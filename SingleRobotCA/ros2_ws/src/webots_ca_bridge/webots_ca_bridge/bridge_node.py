#!/usr/bin/env python3
import json
import math
import socket
import threading

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped, PoseStamped, Quaternion, Point
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros

UDP_HOST = "0.0.0.0"
UDP_PORT = 9871


def _yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class WebotsBridgeNode(Node):

    ROBOT_HEIGHT = 0.075  

    def __init__(self):
        super().__init__('webots_ca_bridge')

        # ── publishers ──────────────────────────────────────────────────────
        self._pub_odom   = self.create_publisher(Odometry,    '/robot/odom',        10)
        self._pub_path   = self.create_publisher(Path,        '/robot/path',        10)
        self._pub_plan   = self.create_publisher(MarkerArray, '/robot/plan',        10)
        self._pub_dists  = self.create_publisher(Marker,      '/robot/distances',   10)
        self._pub_detour = self.create_publisher(MarkerArray, '/robot/detour',      10)
        self._pub_action = self.create_publisher(String,      '/robot/action',      10)
        self._pub_diag   = self.create_publisher(String,      '/robot/diagnostics', 10)


        self._tf_br = tf2_ros.TransformBroadcaster(self)
        self._path = Path()
        self._path.header.frame_id = 'map'
        self._had_motion = False   
        self._state: dict | None = None
        self._lock  = threading.Lock()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((UDP_HOST, UDP_PORT))
        self._sock.settimeout(0.1)

        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()
        self.create_timer(0.05, self._publish_cb)

        self.get_logger().info(
            f'WebotsBridgeNode ready — listening on UDP {UDP_HOST}:{UDP_PORT}')

    def _recv_loop(self):
        while rclpy.ok():
            try:
                data, _ = self._sock.recvfrom(8192)
                state = json.loads(data.decode())
                with self._lock:
                    self._state = state
            except socket.timeout:
                pass
            except json.JSONDecodeError as e:
                self.get_logger().warn(f'UDP JSON decode error: {e}')
            except OSError:
                break   # socket closed on shutdown

    def _on_scene_reset(self, stamp):
        self.get_logger().info('New scene detected — clearing RViz visualization.')

        self._path = Path()
        self._path.header.frame_id = 'map'
        empty_path = Path()
        empty_path.header.stamp    = stamp
        empty_path.header.frame_id = 'map'
        self._pub_path.publish(empty_path)

        clear_plan = MarkerArray()
        for ns in ('plan_arrows', 'plan_labels'):
            m = Marker()
            m.header.stamp    = stamp
            m.header.frame_id = 'map'
            m.ns              = ns
            m.id              = 0
            m.action          = Marker.DELETEALL
            clear_plan.markers.append(m)
        self._pub_plan.publish(clear_plan)
        dm = Marker()
        dm.header.stamp    = stamp
        dm.header.frame_id = 'map'
        dm.ns              = 'distances'
        dm.id              = 0
        dm.action          = Marker.DELETE
        self._pub_dists.publish(dm)

        clear_detour = MarkerArray()
        for ns in ('detour_arrow', 'detour_label'):
            m = Marker()
            m.header.stamp    = stamp
            m.header.frame_id = 'map'
            m.ns              = ns
            m.id              = 0
            m.action          = Marker.DELETEALL
            clear_detour.markers.append(m)
        self._pub_detour.publish(clear_detour)

    def _publish_cb(self):
        with self._lock:
            state = self._state
        if state is None:
            return

        now = self.get_clock().now().to_msg()
        x   = float(state['x'])
        y   = float(state['y'])
        hdg = float(state['hdg'])
        q   = _yaw_to_quaternion(hdg)

        dist_from_origin = math.hypot(x, y)
        if self._had_motion and dist_from_origin < 0.15:
            self._on_scene_reset(now)
            self._had_motion = False
        if dist_from_origin > 0.20:
            self._had_motion = True
        tf_msg = TransformStamped()
        tf_msg.header.stamp       = now
        tf_msg.header.frame_id    = 'map'
        tf_msg.child_frame_id     = 'base_link'
        tf_msg.transform.translation.x = x
        tf_msg.transform.translation.y = y
        tf_msg.transform.translation.z = self.ROBOT_HEIGHT
        tf_msg.transform.rotation = q
        self._tf_br.sendTransform(tf_msg)

      
        odom = Odometry()
        odom.header.stamp          = now
        odom.header.frame_id       = 'map'
        odom.child_frame_id        = 'base_link'
        odom.pose.pose.position.x  = x
        odom.pose.pose.position.y  = y
        odom.pose.pose.position.z  = self.ROBOT_HEIGHT
        odom.pose.pose.orientation = q
        self._pub_odom.publish(odom)

        ps = PoseStamped()
        ps.header.stamp          = now
        ps.header.frame_id       = 'map'
        ps.pose.position.x       = x
        ps.pose.position.y       = y
        ps.pose.position.z       = self.ROBOT_HEIGHT
        ps.pose.orientation      = q
        self._path.header.stamp  = now
        self._path.poses.append(ps)
        self._pub_path.publish(self._path)


        segs    = state.get('segs', [])
        seg_idx = int(state.get('seg_idx', 0))
        seg_rem = int(state.get('seg_rem', 0))
        n_s     = int(state.get('n', 36))
        fov_s   = float(state.get('fov', 2 * math.pi))

        if segs:
            ma = MarkerArray()
            arrow_id  = 0
            text_id   = len(segs)  

            for i, seg in enumerate(segs):
                si    = int(seg['slice'])
                steps = int(seg['steps'])
                label = str(seg.get('label', f'S{si}'))


                world_angle = ((si + 0.5) / n_s - 0.5) * fov_s

                if i < seg_idx:
                    length = 0.4
                    r, g, b, a = 0.4, 0.4, 0.4, 0.4
                    shaft_d, head_d = 0.03, 0.07
                elif i == seg_idx:
                    frac   = seg_rem / max(steps, 1)
                    length = 1.0 + 2.5 * frac   
                    r, g, b, a = 0.1, 1.0, 0.2, 1.0
                    shaft_d, head_d = 0.07, 0.18
                else:
                    fade   = 0.85 ** (i - seg_idx)   
                    length = 1.5
                    r, g, b, a = 0.3, 0.6, 1.0, fade
                    shaft_d, head_d = 0.04, 0.10

                end_x = x + length * math.cos(world_angle)
                end_y = y + length * math.sin(world_angle)

                am = Marker()
                am.header.stamp    = now
                am.header.frame_id = 'map'
                am.ns              = 'plan_arrows'
                am.id              = arrow_id
                am.type            = Marker.ARROW
                am.action          = Marker.ADD
                am.scale.x         = shaft_d
                am.scale.y         = head_d
                am.scale.z         = head_d
                am.color.r, am.color.g = r, g
                am.color.b, am.color.a = b, a

                p_start = Point()
                p_start.x, p_start.y, p_start.z = x, y, self.ROBOT_HEIGHT
                p_end   = Point()
                p_end.x, p_end.y, p_end.z = end_x, end_y, self.ROBOT_HEIGHT
                am.points = [p_start, p_end]
                ma.markers.append(am)
                arrow_id += 1

                tm = Marker()
                tm.header.stamp    = now
                tm.header.frame_id = 'map'
                tm.ns              = 'plan_labels'
                tm.id              = text_id
                tm.type            = Marker.TEXT_VIEW_FACING
                tm.action          = Marker.ADD
                tm.pose.position.x = end_x
                tm.pose.position.y = end_y
                tm.pose.position.z = self.ROBOT_HEIGHT + 0.25
                tm.pose.orientation.w = 1.0
                tm.scale.z         = 0.20
                tm.color.r, tm.color.g = r, g
                tm.color.b, tm.color.a = b, max(a, 0.6)
                if i == seg_idx:
                    tm.text = f"{label}\n{seg_rem}steps"
                else:
                    tm.text = label
                ma.markers.append(tm)
                text_id += 1

            self._pub_plan.publish(ma)

        smooth_dists = state.get('smooth_dists')
        dists = smooth_dists if smooth_dists else state.get('dists')
        if dists is not None:
            n             = int(state.get('n',            36))
            fov           = float(state.get('fov',        2 * math.pi))
            cone_half_deg = float(state.get('cone_half_deg', 40.0))

            dm = Marker()
            dm.header.stamp    = now
            dm.header.frame_id = 'map'
            dm.ns              = 'distances'
            dm.id              = 0
            dm.type            = Marker.POINTS
            dm.action          = Marker.ADD
            dm.scale.x         = 0.10  
            dm.scale.y         = 0.10
            dm.pose.orientation.w = 1.0

            cone_half_rad = math.radians(cone_half_deg)

            for i, d in enumerate(dists):
                world_angle = hdg + ((i + 0.5) / n - 0.5) * fov

                pt = Point()
                pt.x = x + d * math.cos(world_angle)
                pt.y = y + d * math.sin(world_angle)
                pt.z = self.ROBOT_HEIGHT
                dm.points.append(pt)
                diff = (world_angle - hdg + math.pi) % (2 * math.pi) - math.pi
                c = ColorRGBA()
                if abs(diff) < cone_half_rad:
                    c.r, c.g, c.b, c.a = 1.0, 0.55, 0.0, 1.0
                else:
                    c.r, c.g, c.b, c.a = 0.0, 0.85, 1.0, 0.75
                dm.colors.append(c)

            self._pub_dists.publish(dm)
        detour_active = bool(state.get('detour_active', False))
        detour_idx    = state.get('detour_idx')

        detour_ma = MarkerArray()
        if detour_active and detour_idx is not None:
            n_d   = int(state.get('n',   36))
            fov_d = float(state.get('fov', 2 * math.pi))
            detour_world_angle = ((int(detour_idx) + 0.5) / n_d - 0.5) * fov_d
            detour_length = 1.8   # metres
            det_end_x = x + detour_length * math.cos(detour_world_angle)
            det_end_y = y + detour_length * math.sin(detour_world_angle)

            # Arrow
            da = Marker()
            da.header.stamp    = now
            da.header.frame_id = 'map'
            da.ns              = 'detour_arrow'
            da.id              = 0
            da.type            = Marker.ARROW
            da.action          = Marker.ADD
            da.scale.x         = 0.08   
            da.scale.y         = 0.20  
            da.scale.z         = 0.20
            da.color.r, da.color.g, da.color.b, da.color.a = 1.0, 0.0, 1.0, 1.0  # magenta
            p0 = Point(); p0.x, p0.y, p0.z = x,         y,         self.ROBOT_HEIGHT
            p1 = Point(); p1.x, p1.y, p1.z = det_end_x, det_end_y, self.ROBOT_HEIGHT
            da.points = [p0, p1]
            detour_ma.markers.append(da)

            dl = Marker()
            dl.header.stamp    = now
            dl.header.frame_id = 'map'
            dl.ns              = 'detour_label'
            dl.id              = 0
            dl.type            = Marker.TEXT_VIEW_FACING
            dl.action          = Marker.ADD
            dl.pose.position.x = det_end_x
            dl.pose.position.y = det_end_y
            dl.pose.position.z = self.ROBOT_HEIGHT + 0.30
            dl.pose.orientation.w = 1.0
            dl.scale.z         = 0.22
            dl.color.r, dl.color.g, dl.color.b, dl.color.a = 1.0, 0.0, 1.0, 1.0
            action_str = str(state.get('action', '?'))
            dl.text = f"DETOUR s{detour_idx}\n{action_str}"
            detour_ma.markers.append(dl)
        else:
            for ns in ('detour_arrow', 'detour_label'):
                dm2 = Marker()
                dm2.header.stamp    = now
                dm2.header.frame_id = 'map'
                dm2.ns              = ns
                dm2.id              = 0
                dm2.action          = Marker.DELETEALL
                detour_ma.markers.append(dm2)

        self._pub_detour.publish(detour_ma)

        action_msg = String()
        action_msg.data = str(state.get('action', '?'))
        self._pub_action.publish(action_msg)

        diag = {
            "action":        state.get('action', '?'),
            "hdg_deg":       round(math.degrees(hdg), 2),
            "hdg_err_deg":   round(math.degrees(float(state.get('hdg_err', 0.0))), 2),
            "max_score":     state.get('max_score', 0.0),
            "min_cone_dist": state.get('min_cone_dist', 99.0),
            "seg_idx":       state.get('seg_idx', 0),
            "seg_rem":       state.get('seg_rem', 0),
            "detour_active": state.get('detour_active', False),
            "detour_idx":    state.get('detour_idx'),
            "certainty":     state.get('certainty', []),
            "smooth_dists":  state.get('smooth_dists', []),
            "cone_data": [
                {"slice": cd[0], "cert": cd[1], "score": cd[2], "smooth_d": cd[3]}
                for cd in state.get('cone_data', [])
            ],
        }
        diag_msg = String()
        diag_msg.data = json.dumps(diag)
        self._pub_diag.publish(diag_msg)

    def destroy_node(self):
        self._sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WebotsBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
