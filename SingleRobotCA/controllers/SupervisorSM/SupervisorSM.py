import json
import math
import sys
import os

import matplotlib
matplotlib.use('Agg')        
import matplotlib.pyplot as plt
from PIL import Image as PILImage

from controller import Supervisor
import numpy as np

_COGNAV_ROOT = os.environ.get(
    "COGNAV_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
VIZ_DIR = os.path.join(_COGNAV_ROOT, "nav_viz")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'randomization'))

try:
    from randomization import sceneRandomizer
except ImportError:
    print("Warning: sceneRandomizer not available")
    sceneRandomizer = None



def _obstacle_net_clearance(robot_xy: np.ndarray, obstacles: list,
                             robot_radius: float = 0.105) -> float:
    min_surface_dist = float('inf')
    for obs in obstacles:
        centre = np.array(obs["t"][:2])
        yaw    = obs["r"][3]
        hl     = obs["size"][0] / 2
        hw     = obs["size"][1] / 2

        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        delta   = robot_xy - centre
        local_x =  cos_y * delta[0] + sin_y * delta[1]
        local_y = -sin_y * delta[0] + cos_y * delta[1]

        nx = np.clip(local_x, -hl, hl)
        ny = np.clip(local_y, -hw, hw)
        d  = float(np.sqrt((local_x - nx) ** 2 + (local_y - ny) ** 2))
        min_surface_dist = min(min_surface_dist, d)

    return min_surface_dist - robot_radius


class SupervisorSM:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.timestep   = int(self.supervisor.getBasicTimeStep())
        self.scene_randomizer = sceneRandomizer(self.supervisor) if sceneRandomizer else None

        self.emitter = None
        self.receiver = None
        self._setup_devices()

    def _setup_devices(self):
        try:
            self.emitter = self.supervisor.getDevice("emitter")
            if self.emitter:
                print(f"[Supervisor] Emitter on channel {self.emitter.getChannel()}")
        except Exception:
            print("Warning: No emitter device found")
            self.emitter = None

        try:
            self.receiver = self.supervisor.getDevice("receiver")
            if self.receiver:
                self.receiver.enable(self.timestep)
                print(f"[Supervisor] Receiver on channel {self.receiver.getChannel()}")
        except Exception:
            print("Warning: No receiver device found")
            self.receiver = None

    def generate_single_scene(self, scene_id, max_robots=None):
        if not self.scene_randomizer:
            print("Error: Scene randomizer not available")
            return False, 0, 0

        try:
            print(f"\n{'='*70}")
            print(f"Processing scene: {scene_id}")
            print(f"{'='*70}")

            metadata   = self.scene_randomizer.read_meta(scene_id)
            num_robots = len(metadata.get("robots", []))
            if max_robots is not None:
                num_robots = min(num_robots, max_robots)
            n_targets  = len(metadata.get("targets", []))

            print("  Clearing previous scene...")
            self.scene_randomizer.clear_scene_objects()

            for _ in range(10):
                if self.supervisor.step(self.timestep) == -1:
                    return False, 0, 0

            success = self.scene_randomizer.generate_scene(scene_id,
                                                           max_robots=max_robots)
            if not success:
                print(f"Failed to generate scene {scene_id}")
                return False, 0, 0

            print("  Waiting for physics stabilization...")
            for _ in range(5):
                if self.supervisor.step(self.timestep) == -1:
                    return False, 0, 0

            print(f"  Scene {scene_id} ready! ({num_robots} robot(s))")
            return True, num_robots, n_targets

        except Exception as e:
            print(f"  Error processing scene {scene_id}: {e}")
            return False, 0, 0

    def _send_capture(self, scene_id: str):
        """Broadcast capture command to all robots on channel 1."""
        if not self.emitter:
            return
        msg = json.dumps({"cmd": "capture", "scene_id": scene_id}).encode()
        self.emitter.send(msg)
        print(f"[Supervisor] Sent capture command for scene {scene_id}")


    def live_position(self, node_name):
        node = self.supervisor.getFromDef(node_name)
        if node is None:
            return None
        pos = node.getField("translation").getSFVec3f()
        return pos[0], pos[1]

    def live_orientation(self, node_name):
        node = self.supervisor.getFromDef(node_name)
        if node is None:
            return None
        ori = node.getField("rotation").getSFRotation()
        return ori[2] * ori[3]


    def run(self, scene_ids, wait_steps=5):
        for scene_id in scene_ids:
            success, n_robots, n_targets = self.generate_single_scene(scene_id)
            if not success:
                print(f"[Supervisor] Skipping scene {scene_id} (generation failed)")
                continue

            self._send_capture(scene_id)

            print(f"[Supervisor] Scene {scene_id}: {n_robots} robot(s), "
                  f"waiting {wait_steps} steps for capture ...")

            for _ in range(wait_steps):
                if self.supervisor.step(self.timestep) == -1:
                    return

            print(f"[Supervisor] Scene {scene_id} complete.\n")

        print("[Supervisor] All scenes processed.")

    def run_navigation(self, scene_id: str, nav_steps: int = 1000) -> dict:
        success, n_robots, _ = self.generate_single_scene(scene_id, max_robots=1)
        if not success:
            print(f"[Supervisor] Navigation aborted: scene {scene_id} failed.")
            return None
        all_obstacles = []
        all_targets   = []   
        target_xy     = None
        if self.scene_randomizer:
            try:
                meta          = self.scene_randomizer.read_meta(scene_id)
                all_obstacles = (meta.get("inner_obstacles", []) +
                                 meta.get("border_obstacles", []))
                for t in meta.get("targets", []):
                    xy = np.array(t["t"][:2])
                    all_targets.append(xy)
                if all_targets:
                    target_xy = all_targets[0]  
            except Exception as e:
                print(f"[Supervisor] Warning: could not read meta ({e})")

        ROBOT_RADIUS  = 0.105   
        NEAR_MISS_THR = 0.20   
        GOAL_THR      = 0.30    
        VIZ_EVERY     = 5      

        dist_traveled       = 0.0
        min_clearance       = float('inf')
        collision_count     = 0
        near_miss_steps     = 0
        in_collision        = False   
        steps_run           = 0
        prev_xy             = None
        initial_dist_goal   = None
        goal_reached        = False
        goal_step           = None

        path_history = []       
        initial_xy   = None
        current_cl   = None     
        viz_dir      = os.path.join(VIZ_DIR, f"scene_{scene_id}")
        os.makedirs(viz_dir, exist_ok=True)

        print(f"\n[Supervisor] Navigation started — scene {scene_id}, "
              f"{nav_steps} steps.")
        if target_xy is not None:
            print(f"[Supervisor] Target: ({target_xy[0]:.2f}, {target_xy[1]:.2f})\n")

        for step in range(nav_steps):
            if self.supervisor.step(self.timestep) == -1:
                break
            steps_run += 1

            pos = self.live_position("ROBOT_000")
            if pos is None:
                continue
            robot_xy = np.array(pos)

            if initial_xy is None:
                initial_xy = robot_xy.copy()
            path_history.append(robot_xy.copy())
            if prev_xy is not None:
                dist_traveled += float(np.linalg.norm(robot_xy - prev_xy))
            prev_xy = robot_xy.copy()

            if all_obstacles:
                net_cl = _obstacle_net_clearance(robot_xy, all_obstacles,
                                                 ROBOT_RADIUS)
                if net_cl < min_clearance:
                    min_clearance = net_cl
                current_cl = net_cl

                if net_cl < 0.0:
                    if not in_collision:
                        collision_count += 1
                        print(f"  [!] COLLISION  step={step+1}  "
                              f"pos=({robot_xy[0]:.2f},{robot_xy[1]:.2f})  "
                              f"penetration={-net_cl:.3f}m")
                    in_collision = True
                else:
                    in_collision = False
                    if net_cl < NEAR_MISS_THR:
                        near_miss_steps += 1
            if target_xy is not None:
                d_goal = float(np.linalg.norm(robot_xy - target_xy))
                if initial_dist_goal is None:
                    initial_dist_goal = d_goal
                if not goal_reached and d_goal < GOAL_THR:
                    goal_reached = True
                    goal_step    = step + 1
                    print(f"GOAL REACHED  step={step+1}  "
                          f"dist={d_goal:.3f}m")

            status_every = max(1, nav_steps // 20)
            if (step + 1) % status_every == 0:
                status = f"step {step+1}/{nav_steps}"
                if target_xy is not None:
                    d = float(np.linalg.norm(robot_xy - target_xy))
                    status += f"  dist_to_goal={d:.2f}m"
                if all_obstacles and min_clearance < float('inf'):
                    status += f"  min_clr={min_clearance:.3f}m"
                print(f"[]] {status}")
            if steps_run % VIZ_EVERY == 0:
                self._save_viz_frame(
                    viz_dir      = viz_dir,
                    step         = steps_run,
                    nav_steps    = nav_steps,
                    scene_id     = scene_id,
                    robot_xy     = robot_xy,
                    heading      = self.live_orientation("ROBOT_000"),
                    path_history = path_history,
                    initial_xy   = initial_xy,
                    target_xy    = target_xy,
                    all_targets  = all_targets,
                    all_obstacles= all_obstacles,
                    net_cl       = current_cl,
                    goal_reached = goal_reached,
                )

            if goal_reached:
                break
        final_xy_safe = prev_xy if prev_xy is not None else np.zeros(2)
        if target_xy is not None and initial_dist_goal is not None:
            final_dist    = float(np.linalg.norm(final_xy_safe - target_xy))
            goal_progress = max(0.0,
                                (initial_dist_goal - final_dist) / initial_dist_goal
                                ) * 100.0
        else:
            goal_progress = None

        if dist_traveled > 1e-3 and initial_dist_goal is not None:
            path_efficiency = min(initial_dist_goal / dist_traveled, 1.0) * 100.0
        else:
            path_efficiency = None

        near_miss_rate = near_miss_steps / max(steps_run, 1)

        self._print_nav_report(
            scene_id          = scene_id,
            steps_run         = steps_run,
            nav_steps         = nav_steps,
            dist_traveled     = dist_traveled,
            min_clearance     = min_clearance,
            collision_count   = collision_count,
            near_miss_steps   = near_miss_steps,
            near_miss_rate    = near_miss_rate,
            goal_reached      = goal_reached,
            goal_step         = goal_step,
            initial_dist_goal = initial_dist_goal,
            final_xy          = final_xy_safe,
            target_xy         = target_xy,
            goal_progress     = goal_progress,
            path_efficiency   = path_efficiency,
            robot_radius      = ROBOT_RADIUS,
            near_miss_thr     = NEAR_MISS_THR,
        )

        return {
            "scene_id"       : scene_id,
            "steps_run"      : steps_run,
            "nav_steps"      : nav_steps,
            "dist_traveled"  : dist_traveled,
            "goal_reached"   : goal_reached,
            "goal_step"      : goal_step,
            "goal_progress"  : goal_progress,    
            "path_efficiency": path_efficiency,  
            "collision_count": collision_count,
            "min_clearance"  : min_clearance,    
            "near_miss_rate" : near_miss_rate,  
        }

    def _print_nav_report(self, *, scene_id, steps_run, nav_steps,
                          dist_traveled, min_clearance, collision_count,
                          near_miss_steps, near_miss_rate, goal_reached,
                          goal_step, initial_dist_goal, final_xy, target_xy,
                          goal_progress, path_efficiency,
                          robot_radius, near_miss_thr):
        SEP = "=" * 50
        print(f"\n{SEP}")
        print(f"  SCENE {scene_id}  —  steps: {steps_run}/{nav_steps}")
        print(SEP)
        if path_efficiency is not None:
            print(f"  Path Efficiency : {path_efficiency:.1f}%")
        coll_flag = "✓" if collision_count == 0 else "✗ COLLISION"
        print(f"  Collisions      : {collision_count}  {coll_flag}")
        print(SEP + "\n")



    @staticmethod
    def _save_viz_frame(*, viz_dir, step, nav_steps, scene_id,
                        robot_xy, heading, path_history, initial_xy,
                        target_xy, all_targets, all_obstacles, net_cl,
                        goal_reached):  
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        DPI             = 200
        DETECT_RANGE    = 2.0
        STOP_THRESHOLD  = 0.15
        HARD_STOP_RANGE = 0.55
        HARD_STOP_HALF  = math.radians(45.0)
        MAX_RANGE_DISP  = 4.0
        TARGET_RADIUS   = 0.15   
        STEP_DIST       = 0.03  
        pfx = os.path.join(viz_dir, f"step_{step:06d}")

        ns = None
        ns_path = os.path.join(VIZ_DIR, "latest_nav_state.npz")
        if os.path.exists(ns_path):
            try:
                ns = np.load(ns_path, allow_pickle=False)
            except Exception:
                pass

        distances     = ns["distances"]     if ns is not None else None
        smooth_dists  = ns["smooth_dists"]  if ns is not None else None
        cert          = ns["cert"]          if ns is not None else None
        nav_heading   = float(ns["heading"]) if ns is not None else (heading or 0.0)
        detour_active = bool(ns["detour_active"]) if ns is not None else False
        detour_idx    = int(ns["detour_idx"])     if ns is not None else -1

        path_current_idx   = int(ns["path_current_idx"])       if ns is not None else 0
        path_steps_rem     = int(ns["path_steps_rem"])         if ns is not None else 0
        path_slice_indices = ns["path_slice_indices"].tolist() if ns is not None else []
        path_steps_arr     = ns["path_steps"].tolist()         if ns is not None else []
        path_n_slices      = int(ns["path_n_slices"])          if ns is not None else 36
        path_fov           = float(ns["path_fov"])             if ns is not None else 2*math.pi

        n_slices = len(distances) if distances is not None else 36

        scores = None
        if distances is not None and smooth_dists is not None and cert is not None:
            scores = cert * np.clip(1.0 - smooth_dists / DETECT_RANGE, 0.0, 1.0)

        def _draw_obstacles(ax):
            for obs in all_obstacles:
                cx, cy = obs["t"][0], obs["t"][1]
                yaw    = obs["r"][3]
                hl, hw = obs["size"][0] / 2, obs["size"][1] / 2
                cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
                corners = (R @ np.array([[-hl, -hw], [hl, -hw],
                                         [ hl,  hw], [-hl,  hw]]).T).T \
                          + np.array([cx, cy])
                ax.add_patch(plt.Polygon(corners, closed=True,
                                         facecolor="#888888", edgecolor="black",
                                         linewidth=0.8, alpha=0.75, zorder=2))

        def _draw_reference_path(ax):
            if not path_slice_indices:
                return
            pts = [robot_xy.copy()]
            cur = robot_xy.copy()
            for seg_i in range(path_current_idx, len(path_slice_indices)):
                si    = path_slice_indices[seg_i]
                n_st  = (path_steps_rem if seg_i == path_current_idx
                         else path_steps_arr[seg_i])
                w_ang = ((si + 0.5) / path_n_slices - 0.5) * path_fov
                cur   = cur + n_st * STEP_DIST * np.array([math.cos(w_ang),
                                                            math.sin(w_ang)])
                pts.append(cur.copy())
            pts = np.array(pts)
            ax.plot(pts[:, 0], pts[:, 1], "-",
                    color="mediumpurple", linewidth=1.8, alpha=0.80,
                    zorder=3, label="Reference path")
            if len(pts) >= 2:
                ax.annotate("",
                            xy=pts[1], xytext=pts[0],
                            arrowprops=dict(arrowstyle="->",
                                            color="mediumpurple", lw=2.0),
                            zorder=4)
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        raw_img_path = os.path.join(VIZ_DIR, "latest_raw_camera.png")
        if os.path.exists(raw_img_path):
            try:
                ax1.imshow(np.array(PILImage.open(raw_img_path)))
            except Exception:
                pass
        ax1.axis("off")
        ax1.set_title(f"Raw Camera  —  scene {scene_id}  step {step}", fontsize=11, pad=6)
        plt.tight_layout()
        fig1.savefig(f"{pfx}_1_cam_raw.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig1)

        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
        aln_img_path = os.path.join(VIZ_DIR, "latest_camera.png")
        if os.path.exists(aln_img_path):
            try:
                ax2.imshow(np.array(PILImage.open(aln_img_path)))
            except Exception:
                pass
        ax2.axis("off")
        ax2.set_title(f"Aligned Camera (DA2 input)  —  scene {scene_id}  step {step}",
                      fontsize=11, pad=6)
        plt.tight_layout()
        fig2.savefig(f"{pfx}_2_cam_aligned.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig2)
        fig3, ax3 = plt.subplots(1, 1, figsize=(9, 9))

        _draw_obstacles(ax3)
        _draw_reference_path(ax3)

        if len(path_history) > 1:
            ph = np.array(path_history)
            ax3.plot(ph[:, 0], ph[:, 1], "-",
                     color="royalblue", linewidth=2.2,
                     label="Avoidance path", zorder=5)
            ax3.plot(ph[0, 0], ph[0, 1], "s",
                     color="royalblue", markersize=8, zorder=5)

        if distances is not None:
            h = nav_heading
            for i in range(n_slices):
                local_angle = ((i + 0.5) / n_slices - 0.5) * 2.0 * math.pi
                world_angle = h + local_angle
                d  = float(distances[i])
                wx = robot_xy[0] + d * math.cos(world_angle)
                wy = robot_xy[1] + d * math.sin(world_angle)
                sc = float(scores[i]) if scores is not None else 0.0
                sz = 10 if sc > STOP_THRESHOLD else 7 if sc > 0.06 else 4
                ax3.plot(wx, wy, "o", color="darkorange",
                         markersize=sz, alpha=0.85, zorder=6)
            ax3.plot([], [], "o", color="darkorange", markersize=6,
                     label="DA2 depth estimate")
                     
                     
                     
                     
                     
        if smooth_dists is not None:
            h = nav_heading
            for i in range(n_slices):
                local_angle = ((i + 0.5) / n_slices - 0.5) * 2.0 * math.pi
                world_angle = h + local_angle
                d  = float(smooth_dists[i])
                wx = robot_xy[0] + d * math.cos(world_angle)
                wy = robot_xy[1] + d * math.sin(world_angle)
                sc = float(scores[i]) if scores is not None else 0.0
                sz = 12 if sc > STOP_THRESHOLD else 9 if sc > 0.06 else 6
                ax3.plot(wx, wy, "o", color="cyan",
                         markersize=sz, alpha=0.85, zorder=7)
            ax3.plot([], [], "o", color="cyan", markersize=8,
                     label="Smoothed DA2 estimate")

        for txy in all_targets:
            is_primary = (target_xy is not None and
                          np.linalg.norm(txy - target_xy) < 0.01)
            ax3.add_patch(plt.Circle(
                txy, TARGET_RADIUS,
                facecolor="limegreen" if is_primary else "#90EE90",
                edgecolor="darkgreen", linewidth=1.5, alpha=0.85, zorder=9))
            if is_primary and goal_reached:
                ax3.add_patch(plt.Circle(txy, 0.35,
                                         color="limegreen", alpha=0.18, zorder=1))
        if all_targets:
            ax3.add_patch(plt.Circle(
                [0, 0], 0, facecolor="limegreen", edgecolor="darkgreen",
                linewidth=1.5, label=f"Target (×{len(all_targets)})"))

        robot_color = ("red"     if net_cl is not None and net_cl < 0.0
                       else "orange" if net_cl is not None and net_cl < 0.20
                       else "royalblue")
        ax3.plot(robot_xy[0], robot_xy[1], "o",
                 color=robot_color, markersize=14,
                 markeredgecolor="black", markeredgewidth=0.8,
                 label="Robot", zorder=10)
        if heading is not None:
            L = 0.35
            ax3.annotate("",
                         xy=(robot_xy[0] + L * math.cos(heading),
                             robot_xy[1] + L * math.sin(heading)),
                         xytext=(robot_xy[0], robot_xy[1]),
                         arrowprops=dict(arrowstyle="->", color=robot_color,
                                         lw=2.5), zorder=11)

        ax3.set_aspect("equal", adjustable="datalim")
        ax3.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.85)
        ax3.set_title(
            f"Overhead Navigation Map  —  scene {scene_id}  step {step}",
            fontsize=11, pad=6)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel("X (m)")
        ax3.set_ylabel("Y (m)")
        plt.tight_layout()
        fig3.savefig(f"{pfx}_3_overhead.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig3)

        fig4, ax4 = plt.subplots(1, 1, figsize=(7, 7),
                                  subplot_kw={"projection": "polar"})
        ax4.set_theta_zero_location("N")
        ax4.set_theta_direction(-1)   
        if distances is not None:
            sw = 2.0 * math.pi / n_slices

            for i in range(n_slices):
                local_angle = ((i + 0.5) / n_slices - 0.5) * 2.0 * math.pi
                pt  = -local_angle
                d_s = (min(float(smooth_dists[i]), MAX_RANGE_DISP)
                       if smooth_dists is not None else 0.0)
                sc  = float(scores[i]) if scores is not None else 0.0
                bar_c = ("red"       if sc > STOP_THRESHOLD
                         else "orange"    if sc > 0.06
                         else "limegreen")
                ax4.bar(pt, d_s, width=sw * 0.92,
                        color=bar_c, alpha=0.65, align="center", zorder=3)

            raw_th, raw_r = [], []
            for i in range(n_slices):
                local_angle = ((i + 0.5) / n_slices - 0.5) * 2.0 * math.pi
                raw_th.append(-local_angle)
                raw_r.append(min(float(distances[i]), MAX_RANGE_DISP))
            raw_th.append(raw_th[0]);  raw_r.append(raw_r[0])
            ax4.plot(raw_th, raw_r,
                     "--", color="royalblue", linewidth=1.4, alpha=0.85, zorder=6)
            for th, r in zip(raw_th[:-1], raw_r[:-1]):
                ax4.plot([th, th], [0, r],
                         "-", color="royalblue", linewidth=0.5, alpha=0.35, zorder=5)

            hs_th = np.linspace(-HARD_STOP_HALF, HARD_STOP_HALF, 60)
            ax4.fill(hs_th, [HARD_STOP_RANGE] * 60, color="red", alpha=0.12, zorder=2)
            ax4.plot(hs_th, [HARD_STOP_RANGE] * 60, "r--", linewidth=0.8, zorder=4)

            h = nav_heading
            if path_slice_indices and path_current_idx < len(path_slice_indices):
                cur_si   = path_slice_indices[path_current_idx]
                ref_world = ((cur_si + 0.5) / path_n_slices - 0.5) * path_fov
                ref_local = ref_world - h
                ref_polar = -ref_local
                ax4.annotate("",
                             xy=(ref_polar, MAX_RANGE_DISP * 0.88),
                             xytext=(0, 0),
                             arrowprops=dict(arrowstyle="->",
                                             color="lime", lw=2.5), zorder=7)
            if detour_active and 0 <= detour_idx < n_slices:
                det_local = ((detour_idx + 0.5) / n_slices - 0.5) * 2.0 * math.pi
                det_polar = -det_local
                ax4.annotate("",
                             xy=(det_polar, MAX_RANGE_DISP * 0.72),
                             xytext=(0, 0),
                             arrowprops=dict(arrowstyle="->",
                                             color="orange", lw=2.5), zorder=7)

        ax4.set_rmax(MAX_RANGE_DISP)
        ax4.set_rticks([0.5, 1.0, 2.0, MAX_RANGE_DISP])
        ax4.set_rlabel_position(22.5)
        ax4.set_title(
            f"Robot-Centric Depth View  —  scene {scene_id}  step {step}",
            fontsize=11, pad=18)
        ax4.grid(True, alpha=0.35)
        ax4.legend(handles=[
            Patch(facecolor="limegreen", alpha=0.7, label="Safe (smoothed)"),
            Patch(facecolor="orange",    alpha=0.7, label="Moderate"),
            Patch(facecolor="red",       alpha=0.7, label="Danger"),
            Line2D([0], [0], color="royalblue", linestyle="--",
                   linewidth=1.4, label="Raw DA2"),
            Line2D([0], [0], color="lime",   linewidth=2.0, label="Reference dir."),
            Line2D([0], [0], color="orange", linewidth=2.0, label="Detour dir."),
        ], loc="lower left", fontsize=7, framealpha=0.80)
        plt.tight_layout()
        fig4.savefig(f"{pfx}_4_polar.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig4)


    def run_multi_scene_eval(self, scene_ids: list, nav_steps: int = 1000):
        results = []
        n_total = len(scene_ids)

        for idx, sid in enumerate(scene_ids):
            print(f"\n[MultiEval] ── Scene {idx+1}/{n_total}  ({sid}) ──")
            r = self.run_navigation(sid, nav_steps=nav_steps)
            if r is not None:
                results.append(r)
            else:
                print(f"[MultiEval] Scene {sid} skipped (generation failed).")

        if not results:
            print("[MultiEval] No valid scenes — cannot compute aggregate metrics.")
            self.supervisor.simulationSetMode(
                self.supervisor.SIMULATION_MODE_PAUSE)
            return
        SEP  = "=" * 72
        SEP2 = "-" * 72
        print(f"\n{SEP}")
        print(f"  MULTI-SCENE SUMMARY  —  {len(results)} / {n_total} scenes valid")
        print(SEP)
        hdr = (f"  {'scene':>6}  {'goal':>6}  {'dist(m)':>7}  "
               f"{'prog%':>6}  {'eff%':>5}  {'coll':>4}  "
               f"{'minClr(m)':>9}  {'nmRate%':>7}")
        print(hdr)
        print(SEP2)
        for r in results:
            prog  = f"{r['goal_progress']:.1f}"   if r['goal_progress']   is not None else "  n/a"
            eff   = f"{r['path_efficiency']:.1f}" if r['path_efficiency'] is not None else " n/a"
            clr   = f"{r['min_clearance']:.3f}"   if r['min_clearance'] < float('inf') else "  n/a"
            goal  = "YES" if r['goal_reached'] else " NO"
            print(f"  {r['scene_id']:>6}  {goal:>6}  {r['dist_traveled']:>7.3f}  "
                  f"{prog:>6}  {eff:>5}  {r['collision_count']:>4d}  "
                  f"{clr:>9}  {r['near_miss_rate']*100:>7.1f}")

        def _stats(vals):
            v = [x for x in vals if x is not None]
            if not v:
                return None, None
            a = np.array(v, dtype=float)
            return float(np.mean(a)), float(np.std(a))

        success_rate   = np.mean([r['goal_reached']   for r in results]) * 100.0
        coll_free_rate = np.mean([r['collision_count'] == 0 for r in results]) * 100.0

        m_dist,  s_dist  = _stats([r['dist_traveled']   for r in results])
        m_prog,  s_prog  = _stats([r['goal_progress']   for r in results])
        m_eff,   s_eff   = _stats([r['path_efficiency'] for r in results])
        m_coll,  s_coll  = _stats([r['collision_count'] for r in results])
        m_clr,   s_clr   = _stats([r['min_clearance']
                                   if r['min_clearance'] < float('inf') else None
                                   for r in results])
        m_nm,    s_nm    = _stats([r['near_miss_rate']  for r in results])

        n_valid     = len(results)
        n_coll_free = sum(r['collision_count'] == 0 for r in results)

        print(SEP2)
        print(f"  AGGREGATE STATISTICS  —  N = {n_valid} valid scenes")
        print(SEP2)
        print()
        if m_eff is not None:
            print(f"  Path Efficiency %  : {m_eff:.2f} ± {s_eff:.2f} %")
        print(f"  Collision Count    : {m_coll:.2f} ± {s_coll:.2f}  "
              f"({n_coll_free}/{n_valid} collision-free scenes)")
        print()
        print(SEP + "\n")

        self.supervisor.simulationSetMode(
            self.supervisor.SIMULATION_MODE_PAUSE)


if __name__ == "__main__":
    supervisor = Supervisor()
    sm = SupervisorSM(supervisor)

    MODE = "eval"   # "eval" "navigate" "capture"

    if MODE == "eval":
        N_SCENES  = 10
        NAV_STEPS = 1000    
        SCENE_IDS = [f"{i:06d}" for i in range(N_SCENES)]
        sm.run_multi_scene_eval(SCENE_IDS, nav_steps=NAV_STEPS)

    elif MODE == "navigate":
        SCENE_ID  = "000000"
        NAV_STEPS = 1000
        sm.run_navigation(SCENE_ID, nav_steps=NAV_STEPS)

    else:  
        SCENE_IDS  = [f"{i:06d}" for i in range(1)]
        WAIT_STEPS = 5
        sm.run(SCENE_IDS, wait_steps=WAIT_STEPS)
