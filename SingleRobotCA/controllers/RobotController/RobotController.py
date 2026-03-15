from controller import Robot, Emitter, Receiver
import math
import os
import sys
import json
import numpy as np
from PIL import Image
from depth_estimator import DepthEstimator
from behavior_tree import (Selector, Sequence,
                           ObstacleCertainty, ObstacleByMagnitude,
                           FollowPath, AvoidAndSteer)
import behavior_tree as _bt
from reference_path import Odometry, build_reference_path
from state_publisher import StatePublisher

PROJECT_ROOT = os.environ.get(
    "COGNAV_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from distance_vector import get_distance_vector

DATASET_ROOT = os.path.join(PROJECT_ROOT, "src", "dataset_360fov", "dataset")
META_DIR     = os.path.join(PROJECT_ROOT, "dataset_360fov", "dataset", "metadata")
LOG_DIR      = os.path.join(PROJECT_ROOT, "logs")
VIZ_DIR      = os.path.join(PROJECT_ROOT, "nav_viz")
BT_DEBUG = True

N_SLICES  = 36
H_FOV     = 2 * np.pi
MAX_RANGE = 20.0

DEPTH_SCALE = 0.85

NAV_SPEED            = 1.0   
K_HEADING            = 4.0   
INFER_EVERY          = 5     
AVOID_SPEED_MAX      = 0.5    
AVOID_SPEED_MIN      = 0.3    
ROTATE_THRESHOLD_DEG = 150.0  
ROTATE_SPEED         = 0.5   

DETECT_RANGE      = 2.0  
CONE_HALF_ANGLE   = 180.0  
ALPHA_DIST        = 0.40   
                           
                           
                          
ALPHA_CERT        = 0.5   
APPROACH_EPS      = 0.10   
                          
                           
STOP_THRESHOLD    = 0.15   
                           
CLEAR_THRESHOLD   = 0.06   
HOLD_RANGE        = 0.75   
HARD_STOP_RANGE   = 0.55   
                        
BLOCK_DIST        = 1.0   
BLOCK_SCORE       = 0.10   
MIN_VALLEY_WIDTH  = 2      
MIN_HOLD_TICKS    = 15    
DETOUR_HOLD_STEPS = 30     
MAX_HOLD_TICKS    = 40     
HARD_STOP_HALF_ANGLE = 45.0   
                       
class RobotController:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        self.robot_name = self.robot.getName()
        print(f"[{self.robot_name}] Initializing robot controller")

        if self.robot_name.startswith("ROBOT_"):
            self.robot_id = self.robot_name.split('_')[1] if '_' in self.robot_name else "000"
            self.robot_idx = int(self.robot_id)
            self.camera_name = "camera"
        else:
            print(f"[{self.robot_name}] Warning: Unknown robot type")
            self.robot_idx = -1
            self.camera_name = "camera"
            self.robot_id = "000"

        self.camera   = None
        self.receiver = None
        self.emitter  = None
        self._setup_devices()

        self.rgb_dir  = os.path.join(DATASET_ROOT, f"rgb_robot_{self.robot_id}")
        self.mask_dir = os.path.join(DATASET_ROOT, "mask", f"rgb_robot_{self.robot_id}")
        os.makedirs(self.rgb_dir,  exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)

        self.left_motor  = None
        self.right_motor = None
        self._setup_motors()

        self.camera_fov_degrees = 360
        self.wheel_radius = 0.033
        self.wheel_base   = 0.16
        self.max_wheel_speed = 6.67

        self.depth_estimator = DepthEstimator(n_slices=36, depth_scale=DEPTH_SCALE)
        self._captured_scenes = set()
        
    def _setup_devices(self):
        self.camera = self.robot.getDevice(self.camera_name)
        if self.camera:
            self.camera.enable(self.timestep)
            self.camera.recognitionEnable(self.timestep)
            self.camera.enableRecognitionSegmentation()
        else:
            print(f"[{self.robot_name}] Warning: Camera not found")

        self.receiver = self.robot.getDevice("receiver")
        if self.receiver:
            self.receiver.enable(self.timestep)
        else:
            print(f"[{self.robot_name}] Warning: Receiver not found")

        self.emitter = self.robot.getDevice("emitter")
        if not self.emitter:
            print(f"[{self.robot_name}] Warning: Emitter not found")

        self.left_sensor  = self.robot.getDevice("left wheel sensor")
        self.right_sensor = self.robot.getDevice("right wheel sensor")
        if self.left_sensor and self.right_sensor:
            self.left_sensor.enable(self.timestep)
            self.right_sensor.enable(self.timestep)
        else:
            print(f"[{self.robot_name}] Warning: Wheel sensors not found")

    def _setup_motors(self):
        try:
            self.left_motor  = self.robot.getDevice("left wheel motor")
            self.right_motor = self.robot.getDevice("right wheel motor")
            if self.left_motor and self.right_motor:
                self.left_motor.setPosition(float('inf'))
                self.right_motor.setPosition(float('inf'))
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
            else:
                print(f"[{self.robot_name}] Warning: Motors not found")
        except Exception as e:
            print(f"[{self.robot_name}] Error setting up motors: {e}")

    def set_velocity(self, left: float, right: float):
        if self.left_motor and self.right_motor:
            lv = float(np.clip(left,  -self.max_wheel_speed, self.max_wheel_speed))
            rv = float(np.clip(right, -self.max_wheel_speed, self.max_wheel_speed))
            self.left_motor.setVelocity(lv)
            self.right_motor.setVelocity(rv)

    def _capture_rgb(self):
        if not self.camera or self.camera.getSamplingPeriod() <= 0:
            return None
        raw = self.camera.getImage()
        if not raw:
            return None
        w = self.camera.getWidth()
        h = self.camera.getHeight()
        arr = np.frombuffer(raw, np.uint8).reshape((h, w, 4))
        return arr[:, :, [2, 1, 0]]   # BGRA to RGB

    def _capture_segmentation_mask(self):
        if not self.camera or self.camera.getSamplingPeriod() <= 0:
            return None
        seg = self.camera.getRecognitionSegmentationImage()
        if not seg:
            return None
        w = self.camera.getWidth()
        h = self.camera.getHeight()
        seg_array = np.frombuffer(seg, np.uint8).reshape((h, w, 4))
        rgb_seg   = seg_array[:, :, [2, 1, 0]]

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[(rgb_seg == [255, 255, 255]).all(axis=2)] = 0  # Ground
        mask[(rgb_seg == [255,   0,   0]).all(axis=2)] = 1  # Walls
        mask[(rgb_seg == [  0, 255,   0]).all(axis=2)] = 2  # Target
        mask[(rgb_seg == [  0,   0, 255]).all(axis=2)] = 3  # Robots
        return mask

    def capture_and_save(self, scene_id: str):
        rgb  = self._capture_rgb()
        mask = self._capture_segmentation_mask()

        if rgb is None:
            print(f"[{self.robot_name}] capture_and_save: camera not ready, skipping {scene_id}")
            return

        rgb_path = os.path.join(self.rgb_dir, f"{scene_id}.png")
        Image.fromarray(rgb).save(rgb_path)

        if mask is None:
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        mask_path = os.path.join(self.mask_dir, f"{scene_id}.png")
        Image.fromarray(mask, mode='L').save(mask_path)

        print(f"[{self.robot_name}] Saved scene {scene_id} → {rgb_path}")
        self.compare_and_log(scene_id, rgb, mask)

    def compare_and_log(self, scene_id: str, rgb: np.ndarray, mask: np.ndarray):
        meta_path = os.path.join(META_DIR, f"{scene_id}.json")
        if not os.path.exists(meta_path):
            print(f"[{self.robot_name}] No metadata for scene {scene_id}, skipping log")
            return

        meta = json.load(open(meta_path))
        gt   = get_distance_vector(meta, self.robot_idx, N_SLICES,
                                   max_range=MAX_RANGE, fov=H_FOV)
        pred = self.depth_estimator.estimate(rgb, mask)

        err     = pred - gt
        mae     = float(np.mean(np.abs(err)))
        max_err = float(np.max(np.abs(err)))
        corr    = float(np.corrcoef(pred, gt)[0, 1]) if np.std(pred) > 0 else float('nan')

        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, f"scene_{scene_id}.out")
        with open(log_path, "a") as f:
            f.write(f"[{self.robot_name}]  MAE={mae:.3f}m  max_err={max_err:.3f}m"
                    f"  Pearson-r={corr:.3f}\n")
            f.write(f"  GT  : {np.round(gt,   2).tolist()}\n")
            f.write(f"  DA2 : {np.round(pred, 2).tolist()}\n")
            f.write(f"  err : {np.round(err,  2).tolist()}\n\n")

        print(f"[{self.robot_name}] scene {scene_id}  MAE={mae:.3f}m  "
              f"max_err={max_err:.3f}m  r={corr:.3f}  → {log_path}")

    def process_supervisor_commands(self):
        if not self.receiver:
            return
        while self.receiver.getQueueLength() > 0:
            data = self.receiver.getBytes()
            self.receiver.nextPacket()
            try:
                msg = json.loads(data.decode())
                if msg.get("cmd") == "capture":
                    scene_id = msg["scene_id"]
                    if scene_id not in self._captured_scenes:
                        self._captured_scenes.add(scene_id)
                        self.capture_and_save(scene_id)
            except Exception as e:
                print(f"[{self.robot_name}] Bad message: {e}")


    def compute_distance_vector(self, heading: float = 0.0,
                                save_camera_path: str = None,
                                save_raw_path: str = None):
        rgb  = self._capture_rgb()
        mask = self._capture_segmentation_mask()
        if rgb is None:
            return None
        if mask is None:
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

        if save_raw_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_raw_path)),
                        exist_ok=True)
            Image.fromarray(rgb).save(save_raw_path)

        return self.depth_estimator.estimate(rgb, mask,
                                             save_camera_path=save_camera_path)


if __name__ == "__main__":
    _bt.BT_DEBUG    = BT_DEBUG  
    _bt.BT_LOG_PATH = os.path.join(LOG_DIR, "bt_debug.log") 

    controller = RobotController()
    print(f"[{controller.robot_name}] Robot controller started.")

    odometry = Odometry(wheel_radius=controller.wheel_radius,
                        wheel_base=controller.wheel_base)
    path = build_reference_path(N_SLICES, H_FOV)
    certainty_tracker = ObstacleCertainty(
        n_slices        = N_SLICES,
        cone_half_angle = CONE_HALF_ANGLE,
        detect_range    = DETECT_RANGE,
        alpha_dist      = ALPHA_DIST,
        alpha_cert      = ALPHA_CERT,
        approach_eps    = APPROACH_EPS,
        fov             = H_FOV,
    )
    print(f"[{controller.robot_name}] Cone slices (±{CONE_HALF_ANGLE}°): "
          f"{certainty_tracker.cone_indices}")
    _avoid = AvoidAndSteer(
        path              = path,
        forward_speed     = NAV_SPEED,
        k_heading         = K_HEADING,
        hard_stop_range   = HARD_STOP_RANGE,
        avoid_speed_max   = AVOID_SPEED_MAX,
        avoid_speed_min   = AVOID_SPEED_MIN,
        n_slices          = N_SLICES,
        fov               = H_FOV,
        min_valley_width  = MIN_VALLEY_WIDTH,
        rotate_threshold  = math.radians(ROTATE_THRESHOLD_DEG),
        rotate_speed      = ROTATE_SPEED,
    )
    _follow = FollowPath(
        path          = path,
        forward_speed = NAV_SPEED,
        k_heading     = K_HEADING,
    )
    bt = Selector([
        Sequence([
            ObstacleByMagnitude(certainty_tracker, STOP_THRESHOLD, CLEAR_THRESHOLD,
                                HOLD_RANGE, HARD_STOP_RANGE,
                                hard_stop_half_angle=HARD_STOP_HALF_ANGLE,
                                block_dist=BLOCK_DIST, block_score=BLOCK_SCORE,
                                min_hold_ticks=MIN_HOLD_TICKS,
                                detour_hold_steps=DETOUR_HOLD_STEPS,
                                max_hold_ticks=MAX_HOLD_TICKS),
            _avoid,
        ]),
        _follow,
    ])
    blackboard = {"controller": controller, "distances": None, "pose": None}

#--------------------------------
    state_pub = StatePublisher()
    print(f"[{controller.robot_name}] State publisher ready (UDP → bridge_node).")

    infer_counter = 0
    global_step   = 0
    while controller.robot.step(controller.timestep) != -1:
        global_step += 1
        if controller.left_sensor and controller.right_sensor:
            odometry.update(controller.left_sensor.getValue(),
                            controller.right_sensor.getValue())
        blackboard["pose"]    = odometry.pose
        blackboard["heading"] = odometry.pose.heading

        controller.process_supervisor_commands()
        infer_counter += 1
        if infer_counter >= INFER_EVERY:
            infer_counter = 0
            heading = odometry.pose.heading
            os.makedirs(VIZ_DIR, exist_ok=True)
            cam_path     = os.path.join(VIZ_DIR, "latest_camera.png")
            raw_cam_path = os.path.join(VIZ_DIR, "latest_raw_camera.png")
            blackboard["distances"] = controller.compute_distance_vector(
                heading,
                save_camera_path=cam_path,
                save_raw_path=raw_cam_path)
            if blackboard["distances"] is not None:
                certainty_tracker.update(blackboard["distances"], heading)
                _smooth = (certainty_tracker._smooth
                           if certainty_tracker._smooth is not None
                           else np.zeros(N_SLICES, dtype=np.float32))
                np.savez(
                    os.path.join(VIZ_DIR, "latest_nav_state.npz"),
                    distances          = blackboard["distances"],
                    smooth_dists       = _smooth,
                    cert               = certainty_tracker._cert,
                    heading            = np.float32(heading),
                    detour_active      = np.bool_(
                        blackboard.get("detour_active", False)),
                    detour_idx         = np.int32(
                        blackboard.get("detour_idx")
                        if blackboard.get("detour_idx") is not None else -1),
                    path_current_idx   = np.int32(path.current_idx),
                    path_steps_rem     = np.int32(path.steps_remaining),
                    path_slice_indices = np.array(
                        [s.slice_idx for s in path.segments], dtype=np.int32),
                    path_steps         = np.array(
                        [s.steps for s in path.segments], dtype=np.int32),
                    path_n_slices      = np.int32(path.n_slices),
                    path_fov           = np.float32(path.fov),
                )

                _action = blackboard.get("last_action", "FOLLOW")
                if _action not in ("STOP", "RECOVER", "ROTATE"):
                    path.advance()

        bt.tick(blackboard)

        if blackboard["distances"] is not None:
            cone_data = certainty_tracker.compute(heading)
            state_pub.send(
                pose          = odometry.pose,
                distances     = blackboard["distances"],
                action        = blackboard.get("last_action", "?"),
                n_slices      = N_SLICES,
                fov           = H_FOV,
                cone_half_deg = CONE_HALF_ANGLE,
                path          = path,
                smooth_dists  = (certainty_tracker._smooth
                                 if certainty_tracker._smooth is not None else []),
                certainty     = certainty_tracker._cert,
                cone_data     = cone_data,
                hdg_err       = blackboard.get("nav_heading_error", 0.0),
                max_score     = blackboard.get("max_score", 0.0),
                min_cone_dist = blackboard.get("min_cone_dist",
                                               certainty_tracker.min_cone_distance(heading)),
                detour_active = blackboard.get("detour_active", False),
                detour_idx    = blackboard.get("detour_idx"),
            )
