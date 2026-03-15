
from enum import Enum, auto
import math
import numpy as np


BT_DEBUG:    bool      = False
BT_LOG_PATH: str|None = None   
_log_fh                = None  


def _dbg_print(msg: str) -> None:
    global _log_fh
    print(msg)
    if BT_LOG_PATH is not None:
        if _log_fh is None:
            import os
            os.makedirs(os.path.dirname(BT_LOG_PATH), exist_ok=True)
            _log_fh = open(BT_LOG_PATH, 'w', buffering=1) 
        _log_fh.write(msg + "\n")

class NodeStatus(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING  = auto()

class BTNode:
    def tick(self, blackboard: dict) -> NodeStatus:
        raise NotImplementedError(f"{self.__class__.__name__}.tick()")


class Selector(BTNode):
    def __init__(self, children: list):
        self.children = children

    def tick(self, bb: dict) -> NodeStatus:
        for child in self.children:
            status = child.tick(bb)
            if status != NodeStatus.FAILURE:
                return status
        return NodeStatus.FAILURE


class Sequence(BTNode):
    def __init__(self, children: list):
        self.children = children

    def tick(self, bb: dict) -> NodeStatus:
        for child in self.children:
            status = child.tick(bb)
            if status != NodeStatus.SUCCESS:
                return status
        return NodeStatus.SUCCESS

def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

_DEADBAND = 0.08  

def _steer(speed: float, k: float, local_err: float, max_v: float):
    if abs(local_err) < _DEADBAND:
        return speed, speed
    lv = float(np.clip(speed - k * local_err, -max_v, max_v))
    rv = float(np.clip(speed + k * local_err, -max_v, max_v))
    return lv, rv


def _steer_arc(speed: float, k: float, local_err: float, max_v: float):
    if abs(local_err) < _DEADBAND:
        return speed, speed
    lv = float(np.clip(speed - k * local_err, 0.0, max_v))
    rv = float(np.clip(speed + k * local_err, 0.0, max_v))
    return lv, rv


class ObstacleCertainty:
    def __init__(self, n_slices: int = 36,
                 cone_half_angle: float = 40.0,
                 detect_range: float = 3.0,
                 alpha_dist: float = 0.4,
                 alpha_cert: float = 0.5,
                 approach_eps: float = 0.05,
                 fov: float = 2 * math.pi):
        self.n_slices      = n_slices
        self.detect_range  = detect_range
        self.alpha_dist    = alpha_dist
        self.alpha_cert    = alpha_cert
        self.approach_eps  = approach_eps
        self.fov           = fov

        # static forward-cone (used only for display/info; detection uses dynamic cone)
        self.cone_half_angle = cone_half_angle
        fov_deg = math.degrees(fov)
        self.cone_indices = [
            i for i in range(n_slices)
            if abs(((i + 0.5) / n_slices - 0.5) * fov_deg) < cone_half_angle
        ]

        self._smooth      = None   # EMA-smoothed distance vector
        self._prev_smooth = None   # previous step's smoothed vector
        # EMA certainty tracked for ALL slices (dynamic cone needs any slice)
        self._cert        = np.zeros(n_slices, dtype=float)

    def _get_cone_for_heading(self, heading: float) -> list:
        half_slices = round(self.cone_half_angle / (360.0 / self.n_slices))
        forward_slice = int(round((heading / self.fov + 0.5) * self.n_slices - 0.5))
        return [(forward_slice + j) % self.n_slices
                for j in range(-half_slices, half_slices + 1)]

    def update(self, distances: np.ndarray, heading: float = 0.0):
        dist = np.array(distances, dtype=float)

        if self._smooth is None:
            self._smooth = dist.copy()
        else:
            self._smooth = self.alpha_dist * dist + (1.0 - self.alpha_dist) * self._smooth

        #Q condition → EMA certainty for ALL slices
        cone = set(self._get_cone_for_heading(heading))
        if self._prev_smooth is not None:
            delta = self._smooth - self._prev_smooth  
            for i in range(self.n_slices):
                if i in cone:
                    in_range    = self._smooth[i] < self.detect_range
                    approaching = delta[i] < -self.approach_eps
                    q = 1.0 if (in_range and approaching) else 0.0
                else:
                    q = 0.0 
                self._cert[i] = self.alpha_cert * q + (1.0 - self.alpha_cert) * self._cert[i]

            if BT_DEBUG:
                fov_deg = math.degrees(self.fov)
                def _adeg(i): return ((i + 0.5) / self.n_slices - 0.5) * fov_deg
                q_active = [
                    f"s{i}({_adeg(i):+.0f}°,{float(self._smooth[i]):.2f}m,Δ{float(delta[i]):+.3f})"
                    for i in sorted(cone)
                    if self._smooth[i] < self.detect_range and delta[i] < -self.approach_eps
                ]
                cert_i   = int(np.argmax(self._cert))
                smooth_i = int(np.argmin(self._smooth))
                _dbg_print(f"[DBG cert  ] "
                      f"Q1={q_active if q_active else '[]'}  "
                      f"cert_max=s{cert_i}({_adeg(cert_i):+.0f}°):{self._cert[cert_i]:.3f}  "
                      f"smooth_min=s{smooth_i}({_adeg(smooth_i):+.0f}°):{float(self._smooth[smooth_i]):.2f}m")

        self._prev_smooth = self._smooth.copy()

    def compute(self, heading: float = 0.0) -> dict:
        if self._smooth is None:
            return {}
        cone = self._get_cone_for_heading(heading)
        result = {}
        for i in cone:
            cert        = float(self._cert[i])
            smooth_dist = float(self._smooth[i])
            proximity   = max(0.0, 1.0 - smooth_dist / self.detect_range)
            score       = cert * proximity
            result[i]   = (cert, score, smooth_dist)
        return result

    def min_cone_distance(self, heading: float = 0.0) -> float:
        if self._smooth is None:
            return float('inf')
        cone = self._get_cone_for_heading(heading)
        return min(float(self._smooth[i]) for i in cone)

    def min_cone_info(self, heading: float = 0.0):
        if self._smooth is None:
            return (None, float('inf'))
        cone = self._get_cone_for_heading(heading)
        idx = min(cone, key=lambda i: self._smooth[i])
        return (idx, float(self._smooth[idx]))

    def min_forward_distance(self, heading: float, half_angle_deg: float) -> tuple:
        if self._smooth is None:
            return (None, float('inf'))
        slice_width_deg = 360.0 / self.n_slices
        half_slices     = round(half_angle_deg / slice_width_deg)
        forward_slice   = int(round((heading / self.fov + 0.5) * self.n_slices - 0.5))
        indices = [(forward_slice + j) % self.n_slices
                   for j in range(-half_slices, half_slices + 1)]
        idx = min(indices, key=lambda i: self._smooth[i])
        return (idx, float(self._smooth[idx]))

    def smooth_distances(self) -> np.ndarray | None:
        return self._smooth.copy() if self._smooth is not None else None

    def cert_array(self) -> np.ndarray:
        return self._cert.copy()


class ObstacleByMagnitude(BTNode):
    def __init__(self, tracker: ObstacleCertainty,
                 stop_threshold:      float = 0.35,
                 clear_threshold:     float = 0.15,
                 hold_range:          float = 1.5,
                 hard_stop_range:     float = 0.5,
                 hard_stop_half_angle: float = 60.0,
                 block_dist:          float = 1.2,
                 block_score:         float = 0.10,
                 min_hold_ticks:      int   = 15,
                 detour_hold_steps:   int   = 10,
                 max_hold_ticks:      int   = 40):
        self.tracker              = tracker
        self.stop_threshold       = stop_threshold
        self.clear_threshold      = clear_threshold
        self.hold_range           = hold_range
        self.hard_stop_range      = hard_stop_range
        self.hard_stop_half_angle = hard_stop_half_angle  # degrees; narrow forward arc for hard stop
        self.block_dist           = block_dist
        self.block_score          = block_score
        self.min_hold_ticks       = min_hold_ticks  # minimum inference steps before CLEAR allowed
        self.detour_hold_steps    = detour_hold_steps  # post-clear steps to continue detour
        self.max_hold_ticks       = max_hold_ticks    # max steps in HOLD before forced detour exit
        self._danger_active       = False
        self._hold_active         = False
        self._hold_countdown      = 0   # counts down from min_hold_ticks on danger entry
        self._hold_ticks          = 0   # counts up while in HOLD; reset on HOLD entry
        self._post_clear_steps    = 0   # >0: post-CLEAR detour holdoff

    def _angle_deg(self, slice_idx) -> float:
        fov_deg = math.degrees(self.tracker.fov)
        return ((slice_idx + 0.5) / self.tracker.n_slices - 0.5) * fov_deg

    def _compute_blocked(self) -> list:
        n      = self.tracker.n_slices
        smooth = self.tracker.smooth_distances()
        if smooth is None:
            return [0] * n
        cert        = self.tracker.cert_array()
        detect_range = self.tracker.detect_range
        blocked = []
        for i in range(n):
            d       = float(smooth[i])
            score_i = float(cert[i]) * max(0.0, 1.0 - d / detect_range)
            blocked.append(1 if (d < self.block_dist or score_i > self.block_score) else 0)
        return blocked

    def tick(self, bb: dict) -> NodeStatus:
        if self._post_clear_steps > 0:
            self._post_clear_steps -= 1
            bb["detour_active"] = True
            bb["blocked"]       = self._compute_blocked()
            bb["danger_dir"]    = None
            bb["danger_dist"]   = None
            if BT_DEBUG:
                _dbg_print(f"[BT] DETOUR  steps_rem={self._post_clear_steps}")
            return NodeStatus.SUCCESS

        bb["detour_active"] = False

        heading             = float(bb.get("heading", 0.0))
        cone_data           = self.tracker.compute(heading)
        min_cone_dist       = self.tracker.min_cone_distance(heading)
        hard_dir, hard_dist = self.tracker.min_cone_info(heading)
        fwd_dir, fwd_dist = self.tracker.min_forward_distance(
            heading, self.hard_stop_half_angle)
        bb["forward_min_dist"] = fwd_dist
        smooth_all      = self.tracker.smooth_distances()
        if smooth_all is not None:
            global_min_idx  = int(np.argmin(smooth_all))
            global_min_dist = float(smooth_all[global_min_idx])
        else:
            global_min_idx  = hard_dir
            global_min_dist = float('inf')

        bb["min_cone_dist"] = min_cone_dist
        bb["blocked"]       = self._compute_blocked()

        if BT_DEBUG:
            blk = bb["blocked"]
            blk_idx = [i for i, b in enumerate(blk) if b]
            blk_str = ''.join(str(b) for b in blk)
            global_str = (
                f"s{global_min_idx}({self._angle_deg(global_min_idx):+.0f}°):{global_min_dist:.2f}m"
                if global_min_idx is not None else "None(tracker not ready)"
            )
            fwd_str = (
                f"s{fwd_dir}({self._angle_deg(fwd_dir):+.0f}°):{fwd_dist:.2f}m"
                if fwd_dir is not None else "None(not ready)"
            )
            _dbg_print(f"[DBG detect] blocked=[{','.join(map(str,blk_idx))}] "
                  f"({len(blk_idx)}/36)  "
                  f"hist={blk_str}  "
                  f"global={global_str}  "
                  f"fwd±{self.hard_stop_half_angle:.0f}°={fwd_str}  "
                  f"cone_min={min_cone_dist:.2f}m")

        if fwd_dist < self.hard_stop_range or global_min_dist < self.hard_stop_range:
            stop_dir  = fwd_dir        if fwd_dist < self.hard_stop_range else global_min_idx
            stop_dist = fwd_dist       if fwd_dist < self.hard_stop_range else global_min_dist
            stop_tag  = "fwd" if fwd_dist < self.hard_stop_range else "global"
            bb["danger_dir"]  = stop_dir
            bb["danger_dist"] = stop_dist
            if not self._danger_active:
                _dbg_print(f"[BT] DANGER (hard/{stop_tag})  dir={stop_dir}  "
                      f"angle={self._angle_deg(stop_dir) if stop_dir is not None else '?':+.1f}°  "
                      f"dist={stop_dist:.2f}m")
                self._danger_active  = True
                self._hold_countdown = self.min_hold_ticks
            bb["cone_data"]     = cone_data
            bb["max_score"]     = 0.0
            bb["max_certainty"] = 0.0
            return NodeStatus.SUCCESS

        if not cone_data:
            if not self._danger_active or fwd_dist > self.hold_range:
                self._danger_active = False
                self._hold_active   = False
                return NodeStatus.FAILURE

        bb["cone_data"] = cone_data
        max_score = max(score for (_, score, _d) in cone_data.values())
        max_cert  = max(cert  for (cert, _s, _d) in cone_data.values())
        bb["max_score"]     = max_score
        bb["max_certainty"] = max_cert

        score_dir  = max(cone_data, key=lambda i: cone_data[i][1])
        score_dist = cone_data[score_dir][2]

        if BT_DEBUG:
            active = sorted(
                [(i, cert, score, sd) for i, (cert, score, sd) in cone_data.items() if score > 0.01],
                key=lambda x: -x[2]
            )
            lines = [f"s{i}({self._angle_deg(i):+.0f}°,{sd:.2f}m,c={c:.2f},sc={sc:.3f})"
                     for i, c, sc, sd in active[:6]]
            _dbg_print(f"[DBG scores] max_score={max_score:.3f}@s{score_dir}({self._angle_deg(score_dir):+.0f}°)  "
                  f"max_cert={max_cert:.2f}  "
                  f"active={lines if lines else '[]'}  "
                  f"state={'DANGER' if self._danger_active else 'CLEAR'}  "
                  f"hold_cd={self._hold_countdown}")
        if max_score > self.stop_threshold:
            bb["danger_dir"]  = score_dir
            bb["danger_dist"] = score_dist
            if not self._danger_active:
                _dbg_print(f"[BT] DANGER (score) dir={score_dir:2d}  "
                      f"angle={self._angle_deg(score_dir):+.1f}°  "
                      f"dist={score_dist:.2f}m  "
                      f"score={max_score:.3f}  cert={max_cert:.2f}")
                self._danger_active  = True
                self._hold_active    = False
                self._hold_countdown = self.min_hold_ticks
            return NodeStatus.SUCCESS
        if self._danger_active:
            if self._hold_countdown > 0:
                self._hold_countdown -= 1

            if fwd_dist < self.hold_range:
                bb["danger_dir"]  = fwd_dir
                bb["danger_dist"] = fwd_dist
                if not self._hold_active:
                    _dbg_print(f"[BT] HOLD         dir={fwd_dir}  "
                          f"angle={self._angle_deg(fwd_dir) if fwd_dir is not None else '?':+.1f}°  "
                          f"dist={fwd_dist:.2f}m")
                    self._hold_active = True
                    self._hold_ticks  = 0
                else:
                    self._hold_ticks += 1

                # Timeout: if stuck in HOLD too long (e.g. moving parallel to a
                # wall), force a CLEAR → post-detour transition so the robot can
                # escape without waiting for the obstacle to leave the hold range.
                if self._hold_ticks >= self.max_hold_ticks:
                    _dbg_print(f"[BT] HOLD TIMEOUT ({self._hold_ticks} ticks)  "
                          f"fwd={fwd_dist:.2f}m > forcing DETOUR")
                    self._danger_active    = False
                    self._hold_active      = False
                    self._hold_countdown   = 0
                    self._hold_ticks       = 0
                    self._post_clear_steps = self.detour_hold_steps
                    bb["danger_dir"]    = None
                    bb["danger_dist"]   = None
                    bb["detour_active"] = True
                    bb["blocked"]       = self._compute_blocked()
                    return NodeStatus.SUCCESS

                return NodeStatus.SUCCESS

            self._hold_active = False
            if max_score < self.clear_threshold:
                _dbg_print(f"[BT] CLEAR          dir={fwd_dir}  "
                      f"angle={self._angle_deg(fwd_dir) if fwd_dir is not None else '?':+.1f}°  "
                      f"dist={fwd_dist:.2f}m  score={max_score:.3f}")
                self._danger_active   = False
                self._hold_active     = False
                self._hold_countdown  = 0
                self._post_clear_steps = self.detour_hold_steps
                bb["danger_dir"]  = None
                bb["danger_dist"] = None
                bb["detour_active"] = True
                bb["blocked"]       = self._compute_blocked()
                if BT_DEBUG:
                    _dbg_print(f"[BT] DETOUR  steps_rem={self._post_clear_steps} (post-CLEAR)")
                return NodeStatus.SUCCESS

        return NodeStatus.FAILURE

def _vfh_best_direction(blocked: list, ref_idx: int, n: int,
                        min_valley_width: int = 2,
                        valley_margin:    int = 2) -> int | None:
    if all(b == 0 for b in blocked):
        if BT_DEBUG:
            _dbg_print(f"[DBG VFH   ] all free → ref=s{ref_idx}")
        return ref_idx

    first_blocked = next((i for i in range(n) if blocked[i] == 1), None)
    if first_blocked is None:
        return ref_idx   
    def _circ_dist(a: int, b: int) -> int:
        d = abs(a - b) % n
        return min(d, n - d)

    valleys: list[tuple[int, int]] = []   
    walked = 0
    while walked < n:
        idx = (first_blocked + walked) % n
        if blocked[idx] == 0:
            v_start = idx
            v_width = 0
            while walked + v_width < n and blocked[(first_blocked + walked + v_width) % n] == 0:
                v_width += 1
            valleys.append((v_start, v_width))
            walked += v_width
        else:
            walked += 1

    valid = [(s, w) for s, w in valleys if w >= min_valley_width]

    if BT_DEBUG:
        blk_idx = [i for i, b in enumerate(blocked) if b]
        _dbg_print(f"[DBG VFH   ] ref=s{ref_idx}  "
              f"blocked=[{','.join(map(str, blk_idx))}]  "
              f"valleys={[(s,w) for s,w in valleys]}  "
              f"valid={[(s,w) for s,w in valid]}")

    if not valid:
        if BT_DEBUG:
            _dbg_print(f"[DBG VFH   ] no passable gap → HARD STOP")
        return None   
    best_target: int | None = None
    best_cost               = n
    best_width              = 0

    for v_start, v_width in valid:
        v_end = (v_start + v_width - 1) % n

        ref_in_valley = any((v_start + k) % n == ref_idx for k in range(v_width))

        if ref_in_valley:
            target = ref_idx
            reason = "ref_in_valley"
        elif v_width <= min_valley_width + 1:
            target = (v_start + v_width // 2) % n
            reason = "narrow_centre"
        else:
            margin = min(valley_margin, v_width - min_valley_width)
            if _circ_dist(v_start, ref_idx) <= _circ_dist(v_end, ref_idx):
                target = (v_start + margin) % n
                reason = f"wide_near_edge(start+{margin})"
            else:
                target = (v_end - margin) % n
                reason = f"wide_near_edge(end-{margin})"

        cost    = _circ_dist(target, ref_idx)
        is_best = cost < best_cost or (cost == best_cost and v_width > best_width)
        if BT_DEBUG:
            _dbg_print(f"[DBG VFH   ]   valley(s{v_start},w={v_width}) → "
                  f"target=s{target} cost={cost} [{reason}]"
                  f"{'  ← best' if is_best else ''}")
        if is_best:
            best_cost   = cost
            best_target = target
            best_width  = v_width

    if BT_DEBUG:
        _dbg_print(f"[DBG VFH   ] → target=s{best_target} cost={best_cost}")

    return best_target


class FollowPath(BTNode):

    def __init__(self, path,
                 forward_speed: float = 1.0,
                 k_heading:     float = 4.0):
        self.path          = path
        self.forward_speed = forward_speed
        self.k_heading     = k_heading

    def tick(self, bb: dict) -> NodeStatus:
        pose = bb.get("pose")
        if pose is None:
            return NodeStatus.FAILURE

        if self.path.finished:
            bb["controller"].set_velocity(0.0, 0.0)
            bb["last_action"]       = "DONE"
            bb["nav_heading_error"] = 0.0
            return NodeStatus.FAILURE

        ref_world = self.path.current_local_angle()   
        ref_local = _wrap(ref_world - pose.heading)   

        max_v = bb["controller"].max_wheel_speed
        lv, rv = _steer(self.forward_speed, self.k_heading, ref_local, max_v)

        bb["controller"].set_velocity(lv, rv)
        bb["last_action"]       = "FOLLOW"
        bb["nav_heading_error"] = ref_local
        return NodeStatus.SUCCESS


# ── collision-aware steering ───────────────────────────────────────────────────

class AvoidAndSteer(BTNode):
    _RECOVER_SPEED = 0.5

    def __init__(self, path,
                 forward_speed:      float = 1.0,
                 k_heading:          float = 4.0,
                 hard_stop_range:    float = 0.7,
                 avoid_speed_max:    float = 0.5,
                 avoid_speed_min:    float = 0.3,
                 n_slices:           int   = 36,
                 fov:                float = 2 * math.pi,
                 min_valley_width:   int   = 2,
                 valley_margin:      int   = 1,
                 stop_recover_ticks: int   = 20,
                 smooth_alpha:       float = 0.35,
                 memory_decay:       float = 0.85,
                 memory_threshold:   float = 0.25,
                 rotate_threshold:   float = math.pi / 2,
                 rotate_speed:       float = 1.0):
        self.path               = path
        self.forward_speed      = forward_speed
        self.k_heading          = k_heading
        self.hard_stop_range    = hard_stop_range
        self.avoid_speed_max    = avoid_speed_max
        self.avoid_speed_min    = avoid_speed_min
        self.n_slices           = n_slices
        self.fov                = fov
        self.min_valley_width   = min_valley_width
        self.valley_margin      = valley_margin
        self.stop_recover_ticks = stop_recover_ticks
        self.smooth_alpha       = smooth_alpha
        self.memory_decay       = memory_decay
        self.memory_threshold   = memory_threshold
        self._stop_ticks        = 0
        self._smooth_target     = None  
        self._blocked_memory    = None  

    def tick(self, bb: dict) -> NodeStatus:
        pose    = bb.get("pose")
        heading = float(bb.get("heading", 0.0))
        if pose is None:
            return NodeStatus.FAILURE

        forward_min_dist = float(bb.get("forward_min_dist", float('inf')))
        blocked_raw = bb.get("blocked", [])
        if not blocked_raw:
            return NodeStatus.FAILURE

        n     = self.n_slices
        max_v = bb["controller"].max_wheel_speed
        if self._blocked_memory is None:
            self._blocked_memory = np.zeros(n, dtype=float)

        self._blocked_memory *= self.memory_decay
        for i, b in enumerate(blocked_raw):
            if b:
                self._blocked_memory[i] = max(self._blocked_memory[i], 1.0)

        blocked = [1 if (blocked_raw[i] or self._blocked_memory[i] > self.memory_threshold)
                   else 0
                   for i in range(n)]

        if BT_DEBUG:
            mem_held = [i for i in range(n)
                        if not blocked_raw[i] and self._blocked_memory[i] > self.memory_threshold]
            if mem_held:
                _dbg_print(f"[DBG mem ] memory-held (fading, not in raw): {mem_held}")
        ref_world = self.path.current_local_angle()
        ref_idx   = int(round((ref_world / self.fov + 0.5) * n - 0.5)) % n

        fresh_target = _vfh_best_direction(blocked, ref_idx, n,
                                           self.min_valley_width, self.valley_margin)

        if fresh_target is None:
            self._stop_ticks    += 1
            self._smooth_target  = None   
            self._blocked_memory = None   
            recover_phase = self._stop_ticks - self.stop_recover_ticks
            if recover_phase > 0:
                bb["controller"].set_velocity(-self._RECOVER_SPEED, -self._RECOVER_SPEED)
                bb["last_action"] = "RECOVER"
                if BT_DEBUG:
                    _dbg_print(f"[DBG avoid ] RECOVER ({recover_phase}/5)  "
                          f"fwd={forward_min_dist:.2f}m  ticks={self._stop_ticks}")
                if recover_phase >= 5:
                    self._stop_ticks = 0
            else:
                bb["controller"].set_velocity(0.0, 0.0)
                bb["last_action"] = "STOP"
                if BT_DEBUG:
                    _dbg_print(f"[DBG avoid ] STOP (no_valley)  "
                          f"fwd={forward_min_dist:.2f}m  "
                          f"ticks={self._stop_ticks}/{self.stop_recover_ticks}")
            bb["nav_heading_error"] = 0.0
            bb["detour_idx"]        = None
            bb["detour_active"]     = True
            return NodeStatus.SUCCESS

        self._stop_ticks = 0

        fresh_angle = ((fresh_target + 0.5) / n - 0.5) * self.fov
        if self._smooth_target is None:
            self._smooth_target = fresh_angle
        else:
            diff = _wrap(fresh_angle - self._smooth_target)
            self._smooth_target = _wrap(self._smooth_target + self.smooth_alpha * diff)

        target_world = self._smooth_target
        target_local = _wrap(target_world - heading)

        clearance_near = self.hard_stop_range
        clearance_far  = self.hard_stop_range * 4.0
        t     = max(0.0, min(1.0, (forward_min_dist - clearance_near) /
                                   (clearance_far - clearance_near)))
        speed = self.forward_speed * (self.avoid_speed_min +
                                      t * (self.avoid_speed_max - self.avoid_speed_min))


        lv, rv = _steer_arc(speed, self.k_heading, target_local, max_v)
        snapped_idx = int(round((self._smooth_target / self.fov + 0.5) * n - 0.5)) % n
        bb["detour_idx"]        = snapped_idx
        bb["detour_active"]     = True
        bb["nav_heading_error"] = target_local
        bb["last_action"]       = "AVOID"

        if BT_DEBUG:
            _dbg_print(f"[DBG avoid ] [SMOOTH]  "
                  f"ref=s{ref_idx}({math.degrees(ref_world):+.1f}°)  "
                  f"fresh=s{fresh_target}  "
                  f"smooth={math.degrees(target_world):+.1f}°  "
                  f"local={math.degrees(target_local):+.1f}°  "
                  f"fwd={forward_min_dist:.2f}m  t={t:.2f}  "
                  f"speed={speed:.3f}  lv={lv:.3f} rv={rv:.3f}")

        bb["controller"].set_velocity(lv, rv)
        return NodeStatus.SUCCESS
