import math



class Pose:
    def __init__(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0):
        self.x = x
        self.y = y
        self.heading = heading  

    def __repr__(self):
        return (f"Pose(x={self.x:.3f}, y={self.y:.3f}, "
                f"hdg={math.degrees(self.heading):.1f}°)")

class Odometry:
    def __init__(self, wheel_radius: float = 0.033,
                 wheel_base: float = 0.16):
        self.wheel_radius = wheel_radius
        self.wheel_base   = wheel_base
        self.pose         = Pose()
        self._prev_left   = None
        self._prev_right  = None

    def reset(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0):
        self.pose        = Pose(x, y, heading)
        self._prev_left  = None
        self._prev_right = None

    def update(self, left_enc: float, right_enc: float):
        if self._prev_left is None:
            self._prev_left  = left_enc
            self._prev_right = right_enc
            return

        d_left  = (left_enc  - self._prev_left)  * self.wheel_radius
        d_right = (right_enc - self._prev_right) * self.wheel_radius
        self._prev_left  = left_enc
        self._prev_right = right_enc

        d_center = (d_left + d_right) / 2.0
        d_theta  = (d_right - d_left) / self.wheel_base

        self.pose.heading += d_theta
        self.pose.x += d_center * math.cos(self.pose.heading)
        self.pose.y += d_center * math.sin(self.pose.heading)

class DirectionSegment:
    def __init__(self, slice_idx: int, steps: int, label: str = ""):
        self.slice_idx = slice_idx
        self.steps     = steps
        self.label     = label or f"S{slice_idx}"


class ReferencePath:

    def __init__(self, segments: list,
                 n_slices: int  = 36,
                 fov:      float = 2 * math.pi):
        self.segments    = list(segments)
        self.n_slices    = n_slices
        self.fov         = fov
        self.current_idx = 0
        self._steps_rem  = segments[0].steps if segments else 0

    @property
    def finished(self) -> bool:
        return self.current_idx >= len(self.segments)

    @property
    def current_segment(self):
        if self.finished:
            return None
        return self.segments[self.current_idx]

    def current_local_angle(self) -> float:
        seg = self.current_segment
        if seg is None:
            return 0.0
        return ((seg.slice_idx + 0.5) / self.n_slices - 0.5) * self.fov

    def advance(self):
        if self.finished:
            return
        self._steps_rem -= 1
        if self._steps_rem <= 0:
            prev = self.segments[self.current_idx]
            self.current_idx += 1
            if not self.finished:
                self._steps_rem = self.segments[self.current_idx].steps
                nxt = self.segments[self.current_idx]
                print(f"[Path] Segment {self.current_idx + 1}/{len(self.segments)}: "
                      f"slice={nxt.slice_idx} ({nxt.label})  steps={nxt.steps}")
            else:
                print(f"[Path] All segments complete (last was {prev.label}).")

    @property
    def steps_remaining(self) -> int:
        return self._steps_rem


FORWARD  = 18
LEFT_90  = 27
RIGHT_90 = 9


def build_reference_path(n_slices: int  = 36,
                         fov:      float = 2 * math.pi) -> ReferencePath:
    segments = [
        DirectionSegment(FORWARD,  40, "FWD-1"),
        DirectionSegment(LEFT_90,  20, "LEFT-1"),
        DirectionSegment(FORWARD,  30, "FWD-2"),
        DirectionSegment(RIGHT_90, 40, "RIGHT-1"),
        DirectionSegment(FORWARD,  40, "FWD-3"),
        # DirectionSegment(LEFT_90,  20, "LEFT-2"),
        DirectionSegment(RIGHT_90, 20, "RIGHT-2"),
        DirectionSegment(FORWARD,  40, "FWD-4"),
        # DirectionSegment(LEFT_90,  20, "LEFT-3"),
        DirectionSegment(RIGHT_90, 20, "RIGHT-3"),
        DirectionSegment(FORWARD,  40, "FWD-5"),
        DirectionSegment(LEFT_90,  20, "LEFT-4"),
        # DirectionSegment(RIGHT_90, 40, "RIGHT-4"),
    ]
    return ReferencePath(segments, n_slices=n_slices, fov=fov)
