import os, json
import numpy as np

DATASET_ROOT = "dataset_360fov/dataset"
IMG_WIDTH    = 336    



def _slice_local_angle(i, N, fov=2 * np.pi):
    return ((i + 0.5) / N - 0.5) * fov


def _ray_rect_distance(ray_origin, world_angle, obj):
    dx = np.cos(world_angle)
    dy = np.sin(world_angle)

    cx, cy = obj["t"][0], obj["t"][1]
    yaw    = obj["r"][3]
    hl     = obj["size"][0] / 2   
    hw     = obj["size"][1] / 2   

    c, s   = np.cos(yaw), np.sin(yaw)
    ox     =  c * (ray_origin[0] - cx) + s * (ray_origin[1] - cy)
    oy     = -s * (ray_origin[0] - cx) + c * (ray_origin[1] - cy)
    ddx    =  c * dx + s * dy
    ddy    = -s * dx + c * dy

    t_min, t_max = -np.inf, np.inf
    for o, d, h in ((ox, ddx, hl), (oy, ddy, hw)):
        if abs(d) < 1e-10:
            if o < -h or o > h:
                return np.inf
        else:
            t1, t2 = (-h - o) / d, (h - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return np.inf

    if t_max < 0:
        return np.inf

    t = t_min if t_min >= 0 else t_max
    return float(t) if t >= 0 else np.inf

def get_distance_vector(meta, robot_id, n_slices, max_range=20.0, fov=2 * np.pi):
    robot         = meta["robots"][robot_id]
    robot_xy      = np.array(robot["t"][:2])
    robot_heading = robot["r"][3]

    all_obstacles = meta["inner_obstacles"] + meta["border_obstacles"]

    distances = np.empty(n_slices)
    for i in range(n_slices):
        local_angle = _slice_local_angle(i, n_slices, fov)
        world_angle = local_angle + robot_heading

        min_d = max_range
        for obs in all_obstacles:
            d = _ray_rect_distance(robot_xy, world_angle, obs)
            if d < min_d:
                min_d = d
        distances[i] = min_d

    return distances

if __name__ == "__main__":
    N = 36

    for sid in ["000000", "000001", "000002"]:
        meta = json.load(
            open(os.path.join(DATASET_ROOT, "metadata", f"{sid}.json"))
        )
        for rid in range(len(meta["robots"])):
            vec = get_distance_vector(meta, rid, N)
            print(f"scene={sid}  robot={rid:03d}  "
                  f"min={vec.min():.3f}m  max={vec.max():.3f}m")
            print(f"  {np.round(vec, 3).tolist()}\n")
