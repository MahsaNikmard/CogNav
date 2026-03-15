import socket
import json

UDP_HOST = "127.0.0.1"
UDP_PORT = 9871


class StatePublisher:
    def __init__(self, host: str = UDP_HOST, port: int = UDP_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (host, port)

    def send(self, pose, distances, action: str,
             n_slices: int, fov: float,
             cone_half_deg: float = 40.0,
             path=None,
             smooth_dists=None,
             certainty=None,
             cone_data: dict = None,
             hdg_err: float = 0.0,
             max_score: float = 0.0,
             min_cone_dist: float = float('inf'),
             detour_active: bool = False,
             detour_idx: int | None = None):
        segs = []
        seg_idx = 0
        seg_rem = 0
        if path is not None:
            seg_idx = path.current_idx
            seg_rem = path.steps_remaining
            segs = [{"slice": s.slice_idx, "steps": s.steps, "label": s.label}
                    for s in path.segments]

        cd_list = []
        if cone_data:
            for idx, (cert, score, sd) in cone_data.items():
                cd_list.append([int(idx),
                                 round(float(cert),  4),
                                 round(float(score), 4),
                                 round(float(sd),    3)])

        msg = {
            "x":             round(pose.x,       4),
            "y":             round(pose.y,       4),
            "hdg":           round(pose.heading, 5),
            "dists":         [round(float(d), 3) for d in distances],
            "smooth_dists":  ([round(float(d), 3) for d in smooth_dists]
                               if smooth_dists is not None else []),
            "certainty":     ([round(float(c), 4) for c in certainty]
                               if certainty is not None else []),
            "cone_data":     cd_list,
            "action":        action,
            "hdg_err":       round(float(hdg_err), 4),
            "max_score":     round(float(max_score), 4),
            "min_cone_dist": round(float(min_cone_dist) if min_cone_dist != float('inf')
                                   else 99.0, 3),
            "n":             n_slices,
            "fov":           round(fov, 5),
            "cone_half_deg": cone_half_deg,
            "seg_idx":       seg_idx,
            "seg_rem":       seg_rem,
            "segs":          segs,
            "detour_active": bool(detour_active),
            "detour_idx":    int(detour_idx) if detour_idx is not None else None,
        }
        try:
            self._sock.sendto(json.dumps(msg).encode(), self._addr)
        except OSError:
            pass   

    def close(self):
        self._sock.close()
