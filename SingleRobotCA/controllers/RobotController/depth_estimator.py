
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Camera constants (from T3withCamera.proto)
H_FOV = 2 * np.pi   # radians (360°)

DEFAULT_MODEL = os.path.join(
    os.environ.get(
        "COGNAV_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    ),
    "checkpoints", "da2_finetuned", "best_ckpt", "best"
)


class DepthEstimator:
    def __init__(self, n_slices: int = 36, max_range: float = 20.0,
                 model_path: str = None, device: str = None,
                 depth_scale: float = 1.0):
        self.n_slices   = n_slices
        self.max_range  = max_range
        self.min_range  = 0.5
        self.depth_scale = depth_scale  
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model_name = model_path if model_path else DEFAULT_MODEL
        print(f"[DepthEstimator] Loading '{model_name}' on {device} …")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForDepthEstimation.from_pretrained(model_name)
        try:
            model = model.to(device)
        except (RuntimeError, Exception) as e:
            print(f"[DepthEstimator] Warning: cannot use {device} ({e}), falling back to cpu")
            device = 'cpu'
            model = model.to(device)
        self.device = device
        self.model = model
        self.model.eval()
        print(f"[DepthEstimator] Ready on {device}. n_slices={n_slices}, max_range={max_range}m")

    def estimate(self, rgb_image: np.ndarray,
                 seg_mask: np.ndarray = None,
                 save_camera_path: str = None) -> np.ndarray:
        H, W = rgb_image.shape[:2]
        rgb_image = np.roll(rgb_image[:, ::-1, :], W // 2, axis=1)
        if seg_mask is not None:
            seg_mask = np.roll(seg_mask[:, ::-1], W // 2, axis=1)
        if save_camera_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_camera_path)), exist_ok=True)
            Image.fromarray(rgb_image).save(save_camera_path)

        depth_map = self._predict_depth(rgb_image)

        obstacle_mask = (seg_mask == 1) if seg_mask is not None \
                        else np.ones((H, W), dtype=bool)

        N = self.n_slices
        distances = np.full(N, self.max_range, dtype=np.float32)

        for i in range(N):
            col_start = int(i       * W / N)
            col_end   = int((i + 1) * W / N)

            slice_mask  = obstacle_mask[:, col_start:col_end]
            slice_depth = depth_map   [:, col_start:col_end]

            if not slice_mask.any():
                continue

            obs_depth = slice_depth[slice_mask]
            min_depth = float(np.percentile(obs_depth, 5)) * self.depth_scale
            distances[i] = np.clip(min_depth, self.min_range, self.max_range)

        return distances

    def _predict_depth(self, rgb_image: np.ndarray) -> np.ndarray:
        H, W   = rgb_image.shape[:2]
        image  = Image.fromarray(rgb_image)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        depth = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        return depth.astype(np.float32)

    @property
    def slice_angles_deg(self) -> np.ndarray:
        indices = np.arange(self.n_slices)
        return np.degrees(((indices + 0.5) / self.n_slices - 0.5) * H_FOV)
