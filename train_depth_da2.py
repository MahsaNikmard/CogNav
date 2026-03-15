
import os, json, glob, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(THIS_DIR, "dataset_360fov", "dataset")
META_DIR    = os.path.join(DATASET_DIR, "metadata")
CKPT_DIR    = os.path.join(THIS_DIR, "checkpoints", "da2_finetuned", "best_ckpt")

sys.path.insert(0, THIS_DIR)
from distance_vector import get_distance_vector

BASE_MODEL  = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

N_SLICES    = 36         
H_FOV       = 2 * np.pi 
MAX_RANGE   = 20.0        
SOFT_TEMP   = 0.3         
EPOCHS      = 30
BATCH_SIZE  = 1           
ACCUM_STEPS = 4           
LR          = 5e-6
GRAD_CLIP   = 1.0

SAVE_EVERY  = 5           
MAX_SCENES  = None       
VAL_SPLIT   = 0.10       


class CylindricalDepthDataset(Dataset):
    def __init__(self, dataset_dir, meta_dir, n_slices, max_range, fov,
                 max_scenes=None):
        self.samples = [] 

        meta_files = sorted(glob.glob(os.path.join(meta_dir, "*.json")))
        if max_scenes:
            meta_files = meta_files[:max_scenes]

        for meta_file in meta_files:
            scene_id = os.path.splitext(os.path.basename(meta_file))[0]
            try:
                meta = json.load(open(meta_file))
            except Exception:
                continue

            for rid in range(len(meta.get("robots", []))):
                rgb_path  = os.path.join(dataset_dir,
                                         f"rgb_robot_{rid:03d}",
                                         f"{scene_id}.png")
                mask_path = os.path.join(dataset_dir, "mask",
                                         f"rgb_robot_{rid:03d}",
                                         f"{scene_id}.png")
                if not (os.path.exists(rgb_path) and os.path.exists(mask_path)):
                    continue

                gt = get_distance_vector(meta, rid, n_slices,
                                         max_range=max_range,
                                         fov=fov).astype(np.float32)
                self.samples.append((rgb_path, mask_path, gt))

        print(f"[Dataset] {len(self.samples)} samples "
              f"({'all' if max_scenes is None else f'≤{max_scenes} scenes'})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, mask_path, gt = self.samples[idx]
        rgb  = np.array(Image.open(rgb_path).convert("RGB"))  
        mask = np.array(Image.open(mask_path).convert("L"))    
        W = rgb.shape[1]
        rgb  = np.roll(rgb [:, ::-1, :], W // 2, axis=1)
        mask = np.roll(mask[:, ::-1   ], W // 2, axis=1)

        return rgb, mask, gt

def depth_to_distances(depth_map: torch.Tensor,
                       obs_mask:  torch.Tensor,
                       n_slices:  int,
                       max_range: float,
                       temperature: float) -> torch.Tensor:
    H, W = depth_map.shape
    distances = []

    for i in range(n_slices):
        col_start = int(i       * W / n_slices)
        col_end   = int((i + 1) * W / n_slices)

        slice_depth = depth_map[:, col_start:col_end]         
        slice_obs   = obs_mask [:, col_start:col_end]          

        obs_depths = slice_depth[slice_obs]                  

        if obs_depths.numel() == 0:
            distances.append(depth_map.new_tensor(max_range))
        else:
            weights = torch.softmax(-obs_depths / temperature, dim=0)
            distances.append((obs_depths * weights).sum())

    return torch.stack(distances)  


def evaluate(model, processor, loader, device):
    model.eval()
    all_pred, all_gt = [], []

    with torch.no_grad():
        for rgb_batch, mask_batch, gt_batch in loader:
            B = rgb_batch.shape[0]
            for b in range(B):
                img_np   = rgb_batch [b].numpy()
                mask_np  = mask_batch[b].numpy()
                H_img, W_img = img_np.shape[:2]

                inputs = processor(images=Image.fromarray(img_np),
                                   return_tensors="pt").to(device)
                outputs = model(**inputs)
                depth = torch.nn.functional.interpolate(
                    outputs.predicted_depth.unsqueeze(1),
                    size=(H_img, W_img), mode="bicubic", align_corners=False,
                ).squeeze()

                obs_mask = torch.from_numpy(mask_np == 1).to(device)
                pred = depth_to_distances(depth, obs_mask,
                                          N_SLICES, MAX_RANGE, SOFT_TEMP)
                all_pred.append(pred.cpu().numpy())
                all_gt  .append(gt_batch[b].numpy())

    model.train()

    pred_arr = np.array(all_pred, dtype=np.float32)  
    gt_arr   = np.array(all_gt,   dtype=np.float32)  

    gt_safe  = np.where(gt_arr > 1e-6, gt_arr, 1e-6)

    err = np.abs(pred_arr - gt_arr)          
    sq  = (pred_arr - gt_arr) ** 2

    mae    = float(err.mean())
    absrel = float((err / gt_safe).mean())
    sqrel  = float((sq  / gt_safe).mean())
    r      = float(np.corrcoef(pred_arr.flatten(), gt_arr.flatten())[0, 1])

    ratio   = np.maximum(pred_arr / gt_safe, gt_safe / np.where(pred_arr > 1e-6,
                                                                  pred_arr, 1e-6))
    delta_1 = float((ratio < 1.25  ).mean())
    delta_2 = float((ratio < 1.25**2).mean())
    delta_3 = float((ratio < 1.25**3).mean())

    return {
        "mae":           mae,
        "absrel":        absrel,
        "sqrel":         sqrel,
        "pearson_r":     r,
        "delta_1":       delta_1,
        "delta_2":       delta_2,
        "delta_3":       delta_3,
        "within_0_5m":   float((err < 0.5).mean()),
        "within_1_0m":   float((err < 1.0).mean()),
        "per_slice_mae": err.mean(axis=0),   # (N_SLICES,)
        "n_samples":     pred_arr.shape[0],
    }


def _print_metrics(tag: str, m: dict) -> None:
    print(f"  [{tag}]  MAE={m['mae']:.3f}m  "
          f"AbsRel={m['absrel']:.3f}  r={m['pearson_r']:.3f}  "
          f"δ<1.25={m['delta_1']*100:.1f}%  "
          f"<0.5m={m['within_0_5m']*100:.1f}%  "
          f"<1.0m={m['within_1_0m']*100:.1f}%")


def _print_metrics_detailed(tag: str, m: dict) -> None:
    SEP  = "=" * 72
    SEP2 = "-" * 72
    n    = m.get("n_samples", "?")
    print(f"\n{SEP}")
    print(f"  DEPTH ESTIMATOR EVALUATION REPORT  —  {tag}")
    print(f"  Samples: {n}   Slices/sample: {N_SLICES}   Max range: {MAX_RANGE} m")
    print(SEP)

    print(f"   MAE      (Mean Abs Error)        : {m['mae']:.4f} m")

    print(f"   Pearson r             : {m['pearson_r']:.4f}")
    print(f"   Def: linear correlation between flattened pred and GT vectors.")

    print(f"\n{SEP2}")


# ── offline evaluation entry point ────────────────────────────────────────────

def eval_only(checkpoint_dir: str,
              dataset_dir:    str = DATASET_DIR,
              meta_dir:       str = META_DIR,
              split:          str = "val",
              max_scenes:     int = None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[eval_only] Device     : {device}")
    print(f"[eval_only] Checkpoint : {checkpoint_dir}")
    print(f"[eval_only] Dataset    : {dataset_dir}")
    print(f"[eval_only] Split      : {split}")

    processor = AutoImageProcessor.from_pretrained(checkpoint_dir)
    model     = (AutoModelForDepthEstimation
                 .from_pretrained(checkpoint_dir)
                 .to(device))
    model.eval()

    full_ds = CylindricalDepthDataset(
        dataset_dir, meta_dir, N_SLICES, MAX_RANGE, H_FOV,
        max_scenes=max_scenes
    )
    print(f"[eval_only] Total samples loaded: {len(full_ds)}")

    if split == "val":
        n_val   = max(1, int(len(full_ds) * VAL_SPLIT))
        n_train = len(full_ds) - n_val
        _, eval_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"[eval_only] Evaluating on val split: {len(eval_ds)} samples")
    else:
        eval_ds = full_ds
        print(f"[eval_only] Evaluating on full dataset: {len(eval_ds)} samples")

    loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2)
    m = evaluate(model, processor, loader, device)
    _print_metrics_detailed(
        f"checkpoint={os.path.basename(checkpoint_dir)}  split={split}",
        m
    )
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    os.makedirs(CKPT_DIR, exist_ok=True)

    full_dataset = CylindricalDepthDataset(
        DATASET_DIR, META_DIR, N_SLICES, MAX_RANGE, H_FOV,
        max_scenes=MAX_SCENES
    )
    n_val   = max(1, int(len(full_dataset) * VAL_SPLIT))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"[Data] train={n_train}  val={n_val}")

    loader     = DataLoader(train_ds, batch_size=BATCH_SIZE,
                            shuffle=True,  num_workers=4,
                            pin_memory=(device == 'cuda'))
    val_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2,
                            pin_memory=(device == 'cuda'))
    print(f"Loading base model: {BASE_MODEL}")
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    model     = (AutoModelForDepthEstimation
                 .from_pretrained(BASE_MODEL)
                 .to(device))
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=1e-4)
    loss_fn   = nn.MSELoss()

    best_val_mae  = float('inf')
    best_ckpt_dir = os.path.join(CKPT_DIR, "best")
    history       = [] 
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, (rgb_batch, mask_batch, gt_batch) in enumerate(loader):
            B = rgb_batch.shape[0]
            gt_batch = gt_batch.to(device) 

            pred_list = []
            for b in range(B):
                img_np  = rgb_batch [b].numpy()
                mask_np = mask_batch[b].numpy()  
                H_img, W_img = img_np.shape[:2]

                inputs  = processor(images=Image.fromarray(img_np),
                                    return_tensors="pt").to(device)
                outputs = model(**inputs)

                depth = torch.nn.functional.interpolate(
                    outputs.predicted_depth.unsqueeze(1),
                    size=(H_img, W_img),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()   

                obs_mask = torch.from_numpy(mask_np == 1).to(device) 
                pred = depth_to_distances(depth, obs_mask,
                                          N_SLICES, MAX_RANGE, SOFT_TEMP)
                pred_list.append(pred)

            pred_batch = torch.stack(pred_list) 
            loss = loss_fn(pred_batch, gt_batch) / ACCUM_STEPS
            loss.backward()
            total_loss += loss.item() * ACCUM_STEPS

            if (step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()

            if step % 200 == 0:
                step_mae = torch.mean(torch.abs(pred_batch.detach() -
                                                gt_batch)).item()
                print(f"  [epoch {epoch:03d}  step {step:5d}/{len(loader)}]"
                      f"  loss={loss.item()*ACCUM_STEPS:.4f}  MAE={step_mae:.3f}m")

        avg_loss = total_loss / len(loader)

        val_m = evaluate(model, processor, val_loader, device)
        _print_metrics(f"epoch {epoch:03d}/{EPOCHS}  train_loss={avg_loss:.4f}", val_m)

        history.append({
            "epoch":      epoch,
            "train_loss": avg_loss,
            "val_mae":    val_m["mae"],
            "val_r":      val_m["pearson_r"],
        })

        if val_m["mae"] < best_val_mae:
            best_val_mae = val_m["mae"]
            model.save_pretrained(best_ckpt_dir)
            processor.save_pretrained(best_ckpt_dir)
            print(f" New best val MAE={best_val_mae:.3f}m → saved to {best_ckpt_dir}")
        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            save_dir = os.path.join(CKPT_DIR, f"epoch_{epoch:03d}")
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            print(f"  → Saved HF checkpoint: {save_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DA2 depth fine-tuning & evaluation")
    parser.add_argument("--eval",       action="store_true",
                        help="Run offline evaluation only (no training)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint dir for --eval  "
                             "(default: checkpoints/da2_finetuned/best_ckpt)")
    parser.add_argument("--split",      type=str, default="val",
                        choices=["val", "all"],
                        help="Dataset split to evaluate on  (default: val)")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Cap number of scenes loaded for --eval")
    args = parser.parse_args()

    if args.eval:
        ckpt = args.checkpoint or os.path.join(CKPT_DIR, "best")
        eval_only(ckpt, split=args.split, max_scenes=args.max_scenes)
    else:
        main()
