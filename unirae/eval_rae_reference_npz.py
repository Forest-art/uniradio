from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset


class NpzImageDataset(Dataset):
    def __init__(self, arr: np.ndarray) -> None:
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"Expected NHWC RGB array, got shape={arr.shape}")
        self.arr = arr

    def __len__(self) -> int:
        return int(self.arr.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = self.arr[idx]
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        return tensor


def _import_rae_class(rae_code_root: str):
    root = Path(rae_code_root)
    src = root / "src" if (root / "src").exists() else root
    if not src.exists():
        raise FileNotFoundError(f"Cannot find RAE source dir under: {rae_code_root}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from stage1.rae import RAE  # type: ignore

    return RAE


def _resolve_stage1_params(stage1_config: str, rae_code_root: str) -> Dict:
    cfg_path = Path(stage1_config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"stage1 config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "stage_1" not in cfg or "params" not in cfg["stage_1"]:
        raise ValueError(f"Invalid stage1 config: missing stage_1.params: {cfg_path}")

    params = dict(cfg["stage_1"]["params"])
    root = Path(rae_code_root).expanduser().resolve()
    for key in ("decoder_config_path", "pretrained_decoder_path", "normalization_stat_path"):
        value = params.get(key, "")
        if isinstance(value, str) and value:
            path = Path(value)
            if not path.is_absolute():
                path = root / path
            params[key] = str(path.resolve())
    return params


def _import_official_eval_fns(rae_code_root: str):
    root = Path(rae_code_root)
    src = root / "src" if (root / "src").exists() else root
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from eval import compute_reconstruction_metrics  # type: ignore

    return compute_reconstruction_metrics


@torch.no_grad()
def reconstruct_npz(
    model: torch.nn.Module,
    ref_arr: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    log_interval: int,
) -> np.ndarray:
    dataset = NpzImageDataset(ref_arr)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=False,
    )
    recon_arr = np.empty_like(ref_arr, dtype=np.uint8)
    seen = 0
    total_batches = len(loader)
    for batch_idx, images in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        recon = model(images).clamp(0.0, 1.0)
        recon_np = recon.mul(255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        batch_size_now = int(recon_np.shape[0])
        recon_arr[seen : seen + batch_size_now] = recon_np
        seen += batch_size_now
        if int(log_interval) > 0 and (
            batch_idx == 1 or batch_idx % int(log_interval) == 0 or batch_idx == total_batches
        ):
            print(f"[reconstruction][progress] batch={batch_idx}/{total_batches} seen={seen}", flush=True)
    return recon_arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate official RAE on official ImageNet256 reference NPZ.")
    parser.add_argument("--rae_code_root", default="/project/peilab/luxiaocheng/projects/RAE")
    parser.add_argument("--stage1_config", required=True)
    parser.add_argument("--reference_npz", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--recon_log_interval", type=int, default=20)
    parser.add_argument("--metrics", default="rfid,psnr,ssim")
    parser.add_argument("--save_recon_npz", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    t0 = time.time()
    reference_npz_path = Path(args.reference_npz).expanduser().resolve()
    if not reference_npz_path.exists():
        raise FileNotFoundError(f"reference_npz not found: {reference_npz_path}")

    print(f"[setup] loading reference NPZ: {reference_npz_path}", flush=True)
    ref_data = np.load(reference_npz_path)
    if "arr_0" not in ref_data:
        raise KeyError(f"Expected key 'arr_0' in {reference_npz_path}, got keys={list(ref_data.files)}")
    ref_arr = ref_data["arr_0"]
    print(f"[setup] reference shape={ref_arr.shape} dtype={ref_arr.dtype}", flush=True)

    stage1_params = _resolve_stage1_params(args.stage1_config, args.rae_code_root)
    RAE = _import_rae_class(args.rae_code_root)
    model = RAE(**stage1_params).to(device)
    model.eval()

    print(f"[setup] device={device} batch_size={int(args.batch_size)} eval_batch_size={int(args.eval_batch_size)}", flush=True)
    print(f"[setup] stage1_config={Path(args.stage1_config).resolve()}", flush=True)

    recon_t0 = time.time()
    recon_arr = reconstruct_npz(
        model,
        ref_arr,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
        log_interval=int(args.recon_log_interval),
    )
    recon_wall = time.time() - recon_t0

    if args.save_recon_npz:
        recon_path = Path(args.save_recon_npz).expanduser().resolve()
        recon_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(recon_path, arr_0=recon_arr)
        print(f"[reconstruction] saved={recon_path}", flush=True)
    else:
        recon_path = None

    diff = recon_arr.astype(np.float32) / 255.0 - ref_arr.astype(np.float32) / 255.0
    recon_mse = float(np.mean(np.square(diff), dtype=np.float64))
    recon_rmse = recon_mse ** 0.5
    print(f"[reconstruction] rmse={recon_rmse:.6f} mse={recon_mse:.6f} wall={recon_wall:.1f}s", flush=True)

    metric_names = tuple(x.strip() for x in str(args.metrics).split(",") if x.strip())
    eval_t0 = time.time()
    compute_reconstruction_metrics = _import_official_eval_fns(args.rae_code_root)
    metrics = compute_reconstruction_metrics(
        ref_arr,
        recon_arr,
        device=device,
        batch_size=int(args.eval_batch_size),
        metrics_to_compute=metric_names,
        disable_bar=True,
    )
    eval_wall = time.time() - eval_t0
    print(f"[metrics] {json.dumps(metrics, ensure_ascii=False)}", flush=True)

    out = {
        "stage1_config": str(Path(args.stage1_config).resolve()),
        "reference_npz": str(reference_npz_path),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "num_samples": int(ref_arr.shape[0]),
        "reconstruction": {
            "recon_mse": float(recon_mse),
            "recon_rmse": float(recon_rmse),
        },
        "metrics": {k: float(v) for k, v in metrics.items()},
        "artifacts": {
            "recon_npz": (str(recon_path) if recon_path is not None else ""),
        },
        "timing_sec": {
            "reconstruction": float(recon_wall),
            "metrics": float(eval_wall),
            "total": float(time.time() - t0),
        },
    }

    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote={out_path}", flush=True)


if __name__ == "__main__":
    main()
