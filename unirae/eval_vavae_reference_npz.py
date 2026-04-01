from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
        return torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)


def _resolve_repo_id(model_name: str) -> str:
    model_name = str(model_name).strip()
    if "/" in model_name:
        return model_name
    return f"stabilityai/{model_name}"


def _import_official_eval_fns(rae_code_root: str):
    import sys

    root = Path(rae_code_root)
    src = root / "src" if (root / "src").exists() else root
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from eval import compute_reconstruction_metrics  # type: ignore

    return compute_reconstruction_metrics


@torch.no_grad()
def reconstruct_npz(
    vae: torch.nn.Module,
    ref_arr: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    posterior_mode: str,
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
    for batch_idx, images_01 in enumerate(loader, start=1):
        images_01 = images_01.to(device, non_blocking=True)
        x = (images_01 * 2.0 - 1.0).clamp(-1.0, 1.0)
        posterior = vae.encode(x).latent_dist
        if posterior_mode == "mode":
            latent = posterior.mode()
        elif posterior_mode == "sample":
            latent = posterior.sample()
        else:
            raise ValueError(f"Unsupported posterior_mode={posterior_mode}")
        recon = vae.decode(latent).sample
        recon_01 = ((recon + 1.0) * 0.5).clamp(0.0, 1.0)
        if recon_01.shape[-2:] != images_01.shape[-2:]:
            recon_01 = F.interpolate(
                recon_01, size=images_01.shape[-2:], mode="bilinear", align_corners=False
            )
        recon_np = recon_01.mul(255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        batch_size_now = int(recon_np.shape[0])
        recon_arr[seen : seen + batch_size_now] = recon_np
        seen += batch_size_now
        if int(log_interval) > 0 and (
            batch_idx == 1 or batch_idx % int(log_interval) == 0 or batch_idx == total_batches
        ):
            print(f"[reconstruction][progress] batch={batch_idx}/{total_batches} seen={seen}", flush=True)
    return recon_arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VA-VAE/SD-VAE on reference NPZ with official rFID.")
    parser.add_argument("--rae_code_root", default="/project/peilab/luxiaocheng/projects/RAE")
    parser.add_argument("--reference_npz", required=True)
    parser.add_argument("--vae_model", default="sd-vae-ft-mse")
    parser.add_argument("--posterior_mode", default="sample", choices=["sample", "mode"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--recon_log_interval", type=int, default=20)
    parser.add_argument("--metrics", default="rfid")
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

    ref_data = np.load(reference_npz_path)
    if "arr_0" not in ref_data:
        raise KeyError(f"Expected key 'arr_0' in {reference_npz_path}, got keys={list(ref_data.files)}")
    ref_arr = ref_data["arr_0"]
    print(f"[setup] loading reference NPZ: {reference_npz_path}", flush=True)
    print(f"[setup] reference shape={ref_arr.shape} dtype={ref_arr.dtype}", flush=True)

    repo_id = _resolve_repo_id(args.vae_model)
    from diffusers.models import AutoencoderKL  # noqa: PLC0415

    vae = AutoencoderKL.from_pretrained(repo_id).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(
        f"[setup] device={device} batch_size={int(args.batch_size)} eval_batch_size={int(args.eval_batch_size)}",
        flush=True,
    )
    print(f"[setup] vae_model={repo_id} posterior_mode={args.posterior_mode}", flush=True)

    recon_t0 = time.time()
    recon_arr = reconstruct_npz(
        vae,
        ref_arr,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
        posterior_mode=str(args.posterior_mode),
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
        "reference_npz": str(reference_npz_path),
        "device": str(device),
        "vae_model": str(args.vae_model),
        "repo_id": repo_id,
        "posterior_mode": str(args.posterior_mode),
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
