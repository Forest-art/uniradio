from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from .train_dsga_rae_lora import (
    _FrozenCLIPVisionEncoder,
    _build_data_loader,
    _import_rae_class,
    _reconstruct_with_encoder_grads,
    _resolve_rae_defaults,
)


def _load_model_and_head(
    checkpoint: str,
    *,
    rae_code_root: str,
    decoder_config_path: str,
    pretrained_decoder_path: str,
    normalization_stat_path: str,
    num_classes: int,
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.nn.Module, Dict]:
    dec_cfg, dec_ckpt, stat_ckpt = _resolve_rae_defaults(
        rae_code_root=rae_code_root,
        decoder_config_path=decoder_config_path,
        pretrained_decoder_path=pretrained_decoder_path,
        normalization_stat_path=normalization_stat_path,
    )
    for p in [dec_cfg, dec_ckpt, stat_ckpt, checkpoint]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required file/path not found: {p}")

    RAE = _import_rae_class(rae_code_root)
    model = RAE(
        encoder_cls="Dinov2withNorm",
        encoder_config_path="facebook/dinov2-with-registers-base",
        encoder_input_size=224,
        encoder_params={"dinov2_path": "facebook/dinov2-with-registers-base", "normalize": True},
        decoder_config_path=dec_cfg,
        pretrained_decoder_path=dec_ckpt,
        noise_tau=0.0,
        reshape_to_2d=True,
        normalization_stat_path=stat_ckpt,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    out_dim = int(num_classes)
    if isinstance(ckpt, dict):
        head_state = ckpt.get("und_head", ckpt.get("cls_head", None))
        if isinstance(head_state, dict) and "weight" in head_state:
            w = head_state["weight"]
            if hasattr(w, "shape") and len(w.shape) == 2:
                out_dim = int(w.shape[0])
    und_head = nn.Linear(int(model.latent_dim), int(out_dim)).to(device)

    model_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(model_state, strict=False)

    if isinstance(ckpt, dict):
        if "und_head" in ckpt:
            und_head.load_state_dict(ckpt["und_head"], strict=False)
        elif "cls_head" in ckpt:
            und_head.load_state_dict(ckpt["cls_head"], strict=False)

    model.eval()
    und_head.eval()
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    return model, und_head, ckpt_args


@torch.no_grad()
def evaluate_full_rmse_rfid(
    model: torch.nn.Module,
    und_head: torch.nn.Module,
    val_loader,
    *,
    device: torch.device,
    understanding_loss: str,
    clip_vision_encoder: Optional[_FrozenCLIPVisionEncoder],
    rfid_num_samples: int,
    rfid_batch_size: int,
    rfid_tmp_dir: str,
) -> Dict[str, float]:
    sum_mse = 0.0
    sum_u = 0.0
    n_samples = 0

    tmp_root = tempfile.TemporaryDirectory(prefix="dsga_rae_lora_rfid_", dir=(rfid_tmp_dir or None))
    root = Path(tmp_root.name)
    real_dir = root / "real"
    fake_dir = root / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        tokens, rec = _reconstruct_with_encoder_grads(model, images)
        rec = rec.clamp(0.0, 1.0)
        target = images
        if rec.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=rec.shape[-2:], mode="bilinear", align_corners=False)

        pooled = tokens.mean(dim=1)
        mode = str(understanding_loss).lower()
        if mode == "ce":
            logits = und_head(pooled)
            u_metric = (logits.argmax(dim=-1) == labels).float().mean()
        elif mode == "clip_cosine":
            if clip_vision_encoder is None:
                raise ValueError("clip_vision_encoder is required when understanding_loss=clip_cosine")
            pred = F.normalize(und_head(pooled), dim=-1)
            tgt = clip_vision_encoder.encode(images)
            u_metric = (pred * tgt).sum(dim=-1).mean()
        else:
            raise ValueError(f"Unsupported understanding_loss: {understanding_loss}")

        mse_per = ((rec - target) ** 2).flatten(1).mean(dim=1)
        bs = int(images.shape[0])
        n_samples += bs
        sum_mse += float(mse_per.sum().item())
        sum_u += float(u_metric.item()) * bs

        for i in range(bs):
            if saved >= int(rfid_num_samples):
                break
            save_image(target[i].cpu(), str(real_dir / f"{saved:07d}.png"))
            save_image(rec[i].cpu(), str(fake_dir / f"{saved:07d}.png"))
            saved += 1

    val_mse = float(sum_mse / max(n_samples, 1))
    val_rmse = float(val_mse**0.5)
    val_u = float(sum_u / max(n_samples, 1))

    rfid = float("nan")
    rfid_error = ""
    if saved > 1:
        try:
            from torch_fidelity import calculate_metrics

            ret = calculate_metrics(
                input1=str(real_dir),
                input2=str(fake_dir),
                cuda=(device.type == "cuda"),
                isc=False,
                fid=True,
                kid=False,
                prc=False,
                verbose=False,
                batch_size=int(rfid_batch_size),
            )
            rfid = float(ret.get("frechet_inception_distance", float("nan")))
        except Exception as e:  # noqa: BLE001
            rfid_error = str(e)

    tmp_root.cleanup()
    out = {
        "eval_mse": val_mse,
        "eval_rmse": val_rmse,
        "eval_num_samples": int(n_samples),
        "val_rfid": rfid,
        "rfid_num_samples": int(saved),
        "rfid_error": rfid_error,
    }
    mode = str(understanding_loss).lower()
    if mode == "ce":
        out["eval_acc"] = val_u
    elif mode == "clip_cosine":
        out["eval_u_cosine"] = val_u
    return out


def _resolve_runs(run_dirs_arg: str, checkpoint_arg: str) -> List[Tuple[str, str]]:
    runs: List[Tuple[str, str]] = []
    if run_dirs_arg.strip():
        for item in run_dirs_arg.split(","):
            rd = item.strip()
            if not rd:
                continue
            ckpt = str(Path(rd) / "latest.pt")
            runs.append((rd, ckpt))
    if checkpoint_arg.strip():
        ckpt = checkpoint_arg.strip()
        rd = str(Path(ckpt).resolve().parent)
        runs.append((rd, ckpt))
    if len(runs) == 0:
        raise ValueError("Please provide --run_dirs or --checkpoint.")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-val rMSE + rFID eval for DSGA-RAE-LoRA checkpoints.")
    parser.add_argument("--run_dirs", default="", help="Comma-separated run dirs. Each run dir should contain latest.pt")
    parser.add_argument("--checkpoint", default="", help="Optional single checkpoint path.")

    parser.add_argument("--hf_dataset", default="clane9/imagenet-100")
    parser.add_argument("--hf_config", default="")
    parser.add_argument("--val_split", default="validation")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--rfid_num_samples", type=int, default=5000)
    parser.add_argument("--rfid_batch_size", type=int, default=64)
    parser.add_argument("--rfid_tmp_dir", default="/scratch/peilab/xlubl/tmp_rfid")

    parser.add_argument("--rae_code_root", default="/project/peilab/luxiaocheng/projects/RAE")
    parser.add_argument("--decoder_config_path", default="")
    parser.add_argument("--pretrained_decoder_path", default="")
    parser.add_argument("--normalization_stat_path", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    Path(args.rfid_tmp_dir).mkdir(parents=True, exist_ok=True)

    val_loader, num_classes = _build_data_loader(
        hf_dataset=str(args.hf_dataset),
        hf_config=str(args.hf_config),
        split=str(args.val_split),
        image_size=int(args.image_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        is_train=False,
    )
    runs = _resolve_runs(args.run_dirs, args.checkpoint)

    summary: List[Dict] = []
    for run_dir, ckpt in runs:
        model, und_head, ckpt_args = _load_model_and_head(
            checkpoint=ckpt,
            rae_code_root=str(args.rae_code_root),
            decoder_config_path=str(args.decoder_config_path),
            pretrained_decoder_path=str(args.pretrained_decoder_path),
            normalization_stat_path=str(args.normalization_stat_path),
            num_classes=int(num_classes),
            device=device,
        )
        understanding_loss = str(ckpt_args.get("understanding_loss", "ce")).lower()
        clip_vision_encoder = None
        if understanding_loss == "clip_cosine":
            clip_vision_encoder = _FrozenCLIPVisionEncoder(
                model_name=str(ckpt_args.get("clip_vision_model", "openai/clip-vit-base-patch16")),
                input_size=int(ckpt_args.get("clip_input_size", 224)),
                device=device,
            )
        ret = evaluate_full_rmse_rfid(
            model=model,
            und_head=und_head,
            val_loader=val_loader,
            device=device,
            understanding_loss=understanding_loss,
            clip_vision_encoder=clip_vision_encoder,
            rfid_num_samples=int(args.rfid_num_samples),
            rfid_batch_size=int(args.rfid_batch_size),
            rfid_tmp_dir=str(args.rfid_tmp_dir),
        )
        out = {
            "run_dir": str(run_dir),
            "checkpoint": str(ckpt),
            "hf_dataset": str(args.hf_dataset),
            "val_split": str(args.val_split),
            "understanding_loss": understanding_loss,
            **ret,
        }
        out_path = Path(run_dir) / "full_eval_rmse_rfid.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        u_key = "eval_acc" if understanding_loss == "ce" else "eval_u_cosine"
        print(
            f"[eval_done] run={run_dir} {u_key}={ret.get(u_key, float('nan')):.6f} "
            f"rmse={ret['eval_rmse']:.6f} rfid={ret['val_rfid']:.4f} n={ret['eval_num_samples']}"
        )
        summary.append(out)

    # save merged summary near first run
    first = Path(runs[0][0])
    summary_path = first.parent / "full_eval_rmse_rfid_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
