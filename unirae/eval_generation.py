import argparse
from typing import Dict, Optional

import torch
from tqdm import tqdm

from .data_imagenet import build_imagenet_loader, make_batch_dict
from .decoder import ReconDecoder, downsample_target
from .losses import feature_recon_loss, lpips_score, pixel_recon_loss
from .radio_wrapper import RadioWrapper
from .utils import apply_overrides, load_yaml, save_json, to_device


def evaluate_generation(
    model: RadioWrapper,
    decoder: ReconDecoder,
    loader,
    device: torch.device,
    recon_mode: str,
    teacher_model: Optional[RadioWrapper] = None,
    pixel_loss_kind: str = "mse",
    max_batches: Optional[int] = None,
    calc_lpips: bool = False,
) -> Dict[str, float]:
    model.eval()
    decoder.eval()
    if teacher_model is not None:
        teacher_model.eval()

    totals = {
        "loss": 0.0,
        "feature_l2": 0.0,
        "feature_cos": 0.0,
        "mse": 0.0,
        "psnr": 0.0,
    }
    n = 0

    lpips_vals = []

    with torch.no_grad():
        iterator = enumerate(loader)
        for bi, batch in tqdm(iterator, total=max_batches, desc="eval_generation", leave=False):
            if max_batches is not None and bi >= max_batches:
                break
            batch = make_batch_dict(batch)
            batch = to_device(batch, device)

            out = model(batch["images"])
            z_dino = out["z_dino"]
            pred = decoder(z_dino)

            if recon_mode == "feature_recon":
                if teacher_model is None:
                    raise ValueError("teacher_model is required for feature_recon evaluation")
                target = teacher_model(batch["images"])["z_dino"].detach()
                loss, m = feature_recon_loss(pred, target)
                totals["feature_l2"] += m["feature_l2"]
                totals["feature_cos"] += m["feature_cos"]
            else:
                target = downsample_target(batch["images"], size=pred.shape[-1])
                loss, m = pixel_recon_loss(pred, target, kind=pixel_loss_kind)
                totals["mse"] += m["mse"]
                totals["psnr"] += m["psnr"]
                if calc_lpips:
                    l = lpips_score(pred, target)
                    if l is not None:
                        lpips_vals.append(l)

            totals["loss"] += float(loss.item())
            n += 1

    metrics = {"recon_mode": recon_mode, "num_batches": n}
    if n > 0:
        for k, v in totals.items():
            metrics[k] = v / n

    if lpips_vals:
        metrics["lpips"] = sum(lpips_vals) / len(lpips_vals)

    return metrics


def _build_decoder_from_cfg(cfg: Dict, dino_dim: int) -> ReconDecoder:
    dec_cfg = cfg.get("decoder", {})
    recon_cfg = cfg.get("recon", {})
    return ReconDecoder(
        in_dim=dino_dim,
        mode=recon_cfg.get("target", "feature_recon"),
        feature_target_dim=dec_cfg.get("feature_target_dim", dino_dim),
        pixel_size=int(dec_cfg.get("pixel_size", 64)),
        hidden_dim=int(dec_cfg.get("hidden_dim", 1024)),
        token_dropout=float(dec_cfg.get("token_dropout", 0.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.set)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_loader, _ = build_imagenet_loader(
        data_root=cfg["data"]["data_root"],
        split="val",
        image_size=int(cfg["data"].get("image_size", 224)),
        batch_size=int(cfg["eval"].get("batch_size", cfg["train"].get("batch_size", 32))),
        num_workers=int(cfg["data"].get("num_workers", 8)),
        class_names_file=cfg["data"].get("class_names_file"),
        shuffle=False,
        drop_last=False,
        data_format=cfg["data"].get("data_format", "auto"),
        hf_load_from_disk=cfg["data"].get("hf_load_from_disk"),
        hf_split_override=cfg["data"].get("hf_split_val"),
        hf_image_key=cfg["data"].get("hf_image_key", "image"),
        hf_label_key=cfg["data"].get("hf_label_key", "label"),
    )

    model = RadioWrapper(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    dummy = torch.zeros(1, 3, int(cfg["data"].get("image_size", 224)), int(cfg["data"].get("image_size", 224)), device=device)
    with torch.no_grad():
        dino_dim = model(dummy)["z_dino"].shape[-1]

    decoder = _build_decoder_from_cfg(cfg, dino_dim=dino_dim).to(device)
    if "decoder" in ckpt:
        decoder.load_state_dict(ckpt["decoder"], strict=False)

    teacher = None
    recon_mode = cfg.get("recon", {}).get("target", "feature_recon")
    if recon_mode == "feature_recon":
        teacher_cfg = dict(cfg)
        teacher_cfg["lora"] = dict(cfg.get("lora", {}))
        teacher_cfg["lora"]["enable"] = False
        teacher_cfg["radio"] = dict(cfg.get("radio", {}))
        teacher_cfg["radio"]["freeze_trunk"] = True
        teacher = RadioWrapper(teacher_cfg).to(device)
        teacher.eval()

    metrics = evaluate_generation(
        model=model,
        decoder=decoder,
        loader=val_loader,
        device=device,
        recon_mode=recon_mode,
        teacher_model=teacher,
        pixel_loss_kind=cfg.get("recon", {}).get("pixel_loss", "mse"),
        max_batches=args.max_batches,
        calc_lpips=bool(cfg.get("eval", {}).get("use_lpips", False)),
    )

    save_json(metrics, args.output)
    print(f"[eval_generation] wrote {args.output}: {metrics}")


if __name__ == "__main__":
    main()
