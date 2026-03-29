import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .data_imagenet import build_imagenet_loader


class HFImageDataset(Dataset):
    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"]
        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image), int(item.get("label", -1))


def _build_rae_eval_transform(image_size: int) -> transforms.Compose:
    # 与 RAE Stage-1 eval 对齐：Resize 后 CenterCrop，不做 Normalize（RAE 内部自己做 encoder normalize）。
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def _import_rae_class(rae_code_root: str):
    root = Path(rae_code_root)
    src = root / "src" if (root / "src").exists() else root
    if not src.exists():
        raise FileNotFoundError(f"Cannot find RAE source dir under: {rae_code_root}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from stage1.rae import RAE  # type: ignore

    return RAE


def _resolve_defaults(
    rae_code_root: str,
    decoder_config_path: Optional[str],
    pretrained_decoder_path: Optional[str],
    normalization_stat_path: Optional[str],
) -> Tuple[str, str, str]:
    root = Path(rae_code_root)
    dec_cfg = decoder_config_path or str(root / "configs" / "decoder" / "ViTXL")
    dec_ckpt = pretrained_decoder_path or str(
        root / "models" / "decoders" / "dinov2" / "wReg_base" / "ViTXL_n08" / "model.pt"
    )
    stat_ckpt = normalization_stat_path or str(
        root / "models" / "stats" / "dinov2" / "wReg_base" / "imagenet1k" / "stat.pt"
    )
    return dec_cfg, dec_ckpt, stat_ckpt


def _evaluate_recon(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> Dict[str, float]:
    model.eval()
    sum_mse = 0.0
    sum_psnr = 0.0
    sum_count = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches > 0 and bi >= max_batches:
                break
            if isinstance(batch, dict):
                images = batch["images"]
            else:
                images = batch[0]
            images = images.to(device, non_blocking=True)
            rec = model(images)
            target = images
            if rec.shape[-2:] != target.shape[-2:]:
                target = F.interpolate(target, size=rec.shape[-2:], mode="bilinear", align_corners=False)
            mse = F.mse_loss(rec, target)
            psnr = -10.0 * torch.log10(mse + 1e-8)

            bs = int(images.shape[0])
            sum_mse += float(mse.item()) * bs
            sum_psnr += float(psnr.item()) * bs
            sum_count += bs

    denom = max(sum_count, 1)
    mse_mean = sum_mse / denom
    rmse = mse_mean ** 0.5
    psnr_mean = sum_psnr / denom
    return {
        "mse": mse_mean,
        "rmse": rmse,
        "psnr": psnr_mean,
        "num_samples": int(sum_count),
    }


def _verify_decoder_load(rae_model: torch.nn.Module, decoder_ckpt: str) -> Dict[str, object]:
    state_dict = torch.load(decoder_ckpt, map_location="cpu")
    keys = rae_model.decoder.load_state_dict(state_dict, strict=False)  # type: ignore[attr-defined]
    missing = list(keys.missing_keys)
    unexpected = list(keys.unexpected_keys)
    return {
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "missing_head": missing[:10],
        "unexpected_head": unexpected[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate official RAE decoder init quality (MSE/rMSE/PSNR).")
    parser.add_argument("--rae_code_root", default="/project/peilab/luxiaocheng/projects/RAE")
    parser.add_argument("--decoder_config_path", default="")
    parser.add_argument("--pretrained_decoder_path", default="")
    parser.add_argument("--normalization_stat_path", default="")

    parser.add_argument("--dataset_mode", default="hf_online", choices=["hf_online", "imagenet"])
    parser.add_argument("--hf_dataset", default="frgfm/imagenette")
    parser.add_argument("--hf_config", default="160px")
    parser.add_argument("--hf_split", default="validation")

    parser.add_argument("--data_root", default="")
    parser.add_argument("--data_format", default="auto", choices=["auto", "imagefolder", "hf_disk"])
    parser.add_argument("--hf_load_from_disk", default="")
    parser.add_argument("--hf_split_override", default="")
    parser.add_argument("--hf_image_key", default="image")
    parser.add_argument("--hf_label_key", default="label")

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=20)

    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--compare_random_decoder", action="store_true")
    parser.add_argument("--out_json", default="")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tfm = _build_rae_eval_transform(image_size=int(args.image_size))
    if args.dataset_mode == "hf_online":
        from datasets import load_dataset

        ds_hf = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split)
        ds = HFImageDataset(ds_hf, transform=tfm)
        loader = DataLoader(
            ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=True,
            drop_last=False,
        )
        dataset_tag = f"{args.hf_dataset}:{args.hf_config}:{args.hf_split}"
    else:
        if not args.data_root:
            raise ValueError("--data_root is required when dataset_mode=imagenet")
        loader, _ = build_imagenet_loader(
            data_root=args.data_root,
            split="val",
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            shuffle=False,
            drop_last=False,
            data_format=args.data_format,
            hf_load_from_disk=(args.hf_load_from_disk or None),
            hf_split_override=(args.hf_split_override or None),
            hf_image_key=args.hf_image_key,
            hf_label_key=args.hf_label_key,
            custom_transform=tfm,
        )
        dataset_tag = f"imagenet:{args.data_root}:val"

    dec_cfg, dec_ckpt, stat_ckpt = _resolve_defaults(
        rae_code_root=args.rae_code_root,
        decoder_config_path=(args.decoder_config_path or None),
        pretrained_decoder_path=(args.pretrained_decoder_path or None),
        normalization_stat_path=(args.normalization_stat_path or None),
    )
    for p in [dec_cfg, dec_ckpt, stat_ckpt]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required file/path not found: {p}")

    RAE = _import_rae_class(args.rae_code_root)

    print(f"[eval_rae_init] device={device} dataset={dataset_tag}")
    print(f"[eval_rae_init] decoder_ckpt={dec_ckpt}")
    print(f"[eval_rae_init] stat_ckpt={stat_ckpt}")

    official = RAE(
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
    load_check = _verify_decoder_load(official, dec_ckpt)
    print(f"[decoder_load_check] {load_check}")

    m_off = _evaluate_recon(
        model=official,
        loader=loader,
        device=device,
        max_batches=int(args.max_batches),
    )
    print(f"[official_pretrained] {m_off}")

    result = {
        "dataset": dataset_tag,
        "device": str(device),
        "image_size": int(args.image_size),
        "max_batches": int(args.max_batches),
        "decoder_load_check": load_check,
        "official_pretrained": m_off,
    }

    if args.compare_random_decoder:
        random_init = RAE(
            encoder_cls="Dinov2withNorm",
            encoder_config_path="facebook/dinov2-with-registers-base",
            encoder_input_size=224,
            encoder_params={"dinov2_path": "facebook/dinov2-with-registers-base", "normalize": True},
            decoder_config_path=dec_cfg,
            pretrained_decoder_path=None,
            noise_tau=0.0,
            reshape_to_2d=True,
            normalization_stat_path=stat_ckpt,
        ).to(device)
        m_rand = _evaluate_recon(
            model=random_init,
            loader=loader,
            device=device,
            max_batches=int(args.max_batches),
        )
        result["random_decoder"] = m_rand
        result["delta_official_minus_random"] = {
            "mse": float(m_off["mse"] - m_rand["mse"]),
            "rmse": float(m_off["rmse"] - m_rand["rmse"]),
            "psnr": float(m_off["psnr"] - m_rand["psnr"]),
        }
        print(f"[random_decoder] {m_rand}")
        print(f"[delta_official_minus_random] {result['delta_official_minus_random']}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[eval_rae_init] wrote: {out_path}")


if __name__ == "__main__":
    main()
