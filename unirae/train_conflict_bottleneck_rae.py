import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .conflict_probe import (
    collect_layerwise_grad_conflict,
    setup_deep_conflict_bottleneck,
    should_log_conflict,
)
from .utils import append_jsonl, ensure_dir, save_json


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


def _import_rae_class(rae_code_root: str):
    root = Path(rae_code_root)
    src = root / "src" if (root / "src").exists() else root
    if not src.exists():
        raise FileNotFoundError(f"Cannot find RAE source dir under: {rae_code_root}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from stage1.rae import RAE  # type: ignore

    return RAE


def _resolve_rae_defaults(
    rae_code_root: str,
    decoder_config_path: str,
    pretrained_decoder_path: str,
    normalization_stat_path: str,
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


def _encode_tokens_with_grad(rae_model, images_01: torch.Tensor) -> torch.Tensor:
    h, w = images_01.shape[-2:]
    if h != int(rae_model.encoder_input_size) or w != int(rae_model.encoder_input_size):
        images_01 = F.interpolate(
            images_01,
            size=(int(rae_model.encoder_input_size), int(rae_model.encoder_input_size)),
            mode="bicubic",
            align_corners=False,
        )
    x = (images_01 - rae_model.encoder_mean.to(images_01)) / rae_model.encoder_std.to(images_01)
    return rae_model.encoder(x)  # [B, N, C]


def _prepare_latent_for_decode(rae_model, tokens: torch.Tensor) -> torch.Tensor:
    z = tokens
    if bool(rae_model.reshape_to_2d):
        b, n, c = z.shape
        hw = int(n**0.5)
        z = z.transpose(1, 2).reshape(b, c, hw, hw)
    if bool(getattr(rae_model, "do_normalization", False)):
        latent_mean = rae_model.latent_mean.to(z.device) if rae_model.latent_mean is not None else 0
        latent_var = rae_model.latent_var.to(z.device) if rae_model.latent_var is not None else 1
        z = (z - latent_mean) / torch.sqrt(latent_var + float(getattr(rae_model, "eps", 1e-5)))
    return z


def _reconstruct_with_encoder_grads(rae_model, images_01: torch.Tensor):
    tokens = _encode_tokens_with_grad(rae_model, images_01)
    z = _prepare_latent_for_decode(rae_model, tokens)
    rec = rae_model.decode(z)
    return tokens, rec


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RAE with deep conflict bottleneck and early high-frequency conflict logging.")
    parser.add_argument("--out_dir", default="results/train_conflict_bottleneck_rae")
    parser.add_argument("--hf_dataset", default="frgfm/imagenette")
    parser.add_argument("--hf_config", default="160px")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--rae_code_root", default="/project/peilab/luxiaocheng/projects/RAE")
    parser.add_argument("--decoder_config_path", default="")
    parser.add_argument("--pretrained_decoder_path", default="")
    parser.add_argument("--normalization_stat_path", default="")
    parser.add_argument("--noise_tau", type=float, default=0.8)

    parser.add_argument("--num_trainable_blocks", type=int, default=4)
    parser.add_argument("--lambda_u", type=float, default=1.0)
    parser.add_argument("--lambda_g", type=float, default=1.0)
    parser.add_argument("--lr_encoder", type=float, default=1e-5)
    parser.add_argument("--lr_decoder", type=float, default=2e-5)
    parser.add_argument("--lr_cls_head", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--early_until", type=int, default=1000)
    parser.add_argument("--early_every", type=int, default=50)
    parser.add_argument("--late_every", type=int, default=500)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))
    ensure_dir(str(out_dir / "conflict"))
    metrics_file = out_dir / "train_metrics.jsonl"

    tfm = transforms.Compose(
        [
            transforms.Resize(int(args.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(int(args.image_size)),
            transforms.ToTensor(),
        ]
    )
    hf_ds = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split)
    ds = HFImageDataset(hf_ds, transform=tfm)
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
    )
    n_classes = len(hf_ds.features["label"].names) if hasattr(hf_ds, "features") and "label" in hf_ds.features else 1000

    dec_cfg, dec_ckpt, stat_ckpt = _resolve_rae_defaults(
        rae_code_root=args.rae_code_root,
        decoder_config_path=args.decoder_config_path,
        pretrained_decoder_path=args.pretrained_decoder_path,
        normalization_stat_path=args.normalization_stat_path,
    )
    for p in [dec_cfg, dec_ckpt, stat_ckpt]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required file/path not found: {p}")

    RAE = _import_rae_class(args.rae_code_root)
    model = RAE(
        encoder_cls="Dinov2withNorm",
        encoder_config_path="facebook/dinov2-with-registers-base",
        encoder_input_size=224,
        encoder_params={"dinov2_path": "facebook/dinov2-with-registers-base", "normalize": True},
        decoder_config_path=dec_cfg,
        pretrained_decoder_path=dec_ckpt,
        noise_tau=float(args.noise_tau),
        reshape_to_2d=True,
        normalization_stat_path=stat_ckpt,
    ).to(device)
    cls_head = nn.Linear(int(model.latent_dim), int(n_classes)).to(device)

    freeze_info = setup_deep_conflict_bottleneck(model.encoder, num_trainable_blocks=int(args.num_trainable_blocks))
    for p in model.decoder.parameters():
        p.requires_grad = True
    for p in cls_head.parameters():
        p.requires_grad = True

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    dec_params = [p for p in model.decoder.parameters() if p.requires_grad]
    cls_params = [p for p in cls_head.parameters() if p.requires_grad]
    if len(enc_params) == 0:
        raise RuntimeError("No trainable encoder params after bottleneck setup.")

    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": float(args.lr_encoder)},
            {"params": dec_params, "lr": float(args.lr_decoder)},
            {"params": cls_params, "lr": float(args.lr_cls_head)},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    save_json(
        {
            "args": vars(args),
            "freeze_info": freeze_info,
            "num_trainable_encoder": int(sum(p.numel() for p in enc_params)),
            "num_trainable_decoder": int(sum(p.numel() for p in dec_params)),
            "num_trainable_cls_head": int(sum(p.numel() for p in cls_params)),
            "decoder_ckpt": dec_ckpt,
            "stat_ckpt": stat_ckpt,
            "num_classes": int(n_classes),
            "device": str(device),
        },
        str(out_dir / "run_setup.json"),
    )

    model.train()
    cls_head.train()

    it = iter(loader)
    pbar = tqdm(range(1, int(args.steps) + 1), desc="train_conflict_bottleneck", disable=False)
    for step in pbar:
        try:
            images, labels = next(it)
        except StopIteration:
            it = iter(loader)
            images, labels = next(it)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        tokens, rec = _reconstruct_with_encoder_grads(model, images)
        logits = cls_head(tokens.mean(dim=1))

        lu = F.cross_entropy(logits, labels)
        target = images
        if rec.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=rec.shape[-2:], mode="bilinear", align_corners=False)
        lg = F.mse_loss(rec, target)
        total = float(args.lambda_u) * lu + float(args.lambda_g) * lg

        if should_log_conflict(
            step=step,
            early_until=int(args.early_until),
            early_every=int(args.early_every),
            late_every=int(args.late_every),
        ):
            log_ret = collect_layerwise_grad_conflict(
                loss_u=lu,
                loss_g=lg,
                named_params=list(model.encoder.named_parameters()),
                step=step,
                out_dir=str(out_dir / "conflict"),
                tag="train",
                retain_graph=True,
            )
            print(
                "[conflict][step={}] cos_mean={:.4f} neg_ratio={:.3f}".format(
                    step,
                    float(log_ret["stats"]["global_cos_mean"]),
                    float(log_ret["stats"]["global_neg_ratio"]),
                )
            )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if float(args.clip_grad) > 0:
            nn.utils.clip_grad_norm_(list(enc_params) + list(dec_params) + list(cls_params), float(args.clip_grad))
        optimizer.step()

        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            rmse = torch.sqrt(lg.detach().clamp_min(0.0))
        row = {
            "step": int(step),
            "lu": float(lu.detach().item()),
            "lg": float(lg.detach().item()),
            "total": float(total.detach().item()),
            "acc": float(acc.item()),
            "rmse": float(rmse.item()),
        }
        append_jsonl(str(metrics_file), row)

        if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
            pbar.set_postfix(lu=f"{row['lu']:.4f}", lg=f"{row['lg']:.4f}", acc=f"{row['acc']:.3f}", rmse=f"{row['rmse']:.4f}")

    torch.save(
        {
            "model": model.state_dict(),
            "cls_head": cls_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(args.steps),
            "args": vars(args),
        },
        str(out_dir / "latest.pt"),
    )
    print(f"[train_conflict_bottleneck] done. out_dir={out_dir}")


if __name__ == "__main__":
    main()
