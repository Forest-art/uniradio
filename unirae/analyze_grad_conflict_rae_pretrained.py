import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .losses import FeatureVarianceLoss
from .utils import ensure_dir, save_json


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
        x = self.transform(image)
        y = int(item.get("label", -1))
        return x, y


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


def _layer_group_dinov2_wrapped(name: str) -> Tuple[str, int]:
    # Dinov2withNorm named params:
    # - encoder.embeddings.*
    # - encoder.encoder.layer.{i}.*
    # - encoder.layernorm.*
    if name.startswith("encoder.embeddings.patch_embeddings."):
        return "patch_embed", 0
    if name.startswith("encoder.embeddings."):
        return "embeddings", 0
    if name.startswith("encoder.encoder.layer."):
        parts = name.split(".")
        try:
            idx = int(parts[3])
        except Exception:
            idx = -1
        return f"block{idx:02d}", idx + 1
    if name.startswith("encoder.layernorm."):
        return "norm", 999
    return "other", 1000


def _tensor_to_flat(g: torch.Tensor, p: torch.nn.Parameter) -> torch.Tensor:
    if g is None:
        return torch.zeros_like(p, memory_format=torch.preserve_format).reshape(-1)
    return g.detach().reshape(-1)


def _cosine(g1: torch.Tensor, g2: torch.Tensor, eps: float = 1e-12) -> float:
    n1 = torch.norm(g1)
    n2 = torch.norm(g2)
    if float(n1.item()) < eps or float(n2.item()) < eps:
        return 0.0
    return float(torch.dot(g1, g2).item() / (n1.item() * n2.item() + eps))


def _build_views(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # 和现有分析保持一致：第二视角做翻转+亮度扰动，构造理解分支的视角一致性损失。
    x2 = torch.flip(x, dims=[-1])
    b = x.shape[0]
    scale = 0.6 + 0.8 * torch.rand((b, 1, 1, 1), device=x.device, dtype=x.dtype)
    x2 = (x2 * scale).clamp(0.0, 1.0)
    return x, x2


def _infer_num_classes(hf_ds, fallback: int = 1000) -> int:
    feat = None
    if hasattr(hf_ds, "features"):
        feat = hf_ds.features.get("label")
    if feat is not None and hasattr(feat, "names") and feat.names is not None and len(feat.names) > 0:
        return int(len(feat.names))
    if hasattr(hf_ds, "unique"):
        try:
            uniq = hf_ds.unique("label")
            if len(uniq) > 0:
                return int(max(int(x) for x in uniq) + 1)
        except Exception:
            pass
    return int(fallback)


def _make_plots(df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    ordered = summary["layer"].tolist()
    heat = df.pivot(index="layer", columns="batch_idx", values="cos").reindex(ordered)

    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.7, 1.2, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(heat.values, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax0.set_yticks(np.arange(len(ordered)))
    ax0.set_yticklabels(ordered)
    xticks = np.linspace(0, heat.shape[1] - 1, num=min(8, heat.shape[1]), dtype=int)
    ax0.set_xticks(xticks)
    ax0.set_xticklabels([str(int(heat.columns[i])) for i in xticks])
    ax0.set_xlabel("Batch Index")
    ax0.set_ylabel("Encoder Layer")
    ax0.set_title("cos(grad_understanding, grad_generation)")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine Similarity")

    ax1 = fig.add_subplot(gs[0, 1])
    colors = ["#2ca02c" if x >= 0 else "#d62728" for x in summary["cos_mean"]]
    ax1.barh(summary["layer"], summary["cos_mean"], color=colors, alpha=0.9)
    ax1.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean cosine")
    ax1.set_title("Mean by Layer")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(summary["depth"], summary["cos_mean"], marker="o", label="mean cos")
    ax2b = ax2.twinx()
    ax2b.plot(summary["depth"], summary["cos_neg_ratio"], marker="s", color="#d62728", label="neg ratio")
    ax2.set_xlabel("Depth Index")
    ax2.set_ylabel("Mean cosine")
    ax2b.set_ylabel("Negative ratio")
    ax2.set_ylim(-1.0, 1.0)
    ax2b.set_ylim(0.0, 1.0)
    ax2.set_title("Depth Profile")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)

    fig.suptitle("RAE Pretrained Decoder: Layerwise Gradient Conflict", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "grad_conflict_layers.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _encode_tokens_with_grad(rae_model, images_01: torch.Tensor) -> torch.Tensor:
    # 复刻 RAE.encode 逻辑（但保留梯度，不能用 @torch.no_grad 的 encode 方法）。
    h, w = images_01.shape[-2:]
    if h != int(rae_model.encoder_input_size) or w != int(rae_model.encoder_input_size):
        images_01 = F.interpolate(
            images_01,
            size=(int(rae_model.encoder_input_size), int(rae_model.encoder_input_size)),
            mode="bicubic",
            align_corners=False,
        )
    x = (images_01 - rae_model.encoder_mean.to(images_01)) / rae_model.encoder_std.to(images_01)
    # [B, N, C], N 为 patch token 数（已去掉 CLS+register）
    tokens = rae_model.encoder(x)
    return tokens


def _prepare_latent_for_decode(rae_model, tokens: torch.Tensor) -> torch.Tensor:
    z = tokens
    if bool(rae_model.reshape_to_2d):
        b, n, c = z.shape
        hw = int(n ** 0.5)
        z = z.transpose(1, 2).reshape(b, c, hw, hw)
    if bool(getattr(rae_model, "do_normalization", False)):
        latent_mean = rae_model.latent_mean.to(z.device) if rae_model.latent_mean is not None else 0
        latent_var = rae_model.latent_var.to(z.device) if rae_model.latent_var is not None else 1
        z = (z - latent_mean) / torch.sqrt(latent_var + float(getattr(rae_model, "eps", 1e-5)))
    return z


def _reconstruct_with_encoder_grads(rae_model, images_01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = _encode_tokens_with_grad(rae_model, images_01)
    z = _prepare_latent_for_decode(rae_model, tokens)
    rec = rae_model.decode(z)
    return tokens, rec


def _warmup_classifier_head(
    rae_model,
    cls_head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    steps: int,
    lr: float,
) -> Dict[str, float]:
    if steps <= 0:
        return {"steps": 0, "loss": float("nan"), "acc": float("nan")}

    opt = torch.optim.AdamW(cls_head.parameters(), lr=float(lr), betas=(0.9, 0.95), weight_decay=0.0)
    rae_model.eval()
    cls_head.train()
    it = iter(loader)

    sum_loss = 0.0
    sum_acc = 0.0
    for _ in range(int(steps)):
        try:
            images, labels = next(it)
        except StopIteration:
            it = iter(loader)
            images, labels = next(it)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            feats = _encode_tokens_with_grad(rae_model, images).mean(dim=1)
        logits = cls_head(feats)
        loss = F.cross_entropy(logits, labels)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == labels).float().mean()
        sum_loss += float(loss.item())
        sum_acc += float(acc.item())

    cls_head.eval()
    return {
        "steps": int(steps),
        "loss": sum_loss / float(steps),
        "acc": sum_acc / float(steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze layer-wise gradient conflict on official RAE-pretrained decoder (understanding vs reconstruction)."
    )
    parser.add_argument("--out_dir", default="results/grad_conflict_rae_pretrained")
    parser.add_argument("--hf_dataset", default="frgfm/imagenette")
    parser.add_argument("--hf_config", default="160px")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_batches", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--lambda_var", type=float, default=0.1)
    parser.add_argument("--understanding_loss", default="cls_ce", choices=["cls_ce", "dino_reg"])
    parser.add_argument("--num_classes", type=int, default=-1)
    parser.add_argument("--cls_warmup_steps", type=int, default=40)
    parser.add_argument("--cls_warmup_lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--rae_code_root", default="/project/peilab/luxiaocheng/projects/RAE")
    parser.add_argument("--decoder_config_path", default="")
    parser.add_argument("--pretrained_decoder_path", default="")
    parser.add_argument("--normalization_stat_path", default="")
    parser.add_argument("--noise_tau", type=float, default=0.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))

    tfm = transforms.Compose(
        [
            transforms.Resize(int(args.image_size * 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ]
    )
    hf_ds = load_dataset(args.hf_dataset, args.hf_config, split=args.split)
    ds = HFImageDataset(hf_ds, transform=tfm)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

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
    model.train()

    # 只对 encoder 分层做冲突统计；开启 encoder 梯度，decoder 也保留可导以便 Lg 反传回 encoder。
    for p in model.encoder.parameters():
        p.requires_grad = True

    cls_head = None
    cls_warmup = None
    if args.understanding_loss == "cls_ce":
        num_classes = int(args.num_classes) if int(args.num_classes) > 0 else _infer_num_classes(hf_ds, fallback=1000)
        cls_head = nn.Linear(int(model.latent_dim), num_classes).to(device)
        cls_warmup = _warmup_classifier_head(
            rae_model=model,
            cls_head=cls_head,
            loader=loader,
            device=device,
            steps=int(args.cls_warmup_steps),
            lr=float(args.cls_warmup_lr),
        )
        print(
            "[grad_conflict_rae_pretrained] cls_warmup steps={} loss={:.4f} acc={:.3f} num_classes={}".format(
                cls_warmup["steps"],
                cls_warmup["loss"],
                cls_warmup["acc"],
                num_classes,
            )
        )

    enc_named_params = [(n, p) for n, p in model.encoder.named_parameters() if p.requires_grad]
    if len(enc_named_params) == 0:
        raise RuntimeError("No encoder params found for gradient analysis.")

    params: List[torch.nn.Parameter] = []
    param_to_group: List[str] = []
    group_depth: Dict[str, int] = {}
    for n, p in enc_named_params:
        g, d = _layer_group_dinov2_wrapped(n)
        params.append(p)
        param_to_group.append(g)
        if g not in group_depth:
            group_depth[g] = d
        else:
            group_depth[g] = min(group_depth[g], d)

    var_loss_fn = FeatureVarianceLoss(gamma=1.0, eps=1e-4)
    rows = []
    for bi, (images, labels) in enumerate(loader):
        if bi >= args.num_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if args.understanding_loss == "dino_reg":
            x1, x2 = _build_views(images)
        else:
            x1 = images
            x2 = None

        tokens1, rec1 = _reconstruct_with_encoder_grads(model, x1)

        if args.understanding_loss == "dino_reg":
            tokens2 = _encode_tokens_with_grad(model, x2)
            # Understanding: DINO-style view consistency + variance regularization。
            feat1 = F.normalize(tokens1.mean(dim=1), dim=-1)
            feat2 = F.normalize(tokens2.mean(dim=1), dim=-1)
            lu_align = 1.0 - (feat1 * feat2).sum(dim=-1).mean()
            lu_var = var_loss_fn(tokens1.mean(dim=1)) + var_loss_fn(tokens2.mean(dim=1))
            lu = lu_align + float(args.lambda_var) * lu_var
        else:
            if cls_head is None:
                raise RuntimeError("cls_head is None while understanding_loss=cls_ce")
            # Understanding: 语义分类 CE（让理解分支显式走语义目标）。
            logits = cls_head(tokens1.mean(dim=1))
            lu = F.cross_entropy(logits, labels)

        # Generation: reconstruction MSE（官方 pretrained decoder）。
        target = x1
        if rec1.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=rec1.shape[-2:], mode="bilinear", align_corners=False)
        lg = F.mse_loss(rec1, target)

        gu = torch.autograd.grad(lu, params, retain_graph=True, allow_unused=True)
        gg = torch.autograd.grad(lg, params, retain_graph=False, allow_unused=True)

        by_group_u = defaultdict(list)
        by_group_g = defaultdict(list)
        for gname, p, g1, g2 in zip(param_to_group, params, gu, gg):
            by_group_u[gname].append(_tensor_to_flat(g1, p))
            by_group_g[gname].append(_tensor_to_flat(g2, p))

        for gname, depth in sorted(group_depth.items(), key=lambda x: (x[1], x[0])):
            gu_flat = torch.cat(by_group_u[gname], dim=0)
            gg_flat = torch.cat(by_group_g[gname], dim=0)
            rows.append(
                {
                    "batch_idx": bi,
                    "layer": gname,
                    "depth": depth,
                    "cos": _cosine(gu_flat, gg_flat),
                    "gu_norm": float(torch.norm(gu_flat).item()),
                    "gg_norm": float(torch.norm(gg_flat).item()),
                    "lu": float(lu.detach().item()),
                    "lg": float(lg.detach().item()),
                }
            )

    if len(rows) == 0:
        raise RuntimeError("No gradient rows collected.")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "layerwise_grad_cos.csv", index=False)

    summary = (
        df.groupby(["layer", "depth"], as_index=False)
        .agg(
            cos_mean=("cos", "mean"),
            cos_std=("cos", "std"),
            cos_neg_ratio=("cos", lambda x: float((x < 0).mean())),
            gu_norm_mean=("gu_norm", "mean"),
            gg_norm_mean=("gg_norm", "mean"),
            lu_mean=("lu", "mean"),
            lg_mean=("lg", "mean"),
            n=("cos", "count"),
        )
        .sort_values(["depth", "layer"])
    )
    summary["cos_std"] = summary["cos_std"].fillna(0.0)
    summary.to_csv(out_dir / "layerwise_grad_summary.csv", index=False)

    global_cos_mean = float(df["cos"].mean())
    global_neg_ratio = float((df["cos"] < 0).mean())
    deep = summary[summary["depth"] >= summary["depth"].median()]
    shallow = summary[summary["depth"] < summary["depth"].median()]
    stats = {
        "dataset": f"{args.hf_dataset}:{args.hf_config}:{args.split}",
        "understanding_loss": str(args.understanding_loss),
        "num_batches": int(df["batch_idx"].nunique()),
        "num_layers": int(summary.shape[0]),
        "global_cos_mean": global_cos_mean,
        "global_neg_ratio": global_neg_ratio,
        "shallow_mean_cos": float(shallow["cos_mean"].mean()) if len(shallow) else float("nan"),
        "deep_mean_cos": float(deep["cos_mean"].mean()) if len(deep) else float("nan"),
        "shallow_neg_ratio": float(shallow["cos_neg_ratio"].mean()) if len(shallow) else float("nan"),
        "deep_neg_ratio": float(deep["cos_neg_ratio"].mean()) if len(deep) else float("nan"),
        "decoder_ckpt": str(dec_ckpt),
        "stat_ckpt": str(stat_ckpt),
        "cls_warmup": cls_warmup,
    }
    save_json(stats, str(out_dir / "conflict_stats.json"))

    _make_plots(df=df, summary=summary, out_dir=out_dir)

    print(f"[grad_conflict_rae_pretrained] out_dir={out_dir}")
    print(
        "[grad_conflict_rae_pretrained] global_cos_mean={:.4f}, global_neg_ratio={:.3f}, shallow_mean={:.4f}, deep_mean={:.4f}".format(
            stats["global_cos_mean"],
            stats["global_neg_ratio"],
            stats["shallow_mean_cos"],
            stats["deep_mean_cos"],
        )
    )


if __name__ == "__main__":
    main()
