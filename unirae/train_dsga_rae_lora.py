from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
from tqdm import tqdm

from .grad_conflict import _cagrad_like_merge, apply_ma_laga, project_conflicting
from .train_dino_rae_stage1 import (
    _calculate_adaptive_weight,
    _import_rae_modules,
    _select_gan_losses,
)
from .utils import append_jsonl, ensure_dir, save_json


class HFImageDataset(Dataset):
    def __init__(self, hf_ds, transform, image_key: str = "image", label_key: str = "label"):
        self.ds = hf_ds
        self.transform = transform
        self.image_key = str(image_key)
        self.label_key = str(label_key)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        image = item[self.image_key]
        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")
        x = self.transform(image)
        y = int(item.get(self.label_key, -1))
        return x, y


def _build_rae_transform(image_size: int, split: str):
    first_crop = 384 if image_size == 256 else int(image_size * 1.5)
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize(first_crop, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(image_size),
                transforms.ToTensor(),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(first_crop, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def _infer_num_classes(hf_ds, label_key: str = "label", fallback: int = 1000) -> int:
    observed_n = None
    if hasattr(hf_ds, "unique"):
        try:
            uniq = hf_ds.unique(label_key)
            if len(uniq) > 0:
                observed_n = int(max(int(x) for x in uniq) + 1)
        except Exception:
            pass
    feat = None
    if hasattr(hf_ds, "features"):
        feat = hf_ds.features.get(label_key)
    if feat is not None and hasattr(feat, "names") and feat.names is not None and len(feat.names) > 0:
        names_n = int(len(feat.names))
        if observed_n is not None and observed_n > 0 and observed_n <= names_n:
            return observed_n
        return names_n
    if observed_n is not None:
        return observed_n
    return int(fallback)


def _build_data_loader(
    hf_dataset: str,
    hf_config: str,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    is_train: bool,
    image_key: str = "image",
    label_key: str = "label",
):
    cfg = hf_config if str(hf_config).strip() else None
    hf_ds = load_dataset(str(hf_dataset), cfg, split=str(split))
    ds = HFImageDataset(hf_ds, transform=_build_rae_transform(int(image_size), "train" if is_train else "val"), image_key=image_key, label_key=label_key)
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(is_train),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(is_train),
    )
    return loader, _infer_num_classes(hf_ds, label_key=label_key)


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


def _encode_tokens_with_encoder_grads(rae_model, images_01: torch.Tensor) -> torch.Tensor:
    h, w = images_01.shape[-2:]
    if h != int(rae_model.encoder_input_size) or w != int(rae_model.encoder_input_size):
        images_01 = F.interpolate(
            images_01,
            size=(int(rae_model.encoder_input_size), int(rae_model.encoder_input_size)),
            mode="bicubic",
            align_corners=False,
        )
    x = (images_01 - rae_model.encoder_mean.to(images_01)) / rae_model.encoder_std.to(images_01)
    return rae_model.encoder(x)


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


def _reconstruct_with_encoder_grads(rae_model, images_01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = _encode_tokens_with_encoder_grads(rae_model, images_01)
    z = _prepare_latent_for_decode(rae_model, tokens)
    rec = rae_model.decode(z)
    return tokens, rec


class _RAEReconstructWrapper(nn.Module):
    def __init__(self, rae_model: nn.Module):
        super().__init__()
        self.rae_model = rae_model

    def forward(self, images_01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _reconstruct_with_encoder_grads(self.rae_model, images_01)


def _compute_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    kind: str,
    rmse_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mse = F.mse_loss(pred, target)
    kind = str(kind).lower()
    if kind == "mse":
        return mse, mse
    if kind == "rmse":
        return torch.sqrt(mse + float(rmse_eps)), mse
    if kind == "l1":
        return F.l1_loss(pred, target), mse
    raise ValueError(f"Unsupported recon_loss={kind}. Use one of: l1|mse|rmse.")


def _materialize_grads(
    grads: Sequence[Optional[torch.Tensor]],
    params: Sequence[nn.Parameter],
) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for g, p in zip(grads, params):
        out.append(torch.zeros_like(p, memory_format=torch.preserve_format) if g is None else g.detach())
    return out


def _accumulate_into(dst: List[torch.Tensor], src: Sequence[torch.Tensor]) -> None:
    for i, g in enumerate(src):
        dst[i] = dst[i] + g


def _assign_grads(params: Sequence[nn.Parameter], grads: Sequence[torch.Tensor]) -> None:
    for p, g in zip(params, grads):
        p.grad = g


def _clone_zero_grads(params: Sequence[nn.Parameter]) -> List[torch.Tensor]:
    return [torch.zeros_like(p, memory_format=torch.preserve_format) for p in params]


def _flatten(grads: Sequence[torch.Tensor]) -> torch.Tensor:
    if not grads:
        return torch.zeros(1)
    return torch.cat([g.reshape(-1) for g in grads], dim=0)


def _layer_group_dinov2_wrapped(name: str) -> Tuple[str, int]:
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


def _build_encoder_groups(named_params: Sequence[Tuple[str, nn.Parameter]], grouping: str) -> Dict[str, List[int]]:
    grouping = str(grouping).lower()
    if grouping == "global":
        return {"global": list(range(len(named_params)))}
    if grouping != "layerwise":
        raise ValueError(f"Unsupported dsga_grouping={grouping}. Use global|layerwise.")
    groups: Dict[str, List[int]] = {}
    for idx, (name, _) in enumerate(named_params):
        group_name, _ = _layer_group_dinov2_wrapped(name)
        groups.setdefault(group_name, []).append(idx)
    return groups


class _FrozenCLIPVisionEncoder:
    def __init__(self, model_name: str, input_size: int, device: torch.device):
        from transformers import AutoImageProcessor, CLIPVisionModelWithProjection

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(device).eval()
        self.input_size = int(input_size)
        self.device = device
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, images_01: torch.Tensor) -> torch.Tensor:
        pil_images = []
        for img in images_01.detach().cpu():
            pil_images.append(to_pil_image(img.clamp(0.0, 1.0)))
        batch = self.processor(images=pil_images, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(**batch)
        feats = out.image_embeds
        return F.normalize(feats, dim=-1)


@dataclass
class TrainStepStats:
    loss_u: float
    loss_g_obj: float
    recon_mse: float
    train_acc: float
    train_rmse: float
    grad_cos_mean: float
    grad_neg_ratio: float
    lpips: float
    gan_g: float
    d_weight: float


def _merge_encoder_grads(
    grads_u: Sequence[torch.Tensor],
    grads_g: Sequence[torch.Tensor],
    strategy: str,
    cagrad_beta: float,
    dsga_groups: Dict[str, List[int]],
    dsga_align_gamma: float,
    dsga_conflict_tau: float,
    dsga_grouping: str,
    dsga_magnitude_scope: str,
    dsga_norm_restore: bool,
    dsga_mode: str,
) -> List[torch.Tensor]:
    st = str(strategy).lower()
    if st == "naive":
        return [gu + gg for gu, gg in zip(grads_u, grads_g)]
    if st == "pcgrad":
        gu_proj, gg_proj = project_conflicting(grads_u, grads_g)
        return [gu + gg for gu, gg in zip(gu_proj, gg_proj)]
    if st == "cagrad":
        return _cagrad_like_merge(grads_u, grads_g, beta=float(cagrad_beta))
    if st == "dsga":
        return apply_ma_laga(
            grads_u=grads_u,
            grads_g=grads_g,
            layers=dsga_groups if str(dsga_grouping).lower() == "layerwise" else {"global": list(range(len(grads_u)))},
            preserve_target="understanding",
            align_gamma=float(dsga_align_gamma),
            norm_restore=bool(dsga_norm_restore),
            mode=str(dsga_mode),
            conflict_threshold=float(dsga_conflict_tau),
            magnitude_scope=str(dsga_magnitude_scope),
        )
    raise ValueError(f"Unsupported shared_strategy={strategy}. Use naive|pcgrad|cagrad|dsga.")


def _safe_mean(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) == 0:
        return float("nan")
    return float(sum(vals) / len(vals))


@torch.no_grad()
def _evaluate_model(
    model,
    und_head: nn.Module,
    val_loader,
    *,
    device: torch.device,
    understanding_loss: str,
    clip_vision_encoder: Optional[_FrozenCLIPVisionEncoder],
    recon_loss: str,
    recon_rmse_eps: float,
    max_batches: int,
    compute_rfid: bool,
    rfid_num_samples: int,
    rfid_batch_size: int,
    rfid_tmp_dir: str,
) -> Dict[str, float]:
    model.eval()
    und_head.eval()

    sum_mse = 0.0
    sum_u = 0.0
    sum_psnr = 0.0
    n_samples = 0

    tmp_root = None
    real_dir = None
    fake_dir = None
    saved = 0
    if compute_rfid:
        tmp_root = tempfile.TemporaryDirectory(prefix="dsga_rae_rfid_", dir=(rfid_tmp_dir or None))
        root = Path(tmp_root.name)
        real_dir = root / "real"
        fake_dir = root / "fake"
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)

    for bi, (images, labels) in enumerate(val_loader):
        if int(max_batches) > 0 and bi >= int(max_batches):
            break
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
            raise ValueError(f"Unsupported understanding_loss={understanding_loss}")

        _, mse = _compute_recon_loss(rec, target, kind=str(recon_loss), rmse_eps=float(recon_rmse_eps))
        mse_per = ((rec - target) ** 2).flatten(1).mean(dim=1)
        psnr_per = -10.0 * torch.log10(mse_per + 1e-8)
        bs = int(images.shape[0])
        n_samples += bs
        sum_mse += float(mse_per.sum().item())
        sum_psnr += float(psnr_per.sum().item())
        sum_u += float(u_metric.item()) * bs

        if compute_rfid and real_dir is not None and fake_dir is not None:
            for i in range(bs):
                if saved >= int(rfid_num_samples):
                    break
                save_image(target[i].cpu(), str(real_dir / f"{saved:07d}.png"))
                save_image(rec[i].cpu(), str(fake_dir / f"{saved:07d}.png"))
                saved += 1

    val_mse = float(sum_mse / max(n_samples, 1))
    val_rmse = float((val_mse + max(0.0, float(recon_rmse_eps))) ** 0.5) if str(recon_loss).lower() == "rmse" else float(val_mse**0.5)
    val_psnr = float(sum_psnr / max(n_samples, 1))
    val_u = float(sum_u / max(n_samples, 1))

    rfid = float("nan")
    rfid_error = ""
    if compute_rfid and saved > 1 and real_dir is not None and fake_dir is not None:
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
    if tmp_root is not None:
        tmp_root.cleanup()

    out = {
        "eval_mse": val_mse,
        "eval_rmse": val_rmse,
        "eval_psnr": val_psnr,
        "eval_num_samples": int(n_samples),
        "val_rfid": rfid,
        "rfid_num_samples": int(saved),
        "rfid_error": rfid_error,
    }
    if str(understanding_loss).lower() == "ce":
        out["eval_acc"] = val_u
    else:
        out["eval_u_cosine"] = val_u
    out["understanding"] = {
        "acc_txt": float(val_u) if str(understanding_loss).lower() == "ce" else float("nan"),
        "u_cosine": float(val_u) if str(understanding_loss).lower() == "clip_cosine" else float("nan"),
    }
    out["generation"] = {
        "recon_rmse": float(val_rmse),
        "recon_mse": float(val_mse),
        "psnr": float(val_psnr),
        "rfid": float(rfid),
    }
    return out


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _trim_jsonl_to_step(path: Path, max_step: int) -> None:
    if not path.exists():
        return
    kept: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            kept.append(line)
            continue
        step = row.get("step", None)
        if step is None or int(step) <= int(max_step):
            kept.append(json.dumps(row, ensure_ascii=False))
    payload = ("\n".join(kept) + "\n") if kept else ""
    path.write_text(payload, encoding="utf-8")


def _save_checkpoint(
    run_dir: Path,
    *,
    model: nn.Module,
    und_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    discriminator: nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> Path:
    ckpt = {
        "model": model.state_dict(),
        "und_head": und_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "discriminator": discriminator.state_dict(),
        "disc_optimizer": disc_optimizer.state_dict(),
        "step": int(step),
        "args": vars(args),
    }
    ckpt_path = run_dir / "latest.pt"
    torch.save(ckpt, str(ckpt_path))
    return ckpt_path


def _load_checkpoint(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    und_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    discriminator: nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")
    model.load_state_dict(ckpt["model"], strict=False)
    if "und_head" in ckpt:
        und_head.load_state_dict(ckpt["und_head"], strict=False)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        _optimizer_to_device(optimizer, device)
    if "discriminator" in ckpt:
        discriminator.load_state_dict(ckpt["discriminator"], strict=False)
    if "disc_optimizer" in ckpt:
        disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
        _optimizer_to_device(disc_optimizer, device)
    return int(ckpt.get("step", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint RAE full-update training with gradient arbitration.")
    parser.add_argument("--out_dir", default="runs")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hf_dataset", default="clane9/imagenet-100")
    parser.add_argument("--hf_config", default="")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="validation")
    parser.add_argument("--image_key", default="image")
    parser.add_argument("--label_key", default="label")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_max_batches", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=20)

    parser.add_argument("--lambda_u", type=float, default=1.0)
    parser.add_argument("--lambda_g", type=float, default=1.0)
    parser.add_argument("--understanding_loss", default="ce", choices=["ce", "clip_cosine"])
    parser.add_argument("--recon_loss", default="rmse", choices=["l1", "mse", "rmse"])
    parser.add_argument("--recon_rmse_eps", type=float, default=1e-12)
    parser.add_argument("--lpips_weight", type=float, default=1.0)
    parser.add_argument("--gan_weight", type=float, default=0.75)
    parser.add_argument("--lpips_start_step", type=int, default=0)
    parser.add_argument("--gan_start_step", type=int, default=1000)
    parser.add_argument("--disc_update_start_step", type=int, default=750)

    parser.add_argument("--shared_strategy", default="naive", choices=["naive", "pcgrad", "cagrad", "dsga"])
    parser.add_argument("--cagrad_beta", type=float, default=0.35)
    parser.add_argument("--dsga_grouping", default="layerwise", choices=["global", "layerwise"])
    parser.add_argument("--dsga_align_gamma", type=float, default=0.5)
    parser.add_argument("--dsga_conflict_tau", type=float, default=0.0)
    parser.add_argument("--dsga_magnitude_scope", default="global", choices=["global", "layerwise"])
    parser.add_argument("--dsga_mode", default="full", choices=["full", "direction_only", "magnitude_only", "nr_laga", "capped_full"])
    parser.add_argument("--dsga_norm_restore", action="store_true", default=True)
    parser.add_argument("--no_dsga_norm_restore", dest="dsga_norm_restore", action="store_false")

    parser.add_argument("--encoder_update", default="full", choices=["full", "frozen"])
    parser.add_argument("--lr_encoder", type=float, default=2e-5)
    parser.add_argument("--lr_decoder", type=float, default=2e-5)
    parser.add_argument("--lr_und", type=float, default=1e-4)
    parser.add_argument("--lr_disc", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--clip_grad", type=float, default=1.0)

    parser.add_argument("--rae_code_root", default="/project/peilab/luxiaocheng/projects/RAE")
    parser.add_argument("--decoder_config_path", default="")
    parser.add_argument("--pretrained_decoder_path", default="")
    parser.add_argument("--normalization_stat_path", default="")

    parser.add_argument("--final_eval_rfid", action="store_true", default=False)
    parser.add_argument("--no_final_eval_rfid", dest="final_eval_rfid", action="store_false")
    parser.add_argument("--rfid_num_samples", type=int, default=5000)
    parser.add_argument("--rfid_batch_size", type=int, default=64)
    parser.add_argument("--rfid_tmp_dir", default="/project/peilab/luxiaocheng/projects/DSGA/results/in100_rfid_tmp")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--data_parallel", action="store_true", default=False)
    parser.add_argument("--no_data_parallel", dest="data_parallel", action="store_false")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--resume_from", default="")
    args = parser.parse_args()

    _set_seed(int(args.seed))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_dir = Path(args.out_dir) / args.run_name
    ensure_dir(str(run_dir))
    ensure_dir(str(Path(args.rfid_tmp_dir)))

    train_loader, num_classes = _build_data_loader(
        hf_dataset=str(args.hf_dataset),
        hf_config=str(args.hf_config),
        split=str(args.train_split),
        image_size=int(args.image_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        is_train=True,
        image_key=str(args.image_key),
        label_key=str(args.label_key),
    )
    val_loader, _ = _build_data_loader(
        hf_dataset=str(args.hf_dataset),
        hf_config=str(args.hf_config),
        split=str(args.val_split),
        image_size=int(args.image_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        is_train=False,
        image_key=str(args.image_key),
        label_key=str(args.label_key),
    )

    dec_cfg, dec_ckpt, stat_ckpt = _resolve_rae_defaults(
        rae_code_root=str(args.rae_code_root),
        decoder_config_path=str(args.decoder_config_path),
        pretrained_decoder_path=str(args.pretrained_decoder_path),
        normalization_stat_path=str(args.normalization_stat_path),
    )
    for p in [dec_cfg, dec_ckpt, stat_ckpt]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required file/path not found: {p}")

    RAE = _import_rae_class(str(args.rae_code_root))
    model = RAE(
        encoder_cls="Dinov2withNorm",
        encoder_config_path="facebook/dinov2-with-registers-base",
        encoder_input_size=224,
        encoder_params={"dinov2_path": "facebook/dinov2-with-registers-base", "normalize": True},
        decoder_config_path=dec_cfg,
        pretrained_decoder_path=dec_ckpt,
        noise_tau=0.8,
        reshape_to_2d=True,
        normalization_stat_path=stat_ckpt,
    ).to(device)
    und_out_dim = int(num_classes) if str(args.understanding_loss).lower() == "ce" else int(model.latent_dim)
    und_head = nn.Linear(int(model.latent_dim), int(und_out_dim)).to(device)
    visible_cuda_count = int(torch.cuda.device_count()) if device.type == "cuda" else 0
    use_data_parallel = bool(args.data_parallel) and device.type == "cuda" and visible_cuda_count > 1
    recon_model: nn.Module = _RAEReconstructWrapper(model).to(device)
    if use_data_parallel:
        device_ids = list(range(visible_cuda_count))
        recon_model = nn.DataParallel(recon_model, device_ids=device_ids)
        print(f"[data_parallel] enabled with device_ids={device_ids}")

    for p in model.encoder.parameters():
        p.requires_grad = bool(str(args.encoder_update).lower() == "full")
    for p in model.decoder.parameters():
        p.requires_grad = True
    for p in und_head.parameters():
        p.requires_grad = True

    encoder_named_params = [(n, p) for n, p in model.encoder.named_parameters() if p.requires_grad]
    encoder_params = [p for _, p in encoder_named_params]
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
    und_params = [p for p in und_head.parameters() if p.requires_grad]
    if str(args.encoder_update).lower() == "full" and len(encoder_params) == 0:
        raise RuntimeError("encoder_update=full but no trainable encoder params were found.")

    optimizer_groups = []
    if len(encoder_params) > 0:
        optimizer_groups.append({"params": encoder_params, "lr": float(args.lr_encoder)})
    optimizer_groups.append({"params": decoder_params, "lr": float(args.lr_decoder)})
    optimizer_groups.append({"params": und_params, "lr": float(args.lr_und)})
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=(0.9, 0.95),
        weight_decay=float(args.weight_decay),
    )

    modules = _import_rae_modules(str(args.rae_code_root))
    discriminator, disc_aug = modules.build_discriminator(
        {
            "arch": {
                "dino_ckpt_path": str(Path(args.rae_code_root) / "models" / "discs" / "dino_vit_small_patch8_224.pth"),
                "ks": 9,
                "norm_type": "bn",
                "using_spec_norm": True,
                "recipe": "S_8",
            },
            "augment": {"prob": 1.0, "cutout": 0.0},
        },
        device,
    )
    discriminator = discriminator.to(device)
    disc_loss_fn, gen_loss_fn = _select_gan_losses("hinge", "vanilla", modules.hinge_d_loss, modules.vanilla_d_loss, modules.vanilla_g_loss)
    lpips_metric = modules.LPIPS().to(device).eval()
    for p in lpips_metric.parameters():
        p.requires_grad = False
    disc_params = [p for p in discriminator.parameters() if p.requires_grad]
    disc_optimizer = torch.optim.AdamW(disc_params, lr=float(args.lr_disc), betas=(0.9, 0.95), weight_decay=float(args.weight_decay))

    clip_vision_encoder = None
    if str(args.understanding_loss).lower() == "clip_cosine":
        clip_vision_encoder = _FrozenCLIPVisionEncoder(
            model_name="openai/clip-vit-base-patch16",
            input_size=int(args.image_size),
            device=device,
        )

    dsga_groups = _build_encoder_groups(encoder_named_params, grouping=str(args.dsga_grouping)) if len(encoder_named_params) > 0 else {"global": []}
    metrics_path = run_dir / "train_metrics.jsonl"
    eval_metrics_path = run_dir / "eval_metrics.jsonl"
    setup = {
        "args": vars(args),
        "decoder_config_path": dec_cfg,
        "pretrained_decoder_path": dec_ckpt,
        "normalization_stat_path": stat_ckpt,
        "num_classes": int(num_classes),
        "num_trainable_encoder": int(sum(p.numel() for p in encoder_params)),
        "num_trainable_decoder": int(sum(p.numel() for p in decoder_params)),
        "num_trainable_und_head": int(sum(p.numel() for p in und_params)),
        "device": str(device),
        "visible_cuda_count": int(visible_cuda_count),
        "data_parallel": bool(use_data_parallel),
    }
    save_json(setup, str(run_dir / "run_setup.json"))

    start_step = 0
    resume_path = None
    if str(args.resume_from).strip():
        resume_path = Path(str(args.resume_from).strip())
    elif bool(args.resume):
        resume_path = run_dir / "latest.pt"
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        start_step = _load_checkpoint(
            resume_path,
            model=model,
            und_head=und_head,
            optimizer=optimizer,
            discriminator=discriminator,
            disc_optimizer=disc_optimizer,
            device=device,
        )
        _trim_jsonl_to_step(metrics_path, start_step)
        _trim_jsonl_to_step(eval_metrics_path, start_step)
        print(f"[resume] loaded checkpoint={resume_path} start_step={start_step}")

    train_iter = iter(train_loader)
    pbar = tqdm(range(start_step + 1, int(args.steps) + 1), desc=f"train_{args.shared_strategy}", disable=False)
    for step in pbar:
        recon_model.train()
        model.train()
        und_head.train()
        discriminator.eval()

        optimizer.zero_grad(set_to_none=True)
        enc_u_acc = _clone_zero_grads(encoder_params)
        enc_g_acc = _clone_zero_grads(encoder_params)
        dec_g_acc = _clone_zero_grads(decoder_params)
        und_u_acc = _clone_zero_grads(und_params)

        loss_u_vals: List[float] = []
        loss_g_vals: List[float] = []
        mse_vals: List[float] = []
        acc_vals: List[float] = []
        rmse_vals: List[float] = []
        cos_vals: List[float] = []
        neg_vals: List[float] = []
        lpips_vals: List[float] = []
        gan_vals: List[float] = []
        dweight_vals: List[float] = []
        last_disc_images = None
        last_disc_recon = None

        for _ in range(int(args.grad_accum_steps)):
            try:
                images, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, labels = next(train_iter)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            tokens, rec = recon_model(images)
            pooled = tokens.mean(dim=1)

            if str(args.understanding_loss).lower() == "ce":
                logits = und_head(pooled)
                raw_u = F.cross_entropy(logits, labels)
                with torch.no_grad():
                    train_acc = float((logits.argmax(dim=-1) == labels).float().mean().item())
            else:
                if clip_vision_encoder is None:
                    raise ValueError("clip_vision_encoder is required when understanding_loss=clip_cosine")
                pred = F.normalize(und_head(pooled), dim=-1)
                tgt = clip_vision_encoder.encode(images)
                raw_u = 1.0 - (pred * tgt).sum(dim=-1).mean()
                with torch.no_grad():
                    train_acc = float((pred * tgt).sum(dim=-1).mean().item())

            target = images
            if rec.shape[-2:] != target.shape[-2:]:
                target = F.interpolate(target, size=rec.shape[-2:], mode="bilinear", align_corners=False)
            rec_loss, mse = _compute_recon_loss(rec, target, kind=str(args.recon_loss), rmse_eps=float(args.recon_rmse_eps))
            rmse = torch.sqrt(mse.detach().clamp_min(0.0))

            lpips_loss = rec_loss.new_zeros(())
            gan_g_loss = rec_loss.new_zeros(())
            d_weight = rec_loss.new_zeros(())
            if step >= int(args.lpips_start_step) and float(args.lpips_weight) > 0.0:
                real_normed = target * 2.0 - 1.0
                recon_normed = rec.clamp(0.0, 1.0) * 2.0 - 1.0
                lpips_loss = lpips_metric(real_normed, recon_normed).mean()
            gen_objective = rec_loss + float(args.lpips_weight) * lpips_loss
            if step >= int(args.gan_start_step) and float(args.gan_weight) > 0.0:
                recon_normed = rec.clamp(0.0, 1.0) * 2.0 - 1.0
                fake_aug = disc_aug.aug(recon_normed)
                logits_fake, _ = discriminator(fake_aug, None)
                gan_g_loss = gen_loss_fn(logits_fake)
                d_weight = _calculate_adaptive_weight(gen_objective, gan_g_loss, model.decoder.project_out.weight if hasattr(model.decoder, "project_out") else next(model.decoder.parameters()))
                gen_objective = gen_objective + float(args.gan_weight) * d_weight * gan_g_loss

            scaled_u = float(args.lambda_u) * raw_u / float(max(1, args.grad_accum_steps))
            scaled_g = float(args.lambda_g) * gen_objective / float(max(1, args.grad_accum_steps))

            if len(encoder_params) > 0:
                gu = _materialize_grads(torch.autograd.grad(scaled_u, encoder_params, retain_graph=True, allow_unused=True), encoder_params)
                gg = _materialize_grads(torch.autograd.grad(scaled_g, encoder_params, retain_graph=True, allow_unused=True), encoder_params)
                _accumulate_into(enc_u_acc, gu)
                _accumulate_into(enc_g_acc, gg)
                vu = _flatten(gu)
                vg = _flatten(gg)
                denom = float((vu.norm() * vg.norm()).clamp_min(1e-12).item())
                cos = float((torch.dot(vu, vg) / max(denom, 1e-12)).item()) if vu.numel() == vg.numel() else 0.0
                cos_vals.append(cos)
                neg_vals.append(1.0 if cos < 0.0 else 0.0)

            und_u = _materialize_grads(torch.autograd.grad(scaled_u, und_params, retain_graph=True, allow_unused=True), und_params)
            dec_g = _materialize_grads(torch.autograd.grad(scaled_g, decoder_params, retain_graph=True, allow_unused=True), decoder_params)
            _accumulate_into(und_u_acc, und_u)
            _accumulate_into(dec_g_acc, dec_g)

            loss_u_vals.append(float(raw_u.detach().item()))
            loss_g_vals.append(float(gen_objective.detach().item()))
            mse_vals.append(float(mse.detach().item()))
            acc_vals.append(float(train_acc))
            rmse_vals.append(float(rmse.item()))
            lpips_vals.append(float(lpips_loss.detach().item()))
            gan_vals.append(float(gan_g_loss.detach().item()))
            dweight_vals.append(float(d_weight.detach().item()))
            last_disc_images = target.detach()
            last_disc_recon = rec.detach().clamp(0.0, 1.0)

        if len(encoder_params) > 0:
            enc_merged = _merge_encoder_grads(
                grads_u=enc_u_acc,
                grads_g=enc_g_acc,
                strategy=str(args.shared_strategy),
                cagrad_beta=float(args.cagrad_beta),
                dsga_groups=dsga_groups,
                dsga_align_gamma=float(args.dsga_align_gamma),
                dsga_conflict_tau=float(args.dsga_conflict_tau),
                dsga_grouping=str(args.dsga_grouping),
                dsga_magnitude_scope=str(args.dsga_magnitude_scope),
                dsga_norm_restore=bool(args.dsga_norm_restore),
                dsga_mode=str(args.dsga_mode),
            )
            _assign_grads(encoder_params, enc_merged)
        _assign_grads(decoder_params, dec_g_acc)
        _assign_grads(und_params, und_u_acc)

        if float(args.clip_grad) > 0:
            clip_params = [p for p in list(encoder_params) + list(decoder_params) + list(und_params) if p.grad is not None]
            if len(clip_params) > 0:
                nn.utils.clip_grad_norm_(clip_params, float(args.clip_grad))
        optimizer.step()

        disc_loss_val = 0.0
        disc_acc_val = 0.0
        if (
            last_disc_images is not None
            and last_disc_recon is not None
            and step >= int(args.disc_update_start_step)
            and float(args.gan_weight) > 0.0
        ):
            discriminator.train()
            disc_optimizer.zero_grad(set_to_none=True)
            real_normed = last_disc_images * 2.0 - 1.0
            fake_detached = last_disc_recon * 2.0 - 1.0
            fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0
            fake_input = disc_aug.aug(fake_detached)
            real_input = disc_aug.aug(real_normed)
            logits_fake, logits_real = discriminator(fake_input, real_input)
            d_loss = disc_loss_fn(logits_real, logits_fake)
            d_loss.backward()
            disc_optimizer.step()
            disc_loss_val = float(d_loss.detach().item())
            disc_acc_val = float((logits_real > logits_fake).float().mean().item())
            discriminator.eval()

        row = {
            "step": int(step),
            "loss_u": _safe_mean(loss_u_vals),
            "loss_g_obj": _safe_mean(loss_g_vals),
            "mse": _safe_mean(mse_vals),
            "rmse": _safe_mean(rmse_vals),
            "acc": _safe_mean(acc_vals),
            "grad_cos_mean": _safe_mean(cos_vals),
            "grad_neg_ratio": _safe_mean(neg_vals),
            "lpips": _safe_mean(lpips_vals),
            "gan_g": _safe_mean(gan_vals),
            "d_weight": _safe_mean(dweight_vals),
            "disc_loss": float(disc_loss_val),
            "disc_acc": float(disc_acc_val),
            "lr_encoder": float(args.lr_encoder),
            "lr_decoder": float(args.lr_decoder),
            "lr_und": float(args.lr_und),
            "lr_disc": float(args.lr_disc),
        }
        append_jsonl(str(metrics_path), row)
        if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
            pbar.set_postfix(
                lu=f"{row['loss_u']:.4f}",
                lg=f"{row['loss_g_obj']:.4f}",
                acc=f"{row['acc']:.3f}",
                rmse=f"{row['rmse']:.4f}",
            )

        if int(args.eval_every) > 0 and (step % int(args.eval_every) == 0 or step == int(args.steps)):
            eval_ret = _evaluate_model(
                model=model,
                und_head=und_head,
                val_loader=val_loader,
                device=device,
                understanding_loss=str(args.understanding_loss),
                clip_vision_encoder=clip_vision_encoder,
                recon_loss=str(args.recon_loss),
                recon_rmse_eps=float(args.recon_rmse_eps),
                max_batches=int(args.eval_max_batches),
                compute_rfid=False,
                rfid_num_samples=0,
                rfid_batch_size=int(args.rfid_batch_size),
                rfid_tmp_dir=str(args.rfid_tmp_dir),
            )
            eval_row = {"step": int(step), **eval_ret}
            append_jsonl(str(eval_metrics_path), eval_row)
            save_json(eval_row, str(run_dir / "eval_last.json"))
            u_key = "eval_acc" if str(args.understanding_loss).lower() == "ce" else "eval_u_cosine"
            print(
                f"[eval] step={step} {u_key}={eval_ret.get(u_key, float('nan')):.6f} "
                f"rmse={eval_ret['eval_rmse']:.6f}"
            )

        if (int(args.save_every) > 0 and step % int(args.save_every) == 0) or step == int(args.steps):
            ckpt_path = _save_checkpoint(
                run_dir,
                model=model,
                und_head=und_head,
                optimizer=optimizer,
                discriminator=discriminator,
                disc_optimizer=disc_optimizer,
                step=step,
                args=args,
            )
            print(f"[checkpoint] step={step} path={ckpt_path}")

    if bool(args.final_eval_rfid):
        final_ret = _evaluate_model(
            model=model,
            und_head=und_head,
            val_loader=val_loader,
            device=device,
            understanding_loss=str(args.understanding_loss),
            clip_vision_encoder=clip_vision_encoder,
            recon_loss=str(args.recon_loss),
            recon_rmse_eps=float(args.recon_rmse_eps),
            max_batches=0,
            compute_rfid=True,
            rfid_num_samples=int(args.rfid_num_samples),
            rfid_batch_size=int(args.rfid_batch_size),
            rfid_tmp_dir=str(args.rfid_tmp_dir),
        )
        final_row = {"step": int(args.steps), "final_eval": True, **final_ret}
        save_json(final_row, str(run_dir / "final_eval.json"))
        save_json(final_row, str(run_dir / "eval_last.json"))
        append_jsonl(str(eval_metrics_path), final_row)
        u_key = "eval_acc" if str(args.understanding_loss).lower() == "ce" else "eval_u_cosine"
        print(
            f"[final_eval] step={args.steps} {u_key}={final_ret.get(u_key, float('nan')):.6f} "
            f"rmse={final_ret['eval_rmse']:.6f} rfid={final_ret['val_rfid']:.4f}"
        )


if __name__ == "__main__":
    main()
