import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from .data_cifar10 import (
    build_cifar10_loader,
    denormalize_cifar10,
    make_batch_dict,
)
from .grad_conflict import (
    apply_naive,
    apply_cagrad,
    apply_conflict_aware,
    compute_grad_cosine,
)
from .models import build_backbone
from .utils import (
    append_jsonl,
    apply_overrides,
    count_parameters,
    cycle_loader,
    ensure_dir,
    load_yaml,
    now_str,
    save_json,
    save_yaml,
    to_device,
)


class CifarTradeoffModel(nn.Module):
    def __init__(self, cfg: Dict, num_classes: int):
        super().__init__()
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})

        image_size = int(data_cfg.get("image_size", 32))
        backbone_name = model_cfg.get("backbone", "resnet18")
        pretrained = bool(model_cfg.get("pretrained", False))

        self.backbone, feat_dim = build_backbone(
            backbone_name=backbone_name,
            image_size=image_size,
            pretrained=pretrained,
        )
        self.feat_dim = feat_dim

        txt_dim = int(model_cfg.get("txt_dim", 256))
        rec_dim = int(model_cfg.get("rec_dim", 256))
        self.recon_size = int(model_cfg.get("recon_size", image_size))
        hidden_dim = int(model_cfg.get("decoder_hidden_dim", 512))

        self.txt_head = nn.Linear(feat_dim, txt_dim)
        self.rec_head = nn.Linear(feat_dim, rec_dim)
        self.decoder = nn.Sequential(
            nn.Linear(rec_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3 * self.recon_size * self.recon_size),
        )

        self.text_prototypes = nn.Parameter(torch.randn(num_classes, txt_dim) * 0.02)

        if bool(model_cfg.get("freeze_backbone", False)):
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(images)
        z_txt = self.txt_head(feat)
        z_rec = self.rec_head(feat)
        recon = self.decoder(z_rec).view(images.shape[0], 3, self.recon_size, self.recon_size)
        return {
            "feat": feat,
            "z_txt": z_txt,
            "z_rec": z_rec,
            "recon": recon,
        }


def text_prototype_loss(
    z_txt: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    z = F.normalize(z_txt, dim=-1)
    p = F.normalize(prototypes, dim=-1)
    logits = z @ p.t()
    logits = logits / max(temperature, 1e-6)
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, {"txt_acc": acc}


def save_checkpoint(path: str, model: CifarTradeoffModel, optimizer: torch.optim.Optimizer, step: int, cfg: Dict) -> None:
    ensure_dir(str(Path(path).parent))
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": cfg,
        },
        path,
    )


def _build_shared_and_aux_params(model: CifarTradeoffModel, shared_mode: str) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    if shared_mode == "all":
        shared = [p for p in model.parameters() if p.requires_grad]
    else:
        shared = [p for p in model.backbone.parameters() if p.requires_grad]

    shared_ids = {id(p) for p in shared}
    aux = [p for p in model.parameters() if p.requires_grad and id(p) not in shared_ids]
    return shared, aux


def _dist_mean_scalar(accelerator: Accelerator, value, device: torch.device) -> float:
    if torch.is_tensor(value):
        t = value.detach().to(device)
        if t.ndim > 0:
            t = t.float().mean()
        else:
            t = t.float()
    else:
        t = torch.tensor(float(value), device=device, dtype=torch.float32)
    reduced = accelerator.reduce(t, reduction="mean")
    return float(reduced.item())


def _save_recon_samples(
    images: torch.Tensor,
    recons: torch.Tensor,
    dataset_name: str,
    out_path: str,
    max_items: int,
) -> None:
    b = min(max_items, images.shape[0])
    gt = torch.clamp(denormalize_cifar10(images[:b], dataset=dataset_name).cpu(), 0.0, 1.0)
    rc = torch.clamp(denormalize_cifar10(recons[:b], dataset=dataset_name).cpu(), 0.0, 1.0)

    rows = []
    for i in range(b):
        rows.append(gt[i])
        rows.append(rc[i])
    grid = make_grid(rows, nrow=2)
    ensure_dir(str(Path(out_path).parent))
    save_image(grid, out_path)


def evaluate_understanding(
    model: CifarTradeoffModel,
    loader,
    device: torch.device,
    class_names: List[str],
    temperature: float,
    max_batches: int,
) -> Dict[str, float]:
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            batch = make_batch_dict(batch)
            batch = to_device(batch, device)

            out = model(batch["images"])
            loss, _ = text_prototype_loss(
                z_txt=out["z_txt"],
                labels=batch["labels"],
                prototypes=model.text_prototypes,
                temperature=temperature,
            )

            logits = F.normalize(out["z_txt"], dim=-1) @ F.normalize(model.text_prototypes, dim=-1).t()
            pred = logits.argmax(dim=1)

            total += batch["labels"].numel()
            correct += (pred == batch["labels"]).sum().item()
            loss_sum += float(loss.item()) * batch["labels"].shape[0]

    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)
    return {
        "acc_txt": acc,
        "zero_shot_acc": acc,
        "zero_shot_loss": avg_loss,
        "num_samples": total,
        "class_names": list(class_names),
    }


def evaluate_generation(
    model: CifarTradeoffModel,
    loader,
    device: torch.device,
    dataset_name: str,
    max_batches: int,
    sample_path: str,
    save_samples: bool,
    sample_images: int,
) -> Dict[str, float]:
    model.eval()

    total_mse = 0.0
    n = 0
    saved = False

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            batch = make_batch_dict(batch)
            batch = to_device(batch, device)

            out = model(batch["images"])
            recon = out["recon"]
            target = batch["images"]

            mse = F.mse_loss(recon, target)
            total_mse += float(mse.item())
            n += 1

            if save_samples and (not saved):
                _save_recon_samples(
                    target,
                    recon,
                    dataset_name=dataset_name,
                    out_path=sample_path,
                    max_items=sample_images,
                )
                saved = True

    recon_mse = total_mse / max(n, 1)
    psnr = -10.0 * torch.log10(torch.tensor(recon_mse + 1e-8)).item()
    return {
        "recon_mode": "pixel_recon",
        "loss": recon_mse,
        "recon_mse": recon_mse,
        "mse": recon_mse,
        "psnr": psnr,
        "num_batches": n,
    }


def run_eval(
    cfg: Dict,
    model: CifarTradeoffModel,
    eval_loader,
    device: torch.device,
    run_dir: str,
    step: int,
    class_names: List[str],
    dataset_name: str,
) -> None:
    eval_cfg = cfg.get("eval", {})
    max_batches = eval_cfg.get("max_batches", None)

    understanding = evaluate_understanding(
        model=model,
        loader=eval_loader,
        device=device,
        class_names=class_names,
        temperature=float(cfg.get("text", {}).get("temperature", 0.07)),
        max_batches=max_batches,
    )

    generation = evaluate_generation(
        model=model,
        loader=eval_loader,
        device=device,
        dataset_name=dataset_name,
        max_batches=max_batches,
        sample_path=os.path.join(run_dir, "samples", f"step_{step:07d}.png"),
        save_samples=bool(eval_cfg.get("save_recon_samples", True)),
        sample_images=int(eval_cfg.get("sample_images", 8)),
    )

    save_json(understanding, os.path.join(run_dir, "understanding.json"))
    save_json(generation, os.path.join(run_dir, "generation.json"))
    save_json({"step": step, "understanding": understanding, "generation": generation}, os.path.join(run_dir, "eval_last.json"))


def _resolve_training_mode(cfg: Dict) -> Tuple[str, int, float, float, str]:
    train_cfg = cfg.get("train", {})

    mode = str(train_cfg.get("mode", "joint")).lower()
    steps = int(train_cfg.get("steps", 1000))
    lambda_txt = float(train_cfg.get("lambda_txt", 1.0))
    lambda_rec = float(train_cfg.get("lambda_rec", 1.0))
    strategy = str(train_cfg.get("grad_strategy", train_cfg.get("strategy", "naive"))).lower()

    if mode == "joint":
        if strategy == "conflict_aware":
            strategy = "pcgrad"
        if strategy not in {"naive", "pcgrad", "cagrad"}:
            raise ValueError(
                f"Unsupported train.strategy={strategy}. "
                "Only naive|pcgrad|cagrad are allowed for joint mode."
            )
        return mode, steps, lambda_txt, lambda_rec, strategy
    if mode == "text_only":
        return mode, steps, lambda_txt, 0.0, "naive"
    if mode == "recon_only":
        return mode, steps, 0.0, lambda_rec, "naive"
    raise ValueError(
        f"Unsupported train.mode={mode}. "
        "Only joint|text_only|recon_only are supported."
    )


def _resolve_grad_norm_mode(cfg: Dict, mode: str) -> str:
    train_cfg = cfg.get("train", {})
    raw = train_cfg.get("grad_norm_mode", train_cfg.get("grad_norm", "none"))

    if isinstance(raw, bool):
        grad_norm_mode = "mean" if raw else "none"
    else:
        grad_norm_mode = str(raw).lower()
        if grad_norm_mode in {"true", "on", "1"}:
            grad_norm_mode = "mean"
        if grad_norm_mode in {"false", "off", "0"}:
            grad_norm_mode = "none"

    if mode != "joint":
        return "none"
    if grad_norm_mode not in {"none", "mean", "geom", "unit"}:
        raise ValueError(
            f"Unsupported train.grad_norm_mode={grad_norm_mode}. "
            "Use one of: none|mean|geom|unit."
        )
    return grad_norm_mode


def _parse_grad_norm_layers(raw_layers) -> List[str]:
    if raw_layers is None:
        return ["layer3", "layer4"]
    if isinstance(raw_layers, str):
        norm = raw_layers.replace("+", ",")
        items = [x.strip().lower() for x in norm.split(",")]
        items = [x for x in items if x]
        return items if items else ["layer3", "layer4"]
    if isinstance(raw_layers, (list, tuple)):
        items = [str(x).strip().lower() for x in raw_layers if str(x).strip()]
        return items if items else ["layer3", "layer4"]
    return ["layer3", "layer4"]


def _layer_tag_from_param_name(name: str) -> str:
    n = str(name)
    if n.startswith("backbone."):
        n = n[len("backbone.") :]

    if n.startswith("stem.0.") or n.startswith("stem.1."):
        return "stem"
    if n.startswith("stem.4.") or n.startswith("layer1."):
        return "layer1"
    if n.startswith("stem.5.") or n.startswith("layer2."):
        return "layer2"
    if n.startswith("stem.6.") or n.startswith("layer3."):
        return "layer3"
    if n.startswith("stem.7.") or n.startswith("layer4."):
        return "layer4"
    return "other"


def _build_shared_param_names(
    model: CifarTradeoffModel,
    shared_params: List[nn.Parameter],
    shared_mode: str,
) -> List[str]:
    if shared_mode == "all":
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    else:
        named = [(f"backbone.{n}", p) for n, p in model.backbone.named_parameters() if p.requires_grad]

    id_to_idx = {id(p): i for i, p in enumerate(shared_params)}
    out = [""] * len(shared_params)
    for n, p in named:
        idx = id_to_idx.get(id(p), None)
        if idx is not None and out[idx] == "":
            out[idx] = n
    for i in range(len(out)):
        if out[i] == "":
            out[i] = f"param_{i}"
    return out


def _resolve_grad_norm_plan(
    cfg: Dict,
    mode: str,
    grad_norm_mode: str,
    model: CifarTradeoffModel,
    shared_params: List[nn.Parameter],
    shared_mode: str,
) -> Tuple[str, Optional[List[int]], bool, List[str]]:
    if mode != "joint" or grad_norm_mode == "none" or len(shared_params) == 0:
        return "all", None, False, []

    train_cfg = cfg.get("train", {})
    scope = str(train_cfg.get("grad_norm_scope", "all")).lower()
    layers = _parse_grad_norm_layers(train_cfg.get("grad_norm_layers", ["layer3", "layer4"]))

    if scope in {"all", "global"}:
        return "all", None, False, layers
    if scope in {"conflict_all", "all_conflict"}:
        return "conflict_all", None, True, layers
    if scope not in {"deep", "conflict_deep"}:
        raise ValueError(
            f"Unsupported train.grad_norm_scope={scope}. "
            "Use one of: all|deep|conflict_all|conflict_deep."
        )

    names = _build_shared_param_names(model=model, shared_params=shared_params, shared_mode=shared_mode)
    indices = [i for i, n in enumerate(names) if _layer_tag_from_param_name(n) in set(layers)]

    if len(indices) == 0:
        # Fallback to all params to avoid silent no-op when naming/layout changes.
        return ("conflict_all", None, True, layers) if scope == "conflict_deep" else ("all", None, False, layers)

    return scope, indices, (scope == "conflict_deep"), layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    cfg = apply_overrides(base_cfg, args.set)

    mixed_precision = cfg.get("accelerate", {}).get("mixed_precision", "no")
    if isinstance(mixed_precision, bool):
        mixed_precision = "no" if mixed_precision is False else "fp16"
    mixed_precision = str(mixed_precision)
    accelerator = Accelerator(mixed_precision=mixed_precision)
    device = accelerator.device

    seed = int(cfg.get("seed", 42))
    set_seed(seed, device_specific=True)

    exp_name = args.run_name or cfg.get("experiment", {}).get("name", "cifar_exp")
    run_dir = os.path.join(cfg.get("output", {}).get("root", "runs"), exp_name)

    if accelerator.is_main_process:
        if os.path.exists(run_dir):
            run_dir = f"{run_dir}_{now_str()}"
        ckpt_dir = ensure_dir(os.path.join(run_dir, "checkpoints"))
        save_yaml(cfg, os.path.join(run_dir, "run_config.yaml"))
    else:
        ckpt_dir = ""

    objs = [run_dir, ckpt_dir]
    broadcast_object_list(objs)
    run_dir, ckpt_dir = objs[0], objs[1]
    accelerator.wait_for_everyone()

    data_cfg = cfg.get("data", {})
    dataset_name = str(data_cfg.get("dataset", "cifar10")).lower()
    train_bs = int(data_cfg.get("batch_size", cfg.get("train", {}).get("batch_size", 128)))
    eval_bs = int(cfg.get("eval", {}).get("batch_size", train_bs))

    train_loader, class_names = build_cifar10_loader(
        dataset=dataset_name,
        data_root=data_cfg.get("root", "./data"),
        split="train",
        image_size=int(data_cfg.get("image_size", 32)),
        batch_size=train_bs,
        num_workers=int(data_cfg.get("num_workers", 8)),
        val_from_train=bool(data_cfg.get("val_from_train", False)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=seed,
        shuffle=True,
        drop_last=True,
        download=bool(data_cfg.get("download", True)),
        use_fake_data=bool(data_cfg.get("use_fake_data", False)),
        fake_train_size=int(data_cfg.get("fake_train_size", 8192)),
        fake_eval_size=int(data_cfg.get("fake_eval_size", 1024)),
        two_view=False,
        aug_strength="light",
        target_source="view1",
    )

    val_loader, _ = build_cifar10_loader(
        dataset=dataset_name,
        data_root=data_cfg.get("root", "./data"),
        split="val",
        image_size=int(data_cfg.get("image_size", 32)),
        batch_size=eval_bs,
        num_workers=int(data_cfg.get("num_workers", 8)),
        val_from_train=bool(data_cfg.get("val_from_train", False)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=seed,
        shuffle=False,
        drop_last=False,
        download=bool(data_cfg.get("download", True)),
        use_fake_data=bool(data_cfg.get("use_fake_data", False)),
        fake_train_size=int(data_cfg.get("fake_train_size", 8192)),
        fake_eval_size=int(data_cfg.get("fake_eval_size", 1024)),
    )

    test_loader, _ = build_cifar10_loader(
        dataset=dataset_name,
        data_root=data_cfg.get("root", "./data"),
        split="test",
        image_size=int(data_cfg.get("image_size", 32)),
        batch_size=eval_bs,
        num_workers=int(data_cfg.get("num_workers", 8)),
        val_from_train=False,
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=seed,
        shuffle=False,
        drop_last=False,
        download=bool(data_cfg.get("download", True)),
        use_fake_data=bool(data_cfg.get("use_fake_data", False)),
        fake_train_size=int(data_cfg.get("fake_train_size", 8192)),
        fake_eval_size=int(data_cfg.get("fake_eval_size", 1024)),
    )

    if accelerator.is_main_process:
        save_json(
            {
                "templates": cfg.get("text", {}).get("prompt_templates", ["a photo of a {class}"]),
                "class_names": list(class_names),
                "dataset": dataset_name,
                "note": "CIFAR lightweight text alignment with learnable prototypes.",
            },
            os.path.join(run_dir, "text_prompts.json"),
        )

    model = CifarTradeoffModel(cfg, num_classes=len(class_names)).to(device)
    backbone_name = str(cfg.get("model", {}).get("backbone", "resnet18"))

    shared_mode = str(cfg.get("train", {}).get("shared_params", "backbone"))
    shared_params, aux_params = _build_shared_and_aux_params(model, shared_mode=shared_mode)

    all_trainable = [p for p in model.parameters() if p.requires_grad]
    if len(all_trainable) == 0:
        raise RuntimeError("No trainable parameters in model.")

    optimizer = torch.optim.AdamW(
        all_trainable,
        lr=float(cfg.get("optim", {}).get("lr", 3e-4)),
        weight_decay=float(cfg.get("optim", {}).get("weight_decay", 1e-4)),
    )

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    mode, steps, lambda_txt, lambda_rec, strategy = _resolve_training_mode(cfg)
    grad_norm_mode = _resolve_grad_norm_mode(cfg, mode=mode)
    train_cfg = cfg.get("train", {})
    cagrad_beta = float(cfg.get("train", {}).get("cagrad_beta", 0.5))
    cagrad_conflict_only_raw = train_cfg.get("cagrad_conflict_only", False)
    if isinstance(cagrad_conflict_only_raw, bool):
        cagrad_conflict_only = cagrad_conflict_only_raw
    else:
        cagrad_conflict_only = str(cagrad_conflict_only_raw).lower() in {"1", "true", "on", "yes"}
    cagrad_conflict_threshold = float(train_cfg.get("cagrad_conflict_threshold", 0.0))
    cagrad_nonconflict_merge = str(train_cfg.get("cagrad_nonconflict_merge", "cagrad")).lower()
    if cagrad_nonconflict_merge not in {"cagrad", "sum", "avg", "average"}:
        raise ValueError(
            f"Unsupported train.cagrad_nonconflict_merge={cagrad_nonconflict_merge}. "
            "Use one of: cagrad|sum|avg."
        )
    grad_norm_scope, grad_norm_indices, grad_norm_conflict_only, grad_norm_layers = _resolve_grad_norm_plan(
        cfg=cfg,
        mode=mode,
        grad_norm_mode=grad_norm_mode,
        model=model,
        shared_params=shared_params,
        shared_mode=shared_mode,
    )

    log_cfg = cfg.get("log", {})
    log_every = int(log_cfg.get("every", 10))
    cos_every = int(log_cfg.get("cos_every", 10))
    save_every = int(log_cfg.get("save_every", 200))
    eval_every = int(log_cfg.get("eval_every", 200))

    metrics_file = os.path.join(run_dir, "metrics.jsonl")
    cos_curve = []
    temperature = float(cfg.get("text", {}).get("temperature", 0.07))

    if accelerator.is_main_process:
        save_json(
            {
                "strategy": strategy,
                "backbone": backbone_name,
                "mode": mode,
                "lambda_txt": lambda_txt,
                "lambda_rec": lambda_rec,
                "cagrad_beta": cagrad_beta,
                "cagrad_conflict_only": bool(cagrad_conflict_only),
                "cagrad_conflict_threshold": cagrad_conflict_threshold,
                "cagrad_nonconflict_merge": cagrad_nonconflict_merge,
                "grad_norm_mode": grad_norm_mode,
                "grad_norm_scope": grad_norm_scope,
                "grad_norm_conflict_only": bool(grad_norm_conflict_only),
                "grad_norm_layers": grad_norm_layers,
                "grad_norm_num_indices": None if grad_norm_indices is None else int(len(grad_norm_indices)),
            },
            os.path.join(run_dir, "train_setup.json"),
        )

    eval_split = str(cfg.get("eval", {}).get("split", "val")).lower()
    eval_loader = test_loader if eval_split == "test" else val_loader

    train_start = time.time()

    if steps <= 0:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_checkpoint(
                os.path.join(ckpt_dir, "latest.pt"),
                model=accelerator.unwrap_model(model),
                optimizer=optimizer,
                step=0,
                cfg=cfg,
            )
            run_eval(
                cfg=cfg,
                model=accelerator.unwrap_model(model),
                eval_loader=eval_loader,
                device=device,
                run_dir=run_dir,
                step=0,
                class_names=class_names,
                dataset_name=dataset_name,
            )
        accelerator.wait_for_everyone()

    train_iter = cycle_loader(train_loader)
    pbar = tqdm(range(1, steps + 1), desc="train_cifar10", disable=not accelerator.is_local_main_process)

    for step in pbar:
        batch = make_batch_dict(next(train_iter))
        batch = to_device(batch, device)

        model.train()
        images = batch["images"]
        images_target = batch.get("images_target", images)

        out = model(images)

        Lu, txt_extra = text_prototype_loss(
            z_txt=out["z_txt"],
            labels=batch["labels"],
            prototypes=model.text_prototypes,
            temperature=temperature,
        )
        Lg = F.mse_loss(out["recon"], images_target)
        total = lambda_txt * Lu + lambda_rec * Lg

        optimizer.zero_grad(set_to_none=True)
        cos = 0.0
        if strategy == "pcgrad":
            cos = apply_conflict_aware(
                loss_txt=Lu,
                loss_rec=Lg,
                lora_params=shared_params,
                aux_params=aux_params,
                lambda_txt=lambda_txt,
                lambda_rec=lambda_rec,
                grad_norm_mode=grad_norm_mode,
                grad_norm_indices=grad_norm_indices,
                grad_norm_conflict_only=grad_norm_conflict_only,
                extra_loss=None,
            )
            if accelerator.num_processes > 1:
                for p in list(shared_params) + list(aux_params):
                    if p.grad is not None:
                        p.grad = accelerator.reduce(p.grad, reduction="mean")
            optimizer.step()
        elif strategy == "cagrad":
            cos = apply_cagrad(
                loss_txt=Lu,
                loss_rec=Lg,
                shared_params=shared_params,
                aux_params=aux_params,
                lambda_txt=lambda_txt,
                lambda_rec=lambda_rec,
                beta=cagrad_beta,
                conflict_only=cagrad_conflict_only,
                conflict_threshold=cagrad_conflict_threshold,
                nonconflict_merge=cagrad_nonconflict_merge,
                grad_norm_mode=grad_norm_mode,
                grad_norm_indices=grad_norm_indices,
                grad_norm_conflict_only=grad_norm_conflict_only,
                extra_loss=None,
            )
            if accelerator.num_processes > 1:
                for p in list(shared_params) + list(aux_params):
                    if p.grad is not None:
                        p.grad = accelerator.reduce(p.grad, reduction="mean")
            optimizer.step()
        else:
            manual_naive = (mode == "joint") and (grad_norm_mode != "none") and (len(shared_params) > 0)
            if manual_naive:
                cos = apply_naive(
                    loss_txt=Lu,
                    loss_rec=Lg,
                    shared_params=shared_params,
                    aux_params=aux_params,
                    lambda_txt=lambda_txt,
                    lambda_rec=lambda_rec,
                    grad_norm_mode=grad_norm_mode,
                    grad_norm_indices=grad_norm_indices,
                    grad_norm_conflict_only=grad_norm_conflict_only,
                    extra_loss=None,
                )
                if accelerator.num_processes > 1:
                    for p in list(shared_params) + list(aux_params):
                        if p.grad is not None:
                            p.grad = accelerator.reduce(p.grad, reduction="mean")
                optimizer.step()
            else:
                if step % cos_every == 0 and len(shared_params) > 0:
                    cos = compute_grad_cosine(Lu, Lg, shared_params)
                accelerator.backward(total)
                optimizer.step()

        cos_global = _dist_mean_scalar(accelerator, cos, device)
        Lu_global = _dist_mean_scalar(accelerator, Lu, device)
        Lg_global = _dist_mean_scalar(accelerator, Lg, device)
        total_global = _dist_mean_scalar(accelerator, total, device)
        txt_acc_global = _dist_mean_scalar(accelerator, txt_extra.get("txt_acc", 0.0), device)

        if step % cos_every == 0 and accelerator.is_main_process:
            cos_curve.append({"step": step, "cos": cos_global})

        row = {
            "step": step,
            "Lu": Lu_global,
            "Lg": Lg_global,
            "loss_txt": Lu_global,
            "loss_g": Lg_global,
            "loss_total": total_global,
            "total": total_global,
            "cos": cos_global,
            "txt_acc": txt_acc_global,
            "recon_mse": Lg_global,
            "strategy": strategy,
            "mode": mode,
        }

        if accelerator.is_main_process and (step % log_every == 0 or step == 1 or step == steps):
            append_jsonl(metrics_file, row)
            pbar.set_postfix(
                Lu=f"{row['Lu']:.4f}",
                Lg=f"{row['Lg']:.4f}",
                total=f"{row['total']:.4f}",
                cos=f"{row['cos']:.4f}",
            )

        if step % save_every == 0 or step == steps:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_checkpoint(
                    os.path.join(ckpt_dir, "latest.pt"),
                    model=accelerator.unwrap_model(model),
                    optimizer=optimizer,
                    step=step,
                    cfg=cfg,
                )
            accelerator.wait_for_everyone()

        if step % eval_every == 0 or step == steps:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                run_eval(
                    cfg=cfg,
                    model=accelerator.unwrap_model(model),
                    eval_loader=eval_loader,
                    device=device,
                    run_dir=run_dir,
                    step=step,
                    class_names=class_names,
                    dataset_name=dataset_name,
                )
            accelerator.wait_for_everyone()

    elapsed_sec = float(time.time() - train_start)

    if accelerator.is_main_process:
        cos_values = [x["cos"] for x in cos_curve]
        if len(cos_values) == 0:
            cos_mean = 0.0
            cos_neg_ratio = 0.0
        else:
            cos_mean = float(sum(cos_values) / len(cos_values))
            cos_neg_ratio = float(sum(1 for c in cos_values if c < 0) / len(cos_values))

        summary = {
            "run_dir": run_dir,
            "dataset": cfg.get("data", {}).get("dataset", "cifar10"),
            "backbone": backbone_name,
            "strategy": strategy,
            "grad_strategy": strategy,
            "mode": mode,
            "seed": seed,
            "lambda_txt": lambda_txt,
            "lambda_rec": lambda_rec,
            "num_trainable": count_parameters([p for p in model.parameters() if p.requires_grad]),
            "num_shared": count_parameters(shared_params),
            "num_aux": count_parameters(aux_params),
            "shared_params_mode": shared_mode,
            "cagrad_beta": cagrad_beta,
            "cagrad_conflict_only": bool(cagrad_conflict_only),
            "cagrad_conflict_threshold": cagrad_conflict_threshold,
            "cagrad_nonconflict_merge": cagrad_nonconflict_merge,
            "grad_norm_mode": grad_norm_mode,
            "grad_norm_scope": grad_norm_scope,
            "grad_norm_conflict_only": bool(grad_norm_conflict_only),
            "grad_norm_layers": grad_norm_layers,
            "grad_norm_num_indices": None if grad_norm_indices is None else int(len(grad_norm_indices)),
            "cos_mean": cos_mean,
            "cos_neg_ratio": cos_neg_ratio,
            "world_size": accelerator.num_processes,
            "train_steps": steps,
            "walltime_sec": elapsed_sec,
        }
        save_json(summary, os.path.join(run_dir, "cos_summary.json"))
        save_json({"curve": cos_curve}, os.path.join(run_dir, "cos_curve.json"))

    accelerator.wait_for_everyone()
    accelerator.print(f"[train_cifar10] done. run_dir={run_dir}")


if __name__ == "__main__":
    main()
