#!/usr/bin/env python
"""Inter-task Feature Representation Alignment via Linear CKA.

This script compares three jointly-trained methods (e.g., Vanilla Joint / CAGrad / DSGA)
against two task specialists (understanding-only and generation-only) across training stages:
Initial (0%), Middle (50%), and Converged (100%).

For each method and stage:
1) Extract shared-backbone feature maps with forward hooks at multiple depths.
2) Compute Linear CKA between joint features and specialist features per layer:
     cka_u = CKA(F_joint, F_understanding_specialist)
     cka_g = CKA(F_joint, F_generation_specialist)
3) Combine task-wise similarity into a coexistence score (harmonic mean by default).
4) Render spatiotemporal heatmaps (methods as subplots).

Outputs:
- cka_alignment_heatmap.pdf
- cka_alignment_summary.csv
- cka_alignment_raw.json

Spec file format (JSON):
{
  "model_family": "cifar_tradeoff",  // or "in100_vitbridge"
  "dataset": {
    "name": "cifar100",              // cifar100 | imagenet100
    "root": "/path/to/data",
    "split": "test",
    "image_size": 32,
    "batch_size": 128,
    "num_workers": 4,
    "max_batches": 0,
    "max_samples": 1024
  },
  "stages": {
    "initial": "Initial",
    "middle": "Middle",
    "converged": "Converged"
  },
  "specialists": {
    "understanding": {
      "config": "/abs/path/run_config.yaml",   // required for cifar_tradeoff
      "stages": {
        "initial": "INIT",
        "middle": "/abs/path/to/run_or_ckpt",
        "converged": "/abs/path/to/run_or_ckpt"
      }
    },
    "generation": {
      "config": "/abs/path/run_config.yaml",
      "stages": {
        "initial": "INIT",
        "middle": "/abs/path/to/run_or_ckpt",
        "converged": "/abs/path/to/run_or_ckpt"
      }
    }
  },
  "methods": {
    "Vanilla Joint": {
      "config": "/abs/path/run_config.yaml",
      "stages": {
        "initial": "INIT",
        "middle": "/abs/path/to/run_or_ckpt",
        "converged": "/abs/path/to/run_or_ckpt"
      }
    },
    "CAGrad": {
      "config": "/abs/path/run_config.yaml",
      "stages": {
        "initial": "INIT",
        "middle": "/abs/path/to/run_or_ckpt",
        "converged": "/abs/path/to/run_or_ckpt"
      }
    },
    "DSGA": {
      "config": "/abs/path/run_config.yaml",
      "stages": {
        "initial": "INIT",
        "middle": "/abs/path/to/run_or_ckpt",
        "converged": "/abs/path/to/run_or_ckpt"
      }
    }
  }
}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Keep numexpr thread config sane to avoid noisy runtime errors from pandas/seaborn import.
_nexpr_max = os.environ.get("NUMEXPR_MAX_THREADS")
_nexpr_num = os.environ.get("NUMEXPR_NUM_THREADS")
if _nexpr_max is not None:
    try:
        max_threads = int(_nexpr_max)
        if _nexpr_num is None or int(_nexpr_num) > max_threads:
            os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)
    except ValueError:
        pass
elif _nexpr_num is None:
    os.environ["NUMEXPR_NUM_THREADS"] = "16"

try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:  # noqa: BLE001
    sns = None
    _HAS_SEABORN = False

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unirae.data_cifar10 import build_cifar10_loader, make_batch_dict
from unirae.train_cifar10 import CifarTradeoffModel
from unirae.train_imagenet100_dynamics import ViTBridgeModel, build_encoder, build_imagenet100_dataloaders
from unirae.utils import load_yaml


STAGE_ORDER = ["initial", "middle", "converged"]
DEFAULT_STAGE_LABELS = {"initial": "Initial", "middle": "Middle", "converged": "Converged"}


@dataclass
class DatasetCfg:
    name: str
    root: str
    split: str
    image_size: int
    batch_size: int
    num_workers: int
    max_batches: int
    max_samples: int


@dataclass
class ModelInstance:
    model: nn.Module
    backbone: nn.Module
    layer_names: List[str]


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_ckpt_path(path_or_run: str) -> str:
    p = Path(path_or_run).expanduser()
    if p.is_file():
        return str(p)
    if p.is_dir():
        c1 = p / "checkpoints" / "latest.pt"
        if c1.exists():
            return str(c1)
        c2 = p / "latest.pt"
        if c2.exists():
            return str(c2)
    raise FileNotFoundError(f"Cannot resolve checkpoint from: {path_or_run}")


def _load_state_dict(ckpt_path: str) -> Tuple[Dict[str, torch.Tensor], int]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"], int(ckpt.get("step", -1))
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"], int(ckpt.get("step", -1))
        # raw state dict
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt, int(ckpt.get("step", -1)) if "step" in ckpt else -1
    raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")


def _strip_prefix_if_needed(sd: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    keys = list(model.state_dict().keys())
    keyset = set(keys)
    prefixes = ["", "module.", "model.", "base_model."]

    best_sd = sd
    best_score = -1
    for pref in prefixes:
        if pref:
            cand = {(k[len(pref):] if k.startswith(pref) else k): v for k, v in sd.items()}
        else:
            cand = sd
        score = sum(1 for k in cand.keys() if k in keyset)
        if score > best_score:
            best_sd = cand
            best_score = score
    return best_sd


def _build_cifar_model(config_path: str, device: torch.device) -> ModelInstance:
    cfg = load_yaml(config_path)
    data_cfg = cfg.get("data", {})
    dataset_name = str(data_cfg.get("dataset", "cifar100")).lower()
    num_classes = 100 if dataset_name == "cifar100" else 10
    model = CifarTradeoffModel(cfg, num_classes=num_classes).to(device)
    model.eval()
    backbone = model.backbone
    layer_names, _ = discover_hook_layers(backbone)
    return ModelInstance(model=model, backbone=backbone, layer_names=layer_names)


def _build_in100_model(run_setup_path: str, device: torch.device) -> ModelInstance:
    with open(run_setup_path, "r", encoding="utf-8") as f:
        setup = json.load(f)
    args = setup.get("args", {})

    encoder = build_encoder(
        str(args.get("encoder_init", "dinov2_vits14")),
        image_size=224,
        encoder_ckpt=str(args.get("encoder_ckpt", "")),
    )
    model = ViTBridgeModel(
        encoder=encoder,
        num_classes=int(setup.get("data_meta", {}).get("num_classes", 100)),
        image_size=224,
        decoder_dim=int(args.get("decoder_dim", 384)),
        decoder_depth=int(args.get("decoder_depth", 4)),
        decoder_heads=int(args.get("decoder_heads", 6)),
        decoder_mlp_ratio=float(args.get("decoder_mlp_ratio", 4.0)),
        decoder_drop_rate=float(args.get("decoder_drop_rate", 0.0)),
    ).to(device)
    model.eval()
    backbone = model.encoder
    layer_names, _ = discover_hook_layers(backbone)
    return ModelInstance(model=model, backbone=backbone, layer_names=layer_names)


def build_model_instance(
    model_family: str,
    config_path: str,
    checkpoint_path: Optional[str],
    device: torch.device,
) -> ModelInstance:
    if model_family == "cifar_tradeoff":
        inst = _build_cifar_model(config_path, device)
    elif model_family == "in100_vitbridge":
        inst = _build_in100_model(config_path, device)
    else:
        raise ValueError(f"Unsupported model_family={model_family}. Use cifar_tradeoff|in100_vitbridge")

    if checkpoint_path is not None:
        sd, _ = _load_state_dict(checkpoint_path)
        sd = _strip_prefix_if_needed(sd, inst.model)
        missing, unexpected = inst.model.load_state_dict(sd, strict=False)
        if len(unexpected) > 0:
            print(f"[warn] unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        if len(missing) > 0:
            print(f"[warn] missing keys ({len(missing)}): {missing[:5]}")
    return inst


def discover_hook_layers(backbone: nn.Module) -> Tuple[List[str], List[nn.Module]]:
    names: List[str] = []
    mods: List[nn.Module] = []

    def add(name: str, module: nn.Module) -> None:
        names.append(name)
        mods.append(module)

    # timm Swin: backbone.model.layers[s].blocks[b]
    if hasattr(backbone, "model") and hasattr(backbone.model, "layers"):
        layers = getattr(backbone.model, "layers")
        if hasattr(layers, "__len__") and len(layers) > 0:
            block_count = 0
            for si, stage in enumerate(layers):
                if hasattr(stage, "blocks") and hasattr(stage.blocks, "__len__"):
                    for bi, blk in enumerate(stage.blocks):
                        add(f"layers.{si}.blocks.{bi}", blk)
                        block_count += 1
            if block_count > 0:
                return names, mods

    # timm ViT-like: backbone.model.blocks
    if hasattr(backbone, "model") and hasattr(backbone.model, "blocks"):
        blocks = getattr(backbone.model, "blocks")
        if hasattr(blocks, "__len__") and len(blocks) > 0:
            for bi, blk in enumerate(blocks):
                add(f"blocks.{bi}", blk)
            return names, mods

    # direct encoder blocks
    if hasattr(backbone, "blocks") and hasattr(backbone.blocks, "__len__") and len(backbone.blocks) > 0:
        for bi, blk in enumerate(backbone.blocks):
            add(f"blocks.{bi}", blk)
        return names, mods

    # custom ResNet18Backbone (stem sequential: [..., layer1, layer2, layer3, layer4, avgpool])
    if hasattr(backbone, "stem") and isinstance(backbone.stem, nn.Sequential):
        stem = backbone.stem
        if len(stem) >= 8:
            add("layer1", stem[4])
            add("layer2", stem[5])
            add("layer3", stem[6])
            add("layer4", stem[7])
            return names, mods

    # torchvision-style fallback
    for lname in ("layer1", "layer2", "layer3", "layer4"):
        if hasattr(backbone, lname):
            add(lname, getattr(backbone, lname))
    if len(mods) > 0:
        return names, mods

    raise RuntimeError("Cannot discover hook layers. Unsupported backbone structure.")


def _extract_tensor(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)) and len(output) > 0:
        for x in reversed(output):
            if torch.is_tensor(x):
                return x
    if isinstance(output, dict):
        for key in ("x", "out", "last_hidden_state", "x_prenorm"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
        for v in output.values():
            if torch.is_tensor(v):
                return v
    raise TypeError(f"Unsupported hook output type: {type(output)}")


def _flatten_feature(x: torch.Tensor) -> torch.Tensor:
    # Keep batch dimension, flatten all spatial/token/channel dimensions.
    if x.ndim == 1:
        return x.unsqueeze(1)
    return x.reshape(x.shape[0], -1)


def _collect_input_batches(
    dataset_cfg: DatasetCfg,
    model_family: str,
    max_samples: int,
) -> List[torch.Tensor]:
    batches: List[torch.Tensor] = []
    seen = 0

    if dataset_cfg.name == "cifar100":
        loader, _ = build_cifar10_loader(
            dataset="cifar100",
            data_root=dataset_cfg.root,
            split=dataset_cfg.split,
            image_size=dataset_cfg.image_size,
            batch_size=dataset_cfg.batch_size,
            num_workers=dataset_cfg.num_workers,
            val_from_train=False,
            val_ratio=0.1,
            seed=42,
            shuffle=False,
            drop_last=False,
            download=True,
            use_fake_data=False,
        )
        for bi, raw in enumerate(loader):
            if dataset_cfg.max_batches > 0 and bi >= dataset_cfg.max_batches:
                break
            images = make_batch_dict(raw)["images"].cpu()
            batches.append(images)
            seen += int(images.shape[0])
            if seen >= max_samples:
                break
    elif dataset_cfg.name == "imagenet100":
        train_loader, val_loader, _meta = build_imagenet100_dataloaders(
            batch_size=dataset_cfg.batch_size,
            num_workers=dataset_cfg.num_workers,
            dataset_path=(dataset_cfg.root if dataset_cfg.root else None),
            distributed=False,
            rank=0,
            world_size=1,
        )
        _ = train_loader
        for bi, (images, _labels) in enumerate(val_loader):
            if dataset_cfg.max_batches > 0 and bi >= dataset_cfg.max_batches:
                break
            batches.append(images.cpu())
            seen += int(images.shape[0])
            if seen >= max_samples:
                break
    else:
        raise ValueError(f"Unsupported dataset={dataset_cfg.name}. Use cifar100|imagenet100")

    if len(batches) == 0:
        raise RuntimeError("No input batches collected.")
    return batches


def collect_layer_features(
    model: nn.Module,
    backbone: nn.Module,
    hook_layers: Sequence[nn.Module],
    layer_names: Sequence[str],
    input_batches_cpu: Sequence[torch.Tensor],
    device: torch.device,
    max_samples: int,
) -> Dict[str, torch.Tensor]:
    store: Dict[str, List[torch.Tensor]] = {name: [] for name in layer_names}
    cache: Dict[str, torch.Tensor] = {}

    hooks = []
    for name, module in zip(layer_names, hook_layers):
        def _mk_hook(layer_name: str):
            def _hook(_m, _inp, out):
                t = _extract_tensor(out)
                cache[layer_name] = _flatten_feature(t).detach()
            return _hook
        hooks.append(module.register_forward_hook(_mk_hook(name)))

    try:
        model.eval()
        seen = 0
        with torch.no_grad():
            for batch_cpu in input_batches_cpu:
                x = batch_cpu.to(device, non_blocking=True)
                _ = model(x)
                bsz = int(x.shape[0])

                for ln in layer_names:
                    if ln not in cache:
                        raise RuntimeError(f"Hook did not capture layer={ln}")
                    f = cache[ln]
                    if f.shape[0] != bsz:
                        raise RuntimeError(f"Layer={ln} batch mismatch: {f.shape[0]} vs {bsz}")
                    store[ln].append(f.cpu())

                seen += bsz
                if seen >= max_samples:
                    break

        out: Dict[str, torch.Tensor] = {}
        for ln in layer_names:
            ft = torch.cat(store[ln], dim=0)
            if ft.shape[0] > max_samples:
                ft = ft[:max_samples]
            out[ln] = ft
        return out
    finally:
        for h in hooks:
            h.remove()


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    # x: [N, Dx], y: [N, Dy]
    if x.shape[0] != y.shape[0]:
        n = min(x.shape[0], y.shape[0])
        x = x[:n]
        y = y[:n]

    x = x.double()
    y = y.double()

    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    k = x @ x.t()  # [N, N]
    l = y @ y.t()  # [N, N]

    hsic = (k * l).sum()
    denom = torch.linalg.norm(k) * torch.linalg.norm(l) + eps
    val = (hsic / denom).clamp(0.0, 1.0)
    return float(val.item())


def combine_scores(u: float, g: float, mode: str = "harmonic", eps: float = 1e-12) -> float:
    if mode == "mean":
        return float(0.5 * (u + g))
    if mode == "harmonic":
        return float((2.0 * u * g) / (u + g + eps))
    raise ValueError(f"Unsupported combine mode={mode}. Use mean|harmonic")


def _stage_ckpt_or_none(stage_path: str) -> Optional[str]:
    if str(stage_path).upper() == "INIT":
        return None
    return _resolve_ckpt_path(stage_path)


def _load_model_for_stage(
    model_family: str,
    entry: Dict,
    stage_path: str,
    device: torch.device,
) -> ModelInstance:
    config_path = _resolve_model_config(entry=entry, stage_path=stage_path, model_family=model_family)
    ckpt = _stage_ckpt_or_none(stage_path)
    return build_model_instance(
        model_family=model_family,
        config_path=config_path,
        checkpoint_path=ckpt,
        device=device,
    )


def _ordered_methods(methods_cfg: Dict[str, Dict]) -> List[str]:
    keys = list(methods_cfg.keys())
    pref = ["Vanilla Joint", "CAGrad", "DSGA"]
    out = []
    for p in pref:
        if p in methods_cfg:
            out.append(p)
    for k in keys:
        if k not in out:
            out.append(k)
    return out


def _to_dataset_cfg(spec: Dict) -> DatasetCfg:
    ds = spec.get("dataset", {})
    return DatasetCfg(
        name=str(ds.get("name", "cifar100")).lower(),
        root=str(ds.get("root", "")),
        split=str(ds.get("split", "test")).lower(),
        image_size=int(ds.get("image_size", 32)),
        batch_size=int(ds.get("batch_size", 128)),
        num_workers=int(ds.get("num_workers", 4)),
        max_batches=int(ds.get("max_batches", 0)),
        max_samples=int(ds.get("max_samples", 1024)),
    )


def _normalize_stage_key(key: str) -> Optional[str]:
    k = str(key).strip().lower()
    if k in {"initial", "init", "0", "0%"}:
        return "initial"
    if k in {"middle", "mid", "50", "50%"}:
        return "middle"
    if k in {"converged", "final", "end", "100", "100%"}:
        return "converged"
    return None


def _normalize_stage_paths(raw: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in raw.items():
        nk = _normalize_stage_key(k)
        if nk is None:
            continue
        out[nk] = str(v)
    return out


def _normalize_stage_labels(raw: Dict[str, str]) -> Dict[str, str]:
    out = dict(DEFAULT_STAGE_LABELS)
    for k, v in raw.items():
        nk = _normalize_stage_key(k)
        if nk is None:
            continue
        out[nk] = str(v)
    return out


def _resolve_model_config(entry: Dict, stage_path: str, model_family: str) -> str:
    if "config" in entry and str(entry["config"]).strip():
        return str(entry["config"])

    p = Path(stage_path).expanduser()
    if p.is_file():
        # Direct ckpt path: try sibling run files.
        run_dir = p.parent.parent if p.parent.name == "checkpoints" else p.parent
    else:
        run_dir = p

    if model_family == "cifar_tradeoff":
        cand = run_dir / "run_config.yaml"
        if cand.exists():
            return str(cand)
    elif model_family == "in100_vitbridge":
        cand = run_dir / "run_setup.json"
        if cand.exists():
            return str(cand)

    raise ValueError(
        f"Cannot resolve config for model_family={model_family}. "
        f"Please provide `config` in spec entry. stage_path={stage_path}"
    )


def run_alignment(spec: Dict, out_dir: Path, combine_mode: str, device: torch.device) -> None:
    model_family = str(spec.get("model_family", "cifar_tradeoff"))
    dataset_cfg = _to_dataset_cfg(spec)
    stage_names_cfg = _normalize_stage_labels(spec.get("stages", {}))
    stage_keys = list(STAGE_ORDER)

    methods_cfg = spec.get("methods", {})
    if len(methods_cfg) == 0:
        raise ValueError("spec.methods is empty")
    methods_cfg = {k: dict(v) for k, v in methods_cfg.items()}
    for mk in methods_cfg:
        methods_cfg[mk]["stages"] = _normalize_stage_paths(methods_cfg[mk].get("stages", {}))
        for sk in stage_keys:
            if sk not in methods_cfg[mk]["stages"]:
                raise ValueError(f"Method={mk} missing stage={sk} checkpoint/run path.")

    specialists = spec.get("specialists", {})
    if "understanding" not in specialists or "generation" not in specialists:
        raise ValueError("spec.specialists must contain understanding and generation")
    specialists = {k: dict(v) for k, v in specialists.items()}
    for sp in ("understanding", "generation"):
        specialists[sp]["stages"] = _normalize_stage_paths(specialists[sp].get("stages", {}))
        for sk in stage_keys:
            if sk not in specialists[sp]["stages"]:
                raise ValueError(f"Specialist={sp} missing stage={sk} checkpoint/run path.")

    method_order = _ordered_methods(methods_cfg)
    input_batches = _collect_input_batches(dataset_cfg, model_family, dataset_cfg.max_samples)

    # Determine canonical layer order from first method at initial stage.
    first_name = method_order[0]
    first_cfg = methods_cfg[first_name]
    first_inst = _load_model_for_stage(
        model_family=model_family,
        entry=first_cfg,
        stage_path=str(first_cfg["stages"][stage_keys[0]]),
        device=device,
    )
    canonical_layers, canonical_modules = discover_hook_layers(first_inst.backbone)
    num_layers = len(canonical_layers)
    print(f"[info] discovered layers={num_layers}")

    # raw storage: method -> metric -> [layers, stages]
    matrices: Dict[str, Dict[str, np.ndarray]] = {}
    for m in method_order:
        matrices[m] = {
            "cka_u": np.zeros((num_layers, len(stage_keys)), dtype=np.float64),
            "cka_g": np.zeros((num_layers, len(stage_keys)), dtype=np.float64),
            "cka_align": np.zeros((num_layers, len(stage_keys)), dtype=np.float64),
        }

    for sj, sk in enumerate(stage_keys):
        print(f"[stage] {sk} ({stage_names_cfg.get(sk, sk)})")

        # Specialists for this stage.
        us_cfg = specialists["understanding"]
        gs_cfg = specialists["generation"]

        u_inst = _load_model_for_stage(
            model_family=model_family,
            entry=us_cfg,
            stage_path=str(us_cfg["stages"][sk]),
            device=device,
        )
        g_inst = _load_model_for_stage(
            model_family=model_family,
            entry=gs_cfg,
            stage_path=str(gs_cfg["stages"][sk]),
            device=device,
        )

        u_layer_names, u_modules = discover_hook_layers(u_inst.backbone)
        g_layer_names, g_modules = discover_hook_layers(g_inst.backbone)

        # Require same logical depth count; align by index.
        common_layers = min(num_layers, len(u_layer_names), len(g_layer_names))
        if common_layers < num_layers:
            print(
                f"[warn] layer count mismatch at stage={sk}: "
                f"canonical={num_layers}, U={len(u_layer_names)}, G={len(g_layer_names)}. "
                f"Using first {common_layers} layers."
            )

        u_feats = collect_layer_features(
            model=u_inst.model,
            backbone=u_inst.backbone,
            hook_layers=u_modules[:common_layers],
            layer_names=u_layer_names[:common_layers],
            input_batches_cpu=input_batches,
            device=device,
            max_samples=dataset_cfg.max_samples,
        )
        g_feats = collect_layer_features(
            model=g_inst.model,
            backbone=g_inst.backbone,
            hook_layers=g_modules[:common_layers],
            layer_names=g_layer_names[:common_layers],
            input_batches_cpu=input_batches,
            device=device,
            max_samples=dataset_cfg.max_samples,
        )

        for method_name in method_order:
            mcfg = methods_cfg[method_name]
            m_inst = _load_model_for_stage(
                model_family=model_family,
                entry=mcfg,
                stage_path=str(mcfg["stages"][sk]),
                device=device,
            )
            m_layer_names, m_modules = discover_hook_layers(m_inst.backbone)
            common = min(common_layers, len(m_layer_names))
            if common < common_layers:
                print(
                    f"[warn] method={method_name} has fewer layers ({len(m_layer_names)}), using first {common}."
                )

            m_feats = collect_layer_features(
                model=m_inst.model,
                backbone=m_inst.backbone,
                hook_layers=m_modules[:common],
                layer_names=m_layer_names[:common],
                input_batches_cpu=input_batches,
                device=device,
                max_samples=dataset_cfg.max_samples,
            )

            for li in range(common):
                ml = m_layer_names[li]
                ul = u_layer_names[li]
                gl = g_layer_names[li]

                fx = m_feats[ml].to(device, non_blocking=True)
                fu = u_feats[ul].to(device, non_blocking=True)
                fg = g_feats[gl].to(device, non_blocking=True)

                c_u = linear_cka(fx, fu)
                c_g = linear_cka(fx, fg)
                c_a = combine_scores(c_u, c_g, mode=combine_mode)

                matrices[method_name]["cka_u"][li, sj] = c_u
                matrices[method_name]["cka_g"][li, sj] = c_g
                matrices[method_name]["cka_align"][li, sj] = c_a

            # pad any unavailable deeper layers with nan for plotting clarity
            if common < num_layers:
                matrices[method_name]["cka_u"][common:, sj] = np.nan
                matrices[method_name]["cka_g"][common:, sj] = np.nan
                matrices[method_name]["cka_align"][common:, sj] = np.nan

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw json
    raw = {
        "model_family": model_family,
        "dataset": dataset_cfg.__dict__,
        "stages": [stage_names_cfg.get(s, s) for s in stage_keys],
        "layer_names": canonical_layers,
        "combine_mode": combine_mode,
        "methods": {},
    }
    for m in method_order:
        raw["methods"][m] = {
            "cka_u": matrices[m]["cka_u"].tolist(),
            "cka_g": matrices[m]["cka_g"].tolist(),
            "cka_align": matrices[m]["cka_align"].tolist(),
            "avg_cka_u": float(np.nanmean(matrices[m]["cka_u"])),
            "avg_cka_g": float(np.nanmean(matrices[m]["cka_g"])),
            "avg_cka_align": float(np.nanmean(matrices[m]["cka_align"])),
        }

    raw_path = out_dir / "cka_alignment_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    # Summary csv
    csv_path = out_dir / "cka_alignment_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("method,avg_cka_u,avg_cka_g,avg_cka_align\n")
        for m in method_order:
            f.write(
                f"{m},{np.nanmean(matrices[m]['cka_u']):.6f},"
                f"{np.nanmean(matrices[m]['cka_g']):.6f},"
                f"{np.nanmean(matrices[m]['cka_align']):.6f}\n"
            )

    # Plot
    n_methods = len(method_order)
    fig, axes = plt.subplots(1, n_methods, figsize=(4.6 * n_methods, 7.2), sharey=True)
    if n_methods == 1:
        axes = [axes]

    stage_labels = [stage_names_cfg.get(s, s) for s in stage_keys]

    for i, m in enumerate(method_order):
        ax = axes[i]
        heat = matrices[m]["cka_align"]
        if _HAS_SEABORN:
            sns.heatmap(
                heat,
                ax=ax,
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
                cbar=(i == n_methods - 1),
                cbar_kws={"shrink": 0.85, "label": "CKA"},
                xticklabels=stage_labels,
                yticklabels=[str(j) for j in range(num_layers)],
                linewidths=0.2,
                linecolor="white",
            )
        else:
            im = ax.imshow(
                heat,
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                origin="upper",
            )
            ax.set_xticks(np.arange(len(stage_labels)))
            ax.set_xticklabels(stage_labels)
            if num_layers <= 24:
                y_ticks = np.arange(num_layers)
            else:
                tick_step = max(1, num_layers // 12)
                y_ticks = np.arange(0, num_layers, tick_step)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([str(j) for j in y_ticks])
            if i == n_methods - 1:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("CKA")
        ax.set_title(f"{m}\nAvg={np.nanmean(heat):.3f}", fontsize=12)
        ax.set_xlabel("Training Progress", fontsize=11)
        if i == 0:
            ax.set_ylabel("Encoder Depth (Layer Index)", fontsize=11)
        else:
            ax.set_ylabel("")
            if not _HAS_SEABORN:
                ax.tick_params(axis="y", labelleft=False)

    fig.suptitle("Inter-task Feature Representation Alignment (Linear CKA)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = out_dir / "cka_alignment_heatmap.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n=== CKA Alignment Summary ===")
    rows = []
    for m in method_order:
        r = {
            "method": m,
            "avg_cka_u": float(np.nanmean(matrices[m]["cka_u"])),
            "avg_cka_g": float(np.nanmean(matrices[m]["cka_g"])),
            "avg_cka_align": float(np.nanmean(matrices[m]["cka_align"])),
        }
        rows.append(r)
    rows = sorted(rows, key=lambda x: x["avg_cka_align"], reverse=True)
    for r in rows:
        print(
            f"{r['method']:<20} "
            f"avg_cka_u={r['avg_cka_u']:.4f} "
            f"avg_cka_g={r['avg_cka_g']:.4f} "
            f"avg_cka_align={r['avg_cka_align']:.4f}"
        )

    print(f"\n[save] heatmap: {fig_path}")
    print(f"[save] summary: {csv_path}")
    print(f"[save] raw:     {raw_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CKA diagnostic for inter-task feature alignment.")
    p.add_argument("--spec", required=True, help="Path to JSON spec file.")
    p.add_argument("--out_dir", default="results/cka_alignment", help="Output directory.")
    p.add_argument("--combine", default="harmonic", choices=["harmonic", "mean"], help="How to combine CKA_u and CKA_g.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device selection.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = _load_json(args.spec)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    out_dir = Path(args.out_dir)
    run_alignment(spec=spec, out_dir=out_dir, combine_mode=args.combine, device=device)


if __name__ == "__main__":
    main()
