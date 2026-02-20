import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .lora import apply_lora


class RadioWrapper(nn.Module):
    """RADIO wrapper that returns z_clip and z_dino features."""

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg

        radio_cfg = cfg.get("radio", {})
        lora_cfg = cfg.get("lora", {})

        self.clip_key = radio_cfg.get("clip_adaptor", "clip")
        self.dino_key = radio_cfg.get("dino_adaptor", "dino_v2")
        self.freeze_trunk = bool(radio_cfg.get("freeze_trunk", True))

        self._build_radio_model(radio_cfg)

        clip_dim = int(radio_cfg.get("output_dim", {}).get("clip", self._infer_clip_dim()))
        dino_dim = int(radio_cfg.get("output_dim", {}).get("dino", self._infer_dino_dim()))
        base_clip_dim = self._infer_clip_dim()
        base_dino_dim = self._infer_dino_dim()

        self.clip_proj = nn.Identity() if clip_dim == base_clip_dim else nn.Linear(base_clip_dim, clip_dim)
        self.dino_proj = nn.Identity() if dino_dim == base_dino_dim else nn.Linear(base_dino_dim, dino_dim)

        if self.freeze_trunk:
            for p in self.trunk.parameters():
                p.requires_grad = False

        self.lora_params: List[nn.Parameter] = []
        if lora_cfg.get("enable", False):
            self.lora_params = apply_lora(
                model=self.trunk,
                rank=int(lora_cfg.get("rank", 8)),
                alpha=float(lora_cfg.get("alpha", 16.0)),
                target_modules=lora_cfg.get(
                    "target_modules",
                    [
                        "blocks\\..*\\.attn\\.qkv",
                        "blocks\\..*\\.attn\\.proj",
                        "blocks\\..*\\.mlp\\.fc1",
                        "blocks\\..*\\.mlp\\.fc2",
                    ],
                ),
            )

            # Ensure base trunk stays frozen while LoRA params are trainable.
            for p in self.trunk.parameters():
                p.requires_grad = False
            for p in self.lora_params:
                p.requires_grad = True

    def _import_hubconf(self, code_root: str):
        code_root = str(code_root)
        hubconf_path = Path(code_root) / "hubconf.py"
        if not hubconf_path.exists():
            raise FileNotFoundError(f"RADIO hubconf not found at: {hubconf_path}")

        radio_pkg = Path(code_root) / "radio"
        if radio_pkg.exists() and code_root not in sys.path:
            sys.path.insert(0, code_root)

        module_name = f"unirae_radio_hubconf_{abs(hash(code_root))}"
        spec = importlib.util.spec_from_file_location(module_name, str(hubconf_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to import hubconf from {hubconf_path}")

        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _build_radio_model(self, radio_cfg: Dict) -> None:
        code_root = radio_cfg.get("code_root")
        ckpt = radio_cfg.get("ckpt", "")
        adaptor_names = [self.clip_key, self.dino_key]

        self._use_fallback = False
        self._fallback = None

        try:
            if code_root:
                hub_mod = self._import_hubconf(code_root)
                self.radio = hub_mod.radio_model(version=ckpt, adaptor_names=adaptor_names)
            else:
                # Optional fallback to installed hubconf module.
                hub_mod = importlib.import_module("hubconf")
                self.radio = hub_mod.radio_model(version=ckpt, adaptor_names=adaptor_names)

            self.trunk = self.radio.model if hasattr(self.radio, "model") else self.radio
        except Exception as e:
            if not radio_cfg.get("allow_fallback", True):
                raise

            import timm

            self._use_fallback = True
            model_name = radio_cfg.get("fallback_model", "vit_base_patch16_224")
            self._fallback = timm.create_model(model_name, pretrained=False, num_classes=0)
            self.trunk = self._fallback
            self.radio = None
            self._fallback_warn = str(e)

    def _infer_clip_dim(self) -> int:
        if self._use_fallback:
            return int(getattr(self.trunk, "num_features", 768))

        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            out = self.radio(dummy, feature_fmt="NLC")
            z_clip = out[self.clip_key].summary
            return int(z_clip.shape[-1])

    def _infer_dino_dim(self) -> int:
        if self._use_fallback:
            return int(getattr(self.trunk, "num_features", 768))

        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            out = self.radio(dummy, feature_fmt="NLC")
            z_dino = out[self.dino_key].features
            return int(z_dino.shape[-1])

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self._use_fallback:
            feat = self.trunk.forward_features(images)
            if feat.ndim == 3:
                tokens = feat
            else:
                tokens = feat.flatten(2).transpose(1, 2)
            pooled = tokens.mean(dim=1)
            z_clip = self.clip_proj(pooled)
            z_dino = self.dino_proj(tokens)
            return {
                "z_clip": z_clip,
                "z_clip_pooled": z_clip,
                "z_dino": z_dino,
            }

        out = self.radio(images, feature_fmt="NLC")
        z_clip = out[self.clip_key].summary
        z_dino = out[self.dino_key].features

        z_clip = self.clip_proj(z_clip)
        z_dino = self.dino_proj(z_dino)

        return {
            "z_clip": z_clip,
            "z_clip_pooled": z_clip,
            "z_dino": z_dino,
        }

    def get_lora_params(self) -> List[nn.Parameter]:
        return list(self.lora_params)

    def get_non_lora_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad and p not in set(self.lora_params)]
