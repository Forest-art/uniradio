import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    target_modules: Sequence[str] = ("blocks\\..*\\.(qkv|proj|fc1|fc2)",)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return out + delta


_SPECIAL_REGEX_TOKENS = set(".^$*+?{}[]|()")


def _compile_patterns(target_modules: Sequence[str]) -> List[re.Pattern]:
    pats = []
    for s in target_modules:
        if any(ch in _SPECIAL_REGEX_TOKENS for ch in s):
            pats.append(re.compile(s))
        else:
            pats.append(re.compile(re.escape(s)))
    return pats


def _matches(name: str, patterns: List[re.Pattern]) -> bool:
    return any(p.search(name) for p in patterns)


def _iter_named_linears(model: nn.Module) -> Iterable[Tuple[str, nn.Linear]]:
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            yield name, mod


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent_name, _, child_name = module_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, new_module)


def apply_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    target_modules: Sequence[str],
) -> List[nn.Parameter]:
    patterns = _compile_patterns(target_modules)
    replace_list = []
    for name, mod in _iter_named_linears(model):
        if _matches(name, patterns):
            replace_list.append((name, mod))

    if not replace_list:
        raise RuntimeError(
            "No Linear module matched lora.target_modules. "
            f"Patterns={list(target_modules)}"
        )

    for name, mod in replace_list:
        _replace_module(model, name, LoRALinear(base=mod, rank=rank, alpha=alpha))

    return list(iter_lora_parameters(model))


def iter_lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            yield mod.lora_A
            yield mod.lora_B


def mark_only_lora_trainable(model: nn.Module) -> List[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = False

    params = list(iter_lora_parameters(model))
    for p in params:
        p.requires_grad = True
    return params
