import copy
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import yaml

try:
    import numpy as np
except ImportError:
    np = None


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_value(raw: str) -> Any:
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        pass

    if raw.startswith("[") and raw.endswith("]"):
        raw_items = [x.strip() for x in raw[1:-1].split(",") if x.strip()]
        return [_parse_value(x) for x in raw_items]

    return raw


def set_by_dotted_key(cfg: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Invalid override: {ov}. Expected key=value")
        k, v = ov.split("=", 1)
        set_by_dotted_key(out, k.strip(), _parse_value(v.strip()))
    return out


def get_by_dotted_key(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    cur = cfg
    for p in key.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        nk = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, nk))
        else:
            out[nk] = v
    return out


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def count_parameters(params: List[torch.nn.Parameter]) -> int:
    return int(sum(p.numel() for p in params))
