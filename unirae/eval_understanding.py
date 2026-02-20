import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from .clip_text import CLIPTextEncoder
from .data_imagenet import build_imagenet_loader, make_batch_dict
from .losses import align_text_dim, text_classification_loss
from .radio_wrapper import RadioWrapper
from .utils import apply_overrides, load_yaml, save_json, to_device


def evaluate_understanding(
    model: RadioWrapper,
    loader,
    text_embeddings: torch.Tensor,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    with torch.no_grad():
        iterator = enumerate(loader)
        for bi, batch in tqdm(iterator, total=max_batches, desc="eval_understanding", leave=False):
            if max_batches is not None and bi >= max_batches:
                break
            batch = make_batch_dict(batch)
            batch = to_device(batch, device)

            out = model(batch["images"])
            z_clip = out["z_clip_pooled"]
            loss, _ = text_classification_loss(z_clip, batch["labels"], text_embeddings)

            text_for_logits = align_text_dim(text_embeddings, z_clip.shape[-1])
            logits = torch.nn.functional.normalize(z_clip, dim=-1) @ torch.nn.functional.normalize(text_for_logits, dim=-1).t()
            pred = logits.argmax(dim=1)

            total += batch["labels"].numel()
            correct += (pred == batch["labels"]).sum().item()
            loss_sum += loss.item() * batch["labels"].shape[0]

    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)
    return {
        "zero_shot_acc": acc,
        "zero_shot_loss": avg_loss,
        "num_samples": total,
    }


def _load_model_for_eval(cfg: Dict, checkpoint_path: str, device: torch.device) -> RadioWrapper:
    model = RadioWrapper(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


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

    val_loader, class_names = build_imagenet_loader(
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

    text_cfg = cfg.get("text", {})
    text_encoder = CLIPTextEncoder(
        model_name=text_cfg.get("clip_model", "ViT-B-32"),
        pretrained=text_cfg.get("clip_pretrained", "openai"),
        device=device,
    )
    text_embeddings = text_encoder.build_class_embeddings(
        class_names=class_names,
        templates=text_cfg.get("prompt_templates", ["a photo of a {class}"]),
        cache_path=None,
        batch_size=int(text_cfg.get("batch_size", 256)),
    )

    model = _load_model_for_eval(cfg, args.checkpoint, device)

    metrics = evaluate_understanding(
        model=model,
        loader=val_loader,
        text_embeddings=text_embeddings,
        device=device,
        max_batches=args.max_batches,
    )

    save_json(metrics, args.output)
    print(f"[eval_understanding] wrote {args.output}: {metrics}")


if __name__ == "__main__":
    main()
