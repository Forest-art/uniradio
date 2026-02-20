import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
from tqdm import tqdm

from .clip_text import CLIPTextEncoder
from .data_imagenet import build_imagenet_loader, make_batch_dict
from .decoder import ReconDecoder, downsample_target
from .eval_generation import evaluate_generation
from .eval_understanding import evaluate_understanding
from .grad_conflict import apply_conflict_aware, compute_grad_cosine
from .losses import feature_recon_loss, pixel_recon_loss, text_classification_loss
from .radio_wrapper import RadioWrapper
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


def build_decoder(cfg: Dict, dino_dim: int) -> ReconDecoder:
    dec_cfg = cfg.get("decoder", {})
    recon_cfg = cfg.get("recon", {})
    return ReconDecoder(
        in_dim=dino_dim,
        mode=recon_cfg.get("target", "feature_recon"),
        feature_target_dim=int(dec_cfg.get("feature_target_dim", dino_dim)),
        pixel_size=int(dec_cfg.get("pixel_size", 64)),
        hidden_dim=int(dec_cfg.get("hidden_dim", 1024)),
        token_dropout=float(dec_cfg.get("token_dropout", 0.0)),
    )


def build_teacher(cfg: Dict, device: torch.device) -> RadioWrapper:
    teacher_cfg = dict(cfg)
    teacher_cfg["lora"] = dict(cfg.get("lora", {}))
    teacher_cfg["lora"]["enable"] = False
    teacher_cfg["radio"] = dict(cfg.get("radio", {}))
    teacher_cfg["radio"]["freeze_trunk"] = True

    teacher = RadioWrapper(teacher_cfg).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def split_trainable_params(cfg: Dict, model: RadioWrapper, decoder: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    optimize_cfg = cfg.get("optimize", {})
    lora_enabled = bool(cfg.get("lora", {}).get("enable", False))

    lora_params = model.get_lora_params() if lora_enabled else []
    aux_params: List[nn.Parameter] = []

    if bool(optimize_cfg.get("decoder", True)):
        aux_params += [p for p in decoder.parameters() if p.requires_grad]

    if bool(optimize_cfg.get("projections", True)):
        aux_params += [p for p in model.clip_proj.parameters() if p.requires_grad]
        aux_params += [p for p in model.dino_proj.parameters() if p.requires_grad]

    if not lora_enabled and bool(optimize_cfg.get("model_when_no_lora", False)):
        aux_params += [p for p in model.parameters() if p.requires_grad]

    seen = set()
    uniq = []
    for p in aux_params:
        key = id(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    aux_params = uniq

    return lora_params, aux_params


def save_checkpoint(path: str, model: RadioWrapper, decoder: ReconDecoder, optimizer: torch.optim.Optimizer, step: int, cfg: Dict) -> None:
    ensure_dir(str(Path(path).parent))
    torch.save(
        {
            "model": model.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": cfg,
        },
        path,
    )


def run_eval(
    cfg: Dict,
    model: RadioWrapper,
    decoder: ReconDecoder,
    teacher_model: RadioWrapper,
    val_loader,
    text_embeddings: torch.Tensor,
    device: torch.device,
    run_dir: str,
    step: int,
) -> None:
    eval_cfg = cfg.get("eval", {})
    max_batches = eval_cfg.get("max_batches", None)

    u = evaluate_understanding(
        model=model,
        loader=val_loader,
        text_embeddings=text_embeddings,
        device=device,
        max_batches=max_batches,
    )
    g = evaluate_generation(
        model=model,
        decoder=decoder,
        loader=val_loader,
        device=device,
        recon_mode=cfg.get("recon", {}).get("target", "feature_recon"),
        teacher_model=teacher_model,
        pixel_loss_kind=cfg.get("recon", {}).get("pixel_loss", "mse"),
        max_batches=max_batches,
        calc_lpips=bool(eval_cfg.get("use_lpips", False)),
    )

    save_json(u, os.path.join(run_dir, "understanding.json"))
    save_json(g, os.path.join(run_dir, "generation.json"))
    save_json({"step": step, "understanding": u, "generation": g}, os.path.join(run_dir, "eval_last.json"))


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    cfg = apply_overrides(base_cfg, args.set)

    mixed_precision = cfg.get("accelerate", {}).get("mixed_precision", "no")
    accelerator = Accelerator(mixed_precision=mixed_precision)
    device = accelerator.device

    seed = int(cfg.get("seed", 42))
    set_seed(seed, device_specific=True)

    exp_name = args.run_name or cfg.get("experiment", {}).get("name", "exp")
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

    train_loader, class_names = build_imagenet_loader(
        data_root=cfg["data"]["data_root"],
        split="train",
        image_size=int(cfg["data"].get("image_size", 224)),
        batch_size=int(cfg["train"].get("batch_size", 32)),
        num_workers=int(cfg["data"].get("num_workers", 8)),
        class_names_file=cfg["data"].get("class_names_file"),
        shuffle=True,
        drop_last=True,
    )
    val_loader, _ = build_imagenet_loader(
        data_root=cfg["data"]["data_root"],
        split="val",
        image_size=int(cfg["data"].get("image_size", 224)),
        batch_size=int(cfg["eval"].get("batch_size", cfg["train"].get("batch_size", 32))),
        num_workers=int(cfg["data"].get("num_workers", 8)),
        class_names_file=cfg["data"].get("class_names_file"),
        shuffle=False,
        drop_last=False,
    )

    model = RadioWrapper(cfg).to(device)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, int(cfg["data"].get("image_size", 224)), int(cfg["data"].get("image_size", 224)), device=device)
        dino_dim = model(dummy)["z_dino"].shape[-1]

    decoder = build_decoder(cfg, dino_dim=dino_dim).to(device)
    teacher_model = None
    if cfg.get("recon", {}).get("target", "feature_recon") == "feature_recon":
        teacher_model = build_teacher(cfg, device=device)

    lora_params, aux_params = split_trainable_params(cfg, model=model, decoder=decoder)
    all_trainable = list(lora_params) + list(aux_params)
    if len(all_trainable) == 0:
        raise RuntimeError("No trainable parameters. Check lora.enable/optimize settings.")

    optim_cfg = cfg.get("optim", {})
    optimizer = torch.optim.AdamW(
        all_trainable,
        lr=float(optim_cfg.get("lr", 1e-4)),
        weight_decay=float(optim_cfg.get("weight_decay", 1e-4)),
    )

    model, decoder, optimizer, train_loader = accelerator.prepare(model, decoder, optimizer, train_loader)

    text_cfg = cfg.get("text", {})
    text_encoder = CLIPTextEncoder(
        model_name=text_cfg.get("clip_model", "ViT-B-32"),
        pretrained=text_cfg.get("clip_pretrained", "openai"),
        device=device,
    )
    text_cache = os.path.join(run_dir, "cache", "text_embeddings.pt")
    text_embeddings = text_encoder.build_class_embeddings(
        class_names=class_names,
        templates=text_cfg.get("prompt_templates", ["a photo of a {class}"]),
        cache_path=text_cache if accelerator.is_main_process else None,
        batch_size=int(text_cfg.get("batch_size", 256)),
    )

    steps = int(cfg.get("train", {}).get("steps", 1000))
    lambda_txt = float(cfg.get("train", {}).get("lambda_txt", 1.0))
    lambda_rec = float(cfg.get("train", {}).get("lambda_rec", 1.0))
    strategy = cfg.get("train", {}).get("strategy", "naive")
    temperature = float(cfg.get("text", {}).get("temperature", 0.07))

    log_every = int(cfg.get("log", {}).get("every", 10))
    cos_every = int(cfg.get("log", {}).get("cos_every", 10))
    save_every = int(cfg.get("log", {}).get("save_every", 200))
    eval_every = int(cfg.get("log", {}).get("eval_every", 200))

    metrics_file = os.path.join(run_dir, "metrics.jsonl")
    cos_curve = []

    if steps <= 0:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_checkpoint(
                os.path.join(ckpt_dir, "latest.pt"),
                model=accelerator.unwrap_model(model),
                decoder=accelerator.unwrap_model(decoder),
                optimizer=optimizer,
                step=0,
                cfg=cfg,
            )
            run_eval(
                cfg=cfg,
                model=accelerator.unwrap_model(model),
                decoder=accelerator.unwrap_model(decoder),
                teacher_model=teacher_model,
                val_loader=val_loader,
                text_embeddings=text_embeddings,
                device=device,
                run_dir=run_dir,
                step=0,
            )
        accelerator.wait_for_everyone()

    train_iter = cycle_loader(train_loader)
    pbar = tqdm(range(1, steps + 1), desc="train", disable=not accelerator.is_local_main_process)

    for step in pbar:
        batch = make_batch_dict(next(train_iter))
        batch = to_device(batch, device)

        model.train()
        decoder.train()

        out = model(batch["images"])
        z_clip = out["z_clip_pooled"]
        z_dino = out["z_dino"]

        Lu, txt_extra = text_classification_loss(
            image_emb=z_clip,
            labels=batch["labels"],
            text_emb=text_embeddings,
            temperature=temperature,
        )

        recon_target = cfg.get("recon", {}).get("target", "feature_recon")
        if recon_target == "feature_recon":
            if teacher_model is None:
                raise RuntimeError("teacher_model is required for feature_recon")
            with torch.no_grad():
                target_feat = teacher_model(batch["images"])["z_dino"].detach()
            pred_feat = decoder(z_dino)
            Lg, rec_extra = feature_recon_loss(pred_feat, target_feat)
        else:
            pred_img = decoder(z_dino)
            tgt_img = downsample_target(batch["images"], pred_img.shape[-1])
            Lg, rec_extra = pixel_recon_loss(
                pred_img,
                tgt_img,
                kind=cfg.get("recon", {}).get("pixel_loss", "mse"),
                huber_delta=float(cfg.get("recon", {}).get("huber_delta", 1.0)),
            )

        total = lambda_txt * Lu + lambda_rec * Lg

        optimizer.zero_grad(set_to_none=True)
        cos = 0.0
        if strategy == "conflict_aware":
            cos = apply_conflict_aware(
                loss_txt=Lu,
                loss_rec=Lg,
                lora_params=lora_params,
                aux_params=aux_params,
                lambda_txt=lambda_txt,
                lambda_rec=lambda_rec,
            )
            if accelerator.num_processes > 1:
                for p in list(lora_params) + list(aux_params):
                    if p.grad is not None:
                        p.grad = accelerator.reduce(p.grad, reduction="mean")
            optimizer.step()
        else:
            if step % cos_every == 0 and len(lora_params) > 0:
                cos = compute_grad_cosine(Lu, Lg, lora_params)
            accelerator.backward(total)
            optimizer.step()

        cos_global = _dist_mean_scalar(accelerator, cos, device)
        Lu_global = _dist_mean_scalar(accelerator, Lu, device)
        Lg_global = _dist_mean_scalar(accelerator, Lg, device)
        total_global = _dist_mean_scalar(accelerator, total, device)
        txt_acc_global = _dist_mean_scalar(accelerator, txt_extra.get("txt_acc", 0.0), device)

        rec_extra_global = {}
        for k, v in rec_extra.items():
            rec_extra_global[k] = _dist_mean_scalar(accelerator, v, device)

        if step % cos_every == 0 and accelerator.is_main_process:
            cos_curve.append({"step": step, "cos": cos_global})

        row = {
            "step": step,
            "Lu": Lu_global,
            "Lg": Lg_global,
            "total": total_global,
            "cos": cos_global,
            "txt_acc": txt_acc_global,
        }
        row.update(rec_extra_global)

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
                    decoder=accelerator.unwrap_model(decoder),
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
                    decoder=accelerator.unwrap_model(decoder),
                    teacher_model=teacher_model,
                    val_loader=val_loader,
                    text_embeddings=text_embeddings,
                    device=device,
                    run_dir=run_dir,
                    step=step,
                )
            accelerator.wait_for_everyone()

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
            "strategy": strategy,
            "seed": seed,
            "lambda_txt": lambda_txt,
            "lambda_rec": lambda_rec,
            "num_trainable_lora": count_parameters(lora_params),
            "num_trainable_aux": count_parameters(aux_params),
            "cos_mean": cos_mean,
            "cos_neg_ratio": cos_neg_ratio,
            "world_size": accelerator.num_processes,
        }
        save_json(summary, os.path.join(run_dir, "cos_summary.json"))
        save_json({"curve": cos_curve}, os.path.join(run_dir, "cos_curve.json"))

    accelerator.wait_for_everyone()
    accelerator.print(f"[train] done. run_dir={run_dir}")


if __name__ == "__main__":
    main()
