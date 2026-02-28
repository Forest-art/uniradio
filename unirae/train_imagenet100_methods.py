import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .grad_conflict import apply_cagrad, apply_conflict_aware, apply_naive
from .losses import FeatureVarianceLoss
from .train_imagenet100_dynamics import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ViTBridgeModel,
    _autograd_grads_by_name,
    _global_grad_norm,
    _group_encoder_params,
    _init_distributed,
    _reduce_mean_scalar,
    _safe_cosine,
    _neg_ratio,
    build_encoder,
    build_imagenet100_dataloaders,
    evaluate_recon_and_understanding,
    save_layerwise_conflict_csv,
)
from .utils import append_jsonl, ensure_dir, save_json, seed_everything


def _materialize_grads(
    loss: torch.Tensor,
    params: Sequence[nn.Parameter],
    retain_graph: bool = False,
) -> List[torch.Tensor]:
    if len(params) == 0:
        return []
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    out: List[torch.Tensor] = []
    for g, p in zip(grads, params):
        if g is None:
            out.append(torch.zeros_like(p, memory_format=torch.preserve_format))
        else:
            out.append(g.detach().clone())
    return out


def _set_grads(params: Sequence[nn.Parameter], grads: Sequence[torch.Tensor]) -> None:
    for p, g in zip(params, grads):
        p.grad = g


def _all_trainable_params(module: nn.Module) -> List[nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def _all_reduce_grads_if_needed(
    params: Sequence[nn.Parameter],
    is_distributed: bool,
    world_size: int,
) -> None:
    if not is_distributed:
        return
    ws = float(max(1, int(world_size)))
    for p in params:
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(ws)


def _flatten(grads: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(grads) == 0:
        return torch.zeros(1)
    return torch.cat([g.reshape(-1) for g in grads], dim=0)


def _is_deep_param(name: str) -> bool:
    # LACAR 深层定义：Blocks.11 与最终 norm。
    return name.startswith("blocks.11") or name.startswith("norm")


def _build_param_groups(base_model: ViTBridgeModel):
    enc_named = [(n, p) for n, p in base_model.encoder.named_parameters() if p.requires_grad]
    cls_named = [(f"cls_head.{n}", p) for n, p in base_model.cls_head.named_parameters() if p.requires_grad]
    dec_named = [(f"decoder.{n}", p) for n, p in base_model.decoder.named_parameters() if p.requires_grad]

    enc_params = [p for _, p in enc_named]
    cls_params = [p for _, p in cls_named]
    dec_params = [p for _, p in dec_named]

    shared_params = enc_params
    aux_params = cls_params + dec_params
    return enc_named, cls_named, dec_named, shared_params, aux_params


def _collect_probe_stats(
    *,
    base_model: ViTBridgeModel,
    images: torch.Tensor,
    labels: torch.Tensor,
    images_01: torch.Tensor,
    run_dir: Path,
    global_step: int,
    encoder_groups: List[Tuple[str, int, List[Tuple[str, nn.Parameter]]]],
    enc_named: List[Tuple[str, nn.Parameter]],
) -> Dict[str, float]:
    probe_outputs = base_model(images)
    probe_logits = probe_outputs["logits"]
    probe_pred_patches = probe_outputs["pred_patches"]
    probe_gt_patches = base_model.patchify(images_01)

    probe_loss_u = F.cross_entropy(probe_logits, labels)
    probe_loss_g = F.mse_loss(probe_pred_patches, probe_gt_patches)

    dict_gu = _autograd_grads_by_name(probe_loss_u, enc_named, retain_graph=True)
    dict_gg = _autograd_grads_by_name(probe_loss_g, enc_named, retain_graph=False)

    layer_stats = save_layerwise_conflict_csv(
        step=global_step,
        dict_gu=dict_gu,
        dict_gg=dict_gg,
        encoder_groups=encoder_groups,
        out_dir=run_dir,
    )

    # 全局冲突统计
    g_all_u = _flatten([dict_gu[n] for n, _ in enc_named])
    g_all_g = _flatten([dict_gg[n] for n, _ in enc_named])
    layer_stats["global_cosine"] = float(_safe_cosine(g_all_u, g_all_g))
    layer_stats["global_neg_ratio"] = float(_neg_ratio(g_all_u, g_all_g))
    layer_stats["global_gu_norm"] = float(torch.norm(g_all_u).item())
    layer_stats["global_gg_norm"] = float(torch.norm(g_all_g).item())

    # 深层冲突统计（blocks.11 + norm）
    deep_names = [n for n, _ in enc_named if _is_deep_param(n)]
    if len(deep_names) > 0:
        g_deep_u = _flatten([dict_gu[n] for n in deep_names])
        g_deep_g = _flatten([dict_gg[n] for n in deep_names])
        layer_stats["deep_cosine"] = float(_safe_cosine(g_deep_u, g_deep_g))
        layer_stats["deep_neg_ratio"] = float(_neg_ratio(g_deep_u, g_deep_g))
        layer_stats["deep_gu_norm"] = float(torch.norm(g_deep_u).item())
        layer_stats["deep_gg_norm"] = float(torch.norm(g_deep_g).item())
    else:
        layer_stats["deep_cosine"] = 0.0
        layer_stats["deep_neg_ratio"] = 0.0
        layer_stats["deep_gu_norm"] = 0.0
        layer_stats["deep_gg_norm"] = 0.0

    return layer_stats


def _run_lacar_step(
    *,
    loss_u: torch.Tensor,
    loss_g: torch.Tensor,
    loss_var: torch.Tensor,
    enc_named: List[Tuple[str, nn.Parameter]],
    cls_params: List[nn.Parameter],
    dec_params: List[nn.Parameter],
    lambda_u: float,
    lambda_g: float,
    lambda_var: float,
) -> float:
    enc_params = [p for _, p in enc_named]

    gu_list = _materialize_grads(loss_u, enc_params, retain_graph=True)
    gg_list = _materialize_grads(loss_g, enc_params, retain_graph=True)
    gv_list = _materialize_grads(loss_var, enc_params, retain_graph=True)

    merged_enc: List[torch.Tensor] = []
    eps = 1e-12
    for (name, _), gu, gg, gv in zip(enc_named, gu_list, gg_list, gv_list):
        if _is_deep_param(name):
            # 深层非对称零空间投影：保留理解梯度，去除生成梯度中沿理解方向的分量。
            gu_flat = gu.reshape(-1)
            gg_flat = gg.reshape(-1)
            denom = torch.dot(gu_flat, gu_flat).clamp_min(eps)
            proj_coeff = torch.dot(gg_flat, gu_flat) / denom
            gg_proj = gg - proj_coeff * gu
            g_merge = float(lambda_u) * gu + float(lambda_g) * gg_proj
        else:
            # 浅层协同路由：直接相加。
            g_merge = float(lambda_u) * gu + float(lambda_g) * gg

        # 全层附加 FVR 梯度。
        g_merge = g_merge + float(lambda_var) * gv
        merged_enc.append(g_merge)

    cls_grads = _materialize_grads(loss_u, cls_params, retain_graph=True)
    dec_grads = _materialize_grads(loss_g, dec_params, retain_graph=False)

    _set_grads(enc_params, merged_enc)
    _set_grads(cls_params, [float(lambda_u) * g for g in cls_grads])
    _set_grads(dec_params, [float(lambda_g) * g for g in dec_grads])

    return float(_safe_cosine(_flatten(gu_list), _flatten(gg_list)))


def main() -> None:
    parser = argparse.ArgumentParser("IN100 Multi-Method Benchmark with DINO-Small")

    parser.add_argument(
        "--method",
        type=str,
        default="joint",
        choices=["und_only", "gen_only", "joint", "pcgrad", "cagrad", "lacar"],
        help="Experiment branch selector.",
    )
    parser.add_argument("--encoder_init", type=str, default="dinov2", choices=["scratch", "dinov2"])
    parser.add_argument("--encoder_ckpt", type=str, default="")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Local dataset root. Supports ImageFolder (train/val) or HF load_from_disk directory.",
    )
    parser.add_argument("--hf_dataset_id", type=str, default="clane9/imagenet-100")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--image_key", type=str, default="image")
    parser.add_argument("--label_key", type=str, default="label")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=10000)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=1000)

    parser.add_argument("--lambda_u", type=float, default=1.0)
    parser.add_argument("--lambda_g", type=float, default=1.0)
    parser.add_argument(
        "--auto_align_lambda_g",
        action="store_true",
        default=True,
        help="At step 1, set lambda_g = detach(Lu/Lg) for 1:1 weighted initialization.",
    )
    parser.add_argument("--disable_auto_align_lambda_g", action="store_true")

    parser.add_argument("--cagrad_beta", type=float, default=0.35)
    parser.add_argument("--lambda_var", type=float, default=0.20)
    parser.add_argument("--var_gamma", type=float, default=1.0)
    parser.add_argument("--var_eps", type=float, default=1e-4)

    parser.add_argument("--decoder_dim", type=int, default=384)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--decoder_heads", type=int, default=6)
    parser.add_argument("--decoder_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--decoder_drop_rate", type=float, default=0.0)

    parser.add_argument("--probe_every", type=int, default=500)
    parser.add_argument("--probe_until", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_max_batches", type=int, default=50)
    parser.add_argument("--eval_rfid_every", type=int, default=0)
    parser.add_argument("--final_eval_rfid", action="store_true", default=True)
    parser.add_argument("--no_final_eval_rfid", action="store_true")
    parser.add_argument("--eval_rfid_num_samples", type=int, default=2048)
    parser.add_argument("--eval_rfid_batch_size", type=int, default=64)
    parser.add_argument("--eval_rfid_tmp_dir", type=str, default="/tmp")

    parser.add_argument("--output_root", type=str, default="results/in100_method_bench")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--log_every", type=int, default=20)

    args = parser.parse_args()

    if args.disable_auto_align_lambda_g:
        args.auto_align_lambda_g = False
    if args.no_final_eval_rfid:
        args.final_eval_rfid = False

    is_distributed, rank, world_size, local_rank = _init_distributed()
    is_main_process = rank == 0

    seed_everything(int(args.seed) + int(rank))

    if is_distributed and args.device == "cpu" and torch.cuda.is_available():
        raise RuntimeError("Distributed launch with CUDA available does not support --device=cpu.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda is set but CUDA is not available.")

    if is_distributed and torch.cuda.is_available() and args.device != "cpu":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_loader, val_loader, data_meta = build_imagenet100_dataloaders(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        dataset_path=(str(args.dataset_path).strip() if str(args.dataset_path).strip() else None),
        hf_dataset_id=str(args.hf_dataset_id),
        cache_dir=(str(args.cache_dir) if args.cache_dir else None),
        image_key=str(args.image_key),
        label_key=str(args.label_key),
        distributed=is_distributed,
        rank=rank,
        world_size=world_size,
    )

    encoder = build_encoder(args.encoder_init, image_size=224, encoder_ckpt=str(args.encoder_ckpt))
    model = ViTBridgeModel(
        encoder=encoder,
        num_classes=int(data_meta["num_classes"]),
        image_size=224,
        decoder_dim=int(args.decoder_dim),
        decoder_depth=int(args.decoder_depth),
        decoder_heads=int(args.decoder_heads),
        decoder_mlp_ratio=float(args.decoder_mlp_ratio),
        decoder_drop_rate=float(args.decoder_drop_rate),
    ).to(device)

    # 使用 DDP 封装做多卡一致前向；梯度由本脚本手动写入并 all_reduce。
    if is_distributed:
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        else:
            model = DDP(model, broadcast_buffers=False)

    base_model = model.module if isinstance(model, DDP) else model
    all_trainable = _all_trainable_params(base_model)

    optimizer = torch.optim.AdamW(
        all_trainable,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
    )

    warmup_steps = max(1, int(args.warmup_steps))

    def _lr_lambda(step_idx: int) -> float:
        # 线性预热到基准 LR，预热后常数保持。
        s = step_idx + 1
        if s <= warmup_steps:
            return float(s) / float(warmup_steps)
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)

    fvr_loss = FeatureVarianceLoss(gamma=float(args.var_gamma), eps=float(args.var_eps)).to(device)

    run_name = str(args.run_name).strip()
    if not run_name:
        run_name = (
            f"in100_{args.method}_{args.encoder_init}_"
            f"bs{int(args.batch_size)}_s{int(args.seed)}"
        )
    run_dir = Path(args.output_root) / run_name
    ensure_dir(str(run_dir))

    if is_main_process:
        save_json(
            {
                "args": vars(args),
                "data_meta": data_meta,
                "device": str(device),
                "rank": int(rank),
                "world_size": int(world_size),
                "effective_global_batch": int(args.batch_size) * int(world_size),
                "num_trainable_params": int(sum(p.numel() for p in all_trainable)),
            },
            str(run_dir / "run_setup.json"),
        )

    metrics_path = run_dir / "train_metrics.jsonl"
    eval_metrics_path = run_dir / "eval_metrics.jsonl"

    encoder_groups = _group_encoder_params(base_model.encoder)
    enc_named, cls_named, dec_named, shared_params, aux_params = _build_param_groups(base_model)
    cls_params = [p for _, p in cls_named]
    dec_params = [p for _, p in dec_named]

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    runtime_lambda_g = float(args.lambda_g)
    aligned_once = False

    global_step = 0
    train_epoch = 0
    train_sampler = train_loader.sampler if isinstance(train_loader.sampler, DistributedSampler) else None
    if train_sampler is not None:
        train_sampler.set_epoch(train_epoch)
    train_iter = iter(train_loader)

    pbar = (
        tqdm(total=int(args.max_steps), desc=f"in100-{args.method}", dynamic_ncols=True)
        if is_main_process
        else None
    )

    while global_step < int(args.max_steps):
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(train_epoch)
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        global_step += 1

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images_01 = (images * std + mean).clamp_(0.0, 1.0)

        outputs = model(images)
        logits = outputs["logits"]
        pred_patches = outputs["pred_patches"]
        gt_patches = base_model.patchify(images_01)

        loss_u = F.cross_entropy(logits, labels)
        loss_g = F.mse_loss(pred_patches, gt_patches)

        # Step-0 自动损失对齐：lambda_g = detach(Lu/Lg)
        if (
            (not aligned_once)
            and args.auto_align_lambda_g
            and (args.method in {"joint", "pcgrad", "cagrad", "lacar"})
        ):
            lu_mean = _reduce_mean_scalar(float(loss_u.detach().item()), device, is_distributed, world_size)
            lg_mean = _reduce_mean_scalar(float(loss_g.detach().item()), device, is_distributed, world_size)
            runtime_lambda_g = float(lu_mean / max(lg_mean, 1e-12))
            aligned_once = True
            if is_main_process:
                print(
                    f"[align] step=1 raw_lu={lu_mean:.6f} raw_lg={lg_mean:.6f} "
                    f"=> lambda_g={runtime_lambda_g:.6f}"
                )

        loss_var = torch.tensor(0.0, device=device)
        if args.method == "lacar":
            # 用 encoder patch token 均值做方差约束，抑制表征坍缩。
            patch_tokens = outputs["tokens"][:, -base_model.num_patches :]
            feat_for_var = patch_tokens.mean(dim=1)
            loss_var = fvr_loss(feat_for_var)

        should_probe = (
            global_step <= int(args.probe_until)
            and (global_step % max(1, int(args.probe_every)) == 0)
        )

        probe_stats: Optional[Dict[str, float]] = None
        if should_probe and is_main_process:
            probe_stats = _collect_probe_stats(
                base_model=base_model,
                images=images,
                labels=labels,
                images_01=images_01,
                run_dir=run_dir,
                global_step=global_step,
                encoder_groups=encoder_groups,
                enc_named=enc_named,
            )

        optimizer.zero_grad(set_to_none=True)

        train_grad_cos = 0.0
        if args.method == "und_only":
            # encoder + cls_head 仅由 CE 监督。
            enc_params = [p for _, p in enc_named]
            enc_grads = _materialize_grads(loss_u, enc_params, retain_graph=True)
            cls_grads = _materialize_grads(loss_u, cls_params, retain_graph=False)
            _set_grads(enc_params, [float(args.lambda_u) * g for g in enc_grads])
            _set_grads(cls_params, [float(args.lambda_u) * g for g in cls_grads])

        elif args.method == "gen_only":
            # encoder + decoder 仅由重建监督。
            enc_params = [p for _, p in enc_named]
            enc_grads = _materialize_grads(loss_g, enc_params, retain_graph=True)
            dec_grads = _materialize_grads(loss_g, dec_params, retain_graph=False)
            _set_grads(enc_params, enc_grads)
            _set_grads(dec_params, dec_grads)

        elif args.method == "joint":
            train_grad_cos = apply_naive(
                loss_txt=loss_u,
                loss_rec=loss_g,
                shared_params=shared_params,
                aux_params=aux_params,
                lambda_txt=float(args.lambda_u),
                lambda_rec=float(runtime_lambda_g),
            )

        elif args.method == "pcgrad":
            train_grad_cos = apply_conflict_aware(
                loss_txt=loss_u,
                loss_rec=loss_g,
                lora_params=shared_params,
                aux_params=aux_params,
                lambda_txt=float(args.lambda_u),
                lambda_rec=float(runtime_lambda_g),
            )

        elif args.method == "cagrad":
            train_grad_cos = apply_cagrad(
                loss_txt=loss_u,
                loss_rec=loss_g,
                shared_params=shared_params,
                aux_params=aux_params,
                lambda_txt=float(args.lambda_u),
                lambda_rec=float(runtime_lambda_g),
                beta=float(args.cagrad_beta),
            )

        elif args.method == "lacar":
            train_grad_cos = _run_lacar_step(
                loss_u=loss_u,
                loss_g=loss_g,
                loss_var=loss_var,
                enc_named=enc_named,
                cls_params=cls_params,
                dec_params=dec_params,
                lambda_u=float(args.lambda_u),
                lambda_g=float(runtime_lambda_g),
                lambda_var=float(args.lambda_var),
            )

        else:
            raise ValueError(f"Unsupported method={args.method}")

        _all_reduce_grads_if_needed(all_trainable, is_distributed, world_size)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            rmse = torch.sqrt(loss_g.detach().clamp_min(0.0))
            loss_u_weighted = float(args.lambda_u) * loss_u
            if args.method == "gen_only":
                loss_g_weighted = loss_g
            else:
                loss_g_weighted = float(runtime_lambda_g) * loss_g
            if args.method == "lacar":
                total_loss = loss_u_weighted + loss_g_weighted + float(args.lambda_var) * loss_var
            elif args.method == "und_only":
                total_loss = loss_u_weighted
            elif args.method == "gen_only":
                total_loss = loss_g
            else:
                total_loss = loss_u_weighted + loss_g_weighted

        loss_u_log = _reduce_mean_scalar(float(loss_u.detach().item()), device, is_distributed, world_size)
        loss_g_log = _reduce_mean_scalar(float(loss_g.detach().item()), device, is_distributed, world_size)
        loss_var_log = _reduce_mean_scalar(float(loss_var.detach().item()), device, is_distributed, world_size)
        loss_total_log = _reduce_mean_scalar(float(total_loss.detach().item()), device, is_distributed, world_size)
        acc_log = _reduce_mean_scalar(float(acc.item()), device, is_distributed, world_size)
        rmse_log = _reduce_mean_scalar(float(rmse.item()), device, is_distributed, world_size)
        lr_now = float(optimizer.param_groups[0]["lr"])

        loss_u_weighted_log = float(args.lambda_u) * loss_u_log
        if args.method == "gen_only":
            loss_g_weighted_log = loss_g_log
        else:
            loss_g_weighted_log = float(runtime_lambda_g) * loss_g_log

        row = {
            "step": int(global_step),
            "method": str(args.method),
            "loss_u": float(loss_u_log),
            "loss_g": float(loss_g_log),
            "loss_var": float(loss_var_log),
            "loss_u_weighted": float(loss_u_weighted_log),
            "loss_g_weighted": float(loss_g_weighted_log),
            "loss_total": float(loss_total_log),
            "lambda_u": float(args.lambda_u),
            "lambda_g": float(runtime_lambda_g),
            "lambda_var": float(args.lambda_var if args.method == "lacar" else 0.0),
            "train_grad_cosine": float(train_grad_cos),
            "acc": float(acc_log),
            "rmse": float(rmse_log),
            "lr": float(lr_now),
            "loss_ratio_u_over_g": float(loss_u_log / max(loss_g_log, 1e-12)),
            "loss_ratio_weighted_u_over_g": float(loss_u_weighted_log / max(loss_g_weighted_log, 1e-12)),
        }

        if is_main_process and probe_stats is not None:
            row["probe_mean_cosine"] = float(probe_stats["mean_cosine"])
            row["probe_mean_neg_ratio"] = float(probe_stats["mean_neg_ratio"])
            row["probe_global_cosine"] = float(probe_stats["global_cosine"])
            row["probe_global_neg_ratio"] = float(probe_stats["global_neg_ratio"])
            row["probe_deep_cosine"] = float(probe_stats["deep_cosine"])
            row["probe_deep_neg_ratio"] = float(probe_stats["deep_neg_ratio"])
            row["probe_csv"] = str(probe_stats["csv_path"])
            print(
                f"[probe] step={global_step} "
                f"global_cos={probe_stats['global_cosine']:.4f} global_neg={probe_stats['global_neg_ratio']:.4f} "
                f"deep_cos={probe_stats['deep_cosine']:.4f} deep_neg={probe_stats['deep_neg_ratio']:.4f} "
                f"csv={probe_stats['csv_path']}"
            )

        if is_main_process:
            append_jsonl(str(metrics_path), row)
            if pbar is not None:
                pbar.update(1)
                if (global_step == 1) or (global_step % int(args.log_every) == 0) or (global_step == int(args.max_steps)):
                    pbar.set_postfix(
                        lu=f"{row['loss_u']:.4f}",
                        lg=f"{row['loss_g']:.4f}",
                        luw=f"{row['loss_u_weighted']:.4f}",
                        lgw=f"{row['loss_g_weighted']:.4f}",
                        acc=f"{row['acc']:.3f}",
                        rmse=f"{row['rmse']:.4f}",
                        lr=f"{row['lr']:.2e}",
                    )

        if int(args.eval_every) > 0 and (global_step % int(args.eval_every) == 0):
            do_rfid = int(args.eval_rfid_every) > 0 and (global_step % int(args.eval_rfid_every) == 0)
            eval_ret = evaluate_recon_and_understanding(
                model=base_model,
                val_loader=val_loader,
                device=device,
                mean=mean,
                std=std,
                max_batches=int(args.eval_max_batches),
                compute_rfid=bool(do_rfid),
                rfid_num_samples=int(args.eval_rfid_num_samples),
                rfid_batch_size=int(args.eval_rfid_batch_size),
                rfid_tmp_dir=str(args.eval_rfid_tmp_dir),
            )
            base_model.train()
            if is_main_process:
                eval_row = {"step": int(global_step), **eval_ret}
                append_jsonl(str(eval_metrics_path), eval_row)
                print(
                    f"[eval] step={global_step} acc={eval_ret['val_top1_acc']:.4f} "
                    f"rMSE={eval_ret['val_rmse']:.6f} rFID={eval_ret['val_rfid']:.4f}"
                )

    # 训练结束后保存 checkpoint。
    if is_main_process:
        ckpt = {
            "model": base_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": int(global_step),
            "runtime_lambda_g": float(runtime_lambda_g),
            "args": vars(args),
        }
        torch.save(ckpt, run_dir / "latest.pt")

    # 训练结束后强制评测一次 rFID。
    if bool(args.final_eval_rfid):
        final_ret = evaluate_recon_and_understanding(
            model=base_model,
            val_loader=val_loader,
            device=device,
            mean=mean,
            std=std,
            max_batches=int(args.eval_max_batches),
            compute_rfid=True,
            rfid_num_samples=int(args.eval_rfid_num_samples),
            rfid_batch_size=int(args.eval_rfid_batch_size),
            rfid_tmp_dir=str(args.eval_rfid_tmp_dir),
        )
        base_model.train()
        if is_main_process:
            final_row = {"step": int(global_step), "final_eval": True, **final_ret}
            append_jsonl(str(eval_metrics_path), final_row)
            save_json(final_row, str(run_dir / "final_eval.json"))
            print(
                f"[final_eval] step={global_step} acc={final_ret['val_top1_acc']:.4f} "
                f"rMSE={final_ret['val_rmse']:.6f} rFID={final_ret['val_rfid']:.4f}"
            )

    if pbar is not None:
        pbar.close()

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
