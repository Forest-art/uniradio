from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from appendix_code.gradient_strategies import apply_two_task_strategy, make_param_groups
from appendix_code.toy_multitask_model import TinyMultiTaskNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal appendix demo for Joint / PCGrad / CAGrad / DSGA.")
    parser.add_argument("--strategy", default="dsga", choices=["joint", "pcgrad", "cagrad", "dsga"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cagrad_beta", type=float, default=0.35)
    parser.add_argument("--lambda_mag", type=float, default=0.2)
    parser.add_argument("--magnitude_scope", default="global", choices=["global", "layerwise"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = TinyMultiTaskNet(width=args.width, depth=args.depth, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    shared_params = [param for param in model.input_proj.parameters() if param.requires_grad]
    for block in model.blocks:
        shared_params.extend(param for param in block.parameters() if param.requires_grad)

    aux_params = [param for param in model.classifier.parameters() if param.requires_grad]
    aux_params.extend(param for param in model.decoder.parameters() if param.requires_grad)

    dsga_groups = make_param_groups(
        shared_params,
        {"input": model.input_proj.parameters(), **{f"block_{i}": block.parameters() for i, block in enumerate(model.blocks)}},
    )

    for step in range(1, args.steps + 1):
        images = torch.rand(args.batch_size, 3, 32, 32, device=device)
        labels = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss_understanding = F.cross_entropy(outputs["logits"], labels)
        loss_generation = F.mse_loss(outputs["recon"], images)

        stats = apply_two_task_strategy(
            loss_understanding=loss_understanding,
            loss_generation=loss_generation,
            shared_params=shared_params,
            aux_params=aux_params,
            strategy=args.strategy,
            cagrad_beta=args.cagrad_beta,
            dsga_groups=dsga_groups,
            dsga_lambda_mag=args.lambda_mag,
            dsga_magnitude_scope=args.magnitude_scope,
            dsga_conflict_threshold=0.0,
            dsga_conflict_only=False,
            dsga_norm_restore=False,
        )
        optimizer.step()

        print(
            f"step={step:03d} "
            f"Lu={loss_understanding.item():.4f} "
            f"Lg={loss_generation.item():.4f} "
            f"cos={stats.shared_grad_cosine:.4f} "
            f"m_t={stats.global_magnitude_gain:.4f}"
        )


if __name__ == "__main__":
    main()
