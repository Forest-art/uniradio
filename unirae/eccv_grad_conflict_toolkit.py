"""
ECCV-ready gradient conflict toolkit.

模块一:
  - 在联合训练中提取两个任务的独立梯度 (g_u / g_g)
  - 计算逐层与全局的 cosine similarity / negative ratio

模块二:
  - 生成符合论文叙事的 dummy data
  - 绘制两张可投稿风格图:
      Figure 1: Layer-wise Spatial Heterogeneity
      Figure 2: Temporal Persistent Conflict
"""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None


# =========================
# Module 1: Hooking & Math
# =========================


@dataclass
class LayerConflictStat:
    layer: str
    cosine_similarity: float
    neg_ratio: float
    gu_norm: float
    gg_norm: float
    numel: int


class MultiTaskGradientProbe:
    """
    用于在同一 forward 图上提取两个任务损失的独立梯度并统计冲突。

    关键点:
    1) 使用两次 backward，并在中间 zero_grad，避免梯度污染。
    2) 对梯度做 detach().clone()，避免后续 inplace/optimizer 修改。
    3) 支持仅统计指定层，降低显存与开销。
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    @staticmethod
    def filter_named_params(
        named_params: Iterable[Tuple[str, nn.Parameter]],
        include_keywords: Optional[Sequence[str]] = None,
        exclude_keywords: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, nn.Parameter]]:
        """
        过滤层名。常见用法:
          - include_keywords=["blocks.", "layer4"]
          - exclude_keywords=["head", "decoder"]
        """
        include_keywords = include_keywords or []
        exclude_keywords = exclude_keywords or []
        out: List[Tuple[str, nn.Parameter]] = []
        for name, p in named_params:
            if not p.requires_grad:
                continue
            if include_keywords and not any(k in name for k in include_keywords):
                continue
            if exclude_keywords and any(k in name for k in exclude_keywords):
                continue
            out.append((name, p))
        return out

    @staticmethod
    def _clone_param_grads(named_params: Sequence[Tuple[str, nn.Parameter]]) -> Dict[str, torch.Tensor]:
        """
        拷贝当前 .grad 到独立字典:
          grad[name] = detached cloned tensor
        """
        grad_dict: Dict[str, torch.Tensor] = {}
        for name, p in named_params:
            if p.grad is None:
                continue
            grad_dict[name] = p.grad.detach().clone()
        return grad_dict

    def collect_two_task_grads_with_backward(
        self,
        loss_u: torch.Tensor,
        loss_g: torch.Tensor,
        named_shared_params: Sequence[Tuple[str, nn.Parameter]],
        optimizer: Optional[torch.optim.Optimizer] = None,
        keep_graph_for_final_joint_backward: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        标准双 backward 范式（用户可直接复用）:
          1) L_U.backward(retain_graph=True) -> 取 g_u
          2) zero_grad
          3) L_G.backward(retain_graph=...) -> 取 g_g
          4) zero_grad

        注意:
          - 若后续还要在同一图上执行 (L_U + L_G).backward()，第二次 backward 也要 retain_graph=True。
        """
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        else:
            for _, p in named_shared_params:
                p.grad = None

        # 第一次 backward: 仅理解任务梯度
        loss_u.backward(retain_graph=True)
        grad_u = self._clone_param_grads(named_shared_params)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        else:
            for _, p in named_shared_params:
                p.grad = None

        # 第二次 backward: 仅生成任务梯度
        loss_g.backward(retain_graph=bool(keep_graph_for_final_joint_backward))
        grad_g = self._clone_param_grads(named_shared_params)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        else:
            for _, p in named_shared_params:
                p.grad = None

        return grad_u, grad_g

    def _tensor_cosine(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.float().reshape(-1)
        b = b.float().reshape(-1)
        denom = (torch.linalg.norm(a) * torch.linalg.norm(b)).clamp_min(self.eps)
        return float(torch.dot(a, b).div(denom).item())

    def _tensor_neg_ratio(self, a: torch.Tensor, b: torch.Tensor) -> float:
        # 元素符号相反比例，忽略任一侧为 0 的元素。
        a = a.float().reshape(-1)
        b = b.float().reshape(-1)
        valid = (a != 0) & (b != 0)
        if int(valid.sum().item()) == 0:
            return 0.0
        opposite = (a[valid] * b[valid]) < 0
        return float(opposite.float().mean().item())

    def compute_layerwise_conflict(
        self,
        grad_u: Dict[str, torch.Tensor],
        grad_g: Dict[str, torch.Tensor],
    ) -> List[LayerConflictStat]:
        """
        对每一层参数计算:
          - cosine_similarity(cos(g_u, g_g))
          - neg_ratio
          - gu_norm / gg_norm
        """
        names = sorted(set(grad_u.keys()) & set(grad_g.keys()))
        stats: List[LayerConflictStat] = []
        for name in names:
            gu = grad_u[name]
            gg = grad_g[name]
            cos = self._tensor_cosine(gu, gg)
            neg = self._tensor_neg_ratio(gu, gg)
            stats.append(
                LayerConflictStat(
                    layer=name,
                    cosine_similarity=cos,
                    neg_ratio=neg,
                    gu_norm=float(torch.linalg.norm(gu.float()).item()),
                    gg_norm=float(torch.linalg.norm(gg.float()).item()),
                    numel=int(gu.numel()),
                )
            )
        return stats

    def compute_global_conflict(
        self,
        grad_u: Dict[str, torch.Tensor],
        grad_g: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        全局拼接后计算冲突统计。
        """
        names = sorted(set(grad_u.keys()) & set(grad_g.keys()))
        if not names:
            return {
                "global_cosine": 0.0,
                "global_neg_ratio": 0.0,
                "global_gu_norm": 0.0,
                "global_gg_norm": 0.0,
            }
        gu = torch.cat([grad_u[n].float().reshape(-1) for n in names], dim=0)
        gg = torch.cat([grad_g[n].float().reshape(-1) for n in names], dim=0)
        return {
            "global_cosine": self._tensor_cosine(gu, gg),
            "global_neg_ratio": self._tensor_neg_ratio(gu, gg),
            "global_gu_norm": float(torch.linalg.norm(gu).item()),
            "global_gg_norm": float(torch.linalg.norm(gg).item()),
        }


def standard_joint_step_example(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    probe: MultiTaskGradientProbe,
    shared_include_keywords: Sequence[str] = ("blocks.",),
) -> Dict[str, object]:
    """
    一个标准训练 step 示例（可粘贴进 train loop）:
      1) 前向得到 loss_u/loss_g
      2) 双 backward 提取 g_u/g_g
      3) 统计冲突
      4) 再做一次联合 backward + optimizer.step

    说明:
      - 此函数只演示流程；真实项目请替换为你的 forward / loss 定义。
    """
    model.train()
    out = model(images)
    logits = out["logits"]
    recon = out["recon"]
    target = out.get("target", images)

    loss_u = torch.nn.functional.cross_entropy(logits, labels)
    loss_g = torch.nn.functional.mse_loss(recon, target)

    named_shared = probe.filter_named_params(
        model.named_parameters(),
        include_keywords=shared_include_keywords,
    )

    # ---- (A) 梯度探针：两次 backward，互不污染 ----
    grad_u, grad_g = probe.collect_two_task_grads_with_backward(
        loss_u=loss_u,
        loss_g=loss_g,
        named_shared_params=named_shared,
        optimizer=optimizer,
        keep_graph_for_final_joint_backward=True,
    )
    layer_stats = probe.compute_layerwise_conflict(grad_u, grad_g)
    global_stats = probe.compute_global_conflict(grad_u, grad_g)

    # ---- (B) 正常联合更新 ----
    optimizer.zero_grad(set_to_none=True)
    (loss_u + loss_g).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return {
        "loss_u": float(loss_u.item()),
        "loss_g": float(loss_g.item()),
        "layer_stats": layer_stats,
        "global_stats": global_stats,
    }


# ===========================
# Module 2: ECCV Plotting API
# ===========================


class ECCVFigureBuilder:
    def __init__(self, seed: int = 2026):
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _configure_style(auto_latex: bool = True) -> None:
        # 优先用 colorblind 调色，避免审稿打印时可读性下降。
        if sns is not None:
            sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
        else:
            plt.style.use("seaborn-v0_8-whitegrid")

        use_latex = bool(auto_latex and shutil.which("latex") is not None)
        plt.rcParams.update(
            {
                "figure.dpi": 300,
                "savefig.dpi": 500,
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                "text.usetex": use_latex,
                "axes.titlesize": 14,
                "axes.labelsize": 13,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 10,
                "axes.linewidth": 1.0,
                "grid.alpha": 0.22,
                "grid.linewidth": 0.7,
            }
        )

    def simulate_spatial_heterogeneity(
        self,
        num_layers: int = 12,
        num_runs: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        构造“浅层协同、深层冲突”的层级数据。
        返回:
          layer_idx, mean_cos, std_cos
        """
        layers = np.arange(1, num_layers + 1)
        # 从正到负的主趋势，并加入轻微形状变化，避免过度线性。
        base = 0.18 - 0.035 * (layers - 1) + 0.015 * np.sin(layers / 1.8)
        samples = base[None, :] + self.rng.normal(0.0, 0.022, size=(num_runs, num_layers))
        mean = samples.mean(axis=0)
        std = samples.std(axis=0, ddof=1)
        return layers, mean, std

    def simulate_temporal_persistence(
        self,
        max_step: int = 10000,
        step_interval: int = 50,
        num_runs: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        构造“早期波动 + 中后期稳定在 ~0.5”的负余弦比例曲线。
        返回:
          steps, mean_neg_ratio, std_neg_ratio
        """
        steps = np.arange(0, max_step + 1, step_interval)
        # 初期从 ~0.42 向 0.5 回归，同时叠加衰减振荡。
        trend = 0.50 - 0.08 * np.exp(-steps / 700.0)
        osc = 0.06 * np.exp(-steps / 1400.0) * np.sin(steps / 170.0)
        sigma = 0.028 * np.exp(-steps / 1800.0) + 0.008

        runs = []
        for _ in range(num_runs):
            noise = self.rng.normal(0.0, sigma)
            y = np.clip(trend + osc + noise, 0.30, 0.70)
            runs.append(y)
        runs = np.stack(runs, axis=0)
        mean = runs.mean(axis=0)
        std = runs.std(axis=0, ddof=1)
        return steps, mean, std

    def plot_figure1_spatial(
        self,
        layers: np.ndarray,
        mean_cos: np.ndarray,
        std_cos: np.ndarray,
        out_path: Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))

        # 背景区分协同区域 / 冲突区域
        y_min = float(np.min(mean_cos - std_cos) - 0.04)
        y_max = float(np.max(mean_cos + std_cos) + 0.04)
        ax.axhspan(0.0, y_max, color="#E6F4EA", alpha=0.55, zorder=0)
        ax.axhspan(y_min, 0.0, color="#FDECEA", alpha=0.55, zorder=0)

        # 主曲线 + 误差带
        ax.plot(layers, mean_cos, color="#1f77b4", linewidth=2.4, marker="o", markersize=4.5, label="Mean Cosine")
        ax.fill_between(
            layers,
            mean_cos - std_cos,
            mean_cos + std_cos,
            color="#1f77b4",
            alpha=0.2,
            linewidth=0.0,
            label=r"$\pm$ 1 std",
        )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)

        # 注释文本
        ax.text(
            1.1,
            y_max - 0.03,
            "Synergistic (Shallow)",
            color="#2E7D32",
            fontsize=11,
            va="top",
        )
        ax.text(
            layers[-1] - 3.5,
            y_min + 0.03,
            "Conflicting (Deep)",
            color="#C62828",
            fontsize=11,
            va="bottom",
        )

        ax.set_xlim(layers[0] - 0.3, layers[-1] + 0.3)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(layers)
        ax.set_xlabel("Network Depth / Layer Index")
        ax.set_ylabel(r"Cosine Similarity $\,\cos(g_u, g_g)$")
        ax.set_title("Figure 1: Layer-wise Spatial Heterogeneity")
        ax.legend(loc="upper right", frameon=True, framealpha=0.95)

        fig.tight_layout()
        fig.savefig(out_path, dpi=500)
        fig.savefig(out_path.with_suffix(".pdf"), dpi=500)
        plt.close(fig)

    def plot_figure2_temporal(
        self,
        steps: np.ndarray,
        mean_neg: np.ndarray,
        std_neg: np.ndarray,
        out_path: Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8.6, 4.8))

        ax.plot(
            steps,
            mean_neg,
            color="#d62728",
            linewidth=2.4,
            label="Negative Cosine Proportion",
        )
        ax.fill_between(
            steps,
            mean_neg - std_neg,
            mean_neg + std_neg,
            color="#d62728",
            alpha=0.2,
            linewidth=0.0,
            label=r"$\pm$ 1 std",
        )

        # Plateau 辅助线（取 2k 后均值）
        plateau_mask = steps >= 2000
        plateau = float(mean_neg[plateau_mask].mean())
        ax.axhline(plateau, color="#2F4F4F", linestyle="--", linewidth=1.4)
        ax.annotate(
            "Structural Conflict Plateau",
            xy=(6200, plateau),
            xytext=(7000, plateau + 0.045),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#2F4F4F"),
            fontsize=10.5,
            color="#2F4F4F",
        )

        # 高亮平台区间
        ax.axvspan(2000, steps[-1], color="#E8EAF6", alpha=0.24, zorder=0)

        ax.set_xlim(0, int(steps[-1]))
        ax.set_ylim(0.30, 0.70)
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Proportion of Negative Cosine Similarity")
        ax.set_title("Figure 2: Temporal Persistent Conflict")
        ax.legend(loc="upper right", frameon=True, framealpha=0.95)

        fig.tight_layout()
        fig.savefig(out_path, dpi=500)
        fig.savefig(out_path.with_suffix(".pdf"), dpi=500)
        plt.close(fig)

    def build_dummy_figures(self, out_dir: Path, auto_latex: bool = True) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._configure_style(auto_latex=auto_latex)

        # Figure 1 data
        layers, mean_cos, std_cos = self.simulate_spatial_heterogeneity()
        np.savez(
            out_dir / "figure1_dummy_data.npz",
            layers=layers,
            mean_cos=mean_cos,
            std_cos=std_cos,
        )
        self.plot_figure1_spatial(
            layers=layers,
            mean_cos=mean_cos,
            std_cos=std_cos,
            out_path=out_dir / "figure1_spatial_heterogeneity.png",
        )

        # Figure 2 data
        steps, mean_neg, std_neg = self.simulate_temporal_persistence()
        np.savez(
            out_dir / "figure2_dummy_data.npz",
            steps=steps,
            mean_neg=mean_neg,
            std_neg=std_neg,
        )
        self.plot_figure2_temporal(
            steps=steps,
            mean_neg=mean_neg,
            std_neg=std_neg,
            out_path=out_dir / "figure2_temporal_persistence.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser("ECCV-style gradient conflict toolkit")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/eccv_dummy_figures",
        help="输出图和 dummy data 的目录",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--no_latex",
        action="store_true",
        help="禁用 LaTeX 文本渲染（当系统没有 latex 时建议打开）",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    builder = ECCVFigureBuilder(seed=args.seed)
    builder.build_dummy_figures(out_dir=out_dir, auto_latex=(not args.no_latex))

    print(f"[done] figures saved to: {out_dir}")
    print(f" - {out_dir / 'figure1_spatial_heterogeneity.png'}")
    print(f" - {out_dir / 'figure2_temporal_persistence.png'}")


if __name__ == "__main__":
    main()
