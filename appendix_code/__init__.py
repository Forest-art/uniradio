"""Minimal appendix-ready DSGA reference implementation."""

from .gradient_strategies import MergeStats, apply_two_task_strategy, make_param_groups
from .toy_multitask_model import TinyMultiTaskNet

__all__ = [
    "MergeStats",
    "TinyMultiTaskNet",
    "apply_two_task_strategy",
    "make_param_groups",
]
