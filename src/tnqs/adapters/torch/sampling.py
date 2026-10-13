"""Sampling policies for the paper's unprojected autoregressive wavefunctions."""
from __future__ import annotations
import torch

SAMPLER_STRATEGIES = ("auto", "cached", "full_prefix")


def select_sampler_strategy(model: object, requested: str) -> str:
    if requested not in SAMPLER_STRATEGIES:
        raise ValueError(f"Unknown sampler strategy: {requested!r}.")
    supports = bool(getattr(model, "supports_full_prefix_sampling", True)) and model.mode != "hierarchical"
    if requested == "full_prefix" and not supports:
        raise ValueError("Use cached sampling for this architecture.")
    if requested != "auto":
        return requested
    if model.device.type in ("mps", "cuda") and supports and model.num_tokens <= 128:
        return "full_prefix"
    return "cached"


@torch.no_grad()
def sample_wavefunction(model, condition: None, num_samples: int, *, strategy: str = "cached",
                        generator: torch.Generator | None = None) -> torch.Tensor:
    if condition is not None:
        raise ValueError("The paper wavefunctions have no conditioning input.")
    if num_samples < 0:
        raise ValueError("num_samples must be nonnegative.")
    if num_samples == 0:
        return torch.empty((0, model.Lx, model.Ly), dtype=torch.long, device=model.device)
    return model.sample(None, num_samples, strategy=strategy, generator=generator)


@torch.no_grad()
def sample_ground_state(model: object, num_samples: int, *, sampler_strategy: str = "auto",
                        generator: torch.Generator | None = None) -> torch.Tensor:
    return sample_wavefunction(model, None, num_samples,
                               strategy=select_sampler_strategy(model, sampler_strategy), generator=generator)


__all__ = ["SAMPLER_STRATEGIES", "sample_ground_state", "sample_wavefunction", "select_sampler_strategy"]
