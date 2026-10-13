"""Numerically safe categorical draws for autoregressive wavefunctions."""

from __future__ import annotations

import torch


@torch.no_grad()
def _inverse_cdf_sample(
    probabilities: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one category per row without ever selecting a zero-mass plateau.

    ``torch.multinomial`` on MPS can very rarely return an index whose probability is exactly
    zero. The strict inverse-CDF comparison below skips leading, interior, and trailing
    zero-probability plateaus, including when the uniform variate lies exactly on a CDF boundary.
    """

    if probabilities.ndim < 1 or probabilities.shape[-1] == 0:
        raise ValueError("Categorical probabilities require a non-empty final dimension.")
    cumulative = probabilities.cumsum(dim=-1)
    # Softmax sums can differ from one by a few ulps. Normalising by the computed total also
    # makes a trailing zero-mass plateau end at exactly one without assigning it rounding mass.
    cumulative = cumulative / cumulative[..., -1:]
    uniform = torch.rand(
        (*probabilities.shape[:-1], 1),
        dtype=probabilities.dtype,
        device=probabilities.device,
        generator=generator,
    )
    return (cumulative <= uniform).sum(dim=-1).clamp_max(probabilities.shape[-1] - 1)


@torch.no_grad()
def sample_probabilities(
    probabilities: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one categorical value, using a zero-mass-safe MPS implementation."""

    if probabilities.device.type == "mps":
        return _inverse_cdf_sample(probabilities, generator)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


__all__ = ["sample_probabilities"]
