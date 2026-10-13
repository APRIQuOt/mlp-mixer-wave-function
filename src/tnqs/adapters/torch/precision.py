"""Selective AMP for neural features, with full-precision probability/physics heads.

The runtime resolver owns eligibility. Numerical modules only consume its selected
precision. Autocast scopes end before heads, losses, backward, and optimizer steps;
parameters, caches, and optimizer state keep their original dtype.
"""

from __future__ import annotations

from functools import wraps
import math

import torch


def configure_model_precision(model: torch.nn.Module, precision: str) -> None:
    if precision not in ("fp32", "bf16", "fp16"):
        raise ValueError(f"Unresolved compute precision: {precision!r}.")
    for module in model.modules():
        module._tnqs_compute_precision = precision


def _restore_feature_dtype(value, dtype):
    if isinstance(value, torch.Tensor):
        return value.to(dtype) if value.dtype in (torch.float16, torch.bfloat16) else value
    if isinstance(value, list):
        return [_restore_feature_dtype(item, dtype) for item in value]
    if isinstance(value, tuple):
        items = tuple(_restore_feature_dtype(item, dtype) for item in value)
        return items if type(value) is tuple else type(value)(*items)
    return value


def autocast_features(function):
    """Autocast one feature encoder and restore outputs before probability heads.

Scopes are local to a single forward call: no process-global precision switches
and no persistent autocast weight cache across no-grad and differentiable passes.
"""
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        precision = getattr(self, "_tnqs_compute_precision", "fp32")
        if precision == "fp32":
            return function(self, *args, **kwargs)
        parameter = next(self.parameters())
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        with torch.autocast(parameter.device.type, dtype=dtype):
            result = function(self, *args, **kwargs)
        return _restore_feature_dtype(result, parameter.dtype)
    return wrapped


def make_grad_scaler(model: torch.nn.Module) -> torch.amp.GradScaler:
    """FP16 needs scaling; BF16 and ordinary precision use the identity scaler."""
    return torch.amp.GradScaler(
        "cuda", enabled=getattr(model, "_tnqs_compute_precision", "fp32") == "fp16"
    )


def scaled_optimizer_step(scaler, optimizer) -> bool:
    """Return whether an update occurred, so overflow does not advance schedules."""
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() >= previous_scale


@torch.no_grad()
def stable_clip_grad_norm_(parameters, max_norm: float) -> float:
    """Clip a global L2 gradient norm without overflowing its FP32 sum of squares.

    ``torch.nn.utils.clip_grad_norm_`` computes tensor norms at the gradient dtype.
    On MPS that is FP32, so a finite heavy-tailed VMC gradient can yield an infinite
    *reported* norm when squared. Rescaling by the largest magnitude first is
    algebraically identical and keeps every term in the accumulation at most one.
    """

    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    if not gradients:
        return 0.0
    maxima = torch.stack([gradient.detach().abs().max() for gradient in gradients])
    largest = maxima.max()
    if not bool(torch.isfinite(largest)):
        raise FloatingPointError("VMC gradient contains NaN or infinity.")
    largest_value = float(largest.item())
    if largest_value == 0.0:
        return 0.0
    scaled_square_norm = torch.stack([
        (gradient.detach() / largest).square().sum()
        for gradient in gradients
    ]).sum()
    if not bool(torch.isfinite(scaled_square_norm)):
        raise FloatingPointError(
            "VMC gradient norm is non-finite after stable rescaling."
        )
    norm = largest_value * math.sqrt(float(scaled_square_norm.item()))
    coefficient = min(1.0, float(max_norm) / (norm + 1e-12))
    if coefficient < 1.0:
        for gradient in gradients:
            gradient.mul_(coefficient)
    return norm
