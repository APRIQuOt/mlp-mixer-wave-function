"""Device-aware optimizer construction kept outside numerical model classes."""

from __future__ import annotations

from collections.abc import Iterable

import torch


ADAM_STRATEGIES = ("auto", "loop", "foreach", "fused")


def make_adam(
    parameters: Iterable[torch.nn.Parameter],
    *,
    learning_rate: float,
    strategy: str = "auto",
) -> torch.optim.Adam:
    """Build Adam with a launch-efficient implementation when the device supports it.

    MPS and CUDA pay heavily for one optimizer launch per parameter tensor, particularly for
    Mixer blocks made from several small linears and normalizations. PyTorch's fused Adam retains
    ordinary ``Parameter`` objects and autograd while updating them in a fused device operation.
    CPU keeps the conservative loop default.
    """
    if strategy not in ADAM_STRATEGIES:
        raise ValueError(f"Unknown Adam strategy: {strategy!r}.")
    values = list(parameters)
    if not values:
        raise ValueError("Adam requires at least one parameter.")
    devices = {parameter.device for parameter in values}
    if len(devices) != 1:
        raise ValueError("Adam parameters must reside on one device.")
    device = next(iter(devices))
    selected = (
        "fused"
        if strategy == "auto" and device.type in ("mps", "cuda")
        else "loop"
        if strategy == "auto"
        else strategy
    )
    options: dict[str, bool] = {}
    if selected == "fused":
        options["fused"] = True
    elif selected == "foreach":
        options["foreach"] = True
    elif selected == "loop":
        options["foreach"] = False
    return torch.optim.Adam(values, lr=learning_rate, **options)


__all__ = ["ADAM_STRATEGIES", "make_adam"]
