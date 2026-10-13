"""Generic runtime parameter inspection kept outside numerical model classes."""

from __future__ import annotations

from torch import nn

from tnqs.adapters.torch.builders import TorchAnsatzRuntime
from tnqs.domain.costs import ParameterCounts


def _counts(module: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return total, trainable


class TorchParameterInspector:
    def inspect(self, runtime: object) -> ParameterCounts:
        if not isinstance(runtime, TorchAnsatzRuntime):
            raise TypeError("TorchParameterInspector expects a TorchAnsatzRuntime.")
        components: dict[str, tuple[int, int]] = {
            "wavefunction": _counts(runtime.wavefunction)
        }
        return ParameterCounts(
            total=sum(values[0] for values in components.values()),
            trainable=sum(values[1] for values in components.values()),
            by_component=components,
        )


__all__ = ["TorchParameterInspector"]
