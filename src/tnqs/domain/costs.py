"""Measured parameter and supported-operation counts."""

from dataclasses import dataclass
from typing import Any, Mapping

@dataclass(frozen=True)
class ParameterCounts:
    total: int
    trainable: int
    by_component: Mapping[str, tuple[int, int]]

@dataclass(frozen=True)
class RuntimeFlopCounts:
    total: int
    all_flops: dict[str, dict[Any, int]]
    by_operator: Mapping[str, int]
    operator_calls: Mapping[str, int]
    unsupported_operator_calls: Mapping[str, int]
    supported_operator_calls: Mapping[str, int]
    batch_size: int
    device: str
    dtype: str
    include_conditioner: bool
    method: str = "torch_dispatch"
