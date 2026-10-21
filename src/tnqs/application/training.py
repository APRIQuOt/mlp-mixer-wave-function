"""Adam VMC settings used by the ground-state experiments."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from tnqs.domain.errors import InvalidExperiment


def _positive_integer(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InvalidExperiment(f"{name} must be a positive integer.")


def _finite(value: object, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or not isfinite(value):
        raise InvalidExperiment(f"{name} must be finite.")
    return float(value)


@dataclass(frozen=True)
class GroundStateTrainingOptions:
    num_samples: int = 256
    num_steps: int = 1000
    learning_rate: float = 1e-3
    final_learning_rate_fraction: float = 0.01
    final_num_samples: int | None = None
    sample_growth_start: float = 0.5
    initial_temperature: float = 0.0
    warmup_steps: int = 0
    annealing_steps: int | None = None
    connected_strategy: str = "auto"
    connection_strategy: str = "auto"
    sampler_strategy: str = "auto"
    compact_connections: bool = True
    connection_capacity_multiple: int = 2048
    chunk_size: int | None = None
    max_connection_elements: int = 1048576
    compile_logpsi: bool | None = None
    optimizer_strategy: str = "auto"
    gradient_chunk_size: int | None = None
    max_grad_norm: float | None = None

    def __post_init__(self) -> None:
        for name in ("num_samples", "num_steps", "connection_capacity_multiple", "max_connection_elements"):
            _positive_integer(getattr(self, name), name)
        for name in ("final_num_samples", "annealing_steps", "chunk_size", "gradient_chunk_size"):
            if getattr(self, name) is not None:
                _positive_integer(getattr(self, name), name)
        if _finite(self.learning_rate, "learning_rate") <= 0:
            raise InvalidExperiment("learning_rate must be positive.")
        if not 0 <= _finite(self.final_learning_rate_fraction, "final_learning_rate_fraction") <= 1:
            raise InvalidExperiment("final_learning_rate_fraction must lie in [0, 1].")
        if not 0 <= _finite(self.sample_growth_start, "sample_growth_start") <= 1:
            raise InvalidExperiment("sample_growth_start must lie in [0, 1].")
        if _finite(self.initial_temperature, "initial_temperature") < 0:
            raise InvalidExperiment("initial_temperature cannot be negative.")
        if not isinstance(self.warmup_steps, int) or isinstance(self.warmup_steps, bool) or not 0 <= self.warmup_steps <= self.num_steps:
            raise InvalidExperiment("warmup_steps must lie within the training run.")
        if self.initial_temperature > 0 and self.warmup_steps == self.num_steps:
            raise InvalidExperiment("Annealing requires warmup_steps < num_steps.")
        if self.connected_strategy not in ("auto", "full_forward", "cached_suffix"):
            raise InvalidExperiment("Unsupported connected_strategy.")
        if self.connection_strategy not in ("auto", "lattice"):
            raise InvalidExperiment("Unsupported connection_strategy.")
        if self.sampler_strategy not in ("auto", "cached", "full_prefix"):
            raise InvalidExperiment("Unsupported sampler_strategy.")
        if self.optimizer_strategy not in ("auto", "loop", "foreach", "fused"):
            raise InvalidExperiment("Unsupported optimizer_strategy.")
        if not isinstance(self.compact_connections, bool):
            raise InvalidExperiment("compact_connections must be a boolean.")
        if self.compile_logpsi is not None and not isinstance(self.compile_logpsi, bool):
            raise InvalidExperiment("compile_logpsi must be a boolean or null.")
        if self.max_grad_norm is not None and _finite(self.max_grad_norm, "max_grad_norm") <= 0:
            raise InvalidExperiment("max_grad_norm must be positive.")


__all__ = ["GroundStateTrainingOptions"]
