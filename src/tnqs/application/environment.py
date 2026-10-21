"""Explicit device and precision settings for portable experiments."""
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class EnvironmentOptions:
    policy: str = "legacy"
    allow_fallback: bool = False
    memory_headroom_fraction: float = .8
    auto_connection_chunking: bool = True
    precision: str = "fp32"
    def __post_init__(self):
        if self.policy not in ("legacy", "auto"):
            raise ValueError("Unknown environment policy.")
        if self.precision not in ("auto", "fp32", "bf16", "fp16"):
            raise ValueError("Unknown compute precision.")
        if not 0 < self.memory_headroom_fraction <= 1:
            raise ValueError("Memory headroom must lie in (0,1].")

@dataclass(frozen=True)
class HardwareProfile:
    cuda_available: bool = False
    mps_available: bool = False
    torch_version: str = ""

@dataclass(frozen=True)
class ResolvedEnvironment:
    device: str
    selected_optimizer_strategy: str = "loop"
    selected_sampler_strategy: str = "cached"
    selected_connected_strategy: str = "auto"
    selected_connection_strategy: str = "lattice"
    selected_compile_model: bool = False
    selected_precision: str = "fp32"
    selected_acceleration_backend: str = "portable"
    selected_model_backend: str = "pytorch"
    selected_connection_chunk_size: int = 65536
    @property
    def device_type(self):
        return self.device.split(":")[0]
    def as_dict(self):
        return asdict(self)

__all__ = ["EnvironmentOptions", "HardwareProfile", "ResolvedEnvironment"]
