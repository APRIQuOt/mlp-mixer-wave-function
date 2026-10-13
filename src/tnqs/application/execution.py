"""Portable model execution used by the paper reproduction."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ExecutionOptions:
    mode: str = "portable"
    model_backend: str = "pytorch"
    strict: bool = False
    def __post_init__(self):
        if self.mode not in ("portable", "auto") or self.model_backend not in ("pytorch", "auto") or self.strict:
            raise ValueError("The paper repository supports portable PyTorch execution.")

@dataclass(frozen=True)
class ExecutionReport:
    requested: ExecutionOptions
    selections: tuple = ()
    @property
    def fully_fused(self):
        return False

__all__ = ["ExecutionOptions", "ExecutionReport"]
