"""Context-local operation progress without presentation dependencies.

Numerical routines may emit completed work here without depending on a notebook,
terminal, or progress-bar package. Outside a reporting context this is a no-op.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressEvent:
    operation: str
    completed: int = 0
    total: int | None = None
    unit: str = "sample"
    status: str = "running"
    detail: str = ""


ProgressCallback = Callable[[ProgressEvent], None]
_callback: ContextVar[ProgressCallback | None] = ContextVar(
    "tnqs_operation_progress", default=None
)


@contextmanager
def progress_reporting(callback: ProgressCallback | None) -> Iterator[None]:
    """Install a callback for this context and restore it even after failure."""
    token = _callback.set(callback)
    try:
        yield
    finally:
        _callback.reset(token)


def emit_progress(
    operation: str,
    completed: int = 0,
    total: int | None = None,
    unit: str = "sample",
    status: str = "running",
    detail: str = "",
) -> None:
    """Report actual completed work; ``total=None`` denotes an unknown budget."""
    callback = _callback.get()
    if callback is not None:
        callback(ProgressEvent(operation, completed, total, unit, status, detail))


__all__ = ["ProgressCallback", "ProgressEvent", "emit_progress", "progress_reporting"]
