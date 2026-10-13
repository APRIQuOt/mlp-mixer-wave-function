"""Framework-independent output records emitted by training use cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


class EnergyReport(NamedTuple):
    """One Adam VMC step's energy estimate and convergence diagnostics."""

    energy: float
    energy_density: float
    error: float
    variance: float
    imaginary: float
    num_samples: int = 0
    learning_rate: float = float("nan")
    entropy: float = float("nan")
    temperature: float = 0.0
    free_energy: float = float("nan")

@dataclass(frozen=True)
class EvaluationReport:
    """Independent physical diagnostics for a trained ground-state runtime."""

    energy: float
    energy_density: float
    variance: float
    standard_error: float
    imaginary: float
    num_samples: int
    entropy: float = float("nan")
    entropy_standard_error: float = float("nan")


__all__ = ["EnergyReport", "EvaluationReport"]
