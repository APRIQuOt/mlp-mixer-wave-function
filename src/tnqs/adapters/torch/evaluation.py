"""Independent Born-sample evaluation of a frozen paper wavefunction."""
from __future__ import annotations
import torch
from tnqs.adapters.torch.evaluator import prepare_evaluator
from tnqs.adapters.torch.local_energy import local_energy
from tnqs.adapters.torch.randomness import independent_sampling
from tnqs.adapters.torch.environment import resolve_runtime_plan
from tnqs.application.reports import EvaluationReport


def _standard_error(values):
    if values.numel() < 2:
        return values.new_full((), float("nan"))
    return (values.var(unbiased=True) / values.numel()).sqrt()


@torch.no_grad()
def evaluate_ground_state(runtime, *, num_samples=1024, seed=None, batch_size=256,
                          connection_chunk_size=65536):
    """Evaluate fresh independent samples without advancing the training stream."""
    if num_samples <= 0 or batch_size <= 0 or connection_chunk_size <= 0:
        raise ValueError("Evaluation sample and batch sizes must be positive.")
    plan = runtime.environment or resolve_runtime_plan(runtime)
    energies, logs = [], []
    with independent_sampling(runtime, seed=seed):
        for start in range(0, num_samples, batch_size):
            sigma = runtime.sample(min(batch_size, num_samples - start))
            evaluator = prepare_evaluator(runtime.wavefunction, sigma, strategy=plan.selected_connected_strategy)
            energies.append(local_energy(runtime.wavefunction, runtime.hamiltonian, sigma,
                evaluator=evaluator, chunk_size=connection_chunk_size))
            logs.append(evaluator.base_logpsi.real)
    values, log_magnitudes = torch.cat(energies), torch.cat(logs)
    if not torch.isfinite(values).all():
        raise RuntimeError("Evaluation produced nonfinite local energies.")
    real = values.real
    energy = real.mean()
    variance = (values - values.mean()).abs().square().mean()
    surprisal = -2 * log_magnitudes
    return EvaluationReport(energy=float(energy), energy_density=float(energy)/runtime.hamiltonian.num_sites,
        variance=float(variance), standard_error=float(_standard_error(real)),
        imaginary=abs(float(values.imag.mean())), num_samples=values.numel(),
        entropy=float(surprisal.mean()), entropy_standard_error=float(_standard_error(surprisal)))

__all__ = ["evaluate_ground_state"]
