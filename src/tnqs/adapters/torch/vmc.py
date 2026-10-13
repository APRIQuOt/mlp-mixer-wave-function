"""Adam variational Monte Carlo for the real autoregressive paper wavefunctions.

The local energy is held fixed in the score-function gradient:
    dE = 2 <d log|psi| (E_local - <E_local>)>.
Sampling and connected-state evaluation share the exact model probabilities.
"""
from __future__ import annotations
import torch
from tnqs.adapters.torch.evaluator import prepare_evaluator, select_connected_strategy
from tnqs.adapters.torch.local_energy import local_energy
from tnqs.adapters.torch.mix import MixerWavefunction
from tnqs.adapters.torch.optimizers import make_adam
from tnqs.adapters.torch.precision import make_grad_scaler, scaled_optimizer_step, stable_clip_grad_norm_
from tnqs.adapters.torch.sampling import sample_ground_state
from tnqs.application.reports import EnergyReport


def energy_gradient_loss(log_psi, local_energies):
    """Differentiable energy-gradient surrogate, rather than an energy estimate."""
    fluctuation = (local_energies - local_energies.mean()).detach()
    return 2.0 * (log_psi.real * fluctuation.real).mean()


def summarise(local_energies, num_sites):
    real = local_energies.real
    correction = 1 if real.numel() > 1 else 0
    real_variance = real.var(unbiased=bool(correction)).item()
    variance = ((local_energies - local_energies.mean()).abs().square().sum()
                / (real.numel() - correction)).item()
    energy = real.mean().item()
    return EnergyReport(energy=energy, energy_density=energy / num_sites,
        error=(real_variance / real.numel()) ** .5, variance=variance,
        imaginary=abs(local_energies.imag.mean().item()), num_samples=real.numel())


def entropy_regulariser(log_psi):
    log_p = 2.0 * log_psi.real
    return ((log_p - log_p.mean()).detach() * log_p).mean()


def annealing_temperature(step, initial_temperature, warmup_steps, annealing_steps):
    if initial_temperature <= 0:
        return 0.0
    if step < warmup_steps:
        return initial_temperature
    progress = (step - warmup_steps) / max(1, annealing_steps - 1)
    return initial_temperature * max(0.0, 1.0 - progress)


def sample_count(step, num_steps, base, final, growth_start):
    """Original geometric sample schedule; explicit frozen counts remain unchanged."""
    if final is None or final <= base:
        return base
    first = int(growth_start * num_steps)
    if step <= first:
        return base
    progress = (step - first) / max(1, num_steps - 1 - first)
    return int(round(base * (final / base) ** min(progress, 1.0)))


def _can_reuse_base_evaluation(model, log_psi_func, compile_logpsi, gradient_chunk_size, connected_strategy):
    if (type(model) is not MixerWavefunction or compile_logpsi or gradient_chunk_size is not None
            or getattr(log_psi_func, "__self__", None) is not model
            or getattr(log_psi_func, "__func__", None) is not MixerWavefunction.log_psi
            or not callable(getattr(model, "evaluate_base", None))):
        return False
    if model.mode == "patch":
        return select_connected_strategy(model, connected_strategy) == "full_forward"
    return model.mode == "hierarchical" and model.subsampler is not None and connected_strategy in ("auto", "cached_suffix")


def optimise_ground_state(model, hamiltonian, *, num_samples, num_steps, learning_rate=1e-3,
        final_learning_rate_fraction=.01, final_num_samples=None, sample_growth_start=.5,
        initial_temperature=0., warmup_steps=0, annealing_steps=None,
        connected_strategy="auto", connection_strategy="auto", sampler_strategy="auto",
        compact_connections=True, connection_capacity_multiple=2048,
        chunk_size=65536, max_connection_elements=1048576, compile_logpsi=False,
        optimizer_strategy="auto", gradient_chunk_size=None, max_grad_norm=None,
        scheduler_factory=None, callback=None):
    """Train one paper design with its explicit sample and learning-rate schedules.

    A supplied scheduler_factory overrides the default cosine floor. Frozen
    campaign configs provide the observed scheduler, including zero-floor Mixer
    runs, rather than inferring it from final_learning_rate_fraction.
    """
    if connection_strategy not in ("auto", "lattice"):
        raise ValueError("Paper Hamiltonians require lattice connections.")
    optimiser = make_adam(model.parameters(), learning_rate=learning_rate, strategy=optimizer_strategy)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=num_steps,
        eta_min=learning_rate * final_learning_rate_fraction) if scheduler_factory is None
        else scheduler_factory(optimiser))
    scaler = make_grad_scaler(model)
    log_psi_func = model.log_psi
    if compile_logpsi:
        if not getattr(model, "supports_compile", True):
            raise ValueError("This recurrent backbone does not support compiled log_psi.")
        log_psi_func = torch.compile(model.log_psi, backend="inductor",
            mode="max-autotune" if model.device.type == "cuda" else "default", fullgraph=True, dynamic=False)
    reuse_base = _can_reuse_base_evaluation(model, log_psi_func, compile_logpsi,
                                          gradient_chunk_size, connected_strategy)
    history = []
    for step in range(num_steps):
        draws = sample_count(step, num_steps, num_samples, final_num_samples, sample_growth_start)
        sigma = sample_ground_state(model, draws, sampler_strategy=sampler_strategy)
        batch_size = draws if gradient_chunk_size is None else min(draws, gradient_chunk_size)
        if batch_size <= 0:
            raise ValueError("gradient_chunk_size must be positive.")
        base_evaluation = model.evaluate_base(sigma) if reuse_base else None
        energy_chunks, log_chunks = [], []
        with torch.no_grad():
            for start in range(0, draws, batch_size):
                batch = sigma[start:start + batch_size]
                evaluator = prepare_evaluator(model, batch, strategy=connected_strategy,
                    log_psi_func=log_psi_func, base_evaluation=base_evaluation)
                energy_chunks.append(local_energy(model, hamiltonian, batch, evaluator=evaluator,
                    chunk_size=chunk_size, compact_connections=compact_connections,
                    max_connection_elements=max_connection_elements))
                if gradient_chunk_size is not None:
                    log_chunks.append(evaluator.base_logpsi)
            local_energies = torch.cat(energy_chunks)
        temperature = annealing_temperature(step, initial_temperature, warmup_steps,
            annealing_steps if annealing_steps is not None else num_steps - warmup_steps)
        optimiser.zero_grad()
        if gradient_chunk_size is None:
            log_psi = base_evaluation.logpsi if base_evaluation is not None else log_psi_func(sigma)
            loss = energy_gradient_loss(log_psi, local_energies)
            if temperature > 0:
                loss = loss + temperature * entropy_regulariser(log_psi)
            scaler.scale(loss).backward()
        else:
            log_psi = torch.cat(log_chunks)
            fluctuations = (local_energies - local_energies.mean()).detach()
            log_p = 2 * log_psi.real
            entropy_weights = (log_p - log_p.mean()).detach()
            for start in range(0, draws, batch_size):
                window = slice(start, start + batch_size)
                batch_log = log_psi_func(sigma[window])
                loss = 2 * (batch_log.real * fluctuations[window].real).sum() / draws
                if temperature > 0:
                    loss = loss + temperature * (entropy_weights[window] * 2 * batch_log.real).sum() / draws
                scaler.scale(loss).backward()
        if base_evaluation is not None:
            log_psi = log_psi.detach()
            base_evaluation = None
        if max_grad_norm is not None:
            scaler.unscale_(optimiser)
            stable_clip_grad_norm_(model.parameters(), max_grad_norm)
        updated = scaled_optimizer_step(scaler, optimiser)
        current_rate = optimiser.param_groups[0]["lr"]
        if updated:
            scheduler.step()
        entropy = -(2 * log_psi.real).mean().item()
        report = summarise(local_energies, hamiltonian.num_sites)._replace(
            learning_rate=current_rate, entropy=entropy, temperature=temperature,
            free_energy=local_energies.real.mean().item() - temperature * entropy)
        history.append(report)
        if callback is not None:
            callback(step, report)
    return history

__all__ = ["optimise_ground_state", "energy_gradient_loss", "summarise", "sample_count"]
