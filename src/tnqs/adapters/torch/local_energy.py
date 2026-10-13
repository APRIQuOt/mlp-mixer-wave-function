"""Hamiltonian ratio sums for ground-state variational Monte Carlo."""
from __future__ import annotations

import torch
from jaxtyping import Complex, Int
from tnqs.adapters.torch.hamiltonian import Hamiltonian
from tnqs.adapters.torch.connections import prepare_streaming_connections
from tnqs.application.progress import emit_progress


def local_energy(model, hamiltonian: Hamiltonian,
                 sigma: Int[torch.Tensor, "batch_size Lx Ly"],
                 condition: None = None, chunk_size: int = 65536,
                 evaluator: object | None = None, compact_connections: bool = True,
                 acceleration_backend: str = "portable", max_connection_elements: int = 1_048_576,
                 base_logpsi: Complex[torch.Tensor, "batch_size"] | None = None
                 ) -> Complex[torch.Tensor, "batch_size"]:
    """Evaluate each sample's diagonal energy plus its connected amplitude ratios."""
    if condition is not None:
        raise ValueError("The paper wavefunctions have no conditioning input.")
    if evaluator is not None and base_logpsi is not None:
        raise ValueError("base_logpsi and evaluator cannot both own the base evaluation.")
    base = evaluator.base_logpsi if evaluator is not None else base_logpsi
    if base is None:
        base = model.log_psi(sigma)
    plan = prepare_streaming_connections(hamiltonian, sigma,
                                        compact_connections=compact_connections,
                                        acceleration_backend=acceleration_backend)
    if plan is None:
        raise TypeError("Only the paper TFIM and Rydberg Hamiltonians are supported.")
    off_diagonal = torch.zeros_like(base)
    completed = 0
    emit_progress("energy.connections", 0, plan.num_connections, unit="connection",
                  detail=f"Local energies for {len(sigma)} samples")
    for tile in plan.iter_tiles(chunk_size, max_connection_elements=max_connection_elements):
        connected = (evaluator.log_psi(tile.values, tile.base_indices) if evaluator is not None
                     else model.log_psi(tile.values))
        ratios = torch.exp(connected - base.index_select(0, tile.base_indices))
        off_diagonal.index_add_(0, tile.base_indices, tile.matrix_elements.to(ratios.dtype) * ratios)
        completed += tile.base_indices.numel()
        emit_progress("energy.connections", completed, plan.num_connections, unit="connection",
                      status="complete" if completed == plan.num_connections else "running",
                      detail=f"Local energies for {len(sigma)} samples")
    if plan.num_connections == 0:
        emit_progress("energy.connections", 0, 0, unit="connection", status="complete")
    return plan.diagonal.to(base.dtype) + off_diagonal


__all__ = ["local_energy"]
