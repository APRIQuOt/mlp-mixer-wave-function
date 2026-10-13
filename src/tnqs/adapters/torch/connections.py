"""Memory-bounded single-spin connections for the paper Hamiltonians."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
import torch

from tnqs.adapters.torch.hamiltonian import Hamiltonian, Rydberg, TransverseFieldIsing


@dataclass(frozen=True)
class ConnectionTile:
    values: torch.Tensor
    base_indices: torch.Tensor
    matrix_elements: torch.Tensor


@dataclass
class StreamingConnectionPlan:
    """Store compact flip instructions and materialize one bounded tile at a time."""
    diagonal: torch.Tensor
    base_values: torch.Tensor
    value_shape: tuple[int, ...]
    base_indices: torch.Tensor
    locations: torch.Tensor
    matrix_elements: torch.Tensor

    @property
    def num_connections(self) -> int:
        return self.base_indices.numel()

    def iter_tiles(self, chunk_size: int, *, max_connection_elements: int = 1_048_576) -> Iterator[ConnectionTile]:
        if chunk_size <= 0 or max_connection_elements <= 0:
            raise ValueError("Connection tile limits must be positive.")
        width = self.base_values.shape[1]
        rows_per_tile = max(1, min(chunk_size, max_connection_elements // width))
        for start in range(0, self.num_connections, rows_per_tile):
            stop = min(start + rows_per_tile, self.num_connections)
            indices = self.base_indices[start:stop]
            connected = self.base_values.index_select(0, indices).clone()
            rows = torch.arange(stop - start, device=connected.device)
            locations = self.locations[start:stop]
            connected[rows, locations] = torch.bitwise_xor(connected[rows, locations], 1)
            yield ConnectionTile(connected.reshape(-1, *self.value_shape), indices,
                                 self.matrix_elements[start:stop])


def prepare_streaming_connections(hamiltonian: Hamiltonian, sigma: torch.Tensor, *,
                                  compact_connections: bool = True,
                                  acceleration_backend: str = "portable") -> StreamingConnectionPlan | None:
    """Construct the diagonal and every one-spin off-diagonal term using PyTorch."""
    if acceleration_backend not in ("auto", "portable", "pytorch"):
        raise ValueError("The paper implementation supports portable PyTorch execution.")
    if isinstance(hamiltonian, TransverseFieldIsing):
        spins = 2 * sigma.long() - 1
        couplings = (spins[:, hamiltonian.bond_ax, hamiltonian.bond_ay]
                     * spins[:, hamiltonian.bond_bx, hamiltonian.bond_by])
        diagonal = -hamiltonian.J * couplings.sum(-1).to(torch.get_default_dtype())
        element = -hamiltonian.h
    elif isinstance(hamiltonian, Rydberg):
        occupation = sigma.to(hamiltonian.pair_strength.dtype)
        interaction = (occupation[:, hamiltonian.pair_ax, hamiltonian.pair_ay]
                       * occupation[:, hamiltonian.pair_bx, hamiltonian.pair_by]
                       * hamiltonian.pair_strength).sum(-1)
        diagonal = -hamiltonian.delta * occupation.flatten(1).sum(-1) + interaction
        element = -0.5 * hamiltonian.Omega
    else:
        return None
    batch, sites = sigma.shape[0], sigma.shape[1] * sigma.shape[2]
    base_indices = torch.arange(batch, device=sigma.device).repeat_interleave(sites)
    locations = torch.arange(sites, device=sigma.device).repeat(batch)
    matrix = torch.full((batch * sites,), element, dtype=diagonal.dtype, device=sigma.device)
    return StreamingConnectionPlan(diagonal, sigma.reshape(batch, -1).contiguous(),
                                   tuple(sigma.shape[1:]), base_indices, locations, matrix)


__all__ = ["ConnectionTile", "StreamingConnectionPlan", "prepare_streaming_connections"]
