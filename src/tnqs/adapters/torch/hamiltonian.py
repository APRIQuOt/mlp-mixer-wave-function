"""TFIM and Rydberg matrix elements in the computational basis."""
from math import isfinite
from abc import ABC, abstractmethod
from typing import NamedTuple
import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int

class LocalTerms(NamedTuple):
    """The nonzero entries of row sigma of H, for a batch of configurations."""
    diagonal: Float[torch.Tensor, 'batch_size']
    configurations: Int[torch.Tensor, 'batch_size num_connections Lx Ly']
    matrix_elements: Float[torch.Tensor, 'batch_size num_connections']

class Hamiltonian(nn.Module, ABC):
    """Base class holding the lattice shape.

    An nn.Module only so that index buffers follow .to(device) along with everything else.
    """

    def __init__(self, *, Lx: int, Ly: int) -> None:
        super().__init__()
        self.Lx, self.Ly = (Lx, Ly)
        self.num_sites = Lx * Ly

    @abstractmethod
    def local_terms(self, sigma: Int[torch.Tensor, 'batch_size Lx Ly']) -> LocalTerms:
        ...

class TransverseFieldIsing(Hamiltonian):

    def __init__(self, *, Lx: int, Ly: int, J: float=1.0, h: float=1.0, periodic: bool=False, device: torch.device=torch.device('cpu')) -> None:
        """H = -J sum_<ij> s^z_i s^z_j - h sum_i s^x_i on an Lx by Ly lattice.

        Every off-diagonal element is -h < 0, so for h > 0 the model is stoquastic: by
        Perron-Frobenius its ground state can be chosen strictly positive in this basis, and the
        exact phase is identically zero. That makes it the right first target, since the phase
        head should receive no gradient at all and any drift in it is a bug rather than physics.
        """
        super().__init__(Lx=Lx, Ly=Ly)
        self.J, self.h, self.periodic = (J, h, periodic)
        bonds = set()
        for x in range(Lx):
            for y in range(Ly):
                for dx, dy in ((1, 0), (0, 1)):
                    nx, ny = (x + dx, y + dy)
                    if periodic:
                        nx, ny = (nx % Lx, ny % Ly)
                    elif nx >= Lx or ny >= Ly:
                        continue
                    if (x, y) != (nx, ny):
                        bonds.add(tuple(sorted([(x, y), (nx, ny)])))
        bonds = sorted(bonds)
        self.num_bonds = len(bonds)
        self.register_buffer('bond_ax', torch.tensor([a[0] for a, _ in bonds]))
        self.register_buffer('bond_ay', torch.tensor([a[1] for a, _ in bonds]))
        self.register_buffer('bond_bx', torch.tensor([b[0] for _, b in bonds]))
        self.register_buffer('bond_by', torch.tensor([b[1] for _, b in bonds]))
        self.to(device)

    def local_terms(self, sigma: Int[torch.Tensor, 'batch_size Lx Ly']) -> LocalTerms:
        batch_size = sigma.shape[0]
        spins = 2 * sigma.long() - 1
        couplings = spins[:, self.bond_ax, self.bond_ay] * spins[:, self.bond_bx, self.bond_by]
        diagonal = -self.J * couplings.sum(-1).to(torch.get_default_dtype())
        flipped = sigma.reshape(batch_size, 1, -1).repeat(1, self.num_sites, 1)
        site = torch.arange(self.num_sites, device=sigma.device)
        flipped[:, site, site] = 1 - flipped[:, site, site]
        return LocalTerms(diagonal=diagonal, configurations=flipped.reshape(batch_size, self.num_sites, self.Lx, self.Ly), matrix_elements=torch.full((batch_size, self.num_sites), -self.h, dtype=diagonal.dtype, device=sigma.device))

class Rydberg(Hamiltonian):

    def __init__(self, *, Lx: int, Ly: int, Omega: float=1.0, delta: float=1.0, blockade_radius: float | None=None, C_6: float | None=None, lattice_spacing: float=1.0, interaction_cutoff: float | None=None, device: torch.device=torch.device('cpu')) -> None:
        """Rydberg atom array:

            H = -(Omega/2) sum_i s^x_i  -  delta sum_i n_i  +  sum_{i<j} V_ij n_i n_j

        with n_i = |r><r|_i the Rydberg occupation, which is exactly the stored bit, and
        V_ij = C_6 / r_ij^6 the van der Waals tail.

        The interaction is set either by C_6 directly or, if that is None, by the blockade radius
        through C_6 = Omega * R_b^6, so that V(R_b) = Omega: R_b is the distance at which the
        interaction matches the drive. The R_b form is the usual dimensionless parameterisation,
        where the phase diagram is drawn in (R_b/a, delta/Omega) and Omega is just the energy unit.
        Note it makes V proportional to Omega, so passing C_6 explicitly is what decouples the two
        and lets Omega -> 0 reach the classical (diagonal) limit with the blockade intact.

        Experiments usually write the drive as +(Omega/2) sum_i s^x. The two differ by conjugation
        with prod_i Z_i, which leaves n_i untouched and the spectrum unchanged, so we take the sign
        that makes every off-diagonal element -Omega/2 < 0 and the model manifestly stoquastic.

        Harder than the Ising case in the way that matters here: the diagonal couples every pair of
        sites rather than nearest neighbours, so blockade correlates distant patches, and delta
        competing against V produces ordered phases (checkerboard and beyond) rather than a single
        transition. The off-diagonal structure is unchanged, so it isolates the ansatz's ability to
        carry long-range correlation across the autoregressive ordering.

        Open boundaries throughout, which is what a finite atom array physically has.
        """
        if blockade_radius is not None and C_6 is not None:
            raise ValueError('blockade_radius and C_6 are mutually exclusive.')
        if blockade_radius is not None and (not isfinite(blockade_radius) or blockade_radius <= 0):
            raise ValueError('blockade_radius must be finite and positive.')
        if C_6 is not None and (not isfinite(C_6) or C_6 <= 0):
            raise ValueError('C_6 must be finite and positive.')
        super().__init__(Lx=Lx, Ly=Ly)
        self.Omega, self.delta = (Omega, delta)
        self.lattice_spacing = lattice_spacing
        if C_6 is None:
            self.blockade_radius = 1.2 if blockade_radius is None else blockade_radius
            self.C_6 = Omega * self.blockade_radius ** 6
        else:
            self.blockade_radius = None
            self.C_6 = C_6
        sites = [(x, y) for x in range(Lx) for y in range(Ly)]
        pairs, strengths = ([], [])
        for i, a in enumerate(sites):
            for b in sites[i + 1:]:
                distance = lattice_spacing * float(np.hypot(a[0] - b[0], a[1] - b[1]))
                if interaction_cutoff is not None and distance > interaction_cutoff:
                    continue
                pairs.append((a, b))
                strengths.append(self.C_6 / distance ** 6)
        self.num_pairs = len(pairs)
        self.register_buffer('pair_ax', torch.tensor([a[0] for a, _ in pairs]))
        self.register_buffer('pair_ay', torch.tensor([a[1] for a, _ in pairs]))
        self.register_buffer('pair_bx', torch.tensor([b[0] for _, b in pairs]))
        self.register_buffer('pair_by', torch.tensor([b[1] for _, b in pairs]))
        self.register_buffer('pair_strength', torch.tensor(strengths, dtype=torch.get_default_dtype()))
        self.to(device)

    def local_terms(self, sigma: Int[torch.Tensor, 'batch_size Lx Ly']) -> LocalTerms:
        batch_size = sigma.shape[0]
        occupation = sigma.to(self.pair_strength.dtype)
        blockade = (occupation[:, self.pair_ax, self.pair_ay] * occupation[:, self.pair_bx, self.pair_by] * self.pair_strength).sum(-1)
        diagonal = -self.delta * occupation.flatten(1).sum(-1) + blockade
        flipped = sigma.reshape(batch_size, 1, -1).repeat(1, self.num_sites, 1)
        site = torch.arange(self.num_sites, device=sigma.device)
        flipped[:, site, site] = 1 - flipped[:, site, site]
        return LocalTerms(diagonal=diagonal, configurations=flipped.reshape(batch_size, self.num_sites, self.Lx, self.Ly), matrix_elements=torch.full((batch_size, self.num_sites), -0.5 * self.Omega, dtype=diagonal.dtype, device=sigma.device))
