"""Frozen TFIM/Rydberg ground-state architectures used in the paper."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from typing import TypeAlias

from tnqs.domain.errors import InvalidExperiment


def _positive_integer(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InvalidExperiment(f"{name} must be a positive integer.")


def _finite(value: object, name: str) -> None:
    if not isinstance(value, Real) or isinstance(value, bool) or not isfinite(value):
        raise InvalidExperiment(f"{name} must be finite.")


class TaskKind(str, Enum):
    GROUND_STATE = "ground_state"


class BackboneKind(str, Enum):
    MIXER = "mixer"
    RNN = "rnn"
    TRANSFORMER = "transformer"


class AutoregressiveMode(str, Enum):
    SITE = "site"
    PATCH = "patch"
    HIERARCHICAL = "hierarchical"


class RecurrentCell(str, Enum):
    GRU = "gru"
    LSTM = "lstm"


class SymmetryKind(str, Enum):
    NONE = "none"


@dataclass(frozen=True)
class LatticeSpec:
    Lx: int
    Ly: int
    patch_size: int = 1

    def __post_init__(self) -> None:
        for name in ("Lx", "Ly", "patch_size"):
            _positive_integer(getattr(self, name), name)
        if self.Lx % self.patch_size or self.Ly % self.patch_size:
            raise InvalidExperiment("Lattice dimensions must be divisible by patch_size.")

    @property
    def num_sites(self) -> int:
        return self.Lx * self.Ly

    @property
    def sites_per_patch(self) -> int:
        return self.patch_size * self.patch_size

    @property
    def num_tokens(self) -> int:
        return (self.Lx // self.patch_size) * (self.Ly // self.patch_size)

    @property
    def patch_outcomes(self) -> int:
        return 1 << self.sites_per_patch


@dataclass(frozen=True)
class SubsamplerSpec:
    backbone: BackboneKind | None = None
    mode: AutoregressiveMode = AutoregressiveMode.SITE
    patch_size: int = 1
    recurrent_cell: RecurrentCell = RecurrentCell.GRU
    num_layers: int | None = None

    def __post_init__(self) -> None:
        if self.backbone is not None and not isinstance(self.backbone, BackboneKind):
            raise InvalidExperiment("Unsupported subsampler backbone.")
        if self.mode not in (AutoregressiveMode.SITE, AutoregressiveMode.PATCH):
            raise InvalidExperiment("Subsamplers must use site or patch mode.")
        _positive_integer(self.patch_size, "subsampler.patch_size")
        if self.num_layers is not None:
            _positive_integer(self.num_layers, "subsampler.num_layers")
        if self.mode is AutoregressiveMode.SITE and self.patch_size != 1:
            raise InvalidExperiment("A site subsampler requires patch_size=1.")
        if not isinstance(self.recurrent_cell, RecurrentCell):
            raise InvalidExperiment("Unsupported recurrent cell.")


@dataclass(frozen=True)
class HierarchySpec:
    block_size: int
    subsampler: SubsamplerSpec = SubsamplerSpec()

    def __post_init__(self) -> None:
        _positive_integer(self.block_size, "hierarchy.block_size")
        if self.block_size % self.subsampler.patch_size:
            raise InvalidExperiment("Hierarchy block_size must be divisible by subsampler patch_size.")

    @property
    def sites_per_block(self) -> int:
        return self.block_size * self.block_size

    @property
    def subsampler_num_tokens(self) -> int:
        side = self.block_size // self.subsampler.patch_size
        return side * side

    @property
    def subsampler_sites_per_token(self) -> int:
        return self.subsampler.patch_size * self.subsampler.patch_size

    @property
    def subsampler_outcomes(self) -> int:
        return 2 if self.subsampler.mode is AutoregressiveMode.SITE else 1 << self.subsampler_sites_per_token


@dataclass(frozen=True)
class TFIMSpec:
    J: float = 1.0
    h: float = 1.0
    periodic: bool = True

    def __post_init__(self) -> None:
        _finite(self.J, "J")
        _finite(self.h, "h")
        if self.h < 0:
            raise InvalidExperiment("The non-negative ground-state ansatz requires h >= 0.")
        if not isinstance(self.periodic, bool):
            raise InvalidExperiment("periodic must be a boolean.")


@dataclass(frozen=True)
class RydbergSpec:
    Omega: float = 1.0
    delta: float = 1.0
    R_b: float | None = None
    C_6: float | None = None
    interaction_cutoff: float | None = None

    def __post_init__(self) -> None:
        _finite(self.Omega, "Omega")
        _finite(self.delta, "delta")
        if self.Omega < 0:
            raise InvalidExperiment("The non-negative ground-state ansatz requires Omega >= 0.")
        if self.R_b is not None and self.C_6 is not None:
            raise InvalidExperiment("R_b and C_6 are mutually exclusive.")
        for name in ("R_b", "C_6", "interaction_cutoff"):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name)
                if value <= 0:
                    raise InvalidExperiment(f"{name} must be positive.")


HamiltonianSpec: TypeAlias = TFIMSpec | RydbergSpec


@dataclass(frozen=True)
class SymmetrySpec:
    """The paper uses the full, unprojected Hilbert space."""

    kind: SymmetryKind = SymmetryKind.NONE

    def __post_init__(self) -> None:
        if self.kind is not SymmetryKind.NONE:
            raise InvalidExperiment("The paper ansatz requires symmetry.kind='none'.")


@dataclass(frozen=True)
class WavefunctionSpec:
    backbone: BackboneKind
    mode: AutoregressiveMode
    lattice: LatticeSpec
    num_layers: int
    channel_dim: int
    token_hidden_dim: int
    use_phase: bool = False
    up_spin_target: None = None
    recurrent_cell: RecurrentCell = RecurrentCell.GRU
    transformer_heads: int = 4
    subsampler_backbone: BackboneKind | None = None
    subsampler_recurrent_cell: RecurrentCell = RecurrentCell.GRU
    hierarchy: HierarchySpec | None = None

    @property
    def resolved_hierarchy(self) -> HierarchySpec | None:
        if self.mode is not AutoregressiveMode.HIERARCHICAL:
            return None
        if self.hierarchy is not None:
            return self.hierarchy
        return HierarchySpec(self.lattice.patch_size, SubsamplerSpec(
            backbone=self.subsampler_backbone, recurrent_cell=self.subsampler_recurrent_cell))

    @property
    def patch_size(self) -> int:
        hierarchy = self.resolved_hierarchy
        return hierarchy.block_size if hierarchy is not None else self.lattice.patch_size

    @property
    def subsampler_num_layers(self) -> int:
        hierarchy = self.resolved_hierarchy
        return self.num_layers if hierarchy is None or hierarchy.subsampler.num_layers is None else hierarchy.subsampler.num_layers

    @property
    def sites_per_patch(self) -> int:
        return self.patch_size * self.patch_size

    @property
    def num_tokens(self) -> int:
        if self.mode is AutoregressiveMode.SITE:
            return self.lattice.num_sites
        return (self.lattice.Lx // self.patch_size) * (self.lattice.Ly // self.patch_size)

    @property
    def patch_outcomes(self) -> int:
        return 1 << self.sites_per_patch

    def __post_init__(self) -> None:
        if not isinstance(self.backbone, BackboneKind) or not isinstance(self.mode, AutoregressiveMode):
            raise InvalidExperiment("Unsupported wavefunction backbone or mode.")
        for name in ("num_layers", "channel_dim", "token_hidden_dim", "transformer_heads"):
            _positive_integer(getattr(self, name), name)
        if self.use_phase is not False:
            raise InvalidExperiment("Paper ground-state models require use_phase=False.")
        if self.up_spin_target is not None:
            raise InvalidExperiment("Paper models use the full Hilbert space.")
        if self.token_hidden_dim < self.num_tokens:
            raise InvalidExperiment("token_hidden_dim must be at least the number of tokens.")
        if self.mode is AutoregressiveMode.SITE and self.lattice.patch_size != 1:
            raise InvalidExperiment("Site mode requires patch_size=1.")
        if self.backbone is BackboneKind.TRANSFORMER and self.channel_dim % self.transformer_heads:
            raise InvalidExperiment("Transformer channel_dim must be divisible by transformer_heads.")
        if self.hierarchy is not None and self.mode is not AutoregressiveMode.HIERARCHICAL:
            raise InvalidExperiment("hierarchy is only valid for hierarchical mode.")
        if self.subsampler_backbone is not None and self.mode is not AutoregressiveMode.HIERARCHICAL:
            raise InvalidExperiment("subsampler_backbone is only valid for hierarchical mode.")
        hierarchy = self.resolved_hierarchy
        if hierarchy is not None:
            if self.lattice.Lx % hierarchy.block_size or self.lattice.Ly % hierarchy.block_size:
                raise InvalidExperiment("Lattice dimensions must be divisible by hierarchy.block_size.")
            local = hierarchy.subsampler.backbone
            if self.subsampler_backbone is not None and local is not None and self.subsampler_backbone is not local:
                raise InvalidExperiment("Conflicting subsampler backbones.")
        if not isinstance(self.recurrent_cell, RecurrentCell) or not isinstance(self.subsampler_recurrent_cell, RecurrentCell):
            raise InvalidExperiment("Unsupported recurrent cell.")


@dataclass(frozen=True)
class AnsatzSpec:
    task: TaskKind
    wavefunction: WavefunctionSpec
    hamiltonian: HamiltonianSpec
    conditioner: None = None
    symmetry: SymmetrySpec = SymmetrySpec()

    def __post_init__(self) -> None:
        if self.task is not TaskKind.GROUND_STATE:
            raise InvalidExperiment("Only ground-state experiments belong to this paper.")
        if not isinstance(self.hamiltonian, (TFIMSpec, RydbergSpec)):
            raise InvalidExperiment("Unsupported paper Hamiltonian.")
        if self.conditioner is not None:
            raise InvalidExperiment("Paper models have no conditioner.")
        if self.symmetry.kind is not SymmetryKind.NONE:
            raise InvalidExperiment("Paper models require symmetry.kind='none'.")


@dataclass(frozen=True)
class LogPsiOperation:
    batch_size: int = 1
    include_conditioner: bool = False

    def __post_init__(self) -> None:
        _positive_integer(self.batch_size, "batch_size")
        if self.include_conditioner is not False:
            raise InvalidExperiment("Paper FLOP counts contain no conditioner.")
