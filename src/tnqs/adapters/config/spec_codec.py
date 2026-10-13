"""Plain JSON codec for the frozen architectures in the paper campaign."""

from __future__ import annotations

from dataclasses import asdict, fields
from enum import Enum
from typing import Mapping

from tnqs.domain.errors import InvalidExperiment
from tnqs.domain.specifications import (
    AnsatzSpec, AutoregressiveMode, BackboneKind, HierarchySpec, LatticeSpec,
    RecurrentCell, RydbergSpec, SubsamplerSpec, SymmetryKind, SymmetrySpec,
    TaskKind, TFIMSpec, WavefunctionSpec,
)


def _plain(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _table(value, label: str) -> dict:
    if not isinstance(value, Mapping):
        raise InvalidExperiment(f"{label} must be an object.")
    return dict(value)


def _keys(data: dict, allowed: set[str], label: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise InvalidExperiment(f"Unknown {label} field(s): {', '.join(unknown)}.")


def specification_to_dict(specification: AnsatzSpec) -> dict:
    wavefunction = _plain(asdict(specification.wavefunction))
    hierarchy = specification.wavefunction.resolved_hierarchy
    wavefunction["hierarchy"] = None if hierarchy is None else _plain(asdict(hierarchy))
    if isinstance(specification.hamiltonian, TFIMSpec):
        kind = "tfim"
    elif isinstance(specification.hamiltonian, RydbergSpec):
        kind = "rydberg"
    else:
        raise InvalidExperiment("Unsupported paper Hamiltonian.")
    return {
        "task": "ground_state",
        "wavefunction": wavefunction,
        "hamiltonian": {"kind": kind, **asdict(specification.hamiltonian)},
        "symmetry": {"kind": "none"},
    }


def specification_from_dict(data: dict) -> AnsatzSpec:
    data = _table(data, "specification")
    _keys(data, {"task", "wavefunction", "hamiltonian", "symmetry"}, "specification")
    try:
        task = TaskKind(data.get("task", "ground_state"))
        wavefunction = _table(data["wavefunction"], "wavefunction")
        _keys(wavefunction, {field.name for field in fields(WavefunctionSpec)}, "wavefunction")
        wavefunction["backbone"] = BackboneKind(wavefunction["backbone"])
        wavefunction["mode"] = AutoregressiveMode(wavefunction["mode"])
        lattice = _table(wavefunction["lattice"], "lattice")
        _keys(lattice, {field.name for field in fields(LatticeSpec)}, "lattice")
        wavefunction["lattice"] = LatticeSpec(**lattice)
        wavefunction["recurrent_cell"] = RecurrentCell(wavefunction.get("recurrent_cell", "gru"))
        wavefunction["subsampler_recurrent_cell"] = RecurrentCell(wavefunction.get("subsampler_recurrent_cell", "gru"))
        if wavefunction.get("subsampler_backbone") is not None:
            wavefunction["subsampler_backbone"] = BackboneKind(wavefunction["subsampler_backbone"])
        hierarchy = wavefunction.get("hierarchy")
        if hierarchy is not None:
            hierarchy = _table(hierarchy, "hierarchy")
            _keys(hierarchy, {"block_size", "subsampler"}, "hierarchy")
            local = _table(hierarchy.get("subsampler", {}), "hierarchy.subsampler")
            _keys(local, {field.name for field in fields(SubsamplerSpec)}, "hierarchy.subsampler")
            if local.get("backbone") is not None:
                local["backbone"] = BackboneKind(local["backbone"])
            local["mode"] = AutoregressiveMode(local.get("mode", "site"))
            local["recurrent_cell"] = RecurrentCell(local.get("recurrent_cell", "gru"))
            hierarchy = HierarchySpec(hierarchy["block_size"], SubsamplerSpec(**local))
        wavefunction["hierarchy"] = hierarchy
        hamiltonian = _table(data["hamiltonian"], "hamiltonian")
        kind = hamiltonian.pop("kind")
        classes = {"tfim": TFIMSpec, "rydberg": RydbergSpec}
        if kind not in classes:
            raise InvalidExperiment("Unsupported paper Hamiltonian.")
        cls = classes[kind]
        _keys(hamiltonian, {field.name for field in fields(cls)}, "hamiltonian")
        symmetry = _table(data.get("symmetry", {"kind": "none"}), "symmetry")
        _keys(symmetry, {"kind"}, "symmetry")
        return AnsatzSpec(task=task, wavefunction=WavefunctionSpec(**wavefunction),
            hamiltonian=cls(**hamiltonian), symmetry=SymmetrySpec(SymmetryKind(symmetry.get("kind", "none"))))
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, InvalidExperiment):
            raise
        raise InvalidExperiment(f"Invalid frozen specification: {error}") from error


__all__ = ["specification_from_dict", "specification_to_dict"]
