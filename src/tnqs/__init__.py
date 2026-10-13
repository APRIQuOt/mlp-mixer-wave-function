"""Neural quantum states used by the TFIM and Rydberg paper experiments."""

from tnqs.adapters.config.spec_codec import specification_from_dict, specification_to_dict
from tnqs.application.training import GroundStateTrainingOptions
from tnqs.domain.specifications import AnsatzSpec, BackboneKind, LatticeSpec, RydbergSpec, TFIMSpec, WavefunctionSpec

__all__ = ["AnsatzSpec", "BackboneKind", "GroundStateTrainingOptions", "LatticeSpec", "RydbergSpec", "TFIMSpec", "WavefunctionSpec", "specification_from_dict", "specification_to_dict"]
