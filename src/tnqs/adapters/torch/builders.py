"""Construct the paper's static PyTorch wavefunctions from frozen specifications."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from tnqs.adapters.torch.environment import resolve_torch_environment, resolve_runtime_plan
from tnqs.application.environment import EnvironmentOptions, ResolvedEnvironment
from tnqs.application.execution import ExecutionOptions, ExecutionReport
from tnqs.domain.specifications import AnsatzSpec, BackboneKind, RydbergSpec, TFIMSpec


@dataclass
class TorchAnsatzRuntime:
    specification: AnsatzSpec
    wavefunction: nn.Module
    hamiltonian: nn.Module
    conditioner: None = None
    seed: int | None = None
    sampling_generator: torch.Generator | None = None
    execution_report: ExecutionReport = ExecutionReport(ExecutionOptions())
    environment: ResolvedEnvironment | None = None
    acceleration_report: tuple = ()
    environment_options: EnvironmentOptions = EnvironmentOptions()
    execution_options: ExecutionOptions = ExecutionOptions()
    requested_optimizer_strategy: str = "auto"
    requested_acceleration_backend: str = "portable"

    def log_psi(self, configurations: torch.Tensor):
        return self.wavefunction.log_psi(configurations)

    @torch.no_grad()
    def sample(self, num_samples: int):
        from tnqs.adapters.torch.sampling import sample_wavefunction
        plan = self.environment or resolve_runtime_plan(self)
        return sample_wavefunction(self.wavefunction, None, num_samples,
                                   generator=self.sampling_generator,
                                   strategy=plan.selected_sampler_strategy)


class TorchRuntimeBuilder:
    def build(self, specification: AnsatzSpec, *, device: str | torch.device = "auto",
              seed: int | None = None, execution: ExecutionOptions = ExecutionOptions(),
              environment: ResolvedEnvironment | None = None,
              environment_options: EnvironmentOptions = EnvironmentOptions(),
              optimizer_strategy: str = "auto", acceleration_backend: str = "portable",
              dtype: torch.dtype | None = None) -> TorchAnsatzRuntime:
        if acceleration_backend not in ("auto", "portable", "pytorch"):
            raise ValueError("The paper implementation uses portable PyTorch execution.")
        resolved = environment or resolve_torch_environment(
            device=str(device), environment=environment_options, execution=execution,
            optimizer_strategy=optimizer_strategy, acceleration_backend="portable")
        target = torch.device(resolved.device)
        if seed is not None:
            from tnqs.adapters.torch.randomness import make_generator, seed_everything
            seed_everything(seed)
            generator = make_generator(target, seed + 1)
        else:
            generator = None
        model = self._wavefunction(specification, target)
        model.set_sampling_generator(generator)
        hamiltonian = self._hamiltonian(specification, target)
        if dtype is not None:
            if dtype not in (torch.float32, torch.float64):
                raise ValueError("Model parameter precision must be float32 or float64.")
            complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
            def cast(tensor):
                if tensor.is_floating_point():
                    return tensor.to(dtype=dtype)
                if tensor.is_complex():
                    return tensor.to(dtype=complex_dtype)
                return tensor
            model._apply(cast)
            hamiltonian._apply(cast)
        runtime = TorchAnsatzRuntime(specification, model, hamiltonian, seed=seed,
                                     sampling_generator=generator, environment=resolved,
                                     environment_options=environment_options,
                                     execution_options=execution,
                                     requested_optimizer_strategy=optimizer_strategy)
        resolve_runtime_plan(runtime)
        return runtime

    @staticmethod
    def _wavefunction(specification: AnsatzSpec, device: torch.device) -> nn.Module:
        from tnqs.adapters.torch.mix import MixerWavefunction
        from tnqs.adapters.torch.sequence import SequenceBackboneWavefunction, RNNWavefunction, TransformerWavefunction
        values = specification.wavefunction
        lattice = values.lattice
        hierarchy = values.resolved_hierarchy
        common = dict(num_layers=values.num_layers, channel_dim=values.channel_dim,
                      patch_size=values.patch_size, Lx=lattice.Lx, Ly=lattice.Ly,
                      token_hidden_dim=values.token_hidden_dim, mode=values.mode.value,
                      use_phase=False, up_spin_target=None, device=device)
        subsampler = None
        if hierarchy is not None:
            local = hierarchy.subsampler
            needs_module = (values.backbone in (BackboneKind.RNN, BackboneKind.TRANSFORMER)
                            or local.backbone is not None or local.mode.value != "site"
                            or local.patch_size != 1 or local.num_layers is not None)
            if needs_module:
                kind = local.backbone or values.backbone
                cell = local.recurrent_cell if local.backbone is not None else values.recurrent_cell
                kwargs = dict(mode=local.mode.value, num_layers=values.subsampler_num_layers,
                              channel_dim=values.channel_dim, token_hidden_dim=values.token_hidden_dim,
                              Lx=hierarchy.block_size, Ly=hierarchy.block_size,
                              patch_size=local.patch_size, use_phase=False, up_spin_target=None,
                              recurrent_cell=cell, transformer_heads=values.transformer_heads,
                              device=device, contextual=True)
                if kind is BackboneKind.RNN:
                    subsampler = RNNWavefunction(**kwargs)
                elif kind is BackboneKind.TRANSFORMER:
                    subsampler = TransformerWavefunction(**kwargs)
                else:
                    subsampler = SequenceBackboneWavefunction(kind=kind, **kwargs)
        if values.backbone in (BackboneKind.RNN, BackboneKind.TRANSFORMER):
            return SequenceBackboneWavefunction(**common, kind=values.backbone,
                                               recurrent_cell=values.recurrent_cell,
                                               transformer_heads=values.transformer_heads,
                                               subsampler=subsampler)
        if values.backbone is BackboneKind.MIXER:
            return MixerWavefunction(**common, subsampler=subsampler)
        raise ValueError("Unsupported paper architecture.")

    @staticmethod
    def _hamiltonian(specification: AnsatzSpec, device: torch.device) -> nn.Module:
        from tnqs.adapters.torch.hamiltonian import Rydberg, TransverseFieldIsing
        lattice = specification.wavefunction.lattice
        common = dict(Lx=lattice.Lx, Ly=lattice.Ly, device=device)
        values = specification.hamiltonian
        if isinstance(values, TFIMSpec):
            return TransverseFieldIsing(**common, J=values.J, h=values.h, periodic=values.periodic)
        if isinstance(values, RydbergSpec):
            return Rydberg(**common, Omega=values.Omega, delta=values.delta,
                           blockade_radius=values.R_b, C_6=values.C_6,
                           interaction_cutoff=values.interaction_cutoff)
        raise TypeError("Unsupported paper Hamiltonian.")


__all__ = ["TorchAnsatzRuntime", "TorchRuntimeBuilder"]
