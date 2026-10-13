"""Ground-state Adam VMC entry point for frozen paper experiments."""
from dataclasses import asdict
from tnqs.adapters.torch.environment import resolve_runtime_plan
from tnqs.adapters.torch.vmc import optimise_ground_state
from tnqs.application.training import GroundStateTrainingOptions


def train_ground_state(runtime, options: GroundStateTrainingOptions, *, scheduler_factory=None, callback=None):
    """Train a resolved TFIM/Rydberg runtime without rerunning budget search."""
    plan = resolve_runtime_plan(runtime, sampler_strategy=options.sampler_strategy,
        connected_strategy=options.connected_strategy, connection_strategy=options.connection_strategy,
        compile_model=options.compile_logpsi, optimizer_strategy=options.optimizer_strategy)
    parameters = asdict(options)
    parameters.update(sampler_strategy=plan.selected_sampler_strategy,
        optimizer_strategy=plan.selected_optimizer_strategy, compile_logpsi=plan.selected_compile_model,
        chunk_size=options.chunk_size or 65536)
    return optimise_ground_state(runtime.wavefunction, runtime.hamiltonian, **parameters,
        scheduler_factory=scheduler_factory, callback=callback)

__all__ = ["GroundStateTrainingOptions", "train_ground_state"]
