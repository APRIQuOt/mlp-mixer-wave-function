"""Portable PyTorch device, optimizer, and feature-precision selection."""
from dataclasses import replace
import torch
from tnqs.application.environment import EnvironmentOptions, HardwareProfile, ResolvedEnvironment
from tnqs.application.execution import ExecutionOptions, ExecutionReport


def detect_torch_environment():
    return HardwareProfile(bool(torch.cuda.is_available()), bool(torch.backends.mps.is_available()), str(torch.__version__))


def get_torch_environment(*, refresh=False):
    return detect_torch_environment()


refresh_torch_environment = detect_torch_environment


def resolve_torch_environment(*, device="auto", environment=EnvironmentOptions(), execution=ExecutionOptions(),
                              optimizer_strategy="auto", acceleration_backend="portable", **unused):
    profile = detect_torch_environment()
    device = str(device).lower()
    if device == "auto":
        device = "cuda" if profile.cuda_available else "mps" if profile.mps_available else "cpu"
    target = torch.device(device)
    if target.type not in ("cpu", "cuda", "mps"):
        raise ValueError("Use cpu, cuda[:index], mps, or auto.")
    if target.type == "cuda" and not profile.cuda_available:
        raise RuntimeError("CUDA requested but unavailable.")
    if target.type == "mps" and not profile.mps_available:
        raise RuntimeError("MPS requested but unavailable.")
    selected_optimizer = optimizer_strategy
    if selected_optimizer == "auto":
        selected_optimizer = "fused" if target.type in ("mps", "cuda") else "loop"
    if selected_optimizer not in ("loop", "foreach", "fused"):
        raise ValueError("Unknown Adam implementation.")
    precision = environment.precision
    if precision == "auto":
        raise ValueError("Specify compute precision explicitly for paper reproduction.")
    if precision == "fp16" and target.type != "cuda":
        raise ValueError("FP16 gradient scaling requires CUDA.")
    if precision == "bf16" and target.type == "mps":
        raise ValueError("BF16 feature computation is unsupported on MPS here.")
    return ResolvedEnvironment(device=str(target), selected_optimizer_strategy=selected_optimizer, selected_precision=precision)


def resolve_runtime_plan(runtime, *, sampler_strategy="auto", connected_strategy="auto", connection_strategy="auto",
                         compile_model=False, optimizer_strategy="auto", **unused):
    from tnqs.adapters.torch.precision import configure_model_precision
    base = runtime.environment or resolve_torch_environment(device=str(runtime.wavefunction.device),
        environment=runtime.environment_options, execution=runtime.execution_options,
        optimizer_strategy=runtime.requested_optimizer_strategy)
    from tnqs.adapters.torch.sampling import select_sampler_strategy
    sampler = select_sampler_strategy(runtime.wavefunction, sampler_strategy)
    if sampler not in ("cached", "full_prefix"):
        raise ValueError("Unknown sampler strategy.")
    if connection_strategy not in ("auto", "lattice"):
        raise ValueError("Paper Hamiltonians use lattice connections.")
    optimizer = base.selected_optimizer_strategy if optimizer_strategy == "auto" else optimizer_strategy
    runtime.environment = replace(base, selected_sampler_strategy=sampler, selected_connected_strategy=connected_strategy,
        selected_compile_model=bool(compile_model), selected_optimizer_strategy=optimizer)
    runtime.execution_report = ExecutionReport(runtime.execution_options)
    configure_model_precision(runtime.wavefunction, runtime.environment.selected_precision)
    return runtime.environment


def connection_chunk_limit(runtime, requested=65536):
    return requested

__all__ = ["detect_torch_environment", "get_torch_environment", "refresh_torch_environment", "resolve_torch_environment", "resolve_runtime_plan", "connection_chunk_limit"]
