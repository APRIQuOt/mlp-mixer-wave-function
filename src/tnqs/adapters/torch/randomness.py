"""Reproducible runtime random streams."""

from __future__ import annotations

import random
import secrets
from contextlib import contextmanager

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch CPU, and CUDA streams used by the experiment.

    MPS uses PyTorch's device stream and is seeded through ``torch.manual_seed``.  Keeping this
    operation at the composition root means model construction, conditioners, and training code
    observe the same configured seed without making domain objects aware of random state.
    """

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generator(device: torch.device | str, seed: int) -> torch.Generator:
    """Create a dedicated device generator for sampling.

    ``torch.Generator(device=...)`` is supported by CPU/CUDA and newer MPS builds.  The fallback
    keeps CPU-only installations usable; callers still receive a deterministic stream, while the
    global PyTorch seed covers devices whose older runtime cannot construct a device generator.
    """

    target = torch.device(device)
    try:
        return torch.Generator(device=target).manual_seed(seed)
    except RuntimeError:
        return torch.Generator(device="cpu").manual_seed(seed)


@contextmanager
def independent_sampling(runtime, *, seed: int | None = None):
    """Isolate evaluation draws and sampler diagnostics, including on exceptions.

    Separate streams protect continuation reproducibility, not MCMC decorrelation:
    the autoregressive/rejection samples are already independent at fixed weights.
    """
    model = runtime.wavefunction
    device = torch.device(model.device)
    if seed is None:
        training_seed = getattr(runtime, "seed", None)
        seed = secrets.randbits(63) if training_seed is None else (training_seed + 1_000_003) % (2**63)
    old_generator = runtime.sampling_generator
    attributes = ("_sampling_generator", "last_sampling_acceptance", "last_sampling_proposals", "last_sampling_rounds")
    missing = object()
    saved = [(module, {key: getattr(module, key, missing) for key in attributes}) for module in model.modules()]
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
    with torch.random.fork_rng(devices=cuda_devices):
        try:
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            if device.type == "mps":
                torch.mps.manual_seed(seed)
            generator = make_generator(device, seed)
            # Older MPS builds have only a global device stream, protected above.
            if generator.device.type != device.type:
                generator = None
            runtime.sampling_generator = generator
            if hasattr(model, "set_sampling_generator"):
                model.set_sampling_generator(generator)
            yield generator
        finally:
            runtime.sampling_generator = old_generator
            for module, values in saved:
                for key, value in values.items():
                    if value is missing:
                        if hasattr(module, key):
                            delattr(module, key)
                    else:
                        setattr(module, key, value)
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)


__all__ = ["independent_sampling", "make_generator", "seed_everything"]
