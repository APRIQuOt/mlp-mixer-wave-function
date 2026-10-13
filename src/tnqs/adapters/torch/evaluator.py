"""Prepared, cache-aware evaluation of connected configurations.

This evaluator is deliberately separate from the differentiable wavefunction path. VMC holds the local energy fixed in its score-function gradient, so these detached
autoregressive caches preserve the gradient while reducing connected-state work.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from jaxtyping import Complex, Int
from tnqs.adapters.torch.mix import MixerBaseEvaluation, MixerLayerCache, MixerWavefunction, base_evaluation_provenance

def _validate_base_evaluation(model, samples, condition_params, log_psi_func, data):
    if data is None:
        return
    function = model.log_psi if log_psi_func is None else log_psi_func
    if type(model) is not MixerWavefunction or condition_params is not None or getattr(function, '__self__', None) is not model or (getattr(function, '__func__', None) is not MixerWavefunction.log_psi):
        raise ValueError('Base reuse requires the original, unconditioned Mixer logpsi boundary.')
    if data.provenance is None or data.provenance != base_evaluation_provenance(model, samples):
        raise ValueError('Base evaluation belongs to different samples, model, or parameter snapshot.')

def _select_caches(caches: list[object], indices: torch.Tensor) -> list[object]:

    def select(cache: object) -> object:
        if hasattr(cache, 'token') and hasattr(cache, '_fields'):
            selected = []
            for name in cache._fields:
                value = getattr(cache, name)
                if name == 'token':
                    selected.append(select(value))
                elif torch.is_tensor(value):
                    selected.append(value.index_select(0, indices))
                else:
                    selected.append(value)
            return type(cache)(*selected)
        if hasattr(cache, 'norm') and hasattr(cache, 'hidden') and hasattr(cache, 'length'):
            return type(cache)(norm=cache.norm.index_select(0, indices), hidden=cache.hidden.index_select(0, indices), length=cache.length)
        raise TypeError(f'Unsupported layer cache type: {type(cache).__name__}.')
    return [select(cache) for cache in caches]

def _suffix_sums(values: torch.Tensor) -> torch.Tensor:
    """Return sums from each position onward, including a terminal zero."""
    reversed_values = torch.flip(values, dims=[1])
    if torch.is_complex(reversed_values):
        reversed_sum = torch.complex(torch.cumsum(reversed_values.real, dim=1), torch.cumsum(reversed_values.imag, dim=1))
    else:
        reversed_sum = torch.cumsum(reversed_values, dim=1)
    suffix = torch.zeros(values.size(0), values.size(1) + 1, dtype=values.dtype, device=values.device)
    suffix[:, :-1] = torch.flip(reversed_sum, dims=[1])
    return suffix

class PreparedMixerEvaluator:
    """Reuse one base autoregressive pass for many connected configurations."""

    def __init__(self, model: MixerWavefunction, samples: Int[torch.Tensor, 'batch_size Lx Ly'], condition_params: None) -> None:
        if condition_params is not None:
            raise ValueError('Paper ground states do not accept conditioning.')
        self.model = model
        self.samples = samples
        self.condition = condition_params
        self.patch_values = model.to_patch_indices(samples)
        self.batch_size = samples.size(0)
        with torch.no_grad():
            self._prepare()

    def _conditional_contribution(self, hidden: torch.Tensor, selected: torch.Tensor, unused_counts: torch.Tensor, position: int, condition_params: None) -> tuple[torch.Tensor, torch.Tensor]:
        logits, _ = self.model._step_outputs(hidden, condition_params)
        valid = torch.ones(selected.size(0), dtype=torch.bool, device=selected.device)
        log_prob = F.log_softmax(logits, dim=-1).gather(-1, selected.unsqueeze(-1)).squeeze(-1)
        return (torch.complex(0.5 * log_prob, torch.zeros_like(log_prob)), valid)

    def _prepare(self) -> None:
        patch_values = self.patch_values
        placed = torch.zeros(self.batch_size, dtype=torch.int64, device=patch_values.device)
        invalid = torch.zeros(self.batch_size, dtype=torch.bool, device=patch_values.device)
        contributions = []
        step_hidden = []
        caches: list[MixerLayerCache] | None = None
        for position in range(self.model.num_tokens):
            if position == 0:
                token = self.model.start_token.expand(self.batch_size, -1, -1)
            else:
                token = self.model._patch_embedding(patch_values[:, position - 1])
            hidden, caches = self.model._encode_step(token, self.condition, caches)
            step_hidden.append(hidden.squeeze(1).detach())
            contribution, valid = self._conditional_contribution(hidden, patch_values[:, position], placed, position, self.condition)
            contributions.append(contribution)
            invalid = invalid | ~valid
        self.step_hidden = torch.stack(step_hidden, dim=1)
        self.contributions = torch.stack(contributions, dim=1)
        self.base_logpsi = torch.complex(self.contributions.real.sum(dim=1), self.contributions.imag.sum(dim=1))
        assert caches is not None
        self.full_caches = caches
        self.suffix = _suffix_sums(self.contributions)

    @torch.no_grad()
    def log_psi(self, configurations: Int[torch.Tensor, 'batch_size Lx Ly'], base_indices: Int[torch.Tensor, 'batch_size'] | None=None) -> Complex[torch.Tensor, 'batch_size']:
        """Evaluate configurations using cached base prefixes.

        ``base_indices[i]`` identifies the prepared sample whose prefix cache and base sample
        belong to row ``i``. This supports the repeat-interleaved connected-state layout used by
        the Hamiltonian helpers without copying a full evaluator per connected configuration.
        """
        if base_indices is None:
            if configurations.size(0) != self.batch_size:
                raise ValueError('base_indices is required for a different batch size.')
            base_indices = torch.arange(self.batch_size, device=configurations.device)
        if base_indices.ndim != 1 or base_indices.size(0) != configurations.size(0):
            raise ValueError('base_indices must contain one row index per configuration.')
        flipped = self.model.to_patch_indices(configurations)
        base_tokens = self.patch_values.index_select(0, base_indices)
        changed = (flipped != base_tokens).any(dim=1)
        first_change = torch.zeros(configurations.size(0), dtype=torch.long, device=configurations.device)
        if changed.any():
            first_change[changed] = (flipped[changed] != base_tokens[changed]).to(torch.float32).argmax(dim=1)
        base_logpsi = self.base_logpsi.index_select(0, base_indices)
        out_real = base_logpsi.real.clone()
        out_imag = base_logpsi.imag.clone()
        for position in torch.unique(first_change[changed]).tolist():
            rows = changed & (first_change == int(position))
            row_indices = base_indices[rows]
            patch_rows = flipped[rows]
            row_condition = None
            placed = torch.zeros(patch_rows.size(0), dtype=torch.int64, device=patch_rows.device)
            hidden = self.step_hidden[:, position, :].index_select(0, row_indices)
            contribution, valid = self._conditional_contribution(hidden.unsqueeze(1), patch_rows[:, position], placed, int(position), row_condition)
            invalid = ~valid
            suffix_real = contribution.real
            suffix_imag = contribution.imag
            caches = None
            if position + 1 < self.model.num_tokens:
                caches = self.model._slice_caches(_select_caches(self.full_caches, row_indices), position + 1)
                for tail_position in range(position + 1, self.model.num_tokens):
                    token = self.model._patch_embedding(patch_rows[:, tail_position - 1])
                    hidden, caches = self.model._encode_step(token, row_condition, caches)
                    contribution, valid = self._conditional_contribution(hidden, patch_rows[:, tail_position], placed, tail_position, row_condition)
                    suffix_real = suffix_real + contribution.real
                    suffix_imag = suffix_imag + contribution.imag
                    invalid = invalid | ~valid
            base_suffix = self.suffix.index_select(0, row_indices)[:, position]
            out_real[rows] = base_logpsi.real[rows] - base_suffix.real + suffix_real
            out_imag[rows] = base_logpsi.imag[rows] - base_suffix.imag + suffix_imag
        return torch.complex(out_real, out_imag)

    @torch.no_grad()
    def ratios(self, configurations: Int[torch.Tensor, 'batch_size Lx Ly'], base_indices: Int[torch.Tensor, 'batch_size'] | None=None) -> Complex[torch.Tensor, 'batch_size']:
        """Return ``psi(configurations) / psi(base_configuration)``."""
        log_psi = self.log_psi(configurations, base_indices)
        if base_indices is None:
            base_indices = torch.arange(self.batch_size, device=configurations.device)
        return torch.exp(log_psi - self.base_logpsi.index_select(0, base_indices))

class PreparedHierarchicalEvaluator:
    """Batch all affected local factors while reusing unchanged hierarchical prefixes.

    Connected configurations still take one accelerator-friendly coarse forward.  Their local
    factors are flattened into one batch, but factors before the first changed coarse block are
    reused from the prepared base configurations.  This removes both the Python block loop and,
    for local changes, roughly half of the subsampler work without replacing large kernels by a
    long sequence of tiny suffix-cache launches.
    """

    def __init__(self, model: MixerWavefunction, samples: Int[torch.Tensor, 'batch_size Lx Ly'], condition_params: None, base_evaluation: MixerBaseEvaluation | None=None, log_psi_func=None) -> None:
        _validate_base_evaluation(model, samples, condition_params, log_psi_func, base_evaluation)
        if model.mode != 'hierarchical' or model.subsampler is None:
            raise ValueError('PreparedHierarchicalEvaluator requires an independent hierarchical subsampler.')
        if condition_params is not None:
            raise ValueError('Paper ground states do not accept conditioning.')
        self.model = model
        self.samples = samples
        self.condition = condition_params
        self.batch_size = samples.size(0)
        self.patch_bits = samples[:, model.patch_rows, model.patch_cols].long()
        with torch.no_grad():
            if base_evaluation is not None and base_evaluation.contributions is not None:
                if condition_params is not None or base_evaluation.contributions.shape != (self.batch_size, model.num_tokens):
                    raise ValueError('Reusable hierarchy factors require matching unconditioned base samples.')
                self.contributions = base_evaluation.contributions.detach()
                self.base_logpsi = base_evaluation.logpsi.detach()
                self.suffix = _suffix_sums(self.contributions)
            else:
                self._prepare()

    def _local_contributions(self, bits: torch.Tensor, context: torch.Tensor, positions: torch.Tensor, placed_before: torch.Tensor, condition_params: None) -> torch.Tensor:
        local_lattice = bits.reshape(bits.size(0), self.model.patch_size, self.model.patch_size)
        local_tokens = self.model.subsampler._tokens(local_lattice)
        amplitude, _ = self.model.subsampler._evaluate_tokens(local_tokens, condition_params, context=context)
        return torch.complex(amplitude, torch.zeros_like(amplitude))

    def _prepare(self) -> None:
        contexts = self.model._encode(self.samples, self.condition)
        positions = torch.arange(self.model.num_tokens, device=self.samples.device)
        pair_positions = positions.unsqueeze(0).expand(self.batch_size, -1).reshape(-1)
        pair_rows = torch.arange(self.batch_size, device=self.samples.device).repeat_interleave(self.model.num_tokens)
        counts = self.patch_bits.sum(-1)
        placed = F.pad(counts.cumsum(-1)[:, :-1], (1, 0))
        pair_condition = None
        contributions = self._local_contributions(self.patch_bits.reshape(-1, self.patch_bits.size(-1)), contexts.reshape(-1, contexts.size(-1)), pair_positions, placed.reshape(-1), pair_condition)
        self.contributions = contributions.reshape(self.batch_size, self.model.num_tokens)
        self.base_logpsi = torch.complex(self.contributions.real.sum(dim=1), self.contributions.imag.sum(dim=1))
        self.suffix = _suffix_sums(self.contributions)

    @torch.no_grad()
    def log_psi(self, configurations: Int[torch.Tensor, 'batch_size Lx Ly'], base_indices: Int[torch.Tensor, 'batch_size'] | None=None) -> Complex[torch.Tensor, 'batch_size']:
        if base_indices is None:
            if configurations.size(0) != self.batch_size:
                raise ValueError('base_indices is required for a different batch size.')
            base_indices = torch.arange(self.batch_size, device=configurations.device)
        if base_indices.ndim != 1 or base_indices.size(0) != configurations.size(0):
            raise ValueError('base_indices must contain one row index per configuration.')
        patch_bits = configurations[:, self.model.patch_rows, self.model.patch_cols].long()
        base_bits = self.patch_bits.index_select(0, base_indices)
        changed_tokens = (patch_bits != base_bits).any(dim=-1)
        changed = changed_tokens.any(dim=1)
        base_logpsi = self.base_logpsi.index_select(0, base_indices)
        out_real = base_logpsi.real.clone()
        out_imag = base_logpsi.imag.clone()
        if not changed.any():
            return torch.complex(out_real, out_imag)
        changed_rows = changed.nonzero(as_tuple=False).squeeze(-1)
        row_bits = patch_bits.index_select(0, changed_rows)
        row_indices = base_indices.index_select(0, changed_rows)
        row_condition = None
        contexts = self.model._encode(configurations.index_select(0, changed_rows), row_condition)
        first_change = changed_tokens.index_select(0, changed_rows).to(torch.float32).argmax(dim=1)
        positions = torch.arange(self.model.num_tokens, device=configurations.device).unsqueeze(0).expand(row_bits.size(0), -1)
        active_rows, active_positions = (positions >= first_change.unsqueeze(-1)).nonzero(as_tuple=True)
        counts = row_bits.sum(-1)
        placed = F.pad(counts.cumsum(-1)[:, :-1], (1, 0))
        pair_condition = None
        contributions = self._local_contributions(row_bits[active_rows, active_positions], contexts[active_rows, active_positions], active_positions, placed[active_rows, active_positions], pair_condition)
        contribution_real = torch.zeros(row_bits.size(0), self.model.num_tokens, dtype=contributions.real.dtype, device=row_bits.device)
        contribution_imag = torch.zeros_like(contribution_real)
        contribution_real[active_rows, active_positions] = contributions.real
        contribution_imag[active_rows, active_positions] = contributions.imag
        suffix_real = contribution_real.sum(-1)
        suffix_imag = contribution_imag.sum(-1)
        selected_base = base_logpsi.index_select(0, changed_rows)
        base_suffix = self.suffix.index_select(0, row_indices).gather(1, first_change.unsqueeze(-1)).squeeze(-1)
        candidate_real = selected_base.real - base_suffix.real + suffix_real
        candidate_imag = selected_base.imag - base_suffix.imag + suffix_imag
        out_real[changed_rows] = candidate_real
        out_imag[changed_rows] = candidate_imag
        return torch.complex(out_real, out_imag)

    @torch.no_grad()
    def ratios(self, configurations: Int[torch.Tensor, 'batch_size Lx Ly'], base_indices: Int[torch.Tensor, 'batch_size'] | None=None) -> Complex[torch.Tensor, 'batch_size']:
        if base_indices is None:
            base_indices = torch.arange(self.batch_size, device=configurations.device)
        return torch.exp(self.log_psi(configurations, base_indices) - self.base_logpsi.index_select(0, base_indices))

class PreparedFullEvaluator:
    """Correct fallback when a mode has no suffix-replay implementation yet."""

    def __init__(self, model, samples: torch.Tensor, condition_params: None, log_psi_func=None, base_evaluation: MixerBaseEvaluation | None=None) -> None:
        _validate_base_evaluation(model, samples, condition_params, log_psi_func, base_evaluation)
        self.model = model
        self.samples = samples
        self.condition = condition_params
        self.log_psi_func = model.log_psi if log_psi_func is None else log_psi_func
        with torch.no_grad():
            self.base_logpsi = self.log_psi_func(samples, condition_params) if base_evaluation is None else base_evaluation.logpsi.detach()
        if self.base_logpsi.shape != (samples.size(0),):
            raise ValueError('Base logpsi must contain one value per sample.')
        self.batch_size = samples.size(0)

    @torch.no_grad()
    def log_psi(self, configurations: torch.Tensor, base_indices: torch.Tensor | None=None) -> torch.Tensor:
        if base_indices is None:
            if configurations.size(0) != self.batch_size:
                raise ValueError('base_indices is required for a different batch size.')
            base_indices = torch.arange(self.batch_size, device=configurations.device)
        row_condition = None
        return self.log_psi_func(configurations, row_condition)

    @torch.no_grad()
    def ratios(self, configurations: torch.Tensor, base_indices: torch.Tensor | None=None) -> torch.Tensor:
        if base_indices is None:
            base_indices = torch.arange(self.batch_size, device=configurations.device)
        return torch.exp(self.log_psi(configurations, base_indices) - self.base_logpsi.index_select(0, base_indices))

class PreparedSequenceHierarchyEvaluator:
    """Reuse unchanged local factors of an unconstrained, real hierarchy.

    Coarse contexts still use the exact full Transformer forward. Only local
    factors from the first changed block onward need evaluating: teacher-forced
    causal contexts before that block cannot depend on the changed spins. This
    detached cache is rebuilt for every VMC sample batch, never across updates.
    """

    @torch.no_grad()
    def __init__(self, model, samples: torch.Tensor) -> None:
        local = model.subsampler
        if local is None:
            raise ValueError('Prepared sequence factors require an unconstrained real hierarchy.')
        self.model = model
        self.samples = samples
        self.batch_size = samples.size(0)
        self.patch_bits = model._tokens(samples)
        contexts = model._encode(self.patch_bits)
        rows = self.batch_size * model.num_tokens
        elements_per_row = local.channel_dim * max(local.num_tokens, local.token_hidden_dim)
        self.local_row_limit = max(1, min(65536, (1 << 26) // elements_per_row))
        lattice = self.patch_bits.reshape(rows, model.patch_size, model.patch_size)
        flat_contexts = contexts.reshape(rows, model.channel_dim)
        amplitudes = []
        for start in range(0, rows, self.local_row_limit):
            window = slice(start, start + self.local_row_limit)
            amplitude, _ = local._evaluate_tokens(local._tokens(lattice[window]), context=flat_contexts[window])
            amplitudes.append(amplitude)
        self.base_factors = torch.cat(amplitudes).reshape(self.batch_size, model.num_tokens)
        self.base_logpsi = torch.complex(self.base_factors.sum(-1), torch.zeros_like(self.base_factors[:, 0]))

    @torch.no_grad()
    def log_psi(self, configurations: torch.Tensor, base_indices=None) -> torch.Tensor:
        if base_indices is None:
            if configurations.size(0) != self.batch_size:
                raise ValueError('base_indices is required for a different batch size.')
            base_indices = torch.arange(self.batch_size, device=configurations.device)
        if base_indices.ndim != 1 or base_indices.numel() != configurations.size(0):
            raise ValueError('base_indices must contain one row index per configuration.')
        model, local = (self.model, self.model.subsampler)
        bits = model._tokens(configurations)
        changed = (bits != self.patch_bits.index_select(0, base_indices)).any(-1)
        affected = changed.to(torch.int32).cumsum(-1) > 0
        rows, positions = affected.nonzero(as_tuple=True)
        result = self.base_logpsi.real.index_select(0, base_indices).clone()
        if rows.numel() == 0:
            return torch.complex(result, torch.zeros_like(result))
        contexts = model._encode(bits)
        row_limit = self.local_row_limit
        for start in range(0, rows.numel(), row_limit):
            selected_rows = rows[start:start + row_limit]
            selected_positions = positions[start:start + row_limit]
            lattice = bits[selected_rows, selected_positions].reshape(-1, model.patch_size, model.patch_size)
            amplitude, _ = local._evaluate_tokens(local._tokens(lattice), context=contexts[selected_rows, selected_positions])
            original = self.base_factors[base_indices.index_select(0, selected_rows), selected_positions]
            result.index_add_(0, selected_rows, amplitude - original)
        return torch.complex(result, torch.zeros_like(result))

    @torch.no_grad()
    def ratios(self, configurations: torch.Tensor, base_indices=None) -> torch.Tensor:
        if base_indices is None:
            base_indices = torch.arange(self.batch_size, device=configurations.device)
        return torch.exp(self.log_psi(configurations, base_indices) - self.base_logpsi.index_select(0, base_indices))
__all__ = ['PreparedFullEvaluator', 'PreparedSequenceHierarchyEvaluator', 'PreparedHierarchicalEvaluator', 'PreparedMixerEvaluator', 'prepare_evaluator', 'select_connected_strategy']

def prepare_evaluator(model, samples, condition=None, strategy="auto", log_psi_func=None, base_evaluation=None):
    """Prepare detached local-energy ratios using exact autoregressive cache reuse."""
    if condition is not None:
        raise ValueError("Paper ground states do not use conditioning.")
    _validate_base_evaluation(model, samples, None, log_psi_func, base_evaluation)
    from tnqs.adapters.torch.sequence import SequenceBackboneWavefunction
    from tnqs.domain.specifications import BackboneKind
    function = log_psi_func or model.log_psi
    if (strategy == "auto" and samples.device.type == "mps"
            and isinstance(model, SequenceBackboneWavefunction)
            and model.kind is BackboneKind.TRANSFORMER
            and model.mode == "hierarchical" and model.subsampler is not None
            and model.subsampler.kind is BackboneKind.MIXER
            and getattr(function, "__func__", None) is SequenceBackboneWavefunction.log_psi):
        return PreparedSequenceHierarchyEvaluator(model, samples)
    selected = select_connected_strategy(model, strategy)
    if selected == "full_forward" or getattr(model, "supports_cached_suffix", True) is False or not all(
            hasattr(model, name) for name in ("_encode_step", "_patch_embedding", "_step_amplitude_logits")):
        return PreparedFullEvaluator(model, samples, None, log_psi_func, base_evaluation)
    if model.mode == "hierarchical":
        return PreparedHierarchicalEvaluator(model, samples, None, base_evaluation, log_psi_func)
    return PreparedMixerEvaluator(model, samples, None)


def select_connected_strategy(model, requested):
    """Select a mathematically identical connected-state evaluation path."""
    if requested not in ("auto", "full_forward", "cached_suffix"):
        raise ValueError(f"Unknown connected-state strategy: {requested!r}.")
    if requested != "auto":
        return requested
    if model.mode == "hierarchical":
        return "cached_suffix" if isinstance(model, MixerWavefunction) else "full_forward"
    if getattr(model, "supports_cached_suffix", True) is False:
        return "full_forward"
    if model.device.type in ("mps", "cuda") and model.num_tokens <= 128:
        return "full_forward"
    return "cached_suffix"

__all__ = ["PreparedFullEvaluator", "PreparedSequenceHierarchyEvaluator", "PreparedHierarchicalEvaluator", "PreparedMixerEvaluator", "prepare_evaluator", "select_connected_strategy"]

