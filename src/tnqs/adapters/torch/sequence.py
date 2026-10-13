"""Shared autoregressive sequence wavefunctions.

This module owns numerical sequence execution. Architecture specifications and campaign
policy remain outside the numerical models.

The implementation deliberately uses one teacher-forced representation for all three backbones:
the input at position ``p`` is the start token or token ``p-1`` and the output at ``p`` predicts
token ``p``.  That makes the RNN, Transformer, and Mixer paths interchangeable for logpsi,
sampling, and hierarchical composition.
"""
from __future__ import annotations
from tnqs.adapters.torch.categorical import sample_probabilities
from tnqs.adapters.torch.precision import autocast_features
from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from tnqs.adapters.torch.mix import MixerLayer
from tnqs.domain.specifications import BackboneKind, RecurrentCell

@dataclass(frozen=True)
class SequenceSamplingState:
    """Explicit state for a sampling trajectory.

    The state is intentionally a value object.  No prefix, hidden state, or random stream is
    stored on the module, so concurrent samplers and compiled evaluation cannot interfere.
    """
    tokens: Tensor
    position: int

class TransformerCache(tuple):
    """Projected key/value tensors for one causal Transformer block."""
    __slots__ = ()

    def __new__(cls, keys: Tensor, values: Tensor):
        return tuple.__new__(cls, (keys, values))

    @property
    def keys(self) -> Tensor:
        return self[0]

    @property
    def values(self) -> Tensor:
        return self[1]

class CausalTransformerBlock(nn.Module):
    """Dropout-free pre-norm Transformer block with a true incremental attention path."""

    def __init__(self, channel_dim: int, heads: int, feedforward_dim: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(channel_dim, heads, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(channel_dim)
        self.linear1 = nn.Linear(channel_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, channel_dim)
        self.norm2 = nn.LayerNorm(channel_dim)
        self.heads = heads
        self.head_dim = channel_dim // heads

    def _project(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        qkv = F.linear(values, self.attn.in_proj_weight, self.attn.in_proj_bias)
        return qkv.chunk(3, dim=-1)

    def _reshape_heads(self, values: Tensor) -> Tensor:
        batch, length, _ = values.shape
        return values.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def _attention(self, query: Tensor, keys: Tensor, values: Tensor, *, causal: bool) -> Tensor:
        q = self._reshape_heads(query)
        k = self._reshape_heads(keys)
        v = self._reshape_heads(values)
        attended = self._attend_heads(q, k, v, causal=causal)
        return self.attn.out_proj(attended.transpose(1, 2).reshape_as(query))

    def _attend_heads(self, q: Tensor, k: Tensor, v: Tensor, *, causal: bool) -> Tensor:
        if q.device.type == 'mps':
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)
        return self._attend_heads_reference(q, k, v, causal=causal)

    def _attend_heads_reference(self, q: Tensor, k: Tensor, v: Tensor, *, causal: bool) -> Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim ** 0.5
        if causal:
            length = scores.size(-1)
            mask = torch.ones(length, length, dtype=torch.bool, device=scores.device).triu(1)
            scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def forward(self, values: Tensor) -> Tensor:
        normalised = self.norm1(values)
        query, keys, projected_values = self._project(normalised)
        mixed = self._attention(query, keys, projected_values, causal=True)
        residual = values + mixed
        feedforward = self.linear2(F.gelu(self.linear1(self.norm2(residual))))
        return residual + feedforward

    def step(self, value: Tensor, cache: TransformerCache | None) -> tuple[Tensor, TransformerCache]:
        normalised = self.norm1(value)
        query, keys, projected_values = self._project(normalised)
        key_heads = self._reshape_heads(keys)
        value_heads = self._reshape_heads(projected_values)
        if cache is not None:
            key_heads = torch.cat((cache.keys, key_heads), dim=2)
            value_heads = torch.cat((cache.values, value_heads), dim=2)
        q_heads = self._reshape_heads(query)
        attended = self._attend_heads(q_heads, key_heads, value_heads, causal=False).transpose(1, 2).reshape_as(value)
        residual = value + self.attn.out_proj(attended)
        feedforward = self.linear2(F.gelu(self.linear1(self.norm2(residual))))
        return (residual + feedforward, TransformerCache(key_heads, value_heads))

def _sample_logits(logits: Tensor, generator: torch.Generator | None) -> Tensor:
    probabilities = F.softmax(logits, dim=-1)
    return sample_probabilities(probabilities, generator)

class SequenceBackboneWavefunction(nn.Module):
    """Autoregressive site, patch, or hierarchical wavefunction.

    ``kind`` selects an RNN, Transformer, or static Mixer backbone.
    Static Mixer backbones also serve as composable local subsamplers.
    """
    supports_cached_suffix = False
    supports_full_prefix_sampling = False

    def __init__(self, *, kind: BackboneKind | str, mode: str, num_layers: int, channel_dim: int, token_hidden_dim: int, Lx: int, Ly: int, patch_size: int=1, use_phase: bool=False, up_spin_target: int | None=None, recurrent_cell: RecurrentCell | str=RecurrentCell.GRU, transformer_heads: int=4, subsampler: 'SequenceBackboneWavefunction | None'=None, contextual: bool=False, device: torch.device | str='cpu', sampling_generator: torch.Generator | None=None) -> None:
        if use_phase:
            raise ValueError('The paper ansatz requires use_phase=False.')
        if up_spin_target is not None:
            raise ValueError('The paper ansatz requires the full Hilbert space.')
        super().__init__()
        self.kind = BackboneKind(kind)
        self.supports_compile = self.kind is not BackboneKind.RNN
        self.mode = str(mode)
        self.num_layers = int(num_layers)
        self.channel_dim = int(channel_dim)
        self.token_hidden_dim = int(token_hidden_dim)
        self.Lx = int(Lx)
        self.Ly = int(Ly)
        self.patch_size = int(patch_size)
        self.L = self.Lx * self.Ly
        self.use_phase = bool(False)
        self.up_spin_target = None
        self.recurrent_cell = RecurrentCell(recurrent_cell)
        self.transformer_heads = int(transformer_heads)
        if self.mode not in ('site', 'patch', 'hierarchical'):
            raise ValueError(f'Unknown sequence wavefunction mode: {self.mode!r}.')
        if self.mode == 'site' and self.patch_size != 1:
            raise ValueError('site mode requires patch_size=1.')
        if self.Lx % self.patch_size or self.Ly % self.patch_size:
            raise ValueError('patch_size must divide both lattice dimensions.')
        if self.mode == 'hierarchical' and subsampler is None:
            raise ValueError('hierarchical mode requires a subsampler.')
        self.num_tokens = self.L if self.mode == 'site' else self.Lx // self.patch_size * (self.Ly // self.patch_size)
        self.sites_per_patch = self.patch_size * self.patch_size
        self.K_patch = 1 << self.sites_per_patch
        self.vocab_size = 2 if self.mode == 'site' else self.K_patch
        self.token_embedding = None if self.mode == 'hierarchical' else nn.Embedding(self.vocab_size, self.channel_dim)
        self.patch_projection = nn.Linear(self.sites_per_patch, self.channel_dim) if self.mode == 'hierarchical' else None
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_tokens, self.channel_dim) * 0.02)
        self.start_token = nn.Parameter(torch.randn(1, 1, self.channel_dim) * 0.02)
        if self.kind is BackboneKind.RNN:
            self.input_projection = nn.Linear(self.channel_dim, self.token_hidden_dim)
            recurrent = nn.GRU if self.recurrent_cell is RecurrentCell.GRU else nn.LSTM
            self.backbone = recurrent(input_size=self.token_hidden_dim, hidden_size=self.channel_dim, num_layers=self.num_layers, batch_first=True)
        elif self.kind is BackboneKind.TRANSFORMER:
            if self.channel_dim % self.transformer_heads:
                raise ValueError('channel_dim must be divisible by transformer_heads.')
            self.backbone = nn.ModuleList([CausalTransformerBlock(self.channel_dim, self.transformer_heads, self.token_hidden_dim) for _ in range(self.num_layers)])
        elif self.kind in (BackboneKind.MIXER,):
            self.backbone = nn.ModuleList([MixerLayer(num_tokens=self.num_tokens, channel_dim=self.channel_dim, token_hidden_dim=self.token_hidden_dim) for _ in range(self.num_layers)])
        else:
            raise ValueError(f'Unsupported sequence backbone: {self.kind.value!r}.')
        self.amplitude_head = None if self.mode == 'hierarchical' else nn.Linear(self.channel_dim, self.vocab_size)
        self.phase_head = None
        self.subsampler = subsampler
        self.supports_compile = bool(self.kind is not BackboneKind.RNN and (subsampler is None or getattr(subsampler, 'supports_compile', True)))
        self.context_projection = nn.Linear(self.channel_dim, self.channel_dim) if contextual else None
        # Retain the historical token metadata for checkpoint tensor compatibility.
        self.register_buffer('patch_magnetisation', torch.tensor([] if self.mode == 'hierarchical' else [value.bit_count() for value in range(self.K_patch)], dtype=torch.long))
        self.register_buffer('patch_rows', self._patch_coordinates()[0])
        self.register_buffer('patch_cols', self._patch_coordinates()[1])
        self._sampling_generator = sampling_generator
        self.to(device)

    def set_sampling_generator(self, generator: torch.Generator | None) -> None:
        self._sampling_generator = generator
        if self.subsampler is not None:
            self.subsampler.set_sampling_generator(generator)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _patch_coordinates(self) -> tuple[Tensor, Tensor]:
        rows: list[list[int]] = []
        cols: list[list[int]] = []
        for block_row in range(self.Lx // self.patch_size):
            for block_col in range(self.Ly // self.patch_size):
                row_values: list[int] = []
                col_values: list[int] = []
                for local_row in range(self.patch_size):
                    for local_col in range(self.patch_size):
                        row_values.append(block_row * self.patch_size + local_row)
                        col_values.append(block_col * self.patch_size + local_col)
                rows.append(row_values)
                cols.append(col_values)
        return (torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long))

    def to_patch_indices(self, sigma: Tensor) -> Tensor:
        bits = sigma[:, self.patch_rows, self.patch_cols].long()
        positions = torch.arange(self.sites_per_patch, device=sigma.device)
        return (bits << positions).sum(-1)

    def from_patch_indices(self, values: Tensor) -> Tensor:
        bits = self._ints_to_bits(values)
        result = torch.empty(values.size(0), self.Lx, self.Ly, dtype=torch.long, device=values.device)
        result[:, self.patch_rows, self.patch_cols] = bits
        return result

    def _ints_to_bits(self, values: Tensor) -> Tensor:
        positions = torch.arange(self.sites_per_patch, device=values.device)
        return (values.unsqueeze(-1) >> positions & 1).long()

    def _tokens(self, sigma: Tensor) -> Tensor:
        if self.mode == 'site':
            return sigma.reshape(sigma.size(0), self.L).long()
        if self.mode == 'hierarchical':
            return sigma[:, self.patch_rows, self.patch_cols].long()
        return self.to_patch_indices(sigma)

    def _embed_tokens(self, tokens: Tensor) -> Tensor:
        if self.mode == 'hierarchical':
            assert self.patch_projection is not None
            if tokens.shape[-1] != self.sites_per_patch:
                tokens = self._ints_to_bits(tokens)
            return self.patch_projection(tokens.to(self.patch_projection.weight.dtype))
        assert self.token_embedding is not None
        return self.token_embedding(tokens)

    def _shifted_embeddings(self, tokens: Tensor) -> Tensor:
        embedded = self._embed_tokens(tokens)
        start = self.start_token.expand(tokens.size(0), -1, -1)
        return torch.cat((start, embedded[:, :-1]), dim=1) + self.position_embedding

    @autocast_features
    def _project_context(self, context: Tensor) -> Tensor:
        """Use the same feature precision for cached and full local decoding."""
        if self.context_projection is None:
            raise ValueError('This subsampler does not accept hierarchical context.')
        return self.context_projection(context).unsqueeze(1)

    @autocast_features
    def _encode(self, tokens: Tensor, condition: None=None, context: Tensor | None=None) -> Tensor:
        hidden = self._shifted_embeddings(tokens)
        if context is not None:
            hidden = hidden + self._project_context(context)
        if self.kind is BackboneKind.RNN:
            hidden = self.input_projection(hidden)
            hidden, _ = self.backbone(hidden)
        elif self.kind is BackboneKind.TRANSFORMER:
            for layer in self.backbone:
                hidden = layer(hidden)
        else:
            for layer in self.backbone:
                hidden = layer(hidden, None)
        return hidden

    @autocast_features
    def _encode_step(self, previous_token: Tensor | None, position: int, state: Tensor | list[object | None] | None, batch_size: int, condition: None=None, context: Tensor | None=None, projected_context: Tensor | None=None) -> tuple[Tensor, Tensor | list[object]]:
        """Advance one autoregressive position while retaining backbone state."""
        if position >= self.num_tokens:
            raise ValueError('Cannot decode beyond the configured sequence length.')
        if context is not None and projected_context is not None:
            raise ValueError('Supply raw or projected context, not both.')
        if previous_token is None:
            hidden = self.start_token.expand(batch_size, -1, -1)
        else:
            hidden = self._embed_tokens(previous_token).unsqueeze(1)
        hidden = hidden + self.position_embedding[:, position:position + 1]
        if projected_context is not None:
            if projected_context.shape != (batch_size, 1, self.channel_dim):
                raise ValueError('Projected context must have shape [batch, 1, channel_dim].')
            hidden = hidden + projected_context
        elif context is not None:
            hidden = hidden + self._project_context(context)
        if self.kind is BackboneKind.RNN:
            hidden = self.input_projection(hidden)
            hidden, new_state = self.backbone(hidden, state)
        elif self.kind is BackboneKind.TRANSFORMER:
            caches = [] if state is None else list(state)
            if len(caches) != len(self.backbone):
                caches = [None] * len(self.backbone)
            new_caches: list[TransformerCache] = []
            for layer, cache in zip(self.backbone, caches, strict=True):
                hidden, new_cache = layer.step(hidden, cache)
                new_caches.append(new_cache)
            new_state = new_caches
        elif self.kind in (BackboneKind.MIXER,):
            caches = [] if state is None else list(state)
            if len(caches) != len(self.backbone):
                caches = [None] * len(self.backbone)
            new_caches = []
            for layer, cache in zip(self.backbone, caches, strict=True):
                hidden, new_cache = layer.forward_step(hidden, None, cache)
                new_caches.append(new_cache)
            new_state = new_caches
        else:
            raise ValueError(f'Incremental decoding is unavailable for {self.kind.value!r}.')
        return (hidden, new_state)

    def _head_values(self, hidden: Tensor, condition: None=None) -> tuple[Tensor, Tensor | None]:
        if self.amplitude_head is None:
            raise RuntimeError('A hierarchical coarse backbone has no probability heads.')
        amplitude = self.amplitude_head(hidden)
        phase = None
        return (amplitude, phase)

    def _evaluate_tokens(self, tokens: Tensor, condition: None=None, *, context: Tensor | None=None) -> tuple[Tensor, Tensor]:
        hidden = self._encode(tokens, None, context)
        amplitude, _ = self._head_values(hidden, None)
        selected = tokens.unsqueeze(-1)
        log_amplitude = 0.5 * F.log_softmax(amplitude, -1).gather(-1, selected).squeeze(-1).sum(-1)
        return (log_amplitude, torch.zeros_like(log_amplitude))

    def _hierarchical_log_psi(self, sigma: Tensor, condition: None) -> tuple[Tensor, Tensor]:
        assert self.subsampler is not None
        patch_bits = self._tokens(sigma)
        coarse_hidden = self._encode(patch_bits, None)
        log_amplitude = torch.zeros(sigma.size(0), device=sigma.device, dtype=coarse_hidden.dtype)
        phase_total = torch.zeros_like(log_amplitude)
        for position in range(self.num_tokens):
            local_lattice = patch_bits[:, position].reshape(sigma.size(0), self.patch_size, self.patch_size)
            local_tokens = self.subsampler._tokens(local_lattice)
            local_amp, local_phase = self.subsampler._evaluate_tokens(local_tokens, None, context=coarse_hidden[:, position])
            log_amplitude = log_amplitude + local_amp
            phase_total = phase_total + local_phase
        return (log_amplitude, phase_total)

    def log_psi_reference(self, sigma: Tensor, condition: None=None) -> Tensor:
        if self.mode == 'hierarchical':
            log_amplitude, phase = self._hierarchical_log_psi(sigma, None)
        else:
            log_amplitude, phase = self._evaluate_tokens(self._tokens(sigma), None)
        return torch.complex(log_amplitude, phase)

    def log_psi_step(self, sigma: Tensor, condition: None=None) -> Tensor:
        return self.log_psi_reference(sigma, None)

    def log_psi(self, sigma: Tensor, condition: None=None) -> Tensor:
        if sigma.device.type == 'mps' and self.mode == 'hierarchical' and (self.subsampler is not None):
            return self._hierarchical_log_psi_batched(sigma)
        return self.log_psi_reference(sigma, None)

    def _hierarchical_log_psi_batched(self, sigma: Tensor, *, max_local_rows: int=65536) -> Tensor:
        """Batch independent local factors after computing their causal contexts.

        This is teacher-forced evaluation, not sampling: block positions can be
        folded into the batch without changing the autoregressive factorisation.
        Bound the largest local activation as connected-state batches can be much larger
        than the training sample batch.
        """
        assert self.subsampler is not None
        patch_bits = self._tokens(sigma)
        contexts = self._encode(patch_bits)
        batch = sigma.size(0)
        local = self.subsampler
        elements_per_row = local.channel_dim * max(local.num_tokens, local.token_hidden_dim)
        if max_local_rows <= 0:
            raise ValueError('max_local_rows must be positive.')
        local_row_limit = max(1, min(max_local_rows, (1 << 26) // elements_per_row))
        blocks_per_call = max(1, min(self.num_tokens, local_row_limit // max(1, batch)))
        amplitudes, phases = ([], [])
        for start in range(0, self.num_tokens, blocks_per_call):
            stop = min(self.num_tokens, start + blocks_per_call)
            count = stop - start
            local_lattice = patch_bits[:, start:stop].reshape(batch * count, self.patch_size, self.patch_size)
            amp, phase = local._evaluate_tokens(local._tokens(local_lattice), context=contexts[:, start:stop].reshape(batch * count, self.channel_dim))
            amplitudes.append(amp.reshape(batch, count).sum(-1))
            phases.append(phase.reshape(batch, count).sum(-1))
        return torch.complex(torch.stack(amplitudes, dim=0).sum(0), torch.stack(phases, dim=0).sum(0))

    def _sample_tokens(self, num_samples: int, condition: None, *, context: Tensor | None=None, generator: torch.Generator | None=None) -> Tensor:
        generator = self._sampling_generator if generator is None else generator
        tokens = torch.zeros(num_samples, self.num_tokens, dtype=torch.long, device=self.device)
        if self.kind in (BackboneKind.RNN, BackboneKind.TRANSFORMER, BackboneKind.MIXER):
            projected_context = None
            if context is not None:
                projected_context = self._project_context(context)
            state: Tensor | list[object | None] | None = None
            previous: Tensor | None = None
            for position in range(self.num_tokens):
                hidden, state = self._encode_step(previous, position, state, num_samples, None, projected_context=projected_context)
                amplitude, _ = self._head_values(hidden, None)
                logits = amplitude[:, 0]
                selected = _sample_logits(logits, generator)
                tokens[:, position] = selected
                previous = selected
            return tokens
        raise ValueError(f'Sampling is unavailable for {self.kind.value!r}.')

    @torch.no_grad()
    def sample(self, condition: None, num_samples: int, *, strategy: str='cached', generator: torch.Generator | None=None) -> Tensor:
        if strategy != 'cached':
            raise ValueError("Sequence backbones support only strategy='cached'; full_prefix sampling is unsupported.")
        if self.mode != 'hierarchical':
            tokens = self._sample_tokens(num_samples, None, generator=generator)
            if self.mode == 'site':
                return tokens.reshape(num_samples, self.Lx, self.Ly)
            return self.from_patch_indices(tokens)
        assert self.subsampler is not None
        patch_bits = torch.zeros(num_samples, self.num_tokens, self.sites_per_patch, dtype=torch.long, device=self.device)
        lattice = torch.zeros(num_samples, self.Lx, self.Ly, dtype=torch.long, device=self.device)
        state: Tensor | list[TransformerCache | None] | None = None
        previous: Tensor | None = None
        for position in range(self.num_tokens):
            if self.kind in (BackboneKind.RNN, BackboneKind.TRANSFORMER):
                context, state = self._encode_step(previous, position, state, num_samples, None)
                context = context[:, 0]
            else:
                context = self._encode(patch_bits, None)[:, position]
            local = self.subsampler._sample_tokens(num_samples, None, context=context, generator=generator)
            if self.subsampler.mode == 'site':
                local_lattice = local.reshape(num_samples, self.patch_size, self.patch_size)
            else:
                local_lattice = self.subsampler.from_patch_indices(local)
            flat = local_lattice.reshape(num_samples, self.sites_per_patch)
            patch_bits[:, position] = flat
            lattice[:, self.patch_rows[position], self.patch_cols[position]] = flat
            previous = flat
        return lattice

class RNNWavefunction(SequenceBackboneWavefunction):
    """Sequence wavefunction with a GRU/LSTM backbone; GRU is the default."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs['kind'] = BackboneKind.RNN
        super().__init__(**kwargs)

class TransformerWavefunction(SequenceBackboneWavefunction):
    """Sequence wavefunction with a causal Transformer encoder."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs['kind'] = BackboneKind.TRANSFORMER
        super().__init__(**kwargs)

class MixerSubsampler(SequenceBackboneWavefunction):
    """Compact Mixer-compatible subsampler used by hierarchical compositions."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs['kind'] = BackboneKind.MIXER
        super().__init__(**kwargs)
__all__ = ['MixerSubsampler', 'RNNWavefunction', 'SequenceBackboneWavefunction', 'SequenceSamplingState', 'TransformerWavefunction']
