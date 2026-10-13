import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple
from jaxtyping import Complex, Float, Int
from functools import cached_property
from tnqs.adapters.torch.categorical import sample_probabilities
from tnqs.adapters.torch.precision import autocast_features

class WavefunctionOutputs(NamedTuple):
    """Per-patch amplitude logits; phase is always absent for the paper ansatz."""
    amplitude_logits: Float[torch.Tensor, 'batch_size num_tokens K_patch']
    phase: Float[torch.Tensor, 'batch_size num_tokens K_patch'] | None

class MixerLayerCache(NamedTuple):
    """Prefix state for one causal Mixer layer.

    Hidden units are finalized once, when their causal position is reached. Future
    groups stay zero: evaluating them before their inputs exist only repeats work.
    Caches are local to one parameter snapshot and may not survive an optimizer update.
    """
    norm: Float[torch.Tensor, 'batch_size channel_dim num_tokens']
    hidden: Float[torch.Tensor, 'batch_size channel_dim token_hidden_dim']
    length: int

class MixerBaseEvaluation(NamedTuple):
    """Differentiable base values with optional teacher-forced hierarchical factors."""
    logpsi: torch.Tensor
    contexts: torch.Tensor | None = None
    contributions: torch.Tensor | None = None
    provenance: tuple | None = None

def base_evaluation_provenance(model, samples):
    """Identify the exact model, sample view, and ordinary mutation versions."""
    if torch.compiler.is_compiling():
        return None
    if samples.is_inference() or any((p.is_inference() for p in model.parameters())):
        return None
    return (id(model), tuple(((id(p), p._version) for p in model.parameters())), samples.data_ptr(), samples._version, tuple(samples.shape), tuple(samples.stride()), samples.dtype, samples.device)

def position_labels(num_units: int, num_tokens: int) -> Int[torch.Tensor, 'num_units']:
    """Tag each unit of a token-mixing layer with the patch position it is allowed to see up to.

    Cycling with modulo spreads the units evenly over positions, so a hidden layer wider than
    num_tokens simply gives each position several units rather than extending anyone's reach.
    """
    return torch.arange(num_units) % num_tokens

class MLinear(nn.Linear):

    def __init__(self, in_labels: Int[torch.Tensor, 'in_features'], out_labels: Int[torch.Tensor, 'out_features'], bias: bool=True) -> None:
        """Linear layer masked by MADE-style connectivity: an output unit may read an input unit
        only if the input's position label is <= its own.

        Labelling rather than a plain torch.tril is what lets the token mixer have a hidden width
        independent of the number of patches. A tril only expresses an ordering when its rows and
        columns both index patches, so on a rectangular weight it would mask arbitrarily.

        The comparison is inclusive, so a unit may read its own position. That is safe because
        MixerWavefunction shifts the patch embeddings one position right beforehand, leaving
        position p holding patch p-1. It is also what keeps two stacked layers composing to the
        window 0..p-1; an exclusive comparison would retreat one patch per layer and drop the
        nearest neighbour.
        """
        super(MLinear, self).__init__(in_features=in_labels.numel(), out_features=out_labels.numel(), bias=bias)
        mask = (out_labels[:, None] >= in_labels[None, :]).float()
        self.register_buffer('mask', mask)

    def forward(self, input: Float[torch.Tensor, 'batch_size in_features']) -> Float[torch.Tensor, 'batch_size out_features']:
        masked_weight = self.weight * self.mask
        return F.linear(input, masked_weight, self.bias)

class MixerLayer(nn.Module):

    def __init__(self, *, num_tokens: int, channel_dim: int, token_hidden_dim: int) -> None:
        """A Mixer block whose token mixing is autoregressive over patches.

        num_tokens is the sequence length, fixed by the lattice and patch size. token_hidden_dim
        is the free interior width of the token-mixing MLP, and must be at least num_tokens: a
        narrower hidden layer has no unit labelled with the later positions, so the last patches
        would lose sight of their immediate predecessors.
        """
        super(MixerLayer, self).__init__()
        self.num_tokens = num_tokens
        self.channel_dim = channel_dim
        assert token_hidden_dim >= num_tokens, f'token_hidden_dim must be at least the number of patches, got token_hidden_dim={token_hidden_dim} and num_tokens={num_tokens}.'
        token_labels = position_labels(num_tokens, num_tokens)
        hidden_labels = position_labels(token_hidden_dim, num_tokens)
        self.token_layer_norm = nn.LayerNorm(channel_dim)
        self.token_mixer = nn.Sequential(MLinear(token_labels, hidden_labels), nn.GELU(), MLinear(hidden_labels, token_labels))
        self.register_buffer('hidden_labels', hidden_labels)
        self.channel_layer_norm = nn.LayerNorm(channel_dim)
        self.channel_mixer = nn.Sequential(nn.Linear(channel_dim, channel_dim), nn.GELU(), nn.Linear(channel_dim, channel_dim))

    def forward(self, x: Float[torch.Tensor, 'batch_size num_tokens channel_dim'], condition: None=None) -> Float[torch.Tensor, 'batch_size num_tokens channel_dim']:
        return self.forward_portable(x, None)

    def forward_portable(self, x: Float[torch.Tensor, 'batch_size num_tokens channel_dim'], condition: None=None) -> Float[torch.Tensor, 'batch_size num_tokens channel_dim']:
        """Portable PyTorch token and channel mixing."""
        y = self.token_layer_norm(x)
        y = y.transpose(1, 2)
        y = self.token_mixer(y)
        y = y.transpose(1, 2) + x
        z = self.channel_layer_norm(y)
        z = self.channel_mixer(z)
        return z + y

    def forward_step(self, x_new: Float[torch.Tensor, 'batch_size 1 channel_dim'], condition: None=None, cache: MixerLayerCache | None=None) -> tuple[Float[torch.Tensor, 'batch_size 1 channel_dim'], MixerLayerCache]:
        return self.forward_step_portable(x_new, None, cache)

    def forward_step_portable(self, x_new: Float[torch.Tensor, 'batch_size 1 channel_dim'], condition: None=None, cache: MixerLayerCache | None=None) -> tuple[Float[torch.Tensor, 'batch_size 1 channel_dim'], MixerLayerCache]:
        """Process one causal token while reusing the prefix token-mixer state.

        This is the step equivalent of :meth:`forward`. It intentionally keeps the cache as an
        explicit value rather than storing it on the module: a cache is valid only for one batch,
        one parameter snapshot.
        """
        if x_new.ndim != 3 or x_new.size(1) != 1:
            raise ValueError(f'MixerLayer.forward_step expects [batch_size, 1, channel_dim], got {tuple(x_new.shape)}')
        first, activation, second = self.token_mixer
        position = 0 if cache is None else cache.length
        if position >= self.num_tokens:
            raise ValueError(f'Cannot append position {position} to a {self.num_tokens}-token Mixer layer.')
        normalized = self.token_layer_norm(x_new).transpose(1, 2)
        if cache is None:
            norm_full = torch.zeros(x_new.size(0), x_new.size(2), self.num_tokens, dtype=x_new.dtype, device=x_new.device)
            hidden_full = x_new.new_zeros(x_new.size(0), x_new.size(2), first.out_features)
        elif torch.is_grad_enabled():
            norm_full = cache.norm.clone()
            hidden_full = cache.hidden.clone()
        else:
            norm_full = cache.norm
            hidden_full = cache.hidden
        norm_full[..., position:position + 1].copy_(normalized)
        hidden_full[..., position::self.num_tokens] = activation(F.linear(norm_full[..., :position + 1], first.weight[position::self.num_tokens, :position + 1], None if first.bias is None else first.bias[position::self.num_tokens]))
        token_mix = F.linear(hidden_full, second.weight[position:position + 1], None if second.bias is None else second.bias[position:position + 1]).transpose(1, 2)
        y = token_mix + x_new
        z = self.channel_layer_norm(y)
        out = self.channel_mixer(z) + y
        return (out, MixerLayerCache(norm_full, hidden_full, position + 1))

    def slice_cache(self, cache: MixerLayerCache, prefix_length: int) -> MixerLayerCache:
        """Turn a full cache into the equivalent cache for a shorter prefix."""
        if not 0 < prefix_length <= cache.length:
            raise ValueError(f'prefix_length must lie in [1, {cache.length}], got {prefix_length}.')
        norm_full = cache.norm.clone()
        hidden_full = cache.hidden.clone()
        norm_full[..., prefix_length:] = 0
        hidden_full.masked_fill_(self.hidden_labels >= prefix_length, 0)
        return MixerLayerCache(norm_full, hidden_full, prefix_length)

class HierarchicalPatchDecoder(nn.Module):
    """Small autoregressive decoder for the sites inside one patch."""

    def __init__(self, *, channel_dim: int, sites_per_patch: int, use_phase: bool) -> None:
        if use_phase:
            raise ValueError('The paper ansatz requires use_phase=False.')
        super().__init__()
        self.channel_dim = channel_dim
        self.sites_per_patch = sites_per_patch
        self.previous_bit = nn.Embedding(2, channel_dim)
        self.start_site = nn.Parameter(torch.randn(channel_dim) * 0.02)
        self.position = nn.Parameter(torch.randn(sites_per_patch, channel_dim) * 0.02)
        self.mlp = nn.Sequential(nn.LayerNorm(channel_dim), nn.Linear(channel_dim, channel_dim), nn.GELU(), nn.Linear(channel_dim, 2))
        self.phase_head = None

    def forward_step(self, context: Float[torch.Tensor, 'batch_size channel_dim'], previous_bit: Int[torch.Tensor, 'batch_size'] | None, position: int) -> tuple[Float[torch.Tensor, 'batch_size 2'], Float[torch.Tensor, 'batch_size'] | None]:
        token = self.start_site.unsqueeze(0).expand(context.size(0), -1) if previous_bit is None else self.previous_bit(previous_bit)
        hidden = context + token + self.position[position]
        logits = self.mlp(hidden)
        phase = None
        return (logits, phase)

class MixerWavefunction(nn.Module):

    def __init__(self, *, num_layers: int, channel_dim: int, patch_size: int, Lx: int, Ly: int, token_hidden_dim: int | None=None, up_spin_target: int | None=None, use_phase: bool=False, mode: str='patch', subsampler: nn.Module | None=None, device: torch.device=torch.device('cpu')) -> None:
        """Autoregressive Mixer over patches of an Lx by Ly lattice.

        The sequence length is fixed by the lattice and patch size, so it is derived rather than
        passed in. token_hidden_dim is the free interior width of the token-mixing MLP and
        defaults to the sequence length, which reproduces a square masked mixer.

        The paper uses real, non-negative amplitudes over the full Hilbert space.
        The constructor rejects phase heads and fixed-magnetisation restrictions.
        """
        if use_phase:
            raise ValueError('The paper ansatz requires use_phase=False.')
        if up_spin_target is not None:
            raise ValueError('The paper ansatz requires the full Hilbert space.')
        super(MixerWavefunction, self).__init__()
        assert Lx % patch_size == 0 and Ly % patch_size == 0, f'Lx and Ly must be divisible by patch_size, got Lx={Lx}, Ly={Ly}, patch_size={patch_size}.'
        self.patch_size = patch_size
        self.Lx, self.Ly = (Lx, Ly)
        self.L = self.Lx * self.Ly
        if mode not in ('site', 'patch', 'hierarchical'):
            raise ValueError(f'Unknown autoregressive mode: {mode!r}.')
        self.mode = mode
        self.use_phase = False
        if subsampler is not None and mode != 'hierarchical':
            raise ValueError('A subsampler is valid only in hierarchical mode.')
        self.subsampler = subsampler
        self._sampling_generator: torch.Generator | None = None
        self.channel_dim = channel_dim
        self.K_patch = 1 << self.patch_size * self.patch_size
        self.num_tokens = self.Lx // self.patch_size * (self.Ly // self.patch_size)
        self.token_hidden_dim = self.num_tokens if token_hidden_dim is None else token_hidden_dim
        self.layers = nn.ModuleList([MixerLayer(num_tokens=self.num_tokens, channel_dim=channel_dim, token_hidden_dim=self.token_hidden_dim) for _ in range(num_layers)])
        self.amplitude_head = nn.Linear(channel_dim, self.K_patch) if mode in ('site', 'patch') else None
        self.phase_head = None
        self.hierarchical_decoder = HierarchicalPatchDecoder(channel_dim=channel_dim, sites_per_patch=self.patch_size * self.patch_size, use_phase=False) if mode == 'hierarchical' and subsampler is None else None
        self.start_token = nn.Parameter(torch.randn(1, 1, channel_dim) * 0.02)
        self.patch_projector = nn.Conv2d(in_channels=1, out_channels=channel_dim, kernel_size=patch_size, stride=patch_size)
        self.up_spin_target = None
        # Retain the historical token metadata for checkpoint tensor compatibility.
        self.register_buffer('patch_magnetisation', torch.tensor([] if mode == 'hierarchical' else [bin(value).count('1') for value in range(self.K_patch)], dtype=torch.long))
        rows = torch.tensor([[r for r, _ in self.block_lookup[k]] for k in range(self.num_tokens)])
        cols = torch.tensor([[c for _, c in self.block_lookup[k]] for k in range(self.num_tokens)])
        self.register_buffer('patch_rows', rows)
        self.register_buffer('patch_cols', cols)
        self.to(device)

    def set_sampling_generator(self, generator: torch.Generator | None) -> None:
        """Attach the deterministic sampling stream owned by the runtime composition root."""
        self._sampling_generator = generator
        if self.subsampler is not None:
            self.subsampler.set_sampling_generator(generator)

    @property
    def device(self) -> torch.device:
        """Read the device off a live parameter, so it stays correct after .to(...)."""
        return next(self.parameters()).device

    @cached_property
    def block_lookup(self) -> dict:
        """
        Computes the row-major lookup dictionary only once per instance life cycle.
        """
        indices = torch.arange(self.Lx * self.Ly)
        y_coords = indices % self.Ly
        x_coords = indices // self.Ly
        coords = torch.stack([x_coords, y_coords], dim=-1)
        grid = coords.view(self.Lx, self.Ly, 2)
        num_blocks_x = self.Lx // self.patch_size
        num_blocks_y = self.Ly // self.patch_size
        blocks = grid.view(num_blocks_x, self.patch_size, num_blocks_y, self.patch_size, 2)
        blocks = blocks.permute(0, 2, 1, 3, 4)
        flat_blocks = blocks.reshape(num_blocks_x * num_blocks_y, self.patch_size * self.patch_size, 2)
        return {k: tuple((tuple(coord.tolist()) for coord in flat_blocks[k])) for k in range(flat_blocks.size(0))}

    def forward(self, x: Int[torch.Tensor, 'batch_size Lx Ly'], condition: None=None) -> WavefunctionOutputs:
        """Run the causal trunk and amplitude head over configurations."""
        if self.amplitude_head is None:
            raise RuntimeError('forward() logits are available only in patch mode; use log_psi().')
        hidden = self._encode(x, None)
        amplitude_logits = self.amplitude_head(hidden)
        phase = None
        return WavefunctionOutputs(amplitude_logits=amplitude_logits, phase=phase)

    @autocast_features
    def _encode(self, x: Int[torch.Tensor, 'batch_size Lx Ly'], condition: None=None) -> Float[torch.Tensor, 'batch_size num_tokens channel_dim']:
        """The shared causal trunk, before either head."""
        x = self._prep_patches(x)
        x = self._shift_right(x)
        for layer in self.layers:
            x = layer(x, None)
        return x

    @autocast_features
    def _encode_step(self, token: Float[torch.Tensor, 'batch_size 1 channel_dim'], condition: None=None, caches: list[MixerLayerCache | None] | None=None) -> tuple[Float[torch.Tensor, 'batch_size 1 channel_dim'], list[MixerLayerCache]]:
        """Advance the causal Mixer stack by one token."""
        if caches is None:
            caches = [None] * len(self.layers)
        if len(caches) != len(self.layers):
            raise ValueError(f'Expected {len(self.layers)} layer caches, got {len(caches)}.')
        hidden = token
        new_caches = []
        for layer, cache in zip(self.layers, caches, strict=True):
            hidden, new_cache = layer.forward_step(hidden, None, cache)
            new_caches.append(new_cache)
        return (hidden, new_caches)

    def _slice_caches(self, caches: list[MixerLayerCache], prefix_length: int) -> list[MixerLayerCache]:
        """Slice every layer cache to the same causal prefix."""
        return [layer.slice_cache(cache, prefix_length) for layer, cache in zip(self.layers, caches, strict=True)]

    @autocast_features
    def _patch_embedding(self, patch_values: Int[torch.Tensor, 'batch_size'], embedding_table: torch.Tensor | None=None) -> Float[torch.Tensor, 'batch_size 1 channel_dim']:
        """Embed only the requested patch tokens.

        The old implementation materialised all ``2**S`` differentiable embeddings before
        gathering a handful of them.  That is catastrophically expensive for large patches.
        Callers may still provide a deliberately bounded table for a measured reuse-heavy path;
        the default is always the workload-proportional projection below.
        """
        if embedding_table is not None:
            return embedding_table[patch_values].unsqueeze(1)
        bits = self._ints_to_bits(patch_values)
        weight = self.patch_projector.weight[:, 0].flatten(1)
        return F.linear(bits.to(weight), weight, self.patch_projector.bias).unsqueeze(1)

    @autocast_features
    def _patch_embedding_table(self) -> Float[torch.Tensor, 'K_patch channel_dim']:
        """Build a bounded lookup table for explicit, measured reuse-heavy callers."""
        max_elements = 1 << 20
        if self.K_patch * self.channel_dim > max_elements:
            raise RuntimeError(f'Refusing to materialize the patch embedding table: K_patch*channel_dim={self.K_patch * self.channel_dim:,} exceeds the safety cap of {max_elements:,}. Use direct patch projection.')
        values = torch.arange(self.K_patch, device=self.patch_projector.weight.device)
        bits = self._ints_to_bits(values).to(self.patch_projector.weight)
        weight = self.patch_projector.weight[:, 0].flatten(1)
        return F.linear(bits, weight, self.patch_projector.bias)

    @autocast_features
    def _encode_patch_indices(self, patch_values: Int[torch.Tensor, 'batch_size num_tokens'], condition: None=None, embedding_table: torch.Tensor | None=None) -> Float[torch.Tensor, 'batch_size num_tokens channel_dim']:
        """Run the causal trunk directly from compact integer patch tokens."""
        if patch_values.shape[1] != self.num_tokens:
            raise ValueError(f'Expected {self.num_tokens} patch tokens, got {patch_values.shape[1]}.')
        if embedding_table is None:
            bits = self._ints_to_bits(patch_values)
            weight = self.patch_projector.weight[:, 0].flatten(1)
            hidden = F.linear(bits.to(weight), weight, self.patch_projector.bias)
        else:
            hidden = embedding_table[patch_values]
        hidden = self._shift_right(hidden)
        for layer in self.layers:
            hidden = layer(hidden, None)
        return hidden

    def _step_amplitude_logits(self, hidden: Float[torch.Tensor, 'batch_size 1 channel_dim'], condition: None=None) -> Float[torch.Tensor, 'batch_size K_patch']:
        """Apply only the amplitude head, as required by Born sampling."""
        assert self.amplitude_head is not None
        logits = self.amplitude_head(hidden.squeeze(1))
        return logits

    def _step_outputs(self, hidden: Float[torch.Tensor, 'batch_size 1 channel_dim'], condition: None=None) -> tuple[Float[torch.Tensor, 'batch_size K_patch'], Float[torch.Tensor, 'batch_size K_patch'] | None]:
        """Apply the amplitude head at one autoregressive position."""
        assert self.amplitude_head is not None
        amplitude_logits = self._step_amplitude_logits(hidden, None)
        phase = None
        return (amplitude_logits, phase)

    def log_psi_reference(self, sigma: Int[torch.Tensor, 'batch_size Lx Ly'], condition: None=None) -> Complex[torch.Tensor, 'batch_size']:
        """Evaluate the square root of normalized conditional probabilities.

        Each position reads only earlier patches; gathering the observed outcome
        supplies its own patch value. The half converts log probability to log
        amplitude, and exact autoregressive normalization needs no partition sum.
        """
        if self.mode == 'hierarchical':
            return self._log_psi_hierarchical_reference(sigma, None)
        return self.log_psi_from_patch_indices(self.to_patch_indices(sigma), None)

    def log_psi_from_patch_indices(self, patch_values: Int[torch.Tensor, 'batch_size num_tokens'], condition: None=None) -> Complex[torch.Tensor, 'batch_size']:
        """Evaluate patch-mode coefficients directly from compact integer tokens."""
        if self.mode not in ('site', 'patch'):
            raise ValueError('Compact patch-token logpsi is available for patch mode only.')
        hidden = self._encode_patch_indices(patch_values, None)
        assert self.amplitude_head is not None
        logits = self.amplitude_head(hidden)
        patch_indices = patch_values.unsqueeze(-1)
        log_conditionals = F.log_softmax(logits, dim=-1)
        log_amplitude = 0.5 * log_conditionals.gather(-1, patch_indices).squeeze(-1).sum(-1)
        phase = torch.zeros_like(log_amplitude)
        return torch.complex(log_amplitude, phase)

    def _log_psi_hierarchical_reference(self, sigma: Int[torch.Tensor, 'batch_size Lx Ly'], condition: None=None) -> Complex[torch.Tensor, 'batch_size']:
        """Full-sequence reference path for the hierarchical patch decoder."""
        if self.subsampler is not None:
            return self._log_psi_subsampled_reference(sigma, None)
        assert self.hierarchical_decoder is not None
        patch_values = self.to_patch_indices(sigma)
        bits = self._ints_to_bits(patch_values).reshape(sigma.size(0), self.num_tokens, self.patch_size * self.patch_size)
        contexts = self._encode(sigma, None)
        sites_per_patch = bits.size(-1)
        decoder = self.hierarchical_decoder
        start = decoder.start_site.view(1, 1, 1, -1).expand(sigma.size(0), self.num_tokens, 1, -1)
        previous = decoder.previous_bit(bits[:, :, :-1]) if sites_per_patch > 1 else start[:, :, :0]
        token = torch.cat([start, previous], dim=2)
        hidden = contexts.unsqueeze(2) + token + decoder.position.view(1, 1, sites_per_patch, -1)
        logits = decoder.mlp(hidden)
        phase = None
        selected = bits.reshape(sigma.size(0), self.L)
        flat_logits = logits.reshape(sigma.size(0), self.L, 2)
        log_amplitude = 0.5 * F.log_softmax(flat_logits, dim=-1).gather(-1, selected.unsqueeze(-1)).squeeze(-1).sum(-1)
        phase_total = torch.zeros_like(log_amplitude)
        return torch.complex(log_amplitude, phase_total)

    def _log_psi_subsampled_reference(self, sigma: Int[torch.Tensor, 'batch_size Lx Ly'], condition: None=None) -> Complex[torch.Tensor, 'batch_size']:
        """Evaluate independently tokenised local blocks from vectorised coarse contexts."""
        assert self.subsampler is not None
        bits = sigma[:, self.patch_rows, self.patch_cols].long()
        contexts = self._encode(sigma, None)
        log_amplitude = torch.zeros(sigma.size(0), dtype=contexts.dtype, device=sigma.device)
        phase_total = torch.zeros_like(log_amplitude)
        for patch in range(self.num_tokens):
            local_lattice = bits[:, patch].reshape(sigma.size(0), self.patch_size, self.patch_size)
            local_tokens = self.subsampler._tokens(local_lattice)
            local_amp, local_phase = self.subsampler._evaluate_tokens(local_tokens, None, context=contexts[:, patch])
            log_amplitude = log_amplitude + local_amp
            phase_total = phase_total + local_phase
        return torch.complex(log_amplitude, phase_total)

    def log_psi_step(self, sigma: Int[torch.Tensor, 'batch_size Lx Ly'], condition: None=None) -> Complex[torch.Tensor, 'batch_size']:
        """Evaluate log psi with the cache-aware causal step path.

        This is kept as a separately testable path because sampling and connected-state evaluation
        benefit from prefix reuse, while the vectorized full-sequence path is faster for a single
        differentiable base-state evaluation on the current static Mixer.
        """
        if self.mode == 'hierarchical':
            return self._log_psi_hierarchical_step(sigma, None)
        patch_values = self.to_patch_indices(sigma)
        batch_size = sigma.size(0)
        caches: list[MixerLayerCache] | None = None
        log_amplitude = torch.zeros(batch_size, dtype=self.patch_projector.weight.dtype, device=sigma.device)
        phase_total = torch.zeros_like(log_amplitude)
        for p in range(self.num_tokens):
            if p == 0:
                token = self.start_token.expand(batch_size, -1, -1)
            else:
                token = self._patch_embedding(patch_values[:, p - 1])
            hidden, caches = self._encode_step(token, None, caches)
            logits, phase = self._step_outputs(hidden, None)
            selected = patch_values[:, p]
            log_probs = F.log_softmax(logits, dim=-1)
            log_amplitude = log_amplitude + 0.5 * log_probs.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
        return torch.complex(log_amplitude, phase_total)

    def _log_psi_hierarchical_step(self, sigma: Int[torch.Tensor, 'batch_size Lx Ly'], condition: None=None) -> Complex[torch.Tensor, 'batch_size']:
        """Cache-aware step path for hierarchical decoding."""
        if self.subsampler is not None:
            return self._log_psi_subsampled_step(sigma, None)
        assert self.hierarchical_decoder is not None
        patch_values = self.to_patch_indices(sigma)
        bits = self._ints_to_bits(patch_values).reshape(sigma.size(0), self.num_tokens, self.patch_size * self.patch_size)
        log_amplitude = torch.zeros(sigma.size(0), dtype=self.patch_projector.weight.dtype, device=sigma.device)
        phase_total = torch.zeros_like(log_amplitude)
        caches: list[MixerLayerCache] | None = None
        sites_per_patch = bits.size(-1)
        for patch in range(self.num_tokens):
            if patch == 0:
                token = self.start_token.expand(sigma.size(0), -1, -1)
            else:
                token = self._patch_embedding(patch_values[:, patch - 1])
            hidden, caches = self._encode_step(token, None, caches)
            context = hidden.squeeze(1)
            previous_bit = None
            for site in range(sites_per_patch):
                logits, phase = self.hierarchical_decoder.forward_step(context, previous_bit, site)
                selected = bits[:, patch, site]
                log_amplitude = log_amplitude + 0.5 * F.log_softmax(logits, dim=-1).gather(-1, selected.unsqueeze(-1)).squeeze(-1)
                previous_bit = selected
        return torch.complex(log_amplitude, phase_total)

    def _log_psi_subsampled_step(self, sigma: Int[torch.Tensor, 'batch_size Lx Ly'], condition: None=None) -> Complex[torch.Tensor, 'batch_size']:
        """Evaluate local subsamplers while advancing the cached coarse Mixer."""
        assert self.subsampler is not None
        patch_values = self.to_patch_indices(sigma)
        bits = sigma[:, self.patch_rows, self.patch_cols].long()
        log_amplitude = torch.zeros(sigma.size(0), dtype=self.patch_projector.weight.dtype, device=sigma.device)
        phase_total = torch.zeros_like(log_amplitude)
        caches: list[MixerLayerCache] | None = None
        for patch in range(self.num_tokens):
            token = self.start_token.expand(sigma.size(0), -1, -1) if patch == 0 else self._patch_embedding(patch_values[:, patch - 1])
            hidden, caches = self._encode_step(token, None, caches)
            local_lattice = bits[:, patch].reshape(sigma.size(0), self.patch_size, self.patch_size)
            local_tokens = self.subsampler._tokens(local_lattice)
            local_amp, local_phase = self.subsampler._evaluate_tokens(local_tokens, None, context=hidden[:, 0])
            log_amplitude = log_amplitude + local_amp
            phase_total = phase_total + local_phase
        return torch.complex(log_amplitude, phase_total)

    def log_psi(self, sigma: Int[torch.Tensor, 'batch_size Lx Ly'], condition: None=None) -> Complex[torch.Tensor, 'batch_size']:
        """Evaluate the wavefunction with the vectorized full-sequence path.

        The static Mixer can evaluate all causal positions in one tensorized pass, which is faster
        than a Python loop for the differentiable base-state calculation used by VMC for ground-state optimization. The cache-aware step path remains available as :meth:`log_psi_step` for sampling
        and prepared connected-state evaluation, where it removes repeated prefix work.
        """
        if type(self) is MixerWavefunction and self.mode == 'hierarchical' and (self.subsampler is not None):
            return self.evaluate_base(sigma).logpsi
        return self.log_psi_reference(sigma, None)

    def evaluate_base(self, sigma: torch.Tensor, *, max_local_rows: int=65536) -> MixerBaseEvaluation:
        """Evaluate a base batch once, retaining its graph and reusable local factors.

        Independent teacher-forced blocks can share a batch after their causal
        contexts are known. No sampling decisions are parallelized or changed.
        Callers must detach these values before using them in local energies.
        """
        if self.mode != 'hierarchical':
            return MixerBaseEvaluation(self.log_psi_reference(sigma), provenance=base_evaluation_provenance(self, sigma))
        local = self.subsampler
        if local is None:
            return MixerBaseEvaluation(self.log_psi_reference(sigma), provenance=base_evaluation_provenance(self, sigma))
        if max_local_rows <= 0:
            raise ValueError('max_local_rows must be positive.')
        if sigma.size(0) == 0:
            return MixerBaseEvaluation(self.log_psi_reference(sigma), provenance=base_evaluation_provenance(self, sigma))
        contexts = self._encode(sigma)
        batch = sigma.size(0)
        rows = batch * self.num_tokens
        bits = sigma[:, self.patch_rows, self.patch_cols].reshape(rows, self.patch_size, self.patch_size)
        flat_contexts = contexts.reshape(rows, self.channel_dim)
        from tnqs.adapters.torch.execution_memory import local_evaluation_row_limit
        row_limit = local_evaluation_row_limit(local, max_local_rows)
        factors = []
        for start in range(0, rows, row_limit):
            window = slice(start, start + row_limit)
            amplitude, phase = local._evaluate_tokens(local._tokens(bits[window]), context=flat_contexts[window])
            factors.append(torch.complex(amplitude, phase))
        contributions = torch.cat(factors).reshape(batch, self.num_tokens)
        logpsi = torch.complex(contributions.real.sum(-1), contributions.imag.sum(-1))
        return MixerBaseEvaluation(logpsi, contexts, contributions, base_evaluation_provenance(self, sigma))

    def to_patch_indices(self, sigma: Int[torch.Tensor, 'batch_size Lx Ly']) -> Int[torch.Tensor, 'batch_size num_tokens']:
        """Read each patch of a lattice configuration as an integer in [0, K_patch).

        Inverse of _ints_to_bits, using the same site ordering as block_lookup, so the index it
        returns selects the very logit that generated that patch during sampling.
        """
        bits = sigma.to(self.patch_rows.device)[:, self.patch_rows, self.patch_cols]
        return self._bits_to_ints(bits)

    def _bits_to_ints(self, bits: Int[torch.Tensor, '... sites_per_patch']) -> Int[torch.Tensor, '...']:
        """Pack bits back into an integer, matching the bit k = (idx >> k) & 1 convention."""
        num_sites = self.patch_size * self.patch_size
        bit_positions = torch.arange(num_sites, device=bits.device)
        return (bits.long() << bit_positions).sum(-1)

    def _ints_to_bits(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: [...], long in [0, 2^S)
        returns: [..., S] bits in {0,1}
        using standard binary expansion:
            bit k = (idx >> k) & 1,  for k=0..S-1
        """
        device = idx.device
        S = self.patch_size * self.patch_size
        bit_positions = torch.arange(S, device=device)
        bits = (idx.unsqueeze(-1) >> bit_positions & 1).long()
        return bits

    def _shift_right(self, patches: Float[torch.Tensor, 'batch_size num_tokens channel_dim']) -> Float[torch.Tensor, 'batch_size num_tokens channel_dim']:
        """Drop the last patch embedding and prepend a learned start token, so that position p
        carries patch p-1 and the logits at p condition on patches 0..p-1 only.

        Without this the residual stream in each MixerLayer would carry patch p's own embedding
        into the logits that predict it, letting the model learn the identity instead of the
        conditional. Masking the token mixer alone cannot prevent that, as the residual bypasses it.
        """
        batch_size = patches.size(0)
        return torch.cat([self.start_token.expand(batch_size, -1, -1), patches[:, :-1, :]], dim=1)

    def _prep_patches(self, x: Int[torch.Tensor, 'batch_size Lx Ly']) -> Float[torch.Tensor, 'batch_size num_tokens channel_dim']:
        """Gather exact projector embeddings from compact integer patch tokens.

        A strided convolution over non-overlapping binary patches is mathematically a lookup.
        Keeping that lookup differentiable removes a poorly occupied tiny convolution without
        changing the model's parameters or analytical parameter count.
        """
        bits = x.to(self.patch_rows.device)[:, self.patch_rows, self.patch_cols]
        weight = self.patch_projector.weight[:, 0].flatten(1)
        return F.linear(bits.to(weight), weight, self.patch_projector.bias)

    @torch.no_grad()
    def sample(self, condition: None, num_samples: int, *, strategy: str='cached', generator: torch.Generator | None=None) -> Int[torch.Tensor, 'num_samples Lx Ly']:
        """Draw independent configurations from the normalized Born distribution.

        Because every conditional is normalised, these are exact independent draws
        from |psi|^2 with no Markov chain and no burn-in.
        """
        if strategy not in ('cached', 'full_prefix'):
            raise ValueError(f'Unknown sampler strategy: {strategy!r}.')
        generator = self._sampling_generator if generator is None else generator
        if self.mode == 'hierarchical':
            if strategy != 'cached':
                raise ValueError('full_prefix sampling is available for patch mode only.')
            return self._sample_hierarchical(None, num_samples, generator=generator)
        if strategy == 'full_prefix':
            return self._sample_full_prefix(None, num_samples, generator=generator)
        caches: list[MixerLayerCache] | None = None
        previous_patch: torch.Tensor | None = None
        patch_values = torch.empty((num_samples, self.num_tokens), dtype=torch.int64, device=self.device)
        embedding_table = None
        for p in range(self.num_tokens):
            if p == 0:
                token = self.start_token.expand(num_samples, -1, -1)
            else:
                assert previous_patch is not None
                token = self._patch_embedding(previous_patch, embedding_table)
            hidden, caches = self._encode_step(token, None, caches)
            logits_p = self._step_amplitude_logits(hidden, None)
            sampled_patches = sample_probabilities(F.softmax(logits_p, dim=-1), generator)
            previous_patch = sampled_patches
            patch_values[:, p] = sampled_patches
        return self.from_patch_indices(patch_values)

    @torch.no_grad()
    def _sample_full_prefix(self, condition: None, num_samples: int, *, generator: torch.Generator | None=None) -> Int[torch.Tensor, 'num_samples Lx Ly']:
        """Trade repeated nominal FLOPs for fewer, larger accelerator launches.

        The causal mask makes the placeholder tokens to the right of the current position
        invisible.  For the four-token 4x4 model, evaluating all prefix positions together is
        substantially faster on MPS than replaying many tiny cache updates.
        """
        patch_values = torch.zeros((num_samples, self.num_tokens), dtype=torch.int64, device=self.device)
        embedding_table = None
        assert self.amplitude_head is not None
        for position in range(self.num_tokens):
            hidden = self._encode_patch_indices(patch_values, None, embedding_table)
            logits = self.amplitude_head(hidden[:, position])
            selected = sample_probabilities(F.softmax(logits, dim=-1), generator)
            patch_values[:, position] = selected
        return self.from_patch_indices(patch_values)

    def from_patch_indices(self, patch_values: Int[torch.Tensor, 'batch_size num_tokens']) -> Int[torch.Tensor, 'batch_size Lx Ly']:
        """Convert compact patch tokens back to a bit lattice without a Python patch loop."""
        if patch_values.ndim != 2 or patch_values.shape[1] != self.num_tokens:
            raise ValueError(f'Expected patch values shaped (batch, {self.num_tokens}).')
        bits = self._ints_to_bits(patch_values)
        lattice = torch.empty((patch_values.shape[0], self.Lx, self.Ly), dtype=torch.int64, device=patch_values.device)
        lattice[:, self.patch_rows, self.patch_cols] = bits
        return lattice

    @torch.no_grad()
    def _sample_hierarchical(self, condition: None, num_samples: int, *, generator: torch.Generator | None=None) -> Int[torch.Tensor, 'num_samples Lx Ly']:
        """Sample sites inside each patch with a shared causal backbone context."""
        if self.subsampler is not None:
            return self._sample_subsampled(None, num_samples, generator=generator)
        assert self.hierarchical_decoder is not None
        lattice = torch.zeros((num_samples, self.Lx, self.Ly), dtype=torch.int64, device=self.device)
        previous_patch: torch.Tensor | None = None
        caches: list[MixerLayerCache] | None = None
        sites_per_patch = self.patch_size * self.patch_size
        for patch in range(self.num_tokens):
            if patch == 0:
                token = self.start_token.expand(num_samples, -1, -1)
            else:
                assert previous_patch is not None
                token = self._patch_embedding(previous_patch)
            hidden, caches = self._encode_step(token, None, caches)
            context = hidden.squeeze(1)
            bits = torch.zeros(num_samples, sites_per_patch, dtype=torch.int64, device=self.device)
            previous_bit: torch.Tensor | None = None
            for site in range(sites_per_patch):
                logits, _ = self.hierarchical_decoder.forward_step(context, previous_bit, site)
                bit = sample_probabilities(F.softmax(logits, dim=-1), generator)
                bits[:, site] = bit
                previous_bit = bit
            previous_patch = self._bits_to_ints(bits)
            p_coords = self.block_lookup[patch]
            rows, cols = zip(*p_coords)
            lattice[:, rows, cols] = bits
        return lattice

    def _sample_subsampled(self, condition: None, num_samples: int, *, generator: torch.Generator | None=None) -> Int[torch.Tensor, 'num_samples Lx Ly']:
        """Sample each coarse block using its independently tokenised local model."""
        assert self.subsampler is not None
        lattice = torch.zeros((num_samples, self.Lx, self.Ly), dtype=torch.long, device=self.device)
        previous_patch: torch.Tensor | None = None
        caches: list[MixerLayerCache] | None = None
        sites_per_patch = self.patch_size * self.patch_size
        for patch in range(self.num_tokens):
            token = self.start_token.expand(num_samples, -1, -1) if patch == 0 else self._patch_embedding(previous_patch)
            hidden, caches = self._encode_step(token, None, caches)
            local = self.subsampler._sample_tokens(num_samples, None, context=hidden[:, 0], generator=generator)
            if self.subsampler.mode == 'site':
                local_lattice = local.reshape(num_samples, self.patch_size, self.patch_size)
            else:
                local_lattice = self.subsampler.from_patch_indices(local)
            bits = local_lattice.reshape(num_samples, sites_per_patch)
            lattice[:, self.patch_rows[patch], self.patch_cols[patch]] = bits
            previous_patch = self._bits_to_ints(bits)
        return lattice
