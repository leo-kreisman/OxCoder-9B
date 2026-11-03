"""Experimental variable-prompt batching, separate from the running v29 engine.

Each prompt is prefetched without padding. Full-attention keys are then padded
and masked; recurrent state is stacked directly. Per-row rotary positions use
the real prompt length. This path requires its own GPU numerical qualification.
"""
import copy
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from . import frozen_tower as tower
from .frozen_tower_cache import fork_layer


@dataclass
class BatchPrefix:
    cache: object
    prompts: tuple
    lengths: tuple
    owner: int
    versions: tuple


@torch.no_grad()
def prepare(backbone, prompts):
    if not prompts:
        raise ValueError('At least one prompt required')
    states = [tower.prepare(backbone, p, alignment=1) for p in prompts]
    lengths = tuple(p.shape[1] for p in prompts)
    maximum = max(lengths)
    cache = copy.copy(states[0].cache)
    cache.layers = []
    for index in range(len(backbone.layers)):
        layer = fork_layer(states[0].cache, index).layers[index]
        originals = [s.cache.layers[index] for s in states]
        for name in ('keys', 'values'):
            if hasattr(layer, name):
                tensors = [F.pad(getattr(x, name), (0, 0, 0, maximum - n)) for x, n in zip(originals, lengths)]
                setattr(layer, name, torch.cat(tensors))
        for name in ('conv_states', 'recurrent_states'):
            if hasattr(layer, name):
                values = getattr(layer, name)
                setattr(layer, name, {k: torch.cat([getattr(x, name)[k] for x in originals]) for k in values})
        cache.layers.append(layer)
    return BatchPrefix(cache, tuple(p.detach().cpu().clone() for p in prompts), lengths,
                       id(backbone), tower.source_versions(backbone))


def repeat_layer(cache, index, groups):
    fork = fork_layer(cache, index)
    layer = fork.layers[index]
    if groups > 1:
        def repeat(x):
            return x.repeat(groups, *([1] * (x.ndim - 1)))
        for name in ('keys', 'values'):
            if hasattr(layer, name):
                setattr(layer, name, repeat(getattr(layer, name)))
        for name in ('conv_states', 'recurrent_states'):
            if hasattr(layer, name):
                setattr(layer, name, {k: repeat(v) for k, v in getattr(layer, name).items()})
    return fork


def hidden(backbone, x, state, level_lengths):
    if state.owner != id(backbone) or state.versions != tower.source_versions(backbone):
        raise ValueError('Wrong or stale multi-prompt prefix')
    batch, groups = len(state.lengths), len(level_lengths)
    if (x.ndim != 3 or x.shape[0] != batch * groups or min(level_lengths) < 1
            or max(level_lengths) > x.shape[1]):
        raise ValueError('Invalid level-major query batch')
    size = x.shape[1]
    if size == 1:
        x = torch.cat((x, torch.zeros_like(x)), 1)
    past = max(state.lengths)
    for index, (layer, adapter) in enumerate(zip(backbone.layers, backbone.adapters)):
        di = backbone.layer_device_indices[index]
        x = x.to(backbone.devices[di], dtype=tower.layer_dtype(layer))
        starts = torch.tensor(state.lengths * groups, device=x.device)
        positions = starts[:, None] + torch.arange(x.shape[1], device=x.device)[None, :]
        rope = backbone.rotaries[di](x, positions[None].expand(3, -1, -1))
        mask = None
        if layer.block_type == 'full_attention':
            keys = torch.arange(past + x.shape[1], device=x.device)
            valid_prefix = keys[None, :] < starts[:, None]
            valid_queries = torch.tensor(level_lengths, device=x.device).repeat_interleave(batch)
            valid_target = (keys[None, :] >= past) & (keys[None, :] < past + valid_queries[:, None])
            mask = torch.zeros(batch * groups, 1, x.shape[1], past + x.shape[1], device=x.device, dtype=x.dtype)
            mask.masked_fill_(~(valid_prefix | valid_target)[:, None, None, :], float('-inf'))

        def step(h, layer=layer, adapter=adapter, rope=rope, mask=mask, index=index):
            cache = repeat_layer(state.cache, index, groups)
            return adapter(layer(h, position_embeddings=rope, attention_mask=mask, past_key_values=cache))

        x = checkpoint(step, x, use_reentrant=False) if backbone.training and torch.is_grad_enabled() else step(x)
    return backbone.norm(x[:, :size])


def forward_levels(model, inputs, prompts, state=None):
    if not model.parallel_attention or len(inputs) != len(model.cfg.levels):
        raise ValueError('Expected all parallel concept levels')
    state = prepare(model.backbone, prompts) if state is None else state
    if len(prompts) != len(state.prompts) or any(not torch.equal(a.cpu(), b) for a, b in zip(prompts, state.prompts)):
        raise ValueError('Prompt batch differs')
    batch = len(prompts)
    queries = []
    for level, (inp, size) in enumerate(zip(inputs, model.cfg.levels)):
        if inp.shape[:2] != (batch, size):
            raise ValueError('Wrong batched concept input shape')
        x = model.project(inp.to(model.project.weight.device)) + model.level.weight[level] + model.position[:size]
        if model.concept_input_norm is not None:
            x = F.normalize(x, dim=-1, eps=1e-6) * model.concept_input_norm
        queries.append(F.pad(x, (0, 0, 0, model.cfg.levels[-1] - size)))
    h = hidden(model.backbone, torch.cat(queries), state, model.cfg.levels).float()
    outputs = []
    for level, size in enumerate(model.cfg.levels):
        part = h[level * batch:(level + 1) * batch, :size]
        logits, latent = model.head(part, level, return_latent=True)
        outputs.append((logits, model.stop_head(part.mean(1)).squeeze(-1), latent))
    return outputs
