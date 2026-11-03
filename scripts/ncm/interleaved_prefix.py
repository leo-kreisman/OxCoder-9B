"""Experimental frozen-prefix prefill with one unpack per layer and dtype.

Process the same independent prompts in layer-major order. Linear arithmetic,
rotary positions, masks and recurrent states are unchanged. Temporary unpacked
weights are shared only during this no-gradient prefill, then discarded.
This is not a persistent prefix cache or an admitted training runtime.
"""
import copy
from contextlib import contextmanager

import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache

from . import frozen_tower as tower
from .frozen_tower_multi import BatchPrefix, prepare as serial_prepare
from .frozen_tower_cache import fork_layer
from .gguf_training_weights import FrozenGGUFLinear, unpack
from .prefix import mask_for, rotary


@contextmanager
def layer_weights(layer, statistics=None):
    """Bounded temporary cache; never install during gradient computation.

Forward replacement is instance-local and restored even on exceptions. Calls
must be serialized; concurrent inference/training on this same module is not
supported by this experimental helper.
    """
    if torch.is_grad_enabled():
        raise RuntimeError('Weight sharing is restricted to no-gradient prefill')
    replaced = []
    missing = object()
    try:
        for module in layer.modules():
            if not isinstance(module, FrozenGGUFLinear):
                continue
            if getattr(module, '_ncm_unpack_scope', False):
                raise RuntimeError('Nested or concurrent unpack scopes are unsupported')
            original = module.__dict__.get('forward', missing)
            cache = {}

            def forward(x, module=module, cache=cache):
                if torch.is_grad_enabled():
                    raise RuntimeError('Cached frozen weights cannot enter gradient execution')
                key = (x.device, x.dtype)
                if statistics is not None:
                    statistics['linear_calls'] = statistics.get('linear_calls', 0) + 1
                if key not in cache:
                    cache[key] = unpack(module.packed, module.shape, module.tensor_type,
                                        module.source_name, dtype=x.dtype)
                    if statistics is not None:
                        statistics['unpacks'] = statistics.get('unpacks', 0) + 1
                return F.linear(x, cache[key])

            module.forward = forward
            module._ncm_unpack_scope = True
            replaced.append((module, original, cache))
        yield
    finally:
        for module, original, cache in reversed(replaced):
            if original is missing:
                del module.forward
            else:
                module.forward = original
            del module._ncm_unpack_scope
            cache.clear()


@torch.no_grad()
def prepare(backbone, prompts, statistics=None):
    if not prompts or any(p.ndim != 2 or p.shape[0] != 1 or p.shape[1] < 1 for p in prompts):
        raise ValueError('Expected nonempty independent prompts')
    if len(prompts) == 1:
        return serial_prepare(backbone, prompts)
    versions = tower.source_versions(backbone)
    lengths = tuple(p.shape[1] for p in prompts)
    caches = [DynamicCache(config=backbone.config) for _ in prompts]
    hidden = [backbone.embed_prompt(p) for p in prompts]
    for index, layer in enumerate(backbone.layers):
        device_index = backbone.layer_device_indices[index]
        with layer_weights(layer, statistics):
            for row in range(len(prompts)):
                x = hidden[row].to(backbone.devices[device_index], dtype=tower.layer_dtype(layer))
                rope = rotary(backbone, x, device_index, 0)
                mask = (mask_for(x.shape[1], 0, x.device, x.dtype)
                        if layer.block_type == 'full_attention' else None)
                hidden[row] = layer(x, position_embeddings=rope, attention_mask=mask,
                                    past_key_values=caches[row])
    del hidden
    maximum = max(lengths)
    cache = copy.copy(caches[0])
    cache.layers = []
    for index in range(len(backbone.layers)):
        layer = fork_layer(caches[0], index).layers[index]
        originals = [c.layers[index] for c in caches]
        for name in ('keys', 'values'):
            if hasattr(layer, name):
                values = [F.pad(getattr(x, name), (0, 0, 0, maximum - n))
                          for x, n in zip(originals, lengths)]
                setattr(layer, name, torch.cat(values))
        for name in ('conv_states', 'recurrent_states'):
            if hasattr(layer, name):
                values = getattr(layer, name)
                setattr(layer, name, {k: torch.cat([getattr(x, name)[k] for x in originals])
                                     for k in values})
        cache.layers.append(layer)
    if tower.source_versions(backbone) != versions:
        raise RuntimeError('Frozen source changed during prefill')
    return BatchPrefix(cache, tuple(p.detach().cpu().clone() for p in prompts), lengths,
                       id(backbone), versions)
