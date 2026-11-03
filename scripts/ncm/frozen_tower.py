"""Experimental frozen-context execution of the original 32-layer NCM backbone.

The prompt never runs adapters. Concept queries run the existing source layers
and adapters, reading independent copies of the frozen prompt's hybrid state.
This is a new training architecture, not an equivalent resume of v24. It is also
distinct from v25's small, independently initialized concept decoder.

No vocabulary head, target answer tokens, or AR generation is used here.
GPU cache execution requires its own whole-model numerical qualification.
"""
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache

from .backbone import concept_attention_mask
from .prefix import fork_layer, mask_for, rotary


def source_versions(backbone):
    modules = [backbone.layers, backbone.norm, backbone.rotaries]
    if hasattr(backbone, 'embedding'):
        modules.append(backbone.embedding)
    tensors = [t for module in modules
               for t in list(module.parameters()) + list(module.buffers())]
    if any(t.requires_grad for t in tensors):
        raise ValueError('Frozen context requires frozen source parameters')
    return tuple((id(t), t._version) for t in tensors)


def layer_dtype(layer):
    # Packed GGUF linears intentionally expose bytes, not a fake .weight.
    weight = getattr(layer.mlp.gate_proj, 'weight', None)
    return weight.dtype if weight is not None else torch.bfloat16


@dataclass
class FrozenPrefix:
    cache: object
    tokens: torch.Tensor
    owner: int
    versions: tuple


@torch.no_grad()
def prepare(backbone, tokens):
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1:
        raise ValueError('Expected one nonempty prompt')
    source_versions(backbone)
    cache = DynamicCache(config=backbone.config)
    x = backbone.embed_prompt(tokens)
    for i, layer in enumerate(backbone.layers):
        di = backbone.layer_device_indices[i]
        x = x.to(backbone.devices[di], dtype=layer_dtype(layer))
        rope = rotary(backbone, x, di, 0)
        mask = (mask_for(x.shape[1], 0, x.device, x.dtype)
                if layer.block_type == 'full_attention' else None)
        x = layer(x, position_embeddings=rope, attention_mask=mask, past_key_values=cache)
    return FrozenPrefix(cache, tokens.detach().cpu().clone(), id(backbone), source_versions(backbone))


def validate(backbone, state):
    if state.owner != id(backbone) or state.versions != source_versions(backbone):
        raise ValueError('Frozen prefix belongs to a different or modified source')


def cached(backbone, x, state):
    validate(backbone, state)
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 1:
        raise ValueError('Expected one nonempty concept level')
    size, past = x.shape[1], state.tokens.shape[1]
    if size == 1:
        # The fused one-token recurrent path is inference-only. The dummy key
        # is masked, and causal recurrence cannot affect earlier real positions.
        x = torch.cat((x, torch.zeros_like(x)), 1)
    for i, (layer, adapter) in enumerate(zip(backbone.layers, backbone.adapters)):
        di = backbone.layer_device_indices[i]
        x = x.to(backbone.devices[di], dtype=layer_dtype(layer))
        rope = rotary(backbone, x, di, past)
        mask = (mask_for(x.shape[1], past, x.device, x.dtype, parallel=True, valid=size)
                if layer.block_type == 'full_attention' else None)

        def step(h, layer=layer, adapter=adapter, rope=rope, mask=mask, index=i):
            cache = fork_layer(state.cache, index)
            return adapter(layer(h, position_embeddings=rope, attention_mask=mask, past_key_values=cache))

        x = (checkpoint(step, x, use_reentrant=False)
             if backbone.training and torch.is_grad_enabled() else step(x))
    return backbone.norm(x[:, :size])


def reference(backbone, x, tokens):
    """Uncached control: adapters apply to concept positions only at each layer."""
    source_versions(backbone)
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1 or x.shape[0] != 1:
        raise ValueError('Expected one prompt and one concept level')
    past = tokens.shape[1]
    prefix = backbone.embed_prompt(tokens).to(x)
    x = torch.cat((prefix, x), 1)
    for i, (layer, adapter) in enumerate(zip(backbone.layers, backbone.adapters)):
        di = backbone.layer_device_indices[i]
        x = x.to(backbone.devices[di], dtype=layer_dtype(layer))
        rope = rotary(backbone, x, di, 0)
        mask = (concept_attention_mask(x.shape[1], past, x.device, x.dtype)
                if layer.block_type == 'full_attention' else None)

        def step(h, layer=layer, adapter=adapter, rope=rope, mask=mask):
            y = layer(h, position_embeddings=rope, attention_mask=mask, past_key_values=None)
            return torch.cat((y[:, :past], adapter(y[:, past:])), 1)

        x = (checkpoint(step, x, use_reentrant=False)
             if backbone.training and torch.is_grad_enabled() else step(x))
    return backbone.norm(x[:, past:])


def level_output(model, inp, prompt, level, use_cache=True, state=None):
    size = model.cfg.levels[level]
    if inp.shape[:2] != (1, size):
        raise ValueError('Invalid concept input shape')
    if use_cache and (state is None or not torch.equal(state.tokens, prompt.cpu())):
        raise ValueError('Matching frozen prefix required')
    x = model.project(inp.to(model.project.weight.device)) + model.level.weight[level] + model.position[:size]
    if model.concept_input_norm is not None:
        x = torch.nn.functional.normalize(x, dim=-1, eps=1e-6) * model.concept_input_norm
    hidden = (cached(model.backbone, x, state) if use_cache
              else reference(model.backbone, x, prompt)).float()
    logits, latent = model.head(hidden, level, return_latent=True)
    stop = model.stop_head(hidden.mean(1)).squeeze(-1)
    return logits, stop, latent


def forward_levels(model, inputs, prompt, use_cache=True, state=None):
    if not model.parallel_attention or len(inputs) != len(model.cfg.levels):
        raise ValueError('Expected the parallel concept-level architecture')
    if use_cache:
        state = prepare(model.backbone, prompt) if state is None else state
        if not torch.equal(state.tokens, prompt.cpu()):
            raise ValueError('Prefix prompt differs')
    elif state is not None:
        raise ValueError('Uncached control cannot accept a cache')
    return [level_output(model, inp, prompt, level, use_cache, state)
            for level, inp in enumerate(inputs)]
