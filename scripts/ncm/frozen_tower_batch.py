"""Independent concept-level batches sharing one frozen source prompt.

An optional execution experiment. Numerical and memory audits must pass before
this replaces sequential levels. Padding never enters the training objective.
"""
import torch
from torch.utils.checkpoint import checkpoint

from .frozen_tower import validate, layer_dtype, prepare
from .prefix import fork_layer, mask_for, rotary


def cached_many(backbone, x, state, lengths):
    validate(backbone, state)
    if (x.ndim != 3 or x.shape[0] != len(lengths) or not lengths
            or min(lengths) < 1 or max(lengths) > x.shape[1]):
        raise ValueError('Invalid independent level batch')
    batch, size, past = x.shape[0], x.shape[1], state.cached_tokens
    tail = state.tokens.shape[1] - past
    if tail:
        prefix = backbone.embed_prompt(state.tokens[:, past:]).to(x).repeat(batch, 1, 1)
        x = torch.cat((prefix, x), 1)
    if x.shape[1] == 1:
        x = torch.cat((x, torch.zeros_like(x)), 1)
    for i, (layer, adapter) in enumerate(zip(backbone.layers, backbone.adapters)):
        di = backbone.layer_device_indices[i]
        x = x.to(backbone.devices[di], dtype=layer_dtype(layer))
        rope = rotary(backbone, x, di, past)
        mask = None
        if layer.block_type == 'full_attention':
            mask = torch.cat([mask_for(x.shape[1], past, x.device, x.dtype, parallel=True, valid=tail + n)
                              for n in lengths])
            if tail:
                keys = torch.arange(past + x.shape[1], device=x.device)
                queries = torch.arange(past, past + tail, device=x.device)
                mask[:, :, :tail] = mask[:, :, :tail].masked_fill(keys[None, :] > queries[:, None], float('-inf'))

        def step(h, layer=layer, adapter=adapter, rope=rope, mask=mask, index=i):
            cache = fork_layer(state.cache, index, batch_size=batch)
            y = layer(h, position_embeddings=rope, attention_mask=mask, past_key_values=cache)
            return torch.cat((y[:, :tail], adapter(y[:, tail:])), 1)

        x = (checkpoint(step, x, use_reentrant=False)
             if backbone.training and torch.is_grad_enabled() else step(x))
    return backbone.norm(x[:, tail:tail + size])


def forward_levels(model, inputs, prompt, state=None):
    if not model.parallel_attention or len(inputs) != len(model.cfg.levels):
        raise ValueError('Expected parallel concept levels')
    state = prepare(model.backbone, prompt) if state is None else state
    if not torch.equal(state.tokens, prompt.cpu()):
        raise ValueError('Prefix prompt differs')
    queries = []
    for level, (inp, size) in enumerate(zip(inputs, model.cfg.levels)):
        if inp.shape[:2] != (1, size):
            raise ValueError('Invalid level input')
        x = model.project(inp.to(model.project.weight.device)) + model.level.weight[level] + model.position[:size]
        if model.concept_input_norm is not None:
            x = torch.nn.functional.normalize(x, dim=-1, eps=1e-6) * model.concept_input_norm
        queries.append(torch.nn.functional.pad(x, (0, 0, 0, model.cfg.levels[-1] - size)))
    hidden = cached_many(model.backbone, torch.cat(queries), state, model.cfg.levels).float()
    results = []
    for level, size in enumerate(model.cfg.levels):
        h = hidden[level:level + 1, :size]
        logits, latent = model.head(h, level, return_latent=True)
        results.append((logits, model.stop_head(h.mean(1)).squeeze(-1), latent))
    return results
