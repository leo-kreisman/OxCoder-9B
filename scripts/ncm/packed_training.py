"""Pack independent shared-prefix NCM examples without cross-example attention.

This is an experimental execution path. It changes neither the example mean
objective nor conditioning: each conversation has its own prefix, recurrent
state, convolution history, positions, and isolated concept-level branches.
Require a full-model forward/backward audit before using it for training.
"""
from contextlib import contextmanager

import torch

from .shared_training import attention_mask
from .vendor.veomni_shared_prefix import SharedPrefixPlan, _Group


def layout(prefixes, sizes, device):
    if not prefixes or min(prefixes) < 64 or not sizes or min(sizes) < 1:
        raise ValueError('Packed shared prefixes must each contain at least 64 tokens')
    groups, total = [], 0
    for prefix in prefixes:
        starts, lengths = [], []
        for size in sizes:
            starts.append(total)
            lengths.append(prefix + size)
            total += prefix + size
        groups.append(_Group(starts, lengths, prefix))
    return SharedPrefixPlan(groups, total, 4, torch.device(device))


def mask_and_positions(prefixes, sizes, device, dtype):
    lengths = [prefix + sum(sizes) for prefix in prefixes]
    mask = torch.full((sum(lengths), sum(lengths)), float('-inf'), device=device, dtype=dtype)
    positions, start = [], 0
    for prefix, length in zip(prefixes, lengths):
        mask[start:start + length, start:start + length] = attention_mask(prefix, sizes, device, dtype)[0, 0]
        positions.extend(range(prefix))
        for size in sizes:
            positions.extend(range(prefix, prefix + size))
        start += length
    position_ids = torch.tensor(positions, device=device)[None, None, :].expand(3, 1, -1)
    return mask[None, None], position_ids


def convolution(original, x, weight, bias, activation, prefixes, sizes):
    outputs, start = [], 0
    history = weight.shape[-1] - 1
    for prefix in prefixes:
        context = x[:, :, start:start + prefix]
        outputs.append(original(context, weight, bias, activation=activation))
        start += prefix
        for size in sizes:
            tail = context[:, :, -history:] if history else context[:, :, :0]
            branch = torch.cat((tail, x[:, :, start:start + size]), dim=2)
            outputs.append(original(branch, weight, bias, activation=activation)[:, :, -size:])
            start += size
    if start != x.shape[-1]:
        raise ValueError('Convolution layout does not cover the compact sequence')
    return torch.cat(outputs, dim=2)


@contextmanager
def mixers(plan, prefixes, sizes):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    original_conv, original_gdn = hf.causal_conv1d_fn, hf.torch_chunk_gated_delta_rule

    def conv(x, weight, bias=None, activation=None, **kwargs):
        return convolution(original_conv, x, weight, bias, activation, prefixes, sizes)

    def gdn(query, key, value, g, beta, **kwargs):
        if kwargs.pop('initial_state', None) is not None or kwargs.pop('output_final_state', False):
            raise ValueError('Packed training cannot consume inference states')
        if kwargs.pop('cu_seqlens', None) is not None:
            raise ValueError('Nested sequence packing is unsupported')
        return plan.gated_delta_rule(original_gdn, query, key, value, g, beta, **kwargs), None

    hf.causal_conv1d_fn, hf.torch_chunk_gated_delta_rule = conv, gdn
    try:
        yield
    finally:
        hf.causal_conv1d_fn, hf.torch_chunk_gated_delta_rule = original_conv, original_gdn


def forward_examples(model, examples):
    from . import backbone as implementation
    if model.cached_prefix or not model.parallel_attention or not examples:
        raise ValueError('Requires uncached parallel-attention NCM examples')
    backbone, sizes = model.backbone, model.cfg.levels
    prefixes, parts = [], []
    first = model.project.weight.device
    for inputs, prompt in examples:
        if prompt.shape[0] != 1 or prompt.shape[1] < 64 or len(inputs) != len(sizes):
            raise ValueError('Invalid packed example shape')
        prefixes.append(prompt.shape[1])
        parts.append(backbone.embed_prompt(prompt).to(first, dtype=model.project.weight.dtype))
        for level, (inp, size) in enumerate(zip(inputs, sizes)):
            if inp.shape[:2] != (1, size):
                raise ValueError('Invalid concept input shape')
            x = model.project(inp.to(first)) + model.level.weight[level] + model.position[:size]
            if model.concept_input_norm is not None:
                x = torch.nn.functional.normalize(x, dim=-1, eps=1e-6) * model.concept_input_norm
            parts.append(x)
    x = torch.cat(parts, dim=1)
    plans, masks, positions = {}, {}, {}
    for device in backbone.devices:
        plans[str(device)] = layout(prefixes, sizes, device)
        masks[str(device)], positions[str(device)] = mask_and_positions(prefixes, sizes, device, torch.bfloat16)
    for i, (layer, adapter) in enumerate(zip(backbone.layers, backbone.adapters)):
        di = backbone.layer_device_indices[i]
        device = backbone.devices[di]
        x = x.to(device, dtype=torch.bfloat16)
        rope = backbone.rotaries[di](x, positions[str(device)])
        mask = masks[str(device)] if i % 4 == 3 else None
        plan = plans[str(device)]

        def step(h, layer=layer, adapter=adapter, rope=rope, mask=mask, plan=plan):
            with mixers(plan, prefixes, sizes):
                return adapter(layer(h, position_embeddings=rope, attention_mask=mask, past_key_values=None))

        x = implementation.checkpoint(step, x, use_reentrant=False) if model.training and torch.is_grad_enabled() else step(x)
    x = backbone.norm(x).float()
    results, start = [], 0
    for prefix in prefixes:
        start += prefix
        outputs = []
        for level, size in enumerate(sizes):
            hidden = x[:, start:start + size]
            start += size
            logits, residual = model.head(hidden, level, return_latent=True)
            outputs.append((logits, model.stop_head(hidden.mean(1)).squeeze(-1), residual))
        results.append(outputs)
    return results
