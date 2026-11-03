"""Experimental unpadded independent concept levels for the frozen source.

Qwen3.5 layer equations follow Transformers (Apache-2.0). GPU recurrence uses
FLA's variable-length API. CPU controls use separate reference recurrences.
This module is not admitted to training without a full numerical/speed audit.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
from . import frozen_tower as tower
from . import frozen_tower_multi as multi
from .frozen_tower_cache import fork_layer


def recurrence(module, h, prefix, lengths):
    batch, total, _ = h.shape
    raw = module.in_proj_qkv(h).transpose(1, 2)
    previous = prefix.conv_states[0][:, :, -(module.conv_kernel_size - 1):]
    parts, offset = [], 0
    for length in lengths:
        x = torch.cat((previous.to(raw), raw[:, :, offset:offset + length]), -1)
        x = F.conv1d(x, module.conv1d.weight, module.conv1d.bias, groups=module.conv_dim)
        parts.append(ACT2FN[module.activation](x))
        offset += length
    qkv = torch.cat(parts, -1).transpose(1, 2)
    q, k, v = torch.split(qkv, (module.key_dim, module.key_dim, module.value_dim), -1)
    q = q.reshape(batch, total, -1, module.head_k_dim)
    k = k.reshape(batch, total, -1, module.head_k_dim)
    v = v.reshape(batch, total, -1, module.head_v_dim)
    z = module.in_proj_z(h).reshape(-1, module.head_v_dim)
    beta = module.in_proj_b(h).sigmoid()
    g = -module.A_log.float().exp() * F.softplus(module.in_proj_a(h).float() + module.dt_bias)
    ratio = module.num_v_heads // module.num_k_heads
    if ratio > 1:
        q, k = q.repeat_interleave(ratio, 2), k.repeat_interleave(ratio, 2)
    initial = prefix.recurrent_states[0]
    if h.is_cuda:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        cumulative = [0]
        for n in lengths * batch:
            cumulative.append(cumulative[-1] + n)
        cpu_lengths = torch.tensor(cumulative, dtype=torch.long)
        def flatten(x):
            return x.reshape(1, batch * total, *x.shape[2:])
        output, _ = chunk_gated_delta_rule(flatten(q), flatten(k), flatten(v),
            g=flatten(g), beta=flatten(beta), initial_state=initial.repeat_interleave(len(lengths), 0),
            output_final_state=False, use_qk_l2norm_in_kernel=True,
            cu_seqlens=cpu_lengths.to(h.device), cu_seqlens_cpu=cpu_lengths)
        output = output.reshape(batch, total, -1, module.head_v_dim)
    else:
        outputs, offset = [], 0
        for length in lengths:
            section = slice(offset, offset + length)
            output, _ = torch_chunk_gated_delta_rule(q[:, section], k[:, section], v[:, section],
                g=g[:, section], beta=beta[:, section], initial_state=initial,
                output_final_state=False, use_qk_l2norm_in_kernel=True)
            outputs.append(output)
            offset += length
        output = torch.cat(outputs, 1)
    output = module.norm(output.reshape(-1, module.head_v_dim), z).reshape(batch, total, -1)
    return module.out_proj(output)


def hidden(backbone, x, state, lengths):
    if state.owner != id(backbone) or state.versions != tower.source_versions(backbone):
        raise ValueError('Wrong or stale prefix')
    if x.shape[:2] != (len(state.lengths), sum(lengths)) or min(lengths) < 1:
        raise ValueError('Invalid packed levels')
    past = max(state.lengths)
    for index, (layer, adapter) in enumerate(zip(backbone.layers, backbone.adapters)):
        di = backbone.layer_device_indices[index]
        x = x.to(backbone.devices[di], dtype=tower.layer_dtype(layer))
        starts = torch.tensor(state.lengths, device=x.device)
        positions = starts[:, None] + torch.cat([torch.arange(n, device=x.device) for n in lengths])[None]
        rope = backbone.rotaries[di](x, positions[None].expand(3, -1, -1))
        mask = None
        if layer.block_type == 'full_attention':
            prefix_allowed = torch.arange(past, device=x.device)[None, :] < starts[:, None]
            levels = torch.repeat_interleave(torch.arange(len(lengths), device=x.device),
                                            torch.tensor(lengths, device=x.device))
            same = levels[:, None] == levels[None, :]
            allowed = torch.cat((prefix_allowed[:, None, :].expand(-1, x.shape[1], -1),
                                 same[None].expand(x.shape[0], -1, -1)), -1)
            mask = torch.zeros_like(allowed[:, None], dtype=x.dtype).masked_fill_(~allowed[:, None], float('-inf'))

        def step(h, layer=layer, adapter=adapter, index=index, rope=rope, mask=mask):
            normalized = layer.input_layernorm(h)
            if layer.block_type == 'linear_attention':
                attention = recurrence(layer.linear_attn, normalized, state.cache.layers[index], lengths)
            else:
                cache = fork_layer(state.cache, index)
                attention = layer.self_attn(normalized, position_embeddings=rope,
                                            attention_mask=mask, past_key_values=cache)[0]
            h = h + attention
            h = h + layer.mlp(layer.post_attention_layernorm(h))
            return adapter(h)

        x = checkpoint(step, x, use_reentrant=False) if backbone.training and torch.is_grad_enabled() else step(x)
    return backbone.norm(x).float()


def forward_levels(model, inputs, prompts, state=None):
    if not model.parallel_attention or len(inputs) != len(model.cfg.levels):
        raise ValueError('Expected parallel concept levels')
    state = multi.prepare(model.backbone, prompts) if state is None else state
    if len(prompts) != len(state.prompts) or any(not torch.equal(a.cpu(), b) for a, b in zip(prompts, state.prompts)):
        raise ValueError('Prompt batch differs')
    queries = []
    for level, (inp, size) in enumerate(zip(inputs, model.cfg.levels)):
        if inp.shape[:2] != (len(prompts), size):
            raise ValueError('Invalid input shape')
        x = model.project(inp.to(model.project.weight.device)) + model.level.weight[level] + model.position[:size]
        if model.concept_input_norm is not None:
            x = F.normalize(x, dim=-1, eps=1e-6) * model.concept_input_norm
        queries.append(x)
    h = hidden(model.backbone, torch.cat(queries, 1), state, model.cfg.levels)
    outputs, offset = [], 0
    for level, size in enumerate(model.cfg.levels):
        part = h[:, offset:offset + size]
        logits, latent = model.head(part, level, return_latent=True)
        outputs.append((logits, model.stop_head(part.mean(1)).squeeze(-1), latent))
        offset += size
    return outputs
