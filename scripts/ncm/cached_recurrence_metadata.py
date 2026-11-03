"""Reuse immutable variable-length boundary tensors across source layers.

The reference builds and synchronously uploads the same boundaries in every
recurrent layer, including backward recomputation. This experiment caches only
those small index tensors. Model weights, activations and equations are unchanged.
"""
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
import torch
from torch.nn import functional as F
from transformers.activations import ACT2FN
from . import frozen_tower_ragged as engine

REFERENCE = engine.recurrence


class Boundaries:
    def __init__(self, capacity=8):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Positive metadata capacity required')
        self.capacity = capacity
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def get(self, batch, lengths, device):
        lengths = tuple(lengths)
        if (type(batch) is not int or batch < 1 or not lengths
                or any(type(n) is not int or n < 1 for n in lengths)):
            raise ValueError('Positive batch and sequence lengths required')
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        key = (batch, lengths, str(device))
        if key in self.entries:
            self.hits += 1
            entry = self.entries.pop(key)
        else:
            self.misses += 1
            cumulative = [0]
            for n in lengths * batch:
                cumulative.append(cumulative[-1] + n)
            cpu = torch.tensor(cumulative, dtype=torch.long)
            gpu = cpu.to(device)
            entry = (cpu, gpu, cpu._version, gpu._version)
        cpu, gpu, cv, gv = entry
        if cpu._version != cv or gpu._version != gv:
            raise RuntimeError('Boundary metadata was mutated')
        self.entries[key] = entry
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return cpu, gpu


def recurrence(module, h, prefix, lengths, *, boundaries):
    if not h.is_cuda:
        return REFERENCE(module, h, prefix, lengths)
    # Same recurrence equations and FLA arguments as frozen_tower_ragged.
    # Only construction/upload of cu_seqlens moves into the immutable cache.
    batch, total, _ = h.shape
    if sum(lengths) != total:
        raise ValueError('Packed lengths differ from input')
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
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    cpu_lengths, gpu_lengths = boundaries.get(batch, lengths, h.device)
    def flatten(x):
        return x.reshape(1, batch * total, *x.shape[2:])
    output, _ = chunk_gated_delta_rule(flatten(q), flatten(k), flatten(v),
        g=flatten(g), beta=flatten(beta),
        initial_state=prefix.recurrent_states[0].repeat_interleave(len(lengths), 0),
        output_final_state=False, use_qk_l2norm_in_kernel=True,
        cu_seqlens=gpu_lengths, cu_seqlens_cpu=cpu_lengths)
    output = output.reshape(batch, total, -1, module.head_v_dim)
    output = module.norm(output.reshape(-1, module.head_v_dim), z).reshape(batch, total, -1)
    return module.out_proj(output)


@contextmanager
def installed():
    if engine.recurrence is not REFERENCE:
        raise RuntimeError('Another recurrence override is installed')
    boundaries = Boundaries()
    engine.recurrence = partial(recurrence, boundaries=boundaries)
    try:
        yield boundaries
    finally:
        engine.recurrence = REFERENCE
