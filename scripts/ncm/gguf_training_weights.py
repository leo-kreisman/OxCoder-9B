"""Frozen Q4_K/Q6_K weights, unpacked exactly for differentiable BF16 GEMMs.

No requantization or straight-through weight estimator is used. The source
GGUF bytes remain immutable. Gradients flow through the input of F.linear;
layer checkpointing releases unpacked weights between forward/backward calls.
This module is experimental until full-model parity and memory audits pass.

Quantization layout follows gguf-py's Q4_K/Q6_K dequantizers and the existing
Kindi Q4 kernel in this repository (AGPL-3.0-only).
SPDX-License-Identifier: AGPL-3.0-only
"""
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def unpack_kernel(packed, output, N: tl.constexpr, C: tl.constexpr,
                  TYPE: tl.constexpr, TRANSFORM: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // C, index % C
    if TRANSFORM == 1:  # qkv: preserve q/k, invert value-head permutation
        offset = row - 4096
        head, inner = offset // 128, offset % 128
        converted = 4096 + ((head % 2) * 16 + head // 2) * 128 + inner
        row = tl.where(row >= 4096, converted, row)
    elif TRANSFORM == 2:  # gated value rows
        head, inner = row // 128, row % 128
        row = ((head % 2) * 16 + head // 2) * 128 + inner
    elif TRANSFORM == 3:  # output-projection columns
        head, inner = col // 128, col % 128
        col = ((head % 2) * 16 + head // 2) * 128 + inner
    elif TRANSFORM == 4:  # per-head alpha/beta rows
        row = (row % 2) * 16 + row // 2
    source = row * C + col
    block, j = source // 256, source % 256
    valid = index < N
    if TYPE == 12:
        base = block * 144
        lo = tl.load(packed + base, valid, other=0).to(tl.uint16)
        hi = tl.load(packed + base + 1, valid, other=0).to(tl.uint16)
        d = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        lo = tl.load(packed + base + 2, valid, other=0).to(tl.uint16)
        hi = tl.load(packed + base + 3, valid, other=0).to(tl.uint16)
        dmin = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        group, sub = j // 32, j % 32
        low = tl.load(packed + base + 4 + group, valid, other=0).to(tl.int32)
        high = tl.load(packed + base + 4 + group + 4, valid, other=0).to(tl.int32)
        previous = tl.load(packed + base + 4 + tl.maximum(group - 4, 0), valid, other=0).to(tl.int32)
        scale = tl.where(group < 4, low & 63, (high & 15) | ((previous >> 6) << 4))
        minimum = tl.where(group < 4, high & 63, (high >> 4) | ((low >> 6) << 4))
        q = tl.load(packed + base + 16 + (group // 2) * 32 + sub, valid, other=0).to(tl.int32)
        q = (q >> ((group % 2) * 4)) & 15
        value = (d * scale.to(tl.float32)) * q.to(tl.float32) - dmin * minimum.to(tl.float32)
    else:
        base = block * 210
        half, group, sub = j // 128, (j % 128) // 32, j % 32
        low = tl.load(packed + base + half * 64 + (group % 2) * 32 + sub, valid, other=0).to(tl.int32)
        low = (low >> ((group // 2) * 4)) & 15
        high = tl.load(packed + base + 128 + half * 32 + sub, valid, other=0).to(tl.int32)
        high = (high >> (2 * group)) & 3
        scale = tl.load(packed + base + 192 + half * 8 + group * 2 + sub // 16, valid, other=0)
        scale = scale.to(tl.int8, bitcast=True).to(tl.float32)
        lo = tl.load(packed + base + 208, valid, other=0).to(tl.uint16)
        hi = tl.load(packed + base + 209, valid, other=0).to(tl.uint16)
        d = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        value = (d * scale) * ((low | (high << 4)) - 32).to(tl.float32)
    tl.store(output + index, value, valid)


TRANSFORMS = {'linear_attn.in_proj_qkv.weight': 1,
              'linear_attn.in_proj_z.weight': 2,
              'linear_attn.out_proj.weight': 3,
              'linear_attn.in_proj_a.weight': 4,
              'linear_attn.in_proj_b.weight': 4}


def unpack(packed, shape, tensor_type, name='', dtype=torch.bfloat16):
    if tensor_type not in (12, 14) or len(shape) != 2 or shape[1] % 256:
        raise ValueError('Only aligned Q4_K/Q6_K matrices are supported')
    expected = shape[0] * shape[1] // 256 * (144 if tensor_type == 12 else 210)
    if packed.dtype != torch.uint8 or packed.numel() != expected or not packed.is_contiguous() or not packed.is_cuda:
        raise ValueError('Invalid packed tensor layout/device')
    output = torch.empty(shape, device=packed.device, dtype=dtype)
    # Triton launches on the current CUDA device rather than inferring it from
    # tensor pointers. Checkpoint replay can enter here from another device.
    with torch.cuda.device(packed.device):
        unpack_kernel[(triton.cdiv(output.numel(), 1024),)](
            packed, output, output.numel(), shape[1], tensor_type, TRANSFORMS.get(name, 0), 1024,
            enable_fp_fusion=False)
    return output


class FrozenGGUFLinear(nn.Module):
    def __init__(self, packed, shape, tensor_type, name):
        super().__init__()
        self.register_buffer('packed', packed, persistent=False)
        self.shape, self.tensor_type, self.source_name = tuple(shape), tensor_type, name
        self.in_features, self.out_features = shape[1], shape[0]

    def forward(self, x):
        weight = unpack(self.packed, self.shape, self.tensor_type, self.source_name, dtype=x.dtype)
        return torch.nn.functional.linear(x, weight)
