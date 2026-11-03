"""Experimental input-only autograd for immutable GGUF linear weights.

Save packed bytes instead of retaining each temporary dense weight for backward.
This allows evaluation of training without whole-layer activation recomputation.
No quantization arithmetic changes. Full-model qualification is still required.
"""
from contextlib import contextmanager

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from .gguf_training_weights import FrozenGGUFLinear, unpack
from . import frozen_tower_ragged as engine


class PackedInputGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, shape, tensor_type, source_name):
        if packed.requires_grad or packed.dtype != torch.uint8:
            raise ValueError('Only immutable packed byte weights are supported')
        ctx.save_for_backward(packed)
        ctx.shape, ctx.tensor_type, ctx.source_name = shape, tensor_type, source_name
        ctx.dtype = x.dtype
        weight = unpack(packed, shape, tensor_type, source_name, dtype=x.dtype)
        return F.linear(x, weight)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        packed, = ctx.saved_tensors
        weight = unpack(packed, ctx.shape, ctx.tensor_type, ctx.source_name, dtype=ctx.dtype)
        result = torch.matmul(gradient, weight)
        return result, None, None, None, None


@contextmanager
def installed(backbone, *, recompute_layers=False):
    """Instance-local experiment; serialize calls and keep scope through backward.

    Prefix-only layer caches may temporarily override these forwards and restore
    them afterward. Global engine checkpoint dispatch is changed only for this
    process and restored on errors. Higher-order gradients are unsupported.
    """
    previous_checkpoint = engine.checkpoint
    if getattr(engine, '_packed_gradient_scope', False):
        raise RuntimeError('Nested packed-gradient scopes are unsupported')
    missing, replaced = object(), []
    engine._packed_gradient_scope = True
    try:
        for module in backbone.modules():
            if not isinstance(module, FrozenGGUFLinear):
                continue
            if getattr(module, '_ncm_unpack_scope', False):
                raise RuntimeError('Enter the packed-gradient scope outside prefix caching')
            original = module.__dict__.get('forward', missing)

            def forward(x, module=module):
                return PackedInputGradient.apply(x, module.packed, module.shape,
                                                 module.tensor_type, module.source_name)

            module.forward = forward
            replaced.append((module, original))
        if not recompute_layers:
            def direct(function, x, *, use_reentrant):
                if use_reentrant:
                    raise ValueError('Unexpected checkpoint policy')
                return function(x)
            engine.checkpoint = direct
        yield
    finally:
        engine.checkpoint = previous_checkpoint
        del engine._packed_gradient_scope
        for module, original in reversed(replaced):
            if original is missing:
                del module.forward
            else:
                module.forward = original
