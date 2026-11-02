"""Training-only byte loss through a frozen codec and discrete concept predictions.

The forward uses the same hard final-level codes as generation. A softmax
straight-through estimator supplies a surrogate gradient to their logits.
Targets are loss labels only. No byte predictor is added to inference.
"""
import torch
from torch.nn import functional as F
from .codec import PAD


def decoded_loss(codec, previous, logits, targets, temperature=1.):
    if not temperature > 0 or previous.shape[1] != codec.config.length:
        raise ValueError('Positive temperature and full-resolution previous latents required')
    if any(parameter.requires_grad for parameter in codec.parameters()):
        raise ValueError('Decoded supervision requires a frozen codec')
    if logits.shape[:2] != targets.shape or logits.shape[-1] != codec.config.codes:
        raise ValueError('Final-level labels do not match concept logits')
    with torch.autocast(logits.device.type, enabled=False):
        weights = codec.books[-1].weight.float()
        hard = F.embedding(logits.argmax(-1), weights)
        soft = (logits.float() / temperature).softmax(-1) @ weights
        # Exact hard forward value, differentiable soft backward. Parentheses
        # avoid cancellation changing the hard forward rounding.
        selected = hard.detach() + (soft - soft.detach())
        output = codec.decode_latents(previous.detach().float() + selected)
        loss = F.cross_entropy(output.flatten(0,1), targets.flatten(), ignore_index=PAD)
    return loss, output
