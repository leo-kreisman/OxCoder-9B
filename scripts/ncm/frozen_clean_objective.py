"""Canonical NCM code labels with corrupted earlier-code conditioning.

Training-only alternative to adaptive residual labels. The frozen-prompt source,
codec, gradient engine and autonomous inference algorithm remain unchanged.
"""
import torch
from torch.nn import functional as F
from .codec import encode
from .train_independent import corrupted_inputs
from . import frozen_tower_multi as prefix
from . import frozen_tower_ragged as engine


@torch.no_grad()
def supervision(codec, target, corruption):
    # Preserve the established per-example nearest-code arithmetic.
    encoded = [codec.targets(row[None])[0] for row in target]
    labels = [torch.cat([row[level] for row in encoded]) for level in range(len(codec.books))]
    inputs = corrupted_inputs(codec, labels, corruption)
    return labels, inputs


def backward_batch(model, codec, items, loss_scale, corruption):
    if not items or any(row['split'] != 'train' for row, _ in items):
        raise ValueError('Only nonempty training batches may receive updates')
    prompts = [torch.tensor([condition['ids']]) for _, condition in items]
    target = torch.tensor([encode(row['target'], codec.config.length) for row, _ in items],
                          device=codec.position.device)
    labels, inputs = supervision(codec, target, corruption)
    model.eval()
    state = prefix.prepare(model.backbone, prompts)
    model.train()
    outputs = engine.forward_levels(model, inputs, prompts, state)
    batch, levels = len(items), len(outputs)
    finals = torch.tensor([float(condition['is_final_chunk']) for _, condition in items], device=target.device)
    loss = 0
    per_example = torch.zeros(batch, device=target.device)
    accuracies = []
    for level, (logits, stop, _) in enumerate(outputs):
        row_loss = F.cross_entropy(logits.flatten(0, 1), labels[level].flatten(), reduction='none').reshape(batch, -1).mean(1) / levels
        if level == levels - 1:
            row_loss = row_loss + .1 * F.binary_cross_entropy_with_logits(stop, finals, reduction='none')
        loss = loss + row_loss.mean()
        per_example += row_loss.detach() / batch
        accuracies.append((logits.argmax(-1) == labels[level]).float().mean(1).detach())
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite canonical code loss')
    (loss * loss_scale).backward()
    return [{'loss': float(per_example[i]), 'conditioning_corruption': corruption,
             'code_accuracy': [float(a[i]) for a in accuracies]} for i in range(batch)]
