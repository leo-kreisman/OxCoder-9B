"""Qualified unpadded gradient pass, retaining v30 on-policy inference exactly."""
import torch
from torch.nn import functional as F
from .codec import encode
from .frozen_multi_objective import trajectory
from . import frozen_tower_ragged as multi


def backward_batch(model, codec, items, loss_scale, self_conditioned):
    if not items or any(row['split'] != 'train' for row, _ in items):
        raise ValueError('Only nonempty training batches may receive updates')
    prompts = [torch.tensor([condition['ids']]) for _, condition in items]
    target = torch.tensor([encode(row['target'], codec.config.length) for row, _ in items],
                          device=codec.position.device)
    model.eval()
    rollout = trajectory(model, codec, prompts, target, self_conditioned)
    model.train()
    outputs = multi.forward_levels(model, rollout['inputs'], prompts, rollout['state'])
    batch, levels = len(items), len(outputs)
    finals = torch.tensor([float(condition['is_final_chunk']) for _, condition in items], device=target.device)
    loss = 0
    per_example = torch.zeros(batch, device=target.device)
    accuracies = []
    for level, (logits, stop, _) in enumerate(outputs):
        labels = rollout['labels'][level]
        row_loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none').reshape(batch, -1).mean(1) / levels
        if level == levels - 1:
            row_loss = row_loss + .1 * F.binary_cross_entropy_with_logits(stop, finals, reduction='none')
        loss = loss + row_loss.mean()
        per_example += row_loss.detach() / batch
        accuracies.append((logits.argmax(-1) == labels).float().mean(1).detach())
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite concept loss')
    (loss * loss_scale).backward()
    return [{'loss': float(per_example[i]), 'self_conditioned': bool(self_conditioned[i]),
             'code_accuracy': [float(a[i]) for a in accuracies]} for i in range(batch)]
