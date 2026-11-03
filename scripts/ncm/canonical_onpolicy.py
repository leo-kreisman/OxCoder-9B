"""Experimental student-conditioned inputs with unchanged canonical code labels.

This differs from adaptive residual supervision: target code IDs never depend
on student mistakes. Student rollouts receive prompts only, no target bytes.
No active trainer imports this objective until a separate qualification stage.
"""
import torch
from torch.nn import functional as F

from .codec import encode
from .frozen_clean_objective import supervision
from .frozen_multi_objective import trajectory
from . import frozen_tower_multi as prefix
from . import frozen_tower_ragged as engine


@torch.no_grad()
def conditioning(model, codec, prompts, target, flags, corruption, state):
    if len(flags) != len(prompts) or any(type(flag) is not bool for flag in flags):
        raise ValueError('One explicit Boolean conditioning flag per example required')
    labels, teacher_inputs = supervision(codec, target, corruption)
    if not any(flags):
        return labels, teacher_inputs
    # No labels or target bytes are passed to the student trajectory.
    rollout = trajectory(model, codec, prompts, state=state)
    mask = torch.tensor(flags, dtype=torch.bool, device=target.device)[:, None, None]
    inputs = [torch.where(mask, student, teacher) for student, teacher in zip(rollout['inputs'], teacher_inputs)]
    return labels, inputs


def backward_batch(model, codec, items, loss_scale, corruption, flags):
    if not items or any(row['split'] != 'train' for row, _ in items):
        raise ValueError('Only nonempty training batches may receive updates')
    prompts = [torch.tensor([condition['ids']]) for _, condition in items]
    target = torch.tensor([encode(row['target'], codec.config.length) for row, _ in items],
                          device=codec.position.device)
    model.eval()
    state = prefix.prepare(model.backbone, prompts)
    labels, inputs = conditioning(model, codec, prompts, target, flags, corruption, state)
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
        raise FloatingPointError('Nonfinite canonical student-conditioned loss')
    (loss * loss_scale).backward()
    return [{'loss': float(per_example[i]), 'conditioning_corruption': corruption,
             'student_conditioned': flags[i], 'labels': 'unchanged canonical codec IDs',
             'code_accuracy': [float(a[i]) for a in accuracies]} for i in range(batch)]
