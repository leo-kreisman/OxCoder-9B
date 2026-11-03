"""Training-only hard-hierarchy byte supervision for student-conditioned rows.

Every level uses hard argmax codes in the forward pass. Softmax expectations
supply a straight-through surrogate gradient, not an inference algorithm.
Targets enter only the loss. The codec, labels and generation path stay fixed.
"""
import math
from functools import lru_cache
import torch
from torch.nn import functional as F
from .codec import PAD, resize, encode
from .canonical_onpolicy import conditioning
from . import frozen_tower_multi as prefix
from . import frozen_tower_ragged as engine


@lru_cache(maxsize=16)
def _interpolation_matrix(source_length, destination_length):
    # Evaluate the linear operator on its basis. Keep this immutable CPU tensor
    # private. CUDA's native linear-interpolation backward uses nondeterministic
    # accumulation; a small matrix product supplies the same linear derivative.
    return resize(torch.eye(source_length)[None], destination_length)[0]


def linear_surrogate(values, length):
    if values.shape[1] == length:
        return values
    matrix = _interpolation_matrix(values.shape[1], length).to(values)
    return torch.matmul(matrix, values)


def hierarchy_loss(codec, logits, targets, flags):
    if any(p.requires_grad for p in codec.parameters()):
        raise ValueError('The codec must remain frozen')
    if len(logits) != len(codec.books) or len(flags) != targets.shape[0]:
        raise ValueError('One prediction per level and flag per row required')
    if any(type(flag) is not bool for flag in flags) or not any(flags):
        raise ValueError('At least one explicitly student-conditioned row required')
    indices = torch.tensor([i for i, flag in enumerate(flags) if flag], device=targets.device)
    with torch.autocast(targets.device.type, enabled=False):
        z = torch.zeros(len(indices), codec.config.length, codec.config.width,
                        device=targets.device)
        for value, size, book in zip(logits, codec.config.levels, codec.books):
            if value.shape != (len(flags), size, codec.config.codes):
                raise ValueError('Hierarchy logit shape differs from codec')
            value = value.index_select(0, indices).float()
            weight = book.weight.float()
            hard = F.embedding(value.argmax(-1), weight)
            soft = value.softmax(-1) @ weight
            soft = linear_surrogate(soft, codec.config.length)
            # Keep the ordinary codec's exact forward interpolation, with a
            # deterministic matrix-product surrogate only for backward.
            selected = resize(hard.detach(), codec.config.length) + (soft - soft.detach())
            z = z + selected
        output = codec.decode_latents(z)
        labels = targets.index_select(0, indices)
        mask = labels != PAD
        if not bool(mask.any(1).all()):
            raise ValueError('Each target must contain at least one supervised symbol')
        ce = F.cross_entropy(output.flatten(0, 1), labels.flatten(),
                             reduction='none', ignore_index=PAD).reshape_as(labels)
        selected_loss = ce.sum(1) / mask.sum(1)
        # Divide over the complete training batch, not only flagged examples.
        per_row = torch.zeros(len(flags), device=targets.device).index_copy(0, indices, selected_loss)
    return per_row, output


def backward_batch(model, codec, items, loss_scale, corruption, flags, byte_weight=.1):
    if not math.isfinite(byte_weight) or byte_weight < 0:
        raise ValueError('Finite nonnegative byte weight required')
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
    finals = torch.tensor([float(c['is_final_chunk']) for _, c in items], device=target.device)
    loss = 0
    per_example = torch.zeros(batch, device=target.device)
    accuracies = []
    for level, (logits, stop, _) in enumerate(outputs):
        row_loss = F.cross_entropy(logits.flatten(0, 1), labels[level].flatten(),
            reduction='none').reshape(batch, -1).mean(1) / levels
        if level == levels - 1:
            row_loss = row_loss + .1 * F.binary_cross_entropy_with_logits(stop, finals, reduction='none')
        loss = loss + row_loss.mean()
        per_example += row_loss.detach() / batch
        accuracies.append((logits.argmax(-1) == labels[level]).float().mean(1).detach())
    byte_loss = torch.zeros(batch, device=target.device)
    if byte_weight and any(flags):
        byte_loss, _ = hierarchy_loss(codec, [value[0] for value in outputs], target, flags)
        loss = loss + byte_weight * byte_loss.mean()
        per_example += byte_weight * byte_loss.detach() / batch
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite decoded-hierarchy loss')
    (loss * loss_scale).backward()
    return [{'loss': float(per_example[i]), 'byte_ce': float(byte_loss[i].detach()),
             'byte_weight': byte_weight, 'student_conditioned': flags[i],
             'labels': 'unchanged canonical codec IDs',
             'code_accuracy': [float(a[i]) for a in accuracies]} for i in range(batch)]
