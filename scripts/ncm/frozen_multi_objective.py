"""Variable-prompt NCM training with per-example on-policy decisions.

Training targets affect residual labels, and teacher-forced rows only. They
cannot affect the rollout inputs of rows marked self-conditioned. Prefixes
are detached frozen-source states, shared by rollout and gradient evaluation.
"""
import torch
from torch.nn import functional as F
from .codec import encode, resize
from . import frozen_tower_multi as multi


def level_output(model, inp, state, level):
    size = model.cfg.levels[level]
    x = model.project(inp.to(model.project.weight.device)) + model.level.weight[level] + model.position[:size]
    if model.concept_input_norm is not None:
        x = F.normalize(x, dim=-1, eps=1e-6) * model.concept_input_norm
    h = multi.hidden(model.backbone, x, state, (size,)).float()
    logits, latent = model.head(h, level, return_latent=True)
    return logits, model.stop_head(h.mean(1)).squeeze(-1), latent


@torch.no_grad()
def trajectory(model, codec, prompts, target_bytes=None, self_conditioned=None, state=None):
    batch = len(prompts)
    flags = [True] * batch if self_conditioned is None else self_conditioned
    if len(flags) != batch or (not all(flags) and target_bytes is None):
        raise ValueError('One conditioning flag per row; teacher forcing requires targets')
    if target_bytes is not None and target_bytes.shape[0] != batch:
        raise ValueError('One target per prompt required')
    state = multi.prepare(model.backbone, prompts) if state is None else state
    z = torch.zeros(batch, codec.config.length, codec.config.width, device=codec.position.device)
    # Keep the established single-example codec encoding, avoiding batch-shape
    # numerical changes at nearest-code boundaries.
    target_latent = (torch.cat([codec.encode_latents(row[None]) for row in target_bytes])
                     if target_bytes is not None else None)
    select_student = torch.tensor(flags, device=z.device)[:, None]
    inputs, labels, decisions = [], [], []
    ending = None
    for level, (book, size) in enumerate(zip(codec.books, codec.config.levels)):
        inp = codec.resample(z, size)
        inputs.append(inp)
        label = None
        if target_latent is not None:
            residual = codec.resample(target_latent - z, size).float()
            # Identical matrix shapes to single-example residual assignment.
            weight = book.weight.float()
            label = torch.stack([(r.square().sum(-1, keepdim=True) + weight.square().sum(-1)
                                  - 2 * r @ weight.T).argmin(-1) for r in residual])
            labels.append(label)
        if any(flags):
            logits, ending, _ = level_output(model, inp, state, level)
            student = logits.argmax(-1)
            selected = student if label is None else torch.where(select_student, student, label)
        else:
            selected = label
        decisions.append(selected)
        z = z + resize(book(selected), codec.config.length)
    return {'inputs': inputs, 'labels': labels, 'decisions': decisions,
            'latent': z, 'stop_logits': ending, 'state': state}


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
