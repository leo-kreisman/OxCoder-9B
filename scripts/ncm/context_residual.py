"""On-policy earlier levels with target-only adaptive residual supervision.

For a self-conditioned trajectory, all model inputs depend only on the prompt
and its own earlier predicted codes. Target text is used to choose correction
labels, never to construct that trajectory's inputs. Clean teacher-forced
trajectories remain in the training mixture. Inference is unchanged.
"""
import torch

from .codec import resize


@torch.no_grad()
def conditioned_targets(model, codec, hidden, valid, target_bytes, probability=.5):
    if not 0 <= probability <= 1:
        raise ValueError('Invalid on-policy trajectory probability')
    batch = hidden.shape[0]
    if probability == 0:
        self_conditioned = torch.zeros(batch, dtype=torch.bool, device=hidden.device)
    elif probability == 1:
        self_conditioned = torch.ones(batch, dtype=torch.bool, device=hidden.device)
    else:
        self_conditioned = torch.rand(batch, device=hidden.device) < probability
    # FP32 matches the frozen codec's original label construction. This tensor
    # is used only below to define supervised labels for the current level.
    with torch.autocast(hidden.device.type, enabled=False):
        target_latents = codec.encode_latents(target_bytes)
    z = torch.zeros_like(target_latents)
    with torch.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=hidden.device.type == 'cuda'):
        memory, mask = model.prepare_memory(hidden, valid)
    inputs, targets = [], []
    for level, (book, size) in enumerate(zip(codec.books, codec.config.levels)):
        inp = codec.resample(z, size)
        inputs.append(inp)
        with torch.autocast(hidden.device.type, enabled=False):
            residual = codec.resample(target_latents - z, size).float()
            flat, weights = residual.flatten(0, 1), book.weight.float()
            distances = flat.square().sum(-1, keepdim=True) + weights.square().sum(-1) - 2 * flat @ weights.T
            label = distances.argmin(-1).reshape(batch, size)
        targets.append(label)
        if level + 1 < len(codec.books):
            if self_conditioned.any():
                with torch.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=hidden.device.type == 'cuda'):
                    logits, _ = model.forward_level(inp, level, memory, mask)
                selected = torch.where(self_conditioned[:, None], logits.argmax(-1), label)
            else:
                selected = label
            z = z + resize(book(selected), codec.config.length)
    return inputs, targets, self_conditioned
