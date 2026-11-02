"""Training-only causal-token to fixed-byte concept alignment.

This is an experimental bridge, not the same-position alignment from Repr-Align.
An AR hidden state at t-1 predicts token t. Pool those states over each concept's
byte interval, excluding padding. Targets never enter student inference.
"""
import torch
from torch.nn import functional as F


def alignment_map(text, target_start, offsets, window=128):
    """Return [window, token_count] averaging weights and a valid-byte mask.

    Offsets are tokenizer character offsets into the *joint* prefix and target.
    Byte fallback tokens may share character offsets. Each byte receives equal
    total mass, divided among the overlapping tokens. Empty/special offsets have
    no text mass. The end-of-chunk position uses the final input hidden state.
    """
    if not 0 <= target_start < len(text) or not offsets:
        raise ValueError('A nonempty target and tokenized context are required')
    boundaries = [0]
    for char in text:
        boundaries.append(boundaries[-1] + len(char.encode('utf-8')))
    begin, length = boundaries[target_start], boundaries[-1] - boundaries[target_start]
    if length >= window:
        raise ValueError('Target does not fit codec window with EOS')
    weights = torch.zeros(window, len(offsets), dtype=torch.float32)
    for token, (left, right) in enumerate(offsets):
        if not 0 <= left <= right <= len(text):
            raise ValueError('Invalid tokenizer character offsets')
        lo, hi = max(boundaries[left] - begin, 0), min(boundaries[right] - begin, length)
        if hi <= lo:
            continue
        if token == 0:
            raise ValueError('First target token has no preceding causal hidden state')
        weights[lo:hi, token - 1] += 1
    mass = weights[:length].sum(-1)
    if (mass == 0).any():
        raise ValueError('Tokenizer offsets leave uncovered target bytes')
    weights[:length] /= mass[:, None]
    weights[length, -1] = 1  # EOS supervision, not a generated target token.
    valid = torch.arange(window) <= length
    return weights, valid


def tokenize_alignment(tokenizer, prompt_ids, target, window=128):
    prefix = tokenizer.decode(prompt_ids, skip_special_tokens=False,
                              clean_up_tokenization_spaces=False)
    if tokenizer.encode(prefix, add_special_tokens=False) != list(prompt_ids):
        raise ValueError('Prompt cannot be reconstructed without changing tokens')
    text = prefix + target
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    weights, valid = alignment_map(text, len(prefix), encoded['offset_mapping'], window)
    return encoded['input_ids'], weights, valid


def pool_teacher(hidden, weights, valid, levels):
    """Map detached teacher states to each scale, with no PAD contribution."""
    if hidden.ndim != 2 or weights.ndim != 2 or hidden.shape[0] != weights.shape[1]:
        raise ValueError('Token-state and byte-map dimensions differ')
    if valid.shape != (weights.shape[0],) or any(weights.shape[0] % n for n in levels):
        raise ValueError('Scales must divide the fixed codec window')
    byte_states = weights.to(hidden.device) @ hidden.detach().float()
    valid = valid.to(hidden.device)
    targets, masks = [], []
    for size in levels:
        values = byte_states.reshape(size, -1, hidden.shape[-1]).sum(1)
        counts = valid.reshape(size, -1).sum(1)
        targets.append(values / counts.clamp_min(1)[:, None])
        masks.append(counts > 0)
    return targets, masks


def cosine_alignment(student, teacher, valid):
    if student.shape != teacher.shape or valid.shape != student.shape[:-1]:
        raise ValueError('Alignment shapes differ')
    if not valid.any():
        raise ValueError('Alignment has no nonpadding positions')
    loss = 1 - F.cosine_similarity(student.float(), teacher.detach().float(), dim=-1)
    return loss[valid].mean()
