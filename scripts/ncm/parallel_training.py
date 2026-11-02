"""Explicit summed-gradient data parallelism for variable NCM level graphs.

Workers may use different numbers of backward calls (four independent levels
or one shared-prefix graph). Collectives run only after all local examples.
Each example already contributes 1/global_batch; SUM is therefore required.
Frozen GGUF buffers are never broadcast or included in gradient collectives.
"""
import hashlib

import torch
import torch.distributed as dist

from .codec import encode, resize
from .shared_training import forward_levels
from .train_accumulated import example_loss
from .training_runtime import execution


def assignments(lengths, workers):
    """Balance a fixed global batch without changing its contents or weights."""
    if workers < 1 or len(lengths) < workers or any(n < 1 for n in lengths):
        raise ValueError('Every worker needs at least one positive-length example')
    loads, groups = [0] * workers, [[] for _ in range(workers)]
    for index in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        rank = min(range(workers), key=lambda r: (loads[r], r))
        groups[rank].append(index)
        loads[rank] += lengths[index]
    return [sorted(group) for group in groups]


def example_seed(seed, step, index):
    raw = f'ncm-parallel-v23:{seed}:{step}:{index}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63 - 1)


@torch.no_grad()
def inputs_for(codec, codes, maximum, seed):
    if not 0 <= maximum <= 1:
        raise ValueError('Corruption probability is outside [0, 1]')
    generator = torch.Generator(device=codes[0].device).manual_seed(seed)
    z = torch.zeros(codes[0].shape[0], codec.config.length, codec.config.width, device=codes[0].device)
    inputs = []
    for level, (ids, size, book) in enumerate(zip(codes, codec.config.levels, codec.books)):
        inputs.append(codec.resample(z, size))
        probability = maximum * (1 - level / max(1, len(codes) - 1))
        chosen = ids
        if probability:
            mask = torch.rand(ids.shape, device=ids.device, generator=generator) < probability
            replacements = torch.randint(book.num_embeddings, ids.shape, device=ids.device, generator=generator)
            chosen = torch.where(mask, replacements, ids)
        z = z + resize(book(chosen), codec.config.length)
    return inputs


def backward_example(model, codec, item, global_batch, corruption, seed, shared_limit=512):
    row, condition = item
    device = codec.position.device
    target = torch.tensor([encode(row['target'], codec.config.length)], device=device)
    with torch.no_grad():
        codes, _ = codec.targets(target)
        inputs = inputs_for(codec, codes, corruption, seed)
    prompt = torch.tensor([condition['ids']])
    loss_value, accuracy = 0., []
    shared = 64 <= prompt.shape[1] <= shared_limit
    with execution(model, 0, 'reference'):
        if shared:
            values = forward_levels(model, inputs, prompt)
        joint = 0
        for level, inp in enumerate(inputs):
            logits, end, _ = values[level] if shared else model(inp, prompt, level, return_aux=True)
            loss = example_loss(logits, codes[level], end, condition['is_final_chunk'],
                                level, len(codes), global_batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss_value += float(loss.detach())
            accuracy.append(float((logits.argmax(-1) == codes[level]).float().mean()) / global_batch)
            # Audit callers explicitly request logits; normal training needs no
            # CPU copy of this large tensor. The caller may install a head hook.
            if shared:
                joint = joint + loss
            else:
                loss.backward()
        if shared:
            joint.backward()
    return {'loss': loss_value, 'code_accuracy': accuracy, 'shared_prefix': shared}


def sum_gradients(named_parameters, bucket_bytes=16 << 20):
    """Bounded FP32 transport, SUM, then copy into the forward gradient dtype."""
    if not dist.is_initialized() or bucket_bytes < 4:
        raise ValueError('Initialized process group and positive bucket size required')
    parameters = [(name, p) for name, p in named_parameters if p.requires_grad]
    # Exchange validity first: a worker must not exit while another enters a
    # different tensor collective and hangs waiting for a missing gradient.
    device = parameters[0][1].device
    valid = all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in parameters)
    flag = torch.tensor(int(valid), device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise FloatingPointError('At least one worker has a missing or nonfinite gradient')
    # Same named inventory is checked once by the audit/trainer before this loop.
    for _, parameter in parameters:
        flat = parameter.grad.view(-1)
        for offset in range(0, flat.numel(), bucket_bytes // 4):
            part = flat[offset:offset + bucket_bytes // 4]
            buffer = part.float().clone()
            dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
            part.copy_(buffer)


def check_inventory(named_parameters):
    inventory = [(name, list(p.shape), str(p.dtype)) for name, p in named_parameters if p.requires_grad]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, inventory)
    if not inventory or any(value != inventory for value in gathered):
        raise ValueError('Parallel trainable parameter inventories differ')
