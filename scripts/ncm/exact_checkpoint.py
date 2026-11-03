"""Restore saved trainable precision after a loader's construction dtype cast."""
from pathlib import Path

import torch

from .backbone import digest


@torch.no_grad()
def restore_trainable(model, path, identity, expected_sha):
    """Call after setting the destination execution dtype, before inference.

    The historical backbone constructor creates BF16 adapters. Loading an FP16
    checkpoint into those adapters rounds values before the caller changes the
    execution dtype. Recopy the original saved values into the final dtype.
    Training's optimizer-state restore already does this, but cold inference
    and standalone diagnostics need it explicitly too.
    """
    path = Path(path)
    if digest(path) != expected_sha:
        raise ValueError('Checkpoint identity changed')
    state = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
    if state['identity'] != identity:
        raise ValueError('Checkpoint metadata changed')
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(parameters) != set(state['model']):
        raise ValueError('Trainable checkpoint inventory changed')
    for name, parameter in parameters.items():
        saved = state['model'][name]
        if saved.shape != parameter.shape or saved.dtype != parameter.dtype or not torch.isfinite(saved).all():
            raise ValueError('Set the exact saved trainable dtype before restoration: ' + name)
    for name, parameter in parameters.items():
        parameter.copy_(state['model'][name].to(parameter.device))
        if not torch.equal(parameter, state['model'][name].to(parameter.device)):
            raise RuntimeError('Saved trainable values were not restored exactly: ' + name)
    if digest(path) != expected_sha:
        raise ValueError('Checkpoint changed during restoration')
