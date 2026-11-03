"""Target-isolated residual trajectories for the original-backbone generator.

On-policy inputs follow the student's own concept decisions. Target bytes
only determine residual supervision. There is no teacher call at inference.
"""
import torch

from .codec import resize
from . import frozen_tower as tower


@torch.no_grad()
def trajectory(model, codec, prompt, state=None, target_bytes=None,
               self_conditioned=True):
    if not self_conditioned and target_bytes is None:
        raise ValueError('Teacher forcing requires training targets')
    state = tower.prepare(model.backbone, prompt) if state is None else state
    z = torch.zeros(1, codec.config.length, codec.config.width,
                    device=codec.position.device)
    target_latent = codec.encode_latents(target_bytes) if target_bytes is not None else None
    inputs, labels, decisions = [], [], []
    ending = None
    for level, (book, size) in enumerate(zip(codec.books, codec.config.levels)):
        inp = codec.resample(z, size)
        inputs.append(inp)
        label = None
        if target_latent is not None:
            residual = codec.resample(target_latent - z, size).float()
            flat, weight = residual.flatten(0, 1), book.weight.float()
            distance = flat.square().sum(-1, keepdim=True) + weight.square().sum(-1) - 2 * flat @ weight.T
            label = distance.argmin(-1).reshape(1, size)
            labels.append(label)
        if self_conditioned:
            logits, ending, _ = tower.level_output(model, inp, prompt, level, True, state)
            selected = logits.argmax(-1)
        else:
            selected = label
        decisions.append(selected)
        z = z + resize(book(selected), codec.config.length)
    return {'inputs': inputs, 'labels': labels, 'decisions': decisions,
            'latent': z, 'stop_logits': ending, 'state': state}


class GenerationAdapter:
    """Expose the unchanged coding evaluator interface for this architecture."""
    def __init__(self, model):
        self.model = model
        self.backbone = model.backbone

    @torch.no_grad()
    def generate_chunk(self, codec, prompt_ids):
        result = trajectory(self.model, codec, prompt_ids)
        return (codec.decode_latents(result['latent']).argmax(-1),
                result['decisions'], result['stop_logits'].sigmoid())
