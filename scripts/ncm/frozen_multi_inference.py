"""Native NCM generation with the same full-prefix path as v30 training."""
import torch
from .frozen_multi_objective import trajectory


class GenerationAdapter:
    def __init__(self, model):
        self.model = model
        self.backbone = model.backbone

    @torch.no_grad()
    def generate_chunk(self, codec, prompt_ids):
        result = trajectory(self.model, codec, [prompt_ids])
        return (codec.decode_latents(result['latent']).argmax(-1),
                result['decisions'], result['stop_logits'].sigmoid())
