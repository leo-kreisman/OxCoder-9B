"""Experimental native concept decoder conditioned on frozen OxCoder states.

This is an encoder-decoder architecture, distinct from the full-backbone
adapter generator. The source 9B model encodes prompts only; this decoder
predicts VQ levels in parallel. Neither component generates AR answer tokens.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DecoderConfig:
    source_width: int = 4096
    latent_width: int = 128
    width: int = 768
    heads: int = 12
    layers: int = 12
    codes: int = 4096
    levels: tuple = (4, 16, 64, 128)
    maximum_prompt: int = 2048

    def __post_init__(self):
        self.levels = tuple(self.levels)
        if (min(self.source_width, self.latent_width, self.width, self.heads,
                self.layers, self.codes, self.maximum_prompt) < 1
                or self.width % self.heads or self.width % 2
                or not self.levels or min(self.levels) < 1
                or any(a >= b for a, b in zip(self.levels, self.levels[1:]))):
            raise ValueError('Invalid native concept decoder configuration')


def positions(length, width, device):
    frequency = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32)
                          * (-math.log(10000.) / width))
    angles = torch.arange(length, device=device, dtype=torch.float32)[:, None] * frequency
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.dimension = heads, width // heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, queries, memory, allowed):
        def split(x):
            return x.reshape(x.shape[0], x.shape[1], self.heads, self.dimension).transpose(1, 2)
        query, key, value = split(self.q(queries)), split(self.k(memory)), split(self.v(memory))
        result = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed, dropout_p=0.)
        return self.out(result.transpose(1, 2).reshape_as(queries))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_norm = nn.LayerNorm(config.width)
        self.cross_norm = nn.LayerNorm(config.width)
        self.ffn_norm = nn.LayerNorm(config.width)
        self.self_attention = Attention(config.width, config.heads)
        self.cross_attention = Attention(config.width, config.heads)
        self.ffn = nn.Sequential(nn.Linear(config.width, 4 * config.width), nn.GELU(),
                                 nn.Linear(4 * config.width, config.width))

    def forward(self, x, memory, level_mask, memory_mask):
        normalized = self.self_norm(x)
        x = x + self.self_attention(normalized, normalized, level_mask)
        x = x + self.cross_attention(self.cross_norm(x), memory, memory_mask)
        return x + self.ffn(self.ffn_norm(x))


class ContextConceptDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.memory = nn.Sequential(nn.LayerNorm(config.source_width),
                                    nn.Linear(config.source_width, config.width), nn.LayerNorm(config.width))
        self.latent = nn.Sequential(nn.LayerNorm(config.latent_width),
                                    nn.Linear(config.latent_width, config.width))
        self.level = nn.Embedding(len(config.levels), config.width)
        self.position = nn.Parameter(torch.randn(config.levels[-1], config.width) * .02)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.heads = nn.ModuleList([nn.Linear(config.width, config.codes) for _ in config.levels])
        self.stop = nn.Linear(config.width, 1)
        self.checkpoint_blocks = False

    def prepare_memory(self, hidden, valid):
        if (hidden.ndim != 3 or hidden.shape[-1] != self.config.source_width
                or valid.shape != hidden.shape[:2] or valid.dtype != torch.bool
                or not 0 < hidden.shape[1] <= self.config.maximum_prompt
                or not bool(valid.any(-1).all()) or not bool(torch.isfinite(hidden).all())):
            raise ValueError('Invalid frozen prompt states or mask')
        memory = self.memory(hidden.to(self.memory[1].weight.dtype))
        memory = memory + positions(memory.shape[1], self.config.width, memory.device).to(memory)
        return memory, valid[:, None, None, :]

    def queries(self, inputs, level):
        if (inputs.ndim != 3 or inputs.shape[1:] !=
                (self.config.levels[level], self.config.latent_width)):
            raise ValueError('Invalid previous-concept input')
        return self.latent(inputs.to(self.latent[1].weight.dtype)) + self.level.weight[level] + self.position[:inputs.shape[1]]

    def decode_queries(self, queries, memory, level_mask, memory_mask):
        for block in self.blocks:
            if self.training and self.checkpoint_blocks and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                queries = checkpoint(block, queries, memory, level_mask, memory_mask, use_reentrant=False)
            else:
                queries = block(queries, memory, level_mask, memory_mask)
        return self.norm(queries)

    def forward(self, inputs, hidden, valid):
        if len(inputs) != len(self.config.levels):
            raise ValueError('Missing concept levels')
        memory, memory_mask = self.prepare_memory(hidden, valid)
        queries = torch.cat([self.queries(inp, i) for i, inp in enumerate(inputs)], dim=1)
        level_ids = torch.repeat_interleave(torch.arange(len(inputs), device=queries.device),
                                           torch.tensor(self.config.levels, device=queries.device))
        level_mask = (level_ids[:, None] == level_ids[None, :])[None, None]
        output = self.decode_queries(queries, memory, level_mask, memory_mask)
        parts = output.split(self.config.levels, dim=1)
        return ([head(part).float() for head, part in zip(self.heads, parts)],
                self.stop(parts[-1].mean(1)).squeeze(-1).float())

    def forward_level(self, inputs, level, memory, memory_mask):
        queries = self.queries(inputs, level)
        output = self.decode_queries(queries, memory, None, memory_mask)
        stop = self.stop(output.mean(1)).squeeze(-1).float() if level == len(self.heads)-1 else None
        return self.heads[level](output).float(), stop

    @torch.no_grad()
    def generate_chunk(self, codec, hidden, valid):
        from .codec import resize
        if (tuple(codec.config.levels) != self.config.levels or codec.config.width != self.config.latent_width
                or codec.config.codes != self.config.codes):
            raise ValueError('Frozen codec does not match decoder')
        memory, mask = self.prepare_memory(hidden, valid)
        z = torch.zeros(hidden.shape[0], codec.config.length, codec.config.width,
                        device=hidden.device, dtype=self.position.dtype)
        generated = []
        for level, book in enumerate(codec.books):
            logits, stop = self.forward_level(codec.resample(z, self.config.levels[level]), level, memory, mask)
            ids = logits.argmax(-1)
            z = z + resize(book(ids), codec.config.length)
            generated.append(ids)
        # The codec label audit uses FP32. Keep its reconstruction arithmetic
        # unchanged even when the concept generator runs under BF16 autocast.
        with torch.autocast(hidden.device.type, enabled=False):
            output = codec.decode_latents(z.float()).argmax(-1)
        return output, generated, stop.sigmoid()
