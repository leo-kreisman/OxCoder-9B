"""Byte-preserving input representation and learned multiscale residual VQ codec.

Independent implementation of the multiscale residual quantization idea in
https://arxiv.org/abs/2610.14162. It is NOT upstream checkpoint compatible.
"""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F

EOS, PAD, VOCAB = 256, 257, 258


def encode(text, length):
    raw = list(text.encode('utf-8'))
    if len(raw) >= length:
        raise ValueError('Text exceeds byte window; prepare an explicit chunk first')
    return raw + [EOS] + [PAD] * (length - len(raw) - 1)


def decode(ids):
    out = []
    for x in ids:
        x = int(x)
        if x == EOS:
            return bytes(out).decode('utf-8', errors='strict')
        if not 0 <= x < 256:
            raise ValueError('PAD or invalid byte before EOS')
        out.append(x)
    raise ValueError('Missing EOS')


def resize(x, length):
    return F.interpolate(x.transpose(1, 2), size=length, mode='linear', align_corners=False).transpose(1, 2)


@dataclass
class CodecConfig:
    length: int = 128
    width: int = 128
    heads: int = 4
    layers: int = 2
    codes: int = 512
    levels: tuple = (1, 4, 16, 64, 128)
    downsample: str = 'linear'
    coarse_loss_weight: float = 0.

    def __post_init__(self):
        self.levels = tuple(self.levels)
        if (not self.levels or self.levels[-1] != self.length
                or any(a >= b for a, b in zip(self.levels, self.levels[1:]))
                or min(self.levels) < 1 or self.width % self.heads or self.layers < 0):
            raise ValueError('Invalid multiscale codec configuration')
        if self.downsample not in ('linear','area') or not 0 <= self.coarse_loss_weight <= 1:
            raise ValueError('Invalid coarse representation configuration')
        if self.coarse_loss_weight and len(self.levels)<2:
            raise ValueError('Coarse supervision requires multiple levels')


class Codec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(VOCAB, config.width)
        self.position = nn.Parameter(torch.randn(config.length, config.width) * .02)
        def stack():
            if config.layers == 0:
                return nn.Identity()
            layer = nn.TransformerEncoderLayer(config.width, config.heads, config.width * 4,
                dropout=0., batch_first=True, norm_first=True, activation='gelu')
            return nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.encoder = stack()
        self.encoder_norm = nn.LayerNorm(config.width)
        self.decoder = stack()
        self.head = nn.Linear(config.width, VOCAB)
        self.books = nn.ModuleList([nn.Embedding(config.codes, config.width) for _ in config.levels])
        for book in self.books:
            nn.init.normal_(book.weight, std=.1)

    def resample(self,x,length):
        if self.config.downsample=='area' and length<x.shape[1]:
            return F.interpolate(x.transpose(1,2),size=length,mode='area').transpose(1,2)
        return resize(x,length)

    def quantize(self, z):
        """Teacher targets depend on text; inputs contain ONLY preceding levels."""
        cumulative = torch.zeros_like(z)
        ids, inputs, losses = [], [], []
        for size, book in zip(self.config.levels, self.books):
            inputs.append(self.resample(cumulative, size).detach())
            residual = self.resample(z - cumulative.detach(), size)
            flat = residual.reshape(-1, z.shape[-1]).float()
            weights = book.weight.float()
            distance = flat.square().sum(-1, keepdim=True) + weights.square().sum(-1) - 2 * flat @ weights.T
            index = distance.argmin(-1).view(z.shape[0], size)
            q = book(index)
            losses.append(F.mse_loss(q, residual.detach()) + .25 * F.mse_loss(residual, q.detach()))
            cumulative = cumulative + resize(q, self.config.length)
            ids.append(index)
        return cumulative, ids, inputs, torch.stack(losses).mean()

    def encode_latents(self, ids):
        if ids.shape[1] != self.config.length:
            raise ValueError('Wrong codec window')
        return self.encoder_norm(self.encoder(self.embedding(ids) + self.position))

    def forward(self, ids):
        z = self.encode_latents(ids)
        q, codes, inputs, vq_loss = self.quantize(z)
        # Straight-through gradient reaches encoder; codebook uses explicit VQ loss.
        straight = z + (q - z).detach()
        logits = self.decode_latents(straight)
        # PAD is supervised too: the decoder must learn a valid termination pattern.
        loss = F.cross_entropy(logits.flatten(0, 1), ids.flatten()) + vq_loss
        if self.training and self.config.coarse_loss_weight:
            # Retain full-resolution reconstruction on every update. A second
            # reconstruction from a random strict prefix encourages earlier
            # levels to encode context. This auxiliary objective differs from
            # upstream's stochastic replacement of the decoder input.
            keep=int(torch.randint(1,len(codes),(),device=ids.device))
            partial=torch.zeros_like(z)
            for book,index in zip(self.books[:keep],codes[:keep]):
                partial=partial+resize(book(index),self.config.length)
            partial_straight=z+(partial-z).detach()
            coarse_logits=self.decode_latents(partial_straight)
            loss=loss+self.config.coarse_loss_weight*F.cross_entropy(coarse_logits.flatten(0,1),ids.flatten())
        return loss, logits, codes, inputs

    def decode_latents(self, z):
        return self.head(self.decoder(z + self.position))

    @torch.no_grad()
    def targets(self, ids):
        _, codes, inputs, _ = self.quantize(self.encode_latents(ids))
        return codes, inputs

    @torch.no_grad()
    def reconstruct_codes(self, codes):
        z = torch.zeros(codes[0].shape[0], self.config.length, self.config.width,
                        device=self.position.device)
        for size, book, ids in zip(self.config.levels, self.books, codes):
            if ids.shape[1] != size:
                raise ValueError('Wrong code level size')
            z += resize(book(ids), self.config.length)
        return self.decode_latents(z)

    def metadata(self):
        return asdict(self.config)
