"""Load the same frozen OxCoder weights in compact GPU storage for training.

Physical placement is a runtime setting. The returned checkpoint identity is
unchanged, so saved adapter/concept-head weights remain reloadable through the
original BF16 reference path. This loader does not authorize a runtime change:
the caller must compare the whole model, gradients, and optimizer memory first.
SPDX-License-Identifier: AGPL-3.0-only
"""
import json

import numpy as np
import torch
from torch import nn

from .backbone import (Source, OxBackbone, Adapter, ConceptGenerator, config,
                       mapping, to_hf, digest, Qwen3_5DecoderLayer,
                       Qwen3_5TextRotaryEmbedding, Qwen3_5RMSNorm)
from .codec import Codec, CodecConfig
from .gguf_training_weights import FrozenGGUFLinear


class PackedOxBackbone(OxBackbone):
    def __init__(self, source, devices, rank, layer_split=None, progress=print):
        nn.Module.__init__(self)
        self.width, self.source = 4096, source
        self.devices = list(map(torch.device, devices))
        counts = layer_split or [32 // len(devices)] * len(devices)
        if len(counts) != len(devices) or sum(counts) != 32 or min(counts) < 1:
            raise ValueError('Physical placement must allocate all 32 layers')
        self.layer_device_indices = [index for index, count in enumerate(counts) for _ in range(count)]
        self.config = config()
        self.config._attn_implementation = 'sdpa'
        self.layers, self.adapters = nn.ModuleList(), nn.ModuleList()
        self.rotaries = nn.ModuleList([Qwen3_5TextRotaryEmbedding(self.config).to(d) for d in self.devices])
        self.transfers = []
        for index in range(32):
            device = self.devices[self.layer_device_indices[index]]
            with torch.device('meta'):
                layer = Qwen3_5DecoderLayer(self.config, index)
            for name, parameter in list(layer.named_parameters()):
                source_name = f'blk.{index}.' + mapping(name)
                tensor = source.tensors[source_name]
                module_name, attr = name.rsplit('.', 1)
                module = layer.get_submodule(module_name)
                kind = int(tensor.tensor_type)
                if isinstance(module, nn.Linear) and attr == 'weight' and kind in (12, 14):
                    shape = tuple(reversed(tensor.shape.tolist()))
                    if shape != tuple(parameter.shape) or module.bias is not None:
                        raise ValueError('Unexpected linear shape or bias')
                    packed = torch.from_numpy(np.array(tensor.data, dtype=np.uint8, copy=True).reshape(-1)).to(device)
                    parent_name, leaf = module_name.rsplit('.', 1)
                    setattr(layer.get_submodule(parent_name), leaf, FrozenGGUFLinear(packed, shape, kind, name))
                    record = {'source': source_name, 'storage': 'immutable GGUF bytes', 'type': kind,
                              'bytes': packed.numel(), 'parameters': parameter.numel(), 'device': str(device)}
                else:
                    value = to_hf(name, source.read(source_name))
                    if tuple(value.shape) != tuple(parameter.shape):
                        raise ValueError('Resident parameter shape differs')
                    dtype = torch.float32 if value.ndim == 1 else torch.bfloat16
                    value = value.to(device=device, dtype=dtype)
                    setattr(module, attr, nn.Parameter(value, requires_grad=False))
                    record = {'source': source_name, 'storage': str(dtype), 'bytes': value.numel() * value.element_size(),
                              'parameters': value.numel(), 'device': str(device)}
                    del value
                self.transfers.append(record)
            self.layers.append(layer)
            self.adapters.append(Adapter(4096, rank).to(device=device, dtype=torch.bfloat16))
            progress(json.dumps({'phase': 'load_packed_backbone', 'layer': index + 1, 'layers': 32}), flush=True)
        self.norm = Qwen3_5RMSNorm(4096, 1e-6).to(self.devices[-1])
        self.norm.weight = nn.Parameter((source.read('output_norm.weight').flatten() - 1).to(self.devices[-1]), requires_grad=False)
        self.source_identity = {'sha256': source.sha, 'path': source.path,
            'reused_parameters': sum(t['parameters'] for t in self.transfers) + 4096,
            'excluded': ['token_embd.weight', 'output.weight', 'vision components absent in GGUF'],
            'architecture': 'hybrid-causal-mixer-next-concept-v1',
            'weight_storage': 'GGUF Q4_K/Q6_K with exact BF16 unpacking'}


def load_candidate(source_path, codec_path, checkpoint_path, devices, layer_split=None):
    checkpoint_sha = digest(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    identity = checkpoint['identity']
    if digest(checkpoint_path) != checkpoint_sha:
        raise RuntimeError('Checkpoint changed during loading')
    if identity.get('full_precision_reductions'):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if digest(codec_path) != identity['codec_sha256']:
        raise ValueError('Codec identity differs')
    source = Source(source_path)
    if source.sha != identity['source_sha256']:
        raise ValueError('Source identity differs')
    ck = torch.load(codec_path, weights_only=True, map_location='cpu')
    codec = Codec(CodecConfig(**ck['config'])).to(devices[-1])
    codec.load_state_dict(ck['model'])
    codec.eval().requires_grad_(False)
    backbone = PackedOxBackbone(source, devices, identity['rank'], layer_split)
    model = ConceptGenerator(backbone, codec.config, parallel_attention='full_attention' in identity,
        termination='conditioning' in identity, concept_input_norm=identity.get('concept_input_norm'),
        cached_prefix=bool(identity.get('cached_prefix', False)),
        head_kind=identity.get('head_kind', 'linear'), codebooks=codec.books,
        head_input_norm=bool(identity.get('head_input_norm', False)),
        nearest_code=bool(identity.get('nearest_code', False)))
    if set(model.trainable_state()) != set(checkpoint['model']):
        raise ValueError('Trainable checkpoint inventory differs')
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(checkpoint['model'][name].to(parameter))
    model.eval()
    return model, codec, identity, checkpoint_sha
