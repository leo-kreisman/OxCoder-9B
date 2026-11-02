"""Standalone inference and unchanged executable evaluation for the v25 experiment.

The original OxCoder backbone encodes the user prompt and previously generated
chunks. A separate trained NCM decoder predicts all codes within each scale in
parallel. The original AR output head is never used for answer generation.
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer

from .backbone import Source, digest
from .codec import Codec, CodecConfig
from .context_decoder import ContextConceptDecoder, DecoderConfig
from .gguf_training_backbone import PackedOxBackbone
from .train import admission, write_json


class ContextRuntime(nn.Module):
    def __init__(self, backbone, decoder):
        super().__init__()
        self.backbone, self.decoder = backbone, decoder

    @torch.no_grad()
    def generate_chunk(self, codec, ids):
        hidden = self.backbone(self.backbone.embed_prompt(ids))
        device = self.decoder.position.device
        hidden = hidden.to(device)
        valid = torch.ones(hidden.shape[:2], dtype=torch.bool, device=device)
        # The same precision policy is used for validation and standalone use.
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            return self.decoder.generate_chunk(codec, hidden, valid)


def load(source_path, codec_path, checkpoint_path, device='cuda:0', backbone=None):
    checksum = digest(checkpoint_path)
    state = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    if digest(checkpoint_path) != checksum:
        raise ValueError('Decoder checkpoint changed during loading')
    identity = state['identity']
    if identity.get('architecture') != 'frozen-oxcoder-prompt-encoder-native-ncm-decoder-v25':
        raise ValueError('Not an encoder-decoder NCM checkpoint')
    if digest(codec_path) != identity['codec_sha256']:
        raise ValueError('Frozen codec checksum differs')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    c = torch.load(codec_path, weights_only=True, map_location='cpu')
    codec = Codec(CodecConfig(**c['config'])).to(device)
    codec.load_state_dict(c['model']); codec.eval().requires_grad_(False)
    decoder = ContextConceptDecoder(DecoderConfig(**identity['config'])).to(device)
    decoder.load_state_dict(state['model'], strict=True); decoder.eval().requires_grad_(False)
    if backbone is None:
        source = Source(source_path)
        if source.sha != identity['source_sha256']:
            raise ValueError('Frozen OxCoder source checksum differs')
        backbone = PackedOxBackbone(source, [device], rank=1).eval().requires_grad_(False)
    if backbone.source.sha != identity['source_sha256'] or any(torch.count_nonzero(a.up.weight) for a in backbone.adapters):
        raise ValueError('Encoder weights differ from the unadapted source')
    runtime = ContextRuntime(backbone, decoder).eval().requires_grad_(False)
    return runtime, codec, identity, checksum


def evaluate(a):
    from .coding_eval import evaluate_model
    torch.set_num_threads(2)
    admission([a.device], 8)
    runtime, codec, identity, checksum = load(a.source, a.codec, a.checkpoint, a.device)
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=False)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists(): raise FileExistsError(out)
    report = evaluate_model(runtime, codec, tokenizer, identity['prompt_tokens'], checksum, out, 8)
    report['architecture'] = identity['architecture']
    report['source_sha256'] = identity['source_sha256']
    report['codec_sha256'] = identity['codec_sha256']
    report['complete'] = report['evaluated'] == 12
    report['passed_all'] = report['complete'] and report['passed'] == 12
    report['quality_ready'] = False  # cold replay and wider validation remain separate
    write_json(out, report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'codec', 'checkpoint', 'output'): p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--tokenizer', default='results/tokenizer')
    evaluate(p.parse_args())
