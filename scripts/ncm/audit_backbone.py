"""Pre-adaptation AR teacher-forced probes against existing llama.cpp fixtures.

The original vocabulary head is used ONLY for this transfer audit, not NCM
generation. Head rows are dequantized in bounded chunks, never kept as a second
full resident model. This diagnostic is not an inference speed benchmark.
"""
import json
from pathlib import Path
import numpy as np
import torch
from gguf.quants import dequantize
from .backbone import digest


@torch.no_grad()
def audit(backbone, source, reference_path):
    ref = json.loads(Path(reference_path).read_text())
    if ref['source_sha256'] != source.sha:
        raise ValueError('AR reference has a different source checksum')
    if any(torch.count_nonzero(a.up.weight) for a in backbone.adapters):
        raise ValueError('Audit requires unadapted backbone')
    was_training = backbone.training
    backbone.eval()
    head = source.tensors['output.weight']
    results = []
    for row in ref['prompts']:
        prompt = row['ids']
        expected = row['expected']
        ids = torch.tensor([prompt + expected[:-1]])
        h = backbone(backbone.embed_prompt(ids))[0, len(prompt)-1:].to(torch.bfloat16)
        logits = torch.empty(len(expected), 248320, dtype=torch.float32)
        for start in range(0, 248320, 2048):
            raw = dequantize(head.data[start:start+2048], head.tensor_type)
            weights = torch.from_numpy(np.array(raw, copy=True)).to(h.device, dtype=h.dtype)
            logits[:, start:start+weights.shape[0]] = (h @ weights.T).float().cpu()
            del weights, raw
        actual = logits.argmax(-1).tolist()
        results.append({'prompt':row['prompt'], 'expected':expected, 'actual':actual,
            'matches':sum(a==b for a,b in zip(actual,expected)), 'positions':len(expected),
            'passed':actual==expected})
    backbone.train(was_training)
    return {'reference_sha256':digest(reference_path), 'reference_commit':ref['reference_commit'],
        'source_sha256':source.sha, 'results':results, 'passed':all(r['passed'] for r in results),
        'scope':'full-backbone teacher-forced greedy probes on two short AR fixtures; not broad quality parity',
        'precision':'BF16 matrices dequantized from Q4_K_M; FP32 norms/decays'}
