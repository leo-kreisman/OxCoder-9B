"""Verify the FP16 frozen prompt path against existing llama.cpp token fixtures.

The vocabulary head is used only for this read-only transfer audit. It is not
attached to native NCM inference, training losses or executable acceptance.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from gguf.quants import dequantize
from .backbone import digest
from .gguf_training_backbone import load_candidate
from . import frozen_tower as tower
from .train import admission, write_json


@torch.no_grad()
def audit(backbone, reference):
    if reference['source_sha256'] != backbone.source.sha:
        raise ValueError('Reference source checksum differs')
    before = tower.source_versions(backbone)
    head = backbone.source.tensors['output.weight']
    records = []
    for row in reference['prompts']:
        ids = torch.tensor([row['ids'] + row['expected'][:-1]])
        captured = []
        handle = backbone.layers[-1].register_forward_hook(lambda module, args, output: captured.append(output.detach()))
        try:
            state = tower.prepare(backbone, ids, alignment=1)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError('Expected one final-layer prefill output')
        h = backbone.norm(captured[0])[0, len(row['ids']) - 1:]
        logits = torch.empty(len(row['expected']), 248320, dtype=torch.float32)
        for start in range(0, 248320, 2048):
            raw = dequantize(head.data[start:start+2048], head.tensor_type)
            weight = torch.from_numpy(np.array(raw, copy=True)).to(h.device, dtype=h.dtype)
            logits[:, start:start+weight.shape[0]] = (h @ weight.T).float().cpu()
            del raw, weight
        actual = logits.argmax(-1).tolist()
        records.append({'prompt': row['prompt'], 'expected': row['expected'], 'actual': actual,
                        'positions': len(actual), 'matches': sum(a == b for a, b in zip(actual, row['expected'])),
                        'passed': actual == row['expected']})
        del state, captured, h, logits
    if tower.source_versions(backbone) != before:
        raise RuntimeError('Source changed during audit')
    return records


def run(a):
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    admission([a.device], 9)
    model, codec, identity, sha = load_candidate(a.source, a.codec, a.checkpoint, [a.device])
    tower.set_compute_dtype(model.backbone, torch.float16)
    model.eval()
    reference = json.loads(Path(a.reference).read_text())
    records = audit(model.backbone, reference)
    report = {'complete': True, 'passed': all(r['passed'] for r in records), 'quality_ready': False,
              'checkpoint_sha256': sha, 'source_sha256': identity['source_sha256'],
              'reference_sha256': digest(a.reference), 'reference_commit': reference['reference_commit'],
              'records': records, 'compute_dtype': 'float16',
              'scope': 'Source-only teacher-forced token parity; no NCM coding qualification or AR fallback',
              'implementation_sha256': {n: digest(Path(__file__).with_name(n)) for n in
                  ('audit_frozen_source.py', 'frozen_tower.py', 'gguf_training_backbone.py', 'gguf_training_weights.py')}}
    write_json(output, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'codec', 'checkpoint', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--reference', default='scripts/kindi_q4/kindi/reference-oxcoder-q4.json')
    p.add_argument('--device', default='cuda:1')
    run(p.parse_args())
