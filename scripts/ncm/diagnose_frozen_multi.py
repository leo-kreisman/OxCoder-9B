"""Separate loss, representation and prompt-conditioning probes for native NCM.

No optimizer is created. Oracle concepts appear only in explicitly labeled
diagnostic paths, never in executable coding evaluation or released generation.
"""
import argparse
import json
from pathlib import Path
import random
import torch
from torch.nn import functional as F
from .backbone import digest
from .gguf_training_backbone import load_candidate
from .codec import encode, decode, resize, PAD
from . import frozen_tower as tower
from . import frozen_tower_multi as multi
from .frozen_multi_objective import trajectory, level_output
from .frozen_multi_qualification import qualify
from .frozen_tower_training import load_corpus
from .train import admission, write_json


def byte_metrics(codec, z, target, text):
    ids = codec.decode_latents(z).argmax(-1)
    try:
        output = decode(ids[0].tolist())
    except (UnicodeError, ValueError):
        output = None
    mask = target != PAD
    return {'text': output, 'exact': output == text,
            'nonpad_byte_accuracy': float((ids == target)[mask].float().mean())}


@torch.no_grad()
def probe(model, codec, prompt, target, text):
    state = multi.prepare(model.backbone, [prompt])
    clean = trajectory(model, codec, [prompt], target, [False], state)
    outputs = multi.forward_levels(model, clean['inputs'], [prompt], state)
    clean_metrics = [{'ce': float(F.cross_entropy(out[0].flatten(0, 1), label.flatten())),
                      'code_accuracy': float((out[0].argmax(-1) == label).float().mean())}
                     for out, label in zip(outputs, clean['labels'])]
    free = trajectory(model, codec, [prompt], target, [True], state)
    last = len(codec.books) - 1
    logits, _, _ = level_output(model, clean['inputs'][last], state, last)
    oracle_prefix = torch.zeros_like(clean['latent'])
    for book, ids in zip(codec.books[:-1], clean['decisions'][:-1]):
        oracle_prefix += resize(book(ids), codec.config.length)
    oracle_final = oracle_prefix + resize(codec.books[-1](logits.argmax(-1)), codec.config.length)
    return {'clean_teacher_forced': clean_metrics,
            'all_oracle_codec': byte_metrics(codec, clean['latent'], target, text),
            'free_generation': byte_metrics(codec, free['latent'], target, text),
            'oracle_earlier_levels_student_final': byte_metrics(codec, oracle_final, target, text),
            'free_stop_probability': float(free['stop_logits'].sigmoid()[0])}


def run(a):
    torch.set_num_threads(2)
    torch.manual_seed(31001)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(output)
    admission([a.device], 9)
    model, codec, identity, sha = load_candidate(a.source, a.codec, a.checkpoint, [a.device])
    qualify(a.execution_audit, identity['source_sha256'], identity['codec_sha256'])
    if identity.get('architecture') != 'frozen-context-original-oxcoder-native-ncm-v30':
        raise ValueError('Requires a v30 checkpoint')
    tower.set_compute_dtype(model.backbone, torch.float16)
    model.eval()
    training, heldout, corpus = load_corpus(a.selection, a.manifest, identity['source_sha256'], identity['codec_sha256'])
    rng = random.Random(31001)
    selected = rng.sample(training, min(a.examples, len(training))) + rng.sample(heldout, min(a.examples, len(heldout)))
    report = {'complete': False, 'quality_ready': False, 'checkpoint_sha256': sha,
              'corpus': corpus, 'records': [], 'optimizer_created': False,
              'scope': 'Training/validation diagnostics; oracle results are not autonomous generation',
              'implementation_sha256': digest(__file__)}
    before = tower.source_versions(model.backbone)
    for row, condition in selected:
        target = torch.tensor([encode(row['target'], codec.config.length)], device=codec.position.device)
        prompt = torch.tensor([condition['ids']])
        result = probe(model, codec, prompt, target, row['target'])
        result.update(id=row['id'], split=row['split'], source=row['source'],
                      is_final_chunk=condition['is_final_chunk'])
        report['records'].append(result)
        write_json(output, report)
        print(json.dumps({'id': row['id'], 'split': row['split'],
                          'free_exact': result['free_generation']['exact']}), flush=True)
    if before != tower.source_versions(model.backbone) or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Read-only diagnostic mutated source or accumulated gradients')
    report['complete'] = True
    write_json(output, report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'codec', 'checkpoint', 'execution-audit', 'selection', 'manifest', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--examples', type=int, default=16)
    p.add_argument('--device', default='cuda:1')
    run(p.parse_args())
