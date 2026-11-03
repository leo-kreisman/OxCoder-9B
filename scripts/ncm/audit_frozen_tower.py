"""Qualify cached frozen-context execution against its uncached control.

Both paths bypass prompt adapters and retain all source layers for concept
generation. This does not compare quality with the old adapted-prompt model.
It does not authorize training unless every numerical gate passes.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch
from torch.nn import functional as F

from . import frozen_tower as tower
from .audit_training_runtime import compare
from .audit_packed_training import compare_groups
from .backbone import digest
from .codec import encode
from .gguf_training_backbone import load_candidate
from .train import admission, write_json


def run_pass(model, codec, inputs, codes, prompt, final, device, cached, repeats):
    times, outputs = [], {}
    for repeat in range(repeats + 1):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.monotonic()
        state = tower.prepare(model.backbone, prompt) if cached else None
        loss_total = 0.
        for level, inp in enumerate(inputs):
            logits, ending, latent = tower.level_output(model, inp, prompt, level, cached, state)
            loss = F.cross_entropy(logits.flatten(0, 1), codes[level].flatten()) / len(inputs)
            if level == len(inputs) - 1:
                loss = loss + .1 * F.binary_cross_entropy_with_logits(ending, torch.full_like(ending, float(final)))
            loss.backward()
            loss_total += float(loss.detach())
            outputs[f'{level}:codes'] = logits.detach().cpu()
            outputs[f'{level}:stop'] = ending.detach().cpu()
            del logits, ending, latent, loss
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - start
        if repeat:
            times.append(elapsed)
        peak = torch.cuda.max_memory_allocated(device)
        del state
    gradients = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()
                 if p.requires_grad and p.grad is not None}
    if set(gradients) != {n for n, p in model.named_parameters() if p.requires_grad}:
        raise RuntimeError('Missing trainable gradient')
    return {'outputs': outputs, 'gradients': gradients, 'seconds': times,
            'median_seconds': statistics.median(times), 'peak_bytes': peak, 'loss': loss_total}


def run(a):
    paused = []
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    report = {'complete': False, 'quality_ready': False, 'records': [],
              'scope': 'Frozen original-source context and original-source concept generator; numerical audit only',
              'thresholds': {'logit_relative_l2': .02, 'gradient_relative_l2': .05, 'gradient_cosine_minimum': .99},
              'checkpoint_sha256': digest(a.checkpoint), 'source_sha256': digest(a.source),
              'codec_sha256': digest(a.codec), 'selection_sha256': digest(Path(a.selection) / 'selection.pt'),
              'implementation_sha256': {n: digest(Path(__file__).parent / n) for n in
                  ('frozen_tower.py', 'audit_frozen_tower.py', 'prefix.py', 'backbone.py',
                   'gguf_training_weights.py', 'gguf_training_backbone.py')},
              'pause_units': a.pause_unit, 'autoregressive_answer_fallback': False}
    write_json(out / 'report.json', report)
    try:
        # The caller must pause the evaluation coordinator before launching,
        # and wait for its current evaluator to finish. These are only the
        # explicitly named experiment services, never unrelated GPU jobs.
        for unit in a.pause_unit:
            if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode == 0:
                subprocess.run(['systemctl', '--user', 'kill', '--signal=STOP', unit], check=True)
                paused.append(unit)
        torch.set_num_threads(2)
        torch.manual_seed(29001)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        report['admission'] = admission([a.device], 9)
        model, codec, identity, sha = load_candidate(a.source, a.codec, a.checkpoint, [a.device])
        model.train()
        data = torch.load(Path(a.selection) / 'selection.pt', weights_only=True, map_location='cpu')['prepared']
        training = [(r, c) for r, c in data if r['split'] == 'train']
        for requested in (64, 69, 512, 2048):
            row, condition = min(training, key=lambda x: abs(len(x[1]['ids']) - requested))
            target = torch.tensor([encode(row['target'], codec.config.length)], device=a.device)
            with torch.no_grad():
                codes, inputs = codec.targets(target)
            prompt = torch.tensor([condition['ids']])
            ref = run_pass(model, codec, inputs, codes, prompt, condition['is_final_chunk'], a.device, False, a.repeats)
            actual = run_pass(model, codec, inputs, codes, prompt, condition['is_final_chunk'], a.device, True, a.repeats)
            ge, le = compare(actual['gradients'], ref['gradients']), compare(actual['outputs'], ref['outputs'])
            groups = compare_groups(actual['gradients'], ref['gradients'])
            record = {'example_id': row['id'], 'requested_prompt_tokens': requested, 'prompt_tokens': prompt.shape[1],
                      'gradient_comparison': ge, 'logit_comparison': le, 'gradient_groups': groups,
                      'reference': {k: v for k, v in ref.items() if k not in ('outputs', 'gradients')},
                      'cached': {k: v for k, v in actual.items() if k not in ('outputs', 'gradients')},
                      'speed_ratio': ref['median_seconds'] / actual['median_seconds']}
            record['passed'] = (ge['finite'] and le['finite'] and ge['relative_l2'] < .05 and ge['cosine'] > .99
                                and le['relative_l2'] < .02
                                and all(g['finite'] and g['relative_l2'] < .05 and g['cosine'] > .99 for g in groups.values()))
            report['records'].append(record)
            write_json(out / 'report.json', report)
            print(json.dumps(record), flush=True)
            del ref, actual
        report['complete'] = True
        report['passed'] = all(r['passed'] for r in report['records'])
    except BaseException as error:
        report['error'] = repr(error)
        report['passed'] = False
        raise
    finally:
        write_json(out / 'report.json', report)
        for unit in reversed(paused):
            subprocess.run(['systemctl', '--user', 'kill', '--signal=CONT', unit], check=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'codec', 'checkpoint', 'selection', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--pause-unit', action='append', default=[])
    a = p.parse_args()
    if a.repeats < 1:
        p.error('Positive repeats required')
    run(a)
