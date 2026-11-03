"""Qualify variable-prompt batching against uncached and qualified v29 execution."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time
import torch
from torch.nn import functional as F

from .backbone import digest
from .gguf_training_backbone import load_candidate
from . import frozen_tower as tower
from . import frozen_tower_batch as levels
from . import frozen_tower_multi as multi
from .codec import encode
from .audit_training_runtime import compare
from .audit_packed_training import compare_groups
from .train import admission, write_json


def measure(model, inputs, labels, prompts, finals, mode, repeats, device):
    seconds, outputs = [], {}
    batch = len(prompts)
    for repeat in range(repeats + 1):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.monotonic()
        total = 0.
        if mode == 'multi':
            state = multi.prepare(model.backbone, prompts)
            values = multi.forward_levels(model, inputs, prompts, state)
            loss = 0
            for level, (logits, ending, _) in enumerate(values):
                loss = loss + F.cross_entropy(logits.flatten(0, 1), labels[level].flatten()) / len(inputs)
                if level == len(inputs) - 1:
                    loss = loss + .1 * F.binary_cross_entropy_with_logits(ending, finals)
                for i in range(batch):
                    outputs[f'{i}:{level}:codes'] = logits[i:i+1].detach().cpu()
                    outputs[f'{i}:{level}:stop'] = ending[i:i+1].detach().cpu()
            (loss * 64).backward()
            total = float(loss.detach())
            del loss, values, state, logits, ending
        else:
            for i, prompt in enumerate(prompts):
                inp = [x[i:i+1] for x in inputs]
                state = tower.prepare(model.backbone, prompt) if mode == 'v29' else None
                values = levels.forward_levels(model, inp, prompt, state) if mode == 'v29' else None
                joint = 0
                for level in range(len(inputs)):
                    logits, ending, _ = (values[level] if values is not None else
                                        tower.level_output(model, inp[level], prompt, level, False))
                    loss = F.cross_entropy(logits.flatten(0, 1), labels[level][i:i+1].flatten()) / (batch * len(inputs))
                    if level == len(inputs) - 1:
                        loss = loss + .1 / batch * F.binary_cross_entropy_with_logits(ending, finals[i:i+1])
                    if values is None:
                        (loss * 64).backward()
                    else:
                        joint = joint + loss
                    total += float(loss.detach())
                    outputs[f'{i}:{level}:codes'] = logits.detach().cpu()
                    outputs[f'{i}:{level}:stop'] = ending.detach().cpu()
                    del logits, ending, loss
                if values is not None:
                    (joint * 64).backward()
                del state, values, joint
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - start
        if repeat:
            seconds.append(elapsed)
        peak = torch.cuda.max_memory_allocated(device)
    gradients = {n: p.grad.detach().float().cpu().clone() / 64 for n, p in model.named_parameters()
                 if p.requires_grad and p.grad is not None}
    if set(gradients) != {n for n, p in model.named_parameters() if p.requires_grad}:
        raise RuntimeError('Missing trainable gradient')
    return {'gradients': gradients, 'outputs': outputs, 'seconds': seconds,
            'median_seconds': statistics.median(seconds), 'peak_bytes': peak, 'loss': total}


def run(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    paused = []
    report = {'complete': False, 'passed': False, 'quality_ready': False, 'records': [],
              'scope': 'Execution and gradient audit only; no training weights retained',
              'compute_dtype': 'float16', 'loss_scale': 64, 'prefix_alignment': 1}
    def pause(unit):
        if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode == 0:
            subprocess.run(['systemctl', '--user', 'kill', '--signal=STOP', unit], check=True)
            paused.append(unit)
    def wait(unit):
        while subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode == 0:
            report['phase'] = 'waiting_for_' + unit
            write_json(out / 'report.json', report)
            time.sleep(10)
    try:
        if a.wait_for_session:
            pause('ncm-frozen-tower-pipeline-v29.service')
            wait('ncm-frozen-tower-training-v29.service')
            pause('ncm-complete-pipeline-v28.service')
            wait('ncm-context-evaluation-v28.service')
            pause('ncm-context-decoded-v28.service')
        torch.set_num_threads(2)
        torch.manual_seed(30001)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report['admission'] = admission([a.device], 9)
        model, codec, identity, checkpoint_sha = load_candidate(a.source, a.codec, a.checkpoint, [a.device])
        tower.set_compute_dtype(model.backbone, torch.float16)
        model.train()
        report.update(phase='auditing', checkpoint_sha256=checkpoint_sha,
            source_sha256=identity['source_sha256'], codec_sha256=identity['codec_sha256'],
            selection_sha256=digest(Path(a.selection) / 'selection.pt'),
            implementation_sha256={n: digest(Path(__file__).with_name(n)) for n in
                ('audit_frozen_multi.py', 'frozen_tower_multi.py', 'frozen_tower.py',
                 'frozen_tower_batch.py', 'frozen_tower_cache.py', 'gguf_training_weights.py',
                 'gguf_training_backbone.py', 'backbone.py', 'prefix.py')})
        data = torch.load(Path(a.selection) / 'selection.pt', weights_only=True, map_location='cpu')['prepared']
        training = [(r, c) for r, c in data if r['split'] == 'train']
        for sizes in ((64, 69, 128, 256), (64, 512, 1024, 2048)):
            items, prompts, observations = [], [], []
            for size in sizes:
                row, condition = min(training, key=lambda x: abs(len(x[1]['ids']) - size))
                ids = condition['ids']
                changed = len(ids) != size
                actual = (ids * ((size + len(ids) - 1) // len(ids)))[:size]
                items.append((row, condition)); prompts.append(torch.tensor([actual]))
                observations.append({'id': row['id'], 'prompt_tokens': size,
                    'original_tokens': len(ids), 'synthetic_extension_or_truncation': changed})
            with torch.no_grad():
                encoded = [codec.targets(torch.tensor([encode(r['target'], codec.config.length)], device=a.device)) for r, _ in items]
                labels = [torch.cat([c[level] for c, _ in encoded]) for level in range(len(codec.books))]
                inputs = [torch.cat([x[level] for _, x in encoded]) for level in range(len(codec.books))]
            finals = torch.tensor([float(c['is_final_chunk']) for _, c in items], device=a.device)
            reference = measure(model, inputs, labels, prompts, finals, 'reference', a.repeats, a.device)
            control = measure(model, inputs, labels, prompts, finals, 'v29', a.repeats, a.device)
            actual = measure(model, inputs, labels, prompts, finals, 'multi', a.repeats, a.device)
            ge, le = compare(actual['gradients'], reference['gradients']), compare(actual['outputs'], reference['outputs'])
            groups = compare_groups(actual['gradients'], reference['gradients'])
            record = {'examples': observations, 'gradient_comparison': ge, 'logit_comparison': le,
                'gradient_groups': groups, 'speed_ratio_over_uncached': reference['median_seconds'] / actual['median_seconds'],
                'speed_ratio_over_v29': control['median_seconds'] / actual['median_seconds'],
                'reference': {k: v for k, v in reference.items() if k not in ('gradients', 'outputs')},
                'v29': {k: v for k, v in control.items() if k not in ('gradients', 'outputs')},
                'multi': {k: v for k, v in actual.items() if k not in ('gradients', 'outputs')}}
            record['passed'] = (ge['finite'] and le['finite'] and ge['relative_l2'] < .05 and ge['cosine'] > .99
                and le['relative_l2'] < .02 and all(g['finite'] and g['relative_l2'] < .05 and g['cosine'] > .99 for g in groups.values()))
            report['records'].append(record)
            write_json(out / 'report.json', report)
            print(json.dumps(record), flush=True)
            del reference, control, actual
        report['complete'] = True
        report['passed'] = all(r['passed'] for r in report['records'])
        report['speed_qualified'] = report['passed'] and all(r['speed_ratio_over_v29'] > 1 for r in report['records'])
    except BaseException as error:
        report['error'] = repr(error)
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
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--wait-for-session', action='store_true')
    run(p.parse_args())
