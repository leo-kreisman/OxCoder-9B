"""Measure independent prompt prefill reuse without retaining model updates."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from .backbone import digest
from .gguf_training_backbone import load_candidate
from . import frozen_tower as tower
from . import frozen_tower_multi as reference
from . import interleaved_prefix as candidate
from .train import admission, headroom, write_json


def state_tensors(state):
    values = {}
    for i, layer in enumerate(state.cache.layers):
        for name in ('keys', 'values', 'conv_states', 'recurrent_states'):
            if hasattr(layer, name):
                item = getattr(layer, name)
                if isinstance(item, dict):
                    values.update({f'{i}.{name}.{k}': v.detach().cpu() for k, v in item.items()})
                else:
                    values[f'{i}.{name}'] = item.detach().cpu()
    return values


def measure(backbone, prompts, function, repeats, device):
    times, peaks = [], []
    values = None
    for iteration in range(repeats + 1):
        headroom([device])
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.monotonic()
        state = function(backbone, prompts)
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - start
        if iteration:
            times.append(elapsed)
            peaks.append(torch.cuda.max_memory_allocated(device))
        if iteration == repeats:
            values = state_tensors(state)
        del state
    return {'seconds': times, 'median_seconds': statistics.median(times),
            'peak_allocated_bytes': max(peaks)}, values


def run(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    report = {'complete': False, 'passed': False, 'quality_ready': False, 'records': [],
              'scope': 'Frozen prompt execution only; no training update or model quality claim'}
    paused = []

    def active(unit):
        return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode == 0

    def pause(unit):
        if active(unit):
            subprocess.run(['systemctl', '--user', 'kill', '--kill-who=main', '--signal=STOP', unit], check=True)
            paused.append(unit)

    def wait(unit):
        while active(unit):
            report['phase'] = 'waiting_for_' + unit
            write_json(out / 'report.json', report)
            time.sleep(10)

    try:
        if a.wait_for_session:
            pause('ncm-compact-curriculum-pipeline-v33.service')
            wait('ncm-compact-tower-worker-v33.service')
            # The worker's exit resumes the older evaluation coordinator.
            pause('ncm-complete-pipeline-v28.service')
            wait('ncm-context-evaluation-v28.service')
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report['admission'] = admission([a.device], 9)
        model, codec, identity, checksum = load_candidate(a.source, a.codec, a.checkpoint, [a.device])
        tower.set_compute_dtype(model.backbone, torch.float16)
        model.eval()
        report.update(phase='auditing', checkpoint_sha256=checksum,
                      source_sha256=identity['source_sha256'], codec_sha256=identity['codec_sha256'],
                      implementation_sha256={n: digest(Path(__file__).with_name(n)) for n in
                          ('audit_interleaved_prefix.py', 'interleaved_prefix.py',
                           'frozen_tower_multi.py', 'frozen_tower.py', 'gguf_training_weights.py')})
        prepared = torch.load(Path(a.selection) / 'selection.pt', weights_only=True, map_location='cpu')['prepared']
        training = [(r, c) for r, c in prepared if r['split'] == 'train']
        source_versions = tower.source_versions(model.backbone)
        for lengths in ((64, 69, 128, 256), (64, 512, 1024, 2048), (256,)):
            prompts, examples = [], []
            for size in lengths:
                row, condition = min(training, key=lambda item: abs(len(item[1]['ids']) - size))
                ids = condition['ids']
                tokens = (ids * ((size + len(ids) - 1) // len(ids)))[:size]
                prompts.append(torch.tensor([tokens]))
                examples.append({'id': row['id'], 'original_tokens': len(ids), 'prompt_tokens': size,
                                 'synthetic_extension_or_truncation': len(ids) != size})
            control, left = measure(model.backbone, prompts, reference.prepare, a.repeats, a.device)
            actual, right = measure(model.backbone, prompts, candidate.prepare, a.repeats, a.device)
            equal = left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)
            finite = all(torch.isfinite(t).all().item() for t in right.values())
            record = {'examples': examples, 'reference': control, 'interleaved': actual,
                      'all_prefix_tensors_bit_exact': equal, 'finite': finite,
                      'source_unchanged': tower.source_versions(model.backbone) == source_versions,
                      'speed_ratio': control['median_seconds'] / actual['median_seconds']}
            record['passed'] = equal and finite and record['source_unchanged']
            report['records'].append(record)
            write_json(out / 'report.json', report)
            print(json.dumps(record), flush=True)
            del left, right
        report['complete'] = True
        report['passed'] = all(r['passed'] for r in report['records'])
        report['speed_qualified'] = report['passed'] and all(r['speed_ratio'] > 1 for r in report['records'][:2])
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        write_json(out / 'report.json', report)
        for unit in reversed(paused):
            subprocess.run(['systemctl', '--user', 'kill', '--kill-who=main', '--signal=CONT', unit], check=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'codec', 'checkpoint', 'selection', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--wait-for-session', action='store_true')
    a = p.parse_args()
    if a.repeats < 1:
        p.error('repeats must be positive')
    run(a)
