"""Read-only full-shape CPU/GPU AdamW comparison from a real saved state.

Uses reproducible synthetic gradients, not model updates for training. No
weights from this audit may become a candidate checkpoint. Timings measure
only head optimizer work; full-training admission remains separate.
"""
import argparse
import copy
import json
from pathlib import Path
import statistics
import time

import torch

from .backbone import digest
from .cpu_adamw import CPUAdamW
from .train import admission, headroom, write_json


def difference(left, right):
    maximum, error, scale = 0., 0., 0.
    for a, b in zip(left, right):
        a, b = a.detach().float().cpu(), b.detach().float().cpu()
        d = a - b
        maximum = max(maximum, float(d.abs().max()))
        error += float(d.double().square().sum())
        scale += float(a.double().square().sum())
    return {'maximum_absolute': maximum, 'relative_l2': (error / max(scale, 1e-30)) ** .5}


def run(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    report = {'complete': False, 'passed': False, 'quality_ready': False, 'records': [],
              'scope': 'Real checkpoint head state; synthetic gradients; all updates discarded'}
    try:
        torch.set_num_threads(2)
        torch.manual_seed(34002)
        report['admission'] = admission([a.device], 3)
        checksum = digest(a.checkpoint)
        ck = torch.load(a.checkpoint, weights_only=True, map_location='cpu')
        if checksum != digest(a.checkpoint):
            raise ValueError('Checkpoint changed while loading')
        saved = ck['optimizer']
        names = [n for n in saved['names'] if n.startswith(('head.', 'stop_head.'))]
        source = [ck['model'][n] for n in names]
        if (not names or saved['format'] != 'frozen-tower-scaled-optimizer-v29'
                or saved['head']['format'] != 'cpu-adamw-v1'
                or saved['head']['shapes'] != [list(t.shape) for t in source]
                or any(t.dtype != torch.float32 for t in source)):
            raise ValueError('Unsupported optimizer layout')
        original = [torch.nn.Parameter(t.to(a.device, copy=True)) for t in source]
        alternative = [torch.nn.Parameter(t.to(a.device, copy=True)) for t in source]
        control = CPUAdamW(original, lr=1e-3, weight_decay=.01, foreach=False)
        actual = torch.optim.AdamW(alternative, lr=1e-3, weight_decay=.01, foreach=False)
        control.load_state_dict(saved['head'])
        actual.load_state_dict(copy.deepcopy(saved['head']['inner']))
        report.update(checkpoint_sha256=checksum, parameters=sum(t.numel() for t in source),
                      parameter_names=names, implementation_sha256=digest(__file__),
                      cpu_adamw_sha256=digest(Path(__file__).with_name('cpu_adamw.py')),
                      torch_version=str(torch.__version__), device=a.device)
        del source, saved, ck
        cpu_times, gpu_times = [], []
        generator = torch.Generator(device=a.device).manual_seed(34002)
        for step in range(4):
            headroom([a.device])
            control.zero_grad(set_to_none=True)
            actual.zero_grad(set_to_none=True)
            for left, right in zip(original, alternative):
                gradient = torch.randn(left.shape, generator=generator, device=a.device, dtype=torch.float32)
                gradient.mul_(1e-4 if step != 2 else 0.)
                left.grad, right.grad = gradient, gradient.clone()
            torch.nn.utils.clip_grad_norm_(original, 1., error_if_nonfinite=True, foreach=False)
            torch.nn.utils.clip_grad_norm_(alternative, 1., error_if_nonfinite=True, foreach=False)
            torch.cuda.synchronize(a.device)
            start = time.monotonic()
            control.step()
            torch.cuda.synchronize(a.device)
            cpu_seconds = time.monotonic() - start
            start = time.monotonic()
            actual.step()
            torch.cuda.synchronize(a.device)
            gpu_seconds = time.monotonic() - start
            if step:
                cpu_times.append(cpu_seconds)
                gpu_times.append(gpu_seconds)
            parameters = difference(original, alternative)
            first = difference([control.inner.state[p]['exp_avg'] for p in control.copies],
                               [actual.state[p]['exp_avg'] for p in alternative])
            second = difference([control.inner.state[p]['exp_avg_sq'] for p in control.copies],
                                [actual.state[p]['exp_avg_sq'] for p in alternative])
            finite = all(torch.isfinite(p).all().item() for p in original + alternative)
            record = {'step': step + 1, 'zero_gradient': step == 2,
                      'cpu_seconds': cpu_seconds, 'gpu_seconds': gpu_seconds,
                      'parameters': parameters, 'first_moments': first, 'second_moments': second,
                      'finite': finite, 'passed': finite and all(d['relative_l2'] < 1e-5
                          and d['maximum_absolute'] < 1e-5 for d in (parameters, first, second))}
            report['records'].append(record)
            write_json(out / 'report.json', report)
            print(json.dumps(record), flush=True)
        report.update(complete=True, passed=all(r['passed'] for r in report['records']),
                      median_cpu_seconds=statistics.median(cpu_times),
                      median_gpu_seconds=statistics.median(gpu_times),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device))
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        write_json(out / 'report.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:1')
    run(p.parse_args())
