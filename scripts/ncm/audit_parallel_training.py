"""Qualify two-replica compact NCM backwards against a sequential global batch.

Launch with torchrun, two local processes, after other owned GPU jobs stop.
No optimizer updates or evaluation answers are used. Speed includes summed
gradient communication. Failure never authorizes a training-runtime change.
"""
import argparse
from contextlib import contextmanager, nullcontext
from datetime import timedelta
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist

from .audit_training_runtime import compare
from .backbone import digest
from .gguf_training_backbone import load_candidate
from .parallel_training import assignments, backward_example, check_inventory, example_seed, sum_gradients
from .train import admission, write_json


@contextmanager
def capture(model, output, example):
    def head(module, args, value):
        logits = value[0] if isinstance(value, tuple) else value
        output[f'{example}:head:{args[1]}'] = logits.detach().cpu().clone()
    def stop(module, args, value):
        output[f'{example}:stop:{sum(k.startswith(str(example)+":stop:") for k in output)}'] = value.detach().cpu().clone()
    handles = [model.head.register_forward_hook(head), model.stop_head.register_forward_hook(stop)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def gradient_groups(actual, reference):
    def selected(name):
        if name.startswith('backbone.adapters.'):
            return 'adapters'
        if name.startswith(('head.', 'stop_head.')):
            return 'classifiers'
        return 'interfaces'
    return {group: compare({n: v for n, v in actual.items() if selected(n) == group},
                           {n: v for n, v in reference.items() if selected(n) == group})
            for group in ('adapters', 'classifiers', 'interfaces')}


def run(a):
    rank = int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 2:
        raise ValueError('This audit requires exactly two local workers')
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    device, out = f'cuda:{rank}', Path(a.output)
    limits = admission([device], a.allocator_cap_gib)
    dist.init_process_group('nccl', timeout=timedelta(minutes=15))
    try:
        if rank == 0:
            out.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        model, codec, identity, sha = load_candidate(a.source, a.codec, a.checkpoint, [device])
        model.train()
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        check_inventory(named)
        prepared = torch.load(Path(a.data) / 'prepared.pt', map_location='cpu', weights_only=True)['prepared']
        rows = [(r, c) for r, c in prepared if r['split'] == 'train']
        report = {'checkpoint_sha256': sha, 'prepared_sha256': digest(Path(a.data) / 'prepared.pt'),
            'model_identity': identity, 'devices': ['cuda:0', 'cuda:1'],
            'implementation_sha256': {name: digest(Path(__file__).parent / name) for name in
                ('audit_parallel_training.py', 'parallel_training.py', 'gguf_training_backbone.py',
                 'gguf_training_weights.py', 'shared_training.py', 'backbone.py', 'train_accumulated.py')},
            'records': [], 'complete': False, 'passed': False, 'quality_ready': False,
            'scope': 'Same global examples, fixed weights and corruption; no optimizer update or quality score',
            'thresholds': {'gradient_relative_l2': .05, 'gradient_cosine_minimum': .99, 'logit_relative_l2': .02}}
        write_json(out / f'admission-rank{rank}.json', limits)
        for case, lengths in enumerate(((64, 69, 128, 256), (64, 512, 1024, 2048))):
            batch = [min(rows, key=lambda item: abs(len(item[1]['ids']) - size)) for size in lengths]
            allocation = assignments([len(c['ids']) for r, c in batch], 2)
            timings, reference_times = [], []
            for repeat in range(a.repeats + 1):
                model.zero_grad(set_to_none=True)
                outputs = {}
                dist.barrier(); torch.cuda.synchronize()
                start = time.perf_counter()
                for index in allocation[rank]:
                    with capture(model, outputs, index) if repeat == a.repeats else nullcontext():
                        backward_example(model, codec, batch[index], 4, .5, example_seed(23001, case, index))
                sum_gradients(named)
                torch.cuda.synchronize()
                elapsed = torch.tensor(time.perf_counter() - start, device=device)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                if repeat:
                    timings.append(float(elapsed))
                if repeat == a.repeats:
                    torch.save(outputs, out / f'case{case}-rank{rank}-outputs.pt')
                    if rank == 0:
                        actual = {n: p.grad.detach().cpu().clone() for n, p in named}
                dist.barrier()
                if rank == 0:
                    model.zero_grad(set_to_none=True)
                    reference_outputs = {}
                    torch.cuda.synchronize(); began = time.perf_counter()
                    for index in range(4):
                        with capture(model, reference_outputs, index) if repeat == a.repeats else nullcontext():
                            backward_example(model, codec, batch[index], 4, .5, example_seed(23001, case, index))
                    torch.cuda.synchronize()
                    if repeat:
                        reference_times.append(time.perf_counter() - began)
                    if repeat == a.repeats:
                        reference = {n: p.grad.detach().cpu().clone() for n, p in named}
                        combined = {}
                        for worker in range(2):
                            combined.update(torch.load(out / f'case{case}-rank{worker}-outputs.pt',
                                                       map_location='cpu', weights_only=True))
                        gradients, logits = compare(actual, reference), compare(combined, reference_outputs)
                        groups = gradient_groups(actual, reference)
                        passed = (logits['finite'] and logits['relative_l2'] < .02
                            and all(g['finite'] and g['relative_l2'] < .05 and g['cosine'] > .99
                                    for g in [gradients, *groups.values()]))
                        record = {'requested_prompt_tokens': lengths, 'prompt_tokens': [len(c['ids']) for r, c in batch],
                            'example_ids': [r['id'] for r, c in batch], 'allocation': allocation,
                            'gradient_comparison': gradients, 'gradient_groups': groups, 'logit_comparison': logits,
                            'parallel_seconds': timings, 'reference_seconds': reference_times,
                            'speed_ratio': statistics.median(reference_times) / statistics.median(timings),
                            'passed': passed}
                        report['records'].append(record)
                        write_json(out / 'report.json', report)
                        print(record, flush=True)
                        del actual, reference, combined
                dist.barrier()
        if rank == 0:
            report['complete'] = True
            report['passed'] = all(r['passed'] for r in report['records'])
            report['speed_qualified'] = report['passed'] and all(r['speed_ratio'] > 1 for r in report['records'])
            write_json(out / 'report.json', report)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'codec', 'checkpoint', 'data', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--allocator-cap-gib', type=float, default=11.25)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('At least one timed repeat is required')
    run(args)
