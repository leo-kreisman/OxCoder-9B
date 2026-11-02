"""Full OxCoder gradient and timing audit for independent-example packing."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from .audit_training_runtime import compare
from .backbone import digest
from .codec import encode
from .generate_candidate import load_candidate
from .packed_training import forward_examples
from .shared_training import forward_levels
from .train import admission, write_json
from .train_accumulated import example_loss
from .train_independent import corrupted_inputs
from .training_runtime import execution


def compare_groups(actual, expected):
    groups = {}
    for name, test in (
        ('adapters', lambda n: n.startswith('backbone.adapters.')),
        ('classifiers', lambda n: n.startswith(('head.', 'stop_head.'))),
        ('interfaces', lambda n: not n.startswith(('backbone.adapters.', 'head.', 'stop_head.')))):
        groups[name] = compare({k: v for k, v in actual.items() if test(k)},
                               {k: v for k, v in expected.items() if test(k)})
    return groups


def run(a):
    torch.set_num_threads(2)
    devices = ['cuda:0', 'cuda:1']
    limits = admission(devices, a.allocator_cap_gib)
    model, codec, identity, sha = load_candidate(a.source, a.codec, a.checkpoint, devices)
    model.train()
    path = Path(a.data) / 'prepared.pt'
    rows = torch.load(path, weights_only=True, map_location='cpu')['prepared']
    rows = [(r, c) for r, c in rows if r['split'] == 'train' and len(c['ids']) >= 64]
    report = {'checkpoint_sha256': sha, 'prepared_sha256': digest(path), 'model_identity': identity,
              'implementation_sha256': {name: digest(Path(__file__).parent / name) for name in
                  ('audit_packed_training.py', 'packed_training.py', 'shared_training.py',
                   'training_runtime.py', 'backbone.py', 'vendor/veomni_shared_prefix.py')},
              'admission': limits, 'records': [], 'complete': False, 'quality_ready': False,
              'optimizer_state_in_memory': False,
              'scope': 'Equal-example four-level forward/backward; no parameter updates',
              'thresholds': {'logit_relative_l2': .02, 'gradient_relative_l2': .05,
                             'gradient_cosine_minimum': .99}}
    for requested in ([64, 69], [64, 69, 128, 256], [512, 1024]):
        selected = []
        used = set()
        for length in requested:
            row, condition = min((x for x in rows if x[0]['id'] not in used),
                                 key=lambda x: abs(len(x[1]['ids']) - length))
            used.add(row['id'])
            selected.append((row, condition))
        inputs, codes, prompts = [], [], []
        torch.manual_seed(19005)
        for row, condition in selected:
            target = torch.tensor([encode(row['target'], codec.config.length)], device=devices[-1])
            with torch.no_grad():
                labels, _ = codec.targets(target)
            inputs.append(corrupted_inputs(codec, labels, .5))
            codes.append(labels)
            prompts.append(torch.tensor([condition['ids']]))
        reference = None
        for packed in (False, True):
            times, peaks = [], []
            try:
                for repeat in range(a.repeats + 1):
                    model.zero_grad(set_to_none=True)
                    for device in devices:
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    start = time.monotonic()
                    outputs = {}
                    loss_total = 0
                    with execution(model, 0, 'reference'):
                        if packed:
                            branches = forward_examples(model, list(zip(inputs, prompts)))
                            joint = 0
                        for index, (_, condition) in enumerate(selected):
                            values = branches[index] if packed else forward_levels(model, inputs[index], prompts[index])
                            loss = 0
                            for level, (logits, end, _) in enumerate(values):
                                loss = loss + example_loss(logits, codes[index][level], end,
                                    condition['is_final_chunk'], level, len(codes[index]), len(selected))
                                outputs[f'{index}:{level}:codes'] = logits.detach()
                                outputs[f'{index}:{level}:end'] = end.detach()
                            loss_total += float(loss.detach())
                            if packed:
                                joint = joint + loss
                            else:
                                loss.backward()
                            del values
                        if packed:
                            joint.backward()
                            del branches, joint
                    for device in devices:
                        torch.cuda.synchronize(device)
                    elapsed = time.monotonic() - start
                    if repeat:
                        times.append(elapsed)
                    peaks.append({device: torch.cuda.max_memory_allocated(device) for device in devices})
                    del logits, end, loss
                outputs = {name: value.cpu() for name, value in outputs.items()}
                gradients = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()
                             if p.requires_grad and p.grad is not None}
                if set(gradients) != {name for name, p in model.named_parameters() if p.requires_grad}:
                    raise ValueError('Missing trainable gradients')
                if not packed:
                    reference = (outputs, gradients, statistics.median(times))
                ge, le = compare(gradients, reference[1]), compare(outputs, reference[0])
                groups = compare_groups(gradients, reference[1])
                passed = (ge['finite'] and le['finite'] and ge['relative_l2'] < .05 and
                          ge['cosine'] > .99 and le['relative_l2'] < .02 and
                          all(g['finite'] and g['relative_l2'] < .05 and g['cosine'] > .99
                              for g in groups.values()))
                row = {'requested_prompt_tokens': requested, 'prompt_tokens': [p.shape[1] for p in prompts],
                       'compact_tokens': sum(p.shape[1] + sum(codec.config.levels) for p in prompts),
                       'examples': [row['id'] for row, _ in selected], 'packed': packed,
                       'loss': loss_total, 'median_seconds': statistics.median(times), 'seconds': times,
                       'speed_ratio': reference[2] / statistics.median(times),
                       'peak_allocated_bytes': {d: max(p[d] for p in peaks) for d in devices},
                       'logit_comparison': le, 'gradient_comparison': ge, 'gradient_groups': groups,
                       'passed': passed}
                del gradients, outputs
            except torch.cuda.OutOfMemoryError:
                row = {'requested_prompt_tokens': requested, 'packed': packed,
                       'passed': False, 'error': 'CUDA allocator cap exceeded'}
                model.zero_grad(set_to_none=True)
                gc.collect()
                for device in devices:
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
            report['records'].append(row)
            write_json(a.output, report)
            print(json.dumps(row), flush=True)
        del reference
    report['complete'] = True
    write_json(a.output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'codec', 'checkpoint', 'data', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--allocator-cap-gib', type=float, default=11.25)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('Positive repeats required')
    run(args)
