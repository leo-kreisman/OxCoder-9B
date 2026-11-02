"""Compare the BF16 resident reference and exact GGUF-storage training path.

Loads the two backends sequentially, never together. No optimizer step or new
quantization occurs. Passing matrix tests alone does not qualify this runtime.
"""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from .audit_training_runtime import compare
from .audit_packed_training import compare_groups
from .backbone import digest
from .codec import encode
from .generate_candidate import load_candidate as reference_loader
from .gguf_training_backbone import load_candidate as compact_loader
from .shared_training import forward_levels
from .train import admission, write_json, save
from .train_accumulated import example_loss
from .train_independent import corrupted_inputs
from .training_runtime import execution


def run_pass(model, inputs, codes, prompt, final, devices, repeats):
    times, peaks = [], []
    for repeat in range(repeats + 1):
        model.zero_grad(set_to_none=True)
        for device in devices:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        with execution(model, 0, 'reference'):
            outputs = forward_levels(model, inputs, prompt)
            loss = sum(example_loss(logits, codes[level], end, final, level, len(codes), 1)
                       for level, (logits, end, _) in enumerate(outputs))
            loss.backward()
        for device in devices:
            torch.cuda.synchronize(device)
        seconds = time.monotonic() - started
        if repeat:
            times.append(seconds)
        peaks.append({device: torch.cuda.max_memory_allocated(device) for device in devices})
    logits = {f'{i}:codes': entry[0].detach().cpu() for i, entry in enumerate(outputs)}
    logits.update({f'{i}:end': entry[1].detach().cpu() for i, entry in enumerate(outputs)})
    gradients = {name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters()
                 if parameter.requires_grad and parameter.grad is not None}
    if set(gradients) != {name for name, p in model.named_parameters() if p.requires_grad}:
        raise ValueError('Missing trainable gradients')
    return {'outputs': logits, 'gradients': gradients,
            'loss': float(loss.detach()), 'seconds': times,
            'median_seconds': statistics.median(times),
            'peak_allocated_bytes': {d: max(row[d] for row in peaks) for d in devices}}


def run(a):
    torch.set_num_threads(2)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((Path(a.data) / 'manifest.json').read_text())
    path = Path(a.data) / 'prepared.pt'
    if digest(path) != manifest['prepared_sha256']:
        raise ValueError('Prepared data checksum differs')
    rows = [(r, c) for r, c in torch.load(path, map_location='cpu', weights_only=True)['prepared']
            if r['split'] == 'train' and len(c['ids']) >= 64]
    selected = [min(rows, key=lambda x: abs(len(x[1]['ids']) - n)) for n in (64, 69, 512, 2048)]
    report = {'source_checkpoint_sha256': digest(a.checkpoint),
              'prepared_sha256': manifest['prepared_sha256'], 'records': [], 'complete': False,
              'quality_ready': False, 'scope': 'Same frozen weights; physical storage and device-placement audit only',
              'implementation_sha256': {name: digest(Path(__file__).parent / name) for name in
                  ('gguf_training_weights.py', 'gguf_training_backbone.py', 'audit_gguf_training_runtime.py',
                   'backbone.py', 'shared_training.py', 'training_runtime.py', 'vendor/veomni_shared_prefix.py')}}
    for compact in (False, True):
        devices = a.device.split(',') if compact else ['cuda:0', 'cuda:1']
        limits = admission(devices, 11.25)
        loader = compact_loader if compact else reference_loader
        kwargs = {'layer_split': report['model_identity']['layer_split']} if compact and len(devices) == 2 else {}
        model, codec, identity, sha = loader(a.source, a.codec, a.checkpoint, devices, **kwargs)
        model.train()
        report['model_identity'] = identity
        if compact:
            report['compact_weight_bytes'] = sum(row['bytes'] for row in model.backbone.transfers)
        for requested, (row, condition) in zip((64, 69, 512, 2048), selected):
            torch.manual_seed(19007)
            target = torch.tensor([encode(row['target'], codec.config.length)], device=devices[-1])
            with torch.no_grad():
                codes, _ = codec.targets(target)
            inputs = corrupted_inputs(codec, codes, .5)
            prompt = torch.tensor([condition['ids']])
            measurement = run_pass(model, inputs, codes, prompt, condition['is_final_chunk'], devices, a.repeats)
            if compact:
                reference = torch.load(out / f'reference-{requested}.pt', weights_only=True, map_location='cpu')
            else:
                reference = measurement
                save(out / f'reference-{requested}.pt', measurement)
            ge = compare(measurement['gradients'], reference['gradients'])
            le = compare(measurement['outputs'], reference['outputs'])
            groups = compare_groups(measurement['gradients'], reference['gradients'])
            record = {key: measurement[key] for key in ('loss', 'seconds', 'median_seconds', 'peak_allocated_bytes')}
            record.update(compact_weights=compact, devices=devices, admission=limits, example=row['id'],
                          requested_prompt_tokens=requested, prompt_tokens=len(condition['ids']),
                          speed_ratio=reference['median_seconds'] / measurement['median_seconds'],
                          gradient_comparison=ge, logit_comparison=le, gradient_groups=groups,
                          bit_exact_outputs=all(torch.equal(v, reference['outputs'][n]) for n, v in measurement['outputs'].items()),
                          bit_exact_gradients=all(torch.equal(v, reference['gradients'][n]) for n, v in measurement['gradients'].items()))
            record['passed'] = (ge['finite'] and le['finite'] and ge['relative_l2'] < .05 and
                ge['cosine'] > .99 and le['relative_l2'] < .02 and
                all(g['finite'] and g['relative_l2'] < .05 and g['cosine'] > .99 for g in groups.values()))
            report['records'].append(record)
            write_json(out / 'report.json', report)
            print(json.dumps(record), flush=True)
            del measurement, reference, inputs, codes, target, prompt
        del model, codec
        gc.collect()
        for device in ('cuda:0', 'cuda:1'):
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    report['complete'] = True
    report['passed'] = all(row['passed'] for row in report['records'])
    write_json(out / 'report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'codec', 'checkpoint', 'data', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('Positive repeats required')
    run(args)
