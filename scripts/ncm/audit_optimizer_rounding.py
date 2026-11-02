"""Measure BF16 update loss using saved real Muon momentum directions.

The fixed-direction replay is a numerical control, not a prediction of future
training gradients or a coding-quality result. No checkpoint is modified.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.optim._muon import _zeropower_via_newtonschulz, _adjust_lr
from .backbone import digest
from .train import write_json


def run(a):
    torch.set_num_threads(2)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    initial = torch.load(a.initialization, map_location='cpu', weights_only=True)
    bundle = checkpoint['optimizer']
    i = bundle['classes'].index('Muon')
    state = bundle['states'][i]
    names = [n for n, p in checkpoint['model'].items()
             if (n.startswith('backbone.adapters.') and p.ndim == 2) or n == 'project.weight']
    if len(state['param_groups']) != 1 or len(names) != len(state['param_groups'][0]['params']):
        raise ValueError('Unexpected Muon parameter inventory')
    group = state['param_groups'][0]
    records = []
    for name, key in zip(names, group['params']):
        parameter = checkpoint['model'][name]
        momentum = state['state'][key]['momentum_buffer']
        if parameter.shape != momentum.shape:
            raise ValueError('Momentum order differs from parameter order')
        direction = _zeropower_via_newtonschulz(momentum.clone(), group['ns_coefficients'], group['ns_steps'], group['eps'])
        lr = _adjust_lr(group['lr'], group['adjust_lr_fn'], parameter.shape)
        resident, master = parameter.clone(), parameter.float().clone()
        for _ in range(a.replays):
            resident.mul_(1 - group['lr'] * group['weight_decay']).add_(direction, alpha=-lr)
            master.mul_(1 - group['lr'] * group['weight_decay']).add_(direction.float(), alpha=-lr)
        reference = master.to(parameter.dtype)
        delta = master - parameter.float()
        records.append({'parameter': name, 'dtype': str(parameter.dtype), 'elements': parameter.numel(),
            'historical_changed_fraction': float((parameter != initial['model'][name]).float().mean()),
            'resident_changed_fraction': float((resident != parameter).float().mean()),
            'master_changed_fraction': float((reference != parameter).float().mean()),
            'resident_delta_relative_error': float((resident.float() - master).norm() / delta.norm().clamp_min(1e-20)),
            'master_bf16_delta_relative_error': float((reference.float() - master).norm() / delta.norm().clamp_min(1e-20))})
    report = {'checkpoint_sha256': digest(a.checkpoint), 'step': checkpoint['step'],
        'initialization_sha256': digest(a.initialization), 'fixed_direction_replays': a.replays,
        'scope': 'Saved real momentum, repeated direction; no new training or quality score',
        'records': records, 'complete': True, 'quality_ready': False}
    write_json(a.output, report)
    for kind in ('down.weight', 'up.weight', 'project.weight'):
        rows = [r for r in records if r['parameter'].endswith(kind)]
        print(json.dumps({'kind': kind, 'matrices': len(rows), **{key: sum(r[key] for r in rows)/len(rows)
            for key in ('historical_changed_fraction', 'resident_changed_fraction', 'master_changed_fraction',
                        'resident_delta_relative_error', 'master_bf16_delta_relative_error')}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'initialization', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--replays', type=int, default=32)
    args = parser.parse_args()
    if args.replays < 1:
        parser.error('Positive replay count required')
    run(args)
