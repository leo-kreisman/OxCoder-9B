"""Create an explicit learning-rate branch without resetting learned state.

The source checkpoint is immutable. Only the body learning rate in the identity
and its two optimizer parameter groups changes. Model weights, FP32 masters,
moments, step counters, sampling RNG, labels and evaluation remain unchanged.
This prepares an experiment; it does not claim better convergence or quality.
"""
import argparse
import copy
import math
from pathlib import Path

import torch

from .backbone import digest
from .train import save, write_json


def branch(checkpoint, learning_rate, source_sha):
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('A finite positive learning rate is required')
    identity = checkpoint['identity']
    previous = identity['body_lr']
    if learning_rate == previous:
        raise ValueError('The branch must explicitly change the learning rate')
    bundle = checkpoint['optimizer']
    if (bundle.get('format') != 'mixed-optimizers-v1'
            or bundle.get('classes') != ['AdamW', 'AdamW', 'FP32MasterMuon']
            or len(bundle.get('states', [])) != 3):
        raise ValueError('Only the qualified FP32-master optimizer layout is supported')
    heads, interfaces, matrices = bundle['states']
    if matrices.get('format') != 'fp32-master-muon-v1':
        raise ValueError('Missing FP32 master state')
    groups = [heads['param_groups'], interfaces['param_groups'], matrices['inner']['param_groups']]
    for selected, expected in zip(groups, [identity['head_lr'], previous, previous]):
        if len(selected) != 1 or selected[0]['lr'] != expected:
            raise ValueError('Optimizer learning rates disagree with the checkpoint identity')
    if identity['optimizer']['body_lr'] != previous:
        raise ValueError('Optimizer identity disagrees with body learning rate')
    result = copy.deepcopy(checkpoint)
    result['identity']['body_lr'] = learning_rate
    result['identity']['optimizer']['body_lr'] = learning_rate
    result['identity']['learning_rate_transition'] = {
        'source_checkpoint_sha256': source_sha, 'source_step': checkpoint['step'],
        'previous_body_lr': previous, 'body_lr': learning_rate,
        'head_lr': identity['head_lr'],
        'scope': 'Explicit hyperparameter branch; weights, optimizer state, RNG and completed updates retained',
        'quality_ready': False}
    result['optimizer']['states'][1]['param_groups'][0]['lr'] = learning_rate
    result['optimizer']['states'][2]['inner']['param_groups'][0]['lr'] = learning_rate
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--body-lr', required=True, type=float)
    args = parser.parse_args()
    source, output = Path(args.checkpoint), Path(args.output)
    if output.exists():
        raise ValueError('Refusing to replace an existing checkpoint branch')
    source_sha = digest(source)
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    if digest(source) != source_sha:
        raise ValueError('Source checkpoint changed while loading')
    changed = branch(checkpoint, args.body_lr, source_sha)
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, changed)
    write_json(output.with_suffix('.json'), {
        **changed['identity']['learning_rate_transition'],
        'checkpoint_sha256': digest(output), 'implementation_sha256': digest(__file__)})


if __name__ == '__main__':
    main()
