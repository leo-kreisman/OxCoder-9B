"""Explicit data-expansion policy for canonical original-tower NCM training.

Only the sampling distribution and admitted training inventory change. The
parent optimizer, global schedule, frozen evaluation and objective survive.
"""
import json
from pathlib import Path
import torch

from .backbone import digest
from .prepare_creator_expansion import frozen_evaluation_hash
from .train_geometry import ResponseSampler


class CompactSampler:
    """Half the updates use complete short responses; half use the original mix."""
    def __init__(self, training):
        base = [x for x in training if x[0].get('curriculum') != 'compact-code-v33']
        compact = [x for x in training if x[0].get('curriculum') == 'compact-code-v33']
        if not base or not compact or any(r['split'] != 'train' for r, _ in training):
            raise ValueError('Both original and compact training partitions are required')
        self.base, self.compact = ResponseSampler(base), ResponseSampler(compact)
        self.report = {'compact_probability': .5, 'base': self.base.report,
                       'compact': self.compact.report,
                       'within_partition': 'Existing response-balanced source/teacher sampler'}

    def sample(self, rng):
        return (self.compact if rng.random() < .5 else self.base).sample(rng)


def qualify_expansion(base, expanded, parent_corpus, manifest):
    """Every original training/held-out row must survive byte-for-byte in value."""
    report = json.loads(Path(manifest).read_text())
    if (report.get('format') != 'frozen-tower-text-admission-v33'
            or not report.get('complete') or report.get('base_corpus') != parent_corpus
            or report.get('validation_unchanged') is not True):
        raise ValueError('Wrong expansion lineage')
    indexed = {r['id']: (r, c) for r, c in expanded}
    if len(indexed) != len(expanded) or any(indexed.get(r['id']) != (r, c) for r, c in base):
        raise ValueError('Original rows changed or disappeared')
    old_ids = {r['id'] for r, _ in base}
    added = [(r, c) for r, c in expanded if r['id'] not in old_ids]
    if not added or any(r['split'] != 'train' or r.get('curriculum') != 'compact-code-v33'
                        for r, _ in added):
        raise ValueError('Unexpected expansion rows')
    frozen = frozen_evaluation_hash(base)
    if frozen != frozen_evaluation_hash(expanded) or frozen != report['frozen_evaluation_sha256']:
        raise ValueError('Frozen evaluation changed')
    return {'admission_sha256': digest(manifest), 'original_chunks': len(base),
            'additional_chunks': len(added), 'frozen_evaluation_sha256': frozen}


def import_parent_state(path, parent, model, optimizer, rng, device, args):
    if parent.get('architecture') != 'frozen-context-original-oxcoder-native-ncm-v32':
        raise ValueError('Canonical-label v32 parent required')
    expected = {'planned_steps': args.steps, 'warmup': args.warmup,
        'effective_batch_examples': args.batch, 'seed': args.seed,
        'compute_dtype': 'float16', 'loss_scale': optimizer.loss_scale,
        'body_lr': 1e-4, 'head_lr': 1e-3, 'on_policy_probability': 0.,
        'conditioning_corruption_maximum': .15, 'autoregressive_answer_fallback': False}
    if any(parent.get(k) != v for k, v in expected.items()):
        raise ValueError('Parent objective, schedule or numerical policy changed')
    for name, sha in parent['implementation_sha256'].items():
        if digest(Path(__file__).with_name(name)) != sha:
            raise ValueError('Parent implementation changed: ' + name)
    state = torch.load(path, weights_only=True, map_location='cpu')
    if state['identity'] != parent or not 0 < state['step'] < args.steps:
        raise ValueError('Invalid parent state')
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(parameters) != set(state['model']):
        raise ValueError('Trainable inventory changed')
    with torch.no_grad():
        for name, p in parameters.items():
            p.copy_(state['model'][name].to(p))
    optimizer.load_state_dict(state['optimizer'])
    rng.setstate(state['rng']['python'])
    torch.set_rng_state(state['rng']['torch'])
    if str(device).startswith('cuda'):
        torch.cuda.set_rng_state(state['rng']['cuda'], device)
    return state['step']
