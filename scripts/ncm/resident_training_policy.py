"""Strict state-preserving continuation into an audited resident-head runtime."""
import json
from pathlib import Path
import torch
from .backbone import digest


def validate_parent(parent, corpus, args):
    expected = {'architecture': 'frozen-context-original-oxcoder-native-ncm-v33',
                'corpus': corpus, 'effective_batch_examples': args.batch,
                'planned_steps': args.steps, 'warmup': args.warmup, 'seed': args.seed,
                'compute_dtype': 'float16', 'body_lr': 1e-4, 'head_lr': 1e-3,
                'loss_scale': 64., 'on_policy_probability': 0.,
                'conditioning_corruption_maximum': .15, 'compact_sampling_probability': .5,
                'autoregressive_answer_fallback': False}
    if any(parent.get(k) != v for k, v in expected.items()):
        raise ValueError('Resident continuation changes training recipe or data')
    for name, checksum in parent['implementation_sha256'].items():
        if digest(Path(__file__).with_name(name)) != checksum:
            raise ValueError('Parent implementation changed: ' + name)


def qualify(path, parent_sha):
    report = json.loads(Path(path).read_text())
    if (not report.get('complete') or not report.get('passed') or not report.get('speed_qualified')
            or report.get('checkpoint_sha256') != parent_sha
            or set(report.get('arms', {})) != {'control', 'resident'}):
        raise ValueError('Passing full runtime audit for this parent required')
    if any(len(arm['records']) != 4 for arm in report['arms'].values()):
        raise ValueError('Incomplete optimizer audit')
    for name, checksum in report['implementation_sha256'].items():
        if digest(Path(__file__).with_name(name)) != checksum:
            raise ValueError('Audited runtime implementation changed: ' + name)
    return report


def restore(path, identity, model, optimizer, rng, device, legacy=False):
    state = torch.load(path, weights_only=True, map_location='cpu')
    if state['identity'] != identity or not 0 < state['step'] < identity['planned_steps']:
        raise ValueError('Invalid continuation identity or completed schedule')
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if (set(parameters) != set(state['model'])
            or any(parameters[n].shape != p.shape for n, p in state['model'].items())):
        raise ValueError('Continuation parameter inventory differs')
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state['model'][name].to(parameter))
    if legacy:
        optimizer.import_legacy_state(state['optimizer'])
    else:
        optimizer.load_state_dict(state['optimizer'])
    rng.setstate(state['rng']['python'])
    torch.set_rng_state(state['rng']['torch'])
    if str(device).startswith('cuda'):
        torch.cuda.set_rng_state(state['rng']['cuda'], device)
    return state['step']
