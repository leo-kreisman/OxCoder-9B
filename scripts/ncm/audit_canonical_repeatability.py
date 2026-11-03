"""Diagnose disabled-objective parity without updates or relaxed admission.

Run two original and two disabled-student passes from identical saved weights
and random states. Record input/label and RNG fingerprints separately from
gradient differences. This diagnostic cannot authorize a training branch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch

from . import canonical_onpolicy as candidate
from . import frozen_clean_objective as reference
from . import frozen_tower as tower
from . import frozen_tower_multi as prefix
from .backbone import digest
from .exact_checkpoint import restore_trainable
from .gguf_training_backbone import load_candidate
from .frozen_tower_training import load_corpus
from .train import admission, headroom, write_json


def fingerprint(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((value.shape, value.dtype)).encode() + value.numpy().tobytes()).hexdigest()


def rng(device):
    return {'cpu': fingerprint(torch.get_rng_state()),
            'cuda': fingerprint(torch.cuda.get_rng_state(device))}


def comparison(parameters, expected):
    error, scale, maximum, unequal = 0., 0., 0., []
    for name, parameter in parameters.items():
        actual = parameter.grad.detach().cpu()
        old = expected[name]
        if not torch.equal(actual, old):
            unequal.append(name)
        delta = actual.float() - old.float()
        error += float(delta.double().square().sum())
        scale += float(old.double().square().sum())
        maximum = max(maximum, float(delta.abs().max()))
    return {'exact': not unequal, 'relative_l2': (error / max(scale, 1e-30)) ** .5,
            'maximum_absolute': maximum, 'unequal_tensors': len(unequal),
            'first_unequal_names': unequal[:12]}


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    report = {'complete': False, 'records': [], 'training_admitted': False,
              'scope': 'Read-only numerical diagnosis; no optimizer, evaluation tasks, or model updates'}
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report['admission'] = admission([args.device], 9)
    model, codec, parent, checksum = load_candidate(args.source, args.codec, args.checkpoint, [args.device])
    tower.set_compute_dtype(model.backbone, torch.float16)
    restore_trainable(model, args.checkpoint, parent, checksum)
    training, _, corpus = load_corpus(args.selection, args.manifest, parent['source_sha256'], parent['codec_sha256'])
    if corpus != parent['corpus']:
        raise ValueError('Parent corpus differs')
    cases = []
    for size in (64, 69, 512, 2048):
        row, condition = min(training, key=lambda item: abs(len(item[1]['ids']) - size))
        ids = condition['ids']
        cases.append((row, {**condition, 'ids': (ids * ((size + len(ids) - 1) // len(ids)))[:size]}))
    report.update(checkpoint_sha256=checksum, corpus=corpus,
        implementation_sha256={n: digest(Path(__file__).with_name(n)) for n in
            ('audit_canonical_repeatability.py', 'canonical_onpolicy.py', 'frozen_clean_objective.py')})
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    source = tower.source_versions(model.backbone)
    original_prepare, original_supervision = prefix.prepare, reference.supervision
    events = []
    def prepare(*a, **kw):
        before = rng(args.device)
        state = original_prepare(*a, **kw)
        events.append({'operation': 'prepare', 'rng_before': before, 'rng_after': rng(args.device)})
        return state
    def supervision(*a, **kw):
        before = rng(args.device)
        labels, inputs = original_supervision(*a, **kw)
        events.append({'operation': 'supervision', 'rng_before': before, 'rng_after': rng(args.device),
                       'labels': [fingerprint(x) for x in labels], 'inputs': [fingerprint(x) for x in inputs]})
        return labels, inputs
    prefix.prepare = prepare
    reference.supervision = candidate.supervision = supervision
    gradients = None
    try:
        for arm in ('reference_1', 'reference_2', 'disabled_1', 'disabled_2'):
            headroom([args.device])
            model.zero_grad(set_to_none=True)
            events.clear()
            torch.manual_seed(36003)
            torch.cuda.manual_seed_all(36003)
            values = (reference.backward_batch(model, codec, cases, 64., .15)
                      if arm.startswith('reference') else
                      candidate.backward_batch(model, codec, cases, 64., .15, [False] * 4))
            torch.cuda.synchronize(args.device)
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters.values()):
                raise ValueError('Missing or nonfinite gradient')
            if gradients is None:
                gradients = {n: p.grad.detach().cpu().clone() for n, p in parameters.items()}
            record = {'arm': arm, 'losses': [r['loss'] for r in values], 'events': list(events),
                      'gradient_comparison_to_first': comparison(parameters, gradients),
                      'source_unchanged': tower.source_versions(model.backbone) == source,
                      'codec_has_gradient': any(p.grad is not None for p in codec.parameters())}
            report['records'].append(record)
            write_json(out / 'report.json', report)
            print(json.dumps({k: v for k, v in record.items() if k != 'events'}), flush=True)
        report['complete'] = True
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        prefix.prepare = original_prepare
        reference.supervision = candidate.supervision = original_supervision
        write_json(out / 'report.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'codec', 'checkpoint', 'selection', 'manifest', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:0')
    run(p.parse_args())
