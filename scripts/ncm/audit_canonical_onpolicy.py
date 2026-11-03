"""Qualify canonical student-conditioned training on the actual frozen 9B source.

No coding-gate examples are used. All qualification updates are discarded.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import torch

from .backbone import digest
from .gguf_training_backbone import load_candidate
from .exact_checkpoint import restore_trainable
from . import frozen_tower as tower
from .frozen_tower_training import load_corpus
from .frozen_ragged_qualification import qualify
from .frozen_tower_optimizer import TowerOptimizer
from .frozen_clean_objective import backward_batch as reference
from .canonical_onpolicy import backward_batch
from .train import admission, headroom, write_json


def run(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    report = {'complete': False, 'passed': False, 'quality_ready': False,
              'scope': 'Numerical and resource qualification only; all updates discarded', 'records': []}
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report['admission'] = admission([a.device], 9)
        model, codec, parent, checksum = load_candidate(a.source, a.codec, a.checkpoint, [a.device])
        execution = qualify(a.execution_audit, parent['source_sha256'], parent['codec_sha256'])
        for name, sha in parent['implementation_sha256'].items():
            if digest(Path(__file__).with_name(name)) != sha:
                raise ValueError('Parent implementation changed: ' + name)
        tower.set_compute_dtype(model.backbone, torch.float16)
        restore_trainable(model, a.checkpoint, parent, checksum)
        training, _, corpus = load_corpus(a.selection, a.manifest, parent['source_sha256'], parent['codec_sha256'])
        if parent['corpus'] != corpus:
            raise ValueError('Parent corpus differs')
        cases = []
        for size in (64, 69, 512, 2048):
            row, condition = min(training, key=lambda item: abs(len(item[1]['ids']) - size))
            ids = condition['ids']
            cases.append((row, {**condition, 'ids': (ids * ((size + len(ids) - 1) // len(ids)))[:size]}))
        report.update(checkpoint_sha256=checksum, execution=execution, corpus=corpus,
            batch_examples=len(cases), loss_scale=64., implementation_sha256={n: digest(Path(__file__).with_name(n))
            for n in ('audit_canonical_onpolicy.py', 'canonical_onpolicy.py', 'exact_checkpoint.py',
                      'frozen_clean_objective.py', 'frozen_multi_objective.py', 'frozen_tower_ragged.py',
                      'frozen_tower_optimizer.py', 'cpu_adamw.py', 'master_muon.py')})
        write_json(out / 'report.json', report)
        frozen = tower.source_versions(model.backbone)
        parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
        gradients = None
        for arm in ('reference', 'disabled_student'):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(36003)
            torch.cuda.manual_seed_all(36003)
            if arm == 'reference':
                values = reference(model, codec, cases, 64., .15)
                gradients = {n: p.grad.detach().cpu().clone() for n, p in parameters.items()}
                expected_loss = [r['loss'] for r in values]
            else:
                values = backward_batch(model, codec, cases, 64., .15, [False] * 4)
                report['disabled_student_exact'] = (expected_loss == [r['loss'] for r in values]
                    and all(torch.equal(p.grad.detach().cpu(), gradients[n]) for n, p in parameters.items()))
            headroom([a.device])
        del gradients
        model.zero_grad(set_to_none=True)
        if not report['disabled_student_exact']:
            raise ValueError('Disabled student conditioning changed reference gradients')
        saved = torch.load(a.checkpoint, weights_only=True, map_location='cpu', mmap=True)
        optimizer = TowerOptimizer(model)
        optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        del saved
        for flags in ([True, False, True, False], [True] * 4):
            headroom([a.device])
            optimizer.zero_grad()
            torch.cuda.synchronize(a.device)
            torch.cuda.reset_peak_memory_stats(a.device)
            start = time.monotonic()
            before = {n: p.detach().cpu().clone() for n, p in parameters.items()}
            values = backward_batch(model, codec, cases, 64., .15, flags)
            norm = optimizer.step()
            torch.cuda.synchronize(a.device)
            delta, finite = {}, True
            for n, p in parameters.items():
                finite = finite and bool(torch.isfinite(p).all())
                group = 'adapters' if n.startswith('backbone.adapters.') else 'classifiers' if n.startswith(('head.', 'stop_head.')) else 'interfaces'
                delta[group] = delta.get(group, 0.) + float((p.detach().cpu().float() - before[n].float()).square().sum())
            record = {'flags': flags, 'loss': sum(r['loss'] for r in values), 'gradient_norm': norm,
                'seconds': time.monotonic() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(a.device),
                'finite_parameters': finite, 'parameter_delta_square_sum': delta,
                'source_unchanged': tower.source_versions(model.backbone) == frozen,
                'codec_has_gradient': any(p.grad is not None for p in codec.parameters())}
            record['passed'] = finite and record['source_unchanged'] and not record['codec_has_gradient'] and set(delta) == {'adapters', 'classifiers', 'interfaces'} and all(v > 0 for v in delta.values())
            report['records'].append(record)
            write_json(out / 'report.json', report)
            print(json.dumps(record), flush=True)
        report.update(complete=True, passed=report['disabled_student_exact'] and all(r['passed'] for r in report['records']))
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        write_json(out / 'report.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'codec', 'checkpoint', 'selection', 'manifest', 'execution-audit', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:0')
    run(p.parse_args())
