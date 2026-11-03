"""Full-shape optimizer smoke for qualified variable-prompt NCM batching."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import torch

from .backbone import digest
from .gguf_training_backbone import load_candidate
from . import frozen_tower as tower
from .frozen_tower_training import load_corpus
from .frozen_multi_qualification import qualify
from .frozen_multi_objective import backward_batch
from .frozen_tower_optimizer import TowerOptimizer
from .train import admission, write_json


def run(a):
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    paused = []
    report = {'complete': False, 'passed': False, 'quality_ready': False, 'records': [],
              'scope': 'Two full-shape training updates only; weights discarded; no generated-code qualification'}
    try:
        for unit in a.pause_unit:
            if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode == 0:
                subprocess.run(['systemctl', '--user', 'kill', '--signal=STOP', unit], check=True)
                paused.append(unit)
        torch.set_num_threads(2)
        torch.manual_seed(29003)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report['admission'] = admission([a.device], 9)
        model, codec, parent, parent_sha = load_candidate(a.source, a.codec, a.checkpoint, [a.device])
        policy = qualify(a.execution_audit, parent['source_sha256'], parent['codec_sha256'])
        tower.set_compute_dtype(model.backbone, torch.float16)
        training, _, corpus = load_corpus(a.selection, a.manifest, parent['source_sha256'], parent['codec_sha256'])
        original = tower.source_versions(model.backbone)
        optimizer = TowerOptimizer(model)
        cases = []
        for requested in (64, 69, 512, 2048):
            row, condition = min(training, key=lambda x: abs(len(x[1]['ids']) - requested))
            if requested == 2048 and len(condition['ids']) < requested:
                ids = condition['ids']
                condition = {**condition, 'ids': (ids * ((requested + len(ids) - 1) // len(ids)))[:requested]}
            cases.append((row, condition))
        report.update(checkpoint_sha256=parent_sha, execution=policy, corpus=corpus,
            body_lr=1e-4, head_lr=1e-3, loss_scale=optimizer.loss_scale, batch_examples=len(cases),
            implementation_sha256={n: digest(Path(__file__).with_name(n)) for n in
                ('audit_frozen_multi_optimizer.py', 'frozen_multi_qualification.py', 'frozen_multi_objective.py', 'frozen_tower_training.py',
                 'frozen_tower_objective.py', 'frozen_tower_optimizer.py', 'cpu_adamw.py', 'master_muon.py')})
        write_json(out / 'report.json', report)
        for update in range(2):
            torch.cuda.synchronize(a.device)
            torch.cuda.reset_peak_memory_stats(a.device)
            began = time.monotonic()
            optimizer.zero_grad()
            before = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
            results = backward_batch(model, codec, cases, optimizer.loss_scale, [bool(update)] * len(cases))
            norm = optimizer.step()
            torch.cuda.synchronize(a.device)
            deltas = {}
            finite = True
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                finite = finite and bool(torch.isfinite(p).all())
                group = 'adapters' if name.startswith('backbone.adapters.') else 'classifiers' if name.startswith(('head.', 'stop_head.')) else 'interfaces'
                delta = (p.detach().cpu().float() - before[name].float()).square().sum().item()
                deltas[group] = deltas.get(group, 0.) + delta
            record = {'update': update + 1, 'loss': sum(r['loss'] for r in results),
                'gradient_norm': norm, 'seconds': time.monotonic() - began,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(a.device),
                'self_conditioned': bool(update), 'parameter_delta_square_sum': deltas,
                'finite_parameters': finite, 'source_unchanged': original == tower.source_versions(model.backbone),
                'codec_has_gradient': any(p.grad is not None for p in codec.parameters())}
            record['passed'] = (finite and record['source_unchanged'] and not record['codec_has_gradient']
                                and set(deltas) == {'adapters', 'classifiers', 'interfaces'} and all(v > 0 for v in deltas.values()))
            report['records'].append(record)
            write_json(out / 'report.json', report)
            print(json.dumps(record), flush=True)
        report['complete'] = True
        report['passed'] = all(r['passed'] for r in report['records'])
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        write_json(out / 'report.json', report)
        for unit in reversed(paused):
            subprocess.run(['systemctl', '--user', 'kill', '--signal=CONT', unit], check=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'codec', 'checkpoint', 'selection', 'manifest', 'execution-audit', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--pause-unit', action='append', default=[])
    run(p.parse_args())
