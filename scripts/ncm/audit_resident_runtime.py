"""Full-shape optimizer qualification for interleaved prefill and resident heads."""
import argparse
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from .backbone import digest
from .gguf_training_backbone import load_candidate
from . import frozen_tower as tower
from .frozen_tower_training import load_corpus
from .frozen_tower_optimizer import TowerOptimizer
from .resident_tower_optimizer import ResidentTowerOptimizer
from .frozen_clean_objective import backward_batch
from .interleaved_context import installed
from .audit_head_residency import difference
from .train import admission, headroom, write_json


def run(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    paused = []
    report = {'complete': False, 'passed': False, 'quality_ready': False, 'arms': {},
              'scope': 'Four full training updates per runtime from identical saved state; all discarded'}

    def active(unit):
        return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode == 0

    def pause(unit):
        if active(unit):
            subprocess.run(['systemctl', '--user', 'kill', '--kill-who=main', '--signal=STOP', unit], check=True)
            paused.append(unit)

    def wait(unit):
        while active(unit):
            report['phase'] = 'waiting_for_' + unit
            write_json(out / 'report.json', report)
            time.sleep(10)

    try:
        if a.wait_for_session:
            pause('ncm-compact-curriculum-pipeline-v33.service')
            wait('ncm-compact-tower-worker-v33.service')
            pause('ncm-complete-pipeline-v28.service')
            wait('ncm-context-evaluation-v28.service')
        if Path('results/ncm-compact-tower-v33/candidate.json').exists():
            report['phase'] = 'existing_candidate_takes_priority'
            return
        prefix_audit = json.loads(Path(a.prefix_audit).read_text())
        if not (prefix_audit.get('complete') and prefix_audit.get('passed') and prefix_audit.get('speed_qualified')):
            raise ValueError('A passing prefix execution and speed audit is required')
        for name, checksum in prefix_audit['implementation_sha256'].items():
            if digest(Path(__file__).with_name(name)) != checksum:
                raise ValueError('Prefix implementation changed: ' + name)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report['admission'] = admission([a.device], 9)
        model, codec, parent, checksum = load_candidate(a.source, a.codec, a.checkpoint, [a.device])
        if (parent.get('architecture') != 'frozen-context-original-oxcoder-native-ncm-v33'
                or prefix_audit['source_sha256'] != parent['source_sha256']
                or prefix_audit['codec_sha256'] != parent['codec_sha256']):
            raise ValueError('Parent/source differs from the qualified architecture')
        for name, sha in parent['implementation_sha256'].items():
            if digest(Path(__file__).with_name(name)) != sha:
                raise ValueError('Parent implementation changed: ' + name)
        tower.set_compute_dtype(model.backbone, torch.float16)
        training, _, corpus = load_corpus(a.selection, a.manifest, parent['source_sha256'], parent['codec_sha256'])
        if corpus != parent['corpus']:
            raise ValueError('Parent training corpus differs')
        state = torch.load(a.checkpoint, weights_only=True, map_location='cpu')
        if digest(a.checkpoint) != checksum:
            raise ValueError('Checkpoint changed')
        cases, examples = [], []
        for size in (64, 69, 512, 2048):
            row, condition = min(training, key=lambda item: abs(len(item[1]['ids']) - size))
            ids = condition['ids']
            cases.append((row, {**condition, 'ids': (ids * ((size + len(ids) - 1) // len(ids)))[:size]}))
            examples.append({'id': row['id'], 'original_tokens': len(ids), 'prompt_tokens': size,
                             'synthetic_extension_or_truncation': len(ids) != size})
        report.update(phase='auditing', checkpoint_sha256=checksum, source_sha256=parent['source_sha256'],
                      codec_sha256=parent['codec_sha256'], corpus=corpus, examples=examples,
                      prefix_audit_sha256=digest(a.prefix_audit), implementation_sha256={n: digest(Path(__file__).with_name(n))
                          for n in ('audit_resident_runtime.py', 'resident_tower_optimizer.py',
                                    'interleaved_context.py', 'interleaved_prefix.py', 'frozen_clean_objective.py',
                                    'frozen_tower_optimizer.py', 'cpu_adamw.py', 'master_muon.py')})
        write_json(out / 'report.json', report)
        frozen = tower.source_versions(model.backbone)
        parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
        outcomes = {}
        for arm in ('control', 'resident'):
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(state['model'][name].to(parameter))
            optimizer = TowerOptimizer(model) if arm == 'control' else ResidentTowerOptimizer(model)
            copied = copy.deepcopy(state['optimizer'])
            if arm == 'control':
                optimizer.load_state_dict(copied)
            else:
                optimizer.import_legacy_state(copied)
            del copied
            torch.manual_seed(34004)
            torch.cuda.manual_seed_all(34004)
            records = []
            with installed() if arm == 'resident' else nullcontext():
                for update in range(4):
                    headroom([a.device])
                    optimizer.zero_grad()
                    torch.cuda.synchronize(a.device)
                    torch.cuda.reset_peak_memory_stats(a.device)
                    start = time.monotonic()
                    result = backward_batch(model, codec, cases, optimizer.loss_scale, .15)
                    norm = optimizer.step()
                    torch.cuda.synchronize(a.device)
                    record = {'update': update + 1, 'seconds': time.monotonic() - start,
                              'loss': sum(r['loss'] for r in result), 'gradient_norm': norm,
                              'peak_allocated_bytes': torch.cuda.max_memory_allocated(a.device),
                              'source_unchanged': tower.source_versions(model.backbone) == frozen,
                              'codec_has_gradient': any(p.grad is not None for p in codec.parameters())}
                    records.append(record)
                    print(json.dumps({'arm': arm, **record}), flush=True)
            outcomes[arm] = [p.detach().cpu().clone() for p in parameters.values()]
            report['arms'][arm] = {'records': records,
                                  'median_seconds': statistics.median(r['seconds'] for r in records[1:])}
            write_json(out / 'report.json', report)
            optimizer.zero_grad()
            del optimizer
        comparison = difference(outcomes['control'], outcomes['resident'])
        ratio = report['arms']['control']['median_seconds'] / report['arms']['resident']['median_seconds']
        records = [r for arm in report['arms'].values() for r in arm['records']]
        report.update(complete=True, parameter_comparison=comparison, speed_ratio=ratio,
                      passed=comparison['relative_l2'] < 1e-4 and comparison['maximum_absolute'] < 2e-4
                          and all(r['source_unchanged'] and not r['codec_has_gradient'] for r in records),
                      speed_qualified=ratio > 1.05)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        write_json(out / 'report.json', report)
        for unit in reversed(paused):
            subprocess.run(['systemctl', '--user', 'kill', '--kill-who=main', '--signal=CONT', unit], check=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'codec', 'checkpoint', 'selection', 'manifest', 'prefix-audit', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--wait-for-session', action='store_true')
    run(p.parse_args())
