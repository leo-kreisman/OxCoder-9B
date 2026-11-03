"""Full-update control for packed-only autograd without layer recomputation."""
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
from .resident_tower_optimizer import ResidentTowerOptimizer
from .frozen_clean_objective import backward_batch
from .interleaved_context import installed as interleaved
from .packed_gradient import installed as packed_gradient
from .audit_head_residency import difference
from .train import admission, headroom, write_json


def run(a):
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'complete': False, 'passed': False, 'quality_ready': False, 'arms': {},
              'scope': 'Four identical real training updates per runtime; all discarded'}
    supervisor = 'ncm-resident-pipeline-v34.service'
    worker = 'ncm-resident-tower-worker-v34.service'
    paused = False

    def active(unit):
        return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode == 0

    def phase(value):
        report['phase'] = value
        write_json(output / 'report.json', report)

    try:
        if a.wait_for_session:
            phase('waiting_for_resident_training_to_start')
            while not active(worker):
                if Path('results/ncm-resident-pipeline-v34/acceptance.json').exists():
                    phase('passing_candidate_takes_priority')
                    return
                if not active(supervisor):
                    raise RuntimeError('Resident training supervisor stopped before the audit')
                time.sleep(10)
            subprocess.run(['systemctl', '--user', 'kill', '--kill-who=main', '--signal=STOP', supervisor], check=True)
            paused = True
            phase('waiting_for_resident_session_to_finish')
            while active(worker):
                time.sleep(10)
        if Path('results/ncm-resident-tower-v34/candidate.json').exists():
            phase('passing_candidate_takes_priority')
            return
        matrices = json.loads(Path(a.matrix_audit).read_text())
        if not matrices.get('complete') or not matrices.get('passed'):
            raise ValueError('Representative matrix gradients did not qualify')
        for name, checksum in matrices['implementation_sha256'].items():
            if digest(Path(__file__).with_name(name)) != checksum:
                raise ValueError('Matrix implementation changed: ' + name)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report['admission'] = admission([a.device], 12)
        model, codec, parent, checksum = load_candidate(a.source, a.codec, a.checkpoint, [a.device])
        if (parent.get('architecture') != 'frozen-context-original-oxcoder-native-ncm-v34'
                or matrices['source_sha256'] != parent['source_sha256']):
            raise ValueError('Expected qualified resident runtime source')
        for name, sha in parent['implementation_sha256'].items():
            if digest(Path(__file__).with_name(name)) != sha:
                raise ValueError('Parent implementation changed: ' + name)
        tower.set_compute_dtype(model.backbone, torch.float16)
        training, _, corpus = load_corpus(a.selection, a.manifest, parent['source_sha256'], parent['codec_sha256'])
        if corpus != parent['corpus']:
            raise ValueError('Corpus differs')
        state = torch.load(a.checkpoint, weights_only=True, map_location='cpu')
        if digest(a.checkpoint) != checksum:
            raise ValueError('Parent checkpoint changed')
        cases, examples = [], []
        for size in (64, 69, 512, 2048):
            row, condition = min(training, key=lambda item: abs(len(item[1]['ids']) - size))
            ids = condition['ids']
            cases.append((row, {**condition, 'ids': (ids * ((size + len(ids) - 1) // len(ids)))[:size]}))
            examples.append({'id': row['id'], 'original_tokens': len(ids), 'prompt_tokens': size,
                             'synthetic_extension_or_truncation': len(ids) != size})
        report.update(checkpoint_sha256=checksum, source_sha256=parent['source_sha256'],
                      codec_sha256=parent['codec_sha256'], corpus=corpus, examples=examples,
                      matrix_audit_sha256=digest(a.matrix_audit), implementation_sha256={n: digest(Path(__file__).with_name(n))
                          for n in ('audit_packed_runtime.py', 'packed_gradient.py', 'gguf_training_weights.py',
                                    'resident_tower_optimizer.py', 'frozen_tower_ragged.py',
                                    'interleaved_context.py', 'interleaved_prefix.py', 'frozen_clean_objective.py')})
        phase('auditing')
        frozen = tower.source_versions(model.backbone)
        parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
        outcomes = {}
        for arm in ('resident_checkpointed', 'packed_no_recompute'):
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(state['model'][name].to(parameter))
            optimizer = ResidentTowerOptimizer(model)
            saved = copy.deepcopy(state['optimizer'])
            optimizer.load_state_dict(saved)
            del saved
            torch.manual_seed(35003)
            torch.cuda.manual_seed_all(35003)
            records = []
            with packed_gradient(model.backbone) if arm == 'packed_no_recompute' else nullcontext(), interleaved():
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
            write_json(output / 'report.json', report)
            optimizer.zero_grad()
            del optimizer
        comparison = difference(outcomes['resident_checkpointed'], outcomes['packed_no_recompute'])
        ratio = report['arms']['resident_checkpointed']['median_seconds'] / report['arms']['packed_no_recompute']['median_seconds']
        records = [r for arm in report['arms'].values() for r in arm['records']]
        report.update(complete=True, parameter_comparison=comparison, speed_ratio=ratio,
                      passed=comparison['relative_l2'] < 1e-4 and comparison['maximum_absolute'] < 2e-4
                          and all(r['source_unchanged'] and not r['codec_has_gradient'] for r in records),
                      speed_qualified=ratio > 1.05)
        if a.profile and report['passed']:
            optimizer = ResidentTowerOptimizer(model)
            optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
            optimizer.zero_grad()
            with packed_gradient(model.backbone), interleaved(), torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                backward_batch(model, codec, cases, optimizer.loss_scale, .15)
                optimizer.step()
                torch.cuda.synchronize(a.device)
            events = profile.key_averages()
            (output / 'profile.txt').write_text(events.table(sort_by='self_device_time_total', row_limit=40))
            write_json(output / 'profile.json', [{'operator': event.key, 'calls': event.count,
                'self_cpu_microseconds': event.self_cpu_time_total,
                'self_device_microseconds': event.self_device_time_total} for event in events])
            report['profile_scope'] = 'One additional discarded update; excluded from timing comparison'
    except BaseException as error:
        report['error'] = repr(error)
        report['complete'] = False
        report['passed'] = False
        raise
    finally:
        write_json(output / 'report.json', report)
        if paused:
            subprocess.run(['systemctl', '--user', 'kill', '--kill-who=main', '--signal=CONT', supervisor], check=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'codec', 'checkpoint', 'selection', 'manifest', 'matrix-audit', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--wait-for-session', action='store_true')
    p.add_argument('--profile', action='store_true')
    run(p.parse_args())
