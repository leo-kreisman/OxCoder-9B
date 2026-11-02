"""Checkpoint-preserving handoff to prompt-feature extraction and native NCM.

This runner owns only the explicitly named experiment services. It never
terminates unrelated GPU programs. Acceptance stays at twelve executable tasks
and a fresh-process replay; training/probe completion is never acceptance.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def read(path):
    try: return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError): return {}


def active(unit):
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode == 0


def run(a):
    root = Path(a.root).resolve()
    out = root / 'results/ncm-context-pipeline-v25'
    out.mkdir(parents=True, exist_ok=a.resume_features)
    def status(phase, **values):
        value = {'phase': phase, 'time': time.time(), 'quality_ready': False, **values}
        temp = out / 'progress.tmp'; temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(out / 'progress.json'); print(json.dumps(value), flush=True)
    def launch(name, arguments, memory='16G', seconds=43200):
        if active(name + '.service'): raise RuntimeError('Experiment service already active')
        subprocess.run(['systemctl', '--user', 'reset-failed', name + '.service'], capture_output=True)
        subprocess.run(['systemd-run', '--user', '--unit=' + name, '--collect',
            '--property=WorkingDirectory=' + str(root), '--property=MemoryMax=' + memory,
            '--property=MemorySwapMax=0', '--property=TasksMax=128',
            '--property=RuntimeMaxSec=' + str(seconds),
            '--property=StandardOutput=append:' + str(root / 'results' / (name + '.log')),
            '--property=StandardError=append:' + str(root / 'results' / (name + '.log')),
            '--setenv=PYTHONPATH=results/ncm-kernels:results/ncm-deps:results/ncm-data-deps',
            '--setenv=OMP_NUM_THREADS=2', '--setenv=TOKENIZERS_PARALLELISM=false',
            '--setenv=HF_HUB_OFFLINE=1', '--setenv=PYTORCH_ALLOC_CONF=expandable_segments:True',
            a.python, *arguments], check=True)
    try:
        from .backbone import digest
        if a.microbatch != 2:
            report = read(a.batch_audit) if a.batch_audit else {}
            matches = [r for r in report.get('results', []) if r['microbatch'] == a.microbatch]
            if (not report.get('complete') or not report.get('all_passed') or len(matches) != 1
                    or not matches[0]['passed'] or matches[0]['relative_gradient_l2'] > .02
                    or matches[0]['relative_loss_delta'] > .01
                    or any(report['implementation_sha256'][name] != digest(root / 'scripts/ncm' / name)
                           for name in ('context_decoder.py', 'train_context_decoder.py'))):
                raise ValueError('Requested microbatch lacks the current numerical audit')
            (out / 'batch-policy.json').write_text(json.dumps({'microbatch': a.microbatch,
                'audit_sha256': digest(a.batch_audit), 'audit': str(Path(a.batch_audit).resolve())}, indent=2) + '\n')
        if a.resume_features:
            handoff = read(out / 'handoff.json')
            if (not handoff.get('optimizer_and_rng_preserved')
                    or digest(handoff['checkpoint']) != handoff['sha256']
                    or active('ncm-parallel-v24.service')):
                raise ValueError('Feature-stage resumption requires the unchanged preserved source run')
            status('resuming_feature_monitor', microbatch=a.microbatch)
            units = [f'ncm-context-features-v25-{i}' for i in (0, 1)]
        else:
            units = begin_features(a, root, out, status, launch)
        while any(active(x + '.service') for x in units): time.sleep(20)
        if not all(read(root / 'results' / x / 'manifest.json').get('complete') for x in units):
            raise RuntimeError('Feature extraction incomplete; see shard logs')
        status('validating_feature_shards')
        subprocess.run([a.python, '-m', 'scripts.ncm.context_cache', '--selection',
            str(root / 'results/ncm-context-selection-v25'), '--shards',
            *[str(root / 'results' / x) for x in units], '--output', str(out / 'features.json')], cwd=root, check=True)
        status('training_native_decoder')
        launch('ncm-context-training-v25', ['-m', 'scripts.ncm.train_context_decoder',
            '--cache', str(out / 'features.json'), '--codec', str(root / 'results/ncm-contextual-decoder-v9/best.pt'),
            '--output', str(root / 'results/ncm-context-training-v25'), '--microbatch', str(a.microbatch)], memory='12G')
        monitor(a, root, out, status, launch)
    except BaseException as error:
        status('error', error=repr(error))
        raise


def begin_features(a, root, out, status, launch):
    status('waiting_for_preserved_v24_checkpoint', boundary=a.boundary)
    while read(root / 'results/ncm-parallel-v24/progress.json').get('step', 0) < a.boundary:
        if not active('ncm-parallel-v24.service'):
            raise RuntimeError('Existing trainer stopped before handoff boundary')
        time.sleep(10)
    # Progress is written only after the trainer atomically publishes its
    # checkpoint and both ranks cross the save barrier.
    subprocess.run(['systemctl', '--user', 'stop', 'ncm-lr-comparison-v24.service'], check=True)
    subprocess.run(['systemctl', '--user', 'stop', 'ncm-parallel-v24.service'], check=True)
    import torch
    from .backbone import digest
    preserved = root / 'results/ncm-parallel-v24/latest.pt'
    state = torch.load(preserved, weights_only=True, map_location='cpu')
    if state['step'] < a.boundary:
        raise RuntimeError('Latest checkpoint predates handoff boundary')
    handoff = {'checkpoint': str(preserved), 'sha256': digest(preserved), 'step': state['step'],
               'optimizer_and_rng_preserved': 'optimizer' in state and 'rng' in state,
               'reason': 'Independent encoder-decoder larger-batch experiment; v24 remains resumable'}
    (out / 'handoff.json').write_text(json.dumps(handoff, indent=2) + '\n')
    del state
    import gc; gc.collect()
    status('extracting_prompt_only_features', handoff=handoff)
    units = []
    for i in (0, 1):
        unit = f'ncm-context-features-v25-{i}'
        launch(unit, ['-m', 'scripts.ncm.context_features', 'extract', '--source', a.source,
            '--codec', str(root / 'results/ncm-contextual-decoder-v9/best.pt'),
            '--selection', str(root / 'results/ncm-context-selection-v25'),
            '--output', str(root / 'results' / unit), '--device', f'cuda:{i}', '--shard', str(i), '--shards', '2'])
        units.append(unit)
    return units


def monitor(a, root, out, status, launch):
    evaluated = set()
    while True:
        folder = root / 'results/ncm-context-training-v25'
        for checkpoint in sorted(folder.glob('evaluated-*.pt'), key=lambda x: int(x.stem.split('-')[-1])):
            if checkpoint.name in evaluated: continue
            step = int(checkpoint.stem.split('-')[-1])
            target = out / f'coding-{step}.json'
            launch('ncm-context-evaluation-v25', ['-m', 'scripts.ncm.context_runtime',
                '--source', a.source, '--codec', str(root / 'results/ncm-contextual-decoder-v9/best.pt'),
                '--checkpoint', str(checkpoint), '--output', str(target), '--device', 'cuda:0'])
            while active('ncm-context-evaluation-v25.service'): time.sleep(10)
            report = read(target)
            if not report.get('complete'): raise RuntimeError('Executable evaluation did not complete')
            evaluated.add(checkpoint.name)
            status('evaluated_native_decoder', step=step, passed=report['passed'], evaluated=report['evaluated'])
            if report.get('passed_all'):
                replay = out / f'cold-replay-{step}.json'
                launch('ncm-context-replay-v25', ['-m', 'scripts.ncm.context_runtime',
                    '--source', a.source, '--codec', str(root / 'results/ncm-contextual-decoder-v9/best.pt'),
                    '--checkpoint', str(checkpoint), '--output', str(replay), '--device', 'cuda:0'])
                while active('ncm-context-replay-v25.service'): time.sleep(10)
                second = read(replay)
                identical = (second.get('passed_all') and second.get('checkpoint_sha256') == report['checkpoint_sha256']
                    and [x['generation']['text'] for x in second['tasks']] == [x['generation']['text'] for x in report['tasks']])
                if not identical: raise RuntimeError('Passing checkpoint failed fresh-process replay')
                (folder / 'STOP_AFTER_CHECKPOINT').touch()
                (out / 'acceptance.json').write_text(json.dumps({'checkpoint': str(checkpoint),
                    'checkpoint_sha256': report['checkpoint_sha256'], 'passed': 12, 'evaluated': 12,
                    'cold_replay_identical': True, 'scope': 'Twelve elementary functions; not broad professional qualification'}, indent=2) + '\n')
                status('twelve_task_gate_and_cold_replay_passed', checkpoint=str(checkpoint))
                return
        if not active('ncm-context-training-v25.service'):
            status('training_stopped_without_acceptance', completed=read(folder / 'completed.json'))
            return
        time.sleep(20)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--python', required=True)
    p.add_argument('--source', required=True); p.add_argument('--boundary', type=int, default=550)
    p.add_argument('--resume-features', action='store_true')
    p.add_argument('--microbatch', type=int, default=2); p.add_argument('--batch-audit')
    run(p.parse_args())
