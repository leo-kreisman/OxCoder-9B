"""Run the admitted short-code curriculum on GPU1 after a saved control boundary.

Only named experiment services are managed. GPU0's canonical-label control
continues. Original optimizer state and the twelve-task gate remain intact.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
from .run_frozen_ragged_stage import active, read, sha, verify_replay


def signal_running(unit, sig):
    result = subprocess.run(['systemctl', '--user', 'kill', '--kill-who=main', '--signal=' + sig, unit],
                            capture_output=True)
    if result.returncode and active(unit):
        raise RuntimeError('Could not signal running experiment unit: ' + unit)


def run(a):
    pipeline = Path(a.root).resolve() / 'results/ncm-compact-curriculum-pipeline-v33'
    pipeline.mkdir(exist_ok=True)
    with (pipeline / 'owner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(a)


def run_locked(a):
    root = Path(a.root).resolve()
    pipeline = root / 'results/ncm-compact-curriculum-pipeline-v33'
    folder = root / 'results/ncm-compact-tower-v33'
    parent = root / 'results/ncm-frozen-clean-training-v32/evaluated-1000.pt'
    parent_report = parent.with_name('coding-1000.json')
    manifest = root / 'results/ncm-compact-training-v33/admission.json'
    smoke = root / 'results/ncm-compact-optimizer-v33/report.json'
    pipeline.mkdir(exist_ok=True)
    child = 'ncm-compact-tower-worker-v33.service'
    previous_parent = 'ncm-frozen-ragged-pipeline-v31.service'
    previous_child = 'ncm-frozen-ragged-training-v31.service'
    coordinator = 'ncm-complete-pipeline-v28.service'
    evaluator = 'ncm-context-evaluation-v28.service'

    def status(**values):
        result = {'time': time.time(), 'quality_ready': False, **values}
        temporary = pipeline / 'progress.tmp'
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(pipeline / 'progress.json')
        print(json.dumps(result), flush=True)

    def canonical_candidate():
        return (root / 'results/ncm-frozen-clean-training-v32/candidate.json').exists()

    while not (manifest.exists() and parent.exists() and read(parent_report).get('evaluated') == 12):
        if canonical_candidate():
            status(phase='canonical_candidate_takes_priority')
            return
        status(phase='waiting_for_complete_curriculum_and_parent')
        time.sleep(30)
    if (not read(manifest).get('complete') or read(parent_report).get('checkpoint_sha256') != sha(parent)):
        raise ValueError('Incomplete curriculum or mismatched parent evaluation')
    if canonical_candidate():
        status(phase='canonical_candidate_takes_priority')
        return

    # Freeze the old supervisor, not its separately managed worker. Let that
    # worker save and evaluate its full session before retiring the old arm.
    if active(previous_parent):
        signal_running(previous_parent, 'STOP')
    while active(previous_child):
        status(phase='waiting_for_old_control_saved_boundary')
        time.sleep(10)
    old_folder = root / 'results/ncm-frozen-ragged-training-v31'
    old_step = read(old_folder / 'session.json').get('step', 0)
    if (old_step < 1000 or old_step != read(old_folder / 'progress.json').get('step')
            or read(old_folder / f'coding-{old_step}.json').get('evaluated') != 12):
        signal_running(previous_parent, 'CONT')
        raise ValueError('Old control did not reach a complete evaluated session')
    if active(previous_parent):
        subprocess.run(['systemctl', '--user', 'stop', previous_parent], check=True)
    (pipeline / 'retired-control.json').write_text(json.dumps({
        'unit': previous_parent, 'step': old_step,
        'checkpoint': str(old_folder / f'evaluated-{old_step}.pt'),
        'coding_report_sha256': sha(old_folder / f'coding-{old_step}.json'),
        'reason': 'Allocate GPU1 to independently checked compact-code curriculum; GPU0 control continues',
        'quality_ready': False}, indent=2) + '\n')

    def launch(module, extra):
        if active(child):
            raise RuntimeError('Compact worker is already running')
        signal_running(coordinator, 'STOP')
        try:
            while active(evaluator):
                status(phase='waiting_for_existing_native_evaluator')
                time.sleep(10)
            command = ['systemd-run', '--user', '--unit=' + child.removesuffix('.service'), '--collect',
                '--property=WorkingDirectory=' + str(root), '--property=MemoryMax=18G',
                '--property=MemorySwapMax=0', '--property=TasksMax=128', '--property=RuntimeMaxSec=21600',
                '--property=ExecStopPost=-/usr/bin/systemctl --user kill --signal=CONT ' + coordinator,
                '--property=StandardOutput=append:' + str(pipeline / 'worker.log'),
                '--property=StandardError=append:' + str(pipeline / 'worker.log'),
                '--setenv=PYTHONPATH=results/ncm-kernels:results/ncm-deps:results/ncm-data-deps',
                '--setenv=OMP_NUM_THREADS=2', '--setenv=TOKENIZERS_PARALLELISM=false',
                '--setenv=HF_HUB_OFFLINE=1', '--setenv=PYTORCH_ALLOC_CONF=expandable_segments:True',
                a.python, '-m', module, '--source', a.source,
                '--codec', 'results/ncm-contextual-decoder-v9/best.pt',
                '--execution-audit', 'results/ncm-frozen-ragged-audit-v31/report.json',
                '--device', 'cuda:1', *extra]
            subprocess.run(command, check=True)
            while active(child):
                time.sleep(10)
        finally:
            signal_running(coordinator, 'CONT')

    common = ['--selection', 'results/ncm-compact-training-v33', '--manifest', str(manifest)]
    if not smoke.exists():
        status(phase='expanded_corpus_optimizer_qualification')
        launch('scripts.ncm.audit_frozen_clean_optimizer', common +
               ['--checkpoint', str(parent), '--output', str(smoke.parent)])
    if not read(smoke).get('complete') or not read(smoke).get('passed'):
        raise RuntimeError('Expanded-corpus optimizer qualification failed')
    while True:
        candidate = read(folder / 'candidate.json')
        if candidate:
            checkpoint = Path(candidate['checkpoint'])
            checksum = sha(checkpoint)
            if checksum != candidate['checkpoint_sha256']:
                raise ValueError('Candidate changed')
            first_path = folder / ('coding-' + checkpoint.stem.split('-')[-1] + '.json')
            cold = pipeline / ('cold-' + checkpoint.stem)
            if not (cold / 'coding.json').exists():
                status(phase='fresh_process_replay', checkpoint=str(checkpoint))
                launch('scripts.ncm.train_compact_tower', ['--init', str(checkpoint),
                    '--output', str(cold), '--evaluate-only'])
            verify_replay(read(first_path), read(cold / 'coding.json'), checksum)
            acceptance = {'checkpoint': str(checkpoint), 'checkpoint_sha256': checksum,
                'passed': 12, 'evaluated': 12, 'cold_replay_identical': True,
                'first_report_sha256': sha(first_path), 'cold_report_sha256': sha(cold / 'coding.json'),
                'scope': 'Twelve elementary coding tasks; broader professional quality not established'}
            (pipeline / 'acceptance.json').write_text(json.dumps(acceptance, indent=2) + '\n')
            status(phase='twelve_task_gate_and_cold_replay_passed', **acceptance)
            return
        previous = read(folder / 'session.json').get('step', 1000)
        if previous >= 10000:
            status(phase='schedule_exhausted_without_acceptance', step=previous)
            return
        status(phase='compact_curriculum_training', start_step=previous)
        extra = common + ['--init', str(parent), '--output', str(folder),
            '--base-selection', 'results/ncm-complete-selection-v27',
            '--base-manifest', 'results/ncm-complete-pipeline-v28/features.json',
            '--optimizer-audit', str(smoke), '--session-steps', '500', '--eval-every', '250']
        if (folder / 'latest.pt').exists():
            extra.append('--resume')
        launch('scripts.ncm.train_compact_tower', extra)
        if (folder / 'candidate.json').exists():
            continue
        current = read(folder / 'session.json').get('step', 0)
        if current <= previous or current != read(folder / 'progress.json').get('step'):
            raise RuntimeError('Compact worker did not finish a saved session')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cleanup', action='store_true')
    for name in ('root', 'python', 'source'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if args.cleanup:
        subprocess.run(['systemctl', '--user', 'stop', 'ncm-compact-tower-worker-v33.service'], capture_output=True)
        for unit in ('ncm-complete-pipeline-v28.service', 'ncm-frozen-ragged-pipeline-v31.service'):
            subprocess.run(['systemctl', '--user', 'kill', '--signal=CONT', unit], capture_output=True)
    else:
        if any(getattr(args, name) is None for name in ('root', 'python', 'source')):
            parser.error('root, python and source are required')
        run(args)
