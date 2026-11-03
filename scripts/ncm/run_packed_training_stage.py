"""Adopt packed-gradient execution only after full-update and data audits."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

from .run_frozen_ragged_stage import active, read, sha, verify_replay
from .run_compact_curriculum_stage import signal_running


def saved_boundary(root):
    folder = Path(root) / 'results/ncm-resident-tower-v34'
    checkpoint = folder / 'evaluated-2500.pt'
    report = read(folder / 'coding-2500.json')
    if (read(folder / 'session.json').get('step') != 2500
            or read(folder / 'progress.json').get('step') != 2500
            or report.get('evaluated') != 12 or report.get('checkpoint_sha256') != sha(checkpoint)):
        raise ValueError('Expected a complete saved/evaluated compact update-2500 boundary')
    if report.get('passed') == 12:
        raise ValueError('Existing passing candidate takes priority')
    return checkpoint


def run(a):
    root = Path(a.root).resolve()
    pipeline = root / 'results/ncm-packed-pipeline-v35'
    pipeline.mkdir(exist_ok=True)
    with (pipeline / 'owner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(a, root, pipeline)


def run_locked(a, root, pipeline):
    old = 'ncm-resident-pipeline-v34.service'
    old_worker = 'ncm-resident-tower-worker-v34.service'
    audit_unit = 'ncm-packed-runtime-audit-v35.service'
    audit_file = root / 'results/ncm-packed-runtime-audit-v35/report.json'
    coordinator = 'ncm-complete-pipeline-v28.service'
    evaluator = 'ncm-context-evaluation-v28.service'
    child = 'ncm-packed-tower-worker-v35.service'
    folder = root / 'results/ncm-packed-tower-v35'

    def status(**values):
        record = {'time': time.time(), 'quality_ready': False, **values}
        temp = pipeline / 'progress.tmp'
        temp.write_text(json.dumps(record, indent=2) + '\n')
        temp.replace(pipeline / 'progress.json')
        print(json.dumps(record), flush=True)

    def earlier_candidate():
        return any((root / 'results' / name / 'candidate.json').exists() for name in
                   ('ncm-resident-tower-v34', 'ncm-frozen-clean-training-v32'))

    while active(old_worker) or not (root / 'results/ncm-resident-tower-v34/session.json').exists() \
            or read(root / 'results/ncm-resident-tower-v34/session.json').get('step', 0) < 2500:
        if earlier_candidate():
            status(phase='earlier_candidate_takes_priority')
            return
        if not active(audit_unit):
            raise RuntimeError('Required audit stopped before the handover boundary')
        status(phase='waiting_for_saved_compact_session')
        time.sleep(10)
    if earlier_candidate():
        status(phase='earlier_candidate_takes_priority')
        return
    parent = saved_boundary(root)
    provisional = False
    try:
        # The audit has paused this supervisor and owns GPU1 after its worker
        # finishes. Retire the supervisor now so audit cleanup cannot launch a
        # duplicate old worker. Its exact saved checkpoint remains recoverable.
        if active(old):
            subprocess.run(['systemctl', '--user', 'stop', old], check=True)
            provisional = True
        signal_running(coordinator, 'STOP')
        while active(audit_unit):
            status(phase='waiting_for_full_runtime_qualification')
            time.sleep(10)
        audit = read(audit_file)
        if not (audit.get('complete') and audit.get('passed') and audit.get('speed_qualified')
                and audit.get('checkpoint_sha256') == sha(parent)):
            raise ValueError('Full runtime qualification did not pass')
        reader_path = root / 'results/ncm-selection-reader-audit-v35.json'
        while active('ncm-selection-reader-audit-v35.service'):
            status(phase='waiting_for_serialization_qualification')
            time.sleep(10)
        reader = read(reader_path)
        if not (reader.get('complete') and reader.get('passed')
                and reader.get('corpus') == audit.get('corpus')):
            raise ValueError('Whole-corpus serialization qualification did not pass')
        (pipeline / 'handover.json').write_text(json.dumps({
            'parent_checkpoint': str(parent), 'parent_sha256': sha(parent),
            'runtime_audit_sha256': sha(audit_file), 'reader_audit_sha256': sha(reader_path), 'step': 2500,
            'policy': 'Preserve all training state; change qualified execution only',
            'quality_ready': False}, indent=2) + '\n')
    except BaseException:
        if provisional and not active(old) and not active(child):
            # A failed optimization must not strand the existing training run.
            subprocess.run(['systemd-run', '--user', '--collect', '--unit=' + old.removesuffix('.service'),
                '--property=WorkingDirectory=' + str(root), '--property=MemoryMax=512M',
                '--property=RuntimeMaxSec=172800',
                '--property=ExecStopPost=-/usr/bin/python3 -S -m scripts.ncm.run_resident_training_stage --cleanup',
                '/usr/bin/python3', '-S', '-m', 'scripts.ncm.run_resident_training_stage',
                '--root', str(root), '--python', a.python, '--source', a.source], check=True)
        signal_running(coordinator, 'CONT')
        raise

    def launch(extra):
        if active(child) or active(old_worker):
            raise RuntimeError('Another source-tower worker is active on GPU1')
        signal_running(coordinator, 'STOP')
        try:
            while active(evaluator):
                status(phase='waiting_for_native_evaluator')
                time.sleep(10)
            command = ['systemd-run', '--user', '--collect', '--unit=' + child.removesuffix('.service'),
                '--property=WorkingDirectory=' + str(root), '--property=MemoryMax=18G',
                '--property=MemorySwapMax=0', '--property=TasksMax=128', '--property=RuntimeMaxSec=21600',
                '--property=ExecStopPost=-/usr/bin/systemctl --user kill --signal=CONT ' + coordinator,
                '--property=StandardOutput=append:' + str(pipeline / 'worker.log'),
                '--property=StandardError=append:' + str(pipeline / 'worker.log'),
                '--setenv=PYTHONPATH=results/ncm-kernels:results/ncm-deps:results/ncm-data-deps',
                '--setenv=OMP_NUM_THREADS=2', '--setenv=TOKENIZERS_PARALLELISM=false',
                '--setenv=HF_HUB_OFFLINE=1', '--setenv=PYTORCH_ALLOC_CONF=expandable_segments:True',
                a.python, '-m', 'scripts.ncm.train_packed_tower', '--source', a.source,
                '--codec', 'results/ncm-contextual-decoder-v9/best.pt',
                '--execution-audit', 'results/ncm-frozen-ragged-audit-v31/report.json',
                '--device', 'cuda:1', *extra]
            subprocess.run(command, check=True)
            while active(child):
                time.sleep(10)
        finally:
            signal_running(coordinator, 'CONT')

    while True:
        candidate = read(folder / 'candidate.json')
        if candidate:
            checkpoint = Path(candidate['checkpoint'])
            checksum = sha(checkpoint)
            if checksum != candidate['checkpoint_sha256']:
                raise ValueError('Candidate changed')
            first = folder / ('coding-' + checkpoint.stem.split('-')[-1] + '.json')
            cold = pipeline / ('cold-' + checkpoint.stem)
            if not (cold / 'coding.json').exists():
                status(phase='fresh_process_replay', checkpoint=str(checkpoint))
                launch(['--init', str(checkpoint), '--output', str(cold), '--evaluate-only'])
            verify_replay(read(first), read(cold / 'coding.json'), checksum)
            acceptance = {'checkpoint': str(checkpoint), 'checkpoint_sha256': checksum,
                          'passed': 12, 'evaluated': 12, 'cold_replay_identical': True,
                          'first_report_sha256': sha(first), 'cold_report_sha256': sha(cold / 'coding.json'),
                          'scope': 'Twelve elementary coding tasks; broader deployment qualification separate'}
            (pipeline / 'acceptance.json').write_text(json.dumps(acceptance, indent=2) + '\n')
            status(phase='twelve_task_gate_and_cold_replay_passed', **acceptance)
            return
        if earlier_candidate():
            status(phase='earlier_candidate_takes_priority')
            return
        previous = read(folder / 'session.json').get('step', 2500)
        if previous >= 10000:
            status(phase='schedule_exhausted_without_acceptance', step=previous)
            return
        status(phase='packed_gradient_training', start_step=previous)
        extra = ['--init', str(parent), '--output', str(folder), '--runtime-audit', str(audit_file),
                 '--selection', 'results/ncm-compact-training-v33',
                 '--manifest', 'results/ncm-compact-training-v33/admission.json',
                 '--reader-audit', str(root / 'results/ncm-selection-reader-audit-v35.json'),
                 '--selection-cache', str(root / 'results/ncm-compact-json-v35'),
                 '--session-steps', '1000', '--eval-every', '250']
        if (folder / 'latest.pt').exists():
            extra.append('--resume')
        launch(extra)
        if (folder / 'candidate.json').exists():
            continue
        current = read(folder / 'session.json').get('step', 0)
        if current <= previous or current != read(folder / 'progress.json').get('step'):
            raise RuntimeError('Resident worker did not complete its saved session')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cleanup', action='store_true')
    for name in ('root', 'python', 'source'):
        p.add_argument('--' + name)
    a = p.parse_args()
    if a.cleanup:
        subprocess.run(['systemctl', '--user', 'stop', 'ncm-packed-tower-worker-v35.service'], capture_output=True)
        signal_running('ncm-complete-pipeline-v28.service', 'CONT')
    else:
        if any(getattr(a, key) is None for key in ('root', 'python', 'source')):
            p.error('root, python and source required')
        run(a)
