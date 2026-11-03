"""Run byte-supervised continuation after retiring a saved compact-data session."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
from .run_frozen_ragged_stage import active, read, sha, verify_replay


OLD = 'ncm-resident-pipeline-v34.service'
OLD_WORKER = 'ncm-resident-tower-worker-v34.service'
CHILD = 'ncm-decoded-hierarchy-worker-v40.service'


def signal(unit, value):
    if active(unit):
        subprocess.run(['systemctl', '--user', 'kill', '--kill-who=main', '--signal=' + value, unit], check=True)


def run(a):
    root = Path(a.root).resolve()
    pipeline = root / 'results/ncm-decoded-hierarchy-pipeline-v40'
    folder = root / 'results/ncm-decoded-hierarchy-v40'
    parent = root / 'results/ncm-deterministic-student-v38/evaluated-3250.pt'
    smoke = root / 'results/ncm-decoded-hierarchy-audit-v40/report.json'
    pipeline.mkdir(exist_ok=True)
    lock = (pipeline / 'owner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def status(**values):
        result = {'time': time.time(), 'quality_ready': False, **values}
        temporary = pipeline / 'progress.tmp'
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(pipeline / 'progress.json')
        print(json.dumps(result), flush=True)

    def launch(module, extra):
        if active(CHILD) or active(OLD_WORKER):
            raise RuntimeError('GPU1 experiment worker is still active')
        command = ['systemd-run', '--user', '--unit=' + CHILD.removesuffix('.service'), '--collect',
            '--property=WorkingDirectory=' + str(root), '--property=MemoryMax=18G',
            '--property=MemorySwapMax=0', '--property=TasksMax=128', '--property=RuntimeMaxSec=21600',
            '--property=StandardOutput=append:' + str(pipeline / 'worker.log'),
            '--property=StandardError=append:' + str(pipeline / 'worker.log'),
            '--setenv=PYTHONPATH=results/ncm-kernels:results/ncm-deps:results/ncm-data-deps',
            '--setenv=OMP_NUM_THREADS=2', '--setenv=TOKENIZERS_PARALLELISM=false',
            '--setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8', '--setenv=HF_HUB_OFFLINE=1',
            '--setenv=PYTORCH_ALLOC_CONF=expandable_segments:True',
            a.python, '-m', module, '--source', a.source,
            '--codec', 'results/ncm-contextual-decoder-v9/best.pt',
            '--execution-audit', 'results/ncm-frozen-ragged-audit-v31/report.json',
            '--device', 'cuda:1', *extra]
        subprocess.run(command, check=True)
        while active(CHILD):
            time.sleep(10)

    # The v38 comparison remains running on GPU0. Do not pause it.
    signal(OLD, 'STOP')
    try:
        while active(OLD_WORKER):
            status(phase='waiting_for_saved_compact_update_4000')
            time.sleep(10)
        old_folder = root / 'results/ncm-resident-tower-v34'
        if (old_folder / 'candidate.json').exists():
            status(phase='existing_candidate_takes_priority')
            return
        old_checkpoint = old_folder / 'evaluated-4000.pt'
        old_report = read(old_folder / 'coding-4000.json')
        if (read(old_folder / 'session.json').get('step') != 4000
                or old_report.get('evaluated') != 12
                or old_report.get('checkpoint_sha256') != sha(old_checkpoint)):
            raise ValueError('Expected a saved and completely evaluated compact update 4000')
        parent_report = read(parent.parent / 'coding-3250.json')
        if parent_report.get('evaluated') != 12 or parent_report.get('checkpoint_sha256') != sha(parent):
            raise ValueError('Byte objective requires the immutable evaluated v38 parent')
        if parent_report.get('passed') == 12:
            status(phase='student_candidate_takes_priority')
            return
        common = ['--selection', 'results/ncm-complete-selection-v27',
                  '--manifest', 'results/ncm-complete-pipeline-v28/features.json']
        if not smoke.exists():
            status(phase='decoded_hierarchy_qualification', parent=str(parent))
            launch('scripts.ncm.audit_decoded_hierarchy', common +
                   ['--checkpoint', str(parent), '--output', str(smoke.parent)])
        audit = read(smoke)
        if (not audit.get('complete') or not audit.get('passed')
                or not audit.get('disabled_byte_exact') or audit.get('checkpoint_sha256') != sha(parent)):
            raise ValueError('Byte-objective qualification failed; resume compact supervisor')
        if active(OLD):
            subprocess.run(['systemctl', '--user', 'stop', OLD], check=True)
        (pipeline / 'handover.json').write_text(json.dumps({
            'parent': str(parent), 'parent_sha256': sha(parent), 'parent_step': 3250,
            'retired_checkpoint': str(old_checkpoint), 'retired_sha256': sha(old_checkpoint),
            'audit_sha256': sha(smoke), 'quality_ready': False,
            'scope': 'Add training-only byte loss; preserve v38 data/optimizer/sampler/global schedule; GPU1 execution'},
            indent=2) + '\n')
    finally:
        signal(OLD, 'CONT')

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
                launch('scripts.ncm.train_decoded_hierarchy', ['--init', str(checkpoint),
                    '--output', str(cold), '--evaluate-only'])
            verify_replay(read(first_path), read(cold / 'coding.json'), checksum)
            acceptance = {'checkpoint': str(checkpoint), 'checkpoint_sha256': checksum,
                'passed': 12, 'evaluated': 12, 'cold_replay_identical': True,
                'first_report_sha256': sha(first_path), 'cold_report_sha256': sha(cold / 'coding.json'),
                'scope': 'Twelve elementary functions; broader professional quality not established'}
            (pipeline / 'acceptance.json').write_text(json.dumps(acceptance, indent=2) + '\n')
            status(phase='twelve_task_gate_and_cold_replay_passed', **acceptance)
            return
        previous = read(folder / 'session.json').get('step', 3250)
        if previous >= 10000:
            status(phase='schedule_exhausted_without_acceptance', step=previous)
            return
        extra = common + ['--init', str(parent), '--output', str(folder),
            '--optimizer-audit', str(smoke), '--session-steps', '1000', '--eval-every', '250']
        if (folder / 'latest.pt').exists():
            extra.append('--resume')
        status(phase='decoded_hierarchy_training', start_step=previous)
        launch('scripts.ncm.train_decoded_hierarchy', extra)
        if (folder / 'candidate.json').exists():
            continue
        current = read(folder / 'session.json').get('step', 0)
        if current <= previous or current != read(folder / 'progress.json').get('step'):
            raise RuntimeError('Worker did not save a complete new session')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cleanup', action='store_true')
    for name in ('root', 'python', 'source'):
        parser.add_argument('--' + name)
    a = parser.parse_args()
    if a.cleanup:
        subprocess.run(['systemctl', '--user', 'stop', CHILD], capture_output=True)
        signal(OLD, 'CONT')
    else:
        if any(getattr(a, name) is None for name in ('root', 'python', 'source')):
            parser.error('root, python and source are required')
        run(a)
