"""Qualify and run student-conditioned canonical NCM training at a saved boundary."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
from .run_frozen_ragged_stage import active, read, sha, verify_replay


def run(a):
    root = Path(a.root).resolve()
    pipeline = root / 'results/ncm-canonical-student-pipeline-v36'
    folder = root / 'results/ncm-canonical-student-v36'
    parent = root / 'results/ncm-frozen-clean-training-v32/evaluated-3000.pt'
    smoke = root / 'results/ncm-canonical-student-audit-v36/report.json'
    pipeline.mkdir(exist_ok=True)
    lock = (pipeline / 'owner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = 'ncm-canonical-student-worker-v36.service'

    def status(**data):
        result = {'time': time.time(), 'quality_ready': False, **data}
        temporary = pipeline / 'progress.tmp'
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(pipeline / 'progress.json')
        print(json.dumps(result), flush=True)

    def launch(module, extra):
        if active(child):
            raise RuntimeError('Control worker already running')
        command = ['systemd-run', '--user', '--unit=' + child.removesuffix('.service'), '--collect',
            '--property=WorkingDirectory=' + str(root), '--property=MemoryMax=18G',
            '--property=MemorySwapMax=0', '--property=TasksMax=128', '--property=RuntimeMaxSec=21600',
            '--property=StandardOutput=append:' + str(pipeline / 'worker.log'),
            '--property=StandardError=append:' + str(pipeline / 'worker.log'),
            '--setenv=PYTHONPATH=results/ncm-kernels:results/ncm-deps:results/ncm-data-deps',
            '--setenv=OMP_NUM_THREADS=2', '--setenv=TOKENIZERS_PARALLELISM=false',
            '--setenv=HF_HUB_OFFLINE=1', '--setenv=PYTORCH_ALLOC_CONF=expandable_segments:True',
            a.python, '-m', module, '--source', a.source,
            '--codec', 'results/ncm-contextual-decoder-v9/best.pt',
            '--execution-audit', 'results/ncm-frozen-ragged-audit-v31/report.json',
            '--device', 'cuda:0', *extra]
        subprocess.run(command, check=True)
        while active(child):
            time.sleep(10)

    old = 'ncm-frozen-clean-pipeline-v32.service'
    old_worker = 'ncm-frozen-clean-worker-v32.service'
    old_folder = root / 'results/ncm-frozen-clean-training-v32'
    if active(old):
        subprocess.run(['systemctl', '--user', 'kill', '--kill-who=main', '--signal=STOP', old], check=True)
    try:
        while active(old_worker):
            status(phase='waiting_for_saved_canonical_session')
            time.sleep(10)
        if (old_folder / 'candidate.json').exists():
            status(phase='existing_candidate_takes_priority')
            return
        first = read(old_folder / 'coding-3000.json')
        if (read(old_folder / 'session.json').get('step') != 3000
                or first.get('evaluated') != 12 or first.get('checkpoint_sha256') != sha(parent)):
            raise ValueError('Expected a complete saved and evaluated update-3000 parent')
        common = ['--selection', 'results/ncm-complete-selection-v27',
                  '--manifest', 'results/ncm-complete-pipeline-v28/features.json']
        if not smoke.exists():
            status(phase='student_conditioning_qualification', parent=str(parent))
            launch('scripts.ncm.audit_canonical_onpolicy', common +
                   ['--checkpoint', str(parent), '--output', str(smoke.parent)])
        audit = read(smoke)
        if (not audit.get('complete') or not audit.get('passed')
                or not audit.get('disabled_student_exact') or audit.get('checkpoint_sha256') != sha(parent)):
            raise ValueError('Student-conditioned training qualification did not pass')
        # Retire only this supervisor after its checkpoint, evaluation and audit.
        subprocess.run(['systemctl', '--user', 'stop', old], check=True)
        (pipeline / 'handover.json').write_text(json.dumps({'parent': str(parent),
            'parent_sha256': sha(parent), 'audit_sha256': sha(smoke), 'step': 3000,
            'scope': 'Objective change only; same original corpus, source, codec and optimizer state',
            'quality_ready': False}, indent=2) + '\n')
    finally:
        if active(old):
            subprocess.run(['systemctl', '--user', 'kill', '--kill-who=main', '--signal=CONT', old], check=False)
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
                launch('scripts.ncm.train_canonical_onpolicy', ['--init', str(checkpoint),
                    '--output', str(cold), '--evaluate-only'])
            verify_replay(read(first_path), read(cold / 'coding.json'), checksum)
            acceptance = {'checkpoint': str(checkpoint), 'checkpoint_sha256': checksum,
                'passed': 12, 'evaluated': 12, 'cold_replay_identical': True,
                'first_report_sha256': sha(first_path), 'cold_report_sha256': sha(cold / 'coding.json'),
                'scope': 'Twelve elementary functions; broader professional quality not established'}
            (pipeline / 'acceptance.json').write_text(json.dumps(acceptance, indent=2) + '\n')
            status(phase='twelve_task_gate_and_cold_replay_passed', **acceptance)
            return
        previous = read(folder / 'session.json').get('step', 3000)
        if previous >= 10000:
            status(phase='schedule_exhausted_without_acceptance', step=previous)
            return
        extra = common + ['--init', str(parent), '--output', str(folder),
            '--optimizer-audit', str(smoke), '--session-steps', '1000', '--eval-every', '250']
        if (folder / 'latest.pt').exists():
            extra.append('--resume')
        status(phase='student_conditioned_canonical_training', start_step=previous)
        launch('scripts.ncm.train_canonical_onpolicy', extra)
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
    args = parser.parse_args()
    if args.cleanup:
        subprocess.run(['systemctl', '--user', 'kill', '--kill-who=main', '--signal=CONT', 'ncm-frozen-clean-pipeline-v32.service'], capture_output=True)
        subprocess.run(['systemctl', '--user', 'stop', 'ncm-canonical-student-worker-v36.service'], capture_output=True)
    else:
        if any(getattr(args, name) is None for name in ('root', 'python', 'source')):
            parser.error('root, python and source are required')
        run(args)
