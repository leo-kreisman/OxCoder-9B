"""Schedule qualified batched NCM continuation around the native-decoder evaluator.

Only named experiment units are paused. A child exit always resumes the older
coordinator. Native training continues; the full-model evaluators never overlap
the new full-model trainer on the same GPU. Acceptance requires a second process.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

EXPECTED_TASKS = ('factorial', 'unique', 'chunks', 'balanced', 'gcd', 'merge_intervals',
                  'flatten', 'rotate', 'word_counts', 'transpose', 'binary_search', 'is_prime')
EXPECTED_CASES = (4, 3, 3, 4, 4, 3, 3, 4, 3, 3, 4, 5)


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def active(unit):
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode == 0


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_replay(first, second, checksum):
    for report in (first, second):
        tasks = report.get('tasks', [])
        if (report.get('checkpoint_sha256') != checksum or report.get('passed') != 12
                or report.get('evaluated') != 12 or len(tasks) != 12
                or tuple(r['task'] for r in tasks) != EXPECTED_TASKS
                or tuple(len(r['checks']) for r in tasks) != EXPECTED_CASES
                or any(not r['passed'] or r['generation'].get('autoregressive_fallback') is not False
                       or r['generation']['stop_reason'] != 'predicted_end'
                       or not r['checks'] or not all(c['passed'] for c in r['checks']) for r in tasks)):
            raise ValueError('Incomplete or failed executable replay')
    if [(r['task'], r['generation']['text']) for r in first['tasks']] != [(r['task'], r['generation']['text']) for r in second['tasks']]:
        raise ValueError('Cold replay text differs')


def run(a):
    root = Path(a.root).resolve()
    pipeline = root / 'results/ncm-frozen-multi-pipeline-v30'
    folder = root / 'results/ncm-frozen-multi-training-v30'
    pipeline.mkdir(exist_ok=True)
    lock = (pipeline / 'owner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    coordinator = 'ncm-complete-pipeline-v28.service'
    evaluator = 'ncm-context-evaluation-v28.service'
    child_unit = 'ncm-frozen-multi-training-v30.service'

    def status(**values):
        data = {'time': time.time(), 'quality_ready': False, **values}
        tmp = pipeline / 'progress.tmp'
        tmp.write_text(json.dumps(data, indent=2) + '\n')
        tmp.replace(pipeline / 'progress.json')
        print(json.dumps(data), flush=True)

    def pending_native():
        for name in ('ncm-context-control-v28', 'ncm-context-decoded-v28'):
            for checkpoint in (root / 'results' / name).glob('evaluated-*.pt'):
                number = checkpoint.stem.split('-')[-1]
                report = read(root / 'results/ncm-complete-pipeline-v28/comparison' / name / ('coding-' + number + '.json'))
                if not report.get('complete'):
                    return True
        return False

    def launch(init, output, extra):
        if active(child_unit):
            raise RuntimeError('Another frozen-tower worker is running')
        subprocess.run(['systemctl', '--user', 'kill', '--signal=STOP', coordinator], capture_output=True)
        try:
            while active(evaluator):
                status(phase='waiting_for_current_native_evaluation')
                time.sleep(10)
            command = ['systemd-run', '--user', '--unit=' + child_unit.removesuffix('.service'), '--collect',
                '--property=WorkingDirectory=' + str(root), '--property=MemoryMax=18G',
                '--property=MemorySwapMax=0', '--property=TasksMax=128', '--property=RuntimeMaxSec=14400',
                '--property=ExecStopPost=-/usr/bin/systemctl --user kill --signal=CONT ' + coordinator,
                '--property=StandardOutput=append:' + str(pipeline / 'worker.log'),
                '--property=StandardError=append:' + str(pipeline / 'worker.log'),
                '--setenv=PYTHONPATH=results/ncm-kernels:results/ncm-deps:results/ncm-data-deps',
                '--setenv=OMP_NUM_THREADS=2', '--setenv=TOKENIZERS_PARALLELISM=false',
                '--setenv=HF_HUB_OFFLINE=1', '--setenv=PYTORCH_ALLOC_CONF=expandable_segments:True',
                a.python, '-m', 'scripts.ncm.train_frozen_multi', '--source', a.source,
                '--codec', 'results/ncm-contextual-decoder-v9/best.pt', '--init', str(init),
                '--execution-audit', 'results/ncm-frozen-multi-audit-v30/report.json',
                '--output', str(output), '--device', 'cuda:1', *extra]
            subprocess.run(command, check=True)
            while active(child_unit):
                time.sleep(10)
        finally:
            subprocess.run(['systemctl', '--user', 'kill', '--signal=CONT', coordinator], capture_output=True)

    smoke = read(root / 'results/ncm-frozen-multi-optimizer-v30/report.json')
    if not smoke.get('complete') or not smoke.get('passed'):
        raise ValueError('Optimizer smoke must complete and pass first')
    while True:
        candidate = read(folder / 'candidate.json')
        if candidate:
            checkpoint = Path(candidate['checkpoint'])
            checksum = sha(checkpoint)
            if candidate['checkpoint_sha256'] != checksum:
                raise ValueError('Candidate checkpoint changed')
            first = read(folder / ('coding-' + checkpoint.stem.split('-')[-1] + '.json'))
            cold = pipeline / ('cold-' + checkpoint.stem)
            if not (cold / 'coding.json').exists():
                status(phase='fresh_process_replay', checkpoint=str(checkpoint))
                launch(checkpoint, cold, ['--evaluate-only'])
            second = read(cold / 'coding.json')
            verify_replay(first, second, checksum)
            acceptance = {'checkpoint': str(checkpoint), 'checkpoint_sha256': checksum,
                'passed': 12, 'evaluated': 12, 'cold_replay_identical': True,
                'first_report_sha256': sha(folder / ('coding-' + checkpoint.stem.split('-')[-1] + '.json')),
                'cold_report_sha256': sha(cold / 'coding.json'),
                'scope': 'Twelve elementary executable functions; broader professional quality not established'}
            (pipeline / 'acceptance.json').write_text(json.dumps(acceptance, indent=2) + '\n')
            for name in ('ncm-context-control-v28', 'ncm-context-decoded-v28'):
                (root / 'results' / name / 'STOP_AFTER_CHECKPOINT').touch()
            status(phase='twelve_task_gate_and_cold_replay_passed', **acceptance)
            return
        previous = read(folder / 'session.json').get('step', 100)
        if previous >= 10000:
            status(phase='schedule_exhausted_without_acceptance', step=previous)
            return
        while active(coordinator) and (active(evaluator) or pending_native()):
            status(phase='allowing_native_evaluations_to_finish', step=previous)
            time.sleep(10)
        status(phase='training_original_backbone', start_step=previous)
        extra = ['--selection', 'results/ncm-complete-selection-v27',
            '--manifest', 'results/ncm-complete-pipeline-v28/features.json',
            '--optimizer-audit', 'results/ncm-frozen-multi-optimizer-v30/report.json',
            '--session-steps', '100', '--eval-every', '250']
        if (folder / 'latest.pt').exists():
            extra.append('--resume')
        launch(root / 'results/ncm-frozen-tower-training-v29/probe-100.pt', folder, extra)
        if (folder / 'candidate.json').exists():
            continue
        current = read(folder / 'session.json').get('step', 0)
        if current <= previous or current != read(folder / 'progress.json').get('step'):
            raise RuntimeError('Training did not yield a new complete checkpoint; inspect worker.log')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cleanup', action='store_true')
    for key in ('root', 'python', 'source'):
        p.add_argument('--' + key)
    a = p.parse_args()
    if a.cleanup:
        subprocess.run(['systemctl', '--user', 'stop', 'ncm-frozen-multi-training-v30.service'], capture_output=True)
        subprocess.run(['systemctl', '--user', 'kill', '--signal=CONT', 'ncm-complete-pipeline-v28.service'], capture_output=True)
    else:
        if any(getattr(a, key) is None for key in ('root', 'python', 'source')):
            p.error('root, python and source required')
        run(a)
