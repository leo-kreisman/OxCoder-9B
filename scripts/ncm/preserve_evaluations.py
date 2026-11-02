"""Preserve evaluated checkpoints without modifying training or declaring quality.

The trainer atomically replaces latest.pt. A copy is retained only when its
SHA256 matches the completed evaluation, preventing a later model from being
mislabelled with an earlier score. Only development evidence is selected here.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verified_copy(source, destination, expected):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix+'.tmp')
    if temporary.exists():
        raise FileExistsError('Another snapshot may be in progress')
    try:
        with Path(source).open('rb') as src, temporary.open('xb') as dst:
            shutil.copyfileobj(src, dst, 4 << 20)
        if digest(temporary) != expected:
            return False
        os.replace(temporary, destination)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def process(directory):
    directory = Path(directory)
    # A service becomes active before its Python trainer creates the output
    # directory. Do not create it here: the trainer owns exclusive creation.
    if not directory.exists():
        return []
    # Multiple monitoring tools can observe the same completed evaluation.
    # Only one may create its immutable copy; another observer can retry later.
    with (directory / '.evaluation-preservation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return []
        return _process(directory)


def _process(directory):
    captured = []
    for report_path in sorted(directory.glob('coding-*.json')):
        report = json.loads(report_path.read_text())
        if report.get('evaluated') != 12 or len(report.get('tasks', [])) != 12:
            continue
        step = int(report_path.stem.split('-')[-1])
        metadata = directory/f'evaluated-{step}.json'
        if metadata.exists():
            continue
        checkpoint = directory/f'evaluated-{step}.pt'
        if shutil.disk_usage(directory).free < 4 << 30:
            raise RuntimeError('Keep at least 4 GiB free while preserving evaluations')
        success = verified_copy(directory/'latest.pt', checkpoint, report['checkpoint_sha256'])
        record = {'update': step, 'passed': report['passed'], 'evaluated': report['evaluated'],
            'checkpoint_sha256': report['checkpoint_sha256'], 'snapshot_preserved': success,
            'quality_ready': False, 'scope': 'reused development tasks; not a release decision'}
        metadata.write_text(json.dumps(record, indent=2)+'\n')
        captured.append(record)
    return captured


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', required=True)
    p.add_argument('--unit', help='Existing user systemd training unit; read-only status query')
    p.add_argument('--seconds', type=int, default=14400)
    a = p.parse_args()
    if a.seconds < 1:
        p.error('Duration must be positive')
    deadline = time.monotonic()+a.seconds
    while True:
        for record in process(a.directory):
            print(json.dumps(record), flush=True)
        if not a.unit or time.monotonic() >= deadline:
            break
        active = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', a.unit], check=False)
        if active.returncode:
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
