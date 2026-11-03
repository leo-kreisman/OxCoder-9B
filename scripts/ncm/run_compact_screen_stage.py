"""Wait for completed source checks and resource headroom, then screen once."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main(a):
    root = Path(a.root).resolve()
    manifest = root / 'results/ncm-compact-code-v33/manifest.json'
    while True:
        active = subprocess.run(['systemctl', '--user', 'is-active', '--quiet',
                                 'ncm-compact-code-curriculum-v33.service']).returncode == 0
        if not active:
            if not manifest.exists() or not json.loads(manifest.read_text()).get('admission_complete'):
                raise RuntimeError('Source-check worker ended without complete admission')
            break
        time.sleep(30)
    while True:
        available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        if available >= 16 * 1024 ** 3:
            break
        print(json.dumps({'phase': 'waiting_for_screening_memory', 'available_bytes': available}), flush=True)
        time.sleep(30)
    subprocess.run([a.python, '-m', 'scripts.ncm.screen_compact_corpus',
        '--base-selection', 'results/ncm-complete-selection-v27',
        '--base-manifest', 'results/ncm-complete-pipeline-v28/features.json',
        '--addition', 'results/ncm-compact-code-v33',
        '--codec', 'results/ncm-contextual-decoder-v9/best.pt',
        '--output', 'results/ncm-compact-training-v33', '--device', a.device], cwd=root, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--device', default='cuda:1')
    main(parser.parse_args())
