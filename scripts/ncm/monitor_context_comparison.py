"""Serial executable evaluation of immutable native NCM experiment snapshots.

Runs full source-model inference on one explicitly selected device. It never
changes the evaluation tasks, accepts training loss as success, or edits/stops
unrelated services. Existing reports are reused only after checkpoint identity
and completeness checks. Acceptance requires a fresh-process identical replay.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .backbone import digest
from .run_context_stage import active, read


def valid_report(report, checksum):
    return (report.get('complete') is True and report.get('evaluated') == 12
            and report.get('checkpoint_sha256') == checksum
            and len(report.get('tasks', [])) == 12
            and report.get('passed') == sum(x.get('passed') is True for x in report['tasks'])
            and report.get('passed_all') == (report.get('passed') == 12)
            and all(x.get('generation', {}).get('autoregressive_fallback') is False for x in report['tasks']))


def run(a):
    root, output = Path(a.root).resolve(), Path(a.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    def status(**values):
        data = {'time': time.time(), 'quality_ready': False, **values}
        tmp = output / 'progress.tmp'; tmp.write_text(json.dumps(data, indent=2) + '\n')
        tmp.replace(output / 'progress.json'); print(json.dumps(data), flush=True)
    def evaluate(checkpoint, target):
        name = a.eval_unit
        if active(name + '.service'): raise RuntimeError('Evaluation service already active')
        subprocess.run(['systemctl', '--user', 'reset-failed', name + '.service'], capture_output=True)
        subprocess.run(['systemd-run', '--user', '--unit=' + name, '--collect',
            '--property=WorkingDirectory=' + str(root), '--property=MemoryMax=16G',
            '--property=MemorySwapMax=0', '--property=TasksMax=128', '--property=RuntimeMaxSec=3600',
            '--property=StandardOutput=append:' + str(output / 'evaluation.log'),
            '--property=StandardError=append:' + str(output / 'evaluation.log'),
            '--setenv=PYTHONPATH=results/ncm-kernels:results/ncm-deps:results/ncm-data-deps',
            '--setenv=OMP_NUM_THREADS=2', '--setenv=TOKENIZERS_PARALLELISM=false',
            '--setenv=HF_HUB_OFFLINE=1', '--setenv=PYTORCH_ALLOC_CONF=expandable_segments:True',
            a.python, '-m', 'scripts.ncm.context_runtime', '--source', a.source,
            '--codec', a.codec, '--checkpoint', str(checkpoint), '--output', str(target),
            '--device', a.device], check=True)
        while active(name + '.service'): time.sleep(10)
    seen = set()
    while True:
        for folder_name in a.folders:
            folder = Path(folder_name).resolve()
            for checkpoint in sorted(folder.glob('evaluated-*.pt'), key=lambda x: int(x.stem.split('-')[-1])):
                key = str(checkpoint)
                if key in seen: continue
                step = int(checkpoint.stem.split('-')[-1]); checksum = digest(checkpoint)
                destination = output / folder.name
                destination.mkdir(exist_ok=True)
                target = destination / f'coding-{step}.json'
                existing = read(target)
                # Reuse the earlier v25 report without changing its provenance.
                if not target.exists() and folder.name == 'ncm-context-training-v25':
                    old_path = root / 'results/ncm-context-pipeline-v25' / target.name
                    old = read(old_path)
                    if valid_report(old, checksum):
                        target.write_text(json.dumps({**old, 'reused_report': str(old_path),
                            'reused_report_sha256': digest(old_path)}, indent=2) + '\n')
                        existing = read(target)
                if target.exists():
                    if not valid_report(existing, checksum): raise RuntimeError('Invalid existing report: ' + str(target))
                else:
                    status(phase='evaluating', checkpoint=key, device=a.device)
                    evaluate(checkpoint, target)
                report = read(target)
                if not valid_report(report, checksum): raise RuntimeError('Evaluation failed validation: ' + str(target))
                seen.add(key)
                status(phase='evaluated', checkpoint=key, passed=report['passed'], evaluated=12)
                if report['passed_all']:
                    replay = destination / f'cold-replay-{step}.json'
                    if not replay.exists(): evaluate(checkpoint, replay)
                    second = read(replay)
                    if (not valid_report(second, checksum) or
                            [x['generation']['text'] for x in report['tasks']] !=
                            [x['generation']['text'] for x in second['tasks']]):
                        raise RuntimeError('Passing checkpoint failed fresh-process replay')
                    (output / 'acceptance.json').write_text(json.dumps({'checkpoint': key,
                        'checkpoint_sha256': checksum, 'passed': 12, 'evaluated': 12,
                        'cold_replay_identical': True, 'scope': 'Twelve elementary functions; not broad professional qualification'}, indent=2) + '\n')
                    for training_folder in a.folders: (Path(training_folder) / 'STOP_AFTER_CHECKPOINT').touch()
                    status(phase='twelve_task_gate_and_cold_replay_passed', checkpoint=key)
                    return
        if not any(active(unit) for unit in a.training_units):
            status(phase='training_stopped_without_acceptance', evaluated_snapshots=len(seen))
            return
        time.sleep(20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'output', 'python', 'source', 'codec'): parser.add_argument('--' + key, required=True)
    parser.add_argument('--folders', nargs='+', required=True)
    parser.add_argument('--training-units', nargs='+', required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--eval-unit', default='ncm-context-evaluation-v26')
    run(parser.parse_args())
