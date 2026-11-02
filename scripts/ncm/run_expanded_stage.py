"""Run the admitted-data next stage after the existing v18 job completes.

Coordinates only the explicitly named experiment services. It never terminates
unrelated processes, substitutes teacher outputs for student evaluation, or
promotes a model to a release. A 12/12 candidate is replayed from disk before
recording elementary-development acceptance.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time
import urllib.request

from .preserve_evaluations import digest, process, verified_copy


def read(path):
    return json.loads(Path(path).read_text())


def active(unit):
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit], check=False).returncode == 0


def stop(unit):
    if active(unit):
        subprocess.run(['systemctl', '--user', 'stop', unit], check=True)


def qualified_policy(report):
    """Select an audited execution path without weakening numerical thresholds.

    The trainer revalidates the complete report, model identity, and source
    hashes before loading data. This preliminary choice must also reject a
    partial or duplicated context-length inventory.
    """
    if not report.get('complete'):
        raise RuntimeError('Runtime audit incomplete')
    for shared in (True, False):
        records = [r for r in report['records']
                   if bool(r['runtime'].get('shared_prefix_training')) == shared
                   and r['runtime'].get('keep_per_device') == 0
                   and r['runtime'].get('convolution') == 'reference']
        if (len(records) == 4 and
                {r['requested_prompt_tokens'] for r in records} == {64, 69, 512, 2048}
                and all(r.get('passed') for r in records)):
            return shared
    raise RuntimeError('No numerically qualified runtime path')


class Pipeline:
    def __init__(self, args, resume=False):
        self.a = args
        self.root = Path(args.root).resolve()
        self.out = self.root / args.output
        if resume and not (self.out / 'progress.json').is_file():
            raise ValueError('Monitor resumption requires an existing pipeline state')
        self.out.mkdir(parents=True, exist_ok=resume)
        self.units = []

    def status(self, phase, **values):
        report = {'phase': phase, 'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'quality_ready': False, **values}
        temporary = self.out / 'progress.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(self.out / 'progress.json')
        print(json.dumps(report), flush=True)

    def launch(self, unit, command, memory='28G', seconds=1800, gpu='0,1', kernels=True, cpu=400):
        if active(unit):
            raise RuntimeError('Refusing to replace active unit: ' + unit)
        # Stopping torchrun at a preserved checkpoint can leave a failed
        # transient unit loaded. Release that inactive unit before reusing its
        # name for an optimizer-preserving resume. The model files are untouched.
        # A never-created unit has nothing to reset; systemd-run remains the
        # authoritative check that creation succeeded.
        subprocess.run(['systemctl', '--user', 'reset-failed', unit], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log = self.root / 'results' / (unit + '.log')
        pythonpath = 'results/ncm-deps:results/ncm-data-deps'
        if kernels:
            pythonpath = 'results/ncm-kernels:' + pythonpath
        invocation = ['systemd-run', '--user', '--unit=' + unit,
            '--property=MemoryMax=' + memory, '--property=MemorySwapMax=0',
            '--property=TasksMax=128', '--property=CPUQuota=' + str(cpu) + '%',
            '--property=RuntimeMaxSec=' + str(seconds),
            '--property=WorkingDirectory=' + str(self.root),
            '--property=StandardOutput=append:' + str(log),
            '--property=StandardError=append:' + str(log),
            '--setenv=PYTHONPATH=' + pythonpath, '--setenv=OMP_NUM_THREADS=2',
            '--setenv=TOKENIZERS_PARALLELISM=false', '--setenv=HF_HUB_OFFLINE=1',
            '--setenv=PYTORCH_ALLOC_CONF=expandable_segments:True',
            '--setenv=CUDA_VISIBLE_DEVICES=' + gpu] + list(map(str, command))
        subprocess.run(invocation, check=True, cwd=self.root)
        self.units.append(unit)
        (self.out / (unit + '-launch.json')).write_text(json.dumps(invocation, indent=2) + '\n')

    def wait(self, unit, directory=None, stop_on_acceptance=False):
        while True:
            if directory:
                process(self.root / directory)
                if stop_on_acceptance:
                    for path in sorted((self.root / directory).glob('evaluated-*.json')):
                        report = read(path)
                        if report['snapshot_preserved'] and report['passed'] == report['evaluated'] == 12:
                            checkpoint = path.with_suffix('.pt')
                            if digest(checkpoint) != report['checkpoint_sha256']:
                                raise RuntimeError('Passing snapshot checksum differs')
                            self.status('candidate_saved_for_reload', checkpoint=str(checkpoint), report=report)
                            stop(unit)
                            return checkpoint
            if not active(unit):
                result = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'Result', '--value'],
                                        capture_output=True, text=True, check=True).stdout.strip()
                if result != 'success':
                    raise RuntimeError(unit + ' ended with ' + result)
                return None
            time.sleep(5)

    def module(self, name, *arguments):
        return [self.a.python, '-m', 'scripts.ncm.' + name, *map(str, arguments)]

    def run(self):
        self.status('waiting_for_v18_completion')
        self.wait('ncm-independent-v18.service')
        previous = self.root / 'results/ncm-independent-v18'
        if not read(previous / 'progress.json')['training_complete']:
            raise RuntimeError('v18 did not finish its planned schedule')
        report = read(previous / 'coding-2000.json')
        if report['evaluated'] != 12:
            raise RuntimeError('v18 final evaluation is incomplete')
        initialization = self.out / 'initialization.pt'
        if not verified_copy(previous / 'latest.pt', initialization, report['checkpoint_sha256']):
            raise RuntimeError('Initialization copy did not match evaluation')
        self.status('finishing_offline_distillation_on_gpu', initialization_sha256=digest(initialization))
        stop('ncm-distillation-v19.service')
        stop('ncm-offline-teacher-v19.service')
        free = subprocess.run(['nvidia-smi', '--id=1', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, check=True)
        if int(free.stdout.strip()) < 9 * 1024:
            raise RuntimeError('GPU 1 lacks the offline teacher reserve')
        teacher = 'ncm-offline-teacher-gpu-v19'
        self.launch(teacher, [self.a.server, '--model', self.a.source, '--alias', 'oxcoder-offline-teacher',
            '--host', '127.0.0.1', '--port', '18119', '--gpu-layers', '99', '--parallel', '1',
            '--ctx-size', '4096', '--threads', '4', '--threads-batch', '4',
            '--batch-size', '256', '--ubatch-size', '128', '--reasoning', 'off'],
            memory='10G', seconds=7200, gpu='1', kernels=False)
        try:
            for _ in range(60):
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18119/health', timeout=2) as response:
                        if json.load(response).get('status') == 'ok':
                            break
                except (OSError, ValueError):
                    pass
                time.sleep(2)
            else:
                raise RuntimeError('Offline teacher did not become ready')
            collector = 'ncm-distillation-gpu-v19'
            self.launch(collector, self.module('distill_admitted',
                '--admitted', 'results/ncm-creator-admitted-v15b', '--frozen', 'results/ncm-open-corpus-v11c',
                '--output', 'results/ncm-distillation-v19', '--teacher-file', self.a.source,
                '--expected-teacher-sha256', 'a171c091c67c3547c45ec58f1b85a888d73725315fde71bc929ff9f8c255bb6f',
                '--server-binary', self.a.server, '--per-source', '64', '--max-tokens', '768', '--prompt-limit', '1536'),
                memory='5G', seconds=7000, gpu='', kernels=False, cpu=150)
            self.wait(collector + '.service')
            if not read(self.root / 'results/ncm-distillation-v19/manifest.json')['admission_complete']:
                raise RuntimeError('Teacher admission did not finish')
        finally:
            stop(teacher + '.service')
        data = 'results/ncm-expanded-training-v19'
        self.status('preparing_combined_training_inventory')
        unit = 'ncm-prepare-combined-v19'
        self.launch(unit, self.module('prepare_training_expansion',
            '--base', 'results/ncm-contextual-expansion-v17', '--addition', 'results/ncm-opencode-v19',
            '--addition', 'results/ncm-distillation-v19', '--output', data),
            memory='5G', gpu='', kernels=False, cpu=150)
        self.wait(unit + '.service')
        audit = self.root / 'results/ncm-runtime-audit-v19.json'
        codec = self.root / 'results/ncm-contextual-decoder-v9/best.pt'
        common = ['--source', self.a.source, '--codec', codec]
        self.status('qualifying_runtime_on_expanded_inventory')
        unit = 'ncm-runtime-audit-v19'
        self.launch(unit, self.module('audit_training_runtime', *common, '--checkpoint', initialization,
            '--data', data, '--output', audit, '--keep', '0', '--shared',
            '--lengths', '64,69,512,2048', '--repeats', '3'))
        self.wait(unit + '.service')
        shared_ok = qualified_policy(read(audit))
        policy = '--shared-prefix-training' if shared_ok else '--no-shared-prefix-training'
        training = [*common, '--init', initialization, '--data', data, '--runtime-audit', audit,
                    '--data-expansion', '--accumulation', '4', policy]
        self.status('smoke_testing_accumulated_training', shared_prefix=shared_ok)
        unit = 'ncm-accumulated-smoke-v19'
        self.launch(unit, self.module('train_accumulated', *training, '--output', 'results/' + unit,
                                     '--steps', '2', '--eval-every', '2'))
        self.wait(unit + '.service')
        smoke = read(self.root / ('results/' + unit + '/progress.json'))
        if smoke['step'] != 2 or not smoke['training_complete']:
            raise RuntimeError('Accumulation smoke did not complete')
        unit = 'ncm-accumulated-v19'
        self.status('training_expanded_student', steps=5000, accumulation=4, shared_prefix=shared_ok)
        self.launch(unit, self.module('train_accumulated', *training, '--output', 'results/' + unit,
                                     '--steps', '5000', '--eval-every', '250', '--save-every', '10'),
                    seconds=43200)
        candidate = self.wait(unit + '.service', 'results/' + unit, stop_on_acceptance=True)
        if candidate is None:
            self.status('stage_complete_requires_next_experiment', acceptance_passed=False)
            return
        replay = self.out / 'coding-reload.json'
        self.status('replaying_saved_candidate', checkpoint_sha256=digest(candidate))
        unit = 'ncm-acceptance-reload-v19'
        self.launch(unit, self.module('coding_eval', *common, '--checkpoint', candidate, '--output', replay))
        self.wait(unit + '.service')
        result = read(replay)
        passed = result['passed'] == result['evaluated'] == 12 and result['checkpoint_sha256'] == digest(candidate)
        self.status('acceptance_replay_complete', acceptance_passed=passed,
                    checkpoint=str(candidate), checkpoint_sha256=digest(candidate),
                    scope='12/12 reused elementary development tasks; not broader professional qualification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'python', 'server', 'source'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--output', default='results/ncm-pipeline-v19')
    pipeline = Pipeline(parser.parse_args())
    try:
        pipeline.run()
    except BaseException as error:
        pipeline.status('requires_intervention', error=repr(error), acceptance_passed=False)
        raise
