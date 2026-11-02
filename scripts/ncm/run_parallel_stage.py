"""Finish the current teacher collection, curate it, then qualify a continuation.

The existing two creator datasets and their admitted distillation remain in the
base. Additional creator-teacher collection is deferred to free both GPUs for
student training. This coordinator owns only the named experiment services.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import torch

from .preserve_evaluations import digest, process, verified_copy
from .run_expanded_stage import Pipeline, active, read, stop


class ParallelPipeline(Pipeline):
    def distributed(self, name, *arguments):
        return [self.a.python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                '-m', 'scripts.ncm.' + name, *map(str, arguments)]

    def replay(self, candidate):
        self.status('replaying_saved_candidate', checkpoint_sha256=digest(candidate))
        replay = self.out / 'coding-reload.json'
        unit = 'ncm-parallel-acceptance-reload-v23'
        self.launch(unit, self.module('coding_eval', '--source', self.a.source,
            '--codec', self.root / 'results/ncm-contextual-decoder-v9/best.pt',
            '--checkpoint', candidate, '--output', replay))
        self.wait(unit + '.service')
        result = read(replay)
        passed = result['passed'] == result['evaluated'] == 12 and result['checkpoint_sha256'] == digest(candidate)
        self.status('acceptance_replay_complete', acceptance_passed=passed,
                    checkpoint=str(candidate), checkpoint_sha256=digest(candidate),
                    scope='12 elementary development tasks, not broader professional qualification')

    def run(self):
        if active('ncm-extra-distillation-pipeline-v20.service'):
            raise RuntimeError('Retire the old collection coordinator before selecting the successor stage')
        self.status('finishing_nvidia_teacher_collection',
            creator_sources='Both creator datasets and their existing distillation retained in the base',
            deferred='Additional 256-per-source creator teacher requests; prioritize two-GPU student continuation')
        self.wait('ncm-distillation-extra-v20.service')
        archive = self.root / 'results/ncm-distillation-opencode-v20'
        if not read(archive / 'manifest.json')['admission_complete']:
            raise RuntimeError('Teacher collection is incomplete')
        stop('ncm-offline-teacher-extra-v20.service')
        verification = self.root / 'results/ncm-verified-distillation-full-v23'
        self.status('checking_complete_teacher_archive')
        unit = 'ncm-verify-distillation-full-v23'
        self.launch(unit, self.module('verify_distillation', '--admitted', 'results/ncm-opencode-v19',
            '--cache', archive, '--output', verification), memory='3G', seconds=3600, gpu='', kernels=False, cpu=150)
        self.wait(unit + '.service')
        curated = self.root / 'results/ncm-distillation-opencode-curated-v23'
        unit = 'ncm-curate-distillation-v23'
        self.launch(unit, self.module('curate_distillation', '--source', archive,
            '--verification', verification, '--output', curated), memory='2G', gpu='', kernels=False, cpu=150)
        self.wait(unit + '.service')
        data = self.root / 'results/ncm-curated-training-v23'
        unit = 'ncm-prepare-curated-v23'
        self.launch(unit, self.module('prepare_curated_expansion', '--base', 'results/ncm-expanded-training-v19',
            '--addition', curated, '--output', data), memory='6G', seconds=3600, gpu='', kernels=False, cpu=150)
        self.wait(unit + '.service')
        self.status('waiting_for_saved_student_boundary', curated_counts=read(curated / 'manifest.json')['counts'])
        old = self.root / 'results/ncm-master-v22'
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            if not active('ncm-master-v22.service'):
                self.status('original_trainer_stopped_before_handoff', original_monitor=read(
                    self.root / 'results/ncm-master-pipeline-v22/progress.json'))
                return
            progress = read(old / 'progress.json')
            if progress['step'] % 50 == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError('No saved continuation boundary observed')
        # Avoid racing the original acceptance monitor during the handoff.
        stop('ncm-master-pipeline-v22.service')
        stop('ncm-master-v22.service')
        process(old)
        accepted = [p.with_suffix('.pt') for p in old.glob('evaluated-*.json')
                    if (lambda r: r['snapshot_preserved'] and r['passed'] == r['evaluated'] == 12)(read(p))]
        if accepted:
            self.replay(accepted[-1])
            return
        parent = self.out / 'initialization.pt'
        if not verified_copy(old / 'latest.pt', parent, digest(old / 'latest.pt')):
            raise RuntimeError('Saved continuation changed while copying')
        checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
        start = checkpoint['step']
        del checkpoint
        self.status('qualifying_parallel_optimizer_continuation', start_step=start,
                    checkpoint_sha256=digest(parent), prepared_sha256=read(data / 'manifest.json')['prepared_sha256'])
        common = ['--source', self.a.source, '--codec', self.root / 'results/ncm-contextual-decoder-v9/best.pt',
            '--init', parent, '--data', data, '--data-expansion',
            '--runtime-audit', self.root / 'results/ncm-runtime-audit-v19.json',
            '--parallel-audit', self.root / 'results/ncm-parallel-audit-v23/report.json']
        unit = 'ncm-parallel-smoke-v23'
        self.launch(unit, self.distributed('train_parallel', *common, '--output', self.root / 'results' / unit,
            '--steps', start + 3, '--eval-every', start + 3), memory='28G', seconds=3600, cpu=600)
        self.wait(unit + '.service')
        smoke = self.root / 'results' / unit
        progress = read(smoke / 'progress.json')
        if progress['step'] != start + 3 or not progress['training_complete']:
            raise RuntimeError('Parallel optimizer smoke is incomplete')
        process(smoke)
        score = read(smoke / f'evaluated-{start+3}.json')
        if score['passed'] == score['evaluated'] == 12:
            self.replay(smoke / f'evaluated-{start+3}.pt')
            return
        unit = 'ncm-parallel-v23'
        if self.a.steps <= start + 3:
            raise RuntimeError('No full-stage budget remains after qualification')
        self.status('training_two_replica_student', start_step=start, planned_steps=self.a.steps,
                    smoke_updates='Qualification only; main run starts from the same immutable parent')
        self.launch(unit, self.distributed('train_parallel', *common, '--output', self.root / 'results' / unit,
            '--steps', self.a.steps, '--eval-every', '250', '--save-every', '50'),
            memory='28G', seconds=86400, cpu=600)
        candidate = self.wait(unit + '.service', 'results/' + unit, stop_on_acceptance=True)
        if candidate is not None:
            self.replay(candidate)
        else:
            self.status('stage_complete_requires_next_experiment', acceptance_passed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'python', 'source'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--output', default='results/ncm-parallel-pipeline-v23')
    parser.add_argument('--steps', type=int, default=5000)
    args = parser.parse_args()
    pipeline = ParallelPipeline(args)
    try:
        pipeline.run()
    except BaseException as error:
        pipeline.status('stage_error', error=repr(error), acceptance_passed=False)
        raise
