"""Run qualified compact-weight training and cold replay of a passing snapshot.

This coordinator does not start any teacher, alter task answers, or publish a
checkpoint. Resource audits must finish before starting it. Its service names
are private to this stage; an existing baseline coordinator must be held/stopped
by the operator so two full-model jobs cannot be started concurrently.
"""
import argparse

from .compact_training_policy import qualified, placement_policy, storage_policy
from .run_expanded_stage import Pipeline, active, read, stop
from .preserve_evaluations import digest


class CompactPipeline(Pipeline):
    def run(self):
        for unit in ('ncm-runtime-audit-v19.service', 'ncm-gguf-runtime-audit-v19.service',
                     'ncm-gguf-storage-audit-v20.service', 'ncm-packed-audit-v19.service', 'ncm-accumulated-v19.service'):
            if active(unit):
                raise RuntimeError('Conflicting experiment is active: ' + unit)
        base = self.root / 'results'
        data = base / 'ncm-expanded-training-v19'
        initialization = base / 'ncm-pipeline-v19/initialization.pt'
        codec = base / 'ncm-contextual-decoder-v9/best.pt'
        weights = base / 'ncm-gguf-storage-audit-v20/report.json'
        placement = base / 'ncm-gguf-runtime-audit-v19/report.json'
        matrices = base / 'ncm-gguf-all-matrices-v20.json'
        repeatability = base / 'ncm-reference-reproducibility-v20.json'
        runtime = base / 'ncm-runtime-audit-v19.json'
        packing = base / 'ncm-packed-audit-v19.json'
        report = read(weights)
        parent = report['model_identity']
        data_sha = read(data / 'manifest.json')['prepared_sha256']
        storage_policy(matrices, weights, repeatability, parent, data_sha)
        placement_policy(weights, placement, parent, data_sha, matrices, repeatability)
        use_packing = False
        if packing.exists():
            try:
                qualified(packing, parent, data_sha, packing=True)
                rows = [r for r in read(packing)['records'] if r.get('packed')]
                # Require an observed advantage, in addition to numerical gates.
                use_packing = all(r['speed_ratio'] > 1 for r in rows)
            except ValueError:
                use_packing = False
        common = ['--source', self.a.source, '--codec', codec]
        training = [*common, '--init', initialization, '--data', data,
                    '--runtime-audit', runtime, '--weight-audit', weights, '--placement-audit', placement,
                    '--matrix-audit', matrices, '--reproducibility-audit', repeatability,
                    '--data-expansion', '--accumulation', '4']
        if use_packing:
            training.extend(['--packed-audit', packing])
        self.status('compact_optimizer_smoke', packed_examples=use_packing,
                    initialization_sha256=digest(initialization))
        smoke = 'ncm-compact-smoke-v20'
        self.launch(smoke, self.module('train_compact', *training, '--output', base / smoke,
                                     '--steps', '2', '--eval-every', '2'), seconds=3600)
        self.wait(smoke + '.service')
        progress = read(base / smoke / 'progress.json')
        if progress['step'] != 2 or not progress['training_complete']:
            raise RuntimeError('Compact optimizer smoke did not finish')
        unit = 'ncm-compact-v20'
        self.status('training_compact_student', steps=self.a.steps, accumulation=4,
                    packed_examples=use_packing, smoke=progress)
        self.launch(unit, self.module('train_compact', *training, '--output', base / unit,
                                     '--steps', self.a.steps, '--eval-every', '250', '--save-every', '50'),
                    seconds=86400)
        candidate = self.wait(unit + '.service', 'results/' + unit, stop_on_acceptance=True)
        if candidate is None:
            self.status('stage_complete_requires_next_experiment', acceptance_passed=False)
            return
        self.status('replaying_saved_candidate_with_original_backend', checkpoint_sha256=digest(candidate))
        # These are this experiment's optional offline collection units; the
        # original reference replay needs both devices free of our own jobs.
        stop('ncm-distillation-extra-v20.service')
        stop('ncm-offline-teacher-extra-v20.service')
        replay = self.out / 'coding-reload.json'
        unit = 'ncm-compact-acceptance-reload-v20'
        self.launch(unit, self.module('coding_eval', *common, '--checkpoint', candidate, '--output', replay))
        self.wait(unit + '.service')
        result = read(replay)
        passed = result['passed'] == result['evaluated'] == 12 and result['checkpoint_sha256'] == digest(candidate)
        self.status('acceptance_replay_complete', acceptance_passed=passed,
                    checkpoint=str(candidate), checkpoint_sha256=digest(candidate),
                    scope='12/12 reused elementary development tasks; not broader professional qualification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'python', 'source'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--output', default='results/ncm-compact-pipeline-v20')
    parser.add_argument('--steps', type=int, default=5000)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('Positive step count required')
    pipeline = CompactPipeline(args)
    try:
        pipeline.run()
    except BaseException as error:
        pipeline.status('stage_error', error=repr(error), acceptance_passed=False)
        raise
