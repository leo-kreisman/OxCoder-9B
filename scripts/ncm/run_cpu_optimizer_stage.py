"""Qualify the repaired training path and monitor the real student checkpoint."""
import argparse

from .preserve_evaluations import digest, process
from .run_expanded_stage import read
from .run_parallel_stage import ParallelPipeline


class CPUOptimizerPipeline(ParallelPipeline):
    def run(self):
        self.status('waiting_for_cpu_optimizer_smoke')
        self.wait('ncm-cpu-head-smoke-v24.service')
        smoke = self.root / 'results/ncm-cpu-head-smoke-v24'
        progress, identity = read(smoke / 'progress.json'), read(smoke / 'identity.json')
        if not progress['training_complete'] or progress['step'] != 360:
            raise ValueError('The recovery smoke did not traverse the failing update')
        for name, expected in identity['implementation_sha256'].items():
            if digest(self.root / 'scripts/ncm' / name) != expected:
                raise ValueError('Implementation changed after the recovery smoke: ' + name)
        data = self.root / 'results/ncm-curated-training-v23'
        cpu_audit = self.root / 'results/ncm-cpu-adamw-audit-final-v24/report.json'
        if (identity['prepared_sha256'] != read(data / 'manifest.json')['prepared_sha256']
                or identity['cpu_head_optimizer']['audit_sha256'] != digest(cpu_audit)):
            raise ValueError('Data or CPU optimizer audit changed')
        process(smoke)
        score = read(smoke / 'evaluated-360.json')
        if not score['snapshot_preserved'] or score['evaluated'] != 12:
            raise ValueError('Recovery smoke evaluation was not preserved')
        if score['passed'] == 12:
            self.replay(smoke / 'evaluated-360.pt')
            return
        self.status('training_with_cpu_head_optimizer', smoke_step=360,
                    smoke_checkpoint_sha256=score['checkpoint_sha256'],
                    planned_steps=self.a.steps,
                    scope='Fresh continuation from the immutable update-350 parent; smoke is qualification only')
        unit = self.a.training_unit
        self.launch(unit, self.distributed('train_parallel',
            '--source', self.a.source,
            '--codec', self.root / 'results/ncm-contextual-decoder-v9/best.pt',
            '--init', self.root / 'results/ncm-parallel-pipeline-v23/initialization.pt',
            '--data', data, '--data-expansion',
            '--runtime-audit', self.root / 'results/ncm-runtime-audit-v19.json',
            '--parallel-audit', self.root / 'results/ncm-parallel-audit-v23/report.json',
            '--cpu-head-optimizer', '--cpu-optimizer-audit', cpu_audit,
            '--output', self.root / 'results' / unit,
            '--steps', self.a.steps, '--eval-every', '250', '--save-every', '50'),
            memory='28G', seconds=86400, cpu=600)
        self.monitor_existing()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'python', 'source']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--output', default='results/ncm-parallel-pipeline-v24')
    parser.add_argument('--training-unit', default='ncm-parallel-v24')
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--monitor-only', action='store_true')
    args = parser.parse_args()
    pipeline = CPUOptimizerPipeline(args, resume=args.monitor_only)
    try:
        if args.monitor_only:
            pipeline.monitor_existing()
        else:
            pipeline.run()
    except BaseException as error:
        pipeline.status('stage_error', error=repr(error), acceptance_passed=False)
        raise
