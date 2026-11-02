"""Continue the compact student with explicitly migrated FP32 Muon masters."""
import argparse
import torch
from .run_expanded_stage import Pipeline, active, read, stop
from .preserve_evaluations import digest


class MasterPipeline(Pipeline):
    def run(self):
        if self.a.monitor_only:
            if not active('ncm-master-v22.service'):
                raise RuntimeError('Cannot resume monitoring without the existing trainer')
            self.status('resumed_existing_training_monitor')
            self.monitor(['--source', self.a.source, '--codec', self.root / 'results/ncm-contextual-decoder-v9/best.pt'])
            return
        if any(active(name) for name in ('ncm-compact-v20.service', 'ncm-compact-pipeline-v20.service')):
            raise RuntimeError('Stop the previous trainer/coordinator before optimizer migration')
        base = self.root / 'results'
        parent = base / 'ncm-master-parent-v22.pt'
        saved = torch.load(parent, map_location='cpu', weights_only=True)
        start = saved['step']
        if saved['optimizer']['classes'] != ['AdamW', 'AdamW', 'Muon']:
            raise ValueError('Expected the legacy optimizer snapshot')
        del saved
        if self.a.steps <= start + 2:
            raise ValueError('No full training budget after the optimizer smoke')
        common = ['--source', self.a.source, '--codec', base / 'ncm-contextual-decoder-v9/best.pt']
        training = [*common, '--init', parent, '--data', base / 'ncm-expanded-training-v19',
            '--runtime-audit', base / 'ncm-runtime-audit-v19.json',
            '--weight-audit', base / 'ncm-gguf-storage-audit-v20/report.json',
            '--placement-audit', base / 'ncm-gguf-runtime-audit-v19/report.json',
            '--matrix-audit', base / 'ncm-gguf-all-matrices-v20.json',
            '--reproducibility-audit', base / 'ncm-reference-reproducibility-v20.json',
            '--accumulation', '4', '--continue-optimizer', '--fp32-master-muon', '--migrate-muon']
        self.status('fp32_master_optimizer_smoke', start_step=start, parent_sha256=digest(parent))
        unit = 'ncm-master-smoke-v22'
        self.launch(unit, self.module('train_compact', *training, '--output', base / unit,
                                     '--steps', start + 2, '--eval-every', start + 2), seconds=3600)
        self.wait(unit + '.service')
        progress = read(base / unit / 'progress.json')
        if progress['step'] != start + 2 or not progress['training_complete']:
            raise RuntimeError('Master optimizer smoke is incomplete')
        self.status('training_with_fp32_master_updates', start_step=start, planned_steps=self.a.steps)
        unit = 'ncm-master-v22'
        self.launch(unit, self.module('train_compact', *training, '--output', base / unit,
                                     '--steps', self.a.steps, '--eval-every', '250', '--save-every', '50'),
                    seconds=86400)
        self.monitor(common)

    def monitor(self, common):
        unit = 'ncm-master-v22'
        candidate = self.wait(unit + '.service', 'results/' + unit, stop_on_acceptance=True)
        if candidate is None:
            self.status('stage_complete_requires_next_experiment', acceptance_passed=False)
            return
        for name in ('ncm-extra-distillation-pipeline-v20', 'ncm-distillation-extra-v20',
                     'ncm-distillation-creator-extra-v20', 'ncm-offline-teacher-extra-v20',
                     'ncm-alignment-features-v21', 'ncm-alignment-readout-v21'):
            stop(name + '.service')
        self.status('replaying_saved_candidate_with_original_backend', checkpoint_sha256=digest(candidate))
        replay = self.out / 'coding-reload.json'
        unit = 'ncm-master-acceptance-reload-v22'
        self.launch(unit, self.module('coding_eval', *common, '--checkpoint', candidate, '--output', replay))
        self.wait(unit + '.service')
        result = read(replay)
        passed = result['passed'] == result['evaluated'] == 12 and result['checkpoint_sha256'] == digest(candidate)
        self.status('acceptance_replay_complete', acceptance_passed=passed,
                    checkpoint=str(candidate), checkpoint_sha256=digest(candidate),
                    scope='12/12 elementary development tasks; not broader professional qualification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'python', 'source'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--output', default='results/ncm-master-pipeline-v22')
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--monitor-only', action='store_true', help='Resume observation of an already running trainer; do not launch training')
    args = parser.parse_args()
    pipeline = MasterPipeline(args, resume=args.monitor_only)
    try:
        pipeline.run()
    except BaseException as error:
        pipeline.status('stage_error', error=repr(error), acceptance_passed=False)
        raise
