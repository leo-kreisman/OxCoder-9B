"""Compare two body learning rates from the same saved state, then continue.

The executable acceptance tasks remain unchanged. Model selection additionally
requires improvement on a frozen, disjoint validation selection. Failed or
incomplete experiments are retained, never reported as accepted models.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from .preserve_evaluations import digest, process
from .run_expanded_stage import active, read, stop
from .run_parallel_stage import ParallelPipeline


def compare_training(baseline_log, candidate_log):
    def sequence(path):
        rows = {}
        for line in Path(path).read_text().splitlines():
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            if isinstance(row, dict) and 351 <= row.get('step', 0) <= 500 and 'examples' in row:
                if row['step'] in rows:
                    raise ValueError('Duplicate training update in comparison log')
                rows[row['step']] = {'examples': row['examples'], 'allocation': row['allocation']}
        if set(rows) != set(range(351, 501)):
            raise ValueError('Incomplete training comparison sequence')
        return rows
    first, second = sequence(baseline_log), sequence(candidate_log)
    if first != second:
        raise ValueError('Learning-rate arms used different examples or allocation')
    return {'updates': 150, 'global_examples': 600,
            'sequence_sha256': hashlib.sha256(json.dumps(first, sort_keys=True).encode()).hexdigest(),
            'same_examples_and_allocation': True}


def compare(baseline, candidate, baseline_coding, candidate_coding):
    for report in (baseline, candidate):
        if not report.get('complete') or len(report.get('records', [])) != 64:
            raise ValueError('Both complete 64-conversation validation probes are required')
    if baseline['selection_sha256'] != candidate['selection_sha256']:
        raise ValueError('Validation selections differ')
    if baseline['implementation_sha256'] != candidate['implementation_sha256']:
        raise ValueError('Validation implementations differ')
    for key in ['source_sha256', 'codec_sha256', 'prepared_sha256', 'head_lr', 'rank',
                'head_kind', 'conditioning_corruption', 'effective_batch_examples', 'seed']:
        if key not in baseline['identity'] or baseline['identity'][key] != candidate['identity'].get(key):
            raise ValueError('Checkpoint comparison identity differs: ' + key)
    if [r['id'] for r in baseline['records']] != [r['id'] for r in candidate['records']]:
        raise ValueError('Paired validation row identities differ')
    for report, coding in [(baseline, baseline_coding), (candidate, candidate_coding)]:
        if (coding.get('evaluated') != 12 or len(coding.get('tasks', [])) != 12
                or coding['checkpoint_sha256'] != report['checkpoint_sha256']):
            raise ValueError('Coding evaluation does not identify the probed checkpoint')
    deltas = [sum(b['teacher_forced_ce']) / len(b['teacher_forced_ce']) -
              sum(a['teacher_forced_ce']) / len(a['teacher_forced_ce'])
              for a, b in zip(baseline['records'], candidate['records'])]
    if not all(math.isfinite(x) for x in deltas):
        raise ValueError('Nonfinite validation loss')
    mean = sum(deltas) / len(deltas)
    # Descriptive paired normal interval; it does not prove generalization.
    se = math.sqrt(sum((x - mean)**2 for x in deltas) / (len(deltas) - 1) / len(deltas))
    a, b = baseline['summary'], candidate['summary']
    checks = {
        'coding_not_worse': candidate_coding['passed'] >= baseline_coding['passed'],
        'exact_windows_not_worse': b['free_exact_windows'] >= a['free_exact_windows'],
        'valid_windows_not_worse': b['free_valid_windows'] >= a['free_valid_windows'],
        'byte_accuracy_not_worse': b['mean_free_nonpad_byte_accuracy'] >= a['mean_free_nonpad_byte_accuracy'],
        'ce_improves_at_least_one_percent': b['mean_level_ce'] < .99 * a['mean_level_ce'],
        'paired_ce_upper_bound_negative': mean + 1.96 * se < 0,
    }
    return {'select_candidate': all(checks.values()), 'checks': checks,
        'paired_ce_delta': mean, 'paired_ce_standard_error': se,
        'paired_ce_approximate_95_percent_interval': [mean - 1.96 * se, mean + 1.96 * se],
        'baseline': a, 'candidate': b, 'quality_ready': False,
        'scope': 'Conservative continuation choice on development data, not model acceptance'}


class LearningRatePipeline(ParallelPipeline):
    def common(self, unit, init):
        return self.distributed('train_parallel', '--source', self.a.source,
            '--codec', self.root / 'results/ncm-contextual-decoder-v9/best.pt',
            '--init', self.root / init, '--data', self.root / 'results/ncm-curated-training-v23',
            '--data-expansion', '--runtime-audit', self.root / 'results/ncm-runtime-audit-v19.json',
            '--parallel-audit', self.root / 'results/ncm-parallel-audit-v23/report.json',
            '--cpu-head-optimizer', '--cpu-optimizer-audit',
            self.root / 'results/ncm-cpu-adamw-audit-final-v24/report.json',
            '--output', self.root / 'results' / unit, '--steps', '5000',
            '--eval-every', '250', '--save-every', '50')

    def boundary(self, unit, previous_monitor=None):
        folder = self.root / 'results' / unit
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            process(folder)
            marker = folder / 'evaluated-500.json'
            if marker.exists():
                score = read(marker)
                if not score['snapshot_preserved']:
                    raise ValueError('Comparison checkpoint was not preserved')
                if previous_monitor and score['passed'] == score['evaluated'] == 12:
                    self.status('existing_acceptance_monitor_owns_passing_baseline')
                    return None
                if previous_monitor:
                    stop(previous_monitor + '.service')
                stop(unit + '.service')
                if digest(folder / 'latest.pt') != score['checkpoint_sha256']:
                    raise ValueError('Trainer advanced beyond the comparison resume boundary')
                return folder / 'evaluated-500.pt'
            if not active(unit + '.service'):
                raise RuntimeError('Trainer exited before the completed comparison boundary')
            time.sleep(2)
        raise TimeoutError('Comparison boundary deadline exceeded')

    def probe(self, label, checkpoint):
        unit = 'ncm-lr-probe-' + label + '-v24'
        destination = self.out / ('probe-' + label)
        self.launch(unit, self.module('validation_probe', 'run', '--source', self.a.source,
            '--codec', self.root / 'results/ncm-contextual-decoder-v9/best.pt',
            '--checkpoint', checkpoint, '--selection', self.root / 'results/ncm-validation-selection-v24',
            '--output', destination, '--device', 'cuda:1'), memory='12G', seconds=3600, cpu=200)
        self.wait(unit + '.service')
        return read(destination / 'report.json')

    def run(self):
        baseline_unit, candidate_unit = 'ncm-parallel-v24', 'ncm-parallel-lr-v24'
        candidate_init = 'results/ncm-lr-parent-v24.pt'
        baseline_init = 'results/ncm-parallel-pipeline-v23/initialization.pt'
        branch = read(self.root / 'results/ncm-lr-parent-v24.json')
        if (branch['source_checkpoint_sha256'] != digest(self.root / baseline_init)
                or branch['checkpoint_sha256'] != digest(self.root / candidate_init)
                or branch['source_step'] != 350 or branch['body_lr'] != 1e-4
                or branch['previous_body_lr'] != 3e-5):
            raise ValueError('Learning-rate branch parent or schedule differs')
        selection = read(self.root / 'results/ncm-validation-selection-v24/manifest.json')
        if selection['count'] != 64:
            raise ValueError('Expected frozen 64-conversation validation selection')
        self.status('waiting_for_baseline_500', branch=branch, validation=selection['selection_sha256'])
        baseline = self.boundary(baseline_unit, 'ncm-parallel-pipeline-v24')
        if baseline is None:
            return
        self.status('training_candidate_from_identical_parent', baseline_checkpoint_sha256=digest(baseline))
        self.launch(candidate_unit, self.common(candidate_unit, candidate_init), memory='28G',
                    seconds=86400, cpu=600)
        candidate = self.boundary(candidate_unit)
        sequence = compare_training(self.root / 'results' / (baseline_unit + '.log'),
                                    self.root / 'results' / (candidate_unit + '.log'))
        candidate_coding = read(candidate.parent / 'coding-500.json')
        if candidate_coding['passed'] == candidate_coding['evaluated'] == 12:
            self.replay(candidate)
            return
        self.status('paired_validation', baseline_checkpoint_sha256=digest(baseline),
                    candidate_checkpoint_sha256=digest(candidate))
        baseline_probe, candidate_probe = self.probe('baseline', baseline), self.probe('candidate', candidate)
        decision = compare(baseline_probe, candidate_probe,
            read(baseline.parent / 'coding-500.json'), candidate_coding)
        decision['training_sequence'] = sequence
        from .train import write_json
        write_json(self.out / 'comparison.json', decision)
        chosen, init = ((candidate_unit, candidate_init) if decision['select_candidate']
                        else (baseline_unit, baseline_init))
        self.status('resuming_selected_continuation', selected=chosen, decision=decision)
        self.launch(chosen, [*self.common(chosen, init), '--resume'], memory='28G', seconds=86400, cpu=600)
        self.a.training_unit = chosen
        self.monitor_existing()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'python', 'source']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--output', default='results/ncm-lr-comparison-v24')
    args = parser.parse_args()
    pipeline = LearningRatePipeline(args)
    try:
        pipeline.run()
    except BaseException as error:
        pipeline.status('comparison_error', error=repr(error), acceptance_passed=False)
        raise
