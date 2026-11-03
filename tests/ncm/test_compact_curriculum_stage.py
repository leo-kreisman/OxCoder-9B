from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts.ncm import run_compact_curriculum_stage as stage


class CompactStageTests(unittest.TestCase):
    def test_early_retirement_needs_saved_state_and_complete_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'results/ncm-frozen-ragged-training-v31'
            folder.mkdir(parents=True)
            checkpoint = folder / 'retired-1100.pt'; checkpoint.write_bytes(b'checkpoint fixture')
            report = folder / 'retired-coding-1100.json'
            report.write_text(json.dumps({'evaluated': 12, 'checkpoint_sha256': stage.sha(checkpoint)}))
            (folder / 'progress.json').write_text(json.dumps({'step': 1100}))
            record = {'step': 1100, 'phase': 'saved_checkpoint_and_independent_evaluation',
                'optimizer_state_preserved': True, 'checkpoint': str(checkpoint),
                'checkpoint_sha256': stage.sha(checkpoint), 'coding_report': str(report),
                'coding_report_sha256': stage.sha(report)}
            (folder / 'retirement.json').write_text(json.dumps(record))
            self.assertEqual(stage.retired_boundary(directory), (1100, checkpoint, report))
            checkpoint.write_bytes(b'changed checkpoint')
            with self.assertRaises(ValueError):
                stage.retired_boundary(directory)
            checkpoint.write_bytes(b'checkpoint fixture')
            record['optimizer_state_preserved'] = False
            (folder / 'retirement.json').write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                stage.retired_boundary(directory)

    def test_signals_target_supervisor_main_only(self):
        with patch.object(stage.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as command:
            stage.signal_running('experiment.service', 'STOP')
        self.assertEqual(command.call_args.args[0], ['systemctl', '--user', 'kill',
                         '--kill-who=main', '--signal=STOP', 'experiment.service'])

    def test_failed_signal_cannot_silently_allow_overlap(self):
        with patch.object(stage.subprocess, 'run', return_value=SimpleNamespace(returncode=1)), \
             patch.object(stage, 'active', return_value=True):
            with self.assertRaises(RuntimeError):
                stage.signal_running('experiment.service', 'STOP')
        with patch.object(stage.subprocess, 'run', return_value=SimpleNamespace(returncode=1)), \
             patch.object(stage, 'active', return_value=False):
            stage.signal_running('experiment.service', 'CONT')

    def test_no_mutation_while_prerequisites_are_incomplete(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(stage, 'signal_running') as signal, \
             patch.object(stage.subprocess, 'run') as command, \
             patch.object(stage.time, 'sleep', side_effect=InterruptedError):
            (Path(directory) / 'results').mkdir()
            with self.assertRaises(InterruptedError):
                stage.run(SimpleNamespace(root=directory, python='python', source='source.gguf'))
            signal.assert_not_called()
            command.assert_not_called()

    def test_existing_canonical_candidate_takes_priority(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(stage, 'signal_running') as signal, \
             patch.object(stage.subprocess, 'run') as command:
            folder = Path(directory) / 'results/ncm-frozen-clean-training-v32'
            folder.mkdir(parents=True)
            (folder / 'candidate.json').write_text('{}')
            stage.run(SimpleNamespace(root=directory, python='python', source='source.gguf'))
            signal.assert_not_called()
            command.assert_not_called()


if __name__ == '__main__':
    unittest.main()
