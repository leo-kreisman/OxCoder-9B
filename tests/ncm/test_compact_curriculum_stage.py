from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts.ncm import run_compact_curriculum_stage as stage


class CompactStageTests(unittest.TestCase):
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
