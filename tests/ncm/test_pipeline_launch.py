import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.ncm.run_expanded_stage import Pipeline


class PipelineLaunchTests(unittest.TestCase):
    def make_pipeline(self, folder):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.root = Path(folder)
        pipeline.out = Path(folder)
        pipeline.units = []
        return pipeline

    def test_active_unit_cannot_be_reset_or_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            pipeline = self.make_pipeline(folder)
            with patch('scripts.ncm.run_expanded_stage.active', return_value=True), \
                 patch('scripts.ncm.run_expanded_stage.subprocess.run') as run:
                with self.assertRaises(RuntimeError):
                    pipeline.launch('ncm-owned-test', ['python', '-m', 'training'])
                run.assert_not_called()

    def test_inactive_failure_released_before_optimizer_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            pipeline = self.make_pipeline(folder)
            with patch('scripts.ncm.run_expanded_stage.active', return_value=False), \
                 patch('scripts.ncm.run_expanded_stage.subprocess.run') as run:
                pipeline.launch('ncm-owned-test', ['python', '-m', 'training', '--resume'])
                first, second = run.call_args_list
                self.assertEqual(first.args[0], ['systemctl', '--user', 'reset-failed', 'ncm-owned-test'])
                self.assertEqual(second.args[0][0], 'systemd-run')
                self.assertEqual(second.args[0][-4:], ['python', '-m', 'training', '--resume'])
                self.assertTrue(second.kwargs['check'])
                saved = json.loads((Path(folder) / 'ncm-owned-test-launch.json').read_text())
                self.assertEqual(saved, second.args[0])


if __name__ == '__main__':
    unittest.main()
