from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.ncm.run_cpu_optimizer_stage import CPUOptimizerPipeline


class CPUStageTests(unittest.TestCase):
    def test_incomplete_or_unpreserved_smoke_never_launches_training(self):
        for failure in ['incomplete', 'wrong_step', 'unpreserved', 'partial_evaluation']:
            with self.subTest(failure=failure), TemporaryDirectory() as directory:
                args = SimpleNamespace(root=directory, output='pipeline', training_unit='ncm-parallel-v24',
                                       steps=5000, python='python', source='source')
                pipeline = CPUOptimizerPipeline(args)
                def read(path):
                    if path.name == 'progress.json':
                        return {'training_complete': failure != 'incomplete', 'step': 359 if failure == 'wrong_step' else 360}
                    if path.name == 'identity.json':
                        return {'implementation_sha256': {'cpu_adamw.py': 'sha'}, 'prepared_sha256': 'data',
                                'cpu_head_optimizer': {'audit_sha256': 'sha'}}
                    if path.name == 'manifest.json': return {'prepared_sha256': 'data'}
                    return {'snapshot_preserved': failure != 'unpreserved',
                            'evaluated': 11 if failure == 'partial_evaluation' else 12, 'passed': 12}
                with patch('scripts.ncm.run_cpu_optimizer_stage.read', side_effect=read), \
                        patch('scripts.ncm.run_cpu_optimizer_stage.digest', return_value='sha'), \
                        patch('scripts.ncm.run_cpu_optimizer_stage.process'), \
                        patch.object(pipeline, 'wait'), patch.object(pipeline, 'launch') as launch, \
                        patch.object(pipeline, 'replay') as replay:
                    with self.assertRaises(ValueError): pipeline.run()
                launch.assert_not_called(); replay.assert_not_called()


if __name__ == '__main__':
    unittest.main()
