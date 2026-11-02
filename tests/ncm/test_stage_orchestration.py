import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.ncm.run_expanded_stage import Pipeline, qualified_policy
from scripts.ncm.preserve_evaluations import digest


def audit(shared=True, passed=True):
    return [{'runtime': {'shared_prefix_training': shared, 'keep_per_device': 0,
                         'convolution': 'reference'},
             'requested_prompt_tokens': n, 'passed': passed} for n in (64, 69, 512, 2048)]


class StageTests(unittest.TestCase):
    def test_shared_runtime_requires_every_context(self):
        self.assertTrue(qualified_policy({'complete': True, 'records': audit() + audit(False)}))
        records = audit()
        records[-1]['requested_prompt_tokens'] = 512
        with self.assertRaises(RuntimeError):
            qualified_policy({'complete': True, 'records': records})

    def test_reference_fallback_requires_its_own_complete_pass(self):
        self.assertFalse(qualified_policy({'complete': True, 'records': audit(passed=False) + audit(False)}))
        with self.assertRaises(RuntimeError):
            qualified_policy({'complete': True, 'records': audit(passed=False) + audit(False)[:-1]})

    def test_incomplete_audit_cannot_select_a_path(self):
        with self.assertRaises(RuntimeError):
            qualified_policy({'complete': False, 'records': audit()})

    def test_passing_checkpoint_is_preserved_and_only_owned_job_stopped(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / 'student'
            directory.mkdir()
            (directory / 'latest.pt').write_bytes(b'evaluated student weights')
            sha = digest(directory / 'latest.pt')
            (directory / 'coding-250.json').write_text(json.dumps({
                'evaluated': 12, 'passed': 12, 'tasks': [{}] * 12, 'checkpoint_sha256': sha}))
            pipeline = Pipeline(argparse.Namespace(root=root, output='pipeline'))
            with patch('scripts.ncm.run_expanded_stage.stop') as stop, patch.object(pipeline, 'status'):
                candidate = pipeline.wait('owned-training.service', 'student', stop_on_acceptance=True)
            self.assertEqual(digest(candidate), sha)
            stop.assert_called_once_with('owned-training.service')

    def test_partial_or_replaced_checkpoint_cannot_trigger_acceptance(self):
        for evaluated, recorded_sha in ((11, None), (12, '0' * 64)):
            with self.subTest(evaluated=evaluated), tempfile.TemporaryDirectory() as root:
                directory = Path(root) / 'student'
                directory.mkdir()
                (directory / 'latest.pt').write_bytes(b'newer student weights')
                (directory / 'coding-250.json').write_text(json.dumps({
                    'evaluated': evaluated, 'passed': evaluated, 'tasks': [{}] * evaluated,
                    'checkpoint_sha256': recorded_sha or digest(directory / 'latest.pt')}))
                pipeline = Pipeline(argparse.Namespace(root=root, output='pipeline'))
                with patch('scripts.ncm.run_expanded_stage.active', return_value=False), \
                        patch('scripts.ncm.run_expanded_stage.subprocess.run') as run, \
                        patch('scripts.ncm.run_expanded_stage.stop') as stop:
                    run.return_value.stdout = 'success\n'
                    self.assertIsNone(pipeline.wait('owned.service', 'student', stop_on_acceptance=True))
                stop.assert_not_called()


if __name__ == '__main__':
    unittest.main()
