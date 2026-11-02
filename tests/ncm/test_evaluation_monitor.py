import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.ncm import preserve_evaluations as preservation
from scripts.ncm.run_expanded_stage import Pipeline


class EvaluationMonitorTests(unittest.TestCase):
    def test_observer_can_start_before_trainer_directory_without_creating_it(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'not-created-by-trainer-yet'
            self.assertEqual(preservation.process(root), [])
            self.assertFalse(root.exists())
            root.mkdir(exist_ok=False)
            self.assertEqual(preservation.process(root), [])

    def test_concurrent_observers_preserve_once_without_stopping_monitor(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'latest.pt').write_bytes(b'immutable checkpoint fixture')
            expected = preservation.digest(root / 'latest.pt')
            report = {'evaluated': 12, 'passed': 0, 'tasks': [{} for _ in range(12)],
                      'checkpoint_sha256': expected}
            (root / 'coding-250.json').write_text(json.dumps(report))
            entered, release = Event(), Event()
            real_copy = preservation.verified_copy
            results, errors = [], []
            def slow_copy(*args):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError('Test observer was not released')
                return real_copy(*args)
            def observe():
                try: results.extend(preservation.process(root))
                except BaseException as error: errors.append(error)
            with patch.object(preservation, 'verified_copy', slow_copy):
                thread = Thread(target=observe)
                thread.start()
                try:
                    self.assertTrue(entered.wait(5))
                    self.assertEqual(preservation.process(root), [])
                finally:
                    release.set(); thread.join(5)
            self.assertFalse(errors)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0]['snapshot_preserved'])
            self.assertEqual(preservation.digest(root / 'evaluated-250.pt'), expected)
            self.assertEqual(preservation.process(root), [])

    def test_monitor_resume_requires_existing_state_and_default_stays_exclusive(self):
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(root=directory, output='pipeline')
            with self.assertRaises(ValueError): Pipeline(args, resume=True)
            original = Pipeline(args)
            original.status('running')
            with self.assertRaises(FileExistsError): Pipeline(args)
            resumed = Pipeline(args, resume=True)
            self.assertEqual(original.out, resumed.out)


if __name__ == '__main__':
    unittest.main()
