import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.ncm.run_resident_training_stage import saved_boundary, retire_obsolete_monitor
from scripts.ncm.run_frozen_ragged_stage import sha


class ResidentStageTests(unittest.TestCase):
    def test_boundary_requires_saved_evaluated_same_checkpoint_and_no_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'results/ncm-compact-tower-v33'
            folder.mkdir(parents=True)
            checkpoint = folder / 'evaluated-2000.pt'
            checkpoint.write_bytes(b'fixture checkpoint')
            for name in ('session', 'progress'):
                (folder / (name + '.json')).write_text(json.dumps({'step': 2000}))
            report = {'evaluated': 12, 'passed': 0, 'checkpoint_sha256': sha(checkpoint)}
            target = folder / 'coding-2000.json'
            target.write_text(json.dumps(report))
            self.assertEqual(saved_boundary(root), checkpoint)
            target.write_text(json.dumps({**report, 'passed': 12}))
            with self.assertRaisesRegex(ValueError, 'candidate'):
                saved_boundary(root)
            target.write_text(json.dumps(report))
            checkpoint.write_bytes(b'changed checkpoint')
            with self.assertRaises(ValueError):
                saved_boundary(root)

    def test_retirement_preserves_pending_checkpoints_and_only_stops_monitor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = root / 'pipeline'
            pipeline.mkdir()
            checkpoints = []
            for name in ('ncm-context-control-v28', 'ncm-context-decoded-v28'):
                folder = root / 'results' / name
                folder.mkdir(parents=True)
                (folder / 'stopped.json').write_text('{}')
                checkpoint = folder / 'evaluated-500.pt'
                checkpoint.write_bytes(b'preserved')
                checkpoints.append(checkpoint)
            monitor = 'ncm-complete-pipeline-v28.service'
            with patch('scripts.ncm.run_resident_training_stage.active', side_effect=lambda u: u == monitor), \
                 patch('scripts.ncm.run_resident_training_stage.subprocess.run') as run:
                retire_obsolete_monitor(root, pipeline)
                run.assert_called_once_with(['systemctl', '--user', 'stop', monitor], check=True)
            manifest = json.loads((pipeline / 'deferred-native-evaluations.json').read_text())
            self.assertEqual(manifest['pending_checkpoints'], sorted(map(str, checkpoints)))
            self.assertTrue(all(p.read_bytes() == b'preserved' for p in checkpoints))
            with patch('scripts.ncm.run_resident_training_stage.active', return_value=True), \
                 patch('scripts.ncm.run_resident_training_stage.subprocess.run') as run:
                with self.assertRaises(RuntimeError):
                    retire_obsolete_monitor(root, pipeline)
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
