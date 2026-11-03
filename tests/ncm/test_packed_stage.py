import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.ncm import run_packed_training_stage as stage


class PackedStageTests(unittest.TestCase):
    def exercise(self, qualified):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / 'results/ncm-resident-tower-v34'
            parent.mkdir(parents=True)
            checkpoint = parent / 'evaluated-2500.pt'
            checkpoint.write_bytes(b'parent checkpoint fixture')
            for name in ('session', 'progress'):
                (parent / (name + '.json')).write_text('{"step":2500}')
            (parent / 'coding-2500.json').write_text(json.dumps({
                'evaluated': 12, 'passed': 0, 'checkpoint_sha256': stage.sha(checkpoint)}))
            audit = root / 'results/ncm-packed-runtime-audit-v35'
            audit.mkdir()
            (audit / 'report.json').write_text(json.dumps({'complete': True, 'passed': qualified,
                'speed_qualified': qualified, 'checkpoint_sha256': stage.sha(checkpoint), 'corpus': {'test': 1}}))
            (root / 'results/ncm-selection-reader-audit-v35.json').write_text(json.dumps({
                'complete': True, 'passed': True, 'corpus': {'test': 1}}))
            target = root / 'results/ncm-packed-tower-v35'
            target.mkdir()
            (target / 'session.json').write_text('{"step":10000}')
            pipeline = root / 'pipeline'
            pipeline.mkdir()
            old = 'ncm-resident-pipeline-v34.service'
            running = {old}
            commands = []

            def execute(command, **kwargs):
                commands.append(command)
                if command[:3] == ['systemctl', '--user', 'stop']:
                    running.discard(command[3])
                return SimpleNamespace(returncode=0)

            with patch.object(stage, 'active', side_effect=lambda unit: unit in running), \
                 patch.object(stage, 'signal_running'), patch.object(stage.subprocess, 'run', side_effect=execute):
                args = SimpleNamespace(python='python', source='fixture.gguf')
                if qualified:
                    stage.run_locked(args, root, pipeline)
                    handover = json.loads((pipeline / 'handover.json').read_text())
                    self.assertEqual(handover['parent_sha256'], stage.sha(checkpoint))
                    self.assertEqual(handover['step'], 2500)
                    self.assertFalse(any(c[0] == 'systemd-run' for c in commands))
                else:
                    with self.assertRaises(ValueError):
                        stage.run_locked(args, root, pipeline)
                    restart = [c for c in commands if c[0] == 'systemd-run']
                    self.assertEqual(len(restart), 1)
                    self.assertIn('scripts.ncm.run_resident_training_stage', restart[0])
                    self.assertFalse((pipeline / 'handover.json').exists())
            self.assertEqual(commands[0], ['systemctl', '--user', 'stop', old])

    def test_qualified_handover_pins_parent(self):
        self.exercise(True)

    def test_failed_audit_restores_original_supervisor(self):
        self.exercise(False)


if __name__ == '__main__':
    unittest.main()
