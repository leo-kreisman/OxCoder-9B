import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.ncm.preserve_evaluations import verified_copy


class SnapshotTests(unittest.TestCase):
    def test_mismatched_checkpoint_cannot_receive_evaluation_label(self):
        with tempfile.TemporaryDirectory() as d:
            source, destination = Path(d)/'latest.pt', Path(d)/'evaluated.pt'
            source.write_bytes(b'newer unevaluated model')
            self.assertFalse(verified_copy(source, destination, hashlib.sha256(b'evaluated model').hexdigest()))
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix('.pt.tmp').exists())

    def test_matching_bytes_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            source, destination = Path(d)/'latest.pt', Path(d)/'evaluated.pt'
            source.write_bytes(b'evaluated model')
            self.assertTrue(verified_copy(source, destination, hashlib.sha256(source.read_bytes()).hexdigest()))
            self.assertEqual(source.read_bytes(), destination.read_bytes())


if __name__ == '__main__':
    unittest.main()
