import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.ncm.selection_reader import prepare, load, digest, restore
from scripts.ncm.prepared_reader import identical
from scripts.ncm.frozen_tower_training import load_corpus as reference
from scripts.ncm.cached_tower_corpus import load_corpus as cached


class SelectionReaderTests(unittest.TestCase):
    def test_roundtrip_preserves_types_order_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'selection.pt'
            data = {'prepared': [({'id': 'x', 'value': None}, {'ids': [0, 7, 999], 'last': True})]}
            torch.save(data, source)
            sha = digest(source)
            cache = root / 'cache'
            prepare(source, sha, cache)
            self.assertTrue(identical(data, load(cache, sha)))
            with self.assertRaises(ValueError):
                load(cache, 'wrong')
            with self.assertRaises(ValueError):
                load(cache, sha, max_bytes=1)
            (cache / 'data.json').write_text('{}')
            with self.assertRaises(ValueError):
                load(cache, sha)

    def test_whole_admission_matches_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            sources = ['TeichAI/Ox-Alpha-10k', 'TeichAI/DeepSeek-v4-Pro-Agent', 'nvidia/OpenCodeInstruct']
            for i in range(67):
                row = {'id': str(i), 'group': str(i), 'source': sources[i % 3],
                       'split': 'train' if i < 3 else 'validation', 'distilled': i == 0}
                condition = {'ids': [1, 2], 'response_id': str(i), 'chunk_index': 0, 'is_final_chunk': True}
                rows.append((row, condition))
            torch.save({'prepared': rows}, root / 'selection.pt')
            sha = digest(root / 'selection.pt')
            report = {'complete': True, 'identity': {'source_sha256': 'source', 'codec_sha256': 'codec',
                      'selection_sha256': sha, 'complete_training_responses_only': True},
                      'records': [{**r, 'index': i, 'prompt_tokens': 2, 'codec_exact': True} for i, (r, _) in enumerate(rows)]}
            manifest = root / 'admission.json'
            manifest.write_text(json.dumps(report))
            prepare(root / 'selection.pt', sha, root / 'cache')
            a = reference(root, manifest, 'source', 'codec')
            b = cached(root, manifest, 'source', 'codec', root / 'cache')
            self.assertTrue(identical(a, b))
            report['records'][0]['codec_exact'] = False
            manifest.write_text(json.dumps(report))
            for reader, extra in [(reference, ()), (cached, (root / 'cache',))]:
                with self.assertRaises(ValueError):
                    reader(root, manifest, 'source', 'codec', *extra)

    def test_wrong_envelope_is_rejected(self):
        for value in ({'prepared': [], 'extra': 1}, {'prepared': [({}, 1)]}, []):
            with self.assertRaises(ValueError):
                restore(value)


if __name__ == '__main__':
    unittest.main()
