import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.ncm.prepared_reader import load, restore, identical


class PreparedReaderTests(unittest.TestCase):
    def fixture(self):
        return {'prepared': [({'id': 'x', 'target': 'é\n', 'split': 'train'},
                              {'ids': [3, 4, 90000], 'is_final_chunk': False})],
                'data_manifest': {'count': 1, 'rate': .8}, 'prompt_tokens': 2048}

    def write_cache(self, folder):
        folder = Path(folder)
        encoded = json.dumps(self.fixture()).encode()
        (folder / 'data.json').write_bytes(encoded)
        manifest = {'format': 'ncm-prepared-json-v1', 'source_sha256': 'source',
                    'json_sha256': hashlib.sha256(encoded).hexdigest(), 'rows': 1,
                    'exact_type_value_comparison': True}
        (folder / 'manifest.json').write_text(json.dumps(manifest))

    def test_exact_types_order_and_values(self):
        data = self.fixture()
        self.assertTrue(identical(data, restore(json.loads(json.dumps(data)))))
        self.assertFalse(identical([1, 2], (1, 2)))
        self.assertFalse(identical(1, True))
        self.assertFalse(identical(float('nan'), float('nan')))
        self.assertFalse(identical({'a': 1, 'b': 2}, {'b': 2, 'a': 1}))

    def test_cache_identity_size_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            self.write_cache(folder)
            self.assertTrue(identical(self.fixture(), load(folder, 'source')))
            with self.assertRaises(ValueError): load(folder, 'other')
            with self.assertRaises(ValueError): load(folder, 'source', max_bytes=20)
            with (Path(folder) / 'data.json').open('ab') as handle:
                handle.write(b' ')
            with self.assertRaises(ValueError): load(folder, 'source')

    def test_invalid_prepared_envelopes(self):
        for data in [{}, {'prepared': [], 'data_manifest': {}, 'prompt_tokens': True},
                     {'prepared': [[1, 2]], 'data_manifest': {}, 'prompt_tokens': 2048}]:
            with self.assertRaises(ValueError): restore(data)


if __name__ == '__main__':
    unittest.main()
