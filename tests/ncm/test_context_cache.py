import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.ncm.backbone import digest
from scripts.ncm.codec import encode
from scripts.ncm.context_cache import FeatureCache, merge
from scripts.ncm.context_decoder import DecoderConfig


class FeatureCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = DecoderConfig(source_width=16, latent_width=8, width=32,
            heads=4, layers=2, codes=17, levels=(1, 3, 8), maximum_prompt=16)
        self.rows = []
        for i, split in enumerate(('train', 'validation')):
            self.rows.append(({'id': str(i), 'group': str(i), 'split': split,
                               'target': 'x', 'source': 'test', 'distilled': split == 'train'},
                              {'ids': [2, 3], 'is_final_chunk': True}))
        selection = self.root / 'selection'; selection.mkdir()
        torch.save({'prepared': self.rows}, selection / 'selection.pt')
        checksum = digest(selection / 'selection.pt')
        (selection / 'manifest.json').write_text(json.dumps({'selection_sha256': checksum,
            'counts': {'train': 1, 'validation': 1}, 'training_sources': {'test': 1}}))
        shard = self.root / 'shard'; shard.mkdir()
        identity = {'selection_sha256': checksum, 'shards': 1, 'shard': 0, 'codec_sha256': 'codec'}
        (shard / 'identity.json').write_text(json.dumps(identity))
        records = []
        for i, (row, condition) in enumerate(self.rows):
            path = shard / f'{i}.pt'
            torch.save({'row': row, 'condition': condition, 'hidden': torch.randn(2, 16).bfloat16(),
                        'codes': [torch.zeros(n, dtype=torch.long) for n in (1, 3, 8)],
                        'bytes': torch.tensor(encode('x', 8)), 'codec_exact': True}, path)
            records.append({**{k: row[k] for k in ('id', 'group', 'split', 'source', 'distilled')},
                            'index': i, 'path': path.name, 'sha256': digest(path),
                            'prompt_tokens': 2, 'codec_exact': True})
        self.report = {'identity_sha256': digest(shard / 'identity.json'),
                       'records': records, 'rejected': [], 'complete': True}
        (shard / 'manifest.json').write_text(json.dumps(self.report))

    def test_merge_load_and_bounded_cache(self):
        path = self.root / 'merged.json'
        result = merge(self.root / 'selection', [self.root / 'shard'], path)
        self.assertEqual(result['counts'], {'train': 1, 'validation': 1})
        cache = FeatureCache(path, self.config, maximum_cached_bytes=180)
        self.assertEqual(cache.get(0)['row']['target'], 'x')
        self.assertEqual(cache.get(1)['row']['split'], 'validation')
        self.assertLessEqual(cache.cached_bytes, 180)

    def test_rejects_missing_validation_and_changed_payload(self):
        shard = self.root / 'shard'
        self.report['records'].pop()
        (shard / 'manifest.json').write_text(json.dumps(self.report))
        with self.assertRaises(ValueError): merge(self.root / 'selection', [shard], self.root / 'merged.json')
        (shard / '0.pt').write_bytes(b'corruption')
        with self.assertRaises(ValueError): merge(self.root / 'selection', [shard], self.root / 'merged.json')

    def test_rejects_invalid_label_even_if_manifest_was_rehashed(self):
        shard = self.root / 'shard'
        payload = torch.load(shard / '0.pt', weights_only=True)
        payload['codes'][0][0] = 17
        torch.save(payload, shard / '0.pt')
        self.report['records'][0]['sha256'] = digest(shard / '0.pt')
        (shard / 'manifest.json').write_text(json.dumps(self.report))
        path = self.root / 'merged.json'
        merge(self.root / 'selection', [shard], path)
        with self.assertRaises(ValueError): FeatureCache(path, self.config).get(0)


if __name__ == '__main__': unittest.main()
