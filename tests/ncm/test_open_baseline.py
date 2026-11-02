import json
import fcntl
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from scripts.ncm.open_corpus import NearDuplicates, eligible, shingles, split_for, file_sha
from scripts.ncm.open_baseline import (prepare, Corpus, PAD, EOS, BOS, LENGTH,
    corrupt_codes, upstream, codec_config, generator_config, generator_inputs, LEVELS,
    validate_codec_config)


def fixture(root):
    source = root/'raw'
    source.mkdir()
    rows = []
    for split in ('train', 'validation', 'test'):
        for kind in ('code', 'assistant'):
            for i in range(4):
                text = (f'def f_{split}_{i}(xs):\n    # café 🧪\n    return sum(xs) + {i}\n'
                        if kind == 'code' else f'Use a loop for case {split} {i}.\n```python\nprint("Olá")\n```')
                rows.append({'id': f'{split}:{kind}:{i}', 'group': f'{split}:{kind}:{i}',
                             'split': split, 'kind': kind, 'text': text, 'text_sha256': __import__('hashlib').sha256(text.encode()).hexdigest()})
    with (source/'documents.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row)+'\n')
    (source/'manifest.json').write_text(json.dumps({'documents_sha256': file_sha(source/'documents.jsonl')}))
    prepare(SimpleNamespace(corpus=str(source), output=str(root/'prepared')))
    return root/'prepared'


class CorpusTests(unittest.TestCase):
    def test_refined_config_acceptance_is_narrow_and_nonmutating(self):
        c=codec_config(8192)
        c['dropout']=0.
        c['vector_quantizer_config']['aux']['fine_drop']['prob']=0.
        before=json.dumps(c,sort_keys=True)
        validate_codec_config(c,8192)
        self.assertEqual(before,json.dumps(c,sort_keys=True))
        c['embed_dim']=512
        with self.assertRaisesRegex(ValueError,'Unreviewed'):
            validate_codec_config(c,8192)

    def test_dedup_and_benchmark_normalization(self):
        d = NearDuplicates()
        self.assertFalse(d.duplicate('def add(x, y):\n return x+y'))
        self.assertTrue(d.duplicate('def add(x,y): return x + y'))
        text = ' '.join('symbol'+str(i) for i in range(80))
        self.assertEqual(shingles(text), shingles(text.replace(' ', '\n')))
        self.assertEqual(split_for('owner/repo'), split_for('owner/repo'))

    def test_filter_requires_license_and_parseable_python(self):
        row = {'language': 'Python', 'license_type': 'permissive', 'file_path': 'a.py',
               'content': '#'+('comment '*20)+'\ndef add(x,y):\n    return x+y\n'}
        self.assertTrue(eligible(row))
        self.assertFalse(eligible(row | {'license_type': 'no_license'}))
        self.assertFalse(eligible(row | {'content': row['content']+'def invalid('}))
        self.assertFalse(eligible(row | {'file_path': 'HumanEval/test.py'}))

    def test_tokenizer_roundtrip_and_window_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Corpus(fixture(Path(tmp)))
            for row in data.rows:
                prefix, target = data.window(row, 0)
                self.assertEqual(prefix[-1], BOS)
                self.assertTrue(np.all(prefix[:-1] == PAD))
                n = row['length']-1
                self.assertEqual(target[n-1], EOS)
                self.assertTrue(np.all(target[n:] == PAD))
            # An interior window's prefix cannot contain its target or future.
            ids = np.arange(500, dtype=np.uint16)+10
            data.arrays['train'] = ids
            row = {'split':'train', 'offset':0, 'length':500}
            prefix, target = data.window(row, 1)
            np.testing.assert_array_equal(prefix, ids[1:129])
            np.testing.assert_array_equal(target, ids[129:257])

    def test_prepared_checksum_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = fixture(Path(tmp))
            with (path/'train.bin').open('ab') as f:
                f.write(b'bad')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                Corpus(path)

    def test_corruption_never_mutates_targets(self):
        clean = torch.arange(14).reshape(1,14) % 8
        original = clean.clone()
        noisy = corrupt_codes(clean, [2,4,8], 8, 1.)
        self.assertTrue(torch.equal(original, clean))
        self.assertTrue(torch.equal(noisy[:,-8:], clean[:,-8:]))
        self.assertTrue(torch.equal(corrupt_codes(clean,[2,4,8],8,0.), clean))

    def test_pipeline_rejects_second_writer_before_state_changes(self):
        from scripts.ncm.run_open_baseline import run
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            marker = out/'pipeline.json'
            marker.write_text('{"phase":"active-test"}')
            with (out/'.pipeline.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                args=SimpleNamespace(output=tmp,resume=True)
                with self.assertRaises(BlockingIOError):
                    run(args)
            self.assertEqual(marker.read_text(),'{"phase":"active-test"}')


class UpstreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path('results/ncm-upstream').exists():
            raise unittest.SkipTest('Pinned upstream checkout required')
        from omegaconf import OmegaConf
        torch.set_num_threads(2)
        torch.manual_seed(37)
        V, G = upstream('results/ncm-upstream')
        c = codec_config(300)
        c['embed_dim'] = 32
        c['n_layers'] = 2
        c['vqvae_vocab_size'] = [32]*len(LEVELS)
        cls.codec = V(**OmegaConf.create(c)).eval().requires_grad_(False)
        g = generator_config()
        g.update(n_embd=32,n_head=2,n_layer=1,vocab_size=[32]*len(LEVELS),dropout=0.)
        cls.model = G(cls.codec,OmegaConf.create(g)).eval()

    def test_finest_targets_cannot_leak_into_inputs(self):
        clean = torch.randint(0,32,(1,sum(LEVELS)))
        modified = clean.clone()
        modified[:,-128:] = (modified[:,-128:]+1)%32
        p = torch.ones(1,128,dtype=torch.long)
        a = generator_inputs(self.model,self.codec,p,clean)
        b = generator_inputs(self.model,self.codec,p,modified)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_sequential_verification_matches_teacher_forced_logits(self):
        # Force each sampled level to the known code and compare the logits
        # computed by the free path with the block-diagonal training path.
        clean = torch.randint(0,32,(1,sum(LEVELS)))
        p = torch.ones(1,128,dtype=torch.long)
        p[:,:100]=0
        with torch.no_grad():
            expected = self.model(generator_inputs(self.model,self.codec,p,clean),prefix_pad_mask=p==PAD)
        pieces = list(clean.split(LEVELS,1))
        seen = []
        original = self.model._sample_level
        def sample(logits,*args):
            seen.append(logits.detach())
            return pieces[len(seen)-1]
        self.model._sample_level = sample
        try:
            self.model.generate(None,p,temperature=0.,top_k=0,top_p=1.,use_kv_cache=False)
        finally:
            self.model._sample_level=original
        torch.testing.assert_close(torch.cat(seen,1),expected,atol=1e-5,rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
