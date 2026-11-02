import copy
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts.ncm.calibrate_open_codec import calibration_loss, representation_digest, token_coverage
from scripts.ncm.open_baseline import codec_config, upstream, LEVELS
from scripts.ncm.open_corpus import file_sha
from scripts.ncm.calibrate_open_codec import run
from test_open_baseline import fixture


class CalibrationTests(unittest.TestCase):
    def test_synthetic_coverage_range_and_termination(self):
        torch.manual_seed(47)
        x=token_coverage(16,8192,'cpu')
        self.assertEqual(tuple(x.shape),(16,128))
        self.assertTrue(bool(((x>=0)&(x<8192)).all()))
        self.assertFalse(bool((x==1).any()))
        for row in x[:8]:
            stop=(row==2).nonzero().flatten()
            self.assertEqual(len(stop),1)
            self.assertTrue(bool((row[int(stop[0])+1:]==0).all()))
        self.assertTrue(bool((x[8:]>=3).all()))
    @classmethod
    def setUpClass(cls):
        if not Path('results/ncm-upstream').exists():
            raise unittest.SkipTest('Pinned upstream checkout required')
        from omegaconf import OmegaConf
        torch.set_num_threads(2)
        V, _ = upstream('results/ncm-upstream')
        cfg = codec_config(300)
        cfg.update(embed_dim=32,n_layers=2,vqvae_vocab_size=[32]*len(LEVELS))
        cls.template = V(**OmegaConf.create(cfg)).eval().requires_grad_(False)

    def test_gradient_step_changes_decoder_without_codes_or_ema(self):
        codec=copy.deepcopy(self.template)
        codec.decoder.requires_grad_(True)
        y=torch.randint(3,300,(2,128))
        frozen=representation_digest(codec)
        before=codec.encode(y).clone()
        decoder_before={k:v.clone() for k,v in codec.decoder.state_dict().items()}
        opt=torch.optim.AdamW(codec.decoder.parameters(),lr=.0001)
        # Accidental train() in a caller must not enable EMA/dropout.
        codec.train()
        loss=calibration_loss(codec,y)
        loss.backward()
        self.assertTrue(all(p.grad is None for n,p in codec.named_parameters() if not n.startswith('decoder.')))
        self.assertTrue(any(p.grad is not None for p in codec.decoder.parameters()))
        opt.step()
        self.assertEqual(frozen,representation_digest(codec))
        torch.testing.assert_close(codec.encode(y),before,rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(decoder_before[k],v) for k,v in codec.decoder.state_dict().items()))

    def test_digest_detects_buffer_mutation_but_allows_decoder_update(self):
        codec=copy.deepcopy(self.template)
        before=representation_digest(codec)
        with torch.no_grad():
            next(codec.decoder.parameters()).add_(1)
        self.assertEqual(before,representation_digest(codec))
        codec.vq.ema_w_0.add_(1)
        self.assertNotEqual(before,representation_digest(codec))

    def test_interrupted_joint_training_matches_uninterrupted(self):
        from omegaconf import OmegaConf
        import json
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            data=fixture(root)
            manifest=json.loads((data/'manifest.json').read_text())
            V,_=upstream('results/ncm-upstream')
            cfg=codec_config(manifest['vocab_size'])
            cfg.update(embed_dim=32,n_layers=2,vqvae_vocab_size=[32]*len(LEVELS))
            initial=V(**OmegaConf.create(cfg))
            checkpoint=root/'initial.pt'
            torch.save({'identity':{'codec_config':cfg,'data_sha256':file_sha(data/'manifest.json')},
                        'model':initial.state_dict()},checkpoint)
            def args(output,**kw):
                return SimpleNamespace(checkpoint=str(checkpoint),data=str(data),output=str(root/output),
                    upstream='results/ncm-upstream',device='cpu',scope='joint',steps=2,batch=2,
                    accumulation=1,lr=.0001,seed=47,eval_every=2,save_every=1,
                    coverage_fraction=.5,
                    **({'resume':False,'stop_after':None}|kw))
            # The diagnostic is separately exercised; no need to repeat 256
            # windows for each restart control on this tiny synthetic fixture.
            with patch('scripts.ncm.calibrate_open_codec.diagnose',return_value={'scope':'test'}):
                run(args('direct'))
                run(args('resumed',stop_after=1))
                run(args('resumed',resume=True))
            direct=torch.load(root/'direct/latest.pt',weights_only=True)
            resumed=torch.load(root/'resumed/latest.pt',weights_only=True)
            self.assertEqual(direct['data_rng'],resumed['data_rng'])
            self.assertEqual(direct['exposure'],resumed['exposure'])
            for k,v in direct['model'].items():
                torch.testing.assert_close(v,resumed['model'][k],rtol=0,atol=0)
            self.assertTrue(any(not torch.equal(initial.state_dict()[k],v)
                for k,v in direct['model'].items() if k.startswith('vq.ema_w_')))


if __name__=='__main__':
    unittest.main()
