import unittest
import tempfile
from pathlib import Path
import torch
from scripts.ncm.codec import Codec,CodecConfig,encode
from scripts.ncm.audit_codec_context import dependency


class ContextProbeTests(unittest.TestCase):
    def test_tokenwise_linear_pooling_observes_only_middle_positions(self):
        torch.manual_seed(7341)
        codec=Codec(CodecConfig(length=8,width=16,heads=2,layers=0,codes=16,levels=(1,4,8)))
        codec.eval().requires_grad_(False)
        ids=torch.tensor([encode('abcdefg',8)])
        self.assertEqual(dependency(codec,ids)['positions_above_1e_9'],[3,4])
        self.assertEqual(dependency(codec,ids,area=True)['positions_above_1e_9'],list(range(8)))
        self.assertFalse(codec.embedding._forward_hooks)
        self.assertTrue(all(p.grad is None for p in codec.parameters()))

    def test_contextual_encoder_reaches_entire_window(self):
        torch.manual_seed(7341)
        codec=Codec(CodecConfig(length=8,width=16,heads=2,layers=1,codes=16,levels=(1,4,8)))
        codec.eval().requires_grad_(False)
        ids=torch.tensor([encode('abcdefg',8)])
        self.assertEqual(dependency(codec,ids)['positions_above_1e_9'],list(range(8)))

    def test_area_configuration_targets_and_inference_inputs_agree(self):
        cfg=CodecConfig(length=8,width=16,heads=2,layers=1,codes=16,levels=(1,4,8),downsample='area')
        codec=Codec(cfg).eval()
        ids=torch.tensor([encode('abcdefg',8)]);codes,inputs=codec.targets(ids)
        z=torch.zeros(1,8,16)
        for size,book,index,wanted in zip(cfg.levels,codec.books,codes,inputs):
            torch.testing.assert_close(codec.resample(z,size),wanted)
            z=z+codec.resample(book(index),cfg.length)
        self.assertEqual(dependency(codec,ids)['positions_above_1e_9'],list(range(8)))

    def test_coarse_auxiliary_loss_has_encoder_gradient_and_preserves_eval(self):
        cfg=CodecConfig(length=8,width=16,heads=2,layers=1,codes=16,levels=(1,4,8),downsample='area',coarse_loss_weight=.1)
        codec=Codec(cfg);ids=torch.tensor([encode('abcdefg',8)])
        state=torch.get_rng_state();loss=codec(ids)[0]
        torch.set_rng_state(state);torch.testing.assert_close(loss,codec(ids)[0],rtol=0,atol=0)
        loss.backward()
        self.assertTrue(torch.isfinite(codec.encoder.layers[0].self_attn.in_proj_weight.grad).all())
        self.assertGreater(codec.encoder.layers[0].self_attn.in_proj_weight.grad.norm(),0)
        codec.eval();state=torch.get_rng_state();a=codec(ids)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        codec.config.coarse_loss_weight=0.;b=codec(ids)
        torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
        torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)

    def test_old_codec_defaults_are_preserved_and_pooling_change_rejected(self):
        from scripts.ncm.train import initialize_codec
        old={'length':8,'width':16,'heads':2,'layers':0,'codes':16,'levels':[1,4,8]}
        legacy=Codec(CodecConfig(**old))
        self.assertEqual(legacy.config.downsample,'linear')
        self.assertEqual(legacy.config.coarse_loss_weight,0.)
        changed=Codec(CodecConfig(**old,downsample='area'))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old.pt';torch.save({'config':old,'model':legacy.state_dict(),'step':1},path)
            initialize_codec(legacy,path)
            with self.assertRaisesRegex(ValueError,'configuration differs'):initialize_codec(changed,path)

    def test_complete_selection_prioritizes_intact_documents(self):
        from scripts.ncm.calibrate_codec import complete_selection_score
        def metric(exact,total,passed=False):
            return {'rows':[{'exact_text':i<exact} for i in range(total)],'passed':passed}
        probes=metric(6,6,True)
        high_bytes=complete_selection_score({'nonpad_symbol_accuracy':.99999},metric(20,30),probes)
        intact=complete_selection_score({'nonpad_symbol_accuracy':.9999},metric(29,30),probes)
        passed=complete_selection_score({'nonpad_symbol_accuracy':.9999},metric(30,30,True),probes)
        self.assertGreater(intact,high_bytes);self.assertGreater(passed,intact)

    def test_frozen_representation_digest_ignores_decoder_but_detects_encoder_changes(self):
        from scripts.ncm.calibrate_codec import frozen_parameter_digest
        codec=Codec(CodecConfig(length=8,width=16,heads=2,layers=1,codes=16,levels=(1,4,8)))
        codec.requires_grad_(False);codec.decoder.requires_grad_(True);codec.head.requires_grad_(True)
        before=frozen_parameter_digest(codec)
        with torch.no_grad():codec.head.weight.add_(.01)
        self.assertEqual(before,frozen_parameter_digest(codec))
        with torch.no_grad():codec.embedding.weight[0,0].add_(.01)
        self.assertNotEqual(before,frozen_parameter_digest(codec))


if __name__=='__main__':unittest.main()
