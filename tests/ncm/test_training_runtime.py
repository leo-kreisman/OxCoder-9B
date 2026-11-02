import unittest
import json
import tempfile
from pathlib import Path
from scripts.ncm.training_runtime import retained_layers


class RuntimeSelection(unittest.TestCase):
    def test_balanced_and_bounded(self):
        devices=[0]*12+[1]*20
        self.assertEqual(retained_layers(devices,0),[])
        self.assertEqual(retained_layers(devices,2),[0,6,12,22])
        self.assertEqual(len(set(retained_layers(devices,12))),24)
        with self.assertRaises(ValueError):retained_layers(devices,13)
        with self.assertRaises(ValueError):retained_layers(devices,-1)

    def test_gate_rejects_missing_context_failed_gradient_and_changed_source(self):
        from scripts.ncm.training_runtime import validate_audit
        from scripts.ncm.backbone import digest
        root=Path(__file__).resolve().parents[2]/'scripts'/'ncm'
        identity={'source_sha256':'source','codec_sha256':'codec','rank':32,'head_kind':'untied'}
        report={'complete':True,'prepared_sha256':'data','model_identity':identity,
                'checkpoint_sha256':'checkpoint','implementation_sha256':{
                    n:digest(root/n) for n in ('training_runtime.py','shared_training.py','vendor/veomni_shared_prefix.py','backbone.py')},
                'records':[{'requested_prompt_tokens':n,'runtime':{'keep_per_device':2,'convolution':'reference'},
                            'passed':True,'gradient_comparison':{'finite':True,'relative_l2':.001,'cosine':.999},
                            'gradient_groups':{k:{'finite':True,'relative_l2':.001,'cosine':.999} for k in ('adapters','interfaces','classifiers')},
                            'logit_comparison':{'finite':True,'relative_l2':.001}} for n in [64,69,512,2048]]}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'audit.json'
            def write():path.write_text(json.dumps(report))
            write();self.assertEqual(validate_audit(path,identity,'data',2,'reference',False)['keep_per_device'],2)
            with self.assertRaises(ValueError):validate_audit(path,{**identity,'source_sha256':'wrong'},'data',2,'reference',False)
            report['records'][-1]['gradient_comparison']['relative_l2']=.5;write()
            with self.assertRaises(ValueError):validate_audit(path,identity,'data',2,'reference',False)
            report['records'].pop();write()
            with self.assertRaises(ValueError):validate_audit(path,identity,'data',2,'reference',False)


if __name__=='__main__':unittest.main()
