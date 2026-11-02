import json
import tempfile
import unittest
from pathlib import Path
from scripts.ncm.continue_generator import validate_initialization


class ContinuationTests(unittest.TestCase):
    def test_requires_completed_audited_initialization_and_matching_codec(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            training={'complete':True,'identity':{'codec_sha256':'codec'}}
            structure={'checkpoint_reload_exact':True,'all_32_adapters_receive_gradients':True}
            audit={'passed':True}
            def write():
                for name,data in [('training',training),('structural-validation',structure),('backbone-audit',audit)]:
                    (root/(name+'.json')).write_text(json.dumps(data))
            write();self.assertEqual(validate_initialization(root,'codec'),training['identity'])
            with self.assertRaisesRegex(ValueError,'different codec'):validate_initialization(root,'wrong')
            training['complete']=False;write()
            with self.assertRaisesRegex(ValueError,'did not complete'):validate_initialization(root,'codec')
            training['complete']=True;structure['checkpoint_reload_exact']=False;write()
            with self.assertRaisesRegex(ValueError,'structural'):validate_initialization(root,'codec')
            structure['checkpoint_reload_exact']=True;audit['passed']=False;write()
            with self.assertRaisesRegex(ValueError,'backbone'):validate_initialization(root,'codec')


if __name__=='__main__':unittest.main()
