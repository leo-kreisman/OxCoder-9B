import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from kindi.q4pack import PROFILE_DIMS,verify_pack,verification_full
class ValidationTests(unittest.TestCase):
 def test_rejects_changed_dimensions_before_buffer_allocation(self):
  with tempfile.TemporaryDirectory() as d:
   bad=dict(PROFILE_DIMS,width=8192)
   Path(d,'kindi_pack.json').write_text(json.dumps(dict(pack_format='kindi-q4-v1',dimensions=bad,head='output.weight',hadamard_block=1)))
   with self.assertRaisesRegex(ValueError,'Unsupported Q4 profile'):verify_pack(d)
 def test_rejects_manifest_traversal(self):
  with tempfile.TemporaryDirectory() as d:
   Path(d,'kindi_pack.json').write_text(json.dumps(dict(pack_format='kindi-q4-v1',dimensions=PROFILE_DIMS,head='output.weight',hadamard_block=1,files={'../x':{}})))
   with self.assertRaisesRegex(ValueError,'Invalid manifest path'):verify_pack(d)
 def test_verification_mode_is_explicit(self):
  for mode,expected in [('full',True),('size',False)]:
   with patch.dict(os.environ,KINDI_Q4_VERIFY=mode):self.assertEqual(verification_full(),expected)
  with patch.dict(os.environ,KINDI_Q4_VERIFY='typo'):
   with self.assertRaises(ValueError):verification_full()
