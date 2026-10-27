import tempfile,unittest
from types import SimpleNamespace
import numpy as np
from kindi.cache import Openings,_host
class CacheTests(unittest.TestCase):
 def test_cpu_disk_snapshot_roundtrip(self):
  e=SimpleNamespace(ctx=16,M=[np.arange(8,dtype=np.float32)],conv=[np.arange(4,dtype=np.float32)],Kc=[np.arange(32,dtype=np.float16).reshape(16,2)],Vc=[np.arange(32,dtype=np.float16).reshape(16,2)+1])
  before=[x.copy() for x in e.M+e.conv+e.Kc+e.Vc]
  with tempfile.TemporaryDirectory() as d:
   c=Openings(e,d,1,lambda *a:None)
   c.save([1,2,3],[_host(x) for x in e.M],[_host(x) for x in e.conv])
   for x in e.M+e.conv+e.Kc+e.Vc:x.fill(0)
   reloaded=Openings(e,d,1,lambda *a:None)
   p,_,_=reloaded.load(next(iter(reloaded.entries)))
   self.assertEqual(p,3)
   for a,b in zip(e.M+e.conv,before):np.testing.assert_array_equal(a,b)
   np.testing.assert_array_equal(e.Kc[0][:3],before[2][:3]);np.testing.assert_array_equal(e.Vc[0][:3],before[3][:3])
