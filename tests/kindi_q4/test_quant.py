"""Verify the transformed pack kernel against independently compiled GGML dequantizers."""
import ctypes as C
import os
from pathlib import Path
import unittest
import numpy as np
from kindi.cpu4 import Mat,PF,PU8,_lib,_p
from kindi.q4pack import decode_blocks,BLOCK

class QuantTests(unittest.TestCase):
 def test_block_layout_and_native_matrix_products(self):
  lib=_lib(); rng=np.random.default_rng(35)
  oracle_path=os.environ.get('GGML_QUANT_ORACLE')
  oracle=C.CDLL(oracle_path) if oracle_path else None
  for ty in (12,14):
   rows,cols=16,512;bs=BLOCK[ty]
   raw=rng.integers(0,256,(rows,cols//256,bs),dtype=np.uint8)
   for offset in ([0,2] if ty==12 else [208]):
    raw[:,:,offset:offset+2]=np.array([0.013],dtype='<f2').view(np.uint8)
   dense=decode_blocks(raw,ty).reshape(rows,cols)
   if oracle:
    reference=np.empty_like(dense);fn=getattr(oracle,'dequantize_row_q%d_K'%(4 if ty==12 else 6))
    fn.argtypes=[PU8,PF,C.c_int64];fn(_p(raw,PU8),_p(reference),rows*cols)
    np.testing.assert_array_equal(dense,reference)
   tile=raw.reshape(rows//8,8,cols//256,bs).transpose(0,2,1,3).copy()
   mat=Mat(_p(tile,PU8),None,rows,cols,ty,0,tile.nbytes)
   restored=np.empty(cols,np.float32)
   for row in (0,7,8,15):
    lib.kc_dequant(C.byref(mat),row,_p(restored));np.testing.assert_array_equal(restored,dense[row])
   x=rng.normal(size=cols).astype(np.float32);y=np.empty(rows,np.float32)
   lib.kc_matvec(C.byref(mat),_p(x),_p(y))
   np.testing.assert_allclose(y,dense.astype(np.float64)@x.astype(np.float64),atol=0.00015,rtol=0.00001)

if __name__=='__main__':unittest.main()
