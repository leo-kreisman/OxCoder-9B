"""Lossless Q4_K/Q6_K -> Kindi row-tiled pack. SPDX-License-Identifier: AGPL-3.0-only.

Each matrix is transposed from [row][group][block_byte] to
[row//8][group][row%8][block_byte]. Original quantized blocks remain intact.
Small gate projections are exactly dequantized to f32 for Kindi's GDN gates.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from .gguf import GGUF

BLOCK={12:144,14:210}
PAGE=16384  # Both Apple Silicon pages and Linux 4 KiB pages divide this.

def expected_shapes():
    shapes={'output.weight':(248320,4096),'token_embd.weight':(248320,4096),'output_norm.weight':(1,4096)}
    for layer in range(32):
        p='blk.%d.'%layer
        common={'attn_norm.weight':(1,4096),'post_attention_norm.weight':(1,4096),
                'ffn_gate.weight':(12288,4096),'ffn_up.weight':(12288,4096),'ffn_down.weight':(4096,12288)}
        if layer%4==3:
            common.update({'attn_q.weight':(8192,4096),'attn_k.weight':(1024,4096),'attn_v.weight':(1024,4096),
                           'attn_output.weight':(4096,4096),'attn_q_norm.weight':(1,256),'attn_k_norm.weight':(1,256)})
        else:
            common.update({'attn_qkv.weight':(8192,4096),'attn_gate.weight':(4096,4096),'ssm_out.weight':(4096,4096),
                           'ssm_alpha.weight':(32,4096),'ssm_beta.weight':(32,4096),'ssm_a':(1,32),
                           'ssm_dt.bias':(1,32),'ssm_conv1d.weight':(8192,4),'ssm_norm.weight':(1,128)})
        shapes.update({p+k:v for k,v in common.items()})
    return shapes

def validate_shapes(index):
    shapes=expected_shapes()
    if set(index)!=set(shapes):raise ValueError('Missing/extra tensors for dense Qwen3.5 9B')
    for name,(rows,cols) in shapes.items():
        rec=index[name]
        if (rec['rows'],rec['ne0'])!=(rows,cols):raise ValueError('Unexpected shape: '+name)
        is_float=rows==1 or name.endswith(('ssm_alpha.weight','ssm_beta.weight','ssm_conv1d.weight'))
        if (rec['type']==0)!=is_float:raise ValueError('Unexpected tensor storage: '+name)

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()

def decode_blocks(raw,ty):
    """Independent numpy reference for conversion verification and small gates."""
    raw=np.asarray(raw,dtype=np.uint8).reshape(-1,BLOCK[ty])
    out=np.empty((len(raw),256),np.float32)
    if ty==12:
        d=raw[:,:2].copy().view('<f2').ravel().astype(np.float32)
        dm=raw[:,2:4].copy().view('<f2').ravel().astype(np.float32)
        s=raw[:,4:16]
        for k in range(8):
            sc=s[:,k]&63 if k<4 else (s[:,k+4]&15)|((s[:,k-4]>>6)<<4)
            mn=s[:,k+4]&63 if k<4 else (s[:,k+4]>>4)|((s[:,k]>>6)<<4)
            q=(raw[:,16+(k//2)*32:16+(k//2+1)*32]>>((k%2)*4))&15
            out[:,k*32:(k+1)*32]=(d*sc)[:,None]*q-(dm*mn)[:,None]
    else:
        d=raw[:,208:210].copy().view('<f2').ravel().astype(np.float32)
        s=raw[:,192:208].view(np.int8)
        for h in range(2):
            for k in range(4):
                low=(raw[:,h*64+(k%2)*32:h*64+(k%2+1)*32]>>((k//2)*4))&15
                high=(raw[:,128+h*32:160+h*32]>>(2*k))&3
                q=(low|(high<<4)).astype(np.int16)-32
                for j in range(2):
                    off=h*128+k*32+j*16
                    out[:,off:off+16]=(d*s[:,h*8+k*2+j])[:,None]*q[:,j*16:(j+1)*16]
    return out

PROFILE_DIMS = {'layers': 32, 'width': 4096, 'ffn': 12288, 'heads': 16, 'kv_heads': 4, 'head_dim': 256, 'rope_dim': 64, 'groups': 16, 'state_size': 128, 'gdn_inner': 4096, 'rope_base': 10000000.0, 'full_attention_interval': 4}

def verification_full():
    import os
    mode=os.environ.get("KINDI_Q4_VERIFY","full")
    if mode not in ("full","size"): raise ValueError("KINDI_Q4_VERIFY must be full or size")
    return mode=="full"

def verify_pack(pack,full=True):
    pack=Path(pack); man=json.loads((pack/'kindi_pack.json').read_text())
    if man['pack_format']!='kindi-q4-v1': raise ValueError('Unsupported pack')
    if man.get('dimensions')!=PROFILE_DIMS or man.get('head')!='output.weight' or man.get('hadamard_block')!=1: raise ValueError('Unsupported Q4 profile')
    for name,rec in man['files'].items():
        if Path(name).name!=name: raise ValueError('Invalid manifest path')
        path=pack/name
        if path.stat().st_size!=rec['size']: raise ValueError('Pack size mismatch: '+name)
        if full and digest(path)!=rec['sha256']: raise ValueError('Pack checksum mismatch: '+name)
    index=json.loads((pack/'index.json').read_text())['tensors']
    validate_shapes(index)
    for rec in index.values():
        if rec['type'] in BLOCK:
            expected=rec['rows']*(rec['ne0']//256)*BLOCK[rec['type']]
            if rec['rows']%8 or rec['ne0']%256 or expected!=rec['bytes']:
                raise ValueError('Invalid packed tensor dimensions')
            if rec['offset']<0 or rec['offset']%PAGE or rec['padded_bytes']!=(expected+PAGE-1)//PAGE*PAGE or rec['offset']+rec['padded_bytes']>man['files']['weights.bin']['size']:
                raise ValueError('Invalid packed tensor span')
        elif rec['type']==0:
            if rec['f32_off']<0 or rec['f32_off']%4 or rec['f32_off']+rec['rows']*rec['ne0']*4>man['files']['f32.bin']['size']:
                raise ValueError('Invalid float tensor span')
        else: raise ValueError('Invalid tensor type')
    return man

def convert(source,out,log=print):
    source=Path(source).resolve(); out=Path(out).resolve()
    if out.exists(): raise FileExistsError('Use a new output directory: '+str(out))
    g=GGUF(source); meta=g.kv
    keys=['block_count','embedding_length','feed_forward_length','attention.head_count',
          'attention.head_count_kv','ssm.group_count','ssm.state_size','ssm.inner_size']
    if meta.get('general.architecture')!='qwen35' or tuple(meta.get('qwen35.'+k) for k in keys)!=(32,4096,12288,16,4,16,128,4096):
        raise ValueError('First validated profile is dense Qwen3.5/OxCoder 9B only')
    if any(k.startswith('prism.hadamard.') for k in meta): raise ValueError('Folded ternary checkpoint is not a Q4 source')
    if set(t[2] for t in g.tensors)-{0,12,14}: raise ValueError('Only F32, Q4_K and Q6_K supported')
    if len(meta['tokenizer.ggml.tokens'])!=248320: raise ValueError('Unexpected vocabulary')
    if meta.get('qwen35.ssm.conv_kernel')!=4 or meta.get('qwen35.rope.dimension_count')!=64:
        raise ValueError('Unsupported convolution/RoPE')
    if meta.get('qwen35.rope.freq_base')!=1e7 or meta.get('qwen35.attention.key_length')!=256 or meta.get('qwen35.attention.value_length')!=256:
        raise ValueError('Unsupported rotary frequency/attention dimensions')
    if abs(meta.get('qwen35.attention.layer_norm_rms_epsilon',0)-1e-6)>1e-10:raise ValueError('Unsupported norm epsilon')
    if meta.get('qwen35.attention.recurrent_layers')!=[i%4!=3 for i in range(32)]:
        raise ValueError('Unsupported layer schedule')
    for name,ne,ty,off in g.tensors:
        length=math.prod(ne)*4 if ty==0 else math.prod(ne)//256*BLOCK[ty]
        if min(ne)<=0 or off<0 or g.data_start+off+length>source.stat().st_size: raise ValueError('Truncated tensor')
        if ty in BLOCK and (ne[0]%256 or math.prod(ne[1:])%8): raise ValueError('Unsupported tile dimensions')
    validate_shapes({name:dict(rows=math.prod(ne[1:]),ne0=ne[0],type=0 if name.endswith(('ssm_alpha.weight','ssm_beta.weight')) else ty)
                     for name,ne,ty,off in g.tensors})
    out.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(out.parent).free < source.stat().st_size+1_000_000_000: raise OSError('Insufficient free disk')
    temp=Path(tempfile.mkdtemp(prefix=out.name+'.partial-',dir=out.parent))
    try:
        index={}; samples=[]
        # Store layer tensors together; output and embedding remain separate resident candidates.
        ordered=sorted(g.tensors,key=lambda t:(int(t[0].split('.')[1])+1 if t[0].startswith('blk.') else 0,t[0]))
        with source.open('rb') as src,(temp/'weights.bin').open('wb') as weights,(temp/'f32.bin').open('wb') as floats:
            for name,ne,ty,off in ordered:
                cols=ne[0]; rows=math.prod(ne[1:]); src.seek(g.data_start+off)
                rec=dict(rows=rows,ne0=cols,dims=list(reversed(ne)),type=ty,folded=False)
                small=name.endswith(('ssm_alpha.weight','ssm_beta.weight'))
                if ty==0 or small:
                    data=src.read(rows*cols*4 if ty==0 else rows*(cols//256)*BLOCK[ty])
                    if small: data=decode_blocks(np.frombuffer(data,np.uint8),ty).astype('<f4').tobytes()
                    rec.update(type=0,f32_off=floats.tell());floats.write(data)
                else:
                    rec['offset']=weights.tell(); nbytes=0; blocks=cols//256; bs=BLOCK[ty]
                    for start in range(0,rows,128):
                        n=min(128,rows-start); raw=src.read(n*blocks*bs)
                        if len(raw)!=n*blocks*bs: raise ValueError('Short tensor read')
                        tiled=np.frombuffer(raw,np.uint8).reshape(n//8,8,blocks,bs).transpose(0,2,1,3).copy()
                        if not np.array_equal(tiled.transpose(0,2,1,3).reshape(-1),np.frombuffer(raw,np.uint8)):
                            raise ValueError('Tile transformation changed weight bytes')
                        weights.write(tiled.tobytes()); nbytes+=len(raw)
                    rec['bytes']=nbytes; rec['padded_bytes']=(nbytes+PAGE-1)//PAGE*PAGE
                    weights.write(bytes(rec['padded_bytes']-nbytes))
                index[name]=rec
                if len(index)%40==0: log('Q4 conversion: %d/%d tensors'%(len(index),len(ordered)))
        (temp/'index.json').write_text(json.dumps({'tensors':index},sort_keys=True))
        (temp/'vocab.json').write_text(json.dumps(meta['tokenizer.ggml.tokens']))
        (temp/'merges.txt').write_text('\n'.join(meta['tokenizer.ggml.merges'])+'\n')
        (temp/'source-metadata.json').write_text(json.dumps(meta))
        dims=dict(layers=32,width=4096,ffn=12288,heads=16,kv_heads=4,head_dim=256,rope_dim=64,
                  groups=16,state_size=128,gdn_inner=4096,rope_base=1e7,full_attention_interval=4)
        man=dict(pack_format='kindi-q4-v1',model=meta['general.name'],head='output.weight',
                 hadamard_block=1,dimensions=dims,source_file=source.name,source_sha256=digest(source),
                 quantization='mixed Q4_K/Q6_K; original blocks preserved',layout='8-row-block-interleave',
                 kindi_base='50fb5e6d67d5b9fd643e5a76ceda4f3d8ac19b5a',
                 files={p.name:dict(size=p.stat().st_size,sha256=digest(p)) for p in temp.iterdir()})
        (temp/'kindi_pack.json').write_text(json.dumps(man,indent=2)+'\n')
        verify_pack(temp,full=False)
        os.rename(temp,out)
        log('Q4 pack complete: '+str(out))
    except BaseException:
        shutil.rmtree(temp)
        raise
