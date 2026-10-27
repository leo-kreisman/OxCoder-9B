"""Native Kindi CUDA Q4/K6 path with a byte-bounded SSD tensor cache.

SPDX-License-Identifier: AGPL-3.0-only. Reuses Kindi's hybrid forward kernels;
replaces its ternary GEMV, embedding, Hadamard and grouped output paths.
"""
import json,os,time
from pathlib import Path
from collections import OrderedDict
import numpy as np
import cupy as cp
from .engine import Engine as Original, Kernels, VOCAB
from .q4pack import verify_pack, verification_full

def rss():
    return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')

class Weights:
    def __init__(self,pack,index):
        self.fd=os.open(str(Path(pack)/'weights.bin'),os.O_RDONLY)
        self.index=index;self.cache=OrderedDict();self.used=0;self.read_bytes=0;self.hits=0;self.misses=0
        self.limit=int(os.environ.get('KINDI_Q4_GPU_CACHE_BYTES','5800000000'))
        if not 900_000_000<=self.limit<=5_800_000_000:raise ValueError('GPU tensor cache must be 0.9..5.8 GB')
        self.disk_bypass=os.environ.get('KINDI_Q4_STREAM','0')=='1'
        self.pin_boundaries=os.environ.get('KINDI_Q4_PIN_BOUNDARIES','1')=='1' and self.limit>=1_600_000_000
    def __getitem__(self,name):
        if name in self.cache:
            self.hits+=1;self.cache.move_to_end(name);return self.cache[name]
        self.misses+=1;rec=self.index[name];size=rec['bytes']
        cp.cuda.get_current_stream().synchronize()
        while self.cache and self.used+size>self.limit:
            protected={'token_embd.weight','output.weight'} if self.pin_boundaries else set()
            key=next((key for key in self.cache if key not in protected),next(iter(self.cache)))
            self.used-=self.cache[key][0].nbytes;del self.cache[key]
        cp.get_default_memory_pool().free_all_blocks()
        if cp.get_default_memory_pool().total_bytes()+rss()+size+64_000_000>7_500_000_000:
            raise MemoryError('Q4 GPU + host admission budget exceeded')
        a=cp.empty(size,cp.uint8)
        for off in range(0,size,8<<20):
            n=min(8<<20,size-off);data=os.pread(self.fd,n,rec['offset']+off)
            if len(data)!=n:raise IOError('Truncated Q4 weight tensor')
            a[off:off+n].set(np.frombuffer(data,np.uint8))
            self.read_bytes+=n
        cp.cuda.get_current_stream().synchronize()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        # Avoid counting GPU residency while hiding a second full copy in host file cache.
        os.posix_fadvise(self.fd,rec['offset'],rec['padded_bytes'],os.POSIX_FADV_DONTNEED)
        value=(a,np.int32(rec['rows']),np.int32(rec['ne0']),np.int32(rec['type']))
        self.cache[name]=value;self.used+=size;return value

class Engine(Original):
    def __init__(self,pack,ctx=4096,clusters=None,log=print):
        if clusters is not None:raise ValueError('No approximate Q4 head')
        if not 32<=ctx<=32768:raise ValueError('Q4 experimental context range: 32..32768')
        t=time.monotonic();self.pack=Path(pack);self.ctx=ctx;self.log=log;self._f32={}
        man=verify_pack(pack,full=verification_full())
        self.dims=d=man['dimensions'];self.head_name=man['head'];self.HB=1
        self.D=d['width'];self.NL=d['layers'];self.NFF=d['ffn'];self.NH=d['heads'];self.NHKV=d['kv_heads']
        self.HD=d['head_dim'];self.NK=d['groups'];self.HK=d['state_size'];self.NV=d['gdn_inner']//self.HK
        self.CONV_CH=2*self.NK*self.HK+self.NV*self.HK;self.NATT=self.NL//d['full_attention_interval']
        self.idx=json.loads((self.pack/'index.json').read_text());self.w=Weights(pack,self.idx['tensors'])
        self.k=Kernels()
        module=cp.RawModule(code=(Path(__file__).parent/'q4.cu').read_text(),options=('--fmad=false',))
        self.q4_gemv=module.get_function('q4_gemv');self.q4_embed=module.get_function('q4_embed')
        self.q4_rms=module.get_function('q4_rms')
        self.k.k_smfold=module.get_function('q4_silu');self.k.k_fold_fwht=module.get_function('q4_copy')
        self.fl=cp.asarray(np.fromfile(self.pack/'f32.bin',dtype=np.float32))
        self.sD=self.sV=self.sA=self.sF=cp.ones(1,cp.float32) # ABI placeholder, unused by non-folded kernels
        self.hsl=None;self.shortlist=0;self.nsm=1;self.chunk_ok=False;self.pending=0
        self.vocab=json.loads((self.pack/'vocab.json').read_text())
        z=lambda n:cp.zeros(n,cp.float32)
        D=self.D;W=max(D,self.NFF,self.NV*self.HK)
        self.y=z(VOCAB);self.xp=z(W);self.h=z(D);self.h2=z(D);self.xn=z(D);self.part=z(W)
        self.zv=z(self.NV*self.HK);self.qkv=z(self.CONV_CH);self.cs=z(self.CONV_CH)
        self.qkbuf=z(2*self.NK*self.HK);self.qkvbuf=z(self.NHKV*self.HD)
        self.kn=z(self.NHKV*self.HD);self.qn=z(self.NH*2*self.HD);self.qr=z(self.NH*self.HD);self.kro=z(self.NHKV*self.HD)
        self.fo=z(self.NH*self.HD);self.fo6=z(self.NFF);self.outgdn=z(self.NV*self.HK)
        self.att=z(self.NH*self.HD);self.ffg=z(self.NFF);self.ffu=z(self.NFF)
        self.gb=z(self.NV);self.gg=z(self.NV);self.gout=z(self.NV*self.HK)
        self.sbuf=z(self.NH*ctx);self.apart=z(((ctx+255)//256)*self.NH*self.HD);self.aout=cp.zeros(1,cp.int32)
        inv=np.float64(d['rope_base'])**(-2*np.arange(d['rope_dim']//2)/d['rope_dim'])
        theta=np.arange(ctx)[:,None]*inv[None,:]
        self.cos_t=cp.asarray(np.cos(theta).astype(np.float32));self.sin_t=cp.asarray(np.sin(theta).astype(np.float32))
        self.M=[z((self.NV,self.HK,self.HK) if (i+1)%4 else (1,)) for i in range(self.NL)]
        self.conv=[z(3*self.CONV_CH) for _ in range(self.NL)]
        self.Kc=[cp.zeros((ctx,self.NHKV,self.HD),cp.float16) for _ in range(self.NATT)]
        self.Vc=[cp.zeros_like(a) for a in self.Kc]
        self.ylog=self.y;self.ylog_n=VOCAB;self.ylog_map=None
        cp.get_default_pinned_memory_pool().free_all_blocks()
        self.peak_combined=0;self._memory_check()
        log('Kindi Q4 CUDA ready; %.2fs; tensor cache %.3f GB'%(time.monotonic()-t,self.w.limit/1e9))
    def _memory_check(self):
        used=rss()+cp.get_default_memory_pool().total_bytes()
        self.peak_combined=max(self.peak_combined,used)
        if used>7_500_000_000:raise MemoryError('Q4 host RSS + CUDA pool exceeded 7.5GB')
    def gemv(self,names,x,outs,parts_ok=False):
        for name,y in zip(names,outs):
            w,R,C,ty=self.w[name]
            self.q4_gemv(((int(R)+3)//4,),(128,),(w,x,y,R,C,ty))
        return 1
    def embed(self,tok):
        if not 0<=tok<VOCAB:raise ValueError('Invalid token')
        self._memory_check();w,R,C,ty=self.w['token_embd.weight']
        self.q4_embed(((int(C)+255)//256,),(256,),(w,self.h,np.int32(tok),C,ty));return self.h
    def rmsfold(self,x,wname):
        self.q4_rms((1,),(256,),(self.h,self.y,self.f32(wname),self.xn,self.xp,np.int32(self.D),np.int32(self.pending)))
        self.pending=0;return self.xn
    def fold_args(self,width):return (width//512,),np.int32(512)
    def fold_hi(self,buf,width):pass
    def feed_batch(self,toks,pos):
        for i,tok in enumerate(toks):self.feed(int(tok),pos+i)
