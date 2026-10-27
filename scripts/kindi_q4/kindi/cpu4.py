"""Native Kindi Q4_K/Q6_K CPU extension for the dense OxCoder/Qwen3.5 9B profile.

Retains the upstream hybrid GDN/attention forward pass and adds exact weight
block decoding, lossless row-tiled storage and optional SSD layer release.
No llama.cpp or MLX inference dependency. Floating-point arithmetic may differ
from other backends; run checkpoint-specific reference validation.
SPDX-License-Identifier: AGPL-3.0-only.
"""
import ctypes
import json
import os
import time
from pathlib import Path

import numpy as np

VOCAB = 248320
DEFAULT_DIMS = {"layers": 24, "width": 2048, "ffn": 6144, "heads": 8, "kv_heads": 2,
                "groups": 16, "head_dim": 256, "rope_dim": 64, "state_size": 128, "gdn_inner": 2048,
                "rope_base": 1e7, "full_attention_interval": 4}
HERE = os.path.dirname(os.path.abspath(__file__))
F32 = np.float32
PF = ctypes.POINTER(ctypes.c_float)
PU8 = ctypes.POINTER(ctypes.c_uint8)
PU16 = ctypes.POINTER(ctypes.c_uint16)


class Mat(ctypes.Structure):
    _fields_ = [("W", PU8), ("S", PF), ("R", ctypes.c_int), ("C", ctypes.c_int), ("type", ctypes.c_int), ("offset", ctypes.c_uint64), ("bytes", ctypes.c_uint64)]


class Layer(ctypes.Structure):
    _fields_ = [("is_attn", ctypes.c_int), ("attn_norm", PF), ("post_norm", PF),
                ("qkv", Mat), ("gate", Mat), ("ssm_out", Mat),
                ("beta_w", PF), ("alpha_w", PF), ("dt_b", PF), ("ssm_a", PF), ("conv_w", PF), ("ssm_norm", PF),
                ("M", PF), ("conv", PF),
                ("q", Mat), ("k", Mat), ("v", Mat), ("o", Mat), ("q_norm", PF), ("k_norm", PF),
                ("Kc", PU16), ("Vc", PU16),
                ("ffg", Mat), ("ffu", Mat), ("ffd", Mat)]


_SCR = ["h", "xn", "xp", "y", "qkv", "zv", "cs", "qk", "gout", "outn", "m", "qn", "kr", "vv", "qr", "kro", "sc",
        "att", "ffg", "ffu", "T", "logits", "gb", "gg"]


class Model(ctypes.Structure):
    _fields_ = ([(n, ctypes.c_int) for n in ("D", "NL", "NFF", "NH", "NHKV", "HD", "NK", "HK", "NV", "CONV_CH", "HB",
                                             "rope_half", "ctx", "vocab")]
                + [("L", ctypes.POINTER(Layer)), ("emb", Mat), ("head", Mat)]
                + [(n, PF) for n in ("out_norm", "sD", "sV", "sA", "sF", "cos_t", "sin_t")]
                + [(n, PF) for n in _SCR] + [("red", ctypes.POINTER(ctypes.c_double))])


def _lib():
    import hashlib, platform, subprocess
    metal=os.environ.get("KINDI_DEVICE")=="metal"
    if metal and platform.system()!="Darwin": raise RuntimeError("Metal requires macOS")
    sources = [os.path.join(HERE,n) for n in ("kcpu4.c","q4kernels.h")]
    if metal: sources += [os.path.join(HERE,n) for n in ("metal4.mm","q4.metal")]
    key=hashlib.sha256(b"".join(Path(p).read_bytes() for p in sources)+platform.machine().encode()).hexdigest()[:16]
    cache=os.environ.get("KINDI_CPU_CACHE",os.path.expanduser("~/.cache/kindi-q4"))
    os.makedirs(cache,exist_ok=True)
    so=os.path.join(cache,"libkcpu4-"+key+".so")
    if not os.path.exists(so):
        flags=["-O3","-ffp-contract=off","-fno-math-errno","-shared","-fPIC"]
        if platform.system()=="Linux": flags += ["-march=native","-fopenmp"]
        # Apple's Clang has no bundled OpenMP: the portable path uses one thread.
        if metal:
            subprocess.run(["clang","-O3","-ffp-contract=off","-fPIC","-DKINDI_METAL","-c",sources[0],"-o",so+".o"],check=True)
            subprocess.run(["clang++","-O3","-std=c++17","-fobjc-arc","-dynamiclib",so+".o",sources[2],"-framework","Foundation","-framework","Metal","-o",so+".tmp"],check=True)
            os.unlink(so+".o")
        else:
            subprocess.run([os.environ.get("CC","cc"),*flags,sources[0],"-o",so+".tmp","-lm"],check=True)
        os.replace(so+".tmp",so)
    lib=ctypes.CDLL(so)
    if metal:
        lib.kq_metal_init.argtypes=[ctypes.c_char_p]
        lib.kq_metal_error.restype=ctypes.c_char_p
        lib.kq_metal_allocated.restype=ctypes.c_uint64
        if lib.kq_metal_init(Path(sources[3]).read_bytes()): raise RuntimeError(lib.kq_metal_error().decode())
    lib.kc_step.argtypes=[ctypes.POINTER(Model),ctypes.c_int,ctypes.c_int,ctypes.c_int]
    lib.kc_step.restype=ctypes.c_int
    lib.kc_matvec.argtypes=[ctypes.POINTER(Mat),PF,PF]
    lib.kc_dequant.argtypes=[ctypes.POINTER(Mat),ctypes.c_int,PF]
    lib.kc_storage.argtypes=[ctypes.c_int,ctypes.c_int]
    assert lib.kc_sizeof_model()==ctypes.sizeof(Model) and lib.kc_sizeof_layer()==ctypes.sizeof(Layer)
    return lib


def _p(a, t=PF):
    return a.ctypes.data_as(t)


class Engine(object):
    def __init__(self, pack, ctx=4096, clusters=None, log=print, threads=None):
        t0 = time.time()
        if threads:
            os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        self.lib = _lib()
        self.ctx = int(ctx)
        man = json.load(open(os.path.join(pack, "kindi_pack.json")))
        d = self.dims = dict(DEFAULT_DIMS, **man.get("dimensions", {}))
        self.HB = int(man.get("hadamard_block", 512))
        self.head_name = man.get("head", "token_embd.weight")
        D, NL, NFF = d["width"], d["layers"], d["ffn"]
        NH, NHKV, HD = d["heads"], d["kv_heads"], d["head_dim"]
        NK, HK = d["groups"], d["state_size"]
        NV = d["gdn_inner"] // HK
        CC = 2 * NK * HK + NV * HK
        interval = d["full_attention_interval"]
        idx = json.load(open(os.path.join(pack, "index.json")))["tensors"]
        if man.get("pack_format") != "kindi-q4-v1": raise ValueError("Not a Kindi Q4 pack")
        if ctx < 32 or ctx > 32768: raise ValueError("Q4 validation supports context 32..32768")
        self.keep=[]
        def keep(a):
            self.keep.append(a); return a
        from .q4pack import verify_pack, verification_full
        verify_pack(pack, full=verification_full())
        self.weights=np.memmap(os.path.join(pack,"weights.bin"),dtype=np.uint8,mode="r")
        self.weight_fd=os.open(os.path.join(pack,"weights.bin"),os.O_RDONLY)
        stream=int(os.environ.get("KINDI_Q4_STREAM","0"))
        if stream not in (0,1): raise ValueError("KINDI_Q4_STREAM must be 0 or 1")
        if os.environ.get("KINDI_DEVICE")=="metal" and stream: raise ValueError("Metal SSD eviction is not validated; use STREAM=0")
        self.lib.kc_storage(self.weight_fd,stream)
        mats={}
        for name,rec in idx.items():
            if rec["type"] in (12,14):
                mats[name]=Mat(_p(self.weights[rec["offset"]:],PU8),None,rec["rows"],rec["ne0"],
                               rec["type"],rec["offset"],rec["padded_bytes"])
        fl=keep(np.fromfile(os.path.join(pack,"f32.bin"),dtype=F32))

        def f(name):
            rec = idx[name]
            off = rec["f32_off"] // 4
            return _p(fl[off:off + rec["rows"] * rec["ne0"]])
        sg = lambda w: keep(np.ones(w,dtype=F32))  # noqa: E731
        z = lambda n, t=F32: keep(np.zeros(n, dtype=t))  # noqa: E731
        Ls = (Layer * NL)()
        self.M, self.conv, self.Kc, self.Vc = [], [], [], []
        for il in range(NL):
            p = "blk.%d." % il
            L = Ls[il]
            L.is_attn = int((il + 1) % interval == 0)
            L.attn_norm, L.post_norm = f(p + "attn_norm.weight"), f(p + "post_attention_norm.weight")
            L.ffg, L.ffu, L.ffd = mats[p + "ffn_gate.weight"], mats[p + "ffn_up.weight"], mats[p + "ffn_down.weight"]
            if L.is_attn:
                L.q, L.k, L.v, L.o = (mats[p + "attn_q.weight"], mats[p + "attn_k.weight"], mats[p + "attn_v.weight"],
                                      mats[p + "attn_output.weight"])
                L.q_norm, L.k_norm = f(p + "attn_q_norm.weight"), f(p + "attn_k_norm.weight")
                Kc, Vc = z(self.ctx * NHKV * HD, np.uint16), z(self.ctx * NHKV * HD, np.uint16)
                self.Kc.append(Kc.reshape(self.ctx, NHKV, HD))
                self.Vc.append(Vc.reshape(self.ctx, NHKV, HD))
                L.Kc, L.Vc = _p(Kc, PU16), _p(Vc, PU16)
            else:
                L.qkv, L.gate, L.ssm_out = mats[p + "attn_qkv.weight"], mats[p + "attn_gate.weight"], mats[p + "ssm_out.weight"]
                L.beta_w, L.alpha_w = f(p + "ssm_beta.weight"), f(p + "ssm_alpha.weight")
                L.dt_b, L.ssm_a, L.conv_w, L.ssm_norm = (f(p + "ssm_dt.bias"), f(p + "ssm_a"), f(p + "ssm_conv1d.weight"),
                                                         f(p + "ssm_norm.weight"))
                Mst, cv = z(NV * HK * HK), z(3 * CC)
                self.M.append(Mst)
                self.conv.append(cv)
                L.M, L.conv = _p(Mst), _p(cv)
        self.Ls = Ls
        m = self.m = Model()
        for k, v in (("D", D), ("NL", NL), ("NFF", NFF), ("NH", NH), ("NHKV", NHKV), ("HD", HD), ("NK", NK),
                     ("HK", HK), ("NV", NV), ("CONV_CH", CC), ("HB", self.HB), ("rope_half", d["rope_dim"] // 2),
                     ("ctx", self.ctx), ("vocab", VOCAB)):
            setattr(m, k, v)
        m.L = ctypes.cast(Ls, ctypes.POINTER(Layer))
        m.emb, m.head = mats["token_embd.weight"], mats[self.head_name]
        m.out_norm = f("output_norm.weight")
        m.sD, m.sV, m.sA, m.sF = (_p(sg(D)), _p(sg(NV * HK)), _p(sg(NH * HD)), _p(sg(NFF)))
        rh = d["rope_dim"] // 2
        inv = np.float64(d["rope_base"]) ** (-2.0 * np.arange(rh) / d["rope_dim"])
        th = np.arange(self.ctx, dtype=np.float64)[:, None] * inv[None, :]
        m.cos_t, m.sin_t = _p(keep(np.cos(th).astype(F32))), _p(keep(np.sin(th).astype(F32)))
        W = max(D, NFF, NV * HK, NH * HD)
        sizes = {"h": D, "xn": D, "xp": W, "y": D, "qkv": CC, "zv": NV * HK, "cs": CC, "qk": 2 * NK * HK,
                 "gout": NV * HK, "outn": NV * HK, "m": W, "qn": NH * 2 * HD, "kr": NHKV * HD, "vv": NHKV * HD,
                 "qr": NH * HD, "kro": NHKV * HD, "sc": NH * self.ctx, "att": NH * HD, "ffg": NFF, "ffu": NFF,
                 "T": W // 2 * 8, "logits": max(VOCAB, mats[self.head_name].R), "gb": NV, "gg": NV}
        for k in _SCR:
            a = z(sizes[k])
            setattr(m, k, _p(a))
            if k == "logits":
                self.logits = a
        m.red = _p(z(256, np.float64), ctypes.POINTER(ctypes.c_double))
        self.ylog, self.ylog_n, self.ylog_map = self.logits, VOCAB, None
        self.hsl, self.shortlist = None, 0
        self.shortlist = 0
        self.vocab = json.load(open(os.path.join(pack, "vocab.json")))
        (log or (lambda *a: None))("kindi-q4-cpu ready: %s, %s threads, ctx %d (%.1fs)" % (
            man.get("model"), os.environ.get("OMP_NUM_THREADS", os.cpu_count()), self.ctx, time.time() - t0))

    def _memory_check(self):
        import resource, sys
        # RSS is a sampled guard; Linux runs additionally require an external cgroup.
        if sys.platform.startswith("linux"):
            pages=int(open("/proc/self/statm").read().split()[1])
            rss=pages*os.sysconf("SC_PAGE_SIZE")
        else:
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss>7_500_000_000: raise MemoryError("Q4 inference exceeded 7.5GB RSS stop threshold")

    def reset(self):
        for a in self.M + self.conv + self.Kc + self.Vc:
            a.fill(0)

    def feed(self, tok, pos):
        if tok < 0 or tok >= VOCAB or pos < 0: raise ValueError("Invalid token or position")
        self._memory_check()
        if pos >= self.ctx:
            raise OverflowError("position %d is past the context (%d)" % (pos, self.ctx))
        if self.lib.kc_step(ctypes.byref(self.m), int(tok), int(pos), 0)==-2: raise RuntimeError(self.lib.kq_metal_error().decode())

    def step(self, tok, pos):
        if tok < 0 or tok >= VOCAB or pos < 0: raise ValueError("Invalid token or position")
        self._memory_check()
        if pos >= self.ctx:
            raise OverflowError("position %d is past the context (%d)" % (pos, self.ctx))
        result=self.lib.kc_step(ctypes.byref(self.m), int(tok), int(pos), 1)
        if result==-2: raise RuntimeError(self.lib.kq_metal_error().decode())
        return result

    def generate(self, ids, n):
        self.reset()
        for i, t in enumerate(ids[:-1]):
            self.feed(t, i)
        pos = len(ids)
        out = [self.step(ids[-1], pos - 1)]
        while len(out) < n:
            out.append(self.step(out[-1], pos))
            pos += 1
        return out
