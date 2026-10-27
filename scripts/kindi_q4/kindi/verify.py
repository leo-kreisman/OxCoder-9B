"""kindi verify: greedy output vs llama.cpp's reference for the same model file (4 prompts, the first 40 tokens
llama.cpp generated from them at temperature 0). Every token must match. reference.json is the 2B; reference-4b.json,
-9b.json and -27b.json the others (the 27B with llama.cpp's q8_0 K/V cache, as Kindi's 27B engine)."""
import json
import os
import re
import time


def reference_for(pack):
    """the reference file of the pack's model (by the size in its source file name), or None"""
    here = os.path.dirname(os.path.abspath(__file__))
    man = json.load(open(os.path.join(pack, "kindi_pack.json")))
    if man.get("pack_format") == "kindi-q4-v1":
        p = os.path.join(here, "reference-oxcoder-q4.json")
        if not os.path.isfile(p): return None
        ref = json.load(open(p))
        return p if ref.get("source_sha256") == man.get("source_sha256") else None
    m = re.search(r"-(\d+)B[-_.]", man.get("source_file", "") + ".") or re.search(r"(\d+)B", man.get("model", ""))
    size = (m.group(1) + "b") if m else "2b"
    p = os.path.join(here, "reference.json" if size == "2b" else "reference-%s.json" % size)
    return p if os.path.exists(p) else None


def run(cfg, log=print):
    try:
        import cupy as cp
    except ImportError:
        cp = None
    from .engine import make_engine
    import numpy as np
    pack = cfg["pack"]
    rp = reference_for(pack)
    if rp is None:
        log("no llama.cpp reference for this model in this version of Kindi")
        return 2
    ref = json.load(open(rp))
    cl = os.path.join(pack, "head_clusters.npz")
    eng = make_engine(pack, ctx=1024, clusters=dict(np.load(cl)) if os.path.exists(cl) else None, log=log)
    ok_all, tok, dt = True, 0, 0.0
    for i, r in enumerate(ref["prompts"]):
        sync = (lambda: cp.cuda.Device().synchronize()) if cp is not None and hasattr(eng, "k") else (lambda: None)
        sync()
        t0 = time.time()
        got = eng.generate(r["ids"], len(r["expected"]))
        sync()
        t = time.time() - t0
        n = sum(1 for a, b in zip(got, r["expected"]) if a == b)
        ok = got == r["expected"]
        ok_all &= ok
        tok += len(r["ids"]) + len(got)
        dt += t
        log("prompt %d %-60s %s  %d/%d tokens match  (%.1fs)" % (i + 1, repr(r["prompt"])[:60], "PASS" if ok else "FAIL",
                                                                  n, len(r["expected"]), t))
        if not ok:
            j = next(k for k, (a, b) in enumerate(zip(got, r["expected"])) if a != b)
            log("   first difference at token %d" % j)
    log("%s - %.1f tokens/s over the run" % ("ALL PASS: token-exact with llama.cpp" if ok_all else "FAIL", tok / dt))
    return 0 if ok_all else 1
