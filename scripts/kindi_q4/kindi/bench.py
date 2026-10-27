"""kindi bench: generation speed at several context positions, prompt-reading speed, and GPU memory."""
import os
import time


def run(cfg, log=print):
    import cupy as cp
    import numpy as np
    from .engine import make_engine
    pack = cfg["pack"]
    cl = os.path.join(pack, "head_clusters.npz")
    eng = make_engine(pack, ctx=cfg["ctx"], clusters=dict(np.load(cl)) if os.path.exists(cl) else None, log=log)
    free, total = cp.cuda.runtime.memGetInfo()
    log("GPU memory: %.0f MiB free of %.0f with a %d-token context" % (free / 2 ** 20, total / 2 ** 20, eng.ctx))
    eng.reset()

    def timed(fn, n):
        fn(0)                                  # warm-up (first launches compile/load kernels)
        cp.cuda.Device().synchronize()
        t0 = time.time()
        for i in range(n):
            fn(i)
        cp.cuda.Device().synchronize()
        return (time.time() - t0) / n * 1000

    ms = timed(lambda i: eng.feed(1000 + i, 10 + i), 20)
    log("prompt reading:       %5.1f tokens/s" % (1000 / ms))
    modes = [("exact", 0)] + ([("fast", cfg["fast_clusters"])] if eng.hsl is not None else [])
    for name, M in modes:
        eng.shortlist = M
        for p in [x for x in (100, 2000, 8000, 16000, 24000) if x + 20 < eng.ctx]:
            ms = timed(lambda i: eng.step(2000 + i, p + i), 10)
            log("generation (%s head) at position %5d: %5.1f tokens/s" % (name, p, 1000 / ms))
    eng.shortlist = 0
    return 0
