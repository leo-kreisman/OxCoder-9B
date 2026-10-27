"""Kindi OpenAI-compatible server.

  GET  /health                 status, live progress of the request being served, GPU memory
  GET  /v1/models              the model
  POST /v1/chat/completions    messages (+ tools) -> answer; stream=true for SSE; tool calls as OpenAI tool_calls

One request at a time (one engine on one GPU); concurrent requests queue. Thinking is off unless the request sets
chat_template_kwargs.enable_thinking (or top-level enable_thinking); a stream then carries the thinking as
delta.reasoning_content. thinking_budget_tokens (or thinking_budget) caps it: at the budget the server closes the
thinking itself and the model goes on to answer. temperature 0 = greedy (token-exact with llama.cpp in the exact head
mode)."""
import asyncio
import codecs
import json
import os
import threading
import time
import uuid

import numpy as np

from . import COMMIT, __version__, chat
from .cache import Cancelled, Runner
from .engine import EOS
from .sampling import Sampler


THINK_CLOSE = "\n\nI have thought enough; time to act.\n</think>\n\n"


def _looping(gen, min_span=240):
    """True when the last tokens are one block of 4..256 tokens repeated (at least 4 times and min_span tokens):
    a model stuck in a loop. Short legitimate repeats (table rules, a few identical lines) stay below it."""
    n = len(gen)
    for P in range(4, 257):
        reps = max(4, -(-min_span // P))
        if P * reps > n:
            break
        tail = gen[n - P:]
        if all(gen[n - (k + 1) * P:n - k * P] == tail for k in range(1, reps)):
            return True
    return False


class Kindi(object):
    """Engine + tokenizer + cache + sampler, shared by the HTTP layer and `kindi chat`."""

    def __init__(self, cfg, log=print):
        try:
            import cupy as cp
        except ImportError:              # a CPU-only host (KINDI_DEVICE=cpu)
            cp = None
        from .engine import make_engine
        from .tokenizer import Tok
        self.cp, self.cfg, self.log = cp, cfg, log
        pack = cfg["pack"]
        if not os.path.exists(os.path.join(pack, "kindi_pack.json")):
            raise SystemExit("kindi: no model pack at %s - run: kindi convert <model.gguf>" % pack)
        man = json.load(open(os.path.join(pack, "kindi_pack.json")))
        self.pack_id = (man.get("source_sha256") or "pack")[:12]
        self.variant = chat.template_variant(pack)      # qwen35 (2B/4B/9B) or bonsai27 (the 27B template)
        clusters = None
        cl_path = os.path.join(pack, "head_clusters.npz")
        if os.path.exists(cl_path):
            clusters = dict(np.load(cl_path))
        self.eng = make_engine(pack, ctx=cfg["ctx"], clusters=clusters, log=log)
        self.tok = Tok(self.eng.vocab, os.path.join(pack, "merges.txt"))
        self.busy = {"active": False, "phase": "", "prompt_tokens": 0, "cached": 0, "read": 0, "generated": 0,
                     "started": 0.0}
        self.runner = Runner(self.eng, cfg, self.pack_id, log, progress=self.busy)
        self.sample = Sampler(self.eng)
        self.lock = threading.Lock()
        self.stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}
        self.t0 = time.time()

    def head_mode(self):
        """exact | fast: the state file (written by `kindi head`) wins over the config."""
        try:
            v = open(os.path.join(self.cfg["state_dir"], "head_mode")).read().strip().lower()
            if v in ("exact", "fast"):
                return v
        except OSError:
            pass
        return self.cfg["head"]

    def _pick(self, nxt, temperature, top_p, top_k, rng):
        """the next token: already drawn by speculative sampling, or sampled now"""
        ps = getattr(self.eng, "spec_presampled", None)
        if ps is not None:
            self.eng.spec_presampled = None
            return ps
        return self.sample(nxt, temperature, top_p, top_k, rng)

    def generate(self, req, cancel=None):
        """Yields ("text", piece) while generating, then ("done", info). cancel: a threading.Event set when the
        client disconnects (prompt reading and generation stop at the next token)."""
        cfg = self.cfg
        messages = req.get("messages") or []
        tools = req.get("tools") or None
        if req.get("tool_choice") == "none":
            tools = None
        think = bool((req.get("chat_template_kwargs") or {}).get("enable_thinking", req.get("enable_thinking", False)))
        budget = req.get("thinking_budget_tokens") or req.get("thinking_budget") or cfg.get("thinking_budget") or 0
        budget = int(budget) if think else 0
        g = lambda k, d: d if req.get(k) is None else req.get(k)  # noqa: E731
        temperature = float(g("temperature", cfg["temperature"]))
        top_p = float(g("top_p", cfg["top_p"]))
        top_k = int(g("top_k", cfg["top_k"]))
        max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or cfg["max_tokens"])
        repeat_stop = bool(cfg.get("repeat_stop"))
        stops = req.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        fast = self.head_mode() == "fast"
        if fast and self.eng.hsl is None:
            fast = False
        self.eng.shortlist = cfg["fast_clusters"] if fast else 0
        seed = req.get("seed")
        rng = np.random.RandomState(seed if seed is not None else int(time.time() * 1000) % 2 ** 31)
        kw = req.get("chat_template_kwargs") or {}
        effort = req.get("reasoning_effort") or kw.get("reasoning_effort") or cfg.get("reasoning_effort") or None
        body, gp = chat.render(messages, tools, think, variant=self.variant, reasoning_effort=effort,
                               preserve_thinking=kw.get("preserve_thinking"))
        Tb = self.tok.encode(body)
        T = Tb + self.tok.encode(gp)
        ctx = self.eng.ctx
        if len(T) + 1 >= ctx:
            raise OverflowError("the prompt has %d tokens; the context is %d" % (len(T), ctx))
        max_new = max(1, min(max_new, ctx - len(T) - 1))
        t0 = time.time()
        self.busy.update(active=True, phase="matching cache", prompt_tokens=len(T), cached=0, read=0, generated=0,
                         started=t0)
        self.log("request: %d prompt tokens (max %d new)" % (len(T), max_new))
        try:
            im_end = 248046                                   # <|im_end|> closes the first message
            opening_end = T.index(im_end) + 1 if im_end in T else len(T)
            nxt, cached = self.runner.prepare(T, len(Tb), opening_end, cancel)
        except Cancelled as e:
            self.busy["active"] = False
            self.log("stopped: %s" % e)
            return
        except BaseException:
            self.busy["active"] = False
            raise
        t1 = time.time()
        self.log("prompt: %d reused, %d read in %.1fs" % (cached, len(T) - cached, t1 - t0))
        self.busy["phase"] = "generating"
        if hasattr(self.eng, "spec_begin"):
            self.eng.spec_begin(temperature <= 0, temperature, top_p, top_k, rng)   # speculative decoding (KINDI_SPEC)
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        text, gen, finish = "", [], "length"
        # while thinking, pieces go out as ("reasoning", ...) - the last 8 characters are held back in case they are
        # the start of "</think>"; after it, as ("text", ...)
        st = {"think": think, "sent": 0, "forced": False, "lead": think}

        def route(piece):
            if not st["think"]:
                if st["lead"]:                                # the newlines that follow </think>
                    piece = piece.lstrip("\n")
                    st["lead"] = not piece
                return [("text", piece)] if piece else []
            out = []
            if "</think>" in text:
                end = text.index("</think>")
                if end > st["sent"]:
                    out.append(("reasoning", text[st["sent"]:end]))
                after = text[end + len("</think>"):].lstrip("\n")
                st["think"] = False
                if after:
                    st["lead"] = False
                    out.append(("text", after))
            elif len(text) - 8 > st["sent"]:
                out.append(("reasoning", text[st["sent"]:len(text) - 8]))
                st["sent"] = len(text) - 8
            return out

        tok = self.sample(nxt, temperature, top_p, top_k, rng)
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    self.log("stopped: the client went away after %d tokens" % len(gen))
                    return
                if tok in EOS:
                    finish = "stop"
                    break
                gen.append(tok)
                self.busy["generated"] = len(gen)
                piece = dec.decode(self.tok.decode_bytes(tok))
                text += piece
                hit = None if st["think"] else next((s for s in stops if s and s in text), None)
                if hit:
                    text = text[:text.index(hit)]
                    finish = "stop"
                    break
                for item in route(piece):
                    yield item
                if len(gen) >= max_new:
                    break
                loop = repeat_stop and len(gen) % 16 == 0 and _looping(gen)
                if loop and not st["think"]:
                    self.log("stopped a repeating loop after %d tokens" % len(gen))
                    finish = "length"
                    break
                nxt = self.runner.step(tok)
                if loop and st["think"] and not st["forced"]:
                    budget = len(gen)                         # a loop in the thinking: close it now, then answer
                if st["think"] and budget and len(gen) >= budget and not st["forced"]:
                    # thinking budget spent: close the thinking for the model, then let it answer
                    st["forced"] = True
                    self.log("thinking budget of %d tokens reached; closing the thinking" % budget)
                    for ft in self.tok.encode(THINK_CLOSE):
                        gen.append(ft)
                        piece = dec.decode(self.tok.decode_bytes(ft))
                        text += piece
                        for item in route(piece):
                            yield item
                        nxt = self.runner.step(ft)
                tok = self._pick(nxt, temperature, top_p, top_k, rng)
        finally:
            sp = self.eng.spec_end() if hasattr(self.eng, "spec_end") else None
            self.busy["active"] = False
            if hasattr(self.runner, "idle"):
                self.runner.idle()                            # (KINDI_CKPT_VRAM: checkpoint host copies now)
        if sp and sp["passes"]:
            self.log("speculative: %d tokens in %d passes, %d of %d drafted tokens accepted" % (
                len(gen), sp["passes"], sp["accepted"], sp["drafted"]))
        if st["think"] and len(text) > st["sent"]:
            yield "reasoning", text[st["sent"]:]
        t2 = time.time()
        self.log("done: %d tokens in %.1fs (%s)" % (len(gen), t2 - t1, finish))
        for k, v in (("requests", 1), ("prompt_tokens", len(T)), ("cached_tokens", cached),
                     ("completion_tokens", len(gen))):
            self.stats[k] += v
        reasoning = ""
        if "</think>" in text:
            reasoning, text = text.split("</think>", 1)
            reasoning = reasoning.replace("<think>", "").strip()
            text = text.lstrip("\n")
        content, calls = chat.parse_tool_calls(text, tools) if tools else (text.strip(), [])
        if calls:
            finish = "tool_calls"
        yield "done", {"content": content, "tool_calls": calls, "reasoning": reasoning, "finish": finish,
                       "usage": {"prompt_tokens": len(T), "completion_tokens": len(gen),
                                 "total_tokens": len(T) + len(gen), "prompt_tokens_details": {"cached_tokens": cached}},
                       "timing": {"prompt_s": round(t1 - t0, 2), "prompt_new_tokens": len(T) - cached,
                                  "gen_s": round(t2 - t1, 2),
                                  "gen_tok_s": round(len(gen) / (t2 - t1), 2) if t2 > t1 else 0.0}}


def create_app(cfg, log=print):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    K = Kindi(cfg, log)
    model_id = cfg["model_name"]
    app = FastAPI(title="Kindi", version=__version__)

    def _err(code, msg, typ="invalid_request_error"):
        return JSONResponse({"error": {"message": msg, "type": typ, "code": typ}}, status_code=code)

    def _auth(request):
        key = cfg["api_key"]
        return not key or request.headers.get("authorization", "") == "Bearer " + key

    @app.get("/health")
    def health():
        free = total = 0
        if K.cp is not None and os.environ.get("KINDI_DEVICE", "gpu") != "cpu":
            free, total = K.cp.cuda.runtime.memGetInfo()
        b = K.busy
        busy = None
        if b["active"] and K.lock.locked():
            busy = {"phase": b["phase"], "elapsed_s": round(time.time() - b["started"]),
                    "prompt_tokens": b["prompt_tokens"], "reused": b["cached"],
                    "prompt_read": "%d/%d" % (b["read"], b["prompt_tokens"] - b["cached"]), "generated": b["generated"]}
        return {"ok": True, "kindi": __version__, "commit": COMMIT, "model": model_id, "ctx": K.eng.ctx, "head": K.head_mode(),
                "uptime_s": round(time.time() - K.t0), "busy": busy, "stats": K.stats,
                "gpu_free_mib": round(free / 1048576.0), "gpu_total_mib": round(total / 1048576.0)}

    @app.get("/v1/models")
    def models(request: Request):
        if not _auth(request):
            return _err(401, "invalid API key", "authentication_error")
        return {"object": "list", "data": [{"id": model_id, "object": "model", "created": int(K.t0),
                                            "owned_by": "kindi", "context_length": K.eng.ctx}]}

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        if not _auth(request):
            return _err(401, "invalid API key", "authentication_error")
        try:
            req = await request.json()
        except Exception:  # noqa: BLE001
            return _err(400, "the body is not JSON")
        cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())

        def chunk(delta, finish=None, usage=None):
            d = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage is not None:
                d["usage"] = usage
            return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"

        cancel = threading.Event()

        if req.get("stream"):
            def sse():
                with K.lock:
                    yield chunk({"role": "assistant", "content": ""})
                    held, in_call = "", False
                    try:
                        for kind, val in K.generate(req, cancel):
                            if kind == "reasoning":
                                yield chunk({"reasoning_content": val})
                            elif kind == "text":
                                held += val
                                if in_call:
                                    continue                  # keep the call text until it parses (or not)
                                if req.get("tools") and "<tool_call>" in held:
                                    pre = held[:held.index("<tool_call>")]
                                    if pre:
                                        yield chunk({"content": pre})
                                    held, in_call = held[held.index("<tool_call>"):], True
                                    continue
                                # hold back a possible start of '<tool_call>' / '<think>' / '</think>'
                                cut = max([held.rfind("<")] + [0]) if "<" in held[-12:] else len(held)
                                out, held = held[:cut], held[cut:]
                                if out:
                                    yield chunk({"content": out})
                            else:
                                info = val
                                if info["tool_calls"]:
                                    yield chunk({"tool_calls": [dict(tc, index=i)
                                                                for i, tc in enumerate(info["tool_calls"])]})
                                elif held:
                                    # also an unfinished call (cut off by max_tokens): send its raw text, do not drop it
                                    yield chunk({"content": held})
                                inc = (req.get("stream_options") or {}).get("include_usage")
                                yield chunk({}, info["finish"], info["usage"] if inc else None)
                    except OverflowError as e:
                        yield "data: " + json.dumps({"error": {"message": str(e), "type": "context_length_exceeded"}}) + "\n\n"
                    except ValueError as e:
                        yield "data: " + json.dumps({"error": {"message": str(e), "type": "invalid_request_error"}}) + "\n\n"
                    yield "data: [DONE]\n\n"
            async def relay():
                # the generator runs in a worker thread; if the client disconnects, Starlette cancels this
                # coroutine, the finally sets `cancel`, and the engine stops at its next token
                loop = asyncio.get_event_loop()
                q = asyncio.Queue()
                END = object()

                def pump():
                    try:
                        for item in sse():
                            loop.call_soon_threadsafe(q.put_nowait, item)
                    finally:
                        loop.call_soon_threadsafe(q.put_nowait, END)
                fut = loop.run_in_executor(None, pump)
                try:
                    while True:
                        # while a long prompt is read, nothing else is sent: an SSE comment every 15 s keeps the
                        # stream alive (Node's fetch, which pi uses, drops a body that is silent for 300 s)
                        try:
                            item = await asyncio.wait_for(q.get(), timeout=15)
                        except asyncio.TimeoutError:
                            yield ": keep-alive\n\n"
                            continue
                        if item is END:
                            break
                        yield item
                finally:
                    cancel.set()
                    await asyncio.wait([fut], timeout=30)
            return StreamingResponse(relay(), media_type="text/event-stream")

        def run():
            with K.lock:
                info = None
                for kind, val in K.generate(req, cancel):
                    if kind == "done":
                        info = val
                return info
        try:
            fut = asyncio.get_event_loop().run_in_executor(None, run)
            while not fut.done():
                done, _ = await asyncio.wait([fut], timeout=1.0)
                if not done and await request.is_disconnected():
                    cancel.set()
            info = fut.result()
            if info is None:
                return _err(499, "request cancelled", "cancelled")
        except OverflowError as e:
            return _err(400, str(e), "context_length_exceeded")
        except ValueError as e:
            return _err(400, str(e))
        msg = {"role": "assistant", "content": info["content"] if (info["content"] or not info["tool_calls"]) else None}
        if info["tool_calls"]:
            msg["tool_calls"] = info["tool_calls"]
        if info["reasoning"]:
            msg["reasoning_content"] = info["reasoning"]
        return {"id": cid, "object": "chat.completion", "created": created, "model": model_id,
                "choices": [{"index": 0, "message": msg, "finish_reason": info["finish"]}],
                "usage": info["usage"], "timing": info["timing"]}

    return app


def serve(cfg, log=print):
    import uvicorn
    app = create_app(cfg, log)
    log("serving on http://%s:%d/v1 (model %s)" % (cfg["host"], cfg["port"], cfg["model_name"]))
    uvicorn.run(app, host=cfg["host"], port=int(cfg["port"]), log_level="warning")
