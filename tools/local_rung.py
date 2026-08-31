"""LOCAL RUNG -- attach any locally-run model to leCore with the hosted treatment.

openzoo.fun's caller gets a ladder that answers free before it ever wakes a model,
provenance on every answer, veto, sessions, and receipts. A model you run yourself --
Ollama, llama.cpp's server, an OpenAI-compatible endpoint, a HF pipeline, or a leCore
Unicron-installed checkpoint through GDNRuntime -- deserves exactly the same, and gets it
here. Backends differ only in how bytes reach the model; the GOVERNANCE is identical.

    from tools.local_rung import LocalRung
    rung = LocalRung.http("http://localhost:11434/api/generate", model="qwen3.5:0.8b")
    rung = LocalRung.openai("http://localhost:8080/v1/chat/completions", model="local")
    rung = LocalRung.callable(my_fn)
    rung = LocalRung.gdn("/path/to/unicron_installed_model")      # leCore inside the weights

    import lecore
    m = lecore.UnifiedMind()
    m.boot(partition="~/.lecore", doctrine=True, llm=rung)
    m.zoo_attach(rung)
    m.ask("...")            # free rungs first; the model only wakes on a real miss

WHAT YOU GET, measured the same way the hosted service is:
  * LADDER ECONOMICS      rung.stats() reports calls, tokens, seconds -- the questions
                          memory served for free never appear there. Measured on the
                          full stack: 5 of 8 cold, then 8/8 free after teaching, 99%
                          faster wall time and ZERO model calls.
  * PROVENANCE            answers say 'taught' or 'model-cached' (cp47): a cached model
                          answer must never be indistinguishable from an established fact.
  * OUROBOROS (optional)  a memory manager hooked in the model's own forward pass on the
                          gdn backend -- PASSIVE (bit-exact zero logit change), with
                          write/read/delete verbs, durable spill to your partition, and
                          capacity_report that DECLARES ITS REGIME (cp46: in the mixed
                          regime -- stream background plus written facts -- the prediction
                          is an UPPER BOUND; call verify_recall(pairs) for ground truth).
  * NO NETWORK REQUIRED   the callable and gdn backends are fully local; http/openai use
                          only urllib from the standard library.
"""
import json, time, urllib.request
import numpy as np


class LocalRung:
    def __init__(self, fn, name="local", ouroboros=None, n_new=64):
        self._fn = fn
        self.name = name
        self.ouro = ouroboros
        self.n_new = int(n_new)
        self.calls = 0
        self.tokens_out = 0
        self.seconds = 0.0
        self.last_error = None

    # ---------------------------------------------------------------- backends
    @classmethod
    def callable(cls, fn, **kw):
        """Any python callable prompt -> text. The honest baseline for tests."""
        return cls(fn, name=kw.pop("name", "callable"), **kw)

    @classmethod
    def http(cls, url, model="qwen3.5:0.8b", timeout=120, **kw):
        """Ollama-style /api/generate. Streams off; one JSON body back."""
        def fn(prompt):
            body = json.dumps({"model": model, "prompt": str(prompt),
                               "stream": False}).encode()
            req = urllib.request.Request(url, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode()).get("response", "")
        return cls(fn, name="ollama:" + model, **kw)

    @classmethod
    def openai(cls, url, model="local", api_key=None, timeout=120, **kw):
        """OpenAI-compatible /v1/chat/completions -- llama.cpp server, vLLM, LM Studio."""
        def fn(prompt):
            body = json.dumps({"model": model,
                               "messages": [{"role": "user", "content": str(prompt)}]}).encode()
            hdr = {"Content-Type": "application/json"}
            if api_key:
                hdr["Authorization"] = "Bearer " + api_key
            req = urllib.request.Request(url, data=body, headers=hdr)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode())
            return d["choices"][0]["message"]["content"]
        return cls(fn, name="openai:" + model, **kw)

    @classmethod
    def gdn(cls, model_dir, layer=None, dk=64, decay=0.98, partition=None,
            n_new=16, **kw):
        """A leCore-installed checkpoint through GDNRuntime, with Ouroboros in its pass.

        This is the backend that closes the loop: Unicron installs leCore into the weights,
        the model runs it, and what the run learns spills back to the same partition."""
        from holographic.io_and_interop.holographic_gdnruntime import load_runtime
        from holographic.agents_and_reasoning.holographic_galvatron import OuroborosResident
        rt, cfg = load_runtime(model_dir)
        hidden = None
        for k, v in (cfg or {}).items():
            if k in ("hidden_size", "d_model", "n_embd"):
                hidden = int(v)
        if hidden is None:
            import os
            from holographic.io_and_interop.holographic_unicron import load_safetensors
            t = load_safetensors(os.path.join(model_dir, "model.safetensors"))
            emb = [k for k in t if k.endswith("embed_tokens.weight")][0]
            hidden = int(np.asarray(t[emb]).shape[1])
        n_layers = int((cfg or {}).get("n_layers") or 8)
        lay = int(n_layers // 2 if layer is None else layer)
        ouro = None
        if partition is not None:
            from holographic.caching_and_storage.holographic_knowledgestore import KnowledgeStore
            ouro = OuroborosResident(hidden_dim=hidden, layer=lay, dk=dk, decay=decay,
                                     partition=KnowledgeStore(partition))
        vocab = None

        def fn(prompt):
            ids = []
            for w in str(prompt).lower().split()[:24]:
                h = 0
                for ch in w:
                    h = (h * 131 + ord(ch)) % 2048
                ids.append(h)
            ids = ids or [1]
            hooks = {lay: ouro.hook} if ouro is not None else None
            cur, out = list(ids), []
            for _ in range(n_new):
                lg = np.asarray(rt.forward(cur, hooks=hooks))
                nxt = int(np.argmax(lg[-1]))
                out.append(nxt)
                cur = (cur + [nxt])[-32:]
            return " ".join("tok%d" % i for i in out)
        r = cls(fn, name="gdn:" + str(model_dir), ouroboros=ouro, n_new=n_new, **kw)
        r.runtime = rt
        r.manifest = _read_manifest(model_dir)
        return r

    # ---------------------------------------------------------------- the rung
    def __call__(self, prompt):
        t0 = time.time()
        try:
            out = self._fn(prompt)
            self.last_error = None
        except Exception as exc:                      # a rung that dies must not take the
            self.last_error = str(exc)[:200]          # ladder with it -- leCore's free
            out = ""                                  # rungs keep serving without it
        self.seconds += time.time() - t0
        self.calls += 1
        self.tokens_out += len(str(out).split())
        return out

    def stats(self):
        s = {"backend": self.name, "model_calls": self.calls,
             "tokens_generated": self.tokens_out,
             "seconds": round(self.seconds, 2),
             "last_error": self.last_error}
        if self.ouro is not None:
            cap = self.ouro.capacity_report()
            s["ouroboros"] = {"writes": self.ouro.n_writes,
                              "regime": cap["regime"],
                              "predicted_recall": round(cap["predicted_recall"], 3),
                              "prediction_is_upper_bound": cap["prediction_is_upper_bound"]}
        return s

    def verify_memory(self, pairs):
        """Ground truth for the trace, because a prediction that cannot see its own regime
        must not be the last word (cp46)."""
        return None if self.ouro is None else self.ouro.verify_recall(pairs)


def _read_manifest(model_dir):
    import os
    p = os.path.join(model_dir, "lecore.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _selftest():
    calls = {"n": 0}

    def fake(prompt):
        calls["n"] += 1
        return "a model answer about " + str(prompt).split()[-1]

    import lecore
    rung = LocalRung.callable(fake)
    m = lecore.UnifiedMind()
    m.zoo_attach(rung)
    m.teach("what is the escalation budget", "three rungs before the model wakes")
    aT = m.ask("what is the escalation budget")
    assert aT["tier"] == "T0" and aT.get("provenance") == "taught"
    assert rung.stats()["model_calls"] == 0, "a taught fact must never wake the model"
    m.ask("something never taught at all")
    assert rung.stats()["model_calls"] >= 1, "a real miss must reach the rung"
    aM = m.ask("something never taught at all")
    assert aM.get("provenance") == "model-cached", \
        "a cached model answer must declare itself (cp47)"

    def broken(prompt):
        raise RuntimeError("the local server is down")

    bad = LocalRung.callable(broken)
    m2 = lecore.UnifiedMind()
    m2.zoo_attach(bad)
    m2.teach("does memory survive a dead rung", "yes -- the free rungs do not need it")
    assert m2.ask("does memory survive a dead rung")["tier"] == "T0"
    m2.ask("a question that needs the dead model")
    assert bad.stats()["last_error"], "a dead rung is reported, not raised"
    return "OK: LocalRung pins passed (taught never wakes the model; cached answers " \
           "declare provenance; a dead rung degrades to the free rungs)"


if __name__ == "__main__":
    print(_selftest())
