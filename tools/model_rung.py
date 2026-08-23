"""The BACK END: the installed model as leCore's LLM rung, with Ouroboros in its pass.

HONEST SCOPE: the mini is RANDOM-INIT with a toy vocab (tok0..tok1991), so the model's
answer TEXT is meaningless. What is real and measured here: a real GDNRuntime forward
pass per escalation, real logits, real sampled tokens, real Ouroboros trace writes from
the live residual stream, real latency, and the real ladder economics (how often leCore's
free rungs answer without waking the model at all).
"""
import time
import numpy as np
from holographic.io_and_interop.holographic_gdnruntime import load_runtime
from holographic.agents_and_reasoning.holographic_galvatron import OuroborosResident
from holographic.caching_and_storage.holographic_knowledgestore import KnowledgeStore


class ModelRung:
    def __init__(self, model_dir, partition, layer=13, dk=64, decay=0.98, n_new=6):
        self.rt, self.cfg = load_runtime(model_dir)
        self.store = KnowledgeStore(partition)
        self.ouro = OuroborosResident(hidden_dim=128, layer=layer, dk=dk,
                                      decay=decay, partition=self.store)
        self.layer = int(layer)
        self.n_new = int(n_new)
        self.calls = 0
        self.tokens_out = 0
        self.seconds = 0.0

    def _encode(self, text, vocab=2048):
        # deterministic word -> id (toy vocab; the runtime only needs valid ids)
        out = []
        for w in str(text).lower().split()[:24]:
            h = 0
            for ch in w:
                h = (h * 131 + ord(ch)) % vocab
            out.append(h)
        return out or [1]

    def __call__(self, prompt):
        """leCore's T4 rung: a REAL forward pass with the memory manager hooked."""
        ids = self._encode(prompt)
        t0 = time.time()
        out_ids = []
        cur = list(ids)
        for _ in range(self.n_new):
            logits = np.asarray(self.rt.forward(cur, hooks={self.layer: self.ouro.hook}))
            nxt = int(np.argmax(logits[-1]))
            out_ids.append(nxt)
            cur = (cur + [nxt])[-32:]
        self.seconds += time.time() - t0
        self.calls += 1
        self.tokens_out += len(out_ids)
        return "tok" + " tok".join(str(i) for i in out_ids)

    def stats(self):
        cap = self.ouro.capacity_report()
        return {"model_calls": self.calls, "tokens_generated": self.tokens_out,
                "forward_seconds": round(self.seconds, 2),
                "ouroboros_writes": self.ouro.n_writes,
                "trace_effective_n": round(cap["n_effective"], 2),
                "predicted_recall": round(cap["predicted_recall"], 3),
                "saturating": cap["saturating"]}
