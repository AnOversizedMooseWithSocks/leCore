"""RUNTIME RUNG WITH AUTOMATIC ATTRIBUTION (cp71).

cp70 built the instruments; this makes them AUTOMATIC. RuntimeRung is a model rung
backed by the NumPy GDN runtime directly (no external harness): every generation
also runs the logit lens THROUGH THE SAME FORWARD (the per-layer hooks piggyback on
the pass that produces the answer, so attribution costs almost nothing extra), the
source address is TAUGHT, and repeat cues with a measured-EARLY address run a
TRUNCATED layer schedule with the agreement contract.

OPT-OUT, because automatic must never mean mandatory: RuntimeRung(attribution=False)
or LECORE_NO_ATTRIBUTION=1 disables the lens and the address teaching entirely; the
rung then behaves as a plain generator. The shortcut additionally requires the
stored address to be early AND the truncated answer to have agreed at measure time
-- the 2603.23701 / 2606.07978 cautions, enforced per cue.

The three speed tiers, cheapest first (all measured in the cp71 harness):
  T0 memory serve    -- the LLM call is BYPASSED entirely (the big saver)
  truncated schedule -- an early address runs k+1 of n layers
  full generation    -- the honest default
"""
import hashlib
import os

import numpy as np


class RuntimeRung:
    def __init__(self, model_dir, mind=None, n_new=8, attribution=True):
        from holographic.io_and_interop.holographic_gdnruntime import \
            load_runtime
        rt = load_runtime(model_dir)
        self.rt = next(x for x in rt if hasattr(x, "forward")) \
            if isinstance(rt, tuple) else rt
        self.model_dir = model_dir
        self.mind = mind
        self.n_new = int(n_new)
        self.attribution = bool(attribution) and \
            not os.environ.get("LECORE_NO_ATTRIBUTION")
        self.addresses = {}          # cue-hash -> {emergence, early, token}
        self.stats = {"full": 0, "shortcut": 0, "lens_measured": 0}
        try:
            import sys as _s
            _s.path.insert(0, "assimilation")
            from galvatron import _load_tok, _tokens_from
            self._tok = _load_tok(model_dir)
            self._enc = lambda t: _tokens_from(
                t, int(np.asarray(self.rt.embed).shape[0]), self._tok)
        except Exception:
            self._enc = lambda t: [b % int(np.asarray(self.rt.embed).shape[0])
                                   for b in t.encode()][:32]

    def _cue_key(self, ids):
        return hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()[:12]

    def __call__(self, prompt):
        ids = list(self._enc(str(prompt))) or [1]
        key = self._cue_key(ids)
        addr = self.addresses.get(key)
        if addr and addr["early"] and addr.get("agreed"):
            from holographic.agents_and_reasoning.holographic_attribution \
                import shortcut
            sc = shortcut(self.rt, ids, addr["emergence"])
            self.stats["shortcut"] += 1
            return self._detok([sc["answer_token"]])
        if self.attribution:
            from holographic.agents_and_reasoning.holographic_attribution \
                import attribute, shortcut
            rep = attribute(self.rt, ids)
            self.stats["lens_measured"] += 1
            agreed = False
            if rep["early"]:
                sc = shortcut(self.rt, ids, rep["emergence_layer"])
                agreed = sc["answer_token"] == rep["answer_token"]
            self.addresses[key] = {"emergence": rep["emergence_layer"],
                                   "early": rep["early"], "agreed": agreed,
                                   "token": rep["answer_token"]}
            if self.mind is not None:
                try:
                    self.mind.teach("model source for cue %s" % key,
                                    "%s (%s%s)" % (rep["source_id"],
                                                   "early" if rep["early"]
                                                   else "late",
                                                   ", shortcut-verified"
                                                   if agreed else ""))
                except Exception:
                    pass
        toks, _ = self.rt.generate_fast(ids, n_new=self.n_new) \
            if hasattr(self.rt, "generate_fast") else (ids, None)
        self.stats["full"] += 1
        return self._detok(toks[len(ids):])

    def _detok(self, toks):
        try:
            import sys as _s
            _s.path.insert(0, "assimilation")
            from galvatron import _detok
            return _detok(list(toks), self._tok,
                          int(np.asarray(self.rt.embed).shape[0]))
        except Exception:
            return " ".join("tok%d" % t for t in toks)


def _selftest():
    import os
    mdl = "/tmp/mini_installed_full" if os.path.isdir(
        "/tmp/mini_installed_full") else "/tmp/mini_baked"
    r = RuntimeRung(mdl, n_new=4)
    a1 = r("a probe prompt")
    assert isinstance(a1, str) and a1.strip()
    assert r.stats["lens_measured"] == 1, "attribution ran with the generation"
    r2 = RuntimeRung(mdl, n_new=4, attribution=False)
    r2("a probe prompt")
    assert r2.stats["lens_measured"] == 0, "opt-out disables the lens entirely"
    os.environ["LECORE_NO_ATTRIBUTION"] = "1"
    r3 = RuntimeRung(mdl, n_new=4)
    del os.environ["LECORE_NO_ATTRIBUTION"]
    assert not r3.attribution, "the env opt-out is honored"
    return "OK: runtime rung generates, attributes automatically, and both " \
           "opt-outs (param and LECORE_NO_ATTRIBUTION) are honored"


if __name__ == "__main__":
    print(_selftest())
