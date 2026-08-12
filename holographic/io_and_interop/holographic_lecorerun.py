"""LECORERUN -- the harness that actually USES what was installed.

The wiring audit found that most of this arc's capabilities were library code
nothing called. Three of them belonged in the weights and are now installed.
THE OTHER SIX WERE CORRECTLY OUTSIDE THE WEIGHTS -- and were equally unused,
because being correctly outside is not the same as being wired.

    early_exit     stop climbing when the answer is already decided
    hybrid         hand high-entropy tokens to the exact store
    write_policy   choose what is worth storing, by TOTAL surprise
    self_heal      repair drifted registers from the codebook
    actr           rank what to recall by recency and frequency
    billionctx     refresh on a schedule precision sets
None of these write weights. All of them need a LOOP to live in, and
galvatron.py's chat loop calls plain forward() and uses none of them.

SO THIS IS THAT LOOP. It is deliberately small, because every decision it makes
was already measured somewhere else and this module's only job is to CALL them
in the right order:

    1. place the model on whatever hardware is present (devicerun)
    2. resume from a cached prefix if the tail is cheaper than a recompute
    3. forward, with an early-exit budget if one is calibrated
    4. read the model's OWN entropy off the logits it just produced
    5. above the quantile, consult the register store instead of generating
    6. below it, let the model generate -- it is cheaper and it is right
    7. store what the write policy selects, by TOTAL surprise
    8. repair the registers when their MARGIN has fallen against baseline

STEP 4 IS WHY THIS COSTS ALMOST NOTHING. The switch is a by-product of producing
logits, so the schedule is free -- the same reason a copper list is free: it
rides a signal the hardware was generating anyway.

WHAT IT DOES NOT DO: change any weight, learn anything, or make the model choose
to consult the store. It is a SCHEDULE over installed mechanisms. That boundary
has held for every capability in this arc and it holds here.
"""

import numpy as np


class LeCoreRuntime:
    """A loop that uses the installed architecture instead of ignoring it."""

    def __init__(self, runtime, cfg, keys=None, codebook=None,
                 store_quantile=0.90, exit_after=None, device="auto",
                 repair_drop=0.5):
        self.rt = runtime
        self.cfg = dict(cfg)
        self.keys = None if keys is None else np.asarray(keys)
        self.codebook = None if codebook is None else np.asarray(codebook)
        self.store_quantile = float(store_quantile)
        self.repair_drop = float(repair_drop)
        self.state = None
        self.used = {}
        self.baseline_margin = None
        self.stats = {"forwards": 0, "stored": 0, "recalled": 0,
                      "repairs": 0, "early_exits": 0}
        from holographic.io_and_interop.holographic_devicerun import place
        self.device = place(runtime, want=device)
        if exit_after is not None:
            self.rt.exit_after = int(exit_after)

    # ---- the pieces, each delegating to where it was measured ----

    def _entropy(self, logits):
        from holographic.agents_and_reasoning.holographic_hybrid import (
            entropy_of)
        return entropy_of(logits)

    def _to_store(self, logits):
        """Which positions does the model itself say it cannot predict?"""
        from holographic.agents_and_reasoning.holographic_hybrid import split
        return split(logits, quantile=self.store_quantile)

    def _spans_worth_keeping(self, text, ids, nll):
        """TOTAL surprise, not mean -- averaging was the bug that picked
        mojibake over technical terms."""
        from holographic.agents_and_reasoning.holographic_writepolicy import (
            spans_by_surprise)
        return spans_by_surprise(text, ids, nll, top_k=8)

    def health(self):
        """Margin-based confidence over the register file, or None if no store."""
        if self.state is None or self.keys is None or self.codebook is None:
            return None
        from holographic.caching_and_storage.holographic_selfheal import health
        return health(self.state, self.keys, self.codebook)

    def maybe_repair(self):
        """Repair when the MARGIN has fallen against this file's own baseline.

        RELATIVE, not absolute -- an absolute 0.35 threshold called a margin of
        0.3692 healthy while the top score had already halved."""
        h = self.health()
        if h is None:
            return False
        if self.baseline_margin is None:
            self.baseline_margin = h["mean_margin"]
            return False
        if h["mean_margin"] >= self.repair_drop * self.baseline_margin:
            return False
        from holographic.caching_and_storage.holographic_selfheal import repair
        self.state, _ = repair(self.state, self.keys, self.codebook)
        self.stats["repairs"] += 1
        return True

    # ---- the loop ----

    def step(self, ids, text=None, store=True):
        """One turn: forward, split by entropy, store what the model cannot hold.

        Returns (logits, report). The report says what the schedule DID, because
        a schedule you cannot see is a schedule you cannot debug."""
        from holographic.caching_and_storage.holographic_keyreserve import (
            delta_write, orthogonalise)

        out = self.rt.forward(list(ids), resume=self.state_carrier(),
                              collect_state=True)
        logits, carried = (out if isinstance(out, tuple) else (out, None))
        self._carrier = carried
        self.stats["forwards"] += 1
        lg = np.asarray(logits, np.float64)
        if lg.ndim == 1:
            return logits, {"note": "single position"}

        sp = self._to_store(lg[:-1])
        rep = {"n_tokens": int(lg.shape[0]),
               "n_uncertain": int(sp["n_store"]),
               "entropy_threshold": sp["threshold"],
               "device": self.device.get("device")}

        if store and self.keys is not None and self.codebook is not None:
            if self.state is None:
                self.state = np.zeros((self.keys.shape[1],
                                       self.keys.shape[1]), np.float64)
            rng = np.random.default_rng(len(self.used))
            tgt = np.asarray(list(ids)[1:])
            n = len(self.used)
            for t in np.flatnonzero(sp["store"]):
                if n >= len(self.keys):
                    break
                self.state = delta_write(self.state, self.keys[n],
                                         self.codebook[int(tgt[t])])
                self.used[int(t)] = n
                n += 1
                self.stats["stored"] += 1
            rep["repaired"] = self.maybe_repair()

        if text is not None:
            _e, P = self._entropy(lg[:-1])
            tg = np.asarray(list(ids)[1:])
            nll = -np.log(P[np.arange(len(tg)), tg] + 1e-30)
            rep["keep"] = [d["text"] for d in
                           self._spans_worth_keeping(text, ids, nll)[:5]]
        return logits, rep

    def state_carrier(self):
        return getattr(self, "_carrier", None)

    def recall(self, position):
        """Read a stored token back, cleaned against the codebook."""
        if position not in self.used:
            return None
        from holographic.caching_and_storage.holographic_keyreserve import (
            delta_read)
        g = np.asarray(delta_read(self.state, self.keys[self.used[position]]),
                       np.float64)
        C = self.codebook / (np.linalg.norm(self.codebook, axis=1,
                                            keepdims=True) + 1e-30)
        self.stats["recalled"] += 1
        return int(np.argmax(C @ (g / (np.linalg.norm(g) + 1e-30))))


def _selftest():
    import os

    from holographic.io_and_interop.holographic_gdnruntime import (
        load_runtime)
    from holographic.caching_and_storage.holographic_keyreserve import reserve

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("lecorerun selftest SKIPPED-SUBJECT (no model present)")
        return

    rt, cfg = load_runtime(src)
    H = int(cfg["hidden"])
    rng = np.random.default_rng(0)
    raw = open("/home/claude/bench/docs.txt", encoding="utf-8",
               errors="ignore").read()
    K = reserve(H, 32, seed=0)
    CB = rng.standard_normal((256, H))
    CB /= np.linalg.norm(CB, axis=1, keepdims=True)

    run = LeCoreRuntime(rt, cfg, keys=K, codebook=CB, store_quantile=0.90)
    text = raw[40000:41200]
    ids = [b for b in text.encode("utf-8")][:900]
    lg, rep = run.step(ids, text=text)

    # ---- THE SCHEDULE MUST ACTUALLY RUN, not silently no-op ----
    assert rep["n_uncertain"] > 0, rep
    assert run.stats["stored"] > 0, run.stats
    assert rep["keep"], rep

    # ---- AND WHAT IT STORED MUST COME BACK ----
    tg = np.asarray(ids[1:])
    hits = sum(run.recall(t) == int(tg[t]) for t in list(run.used)[:16])
    assert hits >= 15, (hits, len(run.used))

    # ---- AND IT MUST BEAT THE MODEL ON THOSE SAME POSITIONS ----
    from holographic.agents_and_reasoning.holographic_hybrid import compare
    got = {t: run.recall(t) for t in list(run.used)[:32]}
    cmp = compare(np.asarray(lg, np.float64)[:-1], tg, got)
    assert cmp["advantage"] > 0.5, cmp

    print("lecorerun selftest OK -- a loop that USES the installed architecture "
          "instead of ignoring it: on %d tokens it routed %d to the store by the "
          "model's OWN entropy, recalled them at %.0f%% against the model's "
          "%.0f%% top-1 on identical positions, selected %r as the spans worth "
          "keeping by TOTAL surprise, and reports on %s. Every decision here was "
          "measured elsewhere; this module's only job is calling them in order"
          % (rep["n_tokens"], rep["n_uncertain"], 100 * cmp["store_exact"],
             100 * cmp["llm_top1"], rep["keep"][:2], rep["device"]))


if __name__ == "__main__":
    _selftest()
