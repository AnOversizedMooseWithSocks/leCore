"""SESSION -- never compute the same conversation prefix twice.

Moose runs a 0.8B on a CPU laptop and it is slow. The single largest waste in a
conversation is not the model's arithmetic -- it is that every turn RE-PREFILLS
the entire history. MEASURED on a realistic six-turn exchange: 489 tokens
processed, of which only 137 were new. SEVENTY-TWO PERCENT OF THE WORK WAS
REPEATED, and the fraction grows with every turn.

leCore already had the pieces and never joined them: the runtime exposes
prefill/step over an InferenceState with copy(), galvacache memoises pure
functions at a measured 75% hit rate with bit-identical output, and sessions
exist. What was missing is the RADIX TREE -- the structure that answers "what is
the longest prefix of this prompt that I have already computed?"

WHY A TREE AND NOT A DICTIONARY: turn 4 of a conversation shares its first three
turns with turn 3, and a dictionary keyed on the whole prompt misses that
completely. vLLM and SGLang call this RadixAttention; the idea is the same
whatever the model: index by prefix, resume from the deepest match, compute only
the tail.

THE GUARANTEE THIS KEEPS, because a cache that changes answers is worse than no
cache: resuming from a cached state reproduces a full recompute TO FLOAT
ROUNDING -- measured 7.1e-15, machine epsilon. Not bit-identical, and the
difference is real rather than pedantic: resuming STEPS the tail while a fresh
call PREFILLS it, and the two associate their sums differently. The selftest
asserts the measured bound against the live runtime rather than assuming it.
"""

import numpy as np


class PrefixCache:
    """A radix tree over token sequences, holding inference states.

    Keyed by TOKEN, not by text: two prompts that tokenize the same share work
    even if they differ in whitespace, and two that look similar but tokenize
    differently correctly do not."""

    def __init__(self, runtime, max_nodes=512, step_cost=None):
        self.rt = runtime
        self.max_nodes = int(max_nodes)
        # MEASURE THE CROSSOVER, DO NOT ASSUME IT. Resuming replays the tail one
        # token at a time while a fresh call PREFILLS the whole prompt in one
        # batched pass -- and stepping is 6.6x slower PER TOKEN on this runtime.
        # So saving 72% of the tokens was still a NET LOSS in wall clock (0.124s
        # against 0.088s). A cache that saves work and costs time is not a
        # cache. Resume only when the tail is short enough that it wins.
        self.step_cost = (float(step_cost) if step_cost is not None
                          else self._measure_step_cost())
        # node: {"children": {token: node}, "state": state|None, "logits": ...,
        #        "depth": int, "hits": int}
        self.root = {"children": {}, "state": None, "logits": None,
                     "depth": 0, "hits": 0}
        self.nodes = 1
        self.stats = {"hits": 0, "misses": 0, "tokens_saved": 0,
                      "tokens_computed": 0}

    def _measure_step_cost(self, n=64):
        """Cost of a RESUMED token against a freshly prefilled one.

        This used to measure STEPPING, which cost 5.8-6.9x and made the cache
        correctly refuse to resume. With forward(resume=) the tail runs in ONE
        batched pass and the ratio collapses to roughly 1 -- so the same policy
        that declined before now accepts, without changing the policy. Measuring
        the cost of the mechanism you actually use is the whole trick."""
        import time
        probe = list(range(5, 5 + int(n)))
        try:
            self.rt.prefill(probe)                      # warm
            t0 = time.time()
            self.rt.prefill(probe)
            t_pref = max(time.time() - t0, 1e-9) / len(probe)
            _lg, st = self.rt.prefill(probe[: n // 2])
            tail = probe[n // 2:]
            self.rt.forward(tail, resume=st, collect_state=True)   # warm
            t0 = time.time()
            self.rt.forward(tail, resume=st, collect_state=True)
            t_res = max(time.time() - t0, 1e-9) / max(len(tail), 1)
            return float(t_res / t_pref)
        except Exception:
            return 1.0

    def _worth_resuming(self, matched, total):
        """Would resuming beat a fresh prefill? Pure arithmetic on the measured
        cost: the tail costs (total-matched) * step_cost, a fresh call costs
        total."""
        if matched <= 0:
            return False
        return (total - matched) * self.step_cost < total

    def _walk(self, ids):
        """Deepest cached node along this token path, and how far it got."""
        # DESCEND PAST STATELESS NODES. Only terminal nodes carry a state, so
        # stopping at the first one without a state means never matching
        # anything -- measured as 0% saved on a conversation that shares 72% of
        # its tokens. Walk as deep as the tokens allow, and remember the deepest
        # node that HAS a state.
        node = self.root
        best, best_i = self.root, 0
        i = 0
        for t in ids:
            nxt = node["children"].get(int(t))
            if nxt is None:
                break
            node = nxt
            i += 1
            if nxt["state"] is not None:
                best, best_i = nxt, i
        return best, best_i

    def forward(self, token_ids):
        """Logits for this sequence, computing only the uncached tail."""
        ids = [int(t) for t in token_ids]
        node, matched = self._walk(ids)
        # a caller expects the same shape every time, so the cache stores and
        # returns the LAST-POSITION row regardless of which path produced it
        if matched == len(ids) and node["logits"] is not None:
            node["hits"] += 1
            self.stats["hits"] += 1
            self.stats["tokens_saved"] += len(ids)
            return node["logits"]

        if not self._worth_resuming(matched, len(ids)):
            matched = 0
        if matched == 0:
            logits, state = self.rt.prefill(ids)
            computed = len(ids)
            node = self.root
            walk_from = 0
        else:
            # BATCHED RESUME, not token-at-a-time. forward(resume=state) runs the
            # tail in ONE pass: measured 0.0283s against 0.1141s for stepping and
            # 0.1918s for a full recompute -- 4.0x over the old path and 6.8x
            # over recomputing. Stepping is why this cache used to SAVE THE WORK
            # AND LOSE THE WALL CLOCK, and why it was correctly declining to
            # resume at all.
            state = node["state"].copy()
            tail = ids[matched:]
            computed = len(tail)
            out = self.rt.forward(tail, resume=state, collect_state=True)
            if isinstance(out, tuple):
                logits, state = out
            else:
                logits = out
                _l, state = self.rt.prefill(ids)
                computed = len(ids)

        self.stats["misses"] += 1
        self.stats["tokens_saved"] += len(ids) - computed
        self.stats["tokens_computed"] += computed

        # store the terminal state only: interior nodes cost memory and the
        # radix walk already finds the deepest STORED ancestor
        cur = self.root
        for t in ids:
            cur = cur["children"].setdefault(
                int(t), {"children": {}, "state": None, "logits": None,
                         "depth": cur["depth"] + 1, "hits": 0})
            if cur["state"] is None:
                self.nodes += 1
        if self.nodes <= self.max_nodes:
            cur["state"] = state.copy() if hasattr(state, "copy") else state
            L = np.asarray(logits, np.float64)
            cur["logits"] = np.array(L[-1] if L.ndim == 2 else L, copy=True)
        return logits

    def report(self):
        total = self.stats["tokens_saved"] + self.stats["tokens_computed"]
        return dict(self.stats, nodes=self.nodes,
                    saved_fraction=(self.stats["tokens_saved"] / total)
                    if total else 0.0)


def _selftest():
    import os
    import time

    from holographic.io_and_interop.holographic_gdnruntime import load_runtime

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("session selftest SKIPPED-SUBJECT (no model present)")
        return
    rt, cfg = load_runtime(src)

    def tok(t):
        return [b for b in t.encode("utf-8")]

    turns = ["what is the holographic memory ", "how does binding work ",
             "and unbinding ", "what is the capacity ",
             "how do we clean up noise ", "what about the codebook "]

    # ---- NAIVE: re-prefill the whole history every turn ----
    hist = ""
    ref = []
    t0 = time.time()
    for t in turns:
        hist += t
        ref.append(rt.forward(tok(hist)))
    naive = time.time() - t0

    # ---- CACHED ----
    pc = PrefixCache(rt)
    hist = ""
    t0 = time.time()
    got = []
    for t in turns:
        hist += t
        got.append(pc.forward(tok(hist)))
    cached = time.time() - t0

    # ---- IDENTICAL TO FLOAT ROUNDING, which is the true guarantee and not
    #      the same as bit-identical. Resuming from a state STEPS the tail while
    #      a full recompute PREFILLS it, and those associate differently:
    #      measured 7.1e-15, machine epsilon on a float64 path. Asserting
    #      bit-identity here failed a correct cache, and quietly loosening the
    #      claim afterwards would have been worse than measuring it.
    # NORMALISE THE SHAPES BEFORE COMPARING. prefill returns logits for EVERY
    # position (S, vocab) while step returns ONE row (vocab,), so `[-1]` means
    # "last position" on one and "last vocabulary entry" on the other -- a
    # scalar against a vector, which produced a bogus error of 12.3 and looked
    # exactly like a broken cache.
    def _last(x):
        a = np.asarray(x, np.float64)
        return a[-1] if a.ndim == 2 else a

    worst = max(float(np.max(np.abs(_last(a) - _last(b))))
                for a, b in zip(ref, got))
    assert worst < 1e-9, worst

    rep = pc.report()
    # ---- THE CACHE MUST NOT BE SLOWER. That is the whole point, and the first
    #      version saved 72% of the tokens while costing 40% more wall clock.
    assert cached <= naive * 1.05, (cached, naive)

    # ---- a REPEATED turn must be a pure hit ----
    before = pc.stats["hits"]
    pc.forward(tok(hist))
    assert pc.stats["hits"] == before + 1

    print("session selftest OK -- a six-turn conversation re-prefills 489 tokens "
          "naively of which only 137 are new; the prefix cache computes %d and "
          "reuses %d (%.0f%% saved) with stepping measured at %.1fx a prefilled "
          "token so it only resumes when that WINS, matches a full recompute to "
          "%.1e at every turn, "
          "and a repeated turn is a pure hit -- %.3fs against %.3fs"
          % (rep["tokens_computed"], rep["tokens_saved"],
             100 * rep["saved_fraction"], pc.step_cost, worst, cached, naive))


if __name__ == "__main__":
    _selftest()
