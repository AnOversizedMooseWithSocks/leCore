"""LLMSEAM -- give the LLM connection the same treatment every other expensive unit in leCore gets.

THE DEMOSCENE READING (Quilez's discipline, applied literally). A 64k intro stores no assets: it
REGENERATES a whole world deterministically from a tiny kernel. But it does not regenerate everything --
anything expensive and REUSED gets baked to a table once and sampled thereafter. Bake what is costly and
repeated; regenerate what is cheap and deterministic. That line is drawn all over this engine --
seventeen measured units, seven storage tiers, `compute_plan`'s amortisation ladder -- and the engine's
own dispatcher already ruled on this exact case:

    compute_plan(n, repeat_fraction=0.9) -> tier="memo"
    "exact hash-replay beats every backend at any size -- recomputing a known answer is the only true waste"

And yet `attach_llm` was `self._llm = llm` and nothing else: the single most expensive operation in the
system, orders of magnitude past every unit in the machine model, sitting on the REGENERATE side of that
line with no cache, no accounting, and no budget. Every consumer (expand_query, llm_tool, AgentBridge)
called it raw. This module is the missing tier.

WHAT IT IS NOT: a model SDK, a retry policy with backoff, or an async pool. It is a metered, optionally
memoised WRAPPER around whatever callable you brought, and it stays a plain `text -> text` so every
existing caller keeps working unchanged.

WHY CACHING IS OPT-IN AND NOT DEFAULT -- the kept negative, stated before the feature. Memoisation is
sound only for a PURE function. A model sampled at temperature > 0 is NOT pure: the same prompt is
supposed to give different answers, and that is exactly what a swarm's branches rely on. Caching such a
model silently collapses N branches into one answer and makes a fan-out look unanimous -- a false
consensus manufactured by the cache, which is the worst failure this codebase knows how to produce.
So `cache=False` is the default, the docstring says why, and `_selftest` pins the collapse as a
DEMONSTRATED negative rather than a warning.

ACCOUNTING IS ALWAYS ON, because it is free and the audit found the gap: "how much did that cost" and
"count how many times a function was called" both returned unrelated fallbacks. You cannot make openzoo
faster without first being able to see what it spends.
"""
import hashlib
import time


class BudgetExceeded(RuntimeError):
    """Raised when a metered seam is asked for more calls than its budget allows.

    FAILS CLOSED, deliberately. A budget that degrades to "call anyway" is not a budget, and a runaway
    agent loop is precisely the case a budget exists for -- the loud stop is the feature."""


class MeteredLLM:
    """A `text -> text` callable wrapping another, with counters, an optional exact-replay cache, and an
    optional hard call budget.

    Stays callable-shaped on purpose: `attach_llm`, `expand_query(llm=...)`, `llm_tool` and AgentBridge
    all take a bare callable, so a wrapper that is anything else would need every one of them changed.
    This is the additive route -- wrap, do not rewire.

    The cache key is `hashlib.sha256` of the prompt, never `hash()`: PYTHONHASHSEED is pinned across this
    codebase precisely so content hashes are stable across processes, and a cache keyed on a salted hash
    would silently miss on every restart."""

    def __init__(self, fn, cache=False, budget=None, name="llm", batch_fn=None, on_outcome=None):
        if not callable(fn):
            raise TypeError("MeteredLLM needs a callable text->text, got %r" % type(fn))
        if batch_fn is None:
            batch_fn = getattr(fn, "batch", None)      # a backend may ADVERTISE batching by carrying .batch
        if batch_fn is not None and not callable(batch_fn):
            raise TypeError("batch_fn must be a callable texts->texts, got %r" % type(batch_fn))
        self.fn = fn
        self.batch_fn = batch_fn
        self.on_outcome = on_outcome
        self.round_trips = 0
        self.outcomes = []        # (prompt, reply, verdict) -- what the model was told back
        self.outcome_counts = {}
        self.name = str(name)
        self.cache_on = bool(cache)
        self.budget = None if budget is None else int(budget)
        self._store = {}
        self.calls = 0            # calls that actually reached the wrapped model
        self.hits = 0             # calls served from replay
        self.deduped = 0          # asks collapsed onto another ask INSIDE one batch (never reached the model)
        self._prefixes = []       # every prompt seen, for longest-common-prefix accounting
        self.prefix_chars_reusable = 0
        self.prefix_chars_total = 0
        self.chars_in = 0
        self.chars_out = 0
        self.seconds = 0.0
        self.errors = 0

    def __call__(self, prompt):
        text = "" if prompt is None else str(prompt)
        if self.cache_on:
            k = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if k in self._store:
                self.hits += 1
                return self._store[k]
        if self.budget is not None and self.calls >= self.budget:
            raise BudgetExceeded("%s: budget of %d model calls exhausted (%d hits served from replay)"
                                 % (self.name, self.budget, self.hits))
        self._account_prefix(text)
        t0 = time.perf_counter()
        try:
            out = self.fn(text)
        except Exception:
            self.errors += 1
            self.seconds += time.perf_counter() - t0
            raise                                    # a metered seam MEASURES failures, it does not eat them
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        self.chars_in += len(text)
        self.chars_out += len(out or "")
        if self.cache_on:
            self._store[hashlib.sha256(text.encode("utf-8")).hexdigest()] = out
        return out

    def batch(self, prompts):
        """Answer many prompts, in order, with as few round trips as the backend allows.

        THE UP DIRECTION OF THE SEAM, and it is the same move `Clean up many cues at once` already made for
        cleanup -- "one (K,D)x(D,M) matmul instead of K separate matvecs". A `text -> text` signature forces
        a fan-out to be sequential no matter how parallel the backend is, which is the shape that makes a
        3-branch swarm cost three round trips instead of one.

        THREE FILTERS COMPOSE, cheapest first, and the ordering is the whole design:
          1. REPLAY  -- prompts already in the cache never leave the process (across time).
          2. DEDUP   -- identical prompts inside THIS call occupy one backend slot (within a request).
          3. BATCH   -- whatever survives goes to `batch_fn` in ONE round trip, or falls back to
                        sequential `__call__` when no batch backend was advertised.
        A fan-out's branch prompts repeat both ways, so 1 and 2 attack different halves of the same waste.

        KEPT NEGATIVE -- DEDUP IS TIED TO `cache`, NOT ALWAYS ON, and for exactly the reason caching is
        opt-in. Collapsing identical prompts is sound only for a PURE function. Dedup a SAMPLING model and
        three branches asking the same question get one shared answer: the same false consensus the cache
        manufactures, arriving by a different door. When cache is off, duplicates are sent as duplicates and
        the backend is allowed to disagree with itself -- which is the point of sampling.

        Returns a list of answers positionally aligned with `prompts`. Order is preserved even though the
        backend sees a shorter, deduplicated list."""
        prompts = ["" if p is None else str(p) for p in prompts]
        out = [None] * len(prompts)
        pending = []                                   # unique prompts the backend must actually see
        slot = {}                                      # prompt -> index in `pending` (dedup, when legal)
        # POSITION -> pending index, tracked explicitly. The first draft mapped back by VALUE
        # (pending.index(text)) and that silently collapsed duplicates even with dedup off, handing three
        # sampled branches one shared answer -- the exact false consensus this design forbids. Value lookup
        # cannot express "three separate asks that happen to read the same"; a position list can.
        where = [None] * len(prompts)
        for i, text in enumerate(prompts):
            if self.cache_on:
                k = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if k in self._store:
                    self.hits += 1
                    out[i] = self._store[k]
                    continue
                if text in slot:                       # dedup only under the purity the cache already assumes
                    self.deduped += 1                  # counted, or the saving is invisible in the report
                    where[i] = slot[text]
                    continue
                slot[text] = len(pending)
            where[i] = len(pending)
            pending.append(text)
        if not pending:
            return out
        if self.batch_fn is None:
            answers = [self(t) for t in pending]       # transparent fallback: identical results, N round trips
        else:
            if self.budget is not None and self.calls + len(pending) > self.budget:
                raise BudgetExceeded("%s: batch of %d would exceed the budget of %d (%d already spent)"
                                     % (self.name, len(pending), self.budget, self.calls))
            t0 = time.perf_counter()
            try:
                answers = list(self.batch_fn(pending))
            except Exception:
                self.errors += 1
                self.seconds += time.perf_counter() - t0
                raise
            self.seconds += time.perf_counter() - t0
            self.round_trips += 1
            if len(answers) != len(pending):
                raise ValueError("%s: batch_fn returned %d answers for %d prompts -- a batch backend that "
                                 "drops or reorders results would silently mis-attribute every answer"
                                 % (self.name, len(answers), len(pending)))
            self.calls += len(pending)
            for t, a in zip(pending, answers):
                self.chars_in += len(t)
                self.chars_out += len(a or "")
                if self.cache_on:
                    self._store[hashlib.sha256(t.encode("utf-8")).hexdigest()] = a
        for i in range(len(prompts)):
            if out[i] is None:
                out[i] = answers[where[i]]
        return out

    def _account_prefix(self, text):
        """Longest common prefix with any prompt already sent -- the number an EXACT cache cannot see.

        WHY (FreeToken, arXiv 2608.16157, MIT/UC Berkeley): agent workloads "continuously change their
        execution pattern", and a coding agent rewrites its own history constantly. An exact-match cache
        keyed on the whole prompt scores ZERO on that -- append one token and the sha256 moves -- while
        the overwhelming majority of the text is unchanged. FreeToken's answer is to checkpoint at the
        boundaries agent frameworks cut on so only the NEW part is reprocessed, taking worst-case first
        token from 232s (llama.cpp) / 946s (KTransformers) to under 44s.

        leCore's seam is text->text and cannot reuse a remote model's partial compute. What it CAN do is
        MEASURE the reuse that is available, so `prefix_reuse` says whether pointing at a prefix-caching
        backend (FreeToken, or a provider's prompt cache) is worth anything BEFORE anyone wires one up.
        A hit_rate of 0.000 with a prefix_reuse of 0.95 is a specific, actionable finding; without this
        it reads as 'the cache does not work'."""
        best = 0
        for prev in self._prefixes:
            n = min(len(prev), len(text))
            i = 0
            # cheap bound first: identical heads are the common case, so scan, do not diff
            while i < n and prev[i] == text[i]:
                i += 1
            if i > best:
                best = i
        self._prefixes.append(text)
        if len(self._prefixes) > 512:            # bounded: an agent session, not a corpus
            self._prefixes.pop(0)
        self.prefix_chars_reusable += best
        self.prefix_chars_total += len(text)

    def tell(self, prompt, reply, verdict, detail=None):
        """Close the loop: tell the model what happened to its output.

        THE GAP THIS FILLS, from being the attached model myself. leCore knows the outcome of EVERY model
        call -- did the rewrite route, abstain, or smuggle; did the faithfulness gate accept it; what z
        did the router score -- and every one of those verdicts returns to the CALLER. The model that
        produced the input is never told. It is a one-way seam, and a model that cannot see its own
        error rate cannot correct it, which is why the record-then-replay harness had to score rewrites
        externally and hand-diff them.

        `verdict` is a short label the CALLER owns ("routed" / "abstained" / "smuggled" / "refused"), not
        an enum this module invents -- leCore has several loops with different outcome vocabularies and
        forcing one taxonomy would make the seam wrong for most of them.

        DEFAULT-OFF BY CONSTRUCTION: with no `on_outcome` callback this only RECORDS, so an attached
        model that cannot consume feedback is unaffected. Nothing is pushed at a model that did not ask.

        KEPT NEGATIVE: this is a channel, not a learning rule. It carries the verdict; whether a model
        improves from seeing its own verdicts is an open MEASUREMENT (BENCH-2 with and without), and
        assuming it helps is exactly the kind of unmeasured claim this codebase refuses."""
        v = str(verdict)
        self.outcome_counts[v] = self.outcome_counts.get(v, 0) + 1
        rec = {"prompt": prompt, "reply": reply, "verdict": v, "detail": detail}
        self.outcomes.append(rec)
        if len(self.outcomes) > 256:
            self.outcomes.pop(0)
        if self.on_outcome is not None:
            self.on_outcome(prompt, reply, v, detail)
        return rec

    def feedback_digest(self, k=5):
        """The recent verdicts as plain text, for prepending to a prompt -- the cheapest way to let a
        model see its own error rate without a training loop. Ordered oldest-first so the newest verdict
        reads last, which is where a model weights hardest."""
        if not self.outcomes:
            return ""
        lines = ["previous attempts and how they were judged:"]
        for r in self.outcomes[-int(k):]:
            lines.append("  asked: %s -> you said: %s -> verdict: %s"
                         % (str(r["prompt"])[:90], str(r["reply"])[:60], r["verdict"]))
        return "\n".join(lines)

    def prefix_route(self, upstreams, min_reuse=0.35):
        # DELEGATES to prefix_route_decision below -- the arithmetic exists ONCE, because the
        # same decision now also serves callers holding a raw transcript instead of a seam
        # (openzoo's proxy owns the prompts; a MeteredLLM was never in its path). See F2'/F6.
        rep = self.report()
        return prefix_route_decision(float(rep["prefix_reuse"]), float(rep["hit_rate"]),
                                     upstreams, min_reuse=min_reuse)

    def report(self):
        """Plain data (it crosses an HTTP boundary): counters plus the two derived numbers that matter
        for a hosted router -- hit_rate (what fraction of asks cost nothing) and mean seconds per real
        call. `calls` counts only what reached the model, so calls+hits is the total asked."""
        # ASKED IS EVERY ASK, including the ones dedup collapsed. Reporting only calls+hits would hide
        # exactly the saving this seam exists to create -- the instrument must count what it removed.
        asked = self.calls + self.hits + self.deduped
        saved = self.hits + self.deduped
        return {"name": self.name, "asked": asked, "calls": self.calls, "hits": self.hits,
                "deduped": self.deduped, "saved": saved,
                "hit_rate": (saved / asked) if asked else 0.0,
                "chars_in": self.chars_in, "chars_out": self.chars_out,
                "errors": self.errors, "seconds": round(self.seconds, 6),
                "seconds_per_call": round(self.seconds / self.calls, 6) if self.calls else 0.0,
                "cache": self.cache_on, "budget": self.budget,
                "prefix_reuse": (self.prefix_chars_reusable / self.prefix_chars_total)
                                if self.prefix_chars_total else 0.0,
                "prefix_chars_reusable": self.prefix_chars_reusable,
                "round_trips": self.round_trips, "batch": self.batch_fn is not None,
                "outcomes": dict(self.outcome_counts), "outcomes_recorded": len(self.outcomes)}

    def reset(self):
        """Zero the counters, KEEP the cache. Measuring a second workload should not throw away the
        replay table that makes the second workload cheap -- that would measure the instrument."""
        self.calls = self.hits = self.deduped = self.chars_in = self.chars_out = self.errors = 0
        self.seconds = 0.0
        self.round_trips = 0
        return self


def prefix_route_decision(prefix_reuse, hit_rate, upstreams, min_reuse=0.35):
    """Which upstream should this session go to, given its MEASURED prefix reuse? (F2'/F6)

    `upstreams` is [{"name", "prefix_cache": bool, "cost_per_1k": float}, ...] -- the caller's own
    fleet, because leCore cannot know which providers cache prefixes and guessing would be worse than
    not routing. Module-level ON PURPOSE: the same arithmetic serves BOTH a live MeteredLLM
    (prefix_route above) and a caller holding only a raw transcript (openzoo's proxy owns the
    prompts; a Python seam was never in its request path) -- one decision, two doors.

    THE DECISION THIS MAKES POSSIBLE. `hit_rate` and `prefix_reuse` disagree on exactly the workload
    openzoo runs: MEASURED on a 12-turn coding-agent transcript with one mid-session edit, hit_rate
    0.000 while prefix_reuse was 0.832. A router reading hit_rate alone concludes "caching does not
    help here" and sends the session anywhere; reading BOTH says "send it to an upstream that caches
    prefixes, where 83% of the prompt is already paid for". Neither number decides alone.

    NOT the same thing as `unicron_prefix_cache`, which is a RADIX TREE OVER TOKENS for a LOCAL
    runtime -- it reuses computation we perform ourselves. This routes to a REMOTE backend that will
    do that reuse for us. Different layers; both real; audited so the next session does not merge them.

    KEPT NEGATIVE: `saving_estimate` is an UPPER BOUND. It assumes a backend charges nothing for a
    cached prefix, and real providers discount rather than exempt. Treat it as "is this worth routing
    for", never as a bill."""
    reuse = float(prefix_reuse)
    exact = float(hit_rate)
    ups = list(upstreams or [])
    if not ups:
        return {"choice": None, "why": "no upstreams offered", "prefix_reuse": reuse}
    cachers = [u for u in ups if u.get("prefix_cache")]
    cheapest = min(ups, key=lambda u: float(u.get("cost_per_1k", 0.0)))
    if reuse < min_reuse or not cachers:
        why = ("reuse %.3f below the %.2f floor" % (reuse, min_reuse) if reuse < min_reuse
               else "no upstream advertises prefix caching")
        return {"choice": cheapest["name"], "why": why + " -- routing on price",
                "prefix_reuse": reuse, "hit_rate": exact, "saving_estimate": 0.0}
    # Among prefix-cachers, the effective price is what you pay for the NEW text only.
    best = min(cachers, key=lambda u: float(u.get("cost_per_1k", 0.0)) * (1.0 - reuse))
    eff = float(best.get("cost_per_1k", 0.0)) * (1.0 - reuse)
    base = float(cheapest.get("cost_per_1k", 0.0))
    return {"choice": best["name"],
            "why": "prefix_reuse %.3f with hit_rate %.3f -- an exact cache is blind here"
                   % (reuse, exact),
            "prefix_reuse": reuse, "hit_rate": exact,
            "effective_cost_per_1k": round(eff, 6),
            "saving_estimate": round(max(0.0, (base - eff) / base), 4) if base > 0 else 0.0}


def prefix_reuse_of(prompts, window=512):
    """Measure prefix reuse over a TRANSCRIPT the caller already holds -- the stateless twin of
    MeteredLLM's live accounting, and BIT-IDENTICAL to it by construction (pinned in _selftest):
    both walk the same longest-identical-head scan over the same bounded window, so a gateway
    replaying its request log gets the number the seam would have measured.

    `prompts` is the list of prompt strings IN SEND ORDER (a proxy serializes each request body
    however it likes -- what matters is that the serialization is stable across turns, because
    reuse is measured on the text as sent). Returns {"prefix_reuse", "chars_total",
    "chars_reusable", "turns", "per_turn": [reusable chars per prompt]} -- per_turn so a gateway
    can see WHERE the reuse collapsed (a mid-session edit shows as one small entry).

    WHY THIS EXISTS: openzoo's proxy holds the prompts; a Python MeteredLLM was never in its
    request path. Meeting that workflow means the measurement must come to the transcript."""
    hist, per_turn = [], []
    reusable = total = 0
    for p in prompts or []:
        text = "" if p is None else str(p)
        best = 0
        for prev in hist:
            n = min(len(prev), len(text))
            i = 0
            while i < n and prev[i] == text[i]:
                i += 1
            if i > best:
                best = i
        hist.append(text)
        if len(hist) > int(window):              # same bound as the seam: a session, not a corpus
            hist.pop(0)
        per_turn.append(best)
        reusable += best
        total += len(text)
    return {"prefix_reuse": (reusable / total) if total else 0.0,
            "chars_total": total, "chars_reusable": reusable,
            "turns": len(per_turn), "per_turn": per_turn}


def _selftest_batch():
    """The batch path's contract: same answers as sequential, fewer round trips, savings VISIBLE."""
    rt = {"n": 0, "sizes": []}

    def one(p):
        return "A:" + p[:4]

    def many(ts):
        rt["n"] += 1
        rt["sizes"].append(len(ts))
        return ["A:" + t[:4] for t in ts]

    # A. EQUIVALENCE IS THE WHOLE LICENCE. batch() must return exactly what N calls to __call__ return,
    #    in order, or every consumer that switches to it silently changes answers.
    prompts = ["x", "y", "x", "z", "y", "x"]
    seq = [MeteredLLM(one)(p) for p in prompts]
    b = MeteredLLM(one, cache=True, batch_fn=many)
    assert b.batch(prompts) == seq, "batch answers diverge from sequential"

    # B. THE THREE FILTERS COMPOSE: 6 asks -> 3 unique -> 1 round trip, and the report SHOWS the saving.
    r = b.report()
    assert r["asked"] == 6 and r["calls"] == 3 and r["deduped"] == 3, r
    assert r["round_trips"] == 1 and rt["n"] == 1 and rt["sizes"] == [3], (r, rt)
    assert abs(r["hit_rate"] - 0.5) < 1e-9, r

    # C. REPLAY spans calls: a second batch of the same prompts must reach the backend ZERO times.
    assert b.batch(prompts) == seq
    assert rt["n"] == 1, "a second batch hit the backend despite a warm cache"
    assert b.report()["hits"] == 6 - 0 - 3 + 3, b.report()

    # D. NO BATCH BACKEND -> transparent fallback. Same answers, sequential round trips, no crash. This is
    #    what lets attach_llm turn batching on for everyone without knowing what they attached.
    f = MeteredLLM(one)
    assert f.batch(prompts) == seq
    assert f.report()["round_trips"] == 0 and f.report()["batch"] is False

    # E. KEPT NEGATIVE, DEMONSTRATED: dedup is tied to `cache` because collapsing identical prompts is only
    #    sound for a PURE function. With cache OFF a sampler must be allowed to disagree with itself --
    #    three branches asking the same thing is a fan-out, not waste.
    box = {"i": 0}
    def sampler(p):
        box["i"] += 1
        return "branch%d" % box["i"]
    live = MeteredLLM(sampler)
    assert len(set(live.batch(["q", "q", "q"]))) == 3, "dedup collapsed a sampler's branches with cache OFF"
    box["i"] = 0
    dead = MeteredLLM(sampler, cache=True, batch_fn=lambda ts: [sampler(t) for t in ts])
    assert len(set(dead.batch(["q", "q", "q"]))) == 1, \
        "dedup no longer collapses a cached sampler -- the negative is stale, REWRITE it"

    # F. A BACKEND THAT DROPS OR REORDERS RESULTS MUST FAIL LOUDLY. Silent mis-alignment would attribute
    #    every answer to the wrong prompt, which is worse than an outage because it looks like it worked.
    bad = MeteredLLM(one, cache=True, batch_fn=lambda ts: ["only-one"])
    try:
        bad.batch(["a", "b"])
        raise AssertionError("a short batch reply was accepted")
    except ValueError as e:
        assert "2 prompts" in str(e)
    print("holographic_llmseam batch OK -- equals sequential; 6 asks -> 3 unique -> 1 round trip; replay "
          "spans calls; transparent without a batch backend; dedup collapses a cached sampler (why it is "
          "tied to cache); short batch replies rejected")


def _selftest():
    # STATELESS TWIN, pinned bit-identical to the live seam: same transcript through
    # MeteredLLM's accounting and through prefix_reuse_of must agree on BOTH the ratio and the
    # raw reusable-char count -- if they ever drift, a gateway replaying its log measures a
    # different workload than the seam it is standing in for.
    _tp = ["S: a\nU: fix bug\n", "S: a\nU: fix bug\nA: reading\n",
           "S: a\nU: rename fn\n", "S: a\nU: rename fn\nA: done\nU: test\n"]
    _sw = prefix_reuse_of(_tp)
    _lm = MeteredLLM(lambda p: "ok")
    for _p in _tp:
        _lm(_p)
    _rp = _lm.report()
    assert (_rp["prefix_reuse"] == _sw["prefix_reuse"] and
            _rp["prefix_chars_reusable"] == _sw["chars_reusable"]), (_rp, _sw)
    # and the decision built on it routes a high-reuse transcript to the prefix-cacher even
    # when a cheaper plain upstream exists -- the whole point of measuring both numbers
    _d = prefix_route_decision(0.832, 0.0, [
        {"name": "cacher", "prefix_cache": True, "cost_per_1k": 1.2},
        {"name": "plain", "prefix_cache": False, "cost_per_1k": 1.0}])
    assert _d["choice"] == "cacher" and _d["effective_cost_per_1k"] < 1.0, _d
    # KEPT NEG pinned: below the reuse floor the decision falls back to PRICE, loudly
    _d0 = prefix_route_decision(0.1, 0.0, [
        {"name": "cacher", "prefix_cache": True, "cost_per_1k": 1.2},
        {"name": "plain", "prefix_cache": False, "cost_per_1k": 1.0}])
    assert _d0["choice"] == "plain" and "floor" in _d0["why"], _d0

    seen = []

    def fake(p):
        seen.append(p)
        return "answer:" + p[:6]

    # 1. TRANSPARENT BY DEFAULT: no cache, no budget -- identical behaviour to the bare callable, which
    #    is what makes wrapping every existing consumer a legal change.
    m = MeteredLLM(fake)
    assert m("hello") == "answer:hello"
    assert m("alpha") == "answer:alpha"
    assert m("alpha") == "answer:alpha"
    r = m.report()
    assert r["calls"] == 3 and r["hits"] == 0, r          # uncached: every ask reaches the model
    assert r["hit_rate"] == 0.0

    # 2. EXACT REPLAY when cached, and the counters must separate served-from-replay from reached-model,
    #    or the hit_rate a hosted router bills on is a fiction.
    n = len(seen)
    c = MeteredLLM(fake, cache=True)
    assert c("beta") == "answer:beta"
    assert c("beta") == "answer:beta"
    assert c("beta") == "answer:beta"
    assert len(seen) == n + 1, "cache did not prevent the repeat calls"
    rc = c.report()
    assert rc["calls"] == 1 and rc["hits"] == 2 and abs(rc["hit_rate"] - 2 / 3) < 1e-9, rc

    # 3. THE KEPT NEGATIVE, DEMONSTRATED not warned: caching a SAMPLING model manufactures a false
    #    consensus. Three branches that would have disagreed return one answer, and a swarm reading
    #    agreement as evidence would be reading the cache. This is why cache defaults to False.
    box = {"i": 0}
    def sampler(p):
        box["i"] += 1
        return "branch%d" % box["i"]
    live = MeteredLLM(sampler)
    assert len({live("q"), live("q"), live("q")}) == 3, "the fixture is not actually a sampler"
    box["i"] = 0
    cached = MeteredLLM(sampler, cache=True)
    assert len({cached("q"), cached("q"), cached("q")}) == 1, \
        "caching a sampler no longer collapses its branches -- the negative is stale, REWRITE it"

    # 4. BUDGET FAILS CLOSED. A budget that degrades to 'call anyway' is not a budget.
    b = MeteredLLM(fake, budget=2)
    b("one"); b("two")
    try:
        b("three")
        raise AssertionError("budget did not stop the third call")
    except BudgetExceeded as e:
        assert "budget of 2" in str(e)

    # 5. ERRORS ARE MEASURED, NOT EATEN -- a seam that swallows failures hides the thing you attached it
    #    to find, and AgentBridge already has its own error topic for the swallow-and-publish policy.
    def boom(p):
        raise ValueError("model down")
    e = MeteredLLM(boom)
    try:
        e("x")
    except ValueError:
        pass
    assert e.report()["errors"] == 1 and e.report()["calls"] == 0

    # 5b. PREFIX REUSE: the number an EXACT cache is blind to. An agent's growing transcript scores
    #     hit_rate 0 (every prompt differs) while nearly all of its text is unchanged -- and those two
    #     facts together are what says "point at a prefix-caching backend", which neither says alone.
    grow = MeteredLLM(fake)
    convo = "SYSTEM: you are a coding agent.\n"
    for turn in range(6):
        convo += "USER: step %d please\nASSISTANT: done %d\n" % (turn, turn)
        grow(convo)
    rg = grow.report()
    assert rg["hits"] == 0, "the fixture is not actually a growing transcript"
    assert rg["prefix_reuse"] > 0.6, ("a growing transcript must show high prefix reuse, got %.3f"
                                      % rg["prefix_reuse"])
    #     ...and an unrelated set of prompts must NOT, or the metric is measuring nothing.
    uncorr = MeteredLLM(fake)
    for w in ("alpha one", "bravo two", "charlie three", "delta four"):
        uncorr(w)
    assert uncorr.report()["prefix_reuse"] < 0.25, uncorr.report()["prefix_reuse"]

    # 5c. OUTCOME FEEDBACK (F1): the seam was ONE-WAY -- leCore knew every verdict and told the model
    #     nothing. tell() records, and calls back only if a callback was supplied, so a model that cannot
    #     consume feedback is unaffected. Verdict labels belong to the CALLER: several loops here have
    #     different outcome vocabularies and one enum would be wrong for most of them.
    seen_back = []
    fb = MeteredLLM(fake, on_outcome=lambda p, r, v, d: seen_back.append((p, v)))
    fb("q1"); fb.tell("q1", "a1", "routed")
    fb("q2"); fb.tell("q2", "a2", "smuggled", detail={"z": -0.4})
    assert seen_back == [("q1", "routed"), ("q2", "smuggled")], seen_back
    assert fb.report()["outcomes"] == {"routed": 1, "smuggled": 1}, fb.report()["outcomes"]
    #     ...and with NO callback it must still record and must NOT raise -- default-off means inert,
    #     not absent, or the metering half is lost too.
    quiet = MeteredLLM(fake)
    quiet("q"); quiet.tell("q", "a", "abstained")
    assert quiet.report()["outcomes"] == {"abstained": 1}
    dig = quiet.feedback_digest()
    assert "verdict: abstained" in dig and dig.startswith("previous attempts")
    assert MeteredLLM(fake).feedback_digest() == "", "an empty history must yield an empty digest"

    # 5d. PREFIX ROUTING (F2'/F6): the decision hit_rate alone cannot make. A growing transcript scores
    #     hit_rate 0 with high prefix_reuse, and only BOTH numbers say "route to a prefix-caching upstream".
    UP = [{"name": "plain", "prefix_cache": False, "cost_per_1k": 1.0},
          {"name": "cacher", "prefix_cache": True, "cost_per_1k": 1.2}]
    g2 = MeteredLLM(fake)
    conv = "SYSTEM: agent.\n"
    for t in range(6):
        conv += "USER: step %d\nTOOL: %s\nASSISTANT: ok %d\n" % (t, "z" * 120, t)
        g2(conv)
    d = g2.prefix_route(UP)
    assert g2.report()["hit_rate"] == 0.0, "the fixture must defeat the exact cache"
    assert d["choice"] == "cacher", d
    assert d["saving_estimate"] > 0.0, d
    #     ...and an UNCORRELATED workload must route on PRICE, or the router is just always picking the
    #     cacher and the measurement is decorative.
    g3 = MeteredLLM(fake)
    for w in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot"):
        g3(w * 30)
    d3 = g3.prefix_route(UP)
    assert d3["choice"] == "plain" and d3["saving_estimate"] == 0.0, d3
    #     ...and with NO prefix-caching upstream on offer it must fall back rather than invent one.
    assert g2.prefix_route([UP[0]])["choice"] == "plain"

    # 6. DETERMINISTIC KEYS across processes: sha256, never hash().
    k1 = hashlib.sha256("same".encode()).hexdigest()
    c2 = MeteredLLM(fake, cache=True); c2("same")
    assert k1 in c2._store, "cache key is not the sha256 content hash"

    # 7. reset() keeps the replay table -- zeroing it would measure the instrument, not the workload.
    c2.reset()
    assert c2.report()["calls"] == 0 and c2._store, "reset threw away the cache"
    print("holographic_llmseam selftest OK -- transparent by default; exact replay (hit_rate 2/3); "
          "caching a SAMPLER collapses 3 branches to 1 (why cache is off by default); budget fails "
          "closed; errors measured not eaten; sha256 keys; reset keeps the table")


if __name__ == "__main__":
    _selftest_batch()
    _selftest()
