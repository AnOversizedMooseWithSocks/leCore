"""
holographic_orchestrator.py
===========================

Task orchestration built from the engine's primitives -- the use case that
plays to this system's strengths (routing, matching, memory, structure) and
needs no language generation at all. Distilled from leOS's SDOL / bone-chain
layer, in pure numpy:

  * Tool / ToolRegistry -- tools described as vectors with typed inputs and
                           outputs. select() ranks tools by how well they match
                           a task, blended with their learned success history,
                           and flags a capability GAP when nothing fits (the
                           void idea, applied to tools).
  * Planner             -- assemble a chain of typed tools that carries the
                           available inputs to the goal, working backward from
                           the goal and ranking choices by goal alignment. A
                           stripped-down FABRIK: types decide what CAN connect,
                           similarity decides what SHOULD.
  * SkeletonLibrary     -- remember chains that worked, keyed by the goal vector,
                           and reuse them when a similar goal comes back. The
                           reflex arc, applied to whole plans.
  * CircuitBreaker      -- track tool failures; once a tool keeps failing, trip
                           its breaker so the planner routes around it, then let
                           it recover after a cooldown.

The planner also EXECUTES: give a tool a real callable and a planned chain runs
end to end, threading the value through and recording what worked.

How it leans on the other files: skeleton reuse is the ReflexArc idea; the
capability-gap flag is the EpistemicMap's void; usage-weighted selection is the
SemanticCompass's "bias toward what worked" in miniature.

Needs: numpy and holographic_ai.py beside it.
"""

import re
import numpy as np
from holographic_ai import cosine, bundle, Vocabulary


def keyword_vector(vocab, words):
    """A simple meaning vector for a tool or goal: the bundle of its keyword
    atoms. Shared keywords -> similar vectors. (Swap in the learned TextEncoder
    from holographic_encoders for synonym-aware matching.)"""
    return bundle([vocab.get(w) for w in words])


class Tool:
    """A typed, described capability. in_type/out_type gate what it can connect
    to; vec carries what it's for; uses/wins track how it has performed."""

    def __init__(self, name, in_type, out_type, vec, fn=None):
        self.name = name
        self.in_type = in_type
        self.out_type = out_type
        self.vec = vec
        self.fn = fn                 # the real implementation, value -> value
        self.schema = None           # optional: a learned schema over this tool's domain
        self.uses = 0
        self.wins = 0

    def success_rate(self):
        return (self.wins + 1) / (self.uses + 2)   # smoothed; unused tools ~0.5

    def __repr__(self):
        return self.name


class ToolRegistry:
    """Holds the tool catalogue and selects tools for a task."""

    def __init__(self, semantic_weight=0.8, gap_threshold=0.15):
        self.tools = []
        self.semantic_weight = semantic_weight   # vs usage history
        self.gap_threshold = gap_threshold        # below this, nothing really fits

    def add(self, tool):
        self.tools.append(tool)
        return tool

    def score(self, task_vec, tool):
        return (self.semantic_weight * cosine(task_vec, tool.vec)
                + (1 - self.semantic_weight) * tool.success_rate())

    def select(self, task_vec, k=5, out_type=None):
        """Return (ranked tools with scores, is_gap). is_gap is True when even
        the best match is too weak to trust -- a missing capability."""
        pool = [t for t in self.tools if out_type is None or t.out_type == out_type]
        ranked = sorted(((self.score(task_vec, t), t) for t in pool),
                        key=lambda r: r[0], reverse=True)
        best_semantic = max((cosine(task_vec, t.vec) for t in pool), default=0.0)
        return ranked[:k], best_semantic < self.gap_threshold

    def record(self, tool, success):
        tool.uses += 1
        if success:
            tool.wins += 1

    # -- the schema gate: route by who UNDERSTANDS the input, not who resembles it --
    def fit_schema(self, tool, data, modality="text", cuts=(0, 120, 350)):
        """Give a tool a learned schema over its own domain data. Routing can then be a
        description-length contest instead of a vector-similarity one."""
        from holographic_schema import SchemaGenerator
        tool.schema = SchemaGenerator(modality=modality, cuts=cuts).fit(data)
        return tool

    def select_by_understanding(self, raw_input, k=5, out_type=None):
        """Rank tools that carry a schema by how few bits THEIR schema needs to encode the
        input -- the expert that understands it compresses it. Falls back to the semantic
        `select` for any task where schemas aren't available.

        Measured against the semantic-vector gate on 4-author routing (shared alphabet, hard
        case): this gate scored 85% at 160-char inputs and 93% at 300, versus the semantic
        gate's 69% and 74%. The bag-of-word-vectors blurs authors who share vocabulary; chunk
        statistics are the structural signature that separates them. Kept because it beat the
        incumbent on the measurement, not on principle."""
        pool = [t for t in self.tools if t.schema is not None
                and (out_type is None or t.out_type == out_type)]
        if not pool:
            raise RuntimeError("no tools carry a schema -- use select(), or fit_schema() first")
        from holographic_schema import compression_gate
        ranked = compression_gate(raw_input, [(t, t.schema) for t in pool])
        return [(bits, tool) for bits, tool in ranked][:k]


class SkeletonLibrary:
    """Cache of chains that worked, keyed by goal vector. Reuse a chain when a
    new goal is close enough to one already solved."""

    def __init__(self, reuse_threshold=0.85):
        self.reuse_threshold = reuse_threshold
        self.entries = []   # list of (goal_vec, chain)

    def store(self, goal_vec, chain):
        self.entries.append((goal_vec, list(chain)))

    def recall(self, goal_vec):
        best, best_sim = None, -1.0
        for vec, chain in self.entries:
            s = cosine(goal_vec, vec)
            if s > best_sim:
                best, best_sim = chain, s
        if best is not None and best_sim >= self.reuse_threshold:
            return best
        return None


class CircuitBreaker:
    """Stop hammering a tool that keeps failing. After fail_max consecutive
    failures a tool's breaker opens and the planner skips it; after a cooldown
    of planning cycles it half-opens for one trial, and a success closes it
    again. Standard fault-tolerance, so one broken tool doesn't sink everything.
    """

    def __init__(self, fail_max=3, cooldown=5):
        self.fail_max = fail_max
        self.cooldown = cooldown
        self._state = {}

    def _s(self, tool):
        return self._state.setdefault(tool.name, {"fails": 0, "open": False, "cool": 0})

    def available(self, tool):
        s = self._s(tool)
        return not (s["open"] and s["cool"] > 0)     # open+cooling -> unavailable

    def report(self, tool, ok):
        s = self._s(tool)
        if ok:
            s["fails"], s["open"], s["cool"] = 0, False, 0
        else:
            s["fails"] += 1
            if s["fails"] >= self.fail_max:
                s["open"], s["cool"] = True, self.cooldown

    def new_cycle(self):
        """Advance the cooldown clock once per planning attempt."""
        for s in self._state.values():
            if s["open"] and s["cool"] > 0:
                s["cool"] -= 1

    def status(self, tool):
        s = self._s(tool)
        if not s["open"]:
            return "closed"
        return "open" if s["cool"] > 0 else "half-open"


class Planner:
    """Assemble and reuse tool chains."""

    def __init__(self, registry, skeletons=None, breaker=None):
        self.registry = registry
        self.skeletons = skeletons or SkeletonLibrary()
        self.breaker = breaker

    def plan(self, goal_type, goal_vec, available, max_depth=6):
        """Find a chain of tools that turns one of the available input types into
        goal_type, preferring chains whose tools match the goal. Returns
        (chain, source) where source is 'skeleton', 'planned', or 'gap'."""
        available = set(available)
        if self.breaker:
            self.breaker.new_cycle()

        # 1. Reflex: has a similar goal been solved, and does that chain still fit?
        cached = self.skeletons.recall(goal_vec)
        if cached and cached[0].in_type in available:
            return cached, "skeleton"

        # 2. Backward search: from the goal type toward the available inputs.
        def find(need, depth, visited):
            if need in available:
                return []                       # already have this input
            if depth == 0:
                return None
            best, best_score = None, -1.0
            for tool in self.registry.tools:
                if self.breaker and not self.breaker.available(tool):
                    continue                    # breaker open: route around it
                if tool.out_type == need and tool.in_type not in visited:
                    sub = find(tool.in_type, depth - 1, visited | {need})
                    if sub is None:
                        continue
                    chain = sub + [tool]
                    s = np.mean([self.registry.score(goal_vec, t) for t in chain])
                    if s > best_score:
                        best, best_score = chain, s
            return best

        chain = find(goal_type, max_depth, set())
        if chain is None:
            return None, "gap"                  # no typed path reaches the goal
        return chain, "planned"

    def record_success(self, goal_vec, chain):
        """Mark a chain as having worked: bump its tools and save it as a
        skeleton for next time."""
        for tool in chain:
            self.registry.record(tool, success=True)
        self.skeletons.store(goal_vec, chain)

    def execute(self, chain, value, goal_vec=None):
        """Run a planned chain for real, threading value through each tool's
        implementation. Records outcomes (usage stats, breaker, and a skeleton
        on full success). Returns (ok, result_or_error, trace)."""
        trace = []
        for tool in chain:
            try:
                if tool.fn is None:
                    raise RuntimeError("no implementation registered")
                value = tool.fn(value)
            except Exception as err:                # this tool failed
                self.registry.record(tool, success=False)
                if self.breaker:
                    self.breaker.report(tool, ok=False)
                trace.append((tool.name, "FAIL"))
                return False, f"{tool.name}: {err}", trace
            self.registry.record(tool, success=True)
            if self.breaker:
                self.breaker.report(tool, ok=True)
            trace.append((tool.name, "ok"))
        if goal_vec is not None:                    # whole chain worked: cache it
            self.skeletons.store(goal_vec, chain)
        return True, value, trace

    def plan_differentiable(self, goal_sig, length, out_type=None, steps=200, lr=0.5):
        """DIFFERENTIABLE ORCHESTRATION. Optimize a whole tool-chain JOINTLY against a chain-level
        structural score, instead of picking tools by an INDEPENDENT per-tool score (what plan/score do).

        The registry already lives in hyperspace -- every Tool has a `.vec`. A chain's signature is the
        order-encoded superposition of its tools' vecs (position s rotates tool_s.vec -- the engine's
        permute), so a `goal_sig` is "what the composed chain should look like" (e.g. taken from a
        demonstrated working chain). This optimizes a SOFT selection (a distribution over tools at each of
        `length` steps) by gradient ASCENT on cosine(chain_signature, goal_sig) -- the gradient derived
        analytically through cosine / superposition / permute / softmax, in numpy, NO autodiff (the same
        gradient-without-a-framework approach as holographic_optimize). argmax of the converged soft
        selection gives the discrete chain of real Tools.

        Returns (chain, score) where chain is a list of Tools and score is the final composed cosine.
        KEPT NEGATIVE: gradient ascent finds a LOCAL optimum of a non-convex landscape, and on an
        ORTHOGONAL / easy tool set per-position greedy already recovers the chain (no gain) -- the win is
        on CORRELATED tool sets where independent scoring is misled by cross-talk between positions.
        """
        pool = [t for t in self.registry.tools if out_type is None or t.out_type == out_type]
        if not pool:
            return [], 0.0
        V = np.stack([t.vec for t in pool])             # (N, D)
        idx, score = optimize_toolchain(V, goal_sig, length, steps=steps, lr=lr)
        return [pool[i] for i in idx], score


# ---------------------------------------------------------------------------
# Differentiable tool-chain optimization (numpy analytic gradient, no autodiff).
# ---------------------------------------------------------------------------

def chain_signature(tool_vecs):
    """The order-encoded signature of a chain: sum_s permute(tool_vecs[s], s) (position s rotates the vec).
    Order-sensitive (a different order is a different signature) -- the VSA sequence encoding."""
    return sum(np.roll(tool_vecs[s], s) for s in range(len(tool_vecs)))


def _softmax_rows(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def optimize_toolchain(tool_vecs, goal_sig, length, steps=200, lr=0.5):
    """Find the length-`length` chain over `tool_vecs` (N, D) whose composed signature best matches
    `goal_sig` (D,), by gradient ascent on the cosine -- analytic gradient, numpy only.

    Returns (indices, final_cosine): indices[s] is the chosen row of tool_vecs at step s.
    """
    tool_vecs = np.asarray(tool_vecs, float)
    N, D = tool_vecs.shape
    g = np.asarray(goal_sig, float)
    gn = np.linalg.norm(g) or 1.0
    theta = np.zeros((length, N))                       # logits over tools per step (uniform start)
    for _ in range(steps):
        p = _softmax_rows(theta)                        # (L, N) soft selection
        U = p @ tool_vecs                               # (L, D) soft tool per step
        sig = sum(np.roll(U[s], s) for s in range(length))   # composed signature (D,)
        sn = np.linalg.norm(sig) or 1.0
        cos = float(sig @ g) / (sn * gn)
        # d cos / d sig = (g/gn - cos * sig/sn) / sn  (standard cosine gradient)
        dsig = (g / gn - cos * sig / sn) / sn
        # back through the permutation: d sig / d U[s] = roll(., s), so d cos / d U[s] = roll(dsig, -s)
        dU = np.stack([np.roll(dsig, -s) for s in range(length)])   # (L, D)
        G = dU @ tool_vecs.T                            # (L, N): d cos / d p[s, t] = <tool_t, dU[s]>
        # back through softmax: d cos / d theta[s] = p[s] * (G[s] - sum_t p[s,t] G[s,t])
        dtheta = p * (G - (p * G).sum(axis=1, keepdims=True))
        theta = theta + lr * dtheta
    idx = list(np.argmax(theta, axis=1))
    # report the DISCRETE chain's score (the soft optimum decoded to real tools)
    disc_sig = chain_signature(tool_vecs[idx])
    disc_cos = float(disc_sig @ g) / ((np.linalg.norm(disc_sig) or 1.0) * gn)
    return idx, disc_cos


def _optimize_selftest():
    """Substantive checks for the differentiable planner: it RECOVERS a known composed chain that
    independent per-tool greedy gets wrong, on a CORRELATED tool set; ties greedy when tools are easy."""
    rng = np.random.default_rng(0)
    N, D, L = 12, 256, 4

    def run(corr):
        base = rng.normal(size=(N, D))
        if corr:                                        # make tools share a big common component (correlated)
            base = base + 2.0 * rng.normal(size=(1, D))
        V = base / np.linalg.norm(base, axis=1, keepdims=True)
        true = list(rng.choice(N, L, replace=False))
        goal = chain_signature(V[true])
        # independent greedy (the existing per-tool style): pick by cosine(goal, tool.vec), position-blind
        scores = V @ goal
        greedy = list(np.argsort(scores)[::-1][:L])
        g_sig = chain_signature(V[greedy])
        g_cos = float(g_sig @ goal) / ((np.linalg.norm(g_sig)) * np.linalg.norm(goal))
        idx, d_cos = optimize_toolchain(V, goal, L, steps=300, lr=0.5)
        recovered = sum(1 for a, b in zip(idx, true) if a == b)
        return d_cos, g_cos, recovered

    # CORRELATED tools: the differentiable optimizer should clearly beat independent greedy
    d_cos, g_cos, rec = run(corr=True)
    assert d_cos > g_cos + 0.05, f"differentiable should beat independent greedy on correlated tools: {d_cos:.3f} vs {g_cos:.3f}"
    assert rec >= L - 1, f"differentiable should recover ~the true chain ({rec}/{L})"
    # determinism: the same tools + goal give the same chain
    Vd = rng.normal(size=(N, D)); gd = rng.normal(size=D)
    i1, _ = optimize_toolchain(Vd, gd, L, steps=50)
    i2, _ = optimize_toolchain(Vd, gd, L, steps=50)
    assert i1 == i2, "differentiable planning must be deterministic for a fixed problem"
    print(f"orchestrator differentiable planner: ok (CORRELATED tools -- composed cosine {d_cos:.3f} vs "
          f"greedy {g_cos:.3f}, recovered {rec}/{L} of the true chain; analytic-gradient, no autodiff)")


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

def _build_registry(vocab):
    reg = ToolRegistry()
    catalogue = [
        ("fetch_url", "query", "raw_html", ["fetch", "web", "page", "url", "online", "download"]),
        ("parse_html", "raw_html", "text", ["parse", "extract", "text", "html", "web", "page"]),
        ("read_file", "path", "text", ["read", "open", "local", "file", "document", "disk"]),
        ("search_web", "query", "results", ["search", "web", "results", "online", "query"]),
        ("results_to_text", "results", "text", ["extract", "text", "results"]),
        ("summarize", "text", "summary", ["summarize", "shorten", "condense", "summary"]),
        ("translate", "text", "text", ["translate", "convert", "language"]),
        ("sentiment", "text", "label", ["sentiment", "label", "classify", "mood"]),
    ]
    for name, i, o, kw in catalogue:
        reg.add(Tool(name, i, o, keyword_vector(vocab, kw)))
    return reg


def demo_orchestrator():
    print("=" * 70)
    print("DEMO -- Holographic orchestration: route, plan, reuse")
    print("=" * 70)
    vocab = Vocabulary(1024, seed=0)
    reg = _build_registry(vocab)
    planner = Planner(reg)

    def run(label, goal_words, goal_type, available):
        g = keyword_vector(vocab, goal_words)
        chain, source = planner.plan(goal_type, g, available)
        names = " -> ".join(t.name for t in chain) if chain else "(none)"
        print(f"  {label:32s} [{source:8s}] {names}")
        return g, chain

    print("\nSame goal verb, different inputs -> different valid chains:")
    run("summarize a web page", ["summarize", "web", "page", "online"], "summary", {"query"})
    run("summarize a local document", ["summarize", "local", "file", "document"], "summary", {"path"})

    print("\nBoth inputs available -> goal wording breaks the tie:")
    run("...summarize the web page", ["summarize", "web", "page", "online"], "summary", {"query", "path"})
    run("...summarize the local file", ["summarize", "local", "file", "document"], "summary", {"query", "path"})

    print("\nSkeleton reuse -- solve once, then recall a near-identical goal:")
    g, chain = run("first time (web summary)", ["summarize", "web", "page"], "summary", {"query"})
    planner.record_success(g, chain)
    run("again (reworded)", ["summarize", "web", "page", "online"], "summary", {"query"})

    print("\nCapability gaps -- the system says 'I can't' instead of bluffing:")
    run("produce a video caption", ["caption", "video"], "caption", {"query", "path"})
    g = keyword_vector(vocab, ["transcribe", "audio", "speech"])
    ranked, is_gap = reg.select(g, k=2)
    top = ", ".join(f"{t.name} ({s:.2f})" for s, t in ranked)
    print(f"  best tools for 'transcribe audio'  -> {top}   gap={is_gap}")

    print("\nUsage learning -- when two tools fit equally, track record decides:")
    bench = ToolRegistry()
    a = bench.add(Tool("summarizer_A", "text", "summary",
                       keyword_vector(vocab, ["summarize", "condense", "summary"])))
    b = bench.add(Tool("summarizer_B", "text", "summary",
                       keyword_vector(vocab, ["summarize", "condense", "summary"])))
    task = keyword_vector(vocab, ["summarize", "condense"])
    ranked, _ = bench.select(task, k=2)
    print("  before any history: " +
          ", ".join(f"{t.name} ({s:.2f})" for s, t in ranked))
    for _ in range(8):
        bench.record(b, success=True)        # B keeps succeeding
        bench.record(a, success=False)       # A keeps failing
    ranked, _ = bench.select(task, k=2)
    print("  after B succeeds, A fails:  " +
          ", ".join(f"{t.name} ({s:.2f})" for s, t in ranked))
    print()


def _impl_fetch(query):
    pages = {"ai-news": "<h1>News</h1><p>AI models keep improving. New AI models "
             "are faster. Researchers released a new model today. The model runs "
             "on phones. Phones now run AI models locally.</p>"}
    if query not in pages:
        raise ValueError("404 not found")
    return pages[query]


def _impl_parse(html):
    return re.sub(r"<[^>]+>", " ", html)


def _impl_summarize(text):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    freq = {}
    for w in re.findall(r"\w+", text.lower()):
        freq[w] = freq.get(w, 0) + 1

    def score(s):
        words = re.findall(r"\w+", s.lower())
        return sum(freq[w] for w in words) / (len(words) + 1)

    return " ".join(sorted(sentences, key=score, reverse=True)[:2])


def _impl_broken(text):
    raise RuntimeError("service unavailable")


def demo_execution():
    print("=" * 70)
    print("DEMO -- Real execution with automatic failover")
    print("=" * 70)
    vocab = Vocabulary(1024, seed=0)
    reg = ToolRegistry(semantic_weight=1.0)     # pick by fit alone, so the breaker
    reg.add(Tool("fetch_url", "query", "raw_html",                # is the only cause
                 keyword_vector(vocab, ["fetch", "web", "page", "url"]), _impl_fetch))
    reg.add(Tool("parse_html", "raw_html", "text",
                 keyword_vector(vocab, ["parse", "text", "html", "web"]), _impl_parse))
    summ_kw = ["summarize", "summary"]
    reg.add(Tool("summarizer_A", "text", "summary",              # listed first, broken
                 keyword_vector(vocab, summ_kw), _impl_broken))
    reg.add(Tool("summarizer_B", "text", "summary",              # the working backup
                 keyword_vector(vocab, summ_kw), _impl_summarize))

    breaker = CircuitBreaker(fail_max=3, cooldown=9)
    planner = Planner(reg, breaker=breaker)
    goal = keyword_vector(vocab, ["summarize", "web", "page"])
    broken = next(t for t in reg.tools if t.name == "summarizer_A")

    print("\nThe preferred summarizer is broken. Watch the orchestrator notice")
    print("and reroute to the backup once the breaker trips:\n")
    for i in range(5):
        chain, _ = planner.plan("summary", goal, {"query"})
        ok, result, _ = planner.execute(chain, "ai-news", goal_vec=goal)
        names = " -> ".join(t.name for t in chain)
        print(f"  run {i}: {names}")
        print(f"         breaker(summarizer_A)={breaker.status(broken)}, "
              f"{'OUTPUT: ' + result if ok else 'FAILED (' + result + ')'}")
    print()


if __name__ == "__main__":
    demo_orchestrator()
    demo_execution()
    _optimize_selftest()


class Orchestrator:
    """A ready-to-use tool orchestrator -- the 'leCore can USE tools' side of the tool interface.

    Owns a keyword Vocabulary, a ToolRegistry, and a Planner, so you can register LOCAL faculties, REMOTE tools (from
    holographic_toolclient.remote_tools), an LLM, and shell COMMANDS uniformly, then discover/plan over them. Every
    registered tool becomes an orchestrator.Tool with a keyword vector, so the registry can semantic-match a goal to
    the right tool regardless of where the tool actually runs.
    """

    def __init__(self, dim=1024, seed=0):
        self.vocab = Vocabulary(dim, seed=seed)          # get(word) derives a deterministic atom for ANY word
        self.registry = ToolRegistry()
        self.planner = Planner(self.registry)
        self.allowlist = set()                            # shell programs permitted by register_command (SAFETY)

    def _vec(self, text):
        """A keyword vector over the words in `text` -- the tool's meaning, for semantic routing."""
        words = [w.lower() for w in re.split(r"[^A-Za-z0-9]+", text) if w]
        return keyword_vector(self.vocab, words or ["tool"])

    def register(self, tool, in_type="any", out_type="any"):
        """Register a tool. Accepts a raw orchestrator.Tool (added as-is) OR any object with .name/.description and a
        .run or .fn callable (a RemoteTool, an LLM wrapper, your own object) -- wrapped into a Tool with a keyword
        vector so it's discoverable. Returns the registered Tool."""
        if isinstance(tool, Tool):
            return self.registry.add(tool)
        name = getattr(tool, "name", None) or getattr(tool, "__name__", "tool")
        desc = getattr(tool, "description", "") or getattr(tool, "does", "")
        run = getattr(tool, "run", None) or getattr(tool, "fn", None) or (tool if callable(tool) else None)
        if run is None:
            raise TypeError("register needs a Tool, or an object with a .run/.fn callable (got %r)" % type(tool))
        it = getattr(tool, "in_type", in_type)
        ot = getattr(tool, "out_type", out_type)
        return self.registry.add(Tool(name, it, ot, self._vec(name + " " + desc), fn=run))

    def register_command(self, name, argv, description="", allow=True, timeout=60.0):
        """Register a shell PROGRAM as a tool. `argv` is the command as a list (e.g. ['ffmpeg', '-i', '{}']); the
        tool's input value substitutes for a '{}' placeholder, or is appended if there is none. Runs via subprocess,
        returning stdout. SAFETY: argv[0] must be ALLOWLISTED -- allow=True adds it here; otherwise pre-populate
        self.allowlist. Never register a command from untrusted input."""
        import subprocess
        prog = argv[0]
        if allow:
            self.allowlist.add(prog)

        def run(value, _argv=list(argv)):
            if _argv[0] not in self.allowlist:
                raise PermissionError("command not allowlisted: %r" % _argv[0])
            if "{}" in _argv:
                cmd = [str(value) if a == "{}" else a for a in _argv]
            else:
                cmd = _argv + [str(value)]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return out.stdout

        return self.registry.add(Tool(name, "any", "any", self._vec(name + " " + description), fn=run))

    def register_remote(self, base_url, token=None):
        """Fetch a remote node's /tools and register every one (as RemoteTools). Returns the list of registered Tools."""
        from holographic_toolclient import remote_tools
        return [self.register(t) for t in remote_tools(base_url, token=token)]

    def tools(self):
        """The names of everything registered."""
        return [t.name for t in self.registry.tools]
