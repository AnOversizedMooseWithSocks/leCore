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
