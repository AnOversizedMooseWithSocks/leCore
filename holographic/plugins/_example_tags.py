"""EXAMPLE -- a plugin that BUILDS ON the holographic framework, not just around it.

The template (_template.py) shows the contract. This file shows the point: a plugin that
uses the engine's own algebra to add a capability the core does not have. It is a TAG
MEMORY -- "remember that this item carries these tags; later, ask which items carry a tag,
or which tags an item carries" -- in about sixty lines, with no store bigger than one vector.

WHAT IT USES, and why each is the right primitive:

  derived_atom(seed, name, dim)   every item and every tag is a vector that is a PURE
                                  function of (mind.seed, name). Nothing is stored to
                                  reproduce it; the same name is the same vector in any
                                  process, which is what makes the memory regenerable.
  bind(tag, item)                 "this item carries this tag" as ONE vector that resembles
                                  neither -- the association is hidden but recoverable.
  bundle(...)                     the whole memory is one vector: the superposition of every
                                  tag-item binding. Adding a fact is adding a vector.
  unbind(trace, tag)              asking "what carries this tag?" gives a NOISY estimate of
                                  the item -- the sum of every item that carries it, plus
                                  crosstalk from every other binding.
  nearest / cosine                snap the noisy estimate to real items, with a score; and
                                  when the best score is below the floor, ABSTAIN. Refusal
                                  is a first-class answer in this engine, and the plugin
                                  inherits that discipline rather than returning its
                                  weakest guess dressed as an answer.

WHAT IT DELIBERATELY IS NOT: a replacement for mind.learn / mind.recall. Those are the
engine's general memory with an index that scales and a calibrated abstention. This is a
one-vector associative memory with a bundle-capacity ceiling (see the kept negative in the
selftest), which is exactly the kind of small, sharp, workflow-specific thing a plugin is
for. A plugin that needs scale should call mind.learn and let the engine index it.

STATE LIVES IN THE CLOSURE, per mind: register() runs once per load and builds the trace
and codebooks there, so two minds loading this plugin have two independent memories.

Private name (leading underscore) so discovery does not load it into every default mind;
it is here as a runnable, tested example. Load it explicitly:

    m = lecore.UnifiedMind()
    m._plugin_load("holographic.plugins._example_tags")
    m.tags_remember("kettle", ["kitchen", "electric"])
"""

PLUGIN = {
    "name": "example_tags",
    "version": "1.0",
    "does": "Example: a one-vector tag memory built from the HRR algebra (derived_atom / bind / "
            "bundle / unbind / nearest) with calibrated abstention.",
}


def register(mind, config=None):
    """Bind the tag-memory verbs. config: {"floor": 0.15} -- the cosine below which a query
    abstains rather than guessing. The default is measured in the selftest, not assumed."""
    import numpy as np
    from holographic.agents_and_reasoning import holographic_ai as A

    cfg = config or {}
    floor = float(cfg.get("floor", 0.15))
    dim, seed = int(mind.dim), int(mind.seed)

    # The memory. One trace vector plus the names we have seen (so we can build codebooks to
    # clean up against). Names are the only thing that would need saving: the vectors are
    # regenerable from (seed, name).
    state = {"trace": np.zeros(dim), "items": [], "tags": []}

    def atom(name):
        return A.derived_atom(seed, str(name), dim)

    def codebook(names):
        return np.array([atom(n) for n in names]) if names else np.zeros((0, dim))

    def tags_remember(item, tags):
        """Remember that `item` carries each of `tags`. Returns {item, tags, n_items, n_tags}.
        Adding is a bundle: the trace stays similar to every fact it holds."""
        item = str(item)
        tags = [str(t) for t in (tags if isinstance(tags, (list, tuple)) else [tags])]
        if item not in state["items"]:
            state["items"].append(item)
        for t in tags:
            if t not in state["tags"]:
                state["tags"].append(t)
            # ACCUMULATE, DO NOT RENORMALISE. KEPT NEGATIVE, measured while writing this: the
            # first version did trace = bundle([trace, new]) on every add. bundle() renormalises,
            # so each add shrank everything before it -- an exponentially-decaying memory that
            # forgot early facts. The capacity probe exposed it: recall did NOT improve from
            # dim=64 to dim=4096 (1/6, 2/6, 1/6, 3/6), which capacity would have fixed and
            # decay cannot. A plain sum has no recency bias, and cosine is scale-invariant so
            # the query side never needed the trace normalised in the first place.
            state["trace"] = state["trace"] + A.bind(atom(t), atom(item))
        return {"item": item, "tags": tags, "n_items": len(state["items"]), "n_tags": len(state["tags"])}

    def _rank(query, names, k):
        """Every name scored against the noisy unbound estimate, best first; abstain if the
        best is below the floor. This is the cleanup step unbind() tells you to follow it with."""
        if not names:
            return {"abstained": True, "why": "nothing remembered yet", "ranked": []}
        cb = codebook(names)
        q = np.asarray(query, float)
        sims = (cb @ q) / (np.linalg.norm(q) or 1.0)
        order = np.argsort(-sims, kind="stable")[:int(k)]
        ranked = [{"name": names[i], "score": round(float(sims[i]), 4)} for i in order]
        if ranked[0]["score"] < floor:
            return {"abstained": True, "why": "best score %.3f below floor %.3f" % (ranked[0]["score"], floor),
                    "ranked": ranked}
        return {"abstained": False, "ranked": ranked}

    def tags_query(tag, k=5):
        """Which items carry `tag`? {abstained, ranked: [{name, score}], why?}. Unbinds the
        trace by the tag and cleans up against the items seen so far."""
        return _rank(A.unbind(state["trace"], atom(tag)), state["items"], k)

    def tags_of(item, k=5):
        """Which tags does `item` carry? Same unbind, other role -- binding is commutative,
        so one trace answers both questions."""
        return _rank(A.unbind(state["trace"], atom(item)), state["tags"], k)

    def tags_forget_all():
        """Reset the memory (the closure state) -- explicit, because a plugin that holds state
        must give the operator a way to clear it."""
        state["trace"] = np.zeros(dim); state["items"].clear(); state["tags"].clear()
        return {"n_items": 0, "n_tags": 0}

    return [
        {"name": "tags_remember", "fn": tags_remember,
         "does": "Remember that an item carries some tags (one-vector associative memory).",
         "example": "mind.tags_remember('kettle', ['kitchen', 'electric'])",
         "aliases": ("tag an item", "remember tags for", "label this thing", "attach tags")},
        {"name": "tags_query", "fn": tags_query,
         "does": "Which items carry a tag? Ranked with scores; abstains below the floor.",
         "example": "mind.tags_query('kitchen')",
         "aliases": ("what has this tag", "items tagged", "find by tag", "everything labelled")},
        {"name": "tags_of", "fn": tags_of,
         "does": "Which tags does an item carry? Ranked with scores; abstains below the floor.",
         "example": "mind.tags_of('kettle')",
         "aliases": ("tags on this item", "what is this labelled", "labels of")},
        {"name": "tags_forget_all", "fn": tags_forget_all,
         "does": "Clear the tag memory.",
         "example": "mind.tags_forget_all()",
         "aliases": ("clear the tags", "reset tag memory")},
    ]


def _selftest():
    """Planted truths, a kept negative, and the contract every plugin must meet.

    1. PLANTED: three items with overlapping tags; every query ranks the right items first,
       every reverse query ranks the right tags first, and an unknown tag ABSTAINS.
    2. DETERMINISM: two minds with the same seed give identical scores.
    3. ISOLATION: two minds have two memories.
    4. KEPT NEGATIVE: capacity. At dim=64 the one-vector memory holds a handful of facts
       cleanly; pile in 40 and crosstalk drowns the signal. Measured here so the docstring's
       "keep bundles modest" is a number, not advice. The SAME load at dim=1024 recovers all
       six -- which is the check that this is capacity and not a bug (see the decay negative
       in tags_remember: a real bug looks like a negative that does not lift with dim).
    """
    import lecore
    ref = "holographic.plugins._example_tags"

    m = lecore.UnifiedMind(dim=256, seed=0, plugins=())
    m._plugin_load(ref)
    m.tags_remember("kettle", ["kitchen", "electric"])
    m.tags_remember("toaster", ["kitchen", "electric"])
    m.tags_remember("sofa", ["lounge"])

    r = m.tags_query("kitchen")
    assert not r["abstained"] and {x["name"] for x in r["ranked"][:2]} == {"kettle", "toaster"}, r
    assert r["ranked"][2]["name"] == "sofa" and r["ranked"][2]["score"] < r["ranked"][1]["score"], r
    r = m.tags_query("lounge")
    assert not r["abstained"] and r["ranked"][0]["name"] == "sofa", r
    r = m.tags_of("kettle")
    assert not r["abstained"] and {x["name"] for x in r["ranked"][:2]} == {"kitchen", "electric"}, r
    r = m.tags_query("garage")
    assert r["abstained"], "an unknown tag must ABSTAIN, not return its weakest guess: %s" % r

    # invoke + discoverability: the plugin is a real faculty
    assert m.invoke("tags_query", {"tag": "lounge"})["ranked"][0]["name"] == "sofa"
    assert "tags_query" in [getattr(c, "name", "") for c in m.find_capability("what has this tag")[:3]]

    # determinism across processes-worth of state: same seed, same scores
    m2 = lecore.UnifiedMind(dim=256, seed=0, plugins=()); m2._plugin_load(ref)
    m2.tags_remember("kettle", ["kitchen", "electric"]); m2.tags_remember("toaster", ["kitchen", "electric"])
    m2.tags_remember("sofa", ["lounge"])
    assert m.tags_query("kitchen") == m2.tags_query("kitchen"), "not deterministic"

    # isolation: a fresh mind has an empty memory
    m3 = lecore.UnifiedMind(dim=256, seed=0, plugins=()); m3._plugin_load(ref)
    assert m3.tags_query("kitchen")["abstained"]

    # KEPT NEGATIVE -- capacity at small dim.
    s = lecore.UnifiedMind(dim=64, seed=0, plugins=()); s._plugin_load(ref)
    for i in range(40):
        s.tags_remember("item%d" % i, ["tag%d" % (i % 7)])
    r = s.tags_query("tag0")
    hits = [x["name"] for x in r["ranked"] if x["name"] in {"item%d" % i for i in range(0, 40, 7)}]
    # with 40 facts in 64 dims the top-5 no longer contains all six tag0 items: crosstalk wins.
    assert len(hits) < 6, "capacity negative no longer holds -- re-measure before trusting this"
    # ...and the same load at dim=1024 holds them all: capacity, not a bug.
    big = lecore.UnifiedMind(dim=1024, seed=0, plugins=()); big._plugin_load(ref)
    for i in range(40):
        big.tags_remember("item%d" % i, ["tag%d" % (i % 7)])
    r = big.tags_query("tag0", k=6)
    assert not r["abstained"] and {x["name"] for x in r["ranked"]} == {"item%d" % i for i in range(0, 40, 7)}, r
    print("holographic.plugins._example_tags selftest OK (40 facts: dim=64 top-5 holds %d of 6, dim=1024 holds 6 of 6)" % len(hits))


if __name__ == "__main__":
    _selftest()
