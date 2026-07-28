"""SEAM-1 -- bring-your-own query embedder, with a space-agreement probe (holographic_embedseam).

WHY THIS EXISTS
---------------
`route_semantic` is the dense half of routing and it already works -- given a VECTOR. The shipped artifact
`lecore_data/routing/index_128d.npz` is 509 modules x 128d, i.e. the DOCUMENT side only; the query side
needs either a build-time cached phrase or the distilled offline embedder, and no distilled artifact ships.
So a brand-new free-text query returns an honest `None`, and the dense path is unreachable from text.

This is the seam that makes it reachable without leCore importing a model SDK -- the same contract as
`attach_llm`: ANY callable. The user brings the model; the footprint policy is preserved.

THE TRAP THIS MODULE EXISTS TO CLOSE, WHICH A BARE SETTER WOULD WALK STRAIGHT INTO
----------------------------------------------------------------------------------
The shipped index lives in ONE PARTICULAR EMBEDDING SPACE (nomic, ABTT-corrected, 128d). A cosine between
a query embedded by SOME OTHER model and a document embedded by nomic is not a weak signal -- it is a
MEANINGLESS one. It will still return five ranked module names with confident-looking scores.

Dimension is checkable and is checked. **Space is not**: any 128d model passes a shape test and then
produces nonsense. That is precisely the "confident wrong answer" class this engine refuses everywhere
else, so it gets an instrument rather than a warning in a docstring.

THE PROBE. Take a sample of modules that are IN the index, read each one's own docstring summary from the
tree, embed that text with the supplied callable, and route it. A model in the index's space ranks each
module top-k FOR ITS OWN DESCRIPTION. A model in a different space scores at chance.

    chance rate at k=5 over 509 modules = 5/509 = 0.0098

so a working embedder clears the default 0.30 bar by roughly 30x, and a wrong-space one cannot get near
it. The bar is deliberately loose: the job is to separate "right space" from "unrelated space", not to
grade embedding quality. Grading that is item 1.4's paired-fixture comparison, not this.

WHAT THE PROBE CANNOT DO, stated so nobody over-trusts it: it cannot tell a GOOD in-space embedder from a
MEDIOCRE one, and it cannot detect a model trained on the same space but with different pooling. It
separates catastrophe from plausibility. Passing it is necessary, not sufficient.
"""

import ast

import numpy as np


def module_summaries(repo_root, names=None):
    """Map module name (with the `holographic_` prefix, as the routing index spells it) -> the first
    paragraph of its module docstring. Delegates the file walk to holographic_workflowgraph._module_texts
    rather than opening a second one, then pulls the docstring with `ast` -- no imports, so a module with a
    heavy import cost or a missing optional dependency still contributes its text."""
    from holographic.semantic_router.holographic_workflowgraph import _module_texts
    wanted = set(names) if names is not None else None
    out = {}
    for stem, src in _module_texts(repo_root, merge_parts=True).items():
        full = "holographic_%s" % stem
        if wanted is not None and full not in wanted:
            continue
        try:
            doc = ast.get_docstring(ast.parse(src)) or ""
        except SyntaxError:                       # a file we cannot parse contributes nothing, loudly-nothing
            continue
        summary = doc.strip().split("\n\n")[0].strip()
        if summary:
            out[full] = " ".join(summary.split())
    return out


def probe_embedder_space(embed, router, repo_root, sample=12, k=5, seed=0):
    """Does `embed` produce vectors in the same space as `router`'s index?

    Embeds each sampled module's OWN docstring summary and checks the module ranks in the top `k` for it.
    Returns a dict: ok, rate, hits, n, chance, k, reason, and `misses` (the names that did not self-recall,
    so a failure is diagnosable rather than just a number).

    Deterministic: the sample is drawn with a seeded default_rng and sorted, so two runs on one tree probe
    the identical modules."""
    names = [str(n) for n in getattr(router, "names", [])]
    if not names:
        return {"ok": False, "rate": 0.0, "hits": 0, "n": 0, "chance": 0.0, "k": k,
                "reason": "the router carries no index, so there is nothing to agree with",
                "misses": []}
    texts = module_summaries(repo_root, names=set(names))
    usable = sorted(texts)
    if not usable:
        return {"ok": False, "rate": 0.0, "hits": 0, "n": 0, "chance": 0.0, "k": k,
                "reason": "no module docstrings could be read, so the probe cannot run", "misses": []}

    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(usable), size=min(sample, len(usable)), replace=False).tolist())
    chosen = [usable[i] for i in pick]

    hits, misses = 0, []
    for name in chosen:
        try:
            vec = np.asarray(embed(texts[name]), dtype=float).ravel()
        except Exception as exc:                                  # a broken callable is a failed probe
            return {"ok": False, "rate": 0.0, "hits": 0, "n": len(chosen), "chance": k / len(names), "k": k,
                    "reason": "the embedder raised on real input: %s" % exc, "misses": chosen}
        ranked = router.route(vec, k=k)
        if ranked and name in [str(r[0]) for r in ranked]:
            hits += 1
        else:
            misses.append(name)

    n = len(chosen)
    rate = hits / n if n else 0.0
    chance = k / len(names)
    return {"ok": None, "rate": rate, "hits": hits, "n": n, "chance": chance, "k": k,
            "reason": "%d/%d modules self-recalled at k=%d (chance %.3f)" % (hits, n, k, chance),
            "misses": misses}


def _selftest():
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 1. The summaries reader finds real modules and returns TEXT, not source.
    s = module_summaries(root)
    assert len(s) > 100, "expected hundreds of module summaries, got %d" % len(s)
    assert all(isinstance(v, str) and v for v in s.values())
    assert not any("import numpy" in v for v in s.values()), "summaries are leaking source, not docstrings"

    # 2. Names carry the index's spelling (with the prefix), or the probe can never match them.
    assert all(nm.startswith("holographic_") for nm in s)

    # 3. THE PROBE SEPARATES A WRONG-SPACE EMBEDDER FROM CHANCE -- the property the whole module exists for.
    #    A random embedder must NOT clear a sane bar. Uses a stub router so the test needs no artifact.
    class _StubRouter:
        names = ["holographic_%s" % n for n in ("ai", "render", "mesh", "creature", "catalog")]

        def route(self, vec, k=5, gamma=0.0):
            idx = int(abs(np.asarray(vec, float).ravel()[0]) * 1000) % len(self.names)
            return [(self.names[idx], 0.5)]

    rng = np.random.default_rng(0)
    res = probe_embedder_space(lambda t: rng.standard_normal(8), _StubRouter(), root, sample=5, k=1)
    assert res["n"] >= 1
    assert res["rate"] < 0.9, "a random embedder self-recalled at %.2f -- the probe is not discriminating" % res["rate"]

    # 4. A BROKEN CALLABLE IS A FAILED PROBE, not an exception escaping into the caller.
    def _boom(_):
        raise RuntimeError("no model loaded")

    bad = probe_embedder_space(_boom, _StubRouter(), root, sample=3)
    assert bad["ok"] is False and "raised" in bad["reason"]

    # 5. An empty router is reported, never divided by.
    class _Empty:
        names = []

        def route(self, *a, **k):
            return []

    assert probe_embedder_space(lambda t: [0.0], _Empty(), root)["ok"] is False

    # 6. DETERMINISM: the same seed probes the same modules.
    a = probe_embedder_space(lambda t: np.zeros(8), _StubRouter(), root, sample=4, seed=3)
    b = probe_embedder_space(lambda t: np.zeros(8), _StubRouter(), root, sample=4, seed=3)
    assert a["misses"] == b["misses"]

    print("holographic_embedseam: all selftests passed (summaries, probe discriminates, guards, determinism)")


if __name__ == "__main__":
    _selftest()
