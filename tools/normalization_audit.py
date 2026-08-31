"""Audit every text entry point for the DOUBLE-NORMALISATION trap, mechanically.

WHY THIS IS A TOOL AND NOT A SCRIPT. Three bugs in one arc came from handing already-normalised
text to something that normalises again -- `tokenize` is deliberately not idempotent
('settings' -> 'setting' -> 'sett'), so the failure is silent and looks like a small numeric
discrepancy. Two were caught by hand; the third was inside the harness written to catch the
second. A guard on one caller is not a guard, and a lesson written in NOTES is not one either.

THE QUESTION THIS ASKS, and it took a correction to get right. My first version compared
result(raw) with result(normalise(raw)) and reported six of eight entry points as failing --
true and useless, because EVERY normaliser fails that by construction, including
`canonical_terms`, which IS the normaliser. The actionable question is:

    CAN A CALLER HOLDING TOKENS PASS THEM THROUGH SAFELY?

  SAFE    accepts a token LIST, so the join-and-re-tokenise workaround has no reason to exist
  TRAP    string-only AND non-idempotent: a caller holding tokens must join, and the join is
          silently re-normalised into different terms
  REFUSED rejects a list outright -- honest, because the caller finds out immediately
"""
import sys

import numpy as np

sys.path.insert(0, ".")

QUERY = "the settings of these classes"
CORPUS = ["smooth a bumpy surface mesh", "the settings of these classes",
          "a process for meshing curved surfaces", "holographic memory recall from a cue",
          "fluid solver on a periodic torus"]


def _same(a, b):
    try:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=1e-9))
        return a == b
    except Exception:
        return str(a) == str(b)


def audit(verbose=True):
    import lecore
    from holographic.semantic_router.holographic_bm25 import BM25, tokenize
    from holographic.caching_and_storage.holographic_catalog import _tokens as _cat_tokens

    mind = lecore.UnifiedMind(dim=256, seed=0)
    toks = tokenize(QUERY)
    rows = []

    def takes_tokens(name, fn):
        try:
            fn(list(toks))
        except Exception as exc:
            joined = " ".join(toks)
            a = fn(joined)
            b = fn(" ".join(tokenize(joined)))
            rows.append((name, "REFUSED" if _same(a, b) else "TRAP",
                         "list rejected (%s)" % type(exc).__name__))
            return
        rows.append((name, "SAFE", "token list accepted"))

    def string_only(name, fn, own_tokenizer=None):
        """TWO hazards, reported separately because they need different repairs.

        SELF   -- the entry point's OWN normaliser applied twice. If that changes the answer the
                  entry point is non-idempotent and the repair is inside it.
        CROSS  -- text normalised by a DIFFERENT component (here BM25's stemmer). If that changes
                  the answer the repair is ONE NORMALISATION BOUNDARY for the pipeline, not a
                  change to this entry point. Conflating the two sends you to fix the wrong file.
        """
        raw = QUERY
        try:
            base = fn(raw)
            own = fn(" ".join((own_tokenizer or tokenize)(raw)))
            cross = fn(" ".join(tokenize(raw)))
        except Exception as exc:
            rows.append((name, "REFUSED", type(exc).__name__)); return
        self_ok = _same(base, own) if own_tokenizer else True
        cross_ok = _same(base, cross)
        if own_tokenizer and not self_ok:
            rows.append((name, "TRAP", "NON-IDEMPOTENT under its own normaliser"))
        elif not cross_ok:
            rows.append((name, "CROSS", "breaks on text normalised by ANOTHER component"))
        else:
            rows.append((name, "SAFE", ""))

    bm = BM25(CORPUS)
    takes_tokens("BM25.scores", lambda q: bm.scores(q))
    takes_tokens("BM25(docs) ctor", lambda q: BM25([q]).docs_tokens)
    takes_tokens("mind.retrieval_verdict",
                 lambda q: mind.retrieval_verdict(q, [tokenize(c) for c in CORPUS])["mode"])
    takes_tokens("mind.encode_hash",
                 lambda q: mind.encode_hash(q if isinstance(q, list) else tokenize(q)))
    string_only("mind.bm25_rank", lambda q: [int(d) for d, _ in mind.bm25_rank(q, CORPUS)])
    string_only("mind.canonical_terms", lambda q: mind.canonical_terms(q), own_tokenizer=tokenize)
    try:
        idx = mind.build_semantic_index(words=["surface", "mesh", "class", "setting"], dim=128)
        string_only("semantic_index.find", lambda q: [h[0] for h in idx.find(q, k=3)],
                    own_tokenizer=_cat_tokens)
    except Exception as exc:
        rows.append(("semantic_index.find", "UNAVAILABLE", type(exc).__name__))
    for fn in ("route_semantic", "find_capability"):
        if hasattr(mind, fn):
            string_only("mind." + fn, lambda q, _f=fn: str(getattr(mind, _f)(q))[:200],
                        own_tokenizer=_cat_tokens)

    if verbose:
        print("NORMALISATION AUDIT -- can a caller holding TOKENS pass them through safely?")
        print("  %-26s %-9s %s" % ("entry point", "verdict", "note"))
        for name, verdict, note in rows:
            print("  %-26s %-9s %s" % (name, verdict, note))
        traps = [r for r in rows if r[1] == "TRAP"]
        cross = [r for r in rows if r[1] == "CROSS"]
        print("\n  %d TRAP (non-idempotent under its OWN normaliser -- fix the entry point)"
              % len(traps))
        print("  %d CROSS (breaks on text normalised elsewhere -- fix the pipeline's ONE "
              "normalisation boundary, not this file)" % len(cross))
    return rows


#: KNOWN sites, and the reason each is acceptable TODAY. Like skill_lint's budget this MAY SHRINK
#: AND MUST NEVER GROW: a new name appearing here is a new hazard, and the gate fails on it.
_BUDGET = {
    # BY DESIGN: canonical_terms IS the normaliser. Its own selftest pins the non-idempotence in
    # both directions, with a message saying that becoming idempotent is a behaviour change.
    "mind.canonical_terms": "TRAP",
    # CROSS sites: each has its OWN idempotent normaliser and breaks only on text stemmed by BM25.
    # The repair is ONE normalisation boundary for the pipeline (canonical_terms/term_id), not a
    # change to these files -- measured: find_capability returns 3 hits on its own tokens and ZERO
    # on BM25-stemmed ones, because 'process' becomes 'proces'.
    "mind.bm25_rank": "CROSS",
    "semantic_index.find": "CROSS",
    "mind.find_capability": "CROSS",
}


def gate():
    rows = audit(verbose=True)
    new = [(n, v) for n, v, _ in rows if v in ("TRAP", "CROSS") and _BUDGET.get(n) != v]
    stale = [n for n in _BUDGET if not any(r[0] == n and r[1] == _BUDGET[n] for r in rows)]
    print()
    if stale:
        print("  BUDGET ENTRIES NOW CLEAN -- delete them so the budget keeps shrinking: %s"
              % sorted(stale))
    if new:
        print("  NEW HAZARD(S): %s" % new)
        return 1
    print("  0 new normalisation hazards (%d budgeted)." % len(_BUDGET))
    return 0


if __name__ == "__main__":
    sys.exit(gate())
