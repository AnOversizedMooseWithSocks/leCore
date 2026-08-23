"""Make benchmark LEAKAGE visible by construction, instead of catching it by instinct.

Three leaks in one arc, each found only because a number was implausibly good:
  * a reverse-dictionary benchmark scored 0.990 top-1 -- the query WAS the document
  * a semantic-vs-lexical comparison scored 0.803 vs 0.090 one way and 0.233 vs 0.840 the other:
    each arm won when queried with the text IT indexed. Neither number was a retrieval result.
  * a work-ratio measurement used queries sampled from the gold document, hiding 4 orders of
    magnitude in the query distribution.
"Notice that it beats the literature" is not a method. This is.

THE CORE OUTPUT IS A MATRIX: arm x query-source accuracy. The DIAGONAL is each arm queried with
the text it was built from; the OFF-DIAGONAL is held-out. A large diagonal with a small
off-diagonal is the signature of lookup wearing a benchmark's name, and it is obvious in a matrix
and invisible in a single row.
"""
import sys

import numpy as np

sys.path.insert(0, ".")


def leak_matrix(arms, sources, gold, k=1):
    """arms: {name: scorer(query_text) -> score array over the corpus}
    sources: {name: [query_text per gold item]} -- one entry per gold document
    gold: [index of the correct document per query]
    Returns {(arm, source): top-k accuracy}."""
    out = {}
    for aname, score in arms.items():
        for sname, qs in sources.items():
            hit = 0
            for q, g in zip(qs, gold):
                order = np.argsort(-np.asarray(score(q), dtype=float))[:k]
                hit += int(g in order)
            out[(aname, sname)] = hit / max(1, len(gold))
    return out


def report(mat, arms, sources, built_from, verbose=True):
    """built_from: {arm: source} -- which query source each arm INDEXES. That pairing is what
    makes a cell diagonal, and it must be declared by the benchmark author; guessing it is how a
    leak checker becomes decoration."""
    rows = []
    for a in arms:
        diag = mat.get((a, built_from[a]))
        off = [v for (aa, s), v in mat.items() if aa == a and s != built_from[a]]
        best_off = max(off) if off else float("nan")
        ratio = (diag / best_off) if off and best_off > 0 else float("inf")
        rows.append((a, built_from[a], diag, best_off, ratio))
    if verbose:
        print("LEAK MATRIX (rows = arms, columns = query sources; * marks the arm's own text)")
        header = "  %-22s" % "arm" + "".join("%-14s" % s for s in sources)
        print(header)
        for a in arms:
            line = "  %-22s" % a
            for s in sources:
                mark = "*" if built_from[a] == s else " "
                line += "%-14s" % ("%.3f%s" % (mat[(a, s)], mark))
            print(line)
        print("\n  arm                    own-text  best held-out  ratio   verdict")
        for a, src, diag, off, ratio in rows:
            verdict = ("LEAK -- the benchmark is measuring lookup" if ratio > 2.0
                       else "ok" if ratio == ratio else "no held-out source")
            print("  %-22s %-9.3f %-14.3f %-7.2f %s"
                  % (a, diag, off, ratio, verdict))
        print("\n  A ratio above 2x means the arm scores far better on text it indexed than on "
              "anything else,\n  which is lookup, not retrieval. Report the held-out column.")
    return rows


def _selftest():
    """Planted: one arm that only ever matches its own text (a pure leak), one that generalises."""
    rng = np.random.default_rng(0)
    K, D = 60, 32
    A = rng.standard_normal((K, D))          # description 1
    B = rng.standard_normal((K, D))          # description 2 -- INDEPENDENT, not A plus noise.
    # The first plant made B = A + small noise, so every arm matched both sources and the ratio
    # came out 1.00 for a genuinely leaky arm. A held-out source has to be genuinely held out.
    leaky = lambda q: A @ np.asarray(q)                    # indexes A only
    fair = lambda q: (A + B) / 2.0 @ np.asarray(q)         # indexes both
    sources = {"srcA": [A[i] for i in range(K)], "srcB": [B[i] for i in range(K)]}
    arms = {"leaky": leaky, "fair": fair}
    mat = leak_matrix(arms, sources, list(range(K)))
    rows = report(mat, arms, sources, {"leaky": "srcA", "fair": "srcA"}, verbose=False)
    by = {r[0]: r for r in rows}
    assert by["leaky"][2] > 0.9, "the planted leaky arm must ace its own text"
    assert by["leaky"][3] < 0.2, ("the planted leaky arm must FAIL the held-out source -- got "
                                  "%.3f, so the plant is not planting anything" % by["leaky"][3])
    assert by["leaky"][4] > 3.0 * by["fair"][4], (
        "the leak ratio must separate the arms: leaky %.2f vs fair %.2f"
        % (by["leaky"][4], by["fair"][4]))
    print("leak_check self-test passed (planted leak ratio %.2f vs fair %.2f)"
          % (by["leaky"][4], by["fair"][4]))


if __name__ == "__main__":
    _selftest()
