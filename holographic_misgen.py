"""B1 probe (cross-cutting: MIS-1 multiple-importance combination -> steered text generation). KEPT NO-OP.

THE PROPOSAL (Pharr's seat, the balance heuristic, with his precondition on record): steered_generate keeps the
candidate with the best verifier (coherence) score among the predictor's top-beam, discarding the predictor's
own ranking at the selection step. So -- the reasoning went -- combine the predictor's coupling score and the
verifier's coherence score by the balance heuristic, weighting each by its reliability, instead of letting the
verifier override.

THE MEASURED ANSWER: it is a NO-OP -- the balance-heuristic combination gives results IDENTICAL to verifier-only
selection, and the reason is structural. steered_generate already uses the predictor as the candidate GATE: it
restricts to the predictor's top-`beam` candidates before the verifier picks among them. WITHIN that beam the
coupling scores are nearly flat (they are all the most-probable continuations), so after the softmax that puts
them on a common scale, the predictor's distribution is close to uniform -- and a near-uniform factor cannot move
the argmax of the product. The verifier's preference dominates the combination, so MIS == verifier. The
predictor's information is ALREADY fully spent on gating the candidate set; re-using it as a within-beam weight
is redundant.

MEASURED (a loop-trap corpus -- a frequent 'ping pong' cycle mixed with coherent clauses): the balance combination
matches the verifier EXACTLY on anti-looping (distinct ratio). No improvement, on a clean corpus or a loopy one.

THE LESSON: MIS combines two estimators OVER A COMMON CANDIDATE SET ON A COMMON DENSITY SCALE (Pharr's
precondition). Here the predictor does not estimate over the same set as the verifier -- it FILTERS to its
top-beam first -- so there is nothing left for the balance heuristic to balance. The right place for MIS would be
combining two estimators of the SAME quantity over the SAME support; a gate followed by a re-ranker is not that.

No faculty, no tour line -- the finding is the no-op.
"""

import numpy as np


def _softmax(x):
    x = np.asarray(x, float)
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def _generate(mp, ver, mode, seed_toks, length=20, beam=6, lookback=8):
    """Steered generation with a selectable selection rule: 'predictor' (greedy coupling), 'verifier' (the
    shipped rule -- best coherence among the beam), or 'balance' (MIS: argmax of softmax(coupling) *
    softmax(verifier) / their sum, over the beam)."""
    out = list(seed_toks)
    for _ in range(length):
        Cn, _nextM = mp._matrix()
        if len(Cn) == 0:
            break
        q = mp.context_vector(out[-mp.order:])
        qn = q / (np.linalg.norm(q) + 1e-12)
        coup = Cn @ qn
        order = np.argsort(coup)[::-1][:beam]
        cands, cs, seen = [], [], set()
        for j in order:
            w = mp._next[j]
            if w not in seen:
                seen.add(w)
                cands.append(w)
                cs.append(coup[j])
        if not cands:
            break
        vs = []
        for w in cands:
            try:
                vs.append(ver.structure_score((out + [w])[-lookback:]))
            except Exception:
                vs.append(0.0)
        cs = np.array(cs)
        vs = np.array(vs)
        if mode == "predictor":
            pick = int(np.argmax(cs))
        elif mode == "verifier":
            pick = int(np.argmax(vs))
        else:                                            # balance heuristic (MIS) over the beam
            pp, pv = _softmax(cs), _softmax(vs)
            pick = int(np.argmax(pp * pv / (pp + pv + 1e-12)))
        out.append(cands[pick])
    return out[len(seed_toks):]


def _selftest():
    """CI-fast: records the B1 no-op. On a loop-trap corpus, the MIS balance-heuristic combination matches
    verifier-only EXACTLY on anti-looping -- the predictor is already spent on gating the beam, so there is nothing
    for the balance heuristic to balance."""
    from holographic_meaning_predict import MeaningPredictor
    from holographic_structure import StructureVerifier
    rng = np.random.default_rng(0)
    clauses = ["the cat chased the mouse", "the dog found the ball",
               "the bird watched the worm", "the fox carried the leaf"]
    corpus = []
    for _ in range(300):
        corpus.append((rng.choice(clauses)).split() if rng.random() < 0.5
                      else "ping pong ping pong ping pong".split())   # the greedy loop trap
    stream = [w for s in corpus for w in s]
    mp = MeaningPredictor(dim=512, order=2, seed=0).fit_space(corpus, window=2).fit_transitions(stream)
    ver = StructureVerifier(mp.vocab, mp.M, mp.idx).calibrate(stream, chunk=150, z_floor=2.0)

    def distinct(mode):
        rs = []
        for s in range(14):
            g = _generate(mp, ver, mode, ["ping", "pong"] if s % 2 else ["the", "cat"], length=22)
            if len(g) >= 2:
                rs.append(len(set(g)) / len(g))
        return float(np.mean(rs))

    d_verif = distinct("verifier")
    d_bal = distinct("balance")
    assert abs(d_bal - d_verif) < 1e-12, (d_bal, d_verif)          # MIS == verifier (the no-op)


if __name__ == "__main__":
    _selftest()
    print("holographic_misgen B1 no-op selftest passed (the predictor is spent on gating; MIS == verifier)")
