"""An EXTERNAL benchmark: the reverse-dictionary task, on the vendored 144,478-word dictionary.

WHY THIS AND NOT MORE SELF-RETRIEVAL. Every retrieval number in this arc came from a corpus built
from this engine's own source, with queries built by sampling a document's own terms. That
measures the system against a fixture I wrote. The reverse-dictionary task is a PUBLISHED one
(Hill et al. 2016; SemEval-2022 CODWOE Track 2): given a definition, name the word. It has
unambiguous labels I did not invent, it is OUT OF DOMAIN relative to source code, and its standard
protocol is median rank over the whole vocabulary plus top-1/10/100 accuracy -- so results are
comparable to published numbers rather than only to themselves.

IT IS ALSO THE REGIME THAT SCORED 0.000 EARLIER. The R3 paraphrase probe failed completely
because a definition shares almost no surface form with the word it defines. This is that failure
made into a benchmark with real labels, which is the honest way to find out how bad it is.
"""
import numpy as np

import holographic.misc.holographic_dictionary as DICT
from holographic.semantic_router.holographic_bm25 import BM25, tokenize


def build(limit=None, min_tokens=3):
    """Two DIFFERENT descriptions per word, so the task is retrieval and not lookup.

    document = synonyms + hypernym chain + usage example -- what the word IS
    query    = the dictionary DEFINITION -- how the word is explained
    The headword is stripped from BOTH. A query that shares its document's surface form makes the
    task trivial: the first version of this benchmark did exactly that and scored 0.990 top-1,
    against ~50-70% top-10 for published neural systems. A benchmark that beats the literature by
    30 points is measuring itself.
    """
    words = list(DICT.words())
    if limit:
        step = max(1, len(words) // limit)
        words = words[::step][:limit]
    docs, queries, keep = [], [], []
    for w in words:
        wt = set(tokenize(w))
        d = DICT.define(w) or ""
        side = []
        for fn in ("synonyms", "hypernym_chain", "example"):
            try:
                v = getattr(DICT, fn)(w)
            except Exception:
                v = None
            if isinstance(v, (list, tuple)):
                side.extend(str(x) for x in v)
            elif v:
                side.append(str(v))
        dt = [t for t in tokenize(" ".join(side)) if t not in wt]
        qt = [t for t in tokenize(d) if t not in wt]
        if len(dt) >= min_tokens and len(qt) >= min_tokens:
            docs.append(" ".join(dt)); queries.append(" ".join(qt)); keep.append(w)
    return keep, docs, queries


if __name__ == "__main__":
    import p01_stage2 as S
    for N in (5000, 20000):
        words, docs, queries = build(limit=N)
        bm = BM25(docs)
        toks = bm.docs_tokens
        rng = np.random.default_rng(0)
        probe = rng.choice(len(words), min(400, len(words)), replace=False)
        ranks, r1, r10, r100, rr1 = [], 0, 0, 0, 0
        for i in probe:
            s = bm.scores(queries[i])
            order = np.argsort(-s)
            pos = int(np.where(order == i)[0][0])
            ranks.append(pos + 1)
            r1 += int(pos == 0); r10 += int(pos < 10); r100 += int(pos < 100)
            top = list(order[:10])
            q = tokenize(queries[i])
            rer = sorted(top, key=lambda d: S.proximity_score(toks[d], q), reverse=True)
            rr1 += int(rer[0] == i)
        n = len(probe)
        print("REVERSE DICTIONARY, vocabulary %d, %d held-out definitions" % (len(words), n))
        print("   median rank %5d   top-1 %.3f   top-10 %.3f   top-100 %.3f   +prox top-1 %.3f"
              % (int(np.median(ranks)), r1 / n, r10 / n, r100 / n, rr1 / n))
        print("   chance top-1 %.5f, chance top-10 %.5f\n"
              % (1.0 / len(words), 10.0 / len(words)))
