"""A FULLY HELD-OUT retrieval benchmark, built from the repo describing itself.

WHY IT IS NEEDED. Every retrieval figure in this arc came from queries sampled out of the target
document, which the leak checker correctly flags as lookup (ratio 2.63 for the lexical arm). The
only fully-held-out source available so far was dictionary glosses, which reach the 57% of terms
the dictionary defines. This is a better one and it is free:

    DOCUMENT = a module's CODE, with every docstring stripped out
    QUERY    = that module's own module-level DOCSTRING
    GOLD     = the module

The docstring is a human-written description of the code that shares NO text with it once removed,
so nothing about the query is in the index. It is also the real use case in one sentence: find the
code that does what this sentence says. That is the inception shape again -- the repo is the
corpus AND the labels, and neither was written for this benchmark.

UNFRIENDLY BY CONSTRUCTION: prose describing code has almost no vocabulary overlap with the code
(identifiers, keywords, numeric literals), which is exactly the vocabulary-mismatch regime lexical
retrieval is worst at. Expect low numbers, and expect them to be the honest ones.
"""
import glob
import re

import numpy as np

STRIP_DOC = re.compile(r'("""|\x27\x27\x27)(?:.|\n)*?\1')


def build(limit=None, min_doc_tokens=12, min_body_tokens=40):
    import hard_corpus as HC
    files = sorted(glob.glob("holographic/*/holographic_*.py"))
    if limit:
        step = max(1, len(files) // limit)
        files = files[::step][:limit]
    names, bodies, queries = [], [], []
    for path in files:
        try:
            src = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        m = re.search(r'"""((?:.|\n)*?)"""', src)
        if not m:
            continue
        doc = HC.tokens(m.group(1))
        body = HC.tokens(STRIP_DOC.sub(" ", src))     # ALL docstrings removed, not just the first
        name = path.split("/")[-1][:-3]
        # the module's own name leaks the answer through the file header; drop it from the query
        nameparts = set(HC.tokens(name.replace("_", " ")))
        doc = [t for t in doc if t not in nameparts]
        if len(doc) >= min_doc_tokens and len(body) >= min_body_tokens:
            names.append(name); bodies.append(body); queries.append(doc)
    return names, bodies, queries


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "tools")
    import lecore, leak_check as LC
    import holographic.agents_and_reasoning.holographic_retrievalpolicy as RP

    mind = lecore.UnifiedMind(dim=256, seed=0)
    names, bodies, queries = build()
    pol = RP.RetrievalPolicy(bodies, pretokenized=True)
    K = len(names)
    overlap = np.mean([len(set(q) & set(b)) / max(1, len(set(q)))
                       for q, b in zip(queries, bodies)])
    print("FULLY HELD-OUT BENCHMARK: %d modules" % K)
    print("  document = code with docstrings stripped, query = the module docstring")
    print("  mean fraction of query terms present in its own document: %.3f" % overlap)
    print("  (a self-sampled benchmark has 1.000 here by construction)\n")

    vocab = sorted({t for b in bodies for t in b})
    dic = {}
    for w in vocab:
        try:
            e = mind.lookup(w)
        except Exception:
            e = None
        if e and e.get("definition"):
            dic[w] = pol.terms(e["definition"])
    exp = RP.RetrievalPolicy([list(b) + [x for t in set(b) for x in dic.get(t, ())] for b in bodies],
                             pretokenized=True)
    pol.attach_semantic(mind, dim=256)

    rng = np.random.default_rng(0)
    gold = list(range(K))
    self_q = []
    for i in gold:
        u = sorted(set(bodies[i]))
        self_q.append([u[j] for j in rng.choice(len(u), min(8, len(u)), replace=False)])
    arms = {"bm25": lambda q: pol.scores(q),
            "bm25+expansion": lambda q: exp.scores(q),
            "semantic": lambda q: pol.semantic_scores(q)}
    sources = {"docstring (held out)": queries, "body terms (leaky)": self_q}
    for k in (1, 10):
        mat = LC.leak_matrix(arms, sources, gold, k=k)
        print("top-%d accuracy" % k)
        for a in arms:
            print("   %-16s docstring %.3f   body-terms %.3f"
                  % (a, mat[(a, "docstring (held out)")], mat[(a, "body terms (leaky)")]))
    mat = LC.leak_matrix(arms, sources, gold, k=1)
    print()
    LC.report(mat, arms, sources, {a: "body terms (leaky)" for a in arms})
    print("\n  chance top-1 %.5f" % (1.0 / K))
