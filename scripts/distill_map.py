#!/usr/bin/env python3
"""distill_map.py -- N31: can a single matrix replace nomic's 12 layers? NumPy + stdlib.

WHERE THIS COMES FROM (measured, rev. 36):
    full encoder (137 MB)      top-1 7/12   top-5 8/12   median rank 1
    SIF token-pool (23 MB)     top-1 4/12   top-5 8/12   median rank 2

The transformer buys TOP-1 PRECISION and nothing else. Twelve layers of attention apply a correction
to the bag-of-token-vectors. **Is that correction LINEAR?**

    encoder(s)  ~=  W . sif(s)        W: 768 x 768

If yes, leCore ships a token table (23 MB q8) plus W (2.3 MB fp32, 0.6 MB q8) and DELETES the
transformer. W is ridge regression -- closed form, seeded, no gradients, no autodiff, constitution-clean.

If no, the encoder stays an OPTIONAL package and we will know exactly what the attention layers buy.

HONEST NOTE ON PRIORS
I tried to bound the answer with a synthetic "encoder" (linear part + tanh interaction). It could not
discriminate: even a 100%-nonlinear target gave ridge R^2 = 0.80 and 100% neighbour recovery, because
tanh(SA)B stays strongly linearly correlated with S. **Third synthetic this program that failed to
separate the hypotheses.** So no prediction is stated here. Only the real data decides.

METHOD (every step closed-form; fit on documents, scored on queries that were never fit)
    S = SIF token-pool vectors for the 413 module docstrings   (from the token table)
    E = the SAME docstrings' encoder vectors                    (from the content-addressed cache)
    W = argmin ||S W - E||^2 + lam ||W||^2                      (ridge; lam swept, chosen on held-out docs)
    score routing with S_query @ W against E_docs.

THE THREE ROWS THAT MATTER
    [ceiling] E_query vs E_docs      -- the 137 MB encoder            7/12 top-1
    [floor]   S_query vs S_docs      -- the 23 MB bag                 4/12 top-1
    [ours]    S_query @ W vs E_docs  -- 23 MB + 2.3 MB, no transformer

USAGE
    python3 distill_map.py                  # paths from lecore_paths.py
"""
import hashlib, json, re
import numpy as np

import lecore_paths as paths          # not `as P`: a local `P = ...` would shadow it (learned the hard way)
import distill_router as dr           # its WordPiece, SIF and accept-sets

try:
    import knowledge_index as ki      # its corpus walk AND its cache-key format -- see below
except ImportError as _e:             # knowledge_index imports nomic_forward at module scope
    raise SystemExit(f"distill_map needs knowledge_index.py and nomic_forward.py in scripts/: {_e}")


# RULE 0, VIOLATED AND CORRECTED. The first version of this file GUESSED how knowledge_index.py builds
# its cache keys and its document text. It got both wrong, and every one of 413 lookups missed:
#   * key text is  f"search_document: {name} -- {body}"  and my `docs` ALREADY contained the name,
#     so I prefixed it twice;
#   * `body` is the first triple-quoted string, WHITESPACE-COLLAPSED, truncated to MAX_CHARS -- I used
#     ast.get_docstring(), which preserves newlines and indentation. Different bytes, different sha256.
# The fix is not a better guess. It is to IMPORT THE MODULE THAT OWNS THE FORMAT.
CACHE_WIRING = "1000.0|12|True|False"    # rope_base|heads|swap_gate|pre_ln -- knowledge_index's defaults


def cache_key(text):
    """Byte-identical to knowledge_index.embed_cached's key. hashlib, never hash()."""
    return hashlib.sha256((CACHE_WIRING + '||' + text).encode()).hexdigest()[:32]


def corpus(repo, max_chars=280):
    """The SAME entries knowledge_index embedded: its collect_code, its MAX_CHARS, its text layout."""
    ki.MAX_CHARS = max_chars
    entries = [e for e in ki.collect_code(repo) if e[0] == 'code']
    names = [name for _, name, _ in entries]
    texts = [f"search_document: {name} -- {body}" for _, name, body in entries]
    bodies = [body for _, _, body in entries]
    return names, texts, bodies


def load_cache():
    if not paths.CACHE.is_file():
        raise SystemExit(f"no embedding cache at {paths.CACHE}; run knowledge_index.py once to build it")
    return json.load(open(paths.CACHE))


def encoder_vec(cache, text):
    k = cache_key(text)
    return np.array(cache[k], dtype=np.float64) if k in cache else None


def ridge(S, E, lam):
    """W = (S'S + lam I)^-1 S'E. Closed form. `solve`, never `inv` -- the conditioning matters here
    because SIF vectors of a 413-document corpus are nowhere near full rank."""
    d = S.shape[1]
    return np.linalg.solve(S.T @ S + lam * np.eye(d), S.T @ E)


def main():
    cache = load_cache()

    # ---- corpus: exactly the entries knowledge_index embedded (its walk, its text, its truncation)
    names, texts, bodies = corpus(str(paths.REPO))
    dr.check_accept_sets(names)
    asks = [q for q, _ in dr.ASKS]
    print(f"  {len(names)} module docstrings | {len(asks)} asks | cache {len(cache)} entries\n")

    # ---- E: encoder vectors, from the cache (the model ran once, at build time)
    E, keep = [], []
    for i, t in enumerate(texts):
        v = encoder_vec(cache, t)
        if v is not None:
            E.append(v); keep.append(i)
    hit = len(keep)
    if hit < 100:
        # Diagnose rather than die. A miss means the TEXT differs, and the text is printable.
        print(f"  CACHE MISS: {hit} of {len(texts)} docstrings found.")
        print(f"  first key text built here:\n    {texts[0][:150]!r}")
        probe = [k for k in list(cache)[:3]]
        print(f"  cache holds {len(cache)} keys, e.g. {probe}")
        print(f"  wiring prefix used: {CACHE_WIRING!r}")
        raise SystemExit("  -> rebuild the cache against THIS repo:  python3 knowledge_index.py "
                         "<weights> <vocab> --repo <repo>   (or check --rope-base/--heads/--swap-gate/--pre-ln)")
    if hit < len(texts):
        # modules added since the cache was built. Fine -- we fit on what the encoder actually saw.
        missing_names = [names[i] for i in range(len(texts)) if i not in set(keep)]
        print(f"  note: {len(texts)-hit} module(s) newer than the cache, excluded from the fit: "
              f"{missing_names[:4]}{'...' if len(missing_names) > 4 else ''}")
    E = np.array(E)
    names_k = [names[i] for i in keep]
    docs_k = [bodies[i] for i in keep]
    Eq = [encoder_vec(cache, f"search_query: {a}") for a in asks]
    missing = [a for a, v in zip(asks, Eq) if v is None]
    if missing:
        raise SystemExit(f"asks not in cache: {missing[:3]} -- run knowledge_index.py on this repo first")
    Eq = np.array(Eq)
    print(f"  encoder vectors: {E.shape} docs, {Eq.shape} queries (cache hit {hit}/{len(texts)})\n")

    # ---- S: SIF token-pool vectors, from the token table alone
    meta, base = dr.read_st(str(paths.nomic_weights()))
    embn = [n for n in meta if re.search(r'(embeddings\.word_embeddings|embed_tokens)\.weight$', n)][0]
    T = dr.load_t(str(paths.nomic_weights()), meta, base, embn).astype(np.float64)
    wp = dr.WordPiece(str(paths.nomic_vocab()))
    import collections
    freq = collections.Counter()
    for d in docs_k:
        freq.update(wp.tokens(d))
    S = dr.sif_vectors(docs_k, wp, T, freq)
    Sq = dr.sif_vectors(asks, wp, T, freq)
    print(f"  SIF vectors: {S.shape} from token table {T.shape}\n")

    # ---- the three rows
    print(f"  {'router':46s} {'bits':>10s}")
    dr.score(E, Eq, names_k, "[ceiling] full encoder (137 MB)")
    dr.score(S, Sq, names_k, "[floor]   SIF token-pool (23 MB q8)")

    # ---- W: ridge, lam chosen on HELD-OUT documents (never on the asks)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(S))
    tr, te = perm[:len(perm) * 3 // 4], perm[len(perm) * 3 // 4:]
    best = (None, -1e9, None)
    for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        W = ridge(S[tr], E[tr], lam)
        pred = S[te] @ W
        r2 = 1 - np.linalg.norm(pred - E[te]) ** 2 / np.linalg.norm(E[te] - E[te].mean(0)) ** 2
        if r2 > best[1]:
            best = (lam, r2, W)
    lam, r2, _ = best
    print(f"\n  ridge: lam* {lam:g} chosen on held-out docs, held-out R^2 {r2:+.3f}")

    W = ridge(S, E, lam)                     # refit on all documents with the chosen lam
    dr.score(E, Sq @ W, names_k, f"[ours]    SIF @ W  (23 MB + {W.size*4/1e6:.1f} MB)")

    # identity, not just variance -- the §18.1 lesson: R^2 and neighbour transfer are different bars
    unit = lambda A: A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-12)
    pred = unit(S[te] @ W)
    sims = pred @ unit(E[te]).T
    top1 = float(np.mean(np.argmax(sims, 1) == np.arange(len(te))))
    print(f"  doc-identity recovery (held-out): the true encoder vector is nearest for {top1:.0%} of docs")

    print(f"\n  bar: [ours] top-1 >= 6/12 and median rank <= 2, at {23 + W.size*4/1e6:.0f} MB total.")
    print(f"  Then the 137 MB transformer becomes a BUILD-TIME tool: embed the corpus once, delete it.")
    print(f"  If [ours] does not beat [floor], the attention layers are doing something a matrix cannot,")
    print(f"  and the encoder stays an optional package. Either answer is worth having.")


if __name__ == '__main__':
    main()
