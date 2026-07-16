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
import hashlib, json, re, pathlib
import numpy as np

import lecore_paths as paths          # not `as P`: a local `P = ...` would shadow it (learned the hard way)
import nomic_forward as nf            # read_st / load_t / WordPiece -- the model machinery (was distill_router)

try:
    import knowledge_index as ki      # its corpus walk AND its cache-key format -- see below
except ImportError as _e:             # knowledge_index imports nomic_forward at module scope
    raise SystemExit(f"distill_map needs knowledge_index.py and nomic_forward.py in scripts/: {_e}")


# --- self-contained replacements for the old distill_router dependency ---------------------------------
# distill_router lived only in the local experiments folder; its pieces are provided here so distill_map
# runs from a plain repo checkout. The heavy machinery (safetensors read, WordPiece, token table) is
# nomic_forward's; only SIF pooling, scoring, and the accept-set check were unique to distill_router.

def _pca_whiten_table(T, keep=None):
    """Model2Vec's key move (Tulkens & van Dongen 2024; Wada et al. EMNLP 2025): PCA-NORMALIZE the TOKEN
    TABLE before pooling. Centering + whitening removes the dominant, sentence-semantics-irrelevant
    directions and equalizes variance across axes, so a plain mean of token rows carries meaning instead
    of being dominated by a few high-variance nuisance components. THIS IS THE STEP N31 SKIPPED -- we
    ABTT'd the OUTPUT space and fit W on raw token rows; the literature normalizes the TABLE itself.

    Full-rank by default (keep=None keeps all dims): even without dimensionality reduction, PCA whitening
    helps purely by normalizing the space (their measured finding). Deterministic: centered SVD, no RNG."""
    mu = T.mean(0)
    Tc = T - mu
    # SVD of the centered table; whiten by dividing each principal direction by its singular value.
    U, S, Vt = np.linalg.svd(Tc, full_matrices=False)
    k = keep or len(S)
    comp = Vt[:k]                                   # principal directions (k x d)
    sv = S[:k]
    Tw = (Tc @ comp.T) / (sv / np.sqrt(len(T)) + 1e-8)   # project + whiten (unit-ish variance per axis)
    return Tw, mu, comp, sv


def _zipf_freq(vocab_size):
    """Rank-based frequency proxy (Model2Vec): tokenizers ship the vocab sorted by frequency, so a token's
    RANK approximates its probability via Zipf's law (freq ~ 1/rank). Lets SIF weighting work with NO
    corpus counts -- a fallback/complement to the measured doc-frequency we already compute."""
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    return 1.0 / ranks                              # p(word) ~ 1/rank; SIF then weights a/(a+p)


def _sif_vectors(texts, wp, T, freq, a=1e-3):
    """SIF sentence vectors (Arora/Liang/Ma): frequency-weighted mean of token rows, ABTT applied by the
    caller. T is the token table (V x d); freq is per-token count. Deterministic, no model forward pass."""
    total = float(freq.sum()) or 1.0
    out = []
    for s in texts:
        ids = [i for i in wp.encode(s) if 0 <= i < len(T)]
        if not ids:
            out.append(np.zeros(T.shape[1])); continue
        w = a / (a + freq[ids] / total)
        out.append((w[:, None] * T[ids]).sum(0) / w.sum())
    return np.array(out)


def _unit(A):
    return A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-12)


def _score(D, Q, names, label, asks=None):
    """Rank each ask's accepted module; print top-1 / top-5 / median. Mirrors distill_router.score."""
    asks = asks or ki.ASKS_MODULE
    Dn, Qn = _unit(D), _unit(Q)
    ranks = []
    for i, (a, acc) in enumerate(asks):
        order = np.argsort(-(Dn @ Qn[i]))
        ranks.append(next((r + 1 for r, x in enumerate(order) if names[x] in acc), len(names)))
    t1 = sum(r == 1 for r in ranks); t5 = sum(r <= 5 for r in ranks)
    print(f"  {label:34s} top-1 {t1:2d}/{len(asks)}  top-5 {t5:2d}/{len(asks)}  median {int(np.median(ranks))}")
    return t1, t5, float(np.median(ranks))


def _check_accept_sets(names):
    """Warn if any ask's accepted modules are not in the live corpus (the phantom-accept-set trap that
    silently caps the score). Non-fatal -- prints, so a stale accept-set is visible, not hidden."""
    have = set(names)
    for a, acc in ki.ASKS_MODULE:
        missing = [m for m in acc if m not in have]
        if missing:
            print(f"  NOTE: accept-set for {a!r} has modules not in corpus: {missing}")


class _DR:
    """Namespace shim so the existing dr.* call sites keep working."""
    read_st = staticmethod(nf.read_st)
    load_t = staticmethod(nf.load_t)
    WordPiece = nf.WordPiece
    sif_vectors = staticmethod(_sif_vectors)
    score = staticmethod(_score)
    check_accept_sets = staticmethod(_check_accept_sets)
    ASKS = ki.ASKS_MODULE


dr = _DR()
# ------------------------------------------------------------------------------------------------------


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
    import argparse
    here = pathlib.Path(__file__).resolve().parent          # tools/semantic
    ap = argparse.ArgumentParser()
    # default repo = two levels up (repo root), NOT paths.REPO which hardcodes a 'leCore' sibling name
    # and breaks on any clone that renamed the repo folder (same bug that hit export_index.py).
    ap.add_argument('--repo', default=str(here.parent.parent))
    a = ap.parse_args()
    repo = pathlib.Path(a.repo)
    if not repo.is_dir():
        raise SystemExit(f"  --repo {repo} is not a directory; pass --repo <repo root>.")

    cache = load_cache()

    # ---- corpus: exactly the entries knowledge_index embedded (its walk, its text, its truncation)
    names, texts, bodies = corpus(str(repo))
    if not names:
        raise SystemExit(
            f"  0 module docstrings found under --repo {repo}.\n"
            f"  collect_code looks for holographic_*.py; none were under that path.\n"
            f"  If running from tools/semantic, the repo root is '../..' (the default). Pass --repo if elsewhere.")
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
    # Frequency of each TOKEN ID across the docs, as an array sized to the token table (SIF weights word i
    # by a/(a+freq_i)). wp.encode returns ids (np array); count them into a vocab-sized vector. (The old
    # distill_router had wp.tokens(); nomic_forward's WordPiece exposes encode() -- ids, not strings.)
    freq = np.ones(len(T), dtype=np.float64)          # 1-smoothed so no id has zero weight
    for d in docs_k:
        for i in wp.encode(d):
            if 0 <= i < len(T):
                freq[i] += 1
    S = dr.sif_vectors(docs_k, wp, T, freq)
    Sq = dr.sif_vectors(asks, wp, T, freq)
    print(f"  SIF vectors: {S.shape} from token table {T.shape}\n")

    # ---- the three rows
    print(f"  {'router':46s} {'bits':>10s}")
    dr.score(E, Eq, names_k, "[ceiling] full encoder (137 MB)")
    dr.score(S, Sq, names_k, "[floor]   SIF token-pool (23 MB q8)")

    # ---- N31b: the Model2Vec fix -- PCA-WHITEN THE TOKEN TABLE, then pool. No W, no model. This is the
    # step §34 says we skipped. If [m2v] clears the bar, free-text routing ships with just a whitened
    # token table + rank/SIF weights (a few MB), no ridge map at all.
    Tw, tmu, tcomp, tsv = _pca_whiten_table(T)          # full-rank whiten (dim unchanged)
    Sw = dr.sif_vectors(docs_k, wp, Tw, freq)
    Swq = dr.sif_vectors(asks, wp, Tw, freq)
    dr.score(Sw, Swq, names_k, "[m2v]     whitened-table SIF (no W)")
    # and the same with rank-based Zipf weights instead of measured doc frequency (their no-corpus path)
    zf = _zipf_freq(len(T))
    Sz = dr.sif_vectors(docs_k, wp, Tw, zf)
    Szq = dr.sif_vectors(asks, wp, Tw, zf)
    dr.score(Sz, Szq, names_k, "[m2v+zipf] whitened + rank-SIF (no W)")

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

    # N31b stacked: whitened table AND a ridge W on top. If a linear map still helps AFTER fixing the
    # table, this is the strongest no-model option; if [m2v] alone already clears the bar, ship that
    # (simpler -- no W). Pick lam the same honest way, on held-out docs.
    bestw = (None, -1e9)
    for lam2 in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        Ww = ridge(Sw[tr], E[tr], lam2)
        r2w = 1 - np.linalg.norm(Sw[te] @ Ww - E[te]) ** 2 / np.linalg.norm(E[te] - E[te].mean(0)) ** 2
        if r2w > bestw[1]:
            bestw = (lam2, r2w)
    Ww = ridge(Sw, E, bestw[0])
    print(f"  ridge (whitened): lam* {bestw[0]:g}, held-out R^2 {bestw[1]:+.3f}")
    dr.score(E, Swq @ Ww, names_k, "[m2v+W]   whitened table + ridge W")

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


def export_query_embed(out_path, token_table, vocab, freqs, W, pc=None, sif_a=1e-3):
    """Write the N31 runtime artifact: token table + freqs + ridge W (+ ABTT pc). This is what
    holographic_queryembed.QueryEmbedder loads. Kept separate from the fit so the fit can be audited
    before anything ships. See holographic_queryembed for the load side."""
    import numpy as np
    kw = dict(tokens=np.array(list(vocab)), token_vecs=token_table.astype(np.float16),
              freqs=np.asarray(freqs, dtype=np.float32), W=W.astype(np.float16), sif_a=float(sif_a))
    if pc is not None:
        kw["pc"] = pc.astype(np.float16)
    np.savez(out_path, **kw)
    import os
    print(f"  wrote {out_path} ({os.path.getsize(out_path)/1e6:.2f} MB) -- ship to lecore_data/routing/query_embed.npz")
