#!/usr/bin/env python3
"""distill_router.py -- THE FOOTPRINT DECISION. NumPy + stdlib. Needs nomic's weights, nothing else.

THE QUESTION
leCore is small and runs anywhere NumPy runs. Everything we have extracted from nomic is tiny and
static -- catalog vectors (9 KB), the N19 remap (194 KB), the knowledge index (407 KB), the negatives
register (190 KB). Under a megabyte of DATA.

But routing a NEW sentence needs an ENCODER, and today that is nomic's 12-layer transformer: a 137 MB
runtime dependency, for exactly one job -- turning a sentence into a vector.

    Do we need the transformer for that, or only its TOKEN EMBEDDING TABLE?

    full encoder (12 layers)                137.0 MB
    token embeddings, 768d q8                23.4 MB
    token embeddings, 64d PCA q8 (N19)        2.0 MB     <- 70x smaller

This is not a hope. It is Arora, Liang & Ma (ICLR 2017), "A Simple but Tough-to-Beat Baseline for
Sentence Embeddings": weight word vectors by smooth inverse frequency (SIF), average them, then
REMOVE THE FIRST PRINCIPAL COMPONENT. That final step is precisely the all-but-the-top correction we
already measured on nomic's output space (mean|cos| 0.693 -> 0.073). The paper's baseline beat LSTM
sentence encoders. Nobody, to our knowledge, has asked whether it beats a modern encoder's own
transformer *on that encoder's own token table*, for a routing task.

WHAT THIS DECIDES
  If SIF-pooled token vectors route within ~1 hit of the full encoder, leCore ships a 2 MB table and
  NO MODEL AT RUNTIME. The transformer becomes a build-time tool: run once to embed the corpus, then
  delete. That is the difference between "leCore depends on a language model" and "leCore ate one".

  If it does not, we keep the 137 MB encoder as an OPTIONAL, composable package -- never in core --
  and we will know exactly what those 12 layers were buying.

METHOD (all closed-form; no gradients, no learned weights)
  [1] baseline    : the full-encoder document vectors (already cached) and query vectors
  [2] SIF pooling : v(s) = sum_w  a/(a + p(w)) * E[w]   with a = 1e-3, p(w) from a word-frequency
                    count over leCore's own corpus. Then subtract the first principal component,
                    fit ON DOCUMENTS ONLY (never on the held-out queries).
  [3] +N19 rotate : apply the mean/rogue/PCA-64 remap and re-score.
  Report the 12-ask routing suite (top-1, top-5, median rank of 405) for each.

USAGE
  python3 distill_router.py                       # paths resolved by lecore_paths.py
  python3 distill_router.py W.safetensors v.txt   # or pass them explicitly
"""
import sys, json, struct, re, argparse, collections, math
import numpy as np

import lecore_paths as paths     # NOT `as P`: `P` is the obvious name for a projection
                                 # matrix, and a local `P = Vt[:d].T` inside main() silently
                                 # shadows the module for the WHOLE function (Python scopes
                                 # a name locally if it is assigned anywhere in the body).
                                 # That is exactly the UnboundLocalError this line prevents.


# --------------------------------------------------------------------------- io
def read_st(p):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return {k: v for k, v in json.loads(f.read(n).decode()).items() if k != '__metadata__'}, 8 + n


def load_t(p, meta, base, name):
    m = meta[name]; dt, sh, (b, e) = m['dtype'], m['shape'], m['data_offsets']
    with open(p, 'rb') as f:
        f.seek(base + b); raw = f.read(e - b)
    if dt == 'BF16': a = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif dt == 'F16': a = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else: a = np.frombuffer(raw, dtype=np.float32)
    return np.ascontiguousarray(a.reshape(sh))


# --------------------------------------------------------------------------- wordpiece, minimal
class WordPiece:
    """Just enough BERT WordPiece to tokenize our own corpus. Greedy longest-match, '##' continuations."""

    def __init__(self, vocab_path):
        self.ids = {}
        with open(vocab_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                self.ids[line.rstrip('\n')] = i
        self.unk = self.ids.get('[UNK]', 100)

    def tokens(self, text):
        out = []
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            i, sub = 0, []
            while i < len(word):
                for j in range(len(word), i, -1):
                    piece = word[i:j] if i == 0 else '##' + word[i:j]
                    if piece in self.ids:
                        sub.append(self.ids[piece]); i = j; break
                else:
                    sub = [self.unk]; break
            out.extend(sub)
        return out


# --------------------------------------------------------------------------- SIF
def sif_vectors(texts, wp, E, freq, a=1e-3):
    """Arora/Liang/Ma: a/(a+p(w)) weighted mean of token vectors. Rare words carry more signal --
    which is why `marching_tetrahedra` should outweigh `the`, and why plain mean-pooling fails."""
    V = np.zeros((len(texts), E.shape[1]), dtype=np.float64)
    total = sum(freq.values()) or 1
    for i, t in enumerate(texts):
        ids = wp.tokens(t)
        if not ids:
            continue
        w = np.array([a / (a + freq.get(j, 0) / total) for j in ids])
        V[i] = (w[:, None] * E[ids]).sum(0) / w.sum()
    return V


def remove_first_pc(V, pc=None):
    """The paper's final step, and our ABTT. Fit on DOCUMENTS; apply to queries. Never fit on the
    thing you score -- that lesson cost us a retracted 32d tier."""
    if pc is None:
        Vc = V - V.mean(0)
        pc = np.linalg.svd(Vc, full_matrices=False)[2][:1]
    return V - (V @ pc.T) @ pc, pc


# --------------------------------------------------------------------------- the suite
ASKS = [
    ("make my picture less grainy", {'holographic_denoise', 'holographic_denoisehome', 'holographic_svgf'}),
    ("figure out the shortest way through a maze", {'holographic_flow', 'holographic_navigator'}),
    ("how sure are we this match isn't luck", {'holographic_honesty', 'holographic_conformal', 'holographic_measure'}),
    ("squish a big array down for storage", {'holographic_ratedistortion', 'holographic_coldstore', 'holographic_codec'}),
    ("teach the creature to want food", {'holographic_creature'}),
    ("what does this scene look like from here", {'holographic_render', 'holographic_scene_render'}),
    ("break a shape into simpler pieces", {'holographic_resonator', 'holographic_meshsubdiv'}),
    ("remember a picture and get it back later", {'holographic_archive', 'holographic_objectarchive'}),
    ("guess where the ball goes next", {'holographic_dynamics', 'holographic_forecast',
                                        'holographic_meaning_predict'}),
    ("smooth out the bumpy surface", {'holographic_meshsmooth', 'holographic_autobump'}),
    ("find things near this point quickly", {'holographic_tree', 'holographic_spatial'}),
    ("water flowing and swirling", {'holographic_fluid', 'holographic_diffusion'}),
]


def check_accept_sets(names):
    """An accept set naming a module that does not exist silently caps the score. Verified against the
    live tree once (`holographic_predict` did not exist and was quietly costing us a hit)."""
    have = set(names)
    bad = sorted({m for _, acc in ASKS for m in acc} - have)
    if bad:
        print(f"  WARNING: accept-set modules absent from this repo: {bad}")
    return bad


def collect_docs(repo, max_chars=280):
    """One entry per module: its name plus the head of its docstring. Same corpus the encoder saw."""
    import pathlib, ast
    docs, names = [], []
    for p in sorted(pathlib.Path(repo).rglob('holographic_*.py')):
        try:
            tree = ast.parse(p.read_text(errors='ignore'))
        except SyntaxError:
            continue
        d = (ast.get_docstring(tree) or '')[:max_chars]
        if not d:
            continue
        names.append(p.stem)
        docs.append(f"{p.stem} -- {d}")
    return names, docs


def score(Dv, Qv, names, label):
    unit = lambda A: A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-12)
    D, Q = unit(Dv), unit(Qv)
    ranks = []
    for i, (ask, accept) in enumerate(ASKS):
        order = np.argsort(-(D @ Q[i]))
        r = next((j + 1 for j, x in enumerate(order) if names[x] in accept), len(names))
        ranks.append(r)
    t1 = sum(r == 1 for r in ranks); t5 = sum(r <= 5 for r in ranks)
    print(f"  {label:44s} top-1 {t1:2d}/12  top-5 {t5:2d}/12  median rank {int(np.median(ranks)):3d}  worst {max(ranks)}")
    return t1, t5, int(np.median(ranks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('weights', nargs='?', default=None, help='default: scripts/nomic_text/model.safetensors')
    ap.add_argument('vocab', nargs='?', default=None, help='default: scripts/nomic_text/vocab.txt')
    ap.add_argument('--repo', default=None, help='default: the sibling leCore/ directory')
    a = ap.parse_args()
    a.weights = a.weights or str(paths.nomic_weights())
    a.vocab = a.vocab or str(paths.nomic_vocab())
    a.repo = a.repo or str(paths.REPO)
    print(f"  weights {a.weights}\n  vocab   {a.vocab}\n  repo    {a.repo}\n")

    meta, base = read_st(a.weights)
    embn = [n for n in meta if re.search(r'(embeddings\.word_embeddings|embed_tokens)\.weight$', n)][0]
    E = load_t(a.weights, meta, base, embn).astype(np.float64)
    wp = WordPiece(a.vocab)
    print(f"  token table {E.shape} from {embn}\n")

    names, docs = collect_docs(a.repo)
    print(f"  {len(names)} module docstrings")
    check_accept_sets(names)
    print()

    # word frequency over OUR corpus -- SIF's p(w) must reflect the domain, not the web
    freq = collections.Counter()
    for d in docs:
        freq.update(wp.tokens(d))

    asks = [q for q, _ in ASKS]

    Dv = sif_vectors(docs, wp, E, freq)
    Qv = sif_vectors(asks, wp, E, freq)
    score(Dv, Qv, names, "[A] SIF token-pool, raw")

    Dc, pc = remove_first_pc(Dv)                      # fit on documents only
    Qc, _ = remove_first_pc(Qv, pc)
    score(Dc, Qc, names, "[B] SIF + remove-first-PC (Arora 2017 / ABTT)")

    # [C] + the N19 remap: PCA-64 of the corrected document space
    mu = Dc.mean(0)
    X = Dc - mu
    Vt = np.linalg.svd(X, full_matrices=False)[2]
    for d in (128, 64):
        proj = Vt[:d].T                       # the 768 x d rotation leCore would ship
        score(X @ proj, (Qc - mu) @ proj, names, f"[C] SIF + ABTT + PCA-{d} (the shippable table)")

    print(f"\n  bar: the FULL ENCODER measured 7/12 top-1, 8/12 top-5, median rank 1.")
    print(f"  If [C] lands within one hit of that, leCore ships a {E.shape[0]*64/1e6:.1f} MB q8 table and")
    print(f"  NO MODEL AT RUNTIME -- the transformer becomes a build-time tool.")
    print(f"  If it does not, the encoder stays an OPTIONAL package and we know what the layers bought.")


if __name__ == '__main__':
    main()
