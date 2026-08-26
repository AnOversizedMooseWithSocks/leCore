#!/usr/bin/env python3
"""align_models.py -- PHASE A: how much of SmolLM2's token-embedding geometry is a linear map of
nomic-embed-text's? NumPy + stdlib. Needs NO forward pass -- just the two tables + two tokenizers.

THE CLAIM UNDER TEST (Moose's): nomic as the BASE space for words; the LLM as DIFFS + STRUCTURE on
top. If smol_emb ~= W @ nomic_emb + small_delta, the base+diff story has legs. Published grounding:
relative representations (Moschella et al., ICLR 2023), Procrustes word-space alignment (Conneau et
al., 2018 / MUSE), the Platonic Representation Hypothesis (Huh et al., ICML 2024).

WHAT IT MEASURES
  [A1] anchor vocabulary : whole words that are single tokens in BOTH vocabs
  [A2] the map           : least-squares AND orthogonal-Procrustes W (closed form, no gradients),
                           judged HELD-OUT (fit on half the anchors, score the other half):
                             - R^2 (explained variance)
                             - neighbor transfer: for a held-out word, is its true smol row the
                               nearest smol row to W @ nomic(word)? top-1 / top-5 over all 49k rows
                           against a RANDOM-W control (same norms) so the number has a floor.
  [A3] the bytes question: smol_table = W @ nomic + DELTA. leCore already ships nomic, so W (1.7 MB)
                           + quantized DELTA competes with plain q8 (30.1 MB) through DELTA alone.
                           STATED PRIOR, on the record: DELTA will be broadband and q8 will win --
                           this law has held three times. A2's R^2 is the prize even if A3 loses:
                           high R^2 IS the bidirectional cross-model bridge.

USAGE:
  python3 align_models.py smol.safetensors smol_tokenizer.json nomic.safetensors nomic_vocab.txt
  (paste the whole report back)
"""
import sys, json, struct, re
import numpy as np


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


def find_emb(meta):
    for n in meta:
        if re.search(r'(embed_tokens|word_embeddings|wte)\.weight$', n):
            return n
    raise KeyError('no embedding table found')


def gq_err_bytes(W, bits, g=32):
    q = 2 ** (bits - 1) - 1
    f = W.reshape(-1).astype(np.float64); pad = (-len(f)) % g
    if pad: f = np.concatenate([f, np.zeros(pad)])
    Wg = f.reshape(-1, g); s = np.abs(Wg).max(1, keepdims=True) / q + 1e-12
    deq = (np.round(Wg / s) * s).reshape(-1)[:W.size]
    err = float(np.linalg.norm(deq - W.reshape(-1)) / (np.linalg.norm(W) + 1e-12))
    return err, W.size * bits / 8 + Wg.shape[0] * 2


def main():
    smol_st, smol_tok, nomic_st, nomic_vocab = sys.argv[1:5]

    # ------------------------------------------------------------- load tables
    ms, bs = read_st(smol_st)
    S = load_t(smol_st, ms, bs, find_emb(ms)).astype(np.float64)      # (49152, 576)
    mn, bn = read_st(nomic_st)
    N = load_t(nomic_st, mn, bn, find_emb(mn)).astype(np.float64)     # (30528, 768)
    print(f"smol table {S.shape} | nomic table {N.shape}")

    # ------------------------------------------------------------- [A1] anchors
    print("\n" + "=" * 92)
    print("[A1] ANCHOR VOCABULARY -- whole words single-token in BOTH vocabs")
    print("=" * 92)
    # nomic: BERT WordPiece, uncased -- vocab.txt line i = token i
    nom_id = {}
    with open(nomic_vocab, encoding='utf-8') as f:
        for i, line in enumerate(f):
            t = line.rstrip('\n')
            if re.fullmatch(r'[a-z]{2,}', t):                          # plain lowercase words only
                nom_id[t] = i
    # smol: GPT-style BPE from tokenizer.json; leading space marker (Ġ or \u0120) = word start
    tj = json.load(open(smol_tok, encoding='utf-8'))
    vocab = tj['model']['vocab']
    smol_id = {}
    for tok, i in vocab.items():
        w = tok
        if w.startswith('\u0120'): w = w[1:]                           # 'Ġword' -> 'word' (word-start form)
        elif w.startswith(' '): w = w[1:]
        if re.fullmatch(r'[A-Za-z]{2,}', w):
            key = w.lower()
            # prefer the word-start (space-prefixed) form; keep first seen otherwise
            if key not in smol_id or tok.startswith('\u0120'):
                smol_id[key] = i
    anchors = sorted(set(nom_id) & set(smol_id))
    print(f"  nomic word-tokens {len(nom_id)} | smol word-tokens {len(smol_id)} | ANCHORS {len(anchors)}")
    if len(anchors) < 500:
        print("  too few anchors -- check tokenizer paths"); return
    Xn = N[[nom_id[w] for w in anchors]]                               # (A, 768)
    Xs = S[[smol_id[w] for w in anchors]]                              # (A, 576)

    # center both spaces (the anisotropy lesson: the mean is not signal)
    mun, mus = Xn.mean(0), Xs.mean(0)
    Xn = Xn - mun; Xs = Xs - mus

    # ------------------------------------------------------------- [A2] the map, HELD OUT
    print("\n" + "=" * 92)
    print("[A2] THE MAP -- fit on half the anchors, judged on the other half")
    print("=" * 92)
    rng = np.random.default_rng(0)
    for trial in range(3):
        perm = rng.permutation(len(anchors))
        tr, te = perm[:len(perm)//2], perm[len(perm)//2:]

        # least squares: W = argmin ||Xn W - Xs||  (closed form)
        W, *_ = np.linalg.lstsq(Xn[tr], Xs[tr], rcond=None)
        pred = Xn[te] @ W
        ss_res = np.linalg.norm(pred - Xs[te]) ** 2
        ss_tot = np.linalg.norm(Xs[te]) ** 2
        r2 = 1 - ss_res / ss_tot

        # neighbor transfer: nearest SMOL ROW (of all 49k, centered) to the mapped vector
        Sc = S - mus
        Sn = Sc / (np.linalg.norm(Sc, axis=1, keepdims=True) + 1e-12)
        sample = list(range(min(400, len(te))))          # positions WITHIN te / pred
        t1 = t5 = 0
        for j in sample:
            v = pred[j] / (np.linalg.norm(pred[j]) + 1e-12)
            order = np.argsort(-(Sn @ v))[:5]
            true = smol_id[anchors[te[j]]]
            t1 += order[0] == true; t5 += true in order
        # random-W control (same output scale)
        Wr = rng.standard_normal(W.shape) * np.linalg.norm(W) / np.sqrt(W.size)
        predr = Xn[te] @ Wr
        r2r = 1 - np.linalg.norm(predr - Xs[te]) ** 2 / ss_tot
        print(f"  trial {trial}: held-out R^2 {r2:+.3f} (random-W {r2r:+.3f}) | "
              f"neighbor top-1 {t1}/{len(sample)} top-5 {t5}/{len(sample)} over {len(S)} rows")

    # orthogonal Procrustes on the last split (rotation-only: the stricter geometric claim)
    k = min(Xn.shape[1], Xs.shape[1])
    U, _, Vt = np.linalg.svd(Xn[tr].T @ Xs[tr], full_matrices=False)
    R = U[:, :k] @ Vt[:k]
    predp = Xn[te] @ R
    r2p = 1 - np.linalg.norm(predp - Xs[te]) ** 2 / np.linalg.norm(Xs[te]) ** 2
    print(f"  Procrustes (rotation-only, 768->576): held-out R^2 {r2p:+.3f}"
          f"   (general-linear above; the gap = how much SCALING the map needs)")

    # ------------------------------------------------------------- [A3] the bytes question
    print("\n" + "=" * 92)
    print("[A3] BYTES -- smol_table = W@nomic + DELTA vs plain group-q8   (prior on record: q8 wins)")
    print("=" * 92)
    # map EVERY smol anchor row; non-anchor rows have no nomic source and stay plain -- so the codec
    # can only ever help on the anchor subset. Report both honestly.
    W, *_ = np.linalg.lstsq(Xn, Xs, rcond=None)                        # fit on all anchors now
    delta = Xs - Xn @ W
    e_tab, b_tab = gq_err_bytes(Xs, 8)
    print(f"  anchor subset ({len(anchors)} rows):")
    print(f"    plain q8          : err {e_tab:.4f}  {b_tab/1e6:6.2f} MB")
    for bits in (4, 3, 2):
        e_d, b_d = gq_err_bytes(delta, bits)
        # reconstruction error of (W@nomic + quantized delta) relative to the true rows
        q = 2 ** (bits - 1) - 1
        f = delta.reshape(-1); pad = (-len(f)) % 32
        fp = np.concatenate([f, np.zeros(pad)]) if pad else f
        Dg = fp.reshape(-1, 32); s = np.abs(Dg).max(1, keepdims=True) / q + 1e-12
        dq = (np.round(Dg / s) * s).reshape(-1)[:delta.size].reshape(delta.shape)
        rec = Xn @ W + dq
        err = float(np.linalg.norm(rec - Xs) / np.linalg.norm(Xs))
        wb = W.size * 4 / 1e6
        print(f"    W@nomic + q{bits} delta: err {err:.4f}  {b_d/1e6:6.2f} MB delta + {wb:.2f} MB map"
              f"   {'<- BEATS q8' if err <= e_tab and (b_d/1e6+wb) < b_tab/1e6 else ''}")
    print(f"  (non-anchor rows: {len(S)-len(anchors)} of {len(S)} have no nomic source -> plain q8 regardless)")
    print("\n  READ: A2's held-out R^2 and neighbor transfer are the finding either way -- a high R^2 is a")
    print("        working CROSS-MODEL BRIDGE (nomic semantics <-> smol tokens), which Phase B consumes.")
    print("\nDONE. Paste the whole report back.")


if __name__ == '__main__':
    main()
