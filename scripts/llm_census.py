#!/usr/bin/env python3
"""llm_census.py -- census a small CAUSAL LM. NumPy + stdlib only. Self-adapting to tensor names.

WHY A DECODER, AND WHY A SMALL ONE:
Every "not fragmented" result we have (zero duplicate neurons, zero duplicate embedding rows, pruning
10% costs 27%) came from nomic-embed-text -- a CONTRASTIVELY trained encoder. Contrastive training
explicitly pushes representations apart; it would be surprising if such a model had duplicates.

**A causal LM is trained by next-token prediction, not by pushing things apart.** So the central
question of this whole thread is genuinely OPEN for decoders, and this is the cheapest way to ask it:

    Is "the model is dense, not fragmented" a property of MODELS, or a property of CONTRASTIVE TRAINING?

That is a real experiment with a real chance of overturning our own conclusion. Kept negatives cut both
ways: if the decoder IS redundant, the defrag thesis comes back to life -- for decoders.

RECOMMENDED TARGET: HuggingFaceTB/SmolLM2-135M (Apache-2.0, ~135M params, single safetensors).
  Alternatives, same script: Qwen/Qwen2.5-0.5B (Apache-2.0), EleutherAI/pythia-160m (Apache-2.0).

THE SECOND REASON DECODERS MATTER (measured, §11.1): decode is BANDWIDTH-bound on CPU, so compression
is literally speed (5.3x fewer bytes -> 5.4x faster). Encoders are compute-bound and get no such win.

THE THIRD, WHICH SMALL MODELS MAKE ACUTE: the token-embedding table is a huge fraction of a small
model (nomic: 17%). If it dominates, quantizing it is the whole story, and D1 (embedding tables are
lookup tables, not GEMM operands) says it can be quantized harder than the matmul weights.

USAGE:  python3 llm_census.py model.safetensors      # paste the whole report back
"""
import sys, json, struct, re, collections
import numpy as np

import lecore_paths as P


def read_st(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n).decode('utf-8'))
    return {k: v for k, v in header.items() if k != '__metadata__'}, 8 + n


def load_t(path, meta, base, name):
    m = meta[name]; dt, shape, (b, e) = m['dtype'], m['shape'], m['data_offsets']
    with open(path, 'rb') as f:
        f.seek(base + b); raw = f.read(e - b)
    if dt == 'F32': a = np.frombuffer(raw, dtype=np.float32)
    elif dt == 'F16': a = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dt == 'BF16': a = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    else: raise ValueError(f"dtype {dt} on {name}")
    return np.ascontiguousarray(a.reshape(shape))


def gq_err(W, bits, g=32):
    """Group-32 symmetric quantization error -- the codec that won on nomic (q8 ~ 0.005 everywhere)."""
    q = 2 ** (bits - 1) - 1
    f = W.reshape(-1).astype(np.float64); pad = (-len(f)) % g
    if pad: f = np.concatenate([f, np.zeros(pad)])
    Wg = f.reshape(-1, g); s = np.abs(Wg).max(1, keepdims=True) / q + 1e-12
    deq = (np.round(Wg / s) * s).reshape(-1)[:W.size]
    return float(np.linalg.norm(deq - W.reshape(-1)) / (np.linalg.norm(W) + 1e-12))


def r_frac(W, frac=0.90, cap=4096):
    if W.shape[0] > cap:
        W = W[np.random.default_rng(0).choice(W.shape[0], cap, replace=False)]
    s = np.linalg.svd(W, compute_uv=False)
    e = np.cumsum(s ** 2); e /= e[-1]
    return (int(np.searchsorted(e, frac)) + 1) / len(s)


def dup_census(W, rng, thresh=0.98, chunk=256, max_exact_rows=300000):
    """SCALE NOTE for Qwen3.5 (248,320 x 1024): exact all-pairs is 6.2e10 dot-products. Chunked BLAS
    handles it in ~10-20 min at bounded memory. Do NOT subsample: the whole point of a duplicate census
    is rare duplicates, and a 6% sample misses a planted pair 99.6% of the time (measured, the hard way)."""
    """EXACT near-duplicate ROWS over ALL rows, in chunks.

    KEPT NEGATIVE (my own bug, caught by a planted test): an earlier version sampled 3,000 rows at
    random and computed pairwise cosines among them. On a 49,152-row embedding table that samples 6%
    of rows, so the chance of drawing BOTH halves of a planted duplicate pair is ~0.4% -- it reported
    ZERO duplicates on a table where I had planted forty. Rare duplicates are exactly what a duplicate
    census exists to find, so subsampling defeats the instrument. Chunked exact search instead:
    O(N^2) comparisons but only O(chunk x N) memory, and N here is at most ~150k."""
    Wn = (W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    N = len(Wn)
    if N > max_exact_rows:
        raise RuntimeError(f"{N} rows: raise max_exact_rows deliberately, do not silently subsample")
    best = np.zeros(N, dtype=np.float32)
    ndup = 0
    for i in range(0, N, chunk):
        block = Wn[i:i + chunk]
        C = np.abs(block @ Wn.T)                       # (chunk, N)
        for r in range(len(block)):
            C[r, i + r] = 0.0                          # drop self
        ndup += int((C > thresh).sum())
        best[i:i + chunk] = C.max(1)
    return ndup // 2, float(np.percentile(best, 99)), float(best.mean())


def main():
    # Default to the Qwen checkpoint, fall back to smol; an explicit path always wins.
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        try:
            path = str(P.qwen_weights())
        except FileNotFoundError:
            path = str(P.smol_weights())
    print(f"  model: {path}\n")
    meta, base = read_st(path)
    names = sorted(meta)
    rng = np.random.default_rng(0)

    print("=" * 94)
    print("[L0] INVENTORY + WHERE THE BYTES ACTUALLY ARE")
    print("=" * 94)
    sizes = {n: int(np.prod(meta[n]['shape'])) for n in names}
    total = sum(sizes.values())
    print(f"  tensors {len(names)} | params {total/1e6:.1f}M | dtypes {dict(collections.Counter(meta[n]['dtype'] for n in names))}")
    print("  largest tensors:")
    for n, s in sorted(sizes.items(), key=lambda kv: -kv[1])[:6]:
        print(f"    {n:56s} {str(meta[n]['shape']):18s} {s/1e6:7.2f}M  ({s/total:5.1%})")

    # embedding table + tied head detection
    emb = [n for n in names if re.search(r'(embed_tokens|wte|word_embeddings|tok_emb)', n)]
    head = [n for n in names if re.search(r'(lm_head|output|unembed)', n) and len(meta[n]['shape']) == 2]
    emb_p = sum(sizes[n] for n in emb); head_p = sum(sizes[n] for n in head)
    print(f"\n  token embedding: {emb} -> {emb_p/1e6:.2f}M ({emb_p/total:.1%} of the model)")
    print(f"  lm head:         {head or 'ABSENT -> weights are TIED to the embedding'}"
          f"{'' if not head else f' -> {head_p/1e6:.2f}M'}")
    print("  READ: on small models the embedding table dominates. It is a LOOKUP table, not a GEMM")
    print("        operand -- per-row scales cost nothing at lookup time, so it tolerates harder quant.")

    print("\n" + "=" * 94)
    print("[L1] THE OPEN QUESTION: is a CAUSAL LM more redundant than a CONTRASTIVE encoder?")
    print("=" * 94)
    print("  (nomic-embed-text, contrastive: ZERO duplicate neurons in 44/48 matrices; ZERO duplicate")
    print("   embedding rows of 30,528, p99 best-cos 0.802. If a decoder differs, defrag lives.)")
    if emb:
        E = load_t(path, meta, base, emb[0])
        d, p99, mean = dup_census(E, rng)
        verdict = ('<- REDUNDANT: duplicate rows exist, defrag is ALIVE for decoders' if d > 0
                   else '<- dense, like nomic (contrastive result generalizes)' if p99 < 0.90
                   else '<- borderline: no exact dups, but a high-similarity tail')
        print(f"\n  EMBEDDING rows ({len(E)} of them): dup>0.98 {d} | p99 best-cos {p99:.3f} | "
              f"mean best-cos {mean:.3f}\n  {verdict}")

    # family discovery: first integer in the name is the layer index
    fams = collections.defaultdict(list)
    for n in names:
        if len(meta[n]['shape']) == 2 and n not in emb and n not in head:
            fams[re.sub(r'\d+', '{L}', n, count=1)].append(n)
    stacked = {k: sorted(v) for k, v in fams.items() if len(v) >= 4}
    print(f"\n  layer-stacked families found: {len(stacked)}")
    print(f"  {'family':52s} {'L':>3s} {'shape':>14s} {'dup':>5s} {'p99':>6s} {'r90':>5s} {'q8':>7s} {'q4':>6s}")
    for k, v in sorted(stacked.items()):
        Ws = [load_t(path, meta, base, n) for n in (v[0], v[len(v)//2], v[-1])]
        dsum = 0; p99s = []
        for W in Ws:
            dd, pp, _ = dup_census(W, rng)
            dsum += dd; p99s.append(pp)
        r90 = float(np.mean([r_frac(W) for W in Ws]))
        q8 = float(np.mean([gq_err(W, 8) for W in Ws]))
        q4 = float(np.mean([gq_err(W, 4) for W in Ws]))
        print(f"  {k[:50]:52s} {len(v):3d} {str(list(Ws[0].shape)):>14s} {dsum:5d} {np.mean(p99s):6.3f} "
              f"{r90:5.2f} {q8:7.4f} {q4:6.3f}")

    print("\n" + "=" * 94)
    print("[L2] SHARED CROSS-LAYER SUBSPACE (the S1 probe that FIRED on nomic: 2.35x over random)")
    print("=" * 94)
    for k, v in sorted(stacked.items()):
        if len(v) < 6: continue
        Ws = [load_t(path, meta, base, n) for n in v]
        if Ws[0].shape[1] > 2048: continue
        stack = np.vstack(Ws).astype(np.float64)
        d = stack.shape[1]; m = max(16, d // 6)
        G = stack.T @ stack
        _, V = np.linalg.eigh(G)
        cov = float(np.linalg.norm(stack @ V[:, ::-1][:, :m]) ** 2 / np.linalg.norm(stack) ** 2)
        R = rng.standard_normal(stack.shape)
        _, Vr = np.linalg.eigh(R.T @ R)
        covr = float(np.linalg.norm(R @ Vr[:, ::-1][:, :m]) ** 2 / np.linalg.norm(R) ** 2)
        joint = r_frac(stack)
        print(f"  {k[:50]:52s} cov@m={m:<4d} {cov:.3f} vs rnd {covr:.3f} ({cov/covr:4.2f}x) | joint r90 {joint:.2f}")
    print("  READ: >>1.0x means the layers share a basis (nomic: Wqkv 2.35x). Necessary, NOT sufficient")
    print("        for a codec -- the residual must also be SMALL, which on nomic it was not (~45%).")

    print("\n" + "=" * 94)
    print("[L3] EMBEDDING TABLE: how hard can we quantize the thing that dominates the bytes?")
    print("=" * 94)
    if emb:
        E = load_t(path, meta, base, emb[0])
        print(f"  {emb[0]} {list(E.shape)}  r90 {r_frac(E):.3f}")
        for bits in (8, 6, 4, 3):
            for g in (32, 128):
                print(f"    q{bits} group-{g:<3d}: rel err {gq_err(E, bits, g):.4f}"
                      f"   bytes {E.size*bits/8/1e6 + E.size/g*2/1e6:6.1f} MB (vs fp32 {E.size*4/1e6:.1f} MB)")
        rown = np.linalg.norm(E, axis=1)
        print(f"  row-norm spread max/median {float(rown.max()/np.median(rown)):.2f}"
              f"  (high -> per-row scales matter; they are free for a lookup table)")

    print("\nDONE. Paste the whole report back.")
    print("BARS this feeds: (a) duplicate census -> is defrag alive for decoders?  (b) q8 error ~0.005?")
    print("                 (c) embedding-table quant -> the dominant-bytes decision  (d) S1 coverage.")


if __name__ == '__main__':
    main()
