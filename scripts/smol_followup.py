#!/usr/bin/env python3
"""smol_followup.py -- N20: two probes the census earned. NumPy + stdlib.

PROBE 1 -- WHAT ARE THE 432 DUPLICATE ROWS?
The census found 432 embedding rows with cosine > 0.98, and mean best-cos 0.788. That mean is the
anisotropy signature (the output space had mean|cos| 0.693 until we removed ONE direction). Three
competing explanations, and they are cleanly separable:

  (a) ANISOTROPY artifact  -> centering the table collapses the similarities
  (b) UNTRAINED TOKENS     -> a 49,152-slot vocab has never-seen ids; those rows sit near their
                              initialization and look alike. Tell-tale: LOW row norm, HIGH token id.
                              This is DEAD VOCABULARY -- droppable, but not "structure".
  (c) GENUINE redundancy   -> duplicates survive centering AND have normal norms across the id range.
                              Only (c) revives the defrag thesis.

PROBE 2 -- THE FINDING THE CENSUS ALMOST BURIED: decoder attention is LOW-RANK.
  SmolLM2  q_proj r90 = 0.25 | o_proj 0.36 | k_proj 0.41   (vs nomic-embed-text Wqkv r90 = 0.56)
A quarter of the singular values carry 90% of q_proj's energy. On nomic the shared-basis codec died
because the residual was broadband (~45% of energy, r90 ~ 0.6). Here the matrices themselves are
low-rank, which is a different -- and much more promising -- situation. This probe measures the honest
frontier: bytes vs reconstruction error, low-rank vs group-q8, per role.

USAGE:  python3 smol_followup.py model.safetensors        # paste the whole report back
"""
import sys, json, struct, re, collections
import numpy as np


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
    else: raise ValueError(dt)
    return np.ascontiguousarray(a.reshape(shape))


def gq(W, bits, g=32):
    """Group-32 quantize-dequantize + the bytes it costs (bits/8 per weight + fp16 scale per group)."""
    q = 2 ** (bits - 1) - 1
    f = W.reshape(-1).astype(np.float32); pad = (-len(f)) % g
    if pad: f = np.concatenate([f, np.zeros(pad, np.float32)])
    Wg = f.reshape(-1, g); s = np.abs(Wg).max(1, keepdims=True) / q + 1e-12
    deq = (np.round(Wg / s) * s).reshape(-1)[:W.size].reshape(W.shape)
    return deq, W.size * bits / 8 + Wg.shape[0] * 2


def rel(A, B):
    return float(np.linalg.norm(A - B) / (np.linalg.norm(A) + 1e-12))


def dup_exact(Wn, thresh=0.98, chunk=512):
    """Exact all-pairs best-cosine. (Subsampling misses rare duplicates -- learned the hard way.)"""
    N = len(Wn)
    best = np.zeros(N, np.float32); partner = np.zeros(N, np.int64)
    ndup = 0
    for i in range(0, N, chunk):
        C = np.abs(Wn[i:i + chunk] @ Wn.T)
        for r in range(C.shape[0]):
            C[r, i + r] = 0.0
        ndup += int((C > thresh).sum())
        best[i:i + chunk] = C.max(1)
        partner[i:i + chunk] = C.argmax(1)
    return ndup // 2, best, partner


def main():
    path = sys.argv[1]
    meta, base = read_st(path)
    names = sorted(meta)

    # ---------------------------------------------------------------- PROBE 1
    print("=" * 94)
    print("[P1] WHAT ARE THE 432 DUPLICATE EMBEDDING ROWS? (anisotropy / untrained tokens / real)")
    print("=" * 94)
    embn = [n for n in names if re.search(r'(embed_tokens|wte|word_embeddings)', n)][0]
    E = load_t(path, meta, base, embn).astype(np.float32)
    V, d = E.shape
    unit = lambda A: A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)

    nd_raw, best_raw, partner = dup_exact(unit(E))
    Ec = E - E.mean(0)
    nd_cen, best_cen, _ = dup_exact(unit(Ec))
    print(f"  rows {V}, dim {d}")
    print(f"  RAW      : dup>0.98 {nd_raw:5d} | mean best-cos {best_raw.mean():.3f} | p99 {np.percentile(best_raw,99):.3f}")
    print(f"  CENTERED : dup>0.98 {nd_cen:5d} | mean best-cos {best_cen.mean():.3f} | p99 {np.percentile(best_cen,99):.3f}")
    drop = 1 - best_cen.mean() / max(best_raw.mean(), 1e-9)
    print(f"  -> centering removes {drop:.0%} of the mean similarity."
          f"  {'ANISOTROPY dominated the raw number.' if drop > 0.5 else 'Raw similarity was NOT anisotropy.'}")
    print(f"  -> duplicates surviving centering: {nd_cen} "
          f"{'(REAL structure)' if nd_cen > 0 else '(the raw duplicates were an artifact)'}")

    # untrained-token test: norms and id positions of the duplicate rows
    norms = np.linalg.norm(E, axis=1)
    dup_ids = np.where(best_cen > 0.98)[0] if nd_cen else np.where(best_raw > 0.98)[0]
    if len(dup_ids):
        print(f"\n  duplicate rows: {len(dup_ids)} involved")
        print(f"    row-norm  : dup median {np.median(norms[dup_ids]):.4f} vs table median {np.median(norms):.4f} "
              f"(ratio {np.median(norms[dup_ids])/np.median(norms):.2f})")
        print(f"    token ids : dup median {int(np.median(dup_ids))} of {V}  "
              f"(uniform would be ~{V//2}); min {dup_ids.min()} max {dup_ids.max()}")
        lowq = float((norms[dup_ids] < np.percentile(norms, 10)).mean())
        print(f"    fraction of dup rows in the bottom-10% of norms: {lowq:.0%}")
        print("    READ: low norms + clustered high ids => UNTRAINED VOCABULARY (dead slots, droppable,")
        print("          but not exploitable structure). Normal norms spread across ids => REAL redundancy.")
        # what would dropping them save?
        print(f"    dropping them saves {len(dup_ids)*d*1/1e6:.2f} MB at q8 ({len(dup_ids)/V:.1%} of the table)")

    # ---------------------------------------------------------------- PROBE 2
    print("\n" + "=" * 94)
    print("[P2] DECODER ATTENTION IS LOW-RANK -- the honest bytes frontier (low-rank vs group-q8)")
    print("=" * 94)
    fams = collections.defaultdict(list)
    for n in names:
        if len(meta[n]['shape']) == 2 and 'embed' not in n:
            fams[re.sub(r'\d+', '{L}', n, count=1)].append(n)
    fams = {k: sorted(v) for k, v in fams.items() if len(v) >= 4}

    print(f"  {'role':40s} {'shape':>13s} {'r90':>5s} {'q8 err':>7s} {'q8 MB':>7s} | best low-rank at <= q8 bytes")
    for k, v in sorted(fams.items()):
        Ws = [load_t(path, meta, base, n).astype(np.float32) for n in (v[0], v[len(v)//2], v[-1])]
        W = Ws[0]
        m, n_ = W.shape
        s = np.linalg.svd(W, compute_uv=False)
        e = np.cumsum(s ** 2); e /= e[-1]
        r90 = (int(np.searchsorted(e, 0.90)) + 1) / len(s)
        _, q8b = gq(W, 8)
        q8e = float(np.mean([rel(x, gq(x, 8)[0]) for x in Ws]))
        # largest rank whose q8-stored factors fit in the dense-q8 byte budget
        best = None
        for r in range(4, min(m, n_) + 1, 4):
            fb = (m * r + r * n_) * 1.0 + (m * r + r * n_) / 32 * 2   # factors at q8, group scales
            if fb > q8b: break
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            A = U[:, :r] * S[:r]; B = Vt[:r]
            Aq, _ = gq(A, 8); Bq, _ = gq(B, 8)
            err = rel(W, Aq @ Bq)
            best = (r, err, fb)
        tag = f"r={best[0]:3d} err {best[1]:.4f} ({best[2]/1e6:.2f} MB)" if best else "none fits"
        win = ""
        if best and best[1] < q8e:
            win = f"  <- LOW-RANK WINS ({q8e/best[1]:.1f}x lower error at equal bytes)"
        print(f"  {k[:38]:40s} {str([m,n_]):>13s} {r90:5.2f} {q8e:7.4f} {q8b/1e6:7.2f} | {tag}{win}")

    print("\n  READ: nomic's Wqkv had r90 0.56 and low-rank lost. If a decoder's q/k/o (r90 0.25-0.41)")
    print("        beat q8 at equal bytes, that is a REAL codec -- and it is a decoder-only win.")
    print("        Bar before shipping: perplexity on a fixed text, not weight error.")
    print("\nDONE. Paste the whole report back.")


if __name__ == '__main__':
    main()
