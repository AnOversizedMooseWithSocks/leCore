#!/usr/bin/env python3
"""nomic_vision_census.py -- N13 stage 1: census the nomic-embed-vision weights. NumPy + stdlib only.

SELF-ADAPTING: I do not know this checkpoint's tensor naming in advance, and guessing is how censuses
lie. The script DISCOVERS the layer families (any name containing a layer index groups into a family;
the index becomes the stack dimension), then runs the probes that earned their keep on the text model:

  [V0] inventory            names, shapes, dtypes, param count, family grouping
  [V1] duplicate census     near-duplicate neurons/rows per family (text model: zero -- expect same)
  [V2] rank profile         per-matrix r50/r90 + JOINT layer-stack rank + subspace coverage vs random
  [V3] outliers + quant     col-norm outliers; group-32 q8/q4 error per family (the codec workhorse)

And the two VISION-SPECIFIC probes -- both are real compression targets the text model did not have:

  [V4] PATCH EMBEDDING      the (d x 3*16*16) conv-as-matrix that lifts pixels. Known from the ViT
                            literature to be highly structured (edge/color filters). Probe: rank,
                            2-D Fourier energy compaction of each filter, DCT-basis fit.
  [V5] POSITION TABLE       learned 2-D positions. Known (Dosovitskiy et al. 2021, fig. 10) to be
                            smooth on the patch grid -- i.e., LOW-RANK on the 2-D lattice. If r90 is
                            tiny, the whole table stores as a few outer products: free compression,
                            and a clean 'the model rediscovered our Gabor/FPE story' measurement.

USAGE:  python3 nomic_vision_census.py model.safetensors
        (paste the whole report back)
"""
import sys, json, struct, re, collections
import numpy as np

# ------------------------------------------------------------------ safetensors reader (as before)
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
    q = 2 ** (bits - 1) - 1
    f = W.reshape(-1).astype(np.float64); pad = (-len(f)) % g
    if pad: f = np.concatenate([f, np.zeros(pad)])
    Wg = f.reshape(-1, g); s = np.abs(Wg).max(1, keepdims=True) / q + 1e-12
    deq = (np.round(Wg / s) * s).reshape(-1)[:W.size]
    return float(np.linalg.norm(deq - W.reshape(-1)) / (np.linalg.norm(W) + 1e-12))

def r_frac(W, frac=0.90):
    s = np.linalg.svd(W, compute_uv=False)
    e = np.cumsum(s ** 2); e /= e[-1]
    return (int(np.searchsorted(e, frac)) + 1) / len(s)

def main():
    path = sys.argv[1]
    meta, base = read_st(path)
    names = sorted(meta)

    print("=" * 92)
    print("[V0] INVENTORY")
    print("=" * 92)
    total = sum(int(np.prod(meta[n]['shape'])) for n in names)
    print(f"  tensors: {len(names)} | params: {total/1e6:.1f}M | dtypes: {collections.Counter(meta[n]['dtype'] for n in names)}")

    # FAMILY DISCOVERY: replace the first integer in each name with '{L}' -> family key.
    fams = collections.defaultdict(list)
    for n in names:
        key = re.sub(r'\d+', '{L}', n, count=1)
        fams[key].append(n)
    stacked = {k: v for k, v in fams.items() if len(v) >= 4 and len(meta[v[0]]['shape']) == 2}
    print(f"\n  layer-stacked 2-D families (>=4 layers):")
    for k, v in sorted(stacked.items()):
        print(f"    {k:64s} x{len(v):2d}  {meta[v[0]]['shape']}")
    singles = [n for n in names if len(meta[n]['shape']) >= 2 and
               not any(n in v for v in stacked.values())]
    print(f"  non-stacked >=2-D tensors (embeddings, heads, patch/pos):")
    for n in singles:
        print(f"    {n:64s} {meta[n]['shape']}")

    print("\n" + "=" * 92)
    print("[V1-V3] PER-FAMILY: duplicates | rank | joint rank + coverage vs random | outliers | quant")
    print("=" * 92)
    rng = np.random.default_rng(0)
    for k, v in sorted(stacked.items()):
        Ws = [load_t(path, meta, base, n) for n in sorted(v)]
        W0 = Ws[0]
        # V1 duplicates within the first/last layer (rows as neurons)
        def dup(W):
            Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
            n = min(len(Wn), 2048)
            idx = rng.choice(len(Wn), n, replace=False)
            C = np.abs(Wn[idx] @ Wn[idx].T); np.fill_diagonal(C, 0)
            return int((C > 0.98).sum() // 2), float(np.percentile(C.max(1), 99))
        d0, p0 = dup(Ws[0]); dL, pL = dup(Ws[-1])
        # V2 rank: per-matrix mean r90, joint stack r90
        r90s = [r_frac(W) for W in (Ws[0], Ws[len(Ws)//2], Ws[-1])]
        stack = np.vstack(Ws)
        joint = r_frac(stack)
        # coverage vs random at m = d/6 (right space)
        d = W0.shape[1]; m = max(16, d // 6)
        G = stack.T @ stack
        w, V = np.linalg.eigh(G)
        B = V[:, ::-1][:, :m]
        cov = float(np.linalg.norm(stack @ B) ** 2 / (np.linalg.norm(stack) ** 2))
        Wr = rng.standard_normal(stack.shape).astype(np.float32)
        Gr = Wr.T @ Wr; wr, Vr = np.linalg.eigh(Gr)
        covr = float(np.linalg.norm(Wr @ Vr[:, ::-1][:, :m]) ** 2 / (np.linalg.norm(Wr) ** 2))
        # V3 outliers + quant
        cn = np.linalg.norm(W0, axis=0); outl = float(cn.max() / (np.median(cn) + 1e-12))
        q8, q4 = gq_err(W0, 8), gq_err(W0, 4)
        fam = k.split('.')[-2] if '.' in k else k
        print(f"  {k[:56]:58s}")
        print(f"      dup>0.98 first/last: {d0}/{dL} (p99 {p0:.2f}/{pL:.2f}) | r90 per {np.mean(r90s):.2f} joint {joint:.2f}"
              f" | cov@m={m} {cov:.2f} vs rnd {covr:.2f} ({cov/max(covr,1e-9):.2f}x)"
              f" | outl {outl:.2f} | q8 {q8:.4f} q4 {q4:.3f}")

    print("\n" + "=" * 92)
    print("[V4] PATCH EMBEDDING -- structure of the pixel-lift")
    print("=" * 92)
    patch = [n for n in names if 'patch' in n.lower() and len(meta[n]['shape']) >= 2]
    if not patch:
        patch = [n for n in singles if np.prod(meta[n]['shape'][1:]) in (768, 588, 1024, 3072)]
    for n in patch[:2]:
        W = load_t(path, meta, base, n)
        Wm = W.reshape(W.shape[0], -1) if W.ndim > 2 else W
        print(f"  {n} {list(W.shape)} -> matrix {Wm.shape}")
        print(f"    rank r50/r90: {r_frac(Wm,0.5):.2f}/{r_frac(Wm,0.9):.2f} | q8 {gq_err(Wm,8):.4f}")
        # DCT-basis energy compaction of filters (are they smooth like edge/color detectors?)
        if W.ndim == 4 and W.shape[-1] == W.shape[-2]:            # (d, 3, p, p) conv layout
            p = W.shape[-1]
            k1 = np.arange(p)
            D = np.cos(np.pi * (2 * k1[None, :] + 1) * k1[:, None] / (2 * p)) * np.sqrt(2 / p); D[0] *= np.sqrt(0.5)
            F = np.einsum('ij,cdjk,lk->cdil', D, W.transpose(1, 0, 2, 3), D)  # DCT2 per filter
            E = (F ** 2).reshape(F.shape[0], F.shape[1], -1)
            E = E / (E.sum(-1, keepdims=True) + 1e-12)
            topk = np.sort(E, axis=-1)[..., ::-1][..., :8].sum(-1)
            print(f"    DCT compaction: top-8 of {p*p} coeffs hold {float(topk.mean()):.1%} of filter energy "
                  f"(smooth filters -> high; noise -> {8/(p*p):.1%})")

    print("\n" + "=" * 92)
    print("[V5] POSITION TABLE -- is it smooth on the 2-D patch grid? (low-rank = free compression)")
    print("=" * 92)
    pos = [n for n in names if 'pos' in n.lower() and 'emb' in n.lower()]
    for n in pos[:2]:
        P = load_t(path, meta, base, n)
        P2 = P.reshape(-1, P.shape[-1]) if P.ndim > 2 else P
        T, d = P2.shape
        side = int(round((T - 1) ** 0.5))
        body = P2[1:1 + side * side] if side * side in (T - 1, T) else P2
        print(f"  {n} {list(P.shape)} | tokens {T} (grid ~{side}x{side})")
        print(f"    table rank r50/r90: {r_frac(P2,0.5):.2f}/{r_frac(P2,0.9):.2f}")
        if side >= 4 and side * side <= T:
            G = body.reshape(side, side, d)
            # smoothness: energy of the discrete gradient vs the signal (low = smooth = compressible)
            gx = np.diff(G, axis=0); gy = np.diff(G, axis=1)
            rough = (np.linalg.norm(gx) ** 2 + np.linalg.norm(gy) ** 2) / (np.linalg.norm(G) ** 2 + 1e-12)
            # separability: does pos(x,y) ~ row(x) + col(y)? fit and report residual
            rows = G.mean(1, keepdims=True); cols = G.mean(0, keepdims=True); mu = G.mean((0, 1), keepdims=True)
            R = G - rows - cols + mu
            sep = 1 - float(np.linalg.norm(R) ** 2 / (np.linalg.norm(G - mu) ** 2 + 1e-12))
            print(f"    grid roughness (grad/signal energy): {rough:.3f}  (random ~2.0; smooth << 1)")
            print(f"    additive row+col separability: {sep:.1%} of variance  (high -> store 2*{side} vectors, not {side*side})")

    print("\nDONE. Paste the whole report back.")

if __name__ == '__main__':
    main()
