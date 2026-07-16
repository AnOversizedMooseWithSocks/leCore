#!/usr/bin/env python3
"""export_index.py -- pull the SHIPPABLE routing index out of the big build cache. stdlib + numpy.

THE DISTINCTION Moose caught (rev. 45): the .knowledge_cache.json is a BUILD INTERMEDIATE -- every
embedded window (18,963 of them, 26 MB xz). The thing that SHIPS is only the module-docstring vectors
(503 of them), at 64d q8 -- about 30 KB. This extracts the second from the first, so a 26 MB local
cache produces a 30 KB committed artifact.

    python3 export_index.py            # writes routing_index_64d.npz (the ship artifact)
    python3 export_index.py --dim 128  # if 64d ever proves too lossy on a corpus (re-measure first)
"""
import argparse, ast, hashlib, io, pathlib, re
import numpy as np
import lecore_paths as paths

WIRING = "1000.0|12|True|False"


def key(t):
    return hashlib.sha256((WIRING + '||' + t).encode()).hexdigest()[:32]


def head(p, n=280):
    try:
        d = ast.get_docstring(ast.parse(p.read_text(errors='ignore'))) or ''
    except SyntaxError:
        d = ''
    return re.sub(r'\s+', ' ', d).strip()[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--out', default='routing_index_64d.npz')
    a = ap.parse_args()

    import json
    cache = json.loads(paths.CACHE.read_text())
    names, vecs = [], []
    for p in sorted(paths.REPO.rglob('holographic_*.py')):
        b = head(p)
        if not b:
            continue
        k = key(f"search_document: {p.stem} -- {b}")
        if k in cache:                         # only what the model actually embedded
            names.append(p.stem); vecs.append(cache[k])
    V = np.array(vecs, dtype=np.float32)[:, :a.dim]

    # ABTT correction baked in (fit on these docs), then q8 -- the shipped form. Store the correction
    # so the query side applies the IDENTICAL transform at load time.
    mu = V.mean(0)
    Vc = V - mu
    pc = np.linalg.svd(Vc, full_matrices=False)[2][:1]
    Vr = Vc - (Vc @ pc.T) @ pc
    lo = Vr.min(1, keepdims=True); hi = Vr.max(1, keepdims=True)
    q = np.round((Vr - lo) / (hi - lo + 1e-12) * 255).astype(np.uint8)

    out = pathlib.Path(a.out)
    buf = io.BytesIO()
    np.savez(buf, names=np.array(names), q=q, lo=lo.astype(np.float16), hi=hi.astype(np.float16),
             mu=mu.astype(np.float16), pc=pc.astype(np.float16))
    out.write_bytes(buf.getvalue())
    print(f"  {len(names)} module vectors @ {a.dim}d q8 -> {out.name}  ({out.stat().st_size/1024:.0f} KB)")
    print(f"  (from a {len(cache)}-entry build cache; the other {len(cache)-len(names)} entries stay local)")


if __name__ == '__main__':
    main()
