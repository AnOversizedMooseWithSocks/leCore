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
    here = pathlib.Path(__file__).resolve().parent          # tools/semantic
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=128)   # measured knee: top-1 plateaus at 128d, 64d cost 4 hits
    # Default output is the path the ENGINE loads from (lecore_data/routing/index_<dim>d.npz), computed
    # from the repo root -- so running this tool ships the index into place, no manual copy/rename. The
    # loader looks for exactly 'index_128d.npz' under the lecore_data 'routing' category.
    ap.add_argument('--out', default=None)
    # Default paths are computed from THIS file's location -- repo root is two levels up, cache is right
    # here -- so the tool works no matter what the repo folder is named. --repo/--cache override if needed.
    ap.add_argument('--repo', default=str(here.parent.parent),
                    help='repo root to scan for holographic_*.py (default: two levels up from this script)')
    ap.add_argument('--cache', default=str(here / '.knowledge_cache.json'),
                    help='embedding cache json (default: alongside this script)')
    a = ap.parse_args()

    import json
    cache_path = pathlib.Path(a.cache)
    repo = pathlib.Path(a.repo)
    if not cache_path.is_file():
        raise SystemExit(f"  no cache at {cache_path}\n  run knowledge_index.py first, or pass --cache.")
    if not repo.is_dir():
        raise SystemExit(f"  repo path {repo} is not a directory; pass --repo <repo root>.")
    if a.out is None:
        a.out = str(repo / 'lecore_data' / 'routing' / f'index_{a.dim}d.npz')
    cache = json.loads(cache_path.read_text())
    names, vecs = [], []
    for p in sorted(repo.rglob('holographic_*.py')):
        b = head(p)
        if not b:
            continue
        k = key(f"search_document: {p.stem} -- {b}")
        if k in cache:                         # only what the model actually embedded
            names.append(p.stem); vecs.append(cache[k])
    if not vecs:
        raise SystemExit(
            f"  0 module vectors matched the cache.\n"
            f"  scanned {repo} and found holographic_*.py files, but none of their docstrings were in\n"
            f"  {cache_path.name}. Likely the cache was built against a DIFFERENT repo path, or is stale --\n"
            f"  re-run: python knowledge_index.py <model> <vocab> --repo {repo}\n"
            f"  (found {sum(1 for _ in repo.rglob('holographic_*.py'))} modules on disk, {len(cache)} cache entries)")
    V = np.array(vecs, dtype=np.float32)[:, :a.dim]

    # ABTT correction baked in (fit on these docs), then q8 -- the shipped form. Store the correction
    # so the query side applies the IDENTICAL transform at load time.
    mu = V.mean(0)
    Vc = V - mu
    pc = np.linalg.svd(Vc, full_matrices=False)[2][:1]
    Vr = Vc - (Vc @ pc.T) @ pc
    lo = Vr.min(1, keepdims=True); hi = Vr.max(1, keepdims=True)
    q = np.round((Vr - lo) / (hi - lo + 1e-12) * 255).astype(np.uint8)

    # WORKFLOW BONES, names-aligned, packed beside the vectors: the measured dense+structure fusion
    # (top-1 6->7/12, median 2->1, ZERO per-ask regressions at the ship dim, gamma=0.5) needs the bone
    # graph at query time, and the router should not re-derive it from source it may not have. ~1.1k
    # edges as flat (src_idx, dst_idx, weight) arrays -- a few KB. The graph keys are bare stems;
    # index names are holographic_<stem> -- joined here, once, at export.
    import sys
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from holographic.semantic_router.holographic_workflowgraph import build_workflow_graph
    wf = build_workflow_graph(str(repo))
    name_idx = {n: j for j, n in enumerate(names)}
    b_src, b_dst, b_w = [], [], []
    for (sa, sb), w in wf["edges"].items():
        ia = name_idx.get("holographic_" + sa)
        ib = name_idx.get("holographic_" + sb)
        if ia is not None and ib is not None:
            b_src.append(ia); b_dst.append(ib); b_w.append(w)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)          # create lecore_data/routing/ if absent
    buf = io.BytesIO()
    np.savez(buf, names=np.array(names), q=q, lo=lo.astype(np.float16), hi=hi.astype(np.float16),
             mu=mu.astype(np.float16), pc=pc.astype(np.float16),
             bone_src=np.array(b_src, dtype=np.int32), bone_dst=np.array(b_dst, dtype=np.int32),
             bone_w=np.array(b_w, dtype=np.float32))
    out.write_bytes(buf.getvalue())
    print(f"  {len(names)} module vectors @ {a.dim}d q8 + {len(b_src)} bone edges -> {out.name}  ({out.stat().st_size/1024:.0f} KB)")
    print(f"  (from a {len(cache)}-entry build cache; the other {len(cache)-len(names)} entries stay local)")


if __name__ == '__main__':
    main()
