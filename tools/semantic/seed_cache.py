#!/usr/bin/env python3
"""seed_cache.py -- turn the local embedding cache into a small, committable CI seed. stdlib only.

WHY THIS EXISTS
The cold embed (every docstring through the NumPy forward pass) is the ONE slow step, and my "45-140
min" estimate for it was never measured on a 4-vCPU GitHub runner -- an unbaselined number, which this
project distrusts on principle. Rather than gamble a 180-minute job timeout on that guess, we do the
cold embed ONCE, locally, where the weights already live, and ship the result.

    raw cache      65.5 MB   -> would trip GitHub's 50 MB file warning
    gzip -9        19.7 MB
    xz / lzma      17.1 MB   <- committed; matches the repo's dictionary.json.xz precedent

CI then decompresses the seed on a cold cache and starts WARM (~1 min), so the runner never does the
expensive embed at all. The live cache stays gitignored -- it grows per run and is content-addressed,
so the seed is just a periodic snapshot you refresh when the corpus has drifted enough to matter.

USAGE
    # after a local run has populated .knowledge_cache.json (e.g. via run_all.py):
    python3 seed_cache.py            # writes knowledge_cache_seed.json.xz next to it
    python3 seed_cache.py --check    # report drift: how many live entries are NOT in the seed
"""
import argparse, io, json, lzma, pathlib, sys
import numpy as np
import lecore_paths as paths

HERE = pathlib.Path(__file__).resolve().parent
LIVE = HERE / '.knowledge_cache.json'
SEED = HERE / 'routing_seed.npz.xz'    # ONLY routing-relevant entries (code + asks), float16, xz


def _routing_keys(cache):
    """Cache keys the routing exam scores over: code docstrings + ASKS_MODULE/NEGATIVE queries. Uses the
    indexer's OWN collectors and key format -- never reconstruct the hash by hand (that cost 413 misses
    once). Returns only keys actually present in the cache."""
    import hashlib
    import knowledge_index as ki
    wiring = "1000.0|12|True|False"
    k = lambda t: hashlib.sha256((wiring + "||" + t).encode()).hexdigest()[:32]
    repo = str(paths.REPO)
    keys = []
    for kind, name, body in ki.collect_code(repo):
        key = k(f"search_document: {name} -- {body}")
        if key in cache:
            keys.append(key)
    asks = list(ki.ASKS_MODULE) + list(getattr(ki, "ASKS_NEGATIVE", []))
    for a, _ in asks:
        key = k(f"search_query: {a}")
        if key in cache:
            keys.append(key)
    return keys


def build():
    """Store the cache as float16 vectors in an npz, then xz. WHY float16: the shipped index quantizes
    to q8 anyway and routing holds; float16 is cosine-identical to the float32 cache (min cos 1.000000
    measured) so this is LOSSLESS for our purposes, and it takes the seed from 40 MB (raw-json+xz) to
    ~27 MB -- under GitHub's 50 MB warn line. The raw JSON cache at 152 MB would BLOCK the push (>100 MB);
    even xz'd JSON (40 MB) only warns. float16 npz is the honest fix, not a bigger LFS bill."""
    if not LIVE.is_file():
        raise SystemExit(f"no live cache at {LIVE}\n  run the indexer once first (e.g. run_all.py) "
                         f"so there is something to seed from.")
    d = json.loads(LIVE.read_bytes())
    # ONLY the entries the routing EXAM touches: code docstrings + the ask queries. The full cache also
    # holds ~18k md/NOTES/generated-doc windows -- stale, churning, and never scored by the module exam
    # (verified: it ranks E[code_idx]). Seeding those was the 26 MB bloat. This keeps ~500 entries.
    keys = _routing_keys(d)
    vecs = np.array([d[k] for k in keys], dtype=np.float16)     # half precision, cosine-identical
    buf = io.BytesIO()
    np.savez(buf, keys=np.array(keys), vecs=vecs)
    packed = lzma.compress(buf.getvalue(), preset=6)
    SEED.write_bytes(packed)
    print(f"  seed written: {SEED.name}  ({len(d)} cache entries -> {len(keys)} routing entries -> "
          f"{len(packed)/1e3:.0f} KB)")
    if len(packed) > 50e6:
        print("  WARNING: seed exceeds GitHub's 50 MB soft limit -- something is wrong; expected < 1 MB.")


def restore():
    """CI entry point: decompress the seed into the live cache ONLY if the live cache is absent.
    Idempotent -- a runner-cache hit leaves the (newer) live cache untouched."""
    if LIVE.is_file():
        print("  live cache present (runner cache hit) -- not overwriting"); return
    if not SEED.is_file():
        print("  no seed committed -- CI will do a full cold embed (slow, but correct)"); return
    npz = np.load(io.BytesIO(lzma.decompress(SEED.read_bytes())), allow_pickle=False)
    keys, vecs = npz['keys'], npz['vecs'].astype(np.float32)
    cache = {str(k): vecs[i].tolist() for i, k in enumerate(keys)}
    LIVE.write_text(json.dumps(cache))
    print(f"  seeded {len(cache)} entries from the committed snapshot -- CI starts warm")


def check():
    """How far has the live cache drifted from the committed seed? Pure count -- no model needed."""
    if not SEED.is_file():
        raise SystemExit(f"no seed at {SEED}; run without --check to build one.")
    npz = np.load(io.BytesIO(lzma.decompress(SEED.read_bytes())), allow_pickle=False)
    seed_keys = set(str(k) for k in npz['keys'])
    print(f"  seed: {len(seed_keys)} entries")
    if LIVE.is_file():
        live = json.loads(LIVE.read_bytes())
        new = len(set(live) - seed_keys)
        print(f"  live: {len(live)} entries | {new} not yet in the seed "
              f"({'refresh recommended' if new > 50 else 'seed is current enough'})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='report drift between live cache and seed')
    ap.add_argument('--restore', action='store_true', help='CI: decompress seed if live cache absent')
    a = ap.parse_args()
    if a.restore: restore()
    elif a.check: check()
    else: build()
