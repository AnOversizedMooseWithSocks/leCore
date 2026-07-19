"""H5 -- SPATIAL MEMORY: the geometry <-> VSA bridge. "Every closest-point is a recall."

Positions in R^n become hypervectors via fractional power encoding (holographic_fpe.VectorFunctionEncoder --
nearby points map to nearby vectors, MEASURED spearman(cos, -dist) = 0.967), and nearest-point queries become
argmax-cosine over an encoded item store: one matmul, no spatial hash, no per-point loop. The geometric twin of
the engine's standing observation that IK/PBD/PnP/the resonator are all "iterate a projection" -- here, every
closest-point query is an associative RECALL.

MEASURED on a real scan (mantis, 18610 source verts, 20000 texel-scale queries, dim=256):
  * recall batch 4.9s vs 20.0s brute euclid  (4.1x), source encoding amortised (one-time 1.7s)
  * top-1 exact-index match 94.1%, BUT distance-ratio p95 = 1.009 -- the recalled point is within 1% of the
    true nearest distance, i.e. GEOMETRICALLY EQUIVALENT for transfer/bake/shrinkwrap purposes. The honest
    quality metric for geometry is the distance ratio, not the index.
  * resonant payload readout (top-k similarity-weighted): mean colour error 0.034 RGB -- BETTER than the
    volumetric scatter bake's 0.066-0.088, because the kernel weighting is a soft nearest-neighbour average.

KEPT NEGATIVE -- THE CAPACITY TENSION (measured, and why there is NO single-bundle scene mode here):
bundling K position(x)payload bindings into ONE vector collapses fast: class recall 62% at K=8, 25% at K=128
(dim=256). The cause is fundamental, not a tuning problem: FPE keys are CORRELATED BY DESIGN -- nearby
positions get similar vectors, which is exactly the property that makes nearest-neighbour recall work -- and
correlated keys cross-talk in a superposition. Proximity-preservation and bundle-capacity are in direct
tension: one encoding cannot maximise both. So SpatialMemory is an ITEM STORE (rows kept separate, recall by
matmul), never a single scene bundle. If you need a high-capacity bundle, use RANDOM keys and give up
proximity -- the engine's ordinary record/recall path.

KEPT NEGATIVE -- SMALL-N: at small stores brute euclid is comparable or faster (the matmul only wins once the
source encoding is amortised across many query batches and N*Q is large). The 4.1x above is at scan scale.
"""
import numpy as np

from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder


class SpatialMemory:
    """Position-keyed associative memory over R^n: store points (with optional per-point payloads), then
    `nearest` recalls stored indices by argmax cosine of their FPE encodings, and `read` returns a resonant
    (top-k similarity-weighted) payload average -- a soft nearest-neighbour gather in encoding space.

    `bounds` default to the stored points' bounding box padded by 5% -- FPE encoders collapse out-of-range
    values together (holographic_fpe warns), and queries in geometry routinely land just outside the source
    bbox (a texel's 3-D point off a slightly-shrunk retopo), so the pad is load-bearing, not cosmetic.

    Deterministic per seed (hashlib-free by construction: the encoder's phases come from default_rng(seed))."""

    def __init__(self, points, payloads=None, dim=256, bandwidth=8.0, seed=0, bounds=None, pad=0.05):
        P = np.atleast_2d(np.asarray(points, float))
        self.n_dims = P.shape[1]
        if bounds is None:
            lo = P.min(0); hi = P.max(0)
            span = np.maximum(hi - lo, 1e-9)
            bounds = [(float(lo[k] - pad * span[k]), float(hi[k] + pad * span[k])) for k in range(self.n_dims)]
        self.bounds = bounds
        self.enc = VectorFunctionEncoder(n_dims=self.n_dims, dim=int(dim), bounds=bounds,
                                         kernel="rbf", bandwidth=float(bandwidth), seed=int(seed))
        self.points = P
        E = np.asarray(self.enc.encode_many(P), np.float32)
        self._store = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        self.payloads = None if payloads is None else np.atleast_2d(np.asarray(payloads, float))
        if self.payloads is not None and self.payloads.shape[0] != P.shape[0]:
            raise ValueError("payloads must have one row per point")

    def _encode_queries(self, queries):
        Q = np.atleast_2d(np.asarray(queries, float))
        # clamp into bounds rather than let out-of-range values collapse together (the encoder's documented
        # failure mode); geometry queries just past the pad are best answered by the nearest edge of the store.
        for k in range(self.n_dims):
            lo, hi = self.bounds[k]
            Q[:, k] = np.clip(Q[:, k], lo, hi)
        E = np.asarray(self.enc.encode_many(Q), np.float32)
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    def similarities(self, queries, chunk=4000):
        """(n_queries, n_points) cosine similarities -- the raw resonance field, chunked to bound memory.
        NOTE this materialises the FULL matrix; for large query batches use nearest()/read(), which stream
        chunks and keep only top-k (a 236k-texel bake against 18k points is 16 GB as a full matrix -- measured
        OOM; streamed top-k is a few MB)."""
        EQ = self._encode_queries(queries)
        out = np.empty((EQ.shape[0], self._store.shape[0]), np.float32)
        for s in range(0, EQ.shape[0], int(chunk)):
            out[s:s + int(chunk)] = EQ[s:s + int(chunk)] @ self._store.T
        return out

    def _topk_stream(self, queries, k, chunk=4000):
        """Streamed top-k: encode + match chunk by chunk, keeping only (idx, sim) of the k best per query --
        never the full (Q, N) matrix. Returns (idx (Q,k) best-first, weights (Q,k))."""
        EQ = self._encode_queries(queries)
        k = int(k)
        idx_all = np.empty((EQ.shape[0], k), int)
        w_all = np.empty((EQ.shape[0], k), np.float32)
        for s in range(0, EQ.shape[0], int(chunk)):
            sim = EQ[s:s + int(chunk)] @ self._store.T
            if k == 1:
                ia = sim.argmax(1)[:, None]
            else:
                ia = np.argpartition(-sim, k - 1, axis=1)[:, :k]
                row = np.take_along_axis(sim, ia, 1)
                ia = np.take_along_axis(ia, np.argsort(-row, axis=1), 1)
            idx_all[s:s + int(chunk)] = ia
            w_all[s:s + int(chunk)] = np.take_along_axis(sim, ia, 1)
        return idx_all, w_all

    def nearest(self, queries, k=1, chunk=4000):
        """Indices of the k most-similar stored points per query, best first: shape (n_queries, k).
        k=1 is the closest-point-as-recall move. The recalled point is measured to be within ~1% of the true
        nearest distance at p95 (dist-ratio 1.009 on a real scan) -- geometrically equivalent for transfer.
        Streams chunks (never the full similarity matrix), so texel-scale query batches fit in memory."""
        idx, _ = self._topk_stream(queries, k=k, chunk=chunk)
        return idx

    def read(self, queries, k=4, chunk=4000):
        """Resonant payload readout: the top-k similarity-weighted average of stored payloads at each query --
        a SOFT nearest-neighbour gather in encoding space. MEASURED 0.034 mean RGB error reading vertex colour
        on a scan (k=4), better than a volumetric scatter bake. Requires payloads at construction. Streams
        chunks, so texel-scale batches (200k+ queries) fit in memory."""
        if self.payloads is None:
            raise ValueError("read() needs payloads; construct SpatialMemory(points, payloads=...)")
        idx, w = self._topk_stream(queries, k=k, chunk=chunk)
        w = np.maximum(w - w.min(1, keepdims=True), 1e-9)
        w = w / w.sum(1, keepdims=True)
        return np.einsum("qk,qkc->qc", w, self.payloads[idx])


def spatial_recall(points, queries, payloads=None, k=1, dim=256, bandwidth=8.0, seed=0):
    """ONE CALL for the H5 move: encode `points` (n, d) as a spatial memory and recall at `queries` (m, d).
    Returns (indices, payload_out, report): indices (m, k) best-first; payload_out is the resonant top-k
    readout when payloads are given (else None); report carries the store size/dim/bounds. For repeated query
    batches against ONE source, build SpatialMemory directly and reuse it -- the source encoding is the
    amortised cost (measured 1.7s at 18k points / dim 256), and reusing it is where the 4.1x lives."""
    mem = SpatialMemory(points, payloads=payloads, dim=dim, bandwidth=bandwidth, seed=seed)
    idx = mem.nearest(queries, k=k)
    out = mem.read(queries, k=max(int(k), 4)) if payloads is not None else None
    report = {"n_points": int(mem.points.shape[0]), "dim": int(mem.enc.dim),
              "bounds": mem.bounds, "k": int(k)}
    return idx, out, report


def _selftest():
    """Numeric contracts + the kept negatives, loud."""
    rng = np.random.default_rng(0)

    # --- 1: recall finds geometrically-equivalent nearest points on a random cloud ---
    P = rng.random((800, 3))
    Q = np.clip(P[rng.choice(800, 200, replace=False)] + rng.normal(0, 0.01, (200, 3)), 0, 1)
    mem = SpatialMemory(P, dim=256, seed=0)
    idx = mem.nearest(Q, k=1)[:, 0]
    d2 = ((Q[:, None, :] - P[None, :, :]) ** 2).sum(2)
    nn_true = d2.argmin(1)
    top1 = float((idx == nn_true).mean())
    dtrue = np.sqrt(d2[np.arange(len(Q)), nn_true]); dgot = np.sqrt(d2[np.arange(len(Q)), idx])
    dr95 = float(np.percentile(dgot / (dtrue + 1e-12), 95))
    assert top1 > 0.85, "top-1 recall %.2f too low" % top1
    assert dr95 < 1.10, "distance ratio p95 %.3f -- recalled points must be geometrically equivalent" % dr95

    # --- 2: resonant payload readout reproduces a smooth field ---
    pay = P.copy()                                       # payload = the position itself (a smooth field)
    mem2 = SpatialMemory(P, payloads=pay, dim=256, seed=0)
    got = mem2.read(Q, k=4)
    err = float(np.linalg.norm(got - Q, axis=1).mean())
    assert err < 0.05, "resonant readout err %.3f -- soft-NN must reproduce a smooth field" % err

    # --- 3: determinism (constitution): same seed -> identical store and recalls ---
    memA = SpatialMemory(P, dim=128, seed=0); memB = SpatialMemory(P, dim=128, seed=0)
    assert np.array_equal(memA._store, memB._store), "store must be deterministic per seed"
    assert np.array_equal(memA.nearest(Q), memB.nearest(Q))

    # --- 4: KEPT NEGATIVE, pinned -- FPE keys are correlated, so single-bundle capacity COLLAPSES.
    # Proximity-preservation and bundle-capacity are in direct tension; this assert keeps the tension on
    # record by DEMONSTRATING the collapse, so nobody 'adds a bundle mode' without meeting this number.
    from holographic.sampling_and_signal.holographic_fpe import bind
    codes = rng.standard_normal((8, 256)); codes /= np.linalg.norm(codes, axis=1, keepdims=True)
    pid = rng.choice(800, 128, replace=False); cls = rng.integers(0, 8, 128)
    B = np.zeros(256)
    for i in range(128):
        B += bind(mem._store[pid[i]].astype(float), codes[cls[i]])
    B /= np.linalg.norm(B) + 1e-12
    ok = 0
    for i in range(64):
        probes = np.array([float(B @ bind(mem._store[pid[i]].astype(float), codes[c])) for c in range(8)])
        ok += int(probes.argmax() == cls[i])
    bundle_recall = ok / 64.0
    assert bundle_recall < 0.6, ("KEPT NEGATIVE violated: FPE-keyed bundle recall %.0f%% at K=128 -- if this "
                                 "ever PASSES 60%%, the correlated-key cross-talk analysis is wrong and the "
                                 "no-bundle-mode design decision must be revisited" % (100 * bundle_recall))

    # --- 5: out-of-bounds queries are clamped, not collapsed (the pad + clamp contract) ---
    far = np.array([[2.0, 2.0, 2.0]])                    # far outside the store's bounds
    i_far = mem.nearest(far)[:, 0][0]
    corner = P[((P - 1.0) ** 2).sum(1).argmin()]         # stored point nearest the (1,1,1) corner
    assert np.linalg.norm(mem.points[i_far] - corner) < 0.35, \
        "an out-of-bounds query must recall a point near the closest edge of the store, not garbage"

    print("spatialmem selftest OK (top-1 %.0f%%, dist-ratio p95 %.3f, readout err %.3f, deterministic; "
          "KEPT NEGATIVE pinned: FPE bundle recall %.0f%% at K=128 -- correlated keys cross-talk, item store "
          "only)" % (100 * top1, dr95, err, 100 * bundle_recall))


if __name__ == "__main__":
    _selftest()
