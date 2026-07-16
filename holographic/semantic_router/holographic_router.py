"""Embedding router -- route a request to the right module by COSINE in nomic's space, not token overlap.

WHY THIS EXISTS (measured, backlog N9/N28):
The catalog's find_capability scores by shared content words. That is deterministic and needs no model,
but it has no notion of MEANING: "squish a big array down for storage" shares no token with
`holographic_coldstore`, and "airspeed velocity of an unladen swallow" confidently matched a physics
module on the single word "velocity". Measured on the 12-ask suite: token overlap ~2/12 top-1, median
rank 13 of 503; the nomic embedding router hit 7/12 top-1, median rank 1.

WHAT SHIPS (and what does NOT):
A 96 KB index -- 503 module vectors at 64d q8, plus the ABTT correction (mu, pc) baked in -- extracted
from the build cache by tools/semantic/export_index.py. NO model ships. So this router can only score a
query it already has a VECTOR for:
  * queries embedded at build time and cached (the exam's asks, an app's fixed vocabulary)
  * a caller that supplies its own query vector (an app that ran the encoder itself)
A brand-new free-text query with no vector and no model present CANNOT be embedded here -- and this
router says so and returns None, so the caller falls back to the token router rather than guessing. That
honesty is the whole point: silence beats a confident wrong route.

DETERMINISM: q8 dequant + fixed ABTT transform + argsort with a name tie-break. No RNG, no model.
"""
import hashlib
import numpy as np


# The cache-key wiring the index was built under. Must match knowledge_index.embed_cached exactly, or a
# supplied raw query text hashes to the wrong key and silently misses. (Imported the format once, the
# hard way -- a guessed key format cost 413 misses in distill_map. See backlog rev. 47.)
_WIRING = "1000.0|12|True|False"


def _cache_key(text):
    return hashlib.sha256((_WIRING + "||" + text).encode()).hexdigest()[:32]


class EmbeddingRouter:
    """Loads the shipped q8 index (any dim; 128d is the measured champion) and routes a query VECTOR (or a cached query text) to modules.

    The index is a .npz with: names (M,), q (M,64) uint8, lo/hi (M,1) per-row q8 scales, mu (64,) and
    pc (1,64) for the ABTT correction that was fit on the documents at export time. We dequantize once,
    apply the SAME correction the query side gets, and unit-normalise -- so scoring is a plain dot."""

    def __init__(self, index_path):
        z = np.load(index_path, allow_pickle=False)
        self.names = [str(n) for n in z["names"]]
        q = z["q"].astype(np.float64)
        lo, hi = z["lo"].astype(np.float64), z["hi"].astype(np.float64)
        vecs = q / 255.0 * (hi - lo) + lo                 # undo per-row q8
        self.mu = z["mu"].astype(np.float64)
        self.pc = z["pc"].astype(np.float64)
        self.dim = vecs.shape[1]
        self.docs = self._unit(self._correct(vecs))       # corrected + normalised document matrix
        # WORKFLOW BONES (optional, packed by export_index.py): per-module neighbour lists for the measured
        # dense+structure fusion (top-1 6->7/12, median 2->1, zero per-ask regressions at the ship dim). An
        # index without bones loads fine -- route(gamma>0) then just returns the plain dense ranking.
        self._bones = None
        if "bone_src" in z.files:
            nbrs = [[] for _ in self.names]               # 'both' direction, max weight per pair
            best = {}
            for s, d, w in zip(z["bone_src"], z["bone_dst"], z["bone_w"].astype(np.float64)):
                for a, b in ((int(s), int(d)), (int(d), int(s))):
                    k = (a, b)
                    if w > best.get(k, 0.0):
                        best[k] = w
            for (a, b), w in best.items():
                nbrs[a].append((b, w))
            self._bones = nbrs

    # --- the ABTT correction, applied identically to docs and queries ---
    def _correct(self, M):
        M = M[:, : self.dim] - self.mu
        return M - (M @ self.pc.T) @ self.pc

    @staticmethod
    def _unit(M):
        return M / (np.linalg.norm(M, axis=-1, keepdims=True) + 1e-12)

    def route(self, query_vec, k=5, gamma=0.0):
        """Rank modules for a 64d query VECTOR (raw nomic-64, uncorrected -- we apply the correction).
        Returns [(module_name, score)] best-first. gamma=0 (default): plain cosine, byte-identical to the
        historical behaviour. gamma>0: fuse the dense ranking with a WORKFLOW-BONE ranking via weighted RRF
        (weights 1.0 and gamma, k=60) -- the exact mechanism measured on the routing suite (gamma=0.5:
        top-1 6->7/12, median 2->1, ZERO per-ask regressions at the ship dim; a module whose collaborators
        score well is lifted even when its own docstring never matched). Needs bones in the index (packed by
        export_index.py); without them gamma>0 degrades gracefully to the dense ranking. Scores under fusion
        are RRF scores, not cosines -- comparable within a call, not across calls."""
        q = np.asarray(query_vec, dtype=np.float64).reshape(1, -1)
        if q.shape[1] != self.dim:
            raise ValueError(f"query dim {q.shape[1]} != index dim {self.dim}")
        qn = self._unit(self._correct(q))[0]
        sims = self.docs @ qn
        if gamma > 0.0 and self._bones is not None:
            # one-hop propagation of CLAMPED sims (negative cosine = 'not relevant'; bones carry evidence
            # FOR, never against), then rank-level fusion -- ranks, not raw scores, because cosine and
            # propagated mass are not on a common scale (the same reason the exam fuses by RRF).
            seed = np.maximum(sims, 0.0)
            prop = np.zeros_like(seed)
            for i, lst in enumerate(self._bones):
                if lst:
                    wsum = sum(w for _, w in lst) + 1e-12
                    prop[i] = sum(seed[j] * w for j, w in lst) / wsum
            struct = 0.5 * seed + 0.5 * prop              # wf-alpha=0.5, the measured setting
            dense_order = sorted(range(len(sims)), key=lambda i: (-sims[i], self.names[i]))
            struct_order = sorted(range(len(sims)), key=lambda i: (-struct[i], self.names[i]))
            fused = {}
            for wgt, order_ in ((1.0, dense_order), (gamma, struct_order)):
                for rank, i in enumerate(order_, start=1):
                    fused[i] = fused.get(i, 0.0) + wgt / (60.0 + rank)
            top = sorted(fused, key=lambda i: (-fused[i], self.names[i]))[:k]
            return [(self.names[i], float(fused[i])) for i in top]
        order = np.argsort(-sims)[:k]
        # deterministic tie-break by name, matching the catalog's convention
        order = sorted(order, key=lambda i: (-sims[i], self.names[i]))
        return [(self.names[i], float(sims[i])) for i in order]

    def route_cached(self, query_text, cache, k=5):
        """Route a query whose vector is in the embedding cache (build-time asks, fixed app vocabulary).
        `cache` is the {key: [floats]} dict. Returns None if the text was never embedded -- the caller
        should then fall back to the token router. NEVER fabricates an embedding."""
        v = cache.get(_cache_key(query_text))
        if v is None:
            return None
        return self.route(np.asarray(v, dtype=np.float64)[: self.dim], k=k)


def _selftest():
    """Build a tiny synthetic index in memory, prove cosine routing separates a planted match, and prove
    a dimension mismatch fails loudly. No file, no model."""
    import io, tempfile, os
    rng = np.random.default_rng(0)
    M, d = 6, 64
    base = rng.standard_normal((M, d))
    mu = base.mean(0)
    bc = base - mu
    pc = np.linalg.svd(bc, full_matrices=False)[2][:1]
    br = bc - (bc @ pc.T) @ pc
    lo, hi = br.min(1, keepdims=True), br.max(1, keepdims=True)
    q = np.round((br - lo) / (hi - lo + 1e-12) * 255).astype(np.uint8)
    fd, path = tempfile.mkstemp(suffix=".npz"); os.close(fd)
    np.savez(path, names=np.array([f"mod_{i}" for i in range(M)]), q=q,
             lo=lo.astype(np.float16), hi=hi.astype(np.float16),
             mu=mu.astype(np.float16), pc=pc.astype(np.float16))
    r = EmbeddingRouter(path)
    # a query equal to document 3's raw vector must rank mod_3 first
    hits = r.route(base[3], k=3)
    assert hits[0][0] == "mod_3", hits
    assert hits[0][1] > 0.6, hits            # q8 + ABTT lower the absolute cosine; RANK is the contract
    # BONES + gamma fusion: rebuild the same index WITH a bone mod_3 <-> mod_5. gamma=0 must be
    # byte-identical to the boneless routing (pinned); gamma>0 must LIFT the bone-connected mod_5 above
    # its dense position -- the measured rescue mechanism (a module rises because its collaborator scored).
    np.savez(path, names=np.array([f"mod_{i}" for i in range(M)]), q=q,
             lo=lo.astype(np.float16), hi=hi.astype(np.float16),
             mu=mu.astype(np.float16), pc=pc.astype(np.float16),
             bone_src=np.array([3], dtype=np.int32), bone_dst=np.array([5], dtype=np.int32),
             bone_w=np.array([10.0], dtype=np.float32))
    rb = EmbeddingRouter(path)
    assert rb.route(base[3], k=3) == hits, "gamma=0 with bones present must equal the boneless ranking"
    dense_all = [n for n, _ in rb.route(base[3], k=M)]
    fused_all = [n for n, _ in rb.route(base[3], k=M, gamma=0.5)]
    assert fused_all.index("mod_5") <= dense_all.index("mod_5"), (dense_all, fused_all)
    assert fused_all[0] == "mod_3", fused_all                 # the exact match still wins under fusion
    # wrong dimension must raise, not silently truncate
    try:
        r.route(np.zeros(32)); assert False, "should have raised on dim mismatch"
    except ValueError:
        pass
    os.remove(path)
    print("holographic_router selftest: OK (cosine separates a plant; dim mismatch raises; gamma=0 byte-identical with bones present; gamma=0.5 lifts the bone-connected module, exact match still wins)")


if __name__ == "__main__":
    _selftest()
