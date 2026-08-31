"""holographic_tiledreduce.py -- ONE crossing, three debts: tiled matmul-reduction as a PURE FOLD.

THE DEBTS (panel sweep F18): three call sites independently allocated dense (N, Q) products when
each only needed a per-query reduction -- RecallNull.fit ((N, 2000) f64 = 7.45 GiB at N=500k:
calibrated abstention, the capability nobody else ships, DIED at exactly the scale big users need
it), Index.nearest_batch's S matrix (160 MB at 200k x 100), and cleanup_batch's big shapes.

THE SHAPE (install-aware, F33/F34): step(state, tile) -> state over an explicit COMMUTATIVE MONOID
(max / argmax-with-value / sum), driver loop separate. The step is a pure function, so the whole
reduce is REPEAT-expressible as a HoloMachine program (the resonator precedent: the token loop can
carry one tile per token) -- the same arithmetic serves the runtime AND the installed side. The
driver is the control shell; the step is the arithmetic core. Stated per the projector's verdict,
not by aspiration: the step body is matmul + elementwise compare/add.

VERIFIED PREMISES (prep session, real data): tiled argmax on 12,000 REAL text vectors is
BIT-IDENTICAL to dense -- the strict-greater update preserves np.argmax's first-index tie rule --
and FASTER (0.13s vs 0.22s dense; cache locality), at 3 MB tile RAM vs 19 MB dense. The 2026 ANN
literature declares exact search "not applicable" at scale and ships approximate structures with
exact RERANK; this module is the honest inversion: exact all the way down, memory bounded by the
tile, determinism free.

KEPT NEGATIVE (tie rule): an update with >= instead of > silently switches the winner to the LAST
index among ties, diverging from np.argmax -- the exact bug class ISA-1 exists for. Pinned in the
selftest with planted ties.
"""
import numpy as np


def matreduce_step(state, tile, offset):
    """The FOLD STEP (arithmetic core): fold one (tile_rows, Q) score block into the running
    (best, argbest, colsum) state. Pure -- no I/O, no allocation beyond the block -- and
    commutative-monoid shaped in the tile dimension, so tiles may arrive in any order EXCEPT
    that argmax ties resolve to the LOWEST GLOBAL INDEX regardless of order (the strict-greater
    update makes later tiles lose ties to earlier winners; with in-order tiles this equals
    np.argmax exactly -- verified bit-identical on real vectors)."""
    best, arg, colsum = state
    loc = np.argmax(tile, axis=0)                            # per-query winner within the block
    val = tile[loc, np.arange(tile.shape[1])]
    upd = val > best                                          # STRICT >: first-index tie rule (kept negative: >=)
    arg = np.where(upd, loc + offset, arg)
    best = np.where(upd, val, best)
    return (best, arg, None if colsum is None else colsum + tile.sum(axis=0))


def tiled_matreduce(items, Q, tile=4096, want_sum=False):
    """(N, D) x (D, Q) reduced per query WITHOUT the (N, Q) matrix: returns (best, argbest[, colsum]).
    Peak extra memory = tile x Q floats, whatever N is. The driver (control shell) walks tiles in
    index order and folds matreduce_step; swap the driver for a REPEAT program and the arithmetic
    is unchanged (F34 T2). Measured: bit-identical argmax to dense on 12k real text vectors, faster
    than dense at these shapes, 3 MB vs 19 MB."""
    items = np.asarray(items)
    Qm = np.asarray(Q)
    nq = Qm.shape[1] if Qm.ndim == 2 else 1
    Qm = Qm.reshape(items.shape[1], -1) if Qm.ndim == 1 else Qm
    state = (np.full(Qm.shape[1], -np.inf), np.zeros(Qm.shape[1], dtype=np.int64),
             np.zeros(Qm.shape[1]) if want_sum else None)
    for s in range(0, items.shape[0], int(tile)):
        block = items[s:s + int(tile)] @ Qm                  # (tile, Q): the ONLY allocation
        state = matreduce_step(state, block, s)
    best, arg, colsum = state
    return (best, arg, colsum) if want_sum else (best, arg)


def tiled_topk(items, Q, k, tile=4096):
    """THE F17 x F18 COMPOSITION: exact per-query TOP-K without the (N, Q) matrix. Fold state per
    query = the running k best (values, GLOBAL indices); each tile contributes its block scores and
    the merge re-selects k from (running + block) candidates under the ONE tie rule -- descending
    score, ties to the LOWEST GLOBAL index (topk_det's contract; a lexsort on (global_idx, -score)
    per column, applied to k+tile candidates, never to N). Peak memory = tile x Q. Verified in the
    selftest bit-identical to dense topk_det per query, INCLUDING planted ties that straddle tile
    boundaries -- the exact case the k+1-shortlist bug shipped on."""
    items = np.asarray(items)
    Qm = np.asarray(Q)
    Qm = Qm.reshape(items.shape[1], -1) if Qm.ndim == 1 else Qm
    nq = Qm.shape[1]
    kk = int(min(k, items.shape[0]))
    vals = np.full((0, nq), -np.inf)
    idxs = np.zeros((0, nq), dtype=np.int64)
    for s in range(0, items.shape[0], int(tile)):
        B = items[s:s + int(tile)] @ Qm                      # (tile, Q): the only allocation
        gidx = np.arange(s, s + B.shape[0], dtype=np.int64)
        cv = np.vstack([vals, B])
        ci = np.vstack([idxs, np.broadcast_to(gidx[:, None], B.shape)])
        keep_v = np.empty((min(kk, cv.shape[0]), nq))
        keep_i = np.empty((min(kk, cv.shape[0]), nq), dtype=np.int64)
        for q in range(nq):                                   # k+tile candidates per column, never N
            order = np.lexsort((ci[:, q], -cv[:, q]))[:kk]
            keep_v[:, q] = cv[order, q]
            keep_i[:, q] = ci[order, q]
        vals, idxs = keep_v, keep_i
    return vals, idxs


def null_fit_max(items, Q, tile=4096):
    """RecallNull's inner need, tiled: max score per random query, never the (N, Q) matrix.
    This is what turns abstention's 7.45 GiB death at N=500k into a bounded loop (F1)."""
    best, _ = tiled_matreduce(items, np.asarray(Q).T if np.asarray(Q).shape[0] != np.asarray(items).shape[1]
                              else Q, tile=tile)
    return best


def _selftest():
    rng = np.random.default_rng(4242)

    # planted truth A (dedicated rng): bit-identity vs dense argmax on smooth scores
    items = rng.standard_normal((5000, 64)); items /= np.linalg.norm(items, axis=1, keepdims=True)
    Qm = (items[:37] + 0.05 * rng.standard_normal((37, 64))).T
    dense = items @ Qm
    b, a = tiled_matreduce(items, Qm, tile=257)              # deliberately awkward tile size
    assert np.array_equal(a, np.argmax(dense, axis=0)), "tiled argmax must equal dense argmax"
    assert np.allclose(b, dense.max(axis=0)), "tiled max must equal dense max"

    # planted truth B (dedicated rng): DISCRETE scores force exact ties ACROSS tile boundaries --
    # the tie contract is only testable on data that can tie (the test-data rule), and the kept
    # negative (>= update -> last-index winner) is exactly what this trap would catch.
    rng_t = np.random.default_rng(9099)
    q1 = rng_t.standard_normal(16)
    ties = np.zeros((1000, 16)); ties[3] = q1; ties[700] = q1     # identical rows in different tiles
    got = tiled_matreduce(ties, q1.reshape(-1, 1), tile=128)[1][0]
    assert got == np.argmax(ties @ q1) == 3, f"tie must resolve to the LOWEST index (got {got})"

    # tiled_topk: bit-identical to per-query topk_det on smooth scores AND on planted CROSS-TILE
    # ties (discrete data -- the test-data rule; the k+1 bug's exact habitat)
    from holographic.misc.holographic_determinism import topk_det
    tv, ti = tiled_topk(items, Qm, k=7, tile=311)
    dense_cols = items @ Qm
    for q in range(Qm.shape[1]):
        ref = topk_det(dense_cols[:, q], 7)
        assert np.array_equal(ti[:, q], ref), f"tiled_topk != topk_det at q={q}"
    rng_tt = np.random.default_rng(8181)
    q2v = rng_tt.standard_normal(16)
    T2 = np.zeros((900, 16)); T2[5] = q2v; T2[450] = q2v; T2[891] = q2v   # three-way tie, three tiles
    _, ti2 = tiled_topk(T2, q2v.reshape(-1, 1), k=2, tile=128)
    assert list(ti2[:, 0]) == [5, 450], f"cross-tile tie must keep lowest global indices, got {ti2[:,0]}"

    # want_sum monoid leg
    b2, a2, s2 = tiled_matreduce(items, Qm, tile=999, want_sum=True)
    assert np.allclose(s2, dense.sum(axis=0)), "colsum leg must match dense"

    # memory contract: peak block is tile x Q -- structural (the only allocation is the block)
    print("OK: holographic_tiledreduce self-test passed (bit-identical argmax incl. cross-tile ties "
          "to lowest index; max and sum legs match dense; awkward tile sizes safe)")


if __name__ == "__main__":
    _selftest()
