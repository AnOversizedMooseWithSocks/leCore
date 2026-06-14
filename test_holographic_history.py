"""Versioned compressed history: lossless rollback, reorganization compression,
proof-gated commits, and the honest dense-update boundary."""
import numpy as np

from holographic_history import VersionedStore


def _seed_store(vs, n=10, dim=64, seed=0):
    rng = np.random.default_rng(seed)
    ids = [vs.new_id() for _ in range(n)]
    rows = {i: rng.standard_normal(dim) for i in ids}
    vs.commit(rows, list(ids))
    return rows, list(ids), rng


def test_checkout_is_lossless_at_every_version():
    # Any past version reconstructs EXACTLY (rollback needs exact state, so the
    # deltas are lossless -- unlike video's lossy spectral coding).
    vs = VersionedStore(dim=64, gop_len=5)
    rows, order, rng = _seed_store(vs, dim=64)
    snapshots = [(dict(rows), list(order))]
    for _ in range(12):                              # a run of structural edits
        op = rng.integers(3)
        rows = dict(rows)
        if op == 0:                                  # insert
            j = vs.new_id(); rows[j] = rng.standard_normal(64); order = order + [j]
        elif op == 1 and len(order) > 3:             # delete (merge)
            d = order[int(rng.integers(len(order)))]; rows.pop(d)
            order = [x for x in order if x != d]
        else:                                        # mutate one row (split-ish)
            m = order[int(rng.integers(len(order)))]; rows[m] = rows[m] + 0.1 * rng.standard_normal(64)
        vs.commit(rows, list(order))
        snapshots.append((dict(rows), list(order)))
    for v, (exp_rows, exp_order) in enumerate(snapshots):
        got_rows, got_order = vs.checkout(v)
        assert got_order == exp_order
        for i in exp_order:
            assert np.array_equal(got_rows[i], exp_rows[i])


def test_reorganization_history_compresses_losslessly():
    # Sparse structural edits make the history compress vs storing every snapshot
    # whole -- the inter-version redundancy is real (row-keyed so a delete is one
    # id, not a whole-matrix realignment).
    vs = VersionedStore(dim=128, gop_len=16)
    rows, order, rng = _seed_store(vs, n=20, dim=128)
    for _ in range(30):
        rows = dict(rows)
        j = vs.new_id(); rows[j] = rng.standard_normal(128); order = order + [j]
        vs.commit(rows, list(order))
    assert vs.full_entries() > 3 * vs.stored_entries()   # clearly compressed


def test_proof_gated_commit_rejects_bad_reorg():
    # A reorganization that violates an invariant is REJECTED; the store stays at
    # the last valid version, and the rejected attempt is still in the audit log.
    vs = VersionedStore(dim=64, gop_len=8)

    def coherent(rows, order):
        return (len(set(order)) == len(order)
                and all(np.isfinite(np.linalg.norm(rows[i])) and np.linalg.norm(rows[i]) > 1e-6
                        for i in order))

    rng = np.random.default_rng(2)
    ids = [vs.new_id() for _ in range(6)]
    rows = {i: rng.standard_normal(64) for i in ids}
    order = list(ids)
    assert vs.commit(rows, order, proof=coherent) == 0
    bad = dict(rows); bad[order[0]] = np.zeros(64)        # degenerate prototype
    assert vs.commit(bad, list(order), proof=coherent, note="buggy") == -1
    assert vs.head() == 0                                 # store unchanged
    audit = vs.history()
    assert audit[-1]["accepted"] is False                 # the attempt is recorded


def test_rollback_restores_exact_state_and_keeps_history():
    # rollback returns the live state to a past version EXACTLY, recorded as a new
    # commit (the act of rolling back is itself history -- nothing is erased).
    vs = VersionedStore(dim=64, gop_len=8)
    rows, order, rng = _seed_store(vs, dim=64, seed=3)
    base_rows, base_order = vs.checkout(0)
    for _ in range(4):
        rows = dict(rows); j = vs.new_id(); rows[j] = rng.standard_normal(64); order = order + [j]
        vs.commit(rows, list(order))
    rv = vs.rollback(0)
    got_rows, got_order = vs.checkout(rv)
    assert got_order == base_order
    for i in base_order:
        assert np.array_equal(got_rows[i], base_rows[i])
    assert len(vs.history()) == 6                          # 1 + 4 + rollback all kept


def test_dense_update_does_not_compress_and_we_say_so():
    # THE HONEST BOUNDARY (the 'deformation' analog): a dense step that nudges
    # EVERY row changes 100% of the matrix, so delta coding cannot compress it --
    # versioning is for structural history, not dense trajectories.
    vs = VersionedStore(dim=64, gop_len=100)              # force all-delta
    rng = np.random.default_rng(4)
    ids = [vs.new_id() for _ in range(10)]
    rows = {i: rng.standard_normal(64) for i in ids}
    order = list(ids)
    vs.commit(rows, order)
    for _ in range(8):
        rows = {i: rows[i] + 0.01 * rng.standard_normal(64) for i in order}  # dense
        vs.commit(dict(rows), list(order))
    # every row changes every step -> deltas as big as snapshots, no compression
    assert vs.stored_entries() >= 0.9 * vs.full_entries()
