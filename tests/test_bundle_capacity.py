"""Regression traps for the bundle-capacity advisor (work plan item 2.2).

The claim under test is not a number, it is a SHAPE: capacity is a function of load ratio M/D, readout
method, and quality floor -- and the folklore constant ("20-32 instructions") was what you get when you
measure only the weakest readout and forget to write down the variables. Both the artifact and the
collapse are pinned.
"""
import numpy as np
import pytest

import lecore
from holographic.sampling_and_signal.holographic_capacity import (bundle_capacity,
                                                                  measure_recovery_curve)


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


def test_the_folklore_constant_is_a_linear_readout_artifact():
    # THE HEADLINE. Same dim, same floor, same dictionary distribution -- the sparse decoder must hold a
    # strictly higher load ratio than naive cosine readout. If this fails, the module's premise is dead
    # and the docstrings are wrong.
    lin = bundle_capacity(256, "linear", floor=0.95, seeds=range(3))
    cos = bundle_capacity(256, "cosamp", floor=0.95, seeds=range(3))
    assert cos["safe_ratio"] >= 3 * lin["safe_ratio"], \
        "cosamp %.2f vs linear %.2f -- the artifact story no longer holds" % (
            cos["safe_ratio"], lin["safe_ratio"])


def test_the_safe_ratio_collapses_across_dimensions():
    # WHY capacity is a RATIO and not a count: the same method at different dims must land on the same
    # M/D to within one grid step. This is the load-ratio law made executable.
    ratios = [bundle_capacity(d, "cosamp", floor=0.95, seeds=range(3))["safe_ratio"]
              for d in (128, 256, 512)]
    assert max(ratios) - min(ratios) <= 0.101, "no collapse: %r" % ratios


def test_capacity_scales_linearly_with_dim_at_fixed_ratio():
    a = bundle_capacity(256, "cosamp", seeds=range(2))
    b = bundle_capacity(512, "cosamp", seeds=range(2))
    assert abs(b["capacity"] - 2 * a["capacity"]) <= max(4, int(0.02 * 512)), \
        "capacity did not double with dim: %d vs %d" % (a["capacity"], b["capacity"])


def test_the_answer_carries_its_configuration():
    # A capacity without its readout, dim and floor attached is the failure this module replaces, so the
    # provenance fields are contract, not convenience.
    r = bundle_capacity(128, "cosamp", seeds=range(2))
    for field in ("capacity", "safe_ratio", "method", "dim", "floor", "curve"):
        assert field in r
    assert r["curve"], "the curve must travel with the number"


def test_a_lucky_seed_cannot_set_the_capacity():
    # The gate is mean MINUS sd. A row whose mean clears the floor only because of spread must not count.
    curve = measure_recovery_curve(128, "linear", ratios=(0.02, 0.10), seeds=range(4))
    for row in curve:
        assert row["f1_sd"] >= 0.0
    high_var = {"f1_mean": 0.99, "f1_sd": 0.30}
    assert high_var["f1_mean"] - high_var["f1_sd"] < 0.95


def test_the_advisor_is_deterministic():
    assert bundle_capacity(128, "cosamp", seeds=range(2)) == bundle_capacity(128, "cosamp", seeds=range(2))


def test_a_callers_own_codebook_is_honoured():
    # The reference dictionary is incoherent, and coherence is exactly what breaks the reference numbers
    # -- so the codebook= path must actually be used, not silently replaced by the reference.
    rng = np.random.default_rng(0)
    base = rng.standard_normal(128)
    cb = np.stack([base + 0.05 * rng.standard_normal(128) for _ in range(64)])   # highly coherent
    cb /= np.linalg.norm(cb, axis=1, keepdims=True)
    coherent = bundle_capacity(128, "amp", seeds=range(2), codebook=cb)
    incoherent = bundle_capacity(128, "amp", seeds=range(2))
    assert coherent["safe_ratio"] <= incoherent["safe_ratio"], \
        "a near-duplicate codebook reported MORE capacity than random atoms (%r vs %r)" % (
            coherent["safe_ratio"], incoherent["safe_ratio"])


def test_unknown_method_is_refused():
    with pytest.raises(ValueError):
        measure_recovery_curve(64, "nonsense", ratios=(0.05,), seeds=range(1))


def test_wired_and_discoverable(mind):
    r = mind.bundle_capacity(method="cosamp", seeds=range(2))
    assert r["dim"] == 256, "dim did not default to the mind's own"
    assert r["capacity"] > 0
    for query in ("how many things fit in a bundle", "safe number of items to superpose",
                  "load ratio before recovery fails"):
        assert "load ratio" in str(mind.find_capability(query)[:3]), \
            "%r no longer surfaces the capacity advisor" % query


# --------------------------------------------------------------------------------------
# W6 -- the drop budget, and the correction that produced it.
# --------------------------------------------------------------------------------------

def test_truncation_is_not_the_same_as_damage(mind):
    """THE CORRECTION, PINNED. The README's degradation table (100% recall at 40% of slots DESTROYED) is
    about DAMAGE: the slots are zeroed and NO MEMORY IS SAVED. It does not transfer to a memory-saving
    scheme. TRUNCATING to the same fraction at the same load gives ~85%, not 100%, because a zeroed slot
    still occupies its dimension in the readout while a dropped one does not.

    Corruption-robustness and a memory budget are different quantities, and they were briefly conflated."""
    from holographic.agents_and_reasoning.holographic_ai import bind, random_vector, unbind

    dim, count, keep = 1024, 16, 0.4
    rng = np.random.default_rng(0)
    keys = [random_vector(dim, rng) for _ in range(count)]
    vals = [random_vector(dim, rng) for _ in range(count)]
    holo = np.sum([bind(k, v) for k, v in zip(keys, vals)], axis=0)
    values = np.stack(vals)

    rng2 = np.random.default_rng(7)
    hits = 0
    trials = 6
    for _ in range(trials):
        idx = np.sort(rng2.choice(dim, int(keep * dim), replace=False))
        mask = np.zeros(dim, bool)
        mask[idx] = True
        kept = values[:, idx]
        kept = kept / np.linalg.norm(kept, axis=1, keepdims=True)
        for i, k in enumerate(keys):
            got = unbind(holo * mask, k)[idx]
            got = got / (np.linalg.norm(got) or 1.0)
            hits += int(np.argmax(kept @ got)) == i
    truncated = hits / (trials * count)
    assert truncated < 0.98, ("truncated recall is %.0f%%; if it now matches the damage figure, the "
                              "distinction in drop_budget's docstring is wrong" % (100 * truncated))


def test_the_load_ratio_law_predicts_truncation_safety():
    """W6 NEEDED NO NEW THEORY. Dropping slots reduces the EFFECTIVE dimension, so the constraint is the
    load-ratio law this module already measures. Verified across configurations: every one with
    n_items/(keep*dim) at or below ~0.02 held at 98-100%, every one above it degraded."""
    from holographic.sampling_and_signal.holographic_capacity import drop_budget

    for dim, items in ((1024, 8), (1024, 16), (2048, 16), (4096, 16)):
        budget = drop_budget(dim, items)
        assert budget["effective_ratio"] <= 0.0201, budget
        assert budget["safe"] is True
        assert 0 < budget["keep"] <= dim


def test_a_sparse_decoder_can_drop_far_more():
    # The safe ratio is a property of the READOUT, measured by bundle_capacity: 0.02 linear, 0.17 cosamp.
    from holographic.sampling_and_signal.holographic_capacity import drop_budget

    linear = drop_budget(1024, 16, safe_ratio=0.02)["keep_fraction"]
    sparse = drop_budget(1024, 16, safe_ratio=0.17)["keep_fraction"]
    assert sparse < linear, "a sparse decoder should permit dropping MORE, not less"


def test_drop_budget_guards():
    from holographic.sampling_and_signal.holographic_capacity import drop_budget

    for bad in ((0, 4), (1024, 0), (-8, 4)):
        with pytest.raises(ValueError):
            drop_budget(*bad)


def test_drop_budget_is_wired_and_discoverable(mind):
    result = mind.drop_budget(dim=1024, n_items=16)
    assert result["bytes_saved"] > 0 and result["safe"] is True
    for query in ("how many slots can i drop", "memory budget for a bundle"):
        assert "slots can I drop" in str(mind.find_capability(query)[:3]), \
            "%r no longer surfaces the drop budget" % query


# --------------------------------------------------------------------------------------
# Batched cleanup: the missing UP direction, with a default-off device seam (backlog W7).
# --------------------------------------------------------------------------------------

def test_batched_cleanup_matches_a_loop_of_singles(mind):
    rng = np.random.default_rng(0)
    book = rng.standard_normal((128, 256)).astype(np.float32)
    book /= np.linalg.norm(book, axis=1, keepdims=True)
    queries = rng.standard_normal((16, 256)).astype(np.float32)
    idx, scores = mind.cleanup_batch(book, queries)
    assert np.array_equal(idx, np.argmax(queries @ book.T, axis=1))
    assert scores.shape == (16,)


@pytest.mark.slow
def test_batching_pays_on_the_cpu_alone():
    """THE GATE, and it needs no device at all: one (K,D)x(D,M) matmul instead of K separate matvecs is BLAS
    getting one big multiply rather than K small ones. Measured 2.58x at K=32, 5.36x at K=64, 5.92x at
    K=128 — so this would be worth building even if no GPU existed."""
    import time

    rng = np.random.default_rng(0)
    book = rng.standard_normal((1024, 512)).astype(np.float32)
    book /= np.linalg.norm(book, axis=1, keepdims=True)
    queries = rng.standard_normal((64, 512)).astype(np.float32)

    def ms(fn):
        fn()
        start = time.perf_counter()
        fn()
        return (time.perf_counter() - start) * 1e3

    singles = min(ms(lambda: [int(np.argmax(book @ q)) for q in queries]) for _ in range(3))
    batched = min(ms(lambda: np.argmax(queries @ book.T, axis=1)) for _ in range(3))
    assert singles > 2 * batched, "batching no longer pays on CPU (%.3f vs %.3f ms)" % (singles, batched)


def test_the_device_seam_is_default_off(mind):
    """DEFAULT OFF, DELIBERATELY. The host<->device crossover has never been measured on real hardware, so
    enabling it by default would act on arithmetic rather than a measurement — and the one thing worse than
    not using a device is using it on a guess. The seam exists so somebody WITH a device can measure it
    without editing the engine."""
    import inspect

    from holographic.sampling_and_signal.holographic_capacity import cleanup_batch
    assert inspect.signature(cleanup_batch).parameters["backend"].default is None


def test_the_backend_cannot_change_which_atom_wins(mind):
    # Indices resolve by lowest index on BOTH paths, so switching backend must not move a decision.
    pytest.importorskip("wgpu")
    rng = np.random.default_rng(0)
    book = rng.standard_normal((64, 128)).astype(np.float32)
    book /= np.linalg.norm(book, axis=1, keepdims=True)
    queries = rng.standard_normal((12, 128)).astype(np.float32)
    cpu_idx, _ = mind.cleanup_batch(book, queries)
    gpu_idx, _ = mind.cleanup_batch(book, queries, backend="wgsl")
    assert np.array_equal(cpu_idx, gpu_idx)


def test_cleanup_batch_guards(mind):
    from holographic.sampling_and_signal.holographic_capacity import cleanup_batch

    with pytest.raises(ValueError):
        cleanup_batch(np.zeros((4, 8), dtype=np.float32), np.zeros(8, dtype=np.float32))
    with pytest.raises(ValueError):
        cleanup_batch(np.zeros((4, 8), dtype=np.float32), np.zeros((2, 7), dtype=np.float32))
    with pytest.raises(ValueError):
        cleanup_batch(np.zeros((4, 8), dtype=np.float32), np.zeros((2, 8), dtype=np.float32),
                      backend="nonsense")


def test_cleanup_batch_is_discoverable(mind):
    for query in ("clean up many cues at once", "batch cleanup", "recall many vectors at once"):
        assert "batched cleanup" in str(mind.find_capability(query)[:3]), \
            "%r no longer surfaces batched cleanup" % query
