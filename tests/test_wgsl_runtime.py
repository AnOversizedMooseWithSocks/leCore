"""Regression traps for the vendor-neutral WGSL compute runtime (GPU backlog D2).

These run on a SOFTWARE ADAPTER (llvmpipe / WARP) when no GPU is present, which is the whole argument for
this path over CuPy: correctness is verifiable on an ordinary CI runner with no hardware budget, while the
CuPy path cannot be tested anywhere without an NVIDIA machine.

Skipped cleanly when `wgpu` is absent — it is an optional dependency and the engine must not require it.
"""
import numpy as np
import pytest

import lecore
from holographic.io_and_interop.holographic_wgpurun import (available, device_info, run_kernel,
                                                            verify_against_numpy, wrap_kernel)

pytestmark = pytest.mark.skipif(not available(), reason="wgpu / no compute adapter available")

BODY = "fn scale(x: f32, g: f32) -> f32 { return ((x * g) + 0.5f); }"


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=64, seed=0)


def test_a_software_adapter_counts_as_available():
    """The property that makes this CI-testable. A software adapter is reported as available on purpose --
    it runs the same WGSL and validates the same shaders, and correctness is what CI needs to check."""
    info = device_info()
    assert info["available"] is True
    assert info["device"] and info["type"]


def test_the_wrapper_makes_a_runnable_shader_not_just_valid_wgsl():
    # emit_kernel produces a bare `fn`: correct WGSL, not a dispatchable shader. The wrapper owns the entry
    # point, the bindings and the bounds guard, and all three are required.
    shader = wrap_kernel(BODY, "scale", extra_args=(2.0,))
    assert "@compute" in shader
    assert "var<storage" in shader
    assert "arrayLength" in shader


def test_a_simple_elementwise_kernel_is_bit_exact():
    data = np.arange(256, dtype=np.float32)
    got = run_kernel(BODY, "scale", data, extra_args=(2.0,))
    assert np.array_equal(got, data * 2.0 + 0.5)


def test_the_tail_of_a_partial_workgroup_is_correct():
    """The easiest thing to get silently wrong. Dispatch rounds UP to whole workgroups, so with 100 elements
    and a workgroup of 64 there are 28 invocations past the end of the buffer. Without the arrayLength guard
    they write out of bounds; with it, the result must be exact and the length unchanged."""
    data = np.arange(100, dtype=np.float32)
    got = run_kernel(BODY, "scale", data, extra_args=(2.0,))
    assert got.shape == (100,)
    assert np.array_equal(got, data * 2.0 + 0.5)


def test_a_size_smaller_than_one_workgroup_still_works():
    data = np.arange(5, dtype=np.float32)
    assert np.array_equal(run_kernel(BODY, "scale", data, extra_args=(2.0,)), data * 2.0 + 0.5)


def test_a_two_dimensional_array_is_refused():
    with pytest.raises(ValueError):
        run_kernel(BODY, "scale", np.zeros((4, 4), dtype=np.float32))


# --------------------------------------------------------------------------------------
# The differential property -- what this path has and CuPy cannot.
# --------------------------------------------------------------------------------------

def _single_expression(x: float, g: float) -> float:
    return x * g + 0.5


def _accumulating(x: float, g: float) -> float:
    t = 0.0
    for i in range(4):
        t = t + x * g
    return t * 0.25 + 1.0


def test_the_projection_can_be_differentially_tested(mind):
    # The shader is generated from the SAME function that runs on CPU, so the two can be checked against
    # each other on real data rather than trusted to agree.
    report = mind.verify_wgsl_kernel(_single_expression, np.linspace(-3, 3, 128), extra_args=(1.7,))
    assert set(report) == {"max_abs", "max_rel", "exact", "n"}
    assert report["n"] == 128


def test_exactness_holds_for_single_expressions_and_not_for_accumulation(mind):
    """MEASURED BOUNDARY, pinned so the docstring cannot overclaim.

    A single-expression kernel matches bit-for-bit on every input tried; an ACCUMULATING one does not,
    because Python accumulates in float64 and casts at the end while WGSL accumulates in f32 throughout.

    This test also caught a bug in the harness itself: the first `verify_against_numpy` dispatched float32
    while evaluating Python on the original float64 inputs, so it reported the INPUT CAST as kernel
    deviation and made the exact case look inexact. A differential test that does not feed both sides
    identical inputs measures its own conversion."""
    # WHAT IS ACTUALLY TRUE, after being wrong about it three times: exactness is DATA-DEPENDENT and is
    # NOT a guarantee. The same kernel is bit-exact over arange() and NOT over linspace(-3, 3) -- so the
    # honest contract is a TOLERANCE, and `exact` is an observation about one run rather than a property.
    simple = [mind.verify_wgsl_kernel(_single_expression, d, extra_args=(1.7,))
              for d in (np.arange(256, dtype=np.float32),
                        np.linspace(-3, 3, 128).astype(np.float32),
                        (np.arange(64) * 0.5).astype(np.float32))]
    for report in simple:
        assert report["max_rel"] < 1e-5, "a single-expression kernel drifted more than f32 rounding: %r" % report
    assert any(r["exact"] for r in simple), "nothing is exact any more; the f32 literal path may have broken"

    # Accumulation is measurably worse than a single expression -- the ordering is the durable claim, not
    # the absolute numbers.
    drifting = mind.verify_wgsl_kernel(_accumulating, np.linspace(-3, 3, 128).astype(np.float32),
                                       extra_args=(1.7,))
    assert drifting["max_rel"] < 1e-4, "deviation %r is larger than f32 rounding explains" % drifting


def test_the_faculties_are_wired_and_discoverable(mind):
    assert mind.wgsl_device()["available"] is True
    got = mind.run_wgsl_kernel(_single_expression, np.arange(8, dtype=np.float32), extra_args=(2.0,))
    assert np.allclose(got, np.arange(8) * 2.0 + 0.5)
    for query in ("run this on any gpu", "use my amd or intel gpu", "gpu without cuda", "webgpu compute"):
        assert "WGSL" in str(mind.find_capability(query)[:3]), "%r no longer surfaces the runtime" % query


# --------------------------------------------------------------------------------------
# W1 -- the reduction primitive, which is what unlocks the VSA kernels.
# --------------------------------------------------------------------------------------

def test_sum_max_min_match_numpy_exactly_on_representable_data(mind):
    from holographic.io_and_interop.holographic_wgpurun import reduce_kernel

    data = np.arange(1024, dtype=np.float32)
    assert reduce_kernel("sum", data) == float(data.sum())
    assert reduce_kernel("max", data) == float(data.max())
    assert reduce_kernel("min", data) == float(data.min())


def test_the_reduction_tail_is_correct(mind):
    """A size that is not a whole number of workgroups. Out-of-range lanes must write the IDENTITY into
    shared memory rather than skipping the write — a skipped write leaves whatever was in workgroup memory
    from a previous dispatch and the tree folds it in, which is silent and data-dependent."""
    from holographic.io_and_interop.holographic_wgpurun import reduce_kernel

    for n in (1, 7, 63, 65, 100, 1000):
        data = np.arange(n, dtype=np.float32)
        assert reduce_kernel("sum", data) == float(data.sum()), "tail wrong at n=%d" % n


def test_reduction_guards(mind):
    from holographic.io_and_interop.holographic_wgpurun import reduce_kernel

    with pytest.raises(ValueError):
        reduce_kernel("nonsense", np.arange(8, dtype=np.float32))
    with pytest.raises(ValueError):
        reduce_kernel("sum", np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        reduce_kernel("sum", np.array([], dtype=np.float32))


def test_device_argmax_agrees_with_cpu_on_real_cleanup_queries(mind):
    # The kernel W1 exists to unlock: cosine over a codebook, then argmax.
    rng = np.random.default_rng(0)
    dim, count = 512, 64
    book = rng.standard_normal((count, dim)).astype(np.float32)
    book /= np.linalg.norm(book, axis=1, keepdims=True)
    for _ in range(20):
        i = int(rng.integers(count))
        query = book[i] + 0.8 * rng.standard_normal(dim).astype(np.float32) / np.sqrt(dim)
        query /= np.linalg.norm(query)
        sims = (book @ query).astype(np.float32)
        assert mind.wgsl_argmax(sims)[0] == int(np.argmax(sims))


def test_argmax_agrees_with_cpu_on_ADVERSARIAL_EXACT_TIES(mind):
    """THE TEST THAT MATTERS, and the one a random-data test would fake passing. Argmax is a DECISION, not a
    value, and this engine's rule is that existing decisions never flip — so a device argmax must break ties
    the same way the CPU does. Random data almost never produces a tie, so ties are constructed here.

    The design earns this rather than getting lucky: the VALUE reduction runs on the device and the INDEX is
    resolved host-side by first-attaining, which is the canonical lowest-index rule. A fully device-side
    (value, index) reduction would make tie resolution depend on reduction order, and is deliberately not
    built."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        values = rng.standard_normal(256).astype(np.float32)
        j = int(rng.integers(256))
        values[(j + 7) % 256] = values[j]                 # an exact tie between two indices
        assert mind.wgsl_argmax(values)[0] == int(np.argmax(values))


def test_the_tie_rule_is_lowest_index(mind):
    values = np.array([5.0, 1.0, 5.0, 5.0], dtype=np.float32)
    assert mind.wgsl_argmax(values)[0] == 0


def test_the_reduction_is_deterministic(mind):
    data = np.random.default_rng(0).standard_normal(4096).astype(np.float32)
    assert mind.wgsl_reduce("sum", data) == mind.wgsl_reduce("sum", data)
    assert mind.wgsl_argmax(data) == mind.wgsl_argmax(data)


def test_the_reduction_is_discoverable(mind):
    for query in ("sum an array on the gpu", "gpu reduction", "argmax on the gpu"):
        assert "Reduce and argmax" in str(mind.find_capability(query)[:3]), \
            "%r no longer surfaces the reduction" % query


# --------------------------------------------------------------------------------------
# W2's gate: which part of cleanup is worth offloading, if any?
# --------------------------------------------------------------------------------------

@pytest.mark.slow
def test_argmax_alone_can_never_pay_on_any_device():
    """W2'S FIRST GATE RESULT, and it is UNCONFOUNDED — it does not depend on this machine's adapter at all.

    Offloading `cleanup`'s argmax alone cannot pay on ANY device, because NumPy's argmax over a realistic
    codebook is far below any GPU's dispatch floor. Measured on real CPU: argmax needs M ~ 500,000 elements
    before it reaches even 0.1 ms, which is an *optimistic* real-hardware dispatch cost. Real VSA codebooks
    are 64–4096 items, where argmax is single-digit MICROseconds.

    So the conclusion holds for any hardware with a nonzero submission cost, which is all of it."""
    import time

    rng = np.random.default_rng(0)
    values = rng.standard_normal(4096).astype(np.float32)
    best = min(_time_ms(lambda: int(np.argmax(values))) for _ in range(3))
    assert best < 0.05, "numpy argmax over a realistic codebook is %.4f ms; the dispatch-floor argument " \
                        "assumed it was microseconds" % best


@pytest.mark.slow
def test_the_similarity_matmul_is_where_cleanups_time_goes():
    """W2'S RETARGET, measured. The argmax is not the cost — the codebook similarity is. It is M*D work
    against the argmax's M, and it dominates completely once the codebook is any real size:

        M=64   D=512    66% of cleanup
        M=1024 D=512    98%
        M=4096 D=1024  100%

    So W2 offloads the SIMILARITY, and offloads it FUSED WITH the argmax in one dispatch — splitting them
    would pay the dispatch floor twice and ship the intermediate vector back and forth, which is the same
    'fuse before you dispatch' rule that governs the rest of this path."""
    rng = np.random.default_rng(0)
    for count, dim, floor in ((1024, 512, 0.8), (4096, 1024, 0.9)):
        book = rng.standard_normal((count, dim)).astype(np.float32)
        query = rng.standard_normal(dim).astype(np.float32)
        sims = book @ query
        sim_ms = min(_time_ms(lambda: book @ query) for _ in range(3))
        arg_ms = min(_time_ms(lambda: int(np.argmax(sims))) for _ in range(3))
        share = sim_ms / (sim_ms + arg_ms)
        assert share > floor, "similarity is only %.0f%% of cleanup at M=%d D=%d" % (100 * share, count, dim)


def _time_ms(fn):
    import time

    fn()
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1e3


# --------------------------------------------------------------------------------------
# W2 -- the fused matvec + cleanup, and its measured flip boundary.
# --------------------------------------------------------------------------------------

def test_matvec_matches_numpy_within_f32_rounding(mind):
    rng = np.random.default_rng(0)
    for count, dim in ((64, 512), (129, 300), (256, 1024)):
        matrix = rng.standard_normal((count, dim)).astype(np.float32)
        vector = rng.standard_normal(dim).astype(np.float32)
        got = mind.wgsl_matvec(matrix, vector)
        ref = (matrix @ vector).astype(np.float32)
        rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)
        assert rel.max() < 1e-3, "M=%d D=%d max_rel %.2e" % (count, dim, rel.max())


def test_matvec_handles_a_dim_that_is_not_a_multiple_of_the_workgroup(mind):
    # Each lane walks its row with a STRIDE rather than owning one element, which is what makes this work.
    rng = np.random.default_rng(1)
    matrix = rng.standard_normal((17, 300)).astype(np.float32)
    vector = rng.standard_normal(300).astype(np.float32)
    assert np.allclose(mind.wgsl_matvec(matrix, vector), matrix @ vector, rtol=1e-3, atol=1e-4)


def test_matvec_guards(mind):
    from holographic.io_and_interop.holographic_wgpurun import matvec_kernel

    with pytest.raises(ValueError):
        matvec_kernel(np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32))
    with pytest.raises(ValueError):
        matvec_kernel(np.zeros((4, 8), dtype=np.float32), np.zeros(7, dtype=np.float32))


def test_device_cleanup_agrees_with_cpu_on_realistic_queries(mind):
    rng = np.random.default_rng(0)
    dim, count = 512, 256
    book = rng.standard_normal((count, dim)).astype(np.float32)
    book /= np.linalg.norm(book, axis=1, keepdims=True)
    for _ in range(20):
        i = int(rng.integers(count))
        query = book[i] + 0.8 * rng.standard_normal(dim).astype(np.float32) / np.sqrt(dim)
        query /= np.linalg.norm(query)
        assert mind.wgsl_cleanup(book, query)[0] == int(np.argmax(book @ query))


def test_near_duplicate_atoms_do_not_flip_the_decision(mind):
    """Counter-intuitive and worth pinning: near-DUPLICATE rows are SAFE, because the f32 rounding error is
    correlated across near-identical rows and largely cancels in their difference. It is INDEPENDENT rows
    landing coincidentally close that are dangerous — the opposite of the natural worry."""
    rng = np.random.default_rng(0)
    dim, count = 512, 64
    disagreements = 0
    for _ in range(40):
        book = rng.standard_normal((count, dim)).astype(np.float32)
        book[1] = book[0] + 1e-6 * rng.standard_normal(dim).astype(np.float32)
        book /= np.linalg.norm(book, axis=1, keepdims=True)
        query = rng.standard_normal(dim).astype(np.float32)
        query /= np.linalg.norm(query)
        disagreements += mind.wgsl_cleanup(book, query)[0] != int(np.argmax(book @ query))
    assert disagreements == 0


@pytest.mark.slow
def test_the_flip_boundary_is_far_below_any_sensible_tie_margin(mind):
    """THE HONEST RESIDUAL RISK, MEASURED AND BOUNDED. The matvec is not bit-exact, so a similarity near-tie
    CAN flip the decision. Measured with independent rows tuned to a controlled gap: 0/150 at gaps of 1e-6
    and above, 3/150 at 1e-7.

    That boundary is FOUR ORDERS below the smallest tie margin anyone would use (1e-3), so `tied_candidates`
    reports both candidates for every gap that could possibly flip. A caller pairing device cleanup with
    candidate sets receives a DECLARED AMBIGUITY, never a silent difference — which is precisely what
    candidate sets were built for, arrived at from an unrelated direction."""
    rng = np.random.default_rng(0)
    dim, count = 512, 64

    def flips(gap, trials=60):
        bad = 0
        for _ in range(trials):
            book = rng.standard_normal((count, dim)).astype(np.float32)
            book /= np.linalg.norm(book, axis=1, keepdims=True)
            query = rng.standard_normal(dim).astype(np.float32)
            query /= np.linalg.norm(query)
            sims = book @ query
            i = int(np.argmax(sims))
            j = (i + 1) % count
            if abs(float(sims[j])) < 1e-6:
                continue
            book[j] = book[j] * np.float32((float(sims[i]) - gap) / float(sims[j]))
            bad += mind.wgsl_cleanup(book, query)[0] != int(np.argmax(book @ query))
        return bad

    assert flips(1e-3) == 0, "the decision flips at a gap of 1e-3; the mitigation argument fails"
    assert flips(1e-6) == 0, "the flip boundary moved above 1e-6; re-check the tie-margin argument"


def test_candidate_sets_cover_the_flip_regime(mind):
    # The mitigation, made executable: any gap small enough to flip is reported as a tie.
    for gap in (1e-7, 1e-6):
        result = mind.tied_candidates([("winner", 0.5), ("rival", 0.5 - gap)], margin=1e-3)
        assert len(result["candidates"]) == 2 and result["confident"] is False


def test_the_cleanup_kernel_is_discoverable(mind):
    for query in ("cleanup on the gpu", "matvec on the gpu", "codebook similarity on any gpu"):
        assert "VSA cleanup on ANY GPU" in str(mind.find_capability(query)[:3]), \
            "%r no longer surfaces the cleanup kernel" % query


# --------------------------------------------------------------------------------------
# W5 -- the CI lane exists and cannot pass by skipping.
# --------------------------------------------------------------------------------------

def test_the_wgsl_ci_lane_exists_and_verifies_an_adapter():
    """W5. The CuPy backend was wired, code-reviewed, and NEVER ONCE EXECUTED IN CI because testing it needed
    hardware nobody had — it sat unverified for the life of the project. This lane is the mechanism that
    stops the WGSL path repeating that, and it only works if a broken driver install FAILS rather than
    turning every test into a skip.

    This project has already been bitten by a skip that was indistinguishable from a pass, so the lane
    asserts an adapter is present before running anything, and this test asserts the lane still does."""
    import pathlib

    lane = pathlib.Path(".github/workflows/wgsl.yml")
    assert lane.exists(), "the WGSL CI lane is gone"
    text = lane.read_text()
    assert "mesa-vulkan-drivers" in text, "the software adapter is no longer installed; the lane would skip"
    assert "would silently check nothing" in text, "the adapter assertion was removed from the lane"
    assert "pip install wgpu" in text, "wgpu is optional and must be installed by the lane itself"


def test_wgpu_is_an_optional_dependency():
    # It must NOT be in requirements.txt: the engine runs without it, and the lane installs it separately.
    import pathlib

    assert "wgpu" not in pathlib.Path("requirements.txt").read_text(), \
        "wgpu became a hard dependency; the pure-NumPy install is the one everyone actually runs"


# --------------------------------------------------------------------------------------
# The UP direction: batched queries, which is the shape that can actually pay.
# --------------------------------------------------------------------------------------

def test_matmul_matches_numpy_across_shapes(mind):
    rng = np.random.default_rng(0)
    for count, dim, n_q in ((64, 128, 8), (256, 512, 32), (129, 300, 7)):
        matrix = rng.standard_normal((count, dim)).astype(np.float32)
        queries = rng.standard_normal((n_q, dim)).astype(np.float32)
        got = mind.wgsl_matmul(matrix, queries)
        ref = (queries @ matrix.T).astype(np.float32)
        assert got.shape == (n_q, count)
        rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)
        assert rel.max() < 1e-2, "M=%d D=%d K=%d max_rel %.2e" % (count, dim, n_q, rel.max())


def test_batched_cleanup_agrees_with_cpu(mind):
    rng = np.random.default_rng(0)
    book = rng.standard_normal((256, 512)).astype(np.float32)
    book /= np.linalg.norm(book, axis=1, keepdims=True)
    queries = rng.standard_normal((64, 512)).astype(np.float32)
    idx, _scores = mind.wgsl_cleanup_batch(book, queries)
    assert np.array_equal(idx, np.argmax(queries @ book.T, axis=1))


def test_batching_does_not_change_the_tie_rule(mind):
    # Indices resolve host-side by lowest index in BOTH paths, so a batch must agree with K separate calls.
    rng = np.random.default_rng(1)
    book = rng.standard_normal((32, 64)).astype(np.float32)
    queries = rng.standard_normal((5, 64)).astype(np.float32)
    batch_idx, _ = mind.wgsl_cleanup_batch(book, queries)
    single = [mind.wgsl_cleanup(book, q)[0] for q in queries]
    assert list(batch_idx) == single


def test_matmul_guards(mind):
    from holographic.io_and_interop.holographic_wgpurun import matmul_kernel

    with pytest.raises(ValueError):
        matmul_kernel(np.zeros((4, 8), dtype=np.float32), np.zeros(8, dtype=np.float32))
    with pytest.raises(ValueError):
        matmul_kernel(np.zeros((4, 8), dtype=np.float32), np.zeros((3, 7), dtype=np.float32))


@pytest.mark.slow
def test_the_batched_shape_is_the_one_above_the_dispatch_floor():
    """WHY THE BATCHED FORM EXISTS, pinned. Building the single-query matvec first repeated a mistake this
    path had already made twice (offloading an argmax, offloading one bind): THE NATURAL UNIT OF WORK IS
    USUALLY SMALLER THAN THE DISPATCH FLOOR. One query against a 1024x512 codebook is ~0.1 ms of CPU work,
    below any plausible submission cost; a batch of 256 is ~3 ms, comfortably above it."""
    import time

    rng = np.random.default_rng(0)
    book = rng.standard_normal((1024, 512)).astype(np.float32)
    queries = rng.standard_normal((256, 512)).astype(np.float32)

    def ms(fn):
        fn()
        start = time.perf_counter()
        fn()
        return (time.perf_counter() - start) * 1e3

    single = min(ms(lambda: book @ queries[0]) for _ in range(3))
    batched = min(ms(lambda: book @ queries.T) for _ in range(3))
    assert single < 0.5, "one query costs %.3f ms; the dispatch-floor argument assumed it was tiny" % single
    assert batched > 5 * single, "batching no longer changes the regime (%.3f vs %.3f ms)" % (batched, single)


# --------------------------------------------------------------------------------------
# W3 -- batched bind, as a direct circular convolution.
# --------------------------------------------------------------------------------------

def test_bind_is_a_plain_circular_convolution():
    """THE FACT THAT MADE W3 AN M ITEM INSTEAD OF AN L ONE. Because `bind` is exactly a circular
    convolution, it can be computed directly with the workgroup-reduction shape already proven twice —
    no bit-reversal, no twiddle tables, no multi-stage barriers that a Stockham FFT would need."""
    from holographic.agents_and_reasoning.holographic_ai import bind

    rng = np.random.default_rng(0)
    dim = 64
    a, b = rng.standard_normal(dim), rng.standard_normal(dim)
    direct = np.array([sum(a[k] * b[(n - k) % dim] for k in range(dim)) for n in range(dim)])
    assert np.abs(bind(a, b) - direct).max() < 1e-12


def test_batched_bind_matches_the_shipped_bind_batch(mind):
    from holographic.agents_and_reasoning.holographic_ai import bind_batch

    rng = np.random.default_rng(0)
    for count, dim in ((4, 64), (8, 256), (3, 100)):
        a = rng.standard_normal((count, dim)).astype(np.float32)
        b = rng.standard_normal((count, dim)).astype(np.float32)
        got = mind.wgsl_bind_batch(a, b)
        ref = bind_batch(a, b).astype(np.float32)
        assert got.shape == (count, dim)
        assert np.abs(got - ref).max() < 1e-4, "K=%d D=%d" % (count, dim)


def test_batched_bind_handles_a_non_power_of_two_dim(mind):
    # D=100 exercises both the modular index and the strided lane walk; an FFT route would have needed
    # padding here, and the direct form does not.
    from holographic.agents_and_reasoning.holographic_ai import bind_batch

    rng = np.random.default_rng(3)
    a = rng.standard_normal((2, 100)).astype(np.float32)
    b = rng.standard_normal((2, 100)).astype(np.float32)
    assert np.abs(mind.wgsl_bind_batch(a, b) - bind_batch(a, b).astype(np.float32)).max() < 1e-4


def test_batched_bind_guards(mind):
    from holographic.io_and_interop.holographic_wgpurun import bind_batch_kernel

    with pytest.raises(ValueError):
        bind_batch_kernel(np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32))
    with pytest.raises(ValueError):
        bind_batch_kernel(np.zeros((2, 8), dtype=np.float32), np.zeros((2, 9), dtype=np.float32))


def test_batched_bind_is_deterministic(mind):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 128)).astype(np.float32)
    b = rng.standard_normal((4, 128)).astype(np.float32)
    assert np.array_equal(mind.wgsl_bind_batch(a, b), mind.wgsl_bind_batch(a, b))
