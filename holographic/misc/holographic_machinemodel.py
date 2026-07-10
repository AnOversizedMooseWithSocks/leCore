"""holographic_machinemodel.py -- THE leCORE VIRTUAL MACHINE, named and measured.

The engine has, piece by piece, grown the parts of a GPU and of a memory hierarchy. They were built in different
sessions, live in different modules, and are named after the domain that first needed them. This module is the
SPEC SHEET: one place that names each unit, points at the real symbol, and -- because a doc rots and a measurement
does not -- MEASURES its own numbers on demand (`spec_sheet()`).

WHY A COST MODEL AND NOT A LATENCY LADDER (the finding that shaped this module)
------------------------------------------------------------------------------
`holographic_memoryhome` opens with the textbook frame: "registers -> L1 -> L2 -> L3 -> RAM -> disk, each ~10x
bigger and ~10x slower than the one before." That frame is WRONG for this engine, and the measurement says so.
Per single scalar access, on this box:

    L0 reuse a compiled transfer      121 ns
    RAM  dense array index X[i, j]    132 ns      <-- as fast as "L0"
    L1   MarginCache hit            3,485 ns      <-- 26x SLOWER than RAM
    L2   BakedGrid trilinear fetch  69,712 ns     <-- 528x SLOWER than RAM
    L2b  texture unit fetch        376,032 ns     <-- 2,850x SLOWER than RAM

A latency-ordered hierarchy would tell you to never use any of them. That reading is nonsense, and the reason is
that **none of these are scalar units.** Every one is a BATCH unit whose per-access cost collapses with N:

    BakedGrid.sample, ns/point:   N=1: 61,765    N=10: 8,541    N=100: 1,021    N=10,000: 274

And the texture unit is stranger still, in the way that makes it a *hardware* unit: `gather(bake, rule)` has a
marginal cost that is **CONSTANT in N** -- one dot product, whatever N was:

    N lookups:      8       32       128       512      2,048
    N x fetch():  2.8 ms  12.1 ms   44.9 ms  189.5 ms  728.2 ms
    gather():     3.7 us   3.4 us    4.1 us    4.3 us    4.0 us      <-- flat
    speedup:      763x    3,534x   11,033x   44,270x  182,010x

So a unit in this model is characterized by **(setup, marginal, how marginal scales in N)**, and the only question
that matters is *does the work amortize the setup*. That is the same `should_jump` gate the modal solver uses, the
same `worth_factoring` gate `LowRankField` uses, and the same break-even the fat-margin cache uses. One question,
asked of every unit. `place()` asks it for you.

HONEST SCOPE
------------
This module is a MAP, not an execution engine. It does not dispatch, wrap, or shim anything -- every entry points
at the real symbol you should call, and `_selftest` fails if a symbol does not resolve. A registry that names a
function which is not there is a lie that looks like documentation.
"""

import importlib
import time

import numpy as np


# kind -> what a GPU person would call it. A unit is (module, symbol, cost model, when NOT to use).
#
# `setup` is what you pay once; `marginal` is what each additional unit of work costs; `scaling` is how the
# marginal cost grows with N (the number of items in one call). O(1) marginal in N is the strongest possible
# statement and only two units make it.
UNITS = {
    # ---- COMPUTE UNITS -------------------------------------------------------------------------------
    "simd_lanes": {
        "kind": "compute", "gpu_name": "ALU / SIMD lanes",
        "module": "numpy", "symbol": "ndarray",
        "setup": "none", "marginal": "O(1) per element", "scaling": "O(N)",
        "use_when": "any elementwise map over an array",
        "do_not_use_when": "the operation is not elementwise; then you want a real unit below",
        "why": "numpy IS the vector unit. Do not reimplement it, and do not reimplement GEMM either -- `@` is BLAS "
               "at 116 GFLOP/s on this box.",
    },
    "simt_width": {
        "kind": "compute", "gpu_name": "warp / SIMT width",
        "module": "holographic.misc.holographic_superposed", "symbol": "pack",
        "setup": "O(K*D) to pack", "marginal": "O(D) per readout", "scaling": "O(1) in K for a scored query",
        "use_when": "K independent keyed computations must be carried and scored together",
        "do_not_use_when": "K is small (< 8) or the readouts must be exact -- fidelity follows 1/sqrt(K)",
        "why": "K computations ride in one vector; `score_all` evaluates all K against a query in one pass. The "
               "capacity law is 1/sqrt(K), NOT sqrt(K/D) -- see holographic_shader's H7 note.",
    },
    "texture_unit": {
        "kind": "compute", "gpu_name": "texture unit (sample + interpolate)",
        "module": "holographic.rendering.holographic_shader", "symbol": "bake_1d",
        "setup": "O(S*D) to bake S samples", "marginal": "O(D) per fetch", "scaling": "O(N) for N fetches",
        "use_when": "an expensive scalar function is sampled at many arbitrary points; interpolation is in the algebra",
        "do_not_use_when": "you need machine precision -- this is a ~1% preview tier, and the phasor bandwidth must "
                           "exceed the field's max frequency (the algebra has a Nyquist)",
        "why": "bake once, query anywhere. Nothing about it is graphical: a sigmoid bakes to mean |err| 0.021.",
    },
    "gather_unit": {
        "kind": "compute", "gpu_name": "gather (N lookups, one instruction)",
        "module": "holographic.rendering.holographic_shader", "symbol": "gather",
        "setup": "O(N*D) to compile the rule (REUSE IT)", "marginal": "ONE dot product", "scaling": "O(1) in N",
        "use_when": "the same weighted lookup pattern is applied many times to a baked field",
        "do_not_use_when": "the rule is used once -- compiling it costs more than N fetches. Measured: rule build "
                           "is O(N) and reaches 724 ms at N=2048; the gather itself is 4.0 us.",
        "why": "the ONLY unit here with O(1) marginal cost in N. Measured 182,010x at N=2048 -- when the rule is "
               "reused. `gather_samples` is the stateless twin: it re-bakes every call and buys reachability, not "
               "speed (its own docstring says so, and it measures 0.03x at N=8).",
    },
    "kernel_fusion": {
        "kind": "compute", "gpu_name": "shader / kernel fusion",
        "module": "holographic.rendering.holographic_shader", "symbol": "Pipeline",
        "setup": "O(D) per stage, once", "marginal": "one elementwise multiply", "scaling": "O(1) in passes",
        "use_when": "a chain of LINEAR, CIRCULAR passes is applied repeatedly",
        "do_not_use_when": "any stage is nonlinear, or the operator is not shift-equivariant. Measured: periodic "
                           "heat matches a 2,000-step loop to 6.66e-16 at 80x; NEUMANN heat is 2.35e-02 WRONG",
        "why": "N filter passes compose into ONE transfer before any query. Rank-agnostic: verified 1-D, 2-D, 3-D.",
    },
    "operator_power": {
        "kind": "compute", "gpu_name": "tensor core / batched linear algebra",
        "module": "holographic.misc.holographic_iterate", "symbol": "affine_step_k_batch",
        "setup": "one batched eigendecomposition, O(M*d^3)", "marginal": "O(M*d^2) per horizon",
        "scaling": "O(1) in k",
        "use_when": "M independent linear recurrences must be advanced k steps, k >= 20*d",
        "do_not_use_when": "the operator is defective (a Jordan block has no eigenbasis -- it RAISES), or k is "
                           "below break-even, or the system is nonlinear",
        "why": "a batched matrix power. Named for physics; nothing about the code is. 4.3x over batched stepping, "
               "exact to 1.9e-12, and the horizon is free.",
    },
    "rt_core": {
        "kind": "compute", "gpu_name": "RT core (ray/scene intersection)",
        "module": "holographic.rendering.holographic_raymarch", "symbol": "sphere_trace",
        "setup": "none", "marginal": "O(steps) per ray, active-set pruned", "scaling": "O(N) rays, vectorised",
        "use_when": "any 'how far until I hit something' query against an SDF",
        "do_not_use_when": "the geometry is a mesh with no SDF; build one or use the mesh path",
        "why": "the same march renders a pixel, computes a time of impact (CCD), and steps Walk-on-Spheres. One "
               "distance query, three faculties.",
    },
    "rng": {
        "kind": "compute", "gpu_name": "per-thread RNG (counter-based)",
        "module": "holographic.misc.holographic_determinism", "symbol": "hash_unit",
        "setup": "none", "marginal": "one hash", "scaling": "O(N), vectorised over key arrays",
        "use_when": "a sample is indexed by WHERE it is, not by how many came before",
        "do_not_use_when": "you need a long-period stream from a single seed; use default_rng",
        "why": "a pure function of the coordinates: order-independent, no draw counter, no seed coordination. "
               "This is exactly a GPU's per-thread RNG.",
    },
    "scheduler": {
        "kind": "compute", "gpu_name": "atomics-free wave scheduler",
        "module": "holographic.simulation_and_physics.holographic_island", "symbol": "color_waves",
        "setup": "O(sum of squared key degrees) to colour", "marginal": "free", "scaling": "O(waves)",
        "use_when": "tasks contend for shared resources and you want lock-free, reproducible parallelism",
        "do_not_use_when": "everything conflicts -- colouring cannot invent parallelism, it serialises honestly",
        "why": "2,000 transactions over 300 keys -> 24 waves, 83x lock-free parallelism, deterministic by "
               "construction (greedy ascending), every wave verified conflict-free.",
    },
    "occupancy_gate": {
        "kind": "compute", "gpu_name": "occupancy / skip idle work",
        "module": "holographic.simulation_and_physics.holographic_island", "symbol": "SleepTracker",
        "setup": "one energy probe per island per frame", "marginal": "zero for a sleeping island",
        "scaling": "saving == the awake fraction",
        "use_when": "most of the work is at its fixed point most of the time",
        "do_not_use_when": "a single sleep threshold is used -- it flickers 11 times in 12 frames. Hysteresis is "
                           "mandatory, not a refinement",
        "why": "a sleeping island IS at its fixed point; skipping it is bit-identical to stepping it.",
    },

    # ---- MEMORY TIERS --------------------------------------------------------------------------------
    # Ordered by SETUP COST, not by latency. See the module docstring: a latency ladder is the wrong frame.
    "t0_compiled": {
        "kind": "memory", "gpu_name": "L0 / registers -- a held, compiled operator",
        "module": "holographic.rendering.holographic_shader", "symbol": "Pipeline",
        "setup": "compose the transfer once", "marginal": "121 ns to read", "scaling": "O(1)",
        "use_when": "the same composed operator is applied every frame",
        "do_not_use_when": "the operator changes each call -- then composition is pure overhead",
        "why": "the transfer is the register file: computed once, multiplied in.",
    },
    "t1_margin_cache": {
        "kind": "memory", "gpu_name": "L1 -- hot, small, hysteresis-guarded",
        "module": "holographic.caching_and_storage.holographic_cachehome", "symbol": "MarginCache",
        "setup": "one bake per rebuild", "marginal": "3,485 ns per hit", "scaling": "O(1)",
        "use_when": "a DRIFTING query (a camera, a cursor, an agent, a recall neighbourhood)",
        "do_not_use_when": "the query jumps randomly -- the margin buys nothing and the rebuilds dominate",
        "why": "400 drifting queries: exact-key caching rebuilds 400/400; margin 6.0 rebuilds 20 at 95% hits.",
    },
    "t2_baked_grid": {
        "kind": "memory", "gpu_name": "L2 -- bake once, sample O(1)",
        "module": "holographic.caching_and_storage.holographic_cachehome", "symbol": "Cache",
        "setup": "res^d evaluator calls", "marginal": "274 ns/point at N=10,000 (61,765 ns at N=1)",
        "scaling": "amortizes with batch size",
        "use_when": "an expensive evaluator is queried many times over a bounded domain",
        "do_not_use_when": "queried once, or the domain is unbounded. A BATCH unit: at N=1 it is 470x slower than "
                           "a dense index",
        "why": "the classic bake-and-query lever, and the reason `Cache.bake(vary=...)` dispatches on what varies.",
    },
    "t3_content_addressed": {
        "kind": "memory", "gpu_name": "L3 -- shared, content-addressed, deduped",
        "module": "holographic.scene_and_pipeline.holographic_compile", "symbol": "CompileCache",
        "setup": "a deterministic hash of the spec", "marginal": "a dict lookup", "scaling": "O(1)",
        "use_when": "the same spec is compiled from many call sites",
        "do_not_use_when": "the evaluator is not deterministic -- a cached nondeterministic evaluator is a bug "
                           "(the D1 coordinate-keyed-sampling rule)",
        "why": "hash the spec, hand the same compiled object out everywhere.",
    },
    "t4_compressed_ram": {
        "kind": "memory", "gpu_name": "RAM, compressed -- never decompress",
        "module": "holographic.caching_and_storage.holographic_tucker", "symbol": "LowRankField",
        "setup": "one SVD", "marginal": "1,977 ns per point query (vs 132 ns dense)",
        "scaling": "O(r) per query; ops touch only the factors",
        "use_when": "the field is low rank and you are BANDWIDTH bound, not latency bound",
        "do_not_use_when": "`worth_factoring` says no. White noise gates to rank 197/256 and costs 1.54x MORE. "
                           "And nonlinear ops do not survive the factorization",
        "why": "171x fewer bytes at 1024^2 rank 3; a separable blur runs 26x faster touching 0.049 MB, not 8.4 MB.",
    },
    "t5_cold_store": {
        "kind": "memory", "gpu_name": "cold storage / paging",
        "module": "holographic.caching_and_storage.holographic_coldstore", "symbol": "Cold",
        "setup": "compress on eviction", "marginal": "16,868 ns to decompress a row", "scaling": "O(bytes)",
        "use_when": "data is idle and large",
        "do_not_use_when": "it is on a hot path -- decompression is 128x a dense index",
        "why": "trade time for space, explicitly, at the point where space is the constraint.",
    },
    "t6_durable": {
        "kind": "memory", "gpu_name": "disk / durable, verified",
        "module": "holographic.agents_and_reasoning.holographic_deltachain", "symbol": "DeltaChain",
        "setup": "hash chain + Merkle root", "marginal": "8,863 ns for a seek+read",
        "scaling": "O(changed rows)",
        "use_when": "an append-only history of SPARSE mutations must be stored and verified",
        "do_not_use_when": "mutation is dense. On a physics trace DeltaChain takes 614,144 bytes against the raw "
                           "460,800 -- a 33% LOSS. Dense mutation with sparse causes wants the event codec instead",
        "why": "base + per-chunk deltas, with a proof.",
    },
}


def units(kind=None):
    """The unit names, optionally filtered to `kind` in {'compute', 'memory'}. The discovery door."""
    return sorted(k for k, v in UNITS.items() if kind is None or v["kind"] == kind)


def unit(name):
    """One unit's full record: {kind, gpu_name, module, symbol, setup, marginal, scaling, use_when,
    do_not_use_when, why}. Plain data -- it crosses an HTTP boundary."""
    if name not in UNITS:
        raise KeyError("no such unit %r; try units()" % (name,))
    return dict(UNITS[name], name=name)


def machine_map(kind=None):
    """The whole spec sheet as plain data: [{name, gpu_name, kind, module, symbol, setup, marginal, scaling,
    use_when, do_not_use_when}]. This is the pattern-recognition table -- read it before building anything that
    smells like a cache, a kernel, a scheduler or a lookup."""
    return [unit(n) for n in units(kind)]


def resolve(name):
    """Import the unit's real symbol and return it. Raises if it does not exist -- which is the point: a registry
    that names a function which is not there is a lie that looks like documentation."""
    rec = unit(name)
    mod = importlib.import_module(rec["module"])
    return getattr(mod, rec["symbol"])


def break_even(setup_ns, marginal_ns, baseline_ns):
    """How many calls N before a unit with (setup, marginal) beats a `baseline` that costs `baseline_ns` per call?

        N * baseline > setup + N * marginal   =>   N > setup / (baseline - marginal)

    Returns `inf` when the unit's marginal cost is not below the baseline (it can NEVER pay), which is the honest
    answer and the one a dispatcher needs. This is `should_jump` and `worth_factoring`, stated once."""
    if marginal_ns >= baseline_ns:
        return float("inf")
    return float(setup_ns) / (float(baseline_ns) - float(marginal_ns))


def place(baseline_ns, n_calls, setup_ns, marginal_ns):
    """Should this work go on the unit, or stay on the baseline? Returns {use_unit, break_even_n, unit_total_ns,
    baseline_total_ns, speedup}. Pure arithmetic over a measured cost model -- no heuristics, no magic numbers.

    THE ONE WAY TO MISUSE THIS, and it is the program's oldest error: **`baseline_ns` must be the cost of what the
    unit REPLACES**, per call, not the cost of some other operation. A ratio is only meaningful when numerator and
    denominator do the same job.

      * `kernel_fusion` replaces N passes, so its baseline is `n_passes * pass_ns` -- not one pass. Priced against
        a single pass it is measured to LOSE (apply() is one FFT), and that verdict would be an artifact of the
        denominator, not a fact about the unit.
      * `gather_unit` replaces N fetches, so its baseline is `n_lookups * fetch_ns`.
      * `t2_baked_grid` replaces the evaluator you baked, so its baseline is that evaluator.

    Measured against a raw array read (130 ns), almost every unit here reports NEVER -- correctly. These units
    replace an expensive evaluator, not an array index. If everything says NEVER, check the denominator first."""
    n = int(n_calls)
    be = break_even(setup_ns, marginal_ns, baseline_ns)
    unit_total = float(setup_ns) + n * float(marginal_ns)
    base_total = n * float(baseline_ns)
    return {"use_unit": bool(unit_total < base_total), "break_even_n": be,
            "unit_total_ns": unit_total, "baseline_total_ns": base_total,
            "speedup": (base_total / unit_total) if unit_total > 0 else float("inf")}


def _time(fn, n):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e9


def spec_sheet(quick=True):
    """MEASURE every unit on this machine, right now, instead of trusting a comment. Returns plain data:
    {unit: {setup_ns, marginal_ns, note}} with one entry per name in `UNITS`, plus `baseline_dense_index`.
    `quick=True` uses small sizes so it can run in a test.

    WHY THIS IS A FUNCTION AND NOT A TABLE: every number in this module's docstring was measured on one box on one
    day. A spec sheet that cannot re-measure itself is a rumour. Run it on your box.

    A `marginal_ns` of exactly 0.0 is a real answer, not a missing one: a sleeping island costs nothing to skip,
    and a colour wave is a list you already have. Those units pay their whole price in `setup_ns`."""
    from holographic.caching_and_storage.holographic_cachehome import Cache, MarginCache
    from holographic.rendering.holographic_shader import (bake_1d, blur_kernel, fetch, gather, gather_rule,
                                                          Pipeline)
    from holographic.caching_and_storage.holographic_tucker import LowRankField
    from holographic.misc.holographic_superposed import pack, score_all
    from holographic.misc.holographic_iterate import affine_step_k_batch, affine_transfer_batch
    from holographic.misc.holographic_determinism import hash_unit
    from holographic.rendering.holographic_raymarch import sphere_trace
    from holographic.simulation_and_physics.holographic_island import (SleepTracker, color_waves, conflict_graph,
                                                                       island_energy)
    from holographic.scene_and_pipeline.holographic_compile import CompileCache
    from holographic.caching_and_storage.holographic_coldstore import Cold
    from holographic.agents_and_reasoning.holographic_deltachain import DeltaChain

    reps = 20 if quick else 200
    dim = 1024 if quick else 4096
    out = {}

    # dense index: the baseline every unit is measured against
    A = np.random.default_rng(0).normal(size=(128, 128))
    out["baseline_dense_index"] = {"setup_ns": 0.0, "marginal_ns": _time(lambda: A[7, 9], 2000),
                                   "note": "the honest baseline: a raw array read"}

    # texture unit + gather: the O(1)-in-N marginal cost, the headline of this whole module
    xs = np.linspace(0, 1, 64)
    ys = np.sin(6 * xs)
    t_bake = _time(lambda: bake_1d(xs, ys, dim=dim), max(2, reps // 10))
    b = bake_1d(xs, ys, dim=dim)
    out["texture_unit"] = {"setup_ns": t_bake, "marginal_ns": _time(lambda: fetch(b, 0.37, normalize=True), reps),
                           "note": "one O(D) dot per fetch"}
    N = 64
    pts = np.random.default_rng(1).uniform(0, 1, N)
    w = np.random.default_rng(2).normal(size=N)
    t_rule = _time(lambda: gather_rule(b, pts, w), max(2, reps // 10))
    rule = gather_rule(b, pts, w)
    out["gather_unit"] = {"setup_ns": t_bake + t_rule, "marginal_ns": _time(lambda: gather(b, rule), reps * 5),
                          "note": "marginal cost is CONSTANT in N (%d lookups here)" % N}

    # baked grid: a batch unit
    ev = lambda P: P[:, 0] ** 2
    t_setup = _time(lambda: Cache.bake(ev, vary="position", lo=(0, 0, 0), hi=(1, 1, 1), res=8), 3)
    bg = Cache.bake(ev, vary="position", lo=(0, 0, 0), hi=(1, 1, 1), res=8)
    Q = np.random.default_rng(3).uniform(0, 1, size=(256, 3))
    out["t2_baked_grid"] = {"setup_ns": t_setup, "marginal_ns": _time(lambda: bg.sample(Q), reps) / len(Q),
                            "note": "per-point, batched over %d points" % len(Q)}

    # margin cache
    mc = MarginCache(lambda p: ("baked",), margin=10.0)
    mc.get(np.zeros(2))
    out["t1_margin_cache"] = {"setup_ns": 0.0, "marginal_ns": _time(lambda: mc.get(np.array([0.1, 0.1])), reps * 5),
                              "note": "per hit; a miss costs one rebuild"}

    # compressed RAM
    x = np.linspace(0, 1, 64)
    X = np.outer(np.sin(3 * np.pi * x), np.cos(2 * np.pi * x))
    t_svd = _time(lambda: LowRankField.from_dense(X, rank=1), max(2, reps // 4))
    lf = LowRankField.from_dense(X, rank=1)
    out["t4_compressed_ram"] = {"setup_ns": t_svd, "marginal_ns": _time(lambda: lf.query(11, 23), reps * 5),
                                "note": "one length-r contraction; %d bytes vs %d dense" % (lf.nbytes(), X.nbytes)}

    # ---- the remaining compute units (M2: place() must run on MEASURED numbers, not hand-supplied ones) ----
    big = np.random.default_rng(4).normal(size=65536)
    out["simd_lanes"] = {"setup_ns": 0.0, "marginal_ns": _time(lambda: big * 2.0, reps) / big.size,
                         "note": "per element, over %d" % big.size}

    K = 16
    rng = np.random.default_rng(5)
    keys, items = rng.normal(size=(K, dim)), rng.normal(size=(K, dim))
    q = rng.normal(size=dim)
    S = pack(keys, items)
    out["simt_width"] = {"setup_ns": _time(lambda: pack(keys, items), reps),
                         "marginal_ns": _time(lambda: score_all(S, keys, q), reps) / K,
                         "note": "per candidate; K=%d scored in one readout. Fidelity ~ 1/sqrt(K)" % K}

    field = np.random.default_rng(6).normal(size=dim)
    kern = blur_kernel((dim,))
    def _compose():
        p = Pipeline((dim,))
        p.blur(kern)
        p.blur(kern)
        return p
    pipe = _compose()
    out["kernel_fusion"] = {"setup_ns": _time(_compose, reps),
                            "marginal_ns": _time(lambda: pipe.apply(field), reps),
                            "note": "2 stages composed once; apply is one elementwise multiply"}
    out["t0_compiled"] = {"setup_ns": out["kernel_fusion"]["setup_ns"],
                          "marginal_ns": _time(lambda: pipe.transfer[0], reps * 20),
                          "note": "the composed transfer, held and re-read"}

    M, d, k = 8, 6, 1000
    As = np.stack([np.eye(d) * 0.9 + 0.01 * rng.normal(size=(d, d)) for _ in range(M)])
    bs = np.zeros((M, d))
    S0 = np.ones((M, d))
    tr = affine_transfer_batch(As)
    out["operator_power"] = {"setup_ns": _time(lambda: affine_transfer_batch(As), max(2, reps // 4)),
                             "marginal_ns": _time(lambda: affine_step_k_batch(S0, As, bs, k, transfer=tr), reps) / M,
                             "note": "per operator; k=%d is free once the transfer is cached" % k}

    wall = lambda P: np.abs(np.asarray(P, float)[:, 0]) - 0.05
    R = 64
    O = np.stack([np.array([-1.0, 0.0, 0.0])] * R)
    D = np.stack([np.array([1.0, 0.0, 0.0])] * R)
    out["rt_core"] = {"setup_ns": 0.0,
                      "marginal_ns": _time(lambda: sphere_trace(wall, O, D, max_dist=5.0), reps) / R,
                      "note": "per ray, batched over %d; active-set pruned" % R}

    tid = np.arange(4096)
    out["rng"] = {"setup_ns": 0.0, "marginal_ns": _time(lambda: hash_unit(tid, 0, 42), reps) / tid.size,
                  "note": "per draw, vectorised; a pure function of the coordinates"}

    tasks = [set(rng.integers(0, 64, 2).tolist()) for _ in range(256)]
    out["scheduler"] = {"setup_ns": _time(lambda: color_waves(*conflict_graph(tasks)), max(2, reps // 4)),
                        "marginal_ns": 0.0,
                        "note": "colouring is the whole price; a wave is a list you already have"}

    tr2 = SleepTracker(sleep_energy=1e-8, sleep_frames=1)
    rest = np.zeros((8, 3))
    out["occupancy_gate"] = {"setup_ns": _time(lambda: island_energy(rest), reps * 5),
                             "marginal_ns": 0.0,
                             "note": "the energy probe is the price; a sleeping island costs zero to skip"}

    # ---- the remaining memory tiers ----
    # NB: measure the REAL operations. A first draft timed `key in cc._store` behind a `hasattr` conditional and
    # `Cold(payload).get()` on a still-WARM object -- both of which measure nothing. `Cold` does not compress until
    # `cool()`, and a warm `get()` is a no-op. A spec sheet that benchmarks a no-op is worse than no spec sheet.
    cc = CompileCache()
    spec = {"kind": "sphere", "r": 1.0}
    expensive = lambda _s: np.linalg.svd(np.ones((64, 64)))     # never runs on a hit -- that IS the measurement
    cc.get_or_compile(spec, expensive)                          # warm the entry
    out["t3_content_addressed"] = {"setup_ns": _time(lambda: cc.key(spec), reps * 5),
                                   "marginal_ns": _time(lambda: cc.get_or_compile(spec, expensive), reps * 5),
                                   "note": "sha256 of the canonical bytes (setup), then an LRU hit (marginal)"}

    payload = np.random.default_rng(7).normal(size=(64, 64))
    def _cool_only():
        c = Cold(payload)
        c.cool()
        return c
    def _cool_then_get():
        c = Cold(payload)
        c.cool()
        c.get()
        return c
    t_cool = _time(_cool_only, max(2, reps // 4))
    t_both = _time(_cool_then_get, max(2, reps // 4))
    out["t5_cold_store"] = {"setup_ns": t_cool, "marginal_ns": max(t_both - t_cool, 0.0),
                            "note": "cool() compresses (setup); the marginal cost IS the inflate on get()"}

    base = np.zeros((32, 4))
    dc = DeltaChain(base)
    for i in range(8):
        chunk = base.copy()
        chunk[i] = float(i)
        dc.append(chunk)
    out["t6_durable"] = {"setup_ns": _time(lambda: DeltaChain(base), reps),
                         "marginal_ns": _time(lambda: dc.get(3), max(2, reps // 2)),
                         "note": "append-only base+delta with a Merkle proof; get() reconstructs AND verifies"}

    missing = set(UNITS) - set(out)
    if missing:                                              # the bar: every unit measures itself
        raise AssertionError("spec_sheet does not cover %s" % sorted(missing))
    return out


def place_unit(name, baseline_ns, n_calls, sheet=None):
    """`place()` on MEASURED numbers: look the unit's (setup, marginal) up in a spec sheet instead of asking the
    caller to supply them. Returns place()'s dict plus {unit, note}.

    This is the whole point of M2 -- the amortization question should be answerable from the machine, not from a
    number somebody remembered. Pass a cached `sheet` to avoid re-measuring."""
    sheet = sheet if sheet is not None else spec_sheet(quick=True)
    if name not in sheet:
        raise KeyError("no measurement for %r; spec_sheet() covers %s" % (name, sorted(sheet)))
    rec = sheet[name]
    got = place(float(baseline_ns), int(n_calls), rec["setup_ns"], rec["marginal_ns"])
    got.update({"unit": name, "note": rec["note"],
                "setup_ns": rec["setup_ns"], "marginal_ns": rec["marginal_ns"]})
    return got


def _selftest():
    """Regression trap: every named symbol must RESOLVE (a registry that lies is worse than no registry), the cost
    arithmetic must refuse to promise a win it cannot deliver, and the two structural findings must stay pinned --
    the latency ladder is not monotone, and the gather unit's marginal cost is constant in N."""
    # 1. EVERY entry points at a real symbol. This is the whole contract.
    for name in units():
        obj = resolve(name)
        assert obj is not None, name
    assert len(units("compute")) >= 8 and len(units("memory")) >= 6
    assert unit("gather_unit")["scaling"] == "O(1) in N"

    # 2. the cost arithmetic. A unit whose marginal cost is not below the baseline can NEVER pay.
    assert break_even(1000.0, 5.0, 10.0) == 200.0
    assert break_even(1000.0, 20.0, 10.0) == float("inf")
    p = place(baseline_ns=10.0, n_calls=1000, setup_ns=1000.0, marginal_ns=5.0)
    assert p["use_unit"] is True and p["break_even_n"] == 200.0
    assert place(baseline_ns=10.0, n_calls=10, setup_ns=1000.0, marginal_ns=5.0)["use_unit"] is False
    assert place(baseline_ns=1.0, n_calls=10 ** 9, setup_ns=0.0, marginal_ns=2.0)["use_unit"] is False

    # 3. KEPT NEGATIVE: the textbook latency ladder does NOT hold here. A dense index beats the "caches" on a
    #    single scalar access, because none of them is a scalar unit. Measured, not assumed.
    sheet = spec_sheet(quick=True)
    dense = sheet["baseline_dense_index"]["marginal_ns"]
    assert sheet["t1_margin_cache"]["marginal_ns"] > dense
    assert sheet["texture_unit"]["marginal_ns"] > dense

    # 4b. M2: every unit measures itself, and place_unit() runs on those numbers rather than hand-supplied ones.
    assert not (set(UNITS) - set(sheet)), sorted(set(UNITS) - set(sheet))
    dense_base = sheet["baseline_dense_index"]["marginal_ns"]
    # against a raw array read almost nothing can pay -- and the model must SAY so rather than flatter a unit
    assert place_unit("texture_unit", dense_base, 10 ** 6, sheet=sheet)["break_even_n"] == float("inf")
    # against a genuinely expensive evaluator the baked grid pays almost immediately
    hot = place_unit("t2_baked_grid", 50_000.0, 10 ** 6, sheet=sheet)
    assert hot["use_unit"] is True and hot["speedup"] > 5.0
    # units whose whole price is setup report marginal 0.0, and that is an answer, not a hole
    assert sheet["scheduler"]["marginal_ns"] == 0.0 and sheet["occupancy_gate"]["marginal_ns"] == 0.0
    # the two tiers whose first measurement was a no-op must now measure real work
    assert sheet["t5_cold_store"]["marginal_ns"] > 0.0        # a real inflate, not a warm get()

    # 4. THE HEADLINE: the gather unit's marginal cost is constant in N. Measured across two N, one bake.
    from holographic.rendering.holographic_shader import bake_1d, gather, gather_rule
    xs = np.linspace(0, 1, 64)
    b = bake_1d(xs, np.sin(6 * xs), dim=1024)
    rng = np.random.default_rng(0)
    costs = []
    for n in (8, 256):
        rule = gather_rule(b, rng.uniform(0, 1, n), rng.normal(size=n))
        costs.append(_time(lambda: gather(b, rule), 200))
    assert costs[1] < 3.0 * costs[0], ("gather marginal cost must not grow with N", costs)

    print("OK: holographic_machinemodel self-test passed (%d compute units + %d memory tiers, every symbol "
          "resolves; break_even returns inf when a unit can never pay; the latency ladder is NOT monotone -- a "
          "dense index (%.0f ns) beats the L1 margin cache (%.0f ns) and the texture unit (%.0f ns) on a scalar "
          "access, because none of them is a scalar unit; and gather's marginal cost is flat in N (%.2f us at "
          "N=8 vs %.2f us at N=256))"
          % (len(units("compute")), len(units("memory")), dense,
             sheet["t1_margin_cache"]["marginal_ns"], sheet["texture_unit"]["marginal_ns"],
             costs[0] / 1e3, costs[1] / 1e3))


if __name__ == "__main__":
    _selftest()
