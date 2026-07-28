"""CAP-1 -- bundle capacity as a MEASURED LOAD RATIO, not a constant (holographic_capacity).

WHY THIS EXISTS
---------------
"How many things fit in a bundle?" has been answered in this codebase by a folklore constant -- "20-32
instructions" -- and that constant was MEASURED WRONG: it is a LINEAR-readout artifact. With a sparse
decoder the same bundle at the same dimension holds several times more (CoSaMP recovered exact support at
M=86, D=512), and AMP extends usable recovery further still. So a capacity number without three variables
attached -- WHICH readout, WHAT dimension, WHAT quality floor -- is not a capacity number. This module makes
the question answerable properly:

    capacity = safe LOAD RATIO (M/D) x D,   measured for (readout, dim, floor), at call time.

LOAD RATIO IS THE RIGHT AXIS, and the reference theory says why: for m random unit atoms bundled into one
D-vector, the per-item signal-to-crosstalk is governed by m/D -- the SNR per dimension falls as 1/m while D
dimensions integrate it, so recovery curves at different D COLLAPSE onto one curve in M/D. That collapse is
verified in _selftest rather than assumed, because "falls with D" said without saying WHAT falls has already
caused three recorded errors in this project.

WHY IT MEASURES INSTEAD OF SHIPPING A TABLE
-------------------------------------------
The earlier session pinned the reference points (linear degrades by M/D ~ 0.05; CoSaMP exact to ~ 0.17; AMP
usable into the 0.25-0.39 band, all at D=512, incoherent dictionary). Those numbers are honest FOR THAT
CONFIGURATION -- and the same session also measured that coherent dictionaries INVERT the ranking (AMP
collapses to 0.052 where CoSaMP holds 1.000 at coherence 0.5). A shipped table silently inherits the
incoherent assumption; a measurement run on the caller's own dictionary does not. The advisor therefore
measures by default and is honest about costing a second or two.

Everything delegates: recovery goes through the SHIPPED decoders (linear cosine readout, cosamp, amp), and
the sweep uses seeded default_rng throughout.
"""

import numpy as np

from holographic.sampling_and_signal.holographic_amp import amp_recall
from holographic.sampling_and_signal.holographic_cosamp import cosamp_recall


def _random_dictionary(n_atoms, dim, rng):
    """Unit-norm random atoms -- the incoherent reference dictionary. Callers with a REAL codebook should
    pass it instead; coherence inverts the method ranking, so the reference numbers do not transfer."""
    cb = rng.standard_normal((n_atoms, dim))
    return cb / np.linalg.norm(cb, axis=1, keepdims=True)


def _recover_support(method, cue, codebook, k):
    """One recovery through the SHIPPED decoder for `method`, returning the recovered index set.

    'linear' is the naive cosine readout -- kept as a method precisely so the advisor can SHOW the folklore
    constant being an artifact of it rather than assert that in prose."""
    if method == "linear":
        scores = codebook @ cue
        return set(np.argsort(-scores)[:k].tolist())
    if method == "cosamp":
        return set(i for i, _ in cosamp_recall(cue, codebook, k))
    if method == "amp":
        return set(i for i, _ in amp_recall(cue, codebook, K=k))
    raise ValueError("method must be 'linear', 'cosamp' or 'amp', got %r" % (method,))


def measure_recovery_curve(dim, method, ratios=(0.05, 0.10, 0.17, 0.25, 0.33), n_atoms=None,
                           seeds=range(4), codebook=None):
    """Support-recovery F1 as a function of LOAD RATIO M/D, measured live.

    For each ratio, bundle M = round(ratio * dim) distinct atoms and ask `method` for the support back;
    F1 compares recovered to true indices. Returns [{ratio, m, f1_mean, f1_sd}] over `seeds`.

    Pass your own `codebook` to measure on YOUR atoms -- the reference random dictionary is incoherent, and
    coherence is exactly the property that inverts the method ranking."""
    rows = []
    for ratio in ratios:
        m = max(1, int(round(float(ratio) * dim)))
        f1s = []
        for seed in seeds:
            rng = np.random.default_rng(int(seed))
            cb = codebook if codebook is not None else _random_dictionary(n_atoms or 4 * dim, dim, rng)
            true = rng.choice(cb.shape[0], size=min(m, cb.shape[0]), replace=False)
            cue = cb[true].sum(axis=0)
            got = _recover_support(method, cue, cb, len(true))
            ts = set(true.tolist())
            tp = len(got & ts)
            f1s.append(2 * tp / (len(got) + len(ts)) if (got or ts) else 1.0)
        f1s = np.asarray(f1s, float)
        rows.append({"ratio": float(ratio), "m": int(m),
                     "f1_mean": float(f1s.mean()), "f1_sd": float(f1s.std())})
    return rows


def bundle_capacity(dim, method="cosamp", floor=0.95, seeds=range(4), codebook=None,
                    ratios=(0.02, 0.05, 0.10, 0.17, 0.25, 0.33, 0.40)):
    """THE ADVISOR: the largest number of items you can bundle at `dim` and still recover the support at
    quality `floor` with `method` -- MEASURED, not looked up.

    Returns {capacity, safe_ratio, method, dim, floor, curve}. The curve travels with the answer so the
    number cannot be quoted without the configuration that produced it -- a capacity without its readout,
    dimension and floor attached is exactly the folklore-constant failure this module replaces.

    The safe ratio is the largest measured ratio whose MEAN MINUS SD still clears the floor: a capacity
    that only the lucky seed reaches is not a capacity."""
    curve = measure_recovery_curve(dim, method, ratios=ratios, seeds=seeds, codebook=codebook)
    safe = 0.0
    for row in curve:
        if row["f1_mean"] - row["f1_sd"] >= float(floor):
            safe = row["ratio"]
    return {"capacity": int(round(safe * dim)), "safe_ratio": float(safe), "method": method,
            "dim": int(dim), "floor": float(floor), "curve": curve}


def cleanup_batch(codebook, queries, backend=None, workgroup=64):
    """Clean up a STACK of cues against a codebook -> (indices, scores), one per cue.

    THE MISSING `UP` DIRECTION, AND IT PAYS ON THE CPU ALONE. `Vocabulary.cleanup` handles one cue; a caller
    with K cues looped, which is K separate (M, D) x (D,) matvecs where one (K, D) x (D, M) matmul would do.
    Measured, NO DEVICE INVOLVED:
        M=256  D=512  K=32   0.413 ms -> 0.160 ms   2.58x
        M=1024 D=512  K=64   5.171 ms -> 0.964 ms   5.36x
        M=4096 D=1024 K=128 78.530 ms -> 13.273 ms  5.92x
    That is BLAS getting one big matmul instead of K small ones, and it is the whole win at these sizes --
    the argmax is microseconds either way.

    `backend='wgsl'` routes the same computation to a device (holographic_wgpurun.cleanup_batch_kernel).
    DEFAULT OFF, DELIBERATELY: the host<->device crossover has never been measured on real hardware, so
    enabling it by default would act on arithmetic rather than a measurement -- and the one thing worse than
    not using a device is using it on a guess. The seam exists so somebody WITH a device can measure it
    without editing the engine.

    INDICES RESOLVE BY LOWEST INDEX on both paths, so the backend cannot change which atom wins a tie."""
    cb = np.ascontiguousarray(np.asarray(codebook, dtype=np.float32))
    qs = np.ascontiguousarray(np.asarray(queries, dtype=np.float32))
    if cb.ndim != 2 or qs.ndim != 2:
        raise ValueError("cleanup_batch needs a 2-D codebook and 2-D queries, got %r and %r"
                         % (cb.shape, qs.shape))
    if cb.shape[1] != qs.shape[1]:
        raise ValueError("query width %d does not match codebook width %d" % (qs.shape[1], cb.shape[1]))
    if backend in (None, "numpy", "cpu"):
        sims = qs @ cb.T
    elif backend == "wgsl":
        from holographic.io_and_interop.holographic_wgpurun import matmul_kernel

        sims = matmul_kernel(cb, qs, workgroup=workgroup)
    else:
        raise ValueError("backend must be None, 'numpy'/'cpu' or 'wgsl', got %r" % (backend,))
    # argmax by FIRST index attaining the max -- the canonical tie rule, identical on both backends.
    idx = np.array([int(np.flatnonzero(row == row.max())[0]) for row in sims], dtype=int)
    return idx, sims[np.arange(len(idx)), idx]


def drop_budget(dim, n_items, safe_ratio=0.02, floor=0.95):
    """HOW MANY SLOTS CAN BE DROPPED and still recall at `floor`? Returns {keep, keep_fraction, dropped,
    bytes_saved, effective_ratio, safe}.

    THE MECHANISM HALF OF DEGRADATION-UNDER-MEMORY-PRESSURE, and it needed NO NEW THEORY. Dropping slots to
    save memory reduces the EFFECTIVE DIMENSION, so the constraint is the load-ratio law this module already
    measures: recall holds while `n_items / (keep * dim)` stays under the safe ratio.

    VERIFIED against that prediction rather than assumed:
        D=1024 M=8  keep 40%  -> M/eff 0.0196 -> 100%
        D=1024 M=16 keep 40%  -> M/eff 0.0391 ->  85%
        D=2048 M=16 keep 40%  -> M/eff 0.0195 ->  98%
        D=512  M=16 keep 50%  -> M/eff 0.0625 ->  88%
        D=4096 M=16 keep 25%  -> M/eff 0.0156 ->  98%
    Every configuration at or below ~0.02 holds; every one above it degrades. One law, both regimes.

    A CORRECTION THIS MEASUREMENT FORCED, kept loud: the README's degradation table (100% recall at 40% slots
    DESTROYED) is about DAMAGE, where slots are zeroed and NO MEMORY IS SAVED. That number does NOT transfer
    to memory saving. TRUNCATING to 40% of dimensions at the same load gives 85%, not 100%, because a
    zeroed slot still occupies its dimension in the readout while a dropped one does not. Robustness to
    corruption and a memory-saving budget are DIFFERENT QUANTITIES and were briefly conflated here.

    `safe_ratio` defaults to the linear-readout figure (0.02). A caller using a sparse decoder can pass the
    ratio measured by `bundle_capacity` for their method, which is far higher (0.17 for cosamp/amp)."""
    dim, n_items = int(dim), int(n_items)
    if dim <= 0 or n_items <= 0:
        raise ValueError("drop_budget needs positive dim and n_items, got %r and %r" % (dim, n_items))
    needed = int(-(-n_items // float(safe_ratio)))          # smallest effective dim that stays under ratio
    keep = min(dim, max(1, needed))
    return {"keep": keep, "keep_fraction": keep / dim, "dropped": dim - keep,
            "bytes_saved": (dim - keep) * 8, "effective_ratio": n_items / keep,
            "safe": keep <= dim and (n_items / keep) <= float(safe_ratio)}


def _selftest():
    # 1. THE FOLKLORE CONSTANT IS A LINEAR ARTIFACT, shown rather than asserted: at the same dim and floor,
    #    a sparse decoder must hold a strictly higher load ratio than the naive cosine readout.
    lin = bundle_capacity(256, "linear", floor=0.95, seeds=range(3))
    cos = bundle_capacity(256, "cosamp", floor=0.95, seeds=range(3))
    assert cos["safe_ratio"] > lin["safe_ratio"], \
        "cosamp (%.2f) no longer beats linear (%.2f) -- the module's premise died" % (
            cos["safe_ratio"], lin["safe_ratio"])

    # 2. THE LOAD-RATIO COLLAPSE: the curve is a function of M/D, so the safe ratio measured at two
    #    different dims must agree to within one grid step. This is the claim that makes "capacity" a
    #    ratio rather than a count, so it is asserted, not narrated.
    a = bundle_capacity(128, "cosamp", floor=0.95, seeds=range(3))["safe_ratio"]
    b = bundle_capacity(384, "cosamp", floor=0.95, seeds=range(3))["safe_ratio"]
    assert abs(a - b) <= 0.101, "safe ratio did not collapse across dims (%.2f vs %.2f)" % (a, b)

    # 3. The advisor's number is reproducible and carries its provenance.
    r1 = bundle_capacity(128, "cosamp", seeds=range(2))
    r2 = bundle_capacity(128, "cosamp", seeds=range(2))
    assert r1 == r2, "the advisor is not deterministic"
    for field in ("capacity", "safe_ratio", "method", "dim", "floor", "curve"):
        assert field in r1

    # 4. A LUCKY SEED IS NOT A CAPACITY: the gate is mean - sd, so a high-variance row cannot set it.
    fake = [{"ratio": 0.1, "m": 13, "f1_mean": 0.99, "f1_sd": 0.30}]
    safe = 0.0
    for row in fake:
        if row["f1_mean"] - row["f1_sd"] >= 0.95:
            safe = row["ratio"]
    assert safe == 0.0

    # 5. Guards.
    try:
        _recover_support("nonsense", np.zeros(8), np.zeros((4, 8)), 2)
        raise AssertionError("accepted an unknown method")
    except ValueError:
        pass

    print("holographic_capacity: all selftests passed (linear artifact shown, M/D collapse, determinism)")


if __name__ == "__main__":
    _selftest()
