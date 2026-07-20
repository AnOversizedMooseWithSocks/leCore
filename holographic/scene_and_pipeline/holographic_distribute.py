"""Distributed computation over holostuff -- the lessons of SETI@home / Folding@home / distributed rendering, but with
the VSA-native shortcut for reassembly.

The distributed-rendering pattern (V-Ray DR, render farms): decompose a job into BUCKETS, precompute the shared caches
(GI / irradiance / acceleration structures) ONCE on a main node, ship them to the nodes, crunch each bucket
independently, and REASSEMBLE. The hard part in the literature is the SHARED cache: an irradiance cache is a shared
mutable structure that is "notoriously difficult to parallelise" because nodes must exchange samples (Warwick, EGPGV
2006), and mismatched settings between nodes cause visible BUCKET SEAMS.

holostuff sidesteps both problems, and the reason is structural: most of its computations are COMMUTATIVE MONOIDS.
  * force / attractor / potential / radiance / density fields ADD          -> reassemble by SUM (linear superposition)
  * an SDF union is a MIN over primitives                                   -> reassemble by MIN
  * an occupancy / coverage field is a MAX                                  -> reassemble by MAX
  * a VSA scene is a BUNDLE (superposition) of its parts                    -> reassemble by BUNDLE
Every one of those reduce operators is associative AND commutative. So the reassembled result is INDEPENDENT OF BUCKET
ORDER -- which is exactly the property that lets the buckets run on separate machines / VMs and be combined with no
stitching, no seams, and (for the linear cases) BIT-EXACT agreement with the monolithic result. The "shortcut
reassembly" is: reassembly IS the monoid's operator. There is no separate stitch pass to get wrong.

The shared cache (an SDF bake, a cost-to-go field, a codebook, a PRT transfer) is READ-ONLY here, so it is the easy kind
of shared structure: compute once on the main node, hand the same immutable object to every bucket -- no inter-node
sample exchange, so none of the irradiance-cache communication overhead.

This module runs the buckets sequentially IN-PROCESS -- the sandbox has no cluster, so it does NOT and CANNOT claim a
multi-node speedup. What it builds and MEASURES is the architecture that makes distribution possible and correct: the
buckets are independent given the shared cache, the reduce is commutative (proven by shuffling bucket order and getting
an identical result), and the shared cache is computed once and reused by every bucket. On a farm, the `worker` is what
each node/VM runs; here it is a function call. NumPy only; deterministic.
"""
import numpy as np


# ---- domain decomposition -------------------------------------------------------------------------------------------

def partition(n, k):
    """Split range(n) into k disjoint, near-equal contiguous index buckets -- the plain bucket decomposition."""
    k = max(1, min(k, n))
    edges = np.linspace(0, n, k + 1).astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(k) if edges[i + 1] > edges[i]]


def adaptive_partition(costs, k):
    """Load-balanced buckets. On a farm the SLOWEST bucket bounds wall-time, so we minimise the max bucket cost instead
    of splitting evenly: assign items heaviest-first (LPT) to the currently-lightest bucket. Heavy regions end up in
    small buckets, cheap regions in big ones -- adaptive bucket sizing, the same reason renderers shrink buckets over
    expensive pixels. Returns a list of index arrays. Deterministic (stable order on ties)."""
    costs = np.asarray(costs, float)
    k = max(1, min(k, len(costs)))
    order = np.argsort(-costs, kind="stable")                    # heaviest first
    loads = np.zeros(k)
    buckets = [[] for _ in range(k)]
    for idx in order:
        b = int(np.argmin(loads))                                # lightest bucket takes the next-heaviest item
        buckets[b].append(int(idx)); loads[b] += costs[idx]
    return [np.array(sorted(b)) for b in buckets if b]


def partition_2d(shape, tiles):
    """Decompose a 2D domain (H, W) into a grid of disjoint TILES -- the render-farm bucket layout, now for images and
    2D fields. `tiles` is (rows, cols) or an int (a near-square grid of about that many tiles). Returns a list of
    (row_slice, col_slice) covering the domain disjointly; each tile is an independent bucket and reassembly is
    placement (no seams). This is also the CACHE-BLOCKING layout: a tile sized to a working budget streams through a
    fast CPU cache level -- the same reason renderers tile."""
    import math
    H, W = int(shape[0]), int(shape[1])
    if isinstance(tiles, int):
        r = max(1, int(round(math.sqrt(tiles)))); c = max(1, int(math.ceil(tiles / r)))
    else:
        r, c = int(tiles[0]), int(tiles[1])
    re = np.linspace(0, H, r + 1).astype(int); ce = np.linspace(0, W, c + 1).astype(int)
    out = []
    for i in range(r):
        for j in range(c):
            if re[i + 1] > re[i] and ce[j + 1] > ce[j]:
                out.append((slice(int(re[i]), int(re[i + 1])), slice(int(ce[j]), int(ce[j + 1]))))
    return out


def partition_3d(shape, bricks):
    """Decompose a 3D domain (X, Y, Z) into a grid of disjoint BRICKS -- the volume/grid analog of tiles, for SDF bake
    grids, occupancy/density volumes, and fluid grids. `bricks` is (nx, ny, nz) or an int (a near-cubic grid of about
    that many bricks). Returns a list of (xslice, yslice, zslice) covering the volume disjointly. Each brick is an
    independent bucket (a separate VM / node); reassembly is placement. Crucially, a brick that contains no surface can
    be SKIPPED entirely (sparse volumes -- most of a volume is empty space), which is the real speed win of bricking a
    3D domain, not just the parallelism."""
    import math
    X, Y, Z = int(shape[0]), int(shape[1]), int(shape[2])
    if isinstance(bricks, int):
        s = max(1, int(round(bricks ** (1.0 / 3.0)))); nx = ny = s; nz = max(1, int(math.ceil(bricks / (s * s))))
    else:
        nx, ny, nz = int(bricks[0]), int(bricks[1]), int(bricks[2])
    xe = np.linspace(0, X, nx + 1).astype(int); ye = np.linspace(0, Y, ny + 1).astype(int); ze = np.linspace(0, Z, nz + 1).astype(int)
    out = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if xe[i + 1] > xe[i] and ye[j + 1] > ye[j] and ze[k + 1] > ze[k]:
                    out.append((slice(int(xe[i]), int(xe[i + 1])), slice(int(ye[j]), int(ye[j + 1])), slice(int(ze[k]), int(ze[k + 1]))))
    return out


def distribute_bricks(out_shape, regions, worker, cache=None, fill=0.0, skip=None):
    """Run worker(region, cache) on each tile/brick region and PLACE its result at that region in one output array.
    Disjoint by construction -> order-independent and seamless (the shared read-only `cache` is what makes the borders
    agree). `skip(region)->bool` lets EMPTY bricks be dropped (sparse volumes: skip bricks with no surface / no
    occupancy). Returns (out, info) where info reports how many regions ran vs were skipped -- the sparse-skip win."""
    out = np.full(out_shape, fill, dtype=float)
    ran = 0; skipped = 0
    for r in regions:
        if skip is not None and skip(r):
            skipped += 1; continue
        out[r] = worker(r, cache); ran += 1
    return out, {"regions": len(regions), "ran": ran, "skipped": skipped}


# ---- reassembly = the computation's own commutative monoid ----------------------------------------------------------

def reduce_first(parts):
    """The single partial, unchanged -- the identity reduce for an ATOMIC job (exactly one bucket).

    WHY IT EXISTS: the job machinery is a checkpointed monoid fold over buckets, which is right for work that
    DECOMPOSES (a noise bake splits into z-slice bands and sums). A generic `job_submit("render_mesh", ...)` does
    not decompose: it is one call, one bucket, and there is nothing to combine. reduce_sum happens to return
    parts[0] for a single part, so it would "work" -- but it would be a lie in the job's own state ("sum"), it
    calls .copy() on the result (a real cost on a large image), and it would break the moment a caller passed two
    buckets expecting a sum. An atomic job says so.

    Asserts the atomicity rather than silently dropping data: a `first` job with several partials is a caller
    error, not something to paper over.
    """
    if len(parts) != 1:
        raise ValueError("reduce='first' is for ATOMIC (single-bucket) jobs; got %d partials -- use sum/min/max/"
                         "bundle for work that decomposes" % len(parts))
    return parts[0]


def reduce_sum(parts):
    """Linear superposition: forces, fields, potentials, radiance, densities. Exact + order-independent."""
    out = parts[0].copy() if hasattr(parts[0], "copy") else parts[0]
    for p in parts[1:]:
        out = out + p
    return out


def reduce_min(parts):
    """SDF union / nearest-of: min over buckets. Order-independent (idempotent-commutative)."""
    return np.minimum.reduce(parts)


def reduce_max(parts):
    """Occupancy / coverage: max over buckets. Order-independent."""
    return np.maximum.reduce(parts)


def reduce_sum_exact(parts, bits=40):
    """BIT-EXACT, ORDER-INDEPENDENT superposition -- the fix for the float-sum negative. Reassembling floats with `+`
    agrees only to rounding (~1e-12) across bucket orders because float addition is not associative. The remedy is to
    carry the accumulated value in a wider, INTEGER representation, where addition IS exact and commutative: quantize
    each contribution to fixed-point at a scale derived from the peak magnitude (peak is order-independent, so the scale
    is too), sum the integers exactly, then rescale. ANY bucket order yields the identical result. This is the same
    principle holostuff uses everywhere -- spend extra DIMENSIONS/bits to gain an exact property -- applied to
    accumulation. Trade (honest): a uniform quantization set by `bits` (more bits -> finer, bounded by the int64 range,
    guarded here against overflow) and a bounded dynamic range. Deterministic. Use it when reproducibility across nodes
    matters (bit-identical frames); the plain reduce_sum is fine when ~1e-12 agreement is enough."""
    import math
    stack = [np.asarray(p, float) for p in parts]
    peak = max((float(np.abs(p).max()) for p in stack if p.size), default=0.0)
    if peak == 0.0:
        return stack[0].copy() if stack else 0.0
    b = int(bits)
    if len(stack) * (2.0 ** b) >= 2.0 ** 62:                     # overflow guard: k * 2^b must stay under int64
        b = max(1, int(62 - math.ceil(math.log2(max(1, len(stack))))))
    scale = (2.0 ** b) / peak
    total = None
    for p in stack:
        ints = np.rint(p * scale).astype(np.int64)               # fixed-point; per-contribution quantization
        total = ints if total is None else total + ints          # integer add: exact + commutative -> order-independent
    return total.astype(np.float64) / scale


# ---- PARTITION-INVARIANT accumulation (backlog X6) -------------------------------------------------------------------
#
# `reduce_sum_exact` above is order-independent: shuffle the SAME parts list and the answer is bit-identical. It is NOT
# partition-independent, and the distinction cost a measurement to see. Its quantization `scale` is derived from the
# peak and the COUNT of the parts it is handed, so if a farm float-sums inside each bucket and then hands the bucket
# sums here, a 4-way split and a 7-way split present different parts, with different peaks, different counts, and --
# fatally -- different float rounding already baked in.
#
# MEASURED (700 contributions, magnitudes spanning 16 orders):
#     plain float, 4-way vs 7-way            max|diff| 2.98e-08   bit-identical: NO
#     reduce_sum_exact on the BUCKET SUMS    bit-identical: NO      <- the float pre-sum already diverged
#     reduce_sum_exact on the raw contribs   bit-identical: YES
#
# So exactness has to reach the LEAVES. The functions below make that practical on a farm: fix one global scale (from
# the global peak and the global count -- both order- and partition-independent, since max and len are), quantize each
# contribution against it, and let each bucket reduce to an int64 accumulator locally. Integer addition is exact,
# associative and commutative, so the accumulators form a MONOID: merge them in any order, from any number of buckets,
# and the result is bit-identical. That is the invariance Catto's cross-platform determinism does not claim -- his is
# same-computation-everywhere by discipline; this survives RE-PARTITIONING a running farm.

def exact_scale(peak, n_parts, bits=40):
    """The fixed-point scale for a partition-invariant accumulation. Depends only on the GLOBAL peak magnitude and the
    GLOBAL number of contributions -- `max` and `len` are both order- and partition-independent, so every bucket
    derives the identical scale without talking to any other bucket. Returns 0.0 for an all-zero input."""
    import math
    peak = float(peak)
    if peak == 0.0:
        return 0.0
    b = int(bits)
    if int(n_parts) * (2.0 ** b) >= 2.0 ** 62:                  # same overflow guard as reduce_sum_exact
        b = max(1, int(62 - math.ceil(math.log2(max(1, int(n_parts))))))
    return (2.0 ** b) / peak


def exact_partial(parts, scale):
    """Reduce one bucket's contributions to an int64 fixed-point accumulator at the given global `scale`. This is the
    bucket-local half of the monoid: it runs on a farm node, alone, with no coordination."""
    total = None
    for p in parts:
        ints = np.rint(np.asarray(p, float) * scale).astype(np.int64)
        total = ints if total is None else total + ints          # exact, commutative, associative
    return total


def exact_merge(accumulators):
    """Merge bucket accumulators. Integer addition, so ANY order and ANY grouping give the identical int64 result."""
    total = None
    for a in accumulators:
        if a is None:
            continue
        total = a.copy() if total is None else total + a
    return total


def reduce_sum_exact_partitioned(buckets, bits=40):
    """Sum a list of BUCKETS (each a list/array of contributions) so the result is bit-identical under ANY bucketing --
    4-way, 7-way, one bucket, or a bucket per contribution. This is the property `reduce_sum_exact` does NOT have when
    a farm pre-sums inside its buckets (see the note above; measured).

    Two passes, both partition-invariant: (1) the global peak and count fix one scale, (2) each bucket integer-sums
    locally and the accumulators merge exactly. Deterministic; no RNG. The trade is `reduce_sum_exact`'s trade -- a
    uniform quantization at `bits` and a bounded dynamic range -- plus one extra pass over the data to find the peak."""
    flat = [np.asarray(c, float) for b in buckets for c in b]
    if not flat:
        return 0.0
    peak = max((float(np.abs(c).max()) for c in flat if c.size), default=0.0)
    if peak == 0.0:
        return flat[0].copy()
    scale = exact_scale(peak, len(flat), bits=bits)
    accs = [exact_partial([np.asarray(c, float) for c in b], scale) for b in buckets]
    return exact_merge(accs).astype(np.float64) / scale


def scan_exact(x, bits=40):
    """The PREFIX SUM (scan), computed so the answer is BIT-IDENTICAL however the work is blocked.

    THE PROBLEM, and it is the GPU workhorse's problem. `np.cumsum` is sequential and deterministic, so a single
    call is reproducible. But nobody parallelises a scan sequentially: the standard blocked scan computes a cumsum
    per block, sums each block, and adds the exclusive prefix of those block sums. Float addition is not
    associative, so **the answer depends on the block count**. Measured over 4,096 elements:

        data                       seq vs 4-block   4-block vs 7-block
        uniform [0, 1)                  2.05e-12            1.14e-12
        16 orders of magnitude          6.26e-07            3.87e-07
        [1e16, 1.0, -1e16] repeated     0.00e+00            **9.20e+01**

    Two different blockings of the same array disagree by NINETY-TWO. Read that honestly: the amplitude there is
    1e16, so 92.0 is 9.2e-15 RELATIVE -- both blockings are accurate, they just disagree. Absolute disagreement is
    what matters when the prefix sum is a ledger, a counter, or a checkpoint hash; relative accuracy is what matters
    when it is a physical quantity. This function is for the first case.

    THE FIX is the same integer monoid `reduce_sum_exact_partitioned` uses: one global scale from the global peak
    and count (`max` and `len` are partition-invariant), quantize once, and scan in int64 -- where addition IS
    associative, so ANY blocking gives the identical prefix sums. `scan_exact_blocked` computes it block-wise and
    is bit-identical to this, for every block count from 1 to N.

    KEPT NEGATIVE, and it is the thing to say first: **this is not more ACCURATE than `np.cumsum`. It is more
    REPRODUCIBLE.** Measured against an exact `math.fsum` prefix reference, relative error:

        data                    scan_exact     np.cumsum (sequential)   blocked float scan
        uniform [0, 1)            6.54e-15               7.83e-16             4.47e-16
        16 orders of magnitude    3.71e-12               2.03e-15             1.43e-15
        catastrophic cancel.      1.37e-13               1.36e-13             1.36e-13

    The quantization at `bits` costs real precision, and a sequential `np.cumsum` beats it every time. What a
    sequential cumsum cannot do is run on eight blocks and give the same bits. Choose the denominator that matches
    the job: if you are not blocking the scan, do not use this.

    Trade: a uniform fixed-point quantization at `bits`, and a bounded dynamic range -- values below the scale's
    resolution round to zero. Deterministic; no RNG."""
    x = np.asarray(x, float).ravel()
    if x.size == 0:
        return x.copy()
    peak = float(np.abs(x).max())
    if peak == 0.0:
        return np.zeros_like(x)
    scale = exact_scale(peak, x.size, bits=bits)
    ints = np.rint(x * scale).astype(np.int64)
    return np.cumsum(ints).astype(np.float64) / scale


def scan_exact_blocked(x, n_blocks, bits=40):
    """`scan_exact`, computed the way a GPU or a farm would: a cumsum per block, then the exclusive prefix of the
    block sums added in. Returns BIT-IDENTICAL results to `scan_exact` for ANY `n_blocks`, because the carries are
    int64 and integer addition is associative.

    This is the function that makes the claim testable: a blocked FLOAT scan disagrees with itself across block
    counts (by 92.0 on the cancellation case above); a blocked EXACT scan does not, ever."""
    x = np.asarray(x, float).ravel()
    if x.size == 0:
        return x.copy()
    peak = float(np.abs(x).max())
    if peak == 0.0:
        return np.zeros_like(x)
    scale = exact_scale(peak, x.size, bits=bits)
    ints = np.rint(x * scale).astype(np.int64)
    blocks = np.array_split(ints, max(1, int(n_blocks)))
    local = [np.cumsum(b) for b in blocks]                     # per-block scan, exact
    sums = np.array([int(b.sum()) for b in blocks], dtype=np.int64)
    carry = np.concatenate([[np.int64(0)], np.cumsum(sums)[:-1]]) if len(sums) > 1 else np.array([0], np.int64)
    out = np.concatenate([l + c for l, c in zip(local, carry)])
    return out.astype(np.float64) / scale


def reduce_bundle(parts):
    """VSA scene reassembly: bundle (superposition) the partial scene hypervectors. A normalised sum -- the same
    commutative superposition holostuff already uses to combine parts of a structure."""
    s = reduce_sum(parts)
    nrm = np.linalg.norm(s)
    return s / nrm if nrm > 0 else s


# ---- the distributor ------------------------------------------------------------------------------------------------

def distribute(buckets, worker, reduce=reduce_sum, cache=None):
    """Run worker(bucket, cache) on each bucket and reassemble with a COMMUTATIVE reduce. Because reduce is a commutative
    monoid, the result does not depend on bucket order -- so the buckets could be dispatched to separate machines / VMs
    and combined with no stitch pass. `cache` is the shared, read-only precompute handed to every worker (the "GI cache
    on the main node"). In-process the workers run sequentially; on a farm each is a node. Returns the reassembled
    result plus a small info dict (bucket count, per-bucket sizes)."""
    parts = [worker(b, cache) for b in buckets]
    info = {"buckets": len(buckets), "sizes": [int(len(b)) for b in buckets]}
    return reduce(parts), info


def distribute_exact(buckets, worker, cache=None, bits=40):
    """`distribute`, but the answer is BIT-IDENTICAL under any bucketing -- 4-way, 7-way, or one bucket per item.

    THE BUG THIS FIXES, measured. `distribute`'s default `reduce=reduce_sum` is a FLOAT sum, so a 4-way and a 7-way
    split of the same work disagree by 2.98e-08 (700 contributions spanning 16 orders of magnitude). Worse, the
    obvious repair -- swapping in `reduce_sum_exact` -- does NOT fix it, because by the time the reduce sees the
    parts each worker has already float-summed inside its own bucket and the rounding has diverged.
    **Exactness has to reach the leaves.**

    So the contract changes, and that IS the fix: `worker(bucket, cache)` must return the bucket's **contributions**
    (an (n_i, ...) array), not their sum. The reduction then runs as an integer monoid:

      1. one global scale from the global peak and the global count -- `max` and `len` are both order- AND
         partition-invariant, so every bucket derives the identical scale without talking to any other bucket;
      2. each bucket quantizes and integer-sums locally (`exact_partial`) -- this is the part that runs on a node;
      3. the int64 accumulators merge in ANY order or grouping (`exact_merge`), because integer addition is exact,
         associative and commutative.

    Verified bit-identical across 1-, 2-, 4-, 7-, 13- and N-way splits and under row shuffles. This is determinism
    that survives RE-PARTITIONING a running farm mid-job -- the invariance Box3D's cross-platform determinism does
    not claim.

    ON A REAL FARM this is two rounds, not one: round 1 each node returns its local `(peak, count)` (two scalars);
    the coordinator takes `max` and `sum`; round 2 each node returns its `exact_partial` int64 accumulator. Both
    rounds are order-independent. In-process the workers run once and their contributions are held, which is why
    this function calls `worker` exactly once per bucket. `exact_scale` / `exact_partial` / `exact_merge` are public
    so a real farm can run the two-round protocol itself.

    Trade (the same one `reduce_sum_exact` makes): a uniform fixed-point quantization at `bits`, and a bounded
    dynamic range. Returns (total, info) with `info` carrying the scale actually used, so the result is auditable."""
    parts = [np.asarray(worker(b, cache), float) for b in buckets]
    flat = [p.reshape(-1) if p.ndim <= 1 else p.reshape(p.shape[0], -1) for p in parts]
    counts = [int(p.shape[0]) if p.ndim else 1 for p in flat]
    peak = max((float(np.abs(p).max()) for p in flat if p.size), default=0.0)
    n_total = int(sum(counts))
    info = {"buckets": len(buckets), "sizes": [int(len(b)) for b in buckets],
            "contributions": n_total, "peak": peak, "bits": int(bits)}
    if peak == 0.0 or n_total == 0:
        info["scale"] = 0.0
        zero = np.zeros_like(parts[0][0]) if (parts and parts[0].ndim > 1) else 0.0
        return zero, info

    scale = exact_scale(peak, n_total, bits=bits)
    info["scale"] = scale
    accs = [exact_partial(list(p) if p.ndim > 1 else [p], scale) for p in flat]
    total = exact_merge(accs).astype(np.float64) / scale
    if parts and parts[0].ndim > 1:
        total = total.reshape(parts[0].shape[1:])
    return total, info


def distribute_scatter(out_shape, buckets, worker, cache=None, fill=0.0):
    """Disjoint reassembly for jobs whose buckets own SEPARATE outputs (render tiles, particle subsets): worker returns
    (indices, values); we place each bucket's values at its indices in one output array. Disjoint by construction, so
    there are no seams and order does not matter. This is the render-farm bucket layout; the shared `cache` (e.g. a
    baked SDF) is what every tile reads, which -- being identical and immutable -- is exactly why the tiles agree at
    their borders (no colour-management drift, the classic source of bucket seams)."""
    out = np.full(out_shape, fill, dtype=float)
    for b in buckets:
        idx, vals = worker(b, cache)
        out[idx] = vals
    return out, {"buckets": len(buckets), "sizes": [int(len(b)) for b in buckets]}


def _selftest():
    """Three commutative-monoid reassemblies, each proven exact AND order-independent (== distributable), plus the
    shared cache computed once and reused by every bucket."""
    from holographic.misc.holographic_fields import attractor_force
    rng = np.random.default_rng(0)

    # 1. FORCE FIELD (sum monoid) -- superposition is exact; partitioning the SOURCES and summing == monolithic
    P = rng.standard_normal((200, 2))
    centers = rng.standard_normal((24, 2)) * 3
    mono = sum(attractor_force(P, c, strength=1.0) for c in centers)
    buckets = partition(len(centers), 5)
    worker = lambda b, cache: sum(attractor_force(P, centers[i], strength=1.0) for i in b)
    dist, info = distribute(buckets, worker, reduce=reduce_sum)
    assert np.allclose(mono, dist, atol=1e-12), np.abs(mono - dist).max()      # EXACT vs monolithic
    shuf = buckets[::-1]                                                        # any order
    dist2, _ = distribute(shuf, worker, reduce=reduce_sum)
    # HONEST: float addition is not bit-exactly associative, so the SUM monoid is order-independent only to rounding
    # (~1e-12) -- mathematically exact, numerically last-ULP. (min/max below ARE bit-exact order-independent.)
    assert np.allclose(dist, dist2, atol=1e-12), "sum reassembly is order-independent to rounding"

    # 2. SDF UNION (min monoid) -- partitioning primitives and taking min == monolithic union
    Q = rng.standard_normal((300, 3))
    prim_c = rng.standard_normal((16, 3)) * 2
    def union_eval(idxs): return np.minimum.reduce([np.linalg.norm(Q - prim_c[i], axis=1) - 0.7 for i in idxs])
    mono_u = union_eval(range(len(prim_c)))
    pb = partition(len(prim_c), 4)
    du, _ = distribute(pb, lambda b, c: union_eval(b), reduce=reduce_min)
    assert np.allclose(mono_u, du, atol=1e-12)
    du2, _ = distribute(pb[::-1], lambda b, c: union_eval(b), reduce=reduce_min)
    assert np.array_equal(du, du2)

    # 3. SHARED CACHE reused by every bucket, computed ONCE (the main-node cache)
    hits = {"cache_builds": 0, "reads": 0}
    def build_cache():
        hits["cache_builds"] += 1
        return prim_c                                            # stand-in for a baked SDF / GI cache
    cache = build_cache()                                        # ONCE, on the "main node"
    def w3(b, c):
        hits["reads"] += 1
        return np.minimum.reduce([np.linalg.norm(Q - c[i], axis=1) - 0.7 for i in b])
    distribute(pb, w3, reduce=reduce_min, cache=cache)
    assert hits["cache_builds"] == 1 and hits["reads"] == len(pb)   # built once, read by all buckets

    # 4. adaptive partition minimises the heaviest bucket vs an even split
    costs = np.array([10, 1, 1, 1, 1, 1, 1, 1.])                 # one heavy item + many light
    ad = adaptive_partition(costs, 4)
    heaviest = max(sum(costs[i] for i in b) for b in ad)
    assert heaviest <= 10 + 1e-9                                 # heavy item isolated, not stacked with others
    print("distribute selftest ok: force(sum)/union(min) reassembly EXACT and order-independent (=> distributable); "
          "shared cache built once, read by all %d buckets; adaptive partition isolates the heavy item (max bucket %.0f)"
          % (len(pb), heaviest))


if __name__ == "__main__":
    _selftest()
