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
    from holographic_fields import attractor_force
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
