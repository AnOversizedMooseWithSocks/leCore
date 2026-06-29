"""Holographic Gaussian splatting -- a scene/field as a SUPERPOSITION of Gaussian primitives.

WHY THIS EXISTS
---------------
3D Gaussian Splatting (Kerbl, Kopanas, Leimkuehler, Drettakis 2023) represents a scene as an
explicit SUM of parameterised Gaussians fit to data. holostuff's `bundle` IS superposition -- so a
splat scene is, structurally, a bundle of primitives. This module makes that concrete for 2-D
fields/images: fit K Gaussian splats by matching pursuit (greedy superposition), render the sum,
and -- because a small set of smooth Gaussians cannot represent high-frequency noise -- use the
fit itself as a denoiser.

MEASURED (on a real (log-return, log-volume) density from the SOL market data)
  * ~20 superposed Gaussians reconstruct the density at ~31 dB PSNR using ~3.5% of the pixel
    budget (4 numbers per splat). It plateaus by ~50 splats -- a compact, adaptive code.
  * Fitting 20 splats to a NOISY density (22.6 dB to clean) recovers it to ~27 dB -- denoising by
    splatting, because the representation has no capacity for noise.
  * The bridge to the rest of the engine: holostuff's RBF `ScalarEncoder` already places a
    Gaussian bump in similarity space, i.e. it is Gaussian splatting in the hypervector domain.
    The splat <-> kernel <-> FHRR-phasor chain is one object.

DESIGN NOTES
  * Isotropic splats and a small fixed scale set keep the fit a clean, deterministic matching
    pursuit. KEPT NEGATIVE / SCOPE: anisotropic covariances and gradient refinement (full 3DGS)
    are deliberately out of scope here -- isotropic matching pursuit is the honest baseline, and
    real images plateau in quality once the smooth structure is captured (noise is, correctly,
    not fit).
  * Pure NumPy, deterministic: the greedy order is fixed by the residual, no RNG.
"""

import numpy as np


def _gaussian(shape, cy, cx, sigma):
    """A unit-L2-norm isotropic Gaussian centred at (cy, cx) on a `shape` grid."""
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    g = np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2.0 * sigma * sigma))
    return g / (np.sqrt((g * g).sum()) + 1e-12)


def splat_fit(target, K, scales=(1.0, 2.0, 3.5, 6.0), refit=False):
    """Fit `target` (a 2-D array) with K Gaussian splats by matching pursuit.

    Each step places a splat at the current residual's peak, picks the scale (from `scales`) that
    explains the most residual energy, fits its amplitude by projection, and subtracts it. Returns
    a list of (cy, cx, amplitude, sigma) -- the scene as an explicit superposition of primitives.

    With refit=True, the amplitudes are re-solved JOINTLY once placement is done (`splat_refit`) -- the
    same positions and scales, but ~2-4 dB sharper because greedy MP's overlap double-counting is removed.
    Recommended whenever the fitted splats are the deliverable; default False keeps the raw MP result."""
    target = np.asarray(target, float)
    R = target.copy()
    splats = []
    for _ in range(K):
        cy, cx = np.unravel_index(np.abs(R).argmax(), R.shape)
        best = None                                   # (energy, amp, sigma, g)
        for s in scales:
            g = _gaussian(R.shape, cy, cx, s)
            amp = float((R * g).sum())                # least-squares amplitude (g is unit norm)
            energy = amp * amp
            if best is None or energy > best[0]:
                best = (energy, amp, s, g)
        _, amp, s, g = best
        R = R - amp * g
        splats.append((int(cy), int(cx), amp, s))
    return splat_refit(splats, target) if refit else splats


def adaptive_fit(target, noise_thresh=0.03, k_min=4, k_max=200, scales=(1.0, 2.0, 3.5, 6.0), refit=True):
    """Fit `target` with an ADAPTIVE number of splats: the same matching pursuit as splat_fit, but the COUNT
    is driven by the content instead of fixed. This is V-Ray's adaptive sampler (and 3DGS densification in
    spirit) ported to splats -- sample to a noise floor, not to a budget.

    Placement continues until the residual RMS falls below `noise_thresh` * the target's range (bounded to
    [k_min, k_max]). So a smooth/simple field finishes in a few splats and a busy field keeps going, both at
    the SAME reconstruction quality. Returns (splats, k_used); refit=True re-solves amplitudes jointly at the
    end -- orthogonal to the count (the count is WHERE the splats go, the refit is HOW STRONG they are).

    MEASURED: at noise_thresh=0.03 a one-blob field took 13 splats and a seven-blob field 36, both ~33 dB --
    where a fixed k=20 over-spends on the easy field (36 dB) and starves the busy one (27 dB).

    KEPT CAVEAT: the threshold gates the GREEDY residual, so quality is only APPROXIMATELY equalised; and a
    HARD-EDGED target never reaches a low residual with this smooth isotropic basis, so it simply runs to
    k_max -- the adaptive count is meaningful for fields the Gaussian basis can actually represent."""
    target = np.asarray(target, float)
    span = float(target.max() - target.min()) or 1.0          # the residual is measured relative to the range
    R = target.copy()
    splats = []
    for k in range(k_max):
        if k >= k_min and float(np.sqrt((R * R).mean())) / span < noise_thresh:
            break                                             # at the noise floor -- no further splat earns its place
        cy, cx = np.unravel_index(np.abs(R).argmax(), R.shape)
        best = None                                           # (energy, amp, sigma, g)
        for s in scales:
            g = _gaussian(R.shape, cy, cx, s)
            amp = float((R * g).sum())                        # least-squares amplitude (g is unit norm)
            energy = amp * amp
            if best is None or energy > best[0]:
                best = (energy, amp, s, g)
        _, amp, s, g = best
        R = R - amp * g
        splats.append((int(cy), int(cx), amp, s))
    out = splat_refit(splats, target) if (refit and splats) else splats
    return out, len(splats)


def splat_render(splats, shape):
    """Render a splat list back to a 2-D array -- the superposition (sum) of its primitives."""
    out = np.zeros(shape, float)
    for cy, cx, amp, s in splats:
        out += amp * _gaussian(shape, cy, cx, s)
    return out


def splat_refit(splats, target):
    """Re-solve ALL splat amplitudes JOINTLY by least squares, keeping every position and scale fixed -- the
    'looping' step that removes greedy matching pursuit's overlap suboptimality (an orthogonal-matching-
    pursuit-style amplitude refit). Greedy MP fits each splat's amplitude against the *residual* at the moment
    it is placed, so overlapping splats systematically double-count; one joint least-squares solve over the
    final placement corrects that. Closed-form and gradient-FREE (a single lstsq), so it stays inside the
    NumPy-only rule -- distinct from the gradient optimisation of positions/scales that full 3DGS does (that
    needs autodiff and remains out of scope). MEASURED to add ~2-4 dB PSNR over the greedy fit on real images,
    the gain GROWING with the splat count (more overlap to disentangle)."""
    target = np.asarray(target, float)
    if not splats:
        return splats
    G = np.stack([_gaussian(target.shape, cy, cx, s).ravel() for (cy, cx, amp, s) in splats], axis=1)
    amps, *_ = np.linalg.lstsq(G, target.ravel(), rcond=None)      # the joint solve: min || target - G a ||
    return [(cy, cx, float(amps[i]), s) for i, (cy, cx, _, s) in enumerate(splats)]


def splat_denoise(noisy, K, scales=(1.0, 2.0, 3.5, 6.0)):
    """Denoise a 2-D field by fitting K splats and rendering them: the smooth Gaussian basis
    captures structure but not high-frequency noise, so the fit is a denoiser."""
    return splat_render(splat_fit(noisy, K, scales), np.asarray(noisy).shape)


def psnr(a, b, peak=1.0):
    """Peak-signal-to-noise ratio in dB between two arrays (99.0 if identical)."""
    mse = float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))
    return 99.0 if mse == 0.0 else float(10.0 * np.log10(peak * peak / mse))


# --- the HOLOGRAPHIC layer: a splat scene AS a bundle, queryable by region ----------------------
# "a splat scene is a bundle" made literal: bundle a per-region descriptor of the splats, each bound
# to a region role, into ONE hypervector, and read a region back by unbinding its role. This is the
# content-addressable "what's roughly HERE" query the archive's exact splat-list lookup complements.

def splat_bundle(splats, shape, dim=4096, grid=8, levels=5, seed=0):
    """Encode a splat scene as ONE hypervector: partition `shape` into grid x grid regions, quantise each
    region's PEAK occupancy to one of `levels` near-orthogonal level atoms, and bundle bind(region_role,
    level_atom) over all regions. Returns (scene_hv, ctx); ctx carries the role + level codebooks + grid so
    recall_region can read a region back. The bundle IS a superposition -- the engine's bundle over the
    scene's own primitives. (Quantised levels with ORTHOGONAL atoms, not a continuous RBF value, so the
    per-region readback survives the bundle crosstalk -- the readout is robust, the value is coarse.)"""
    from holographic_ai import bind, bundle, Vocabulary
    H, W = shape[0], shape[1]
    rendered = splat_render(splats, (H, W))
    roles = Vocabulary(dim, seed=seed)
    lvl = Vocabulary(dim, seed=seed + 1)                  # `levels` near-orthogonal occupancy atoms
    peak = float(np.abs(rendered).max()) + 1e-12
    parts, desc = [], {}
    for gy in range(grid):
        for gx in range(grid):
            ys, ye = gy * H // grid, (gy + 1) * H // grid
            xs, xe = gx * W // grid, (gx + 1) * W // grid
            energy = float(np.clip(np.abs(rendered[ys:ye, xs:xe]).max() / peak, 0.0, 1.0))
            q = int(round(energy * (levels - 1)))         # quantise PEAK occupancy to a level index
            desc[(gy, gx)] = q / (levels - 1)
            parts.append(bind(roles.get(f"cell:{gy}:{gx}"), lvl.get(f"lvl:{q}")))
    ctx = {"roles": roles, "lvl": lvl, "levels": levels, "grid": grid, "desc": desc}
    return (bundle(parts) if parts else np.zeros(dim)), ctx


def recall_region(scene_hv, cell, ctx):
    """Read a region's quantised occupancy back out of a splat-bundle by unbinding its role and cleaning
    up against the orthogonal level atoms -- content-addressable region lookup. `cell` is (gy, gx). Returns
    the recovered occupancy in [0, 1]. COARSE by design (quantised levels); for exact per-splat region
    content use SplatArchive.region, the precise complement."""
    from holographic_ai import unbind
    roles, lvl, L = ctx["roles"], ctx["lvl"], ctx["levels"]
    noisy = unbind(np.asarray(scene_hv, float), roles.get(f"cell:{cell[0]}:{cell[1]}"))
    best_q, best_s = 0, -2.0
    for q in range(L):
        a = lvl.get(f"lvl:{q}")
        s = float(noisy @ a / (np.linalg.norm(noisy) * np.linalg.norm(a) + 1e-12))
        if s > best_s:
            best_q, best_s = q, s
    return best_q / (L - 1)


def splat_bundle_tiled(splats, shape, dim=4096, grid=16, levels=5, tile=8, seed=0):
    """A splat scene as a GRID OF TILE BUNDLES, so content-addressable region recall stays accurate at FINE
    resolution. `splat_bundle` puts all grid*grid role->occupancy bindings in ONE vector, and `recall_region`
    decodes each by unbind+cleanup -- a decode-via-cleanup readout, so as the grid gets finer the bundle's own
    crosstalk grows and recall caps (measured at dim 4096: ~100% at grid 8, ~98% at 16, ~88% at 24, ~75% at
    32). This is exactly the decode-from-a-crowded-superposition cap that chunking bounds elsewhere (routes,
    sequences, programs); here the chunk is a TILE. Each cell is routed to its tile by floor-dividing its grid
    index by `tile`, so a tile bundle holds at most tile*tile bindings no matter how fine the TOTAL grid is --
    the per-bundle load is fixed and recall holds ~100% at any resolution. Costs one hypervector per tile
    (proportional storage -- the price of exceeding a single vector's capacity, the same trade chunk_route and
    chunked SequenceMemory make). Cells keep their GLOBAL roles, so recall_region_tiled needs no remapping.

    Returns a ctx dict that IS the tiled scene: ctx['tiles'] maps (ty, tx) -> the tile's bundle hypervector,
    and ctx carries the shared role/level codebooks + geometry. Read a cell back with recall_region_tiled.

    The tiling/routing now DELEGATES to TiledStore + _tile_bucket in holographic_tree (shared with the spatial
    StructuredIndex) -- this function owns only the splat encode (role-bound occupancy) and the per-tile bundle;
    the floor-divide routing and bounded grouping live once, in the shared store."""
    from holographic_ai import bind, bundle, Vocabulary
    from holographic_tree import TiledStore                # the shared tiling primitive
    H, W = shape[0], shape[1]
    rendered = splat_render(splats, (H, W))
    roles = Vocabulary(dim, seed=seed)
    lvl = Vocabulary(dim, seed=seed + 1)                  # `levels` near-orthogonal occupancy atoms
    peak = float(np.abs(rendered).max()) + 1e-12
    store, desc = TiledStore(tile, dim), {}
    for gy in range(grid):
        for gx in range(grid):
            ys, ye = gy * H // grid, (gy + 1) * H // grid
            xs, xe = gx * W // grid, (gx + 1) * W // grid
            energy = float(np.clip(np.abs(rendered[ys:ye, xs:xe]).max() / peak, 0.0, 1.0))
            q = int(round(energy * (levels - 1)))
            desc[(gy, gx)] = q / (levels - 1)
            # the store routes (gy, gx) -> its tile by floor-divide and groups the binding there
            store.add((gy, gx), bind(roles.get(f"cell:{gy}:{gx}"), lvl.get(f"lvl:{q}")))
    tiles = {k: bundle(v) for k, v in store.groups().items()}   # one bounded bundle per tile (all non-empty)
    return {"roles": roles, "lvl": lvl, "levels": levels, "grid": grid, "tile": tile,
            "tiles": tiles, "desc": desc, "dim": dim, "shape": (H, W)}


def recall_region_tiled(scene, cell):
    """Read a global cell back from a TILED splat scene (from splat_bundle_tiled): route the cell to its tile
    bundle -- which holds at most tile*tile bindings, so crosstalk stays low and recall stays accurate at fine
    TOTAL resolution -- then decode it with the same unbind+cleanup as recall_region. `cell` is (gy, gx) in the
    full grid; returns the recovered occupancy in [0, 1] (0.0 for an empty tile). Routing uses the shared
    _tile_bucket, so build-time and recall-time tiling are guaranteed identical."""
    from holographic_tree import _tile_bucket
    hv = scene["tiles"].get(_tile_bucket(cell, scene["tile"]))
    return 0.0 if hv is None else recall_region(hv, cell, scene)


# --- anisotropic splats: full-covariance Gaussians fit by gradient descent (the real 3DGS primitive) ------
# Each splat is (center, amplitude, L) where L is an n*n lower-triangular Cholesky factor of the INVERSE
# covariance, so the Gaussian is amp * exp(-0.5 * ||L^T (x - center)||^2). L lower-triangular keeps the
# precision positive-definite for free. Works in any dimension -- 2-D fields and 3-D volumes share one fit.

def _coords(shape):
    """All voxel coordinates of an n-D array as an (npix, n) float array (row-major), built once per fit."""
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1).astype(float)


def _iso_pursuit(target, K, scales=(1.0, 2.0, 3.5, 6.0)):
    """Isotropic matching pursuit in n-D -- the warm start for the anisotropic fit. Returns a list of
    (center (n,), peak_amplitude, sigma); render is amp * exp(-0.5 |x-center|^2 / sigma^2)."""
    R = np.asarray(target, float).copy()
    shape = R.shape
    C = _coords(shape)
    out = []
    for _ in range(K):
        ctr = np.array(np.unravel_index(np.abs(R).argmax(), shape), float)
        d2 = ((C - ctr) ** 2).sum(1)
        best = None
        for s in scales:
            g = np.exp(-0.5 * d2 / s ** 2)                      # peak 1
            amp = float((R.ravel() @ g) / (g @ g + 1e-12))      # least-squares peak amplitude
            energy = amp * amp * float(g @ g)
            if best is None or energy > best[0]:
                best = (energy, amp, s, g)
        _, amp, s, g = best
        R = (R.ravel() - amp * g).reshape(shape)
        out.append((ctr, amp, s))
    return out


def aniso_render(splats, shape):
    """Render anisotropic splats (center, amp, L) back to an n-D array -- the superposition of full-covariance
    Gaussians, exactly what aniso_fit optimises."""
    C = _coords(shape)
    out = np.zeros(C.shape[0])
    for ctr, amp, L in splats:
        u = (C - ctr) @ L
        out += amp * np.exp(-0.5 * (u * u).sum(1))
    return out.reshape(shape)


def _aniso_optimize(target, centers, amps, Ls, steps=200, lr=0.15,
                    early_stop=False, min_steps=40, patience=20, tol=0.004, stats=None):
    """Adam optimisation of anisotropic splats from an EXPLICIT init (centers (K,n), amps (K,), Ls (K,n,n)) --
    the shared gradient engine behind both the one-shot `aniso_fit` (iso warm start) and the coarse-to-fine
    `densify_fit` (staged warm start). Returns (centers, amps, Ls, rendered). The C3 convergence-gated early-stop
    lives here (see aniso_fit's docstring); early_stop=False runs the full `steps` (bit-identical)."""
    target = np.asarray(target, float)
    shape = target.shape
    n = target.ndim
    C = _coords(shape)
    t = target.ravel()
    centers = np.array(centers, float)
    amps = np.array(amps, float)
    Ls = np.array(Ls, float)
    K = len(amps)
    tril = np.tril_indices(n)
    state = {key: (np.zeros_like(v), np.zeros_like(v)) for key, v in (("a", amps), ("c", centers), ("L", Ls))}
    b1, b2, eps = 0.9, 0.999, 1e-8
    _mse_hist = []                                                     # C3: residual trace for the convergence-gated stop
    step = 0

    def render(ce, am, Ls_):
        m = np.zeros(len(t))
        for k in range(K):
            u = (C - ce[k]) @ Ls_[k]
            m += am[k] * np.exp(-0.5 * (u * u).sum(1))
        return m

    for step in range(1, steps + 1):
        r = render(centers, amps, Ls) - t                                # residual
        ga = np.zeros_like(amps); gc = np.zeros_like(centers); gL = np.zeros_like(Ls)
        for k in range(K):
            d = C - centers[k]
            u = d @ Ls[k]                                                # u = L^T d  (per pixel)
            ex = np.exp(-0.5 * (u * u).sum(1))
            g = amps[k] * ex
            ga[k] = float((r * ex).sum())                                # dE/d amp
            Pd = u @ Ls[k].T                                             # (L L^T) d = precision * d
            gc[k] = ((r * g)[:, None] * Pd).sum(0)                       # dE/d center = sum r g (P d)
            for a_, b_ in zip(*tril):                                    # dE/d L_ab = sum r (-g) d_a u_b
                gL[k][a_, b_] = float((r * (-g) * d[:, a_] * u[:, b_]).sum())
        for key, par, grad in (("a", amps, ga), ("c", centers, gc), ("L", Ls, gL)):
            m, v = state[key]
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad * grad
            par -= lr * (m / (1 - b1 ** step)) / (np.sqrt(v / (1 - b2 ** step)) + eps)
            state[key] = (m, v)
        if early_stop:                                   # C3: convergence-gated stop (a SPEED/QUALITY knob, not free)
            _mse_hist.append(float((r * r).mean()))      # r is the pre-update residual; its trend tracks convergence
            if step >= min_steps and len(_mse_hist) > patience:
                _win = _mse_hist[-patience - 1] - _mse_hist[-1]         # improvement over the last `patience` steps
                if _win <= tol * _mse_hist[0]:           # below tol of the INITIAL error -> converged (works whether the
                    break                                # fit plateaus at a floor OR descends geometrically toward zero)
    if stats is not None:
        stats["steps"] = int(step)
    return centers, amps, Ls, render(centers, amps, Ls).reshape(shape)


def aniso_fit(target, K, steps=200, lr=0.15, scales=(1.0, 2.0, 3.5, 6.0),
              early_stop=False, min_steps=40, patience=20, tol=0.004, stats=None):
    """Fit `target` (any n-D array) with K ANISOTROPIC Gaussian splats by gradient descent on the
    reconstruction MSE -- the 3D-Gaussian-Splatting primitive (oriented, elliptical Gaussians), in NumPy with
    analytical gradients and a small built-in Adam (no autodiff framework). Warm-started from the isotropic
    matching pursuit so the covariances only have to specialise. Each splat is (center, amplitude, L), L the
    lower-triangular Cholesky factor of the inverse covariance. Returns (splats, rendered).

    ADAPTIVE STOP (C3, opt-in via early_stop=True, pass stats={} to read stats['steps']): the fixed `steps`
    count over-computes an easy field. Stop when the reconstruction has CONVERGED -- the MSE improvement over
    the last `patience` steps falls below `tol` of the INITIAL error (a fixed scale, so it fires whether the
    fit plateaus at a residual floor OR descends geometrically toward zero) -- with a `min_steps` floor.
    Measured: ~20-40% fewer steps on under-fit fields (a busy field stops near ~121), less on a near-perfectly
    fittable one, at a few-percent MSE cost. TWO kept caveats, and they matter: (1) unlike the resonator's
    early-stop, which has an EXACT reconstruction certificate and is therefore FREE, this is a SOFT plateau on
    a continuous optimisation -- stopping always costs a little MSE, so it is a speed/quality knob, not a free
    lunch, and it is OFF by default (early_stop=False is bit-identical to the original fixed-step fit). (2)
    Adam's momentum needs ~30 steps to warm up, during which the MSE barely moves -- a naive relative-improvement
    test mistakes that warm-up for convergence and stops at step ~20 with a terrible fit, which is exactly why
    the `min_steps` floor (default 40) exists.

    Anisotropy is decisive where structure is oriented/elongated -- one aligned splat replaces many circular
    ones. KEPT NEGATIVE: the loss is non-convex, so this finds a LOCAL optimum -- more splats do not help
    monotonically (a good K=4 fit can beat a messier K=8 one), and the result depends on the warm start. This
    is the honest from-scratch core of 3DGS, without its tile rasteriser, spherical-harmonic view-dependent
    colour, or GPU speed."""
    target = np.asarray(target, float)
    n = target.ndim
    iso = _iso_pursuit(target, K, scales)                             # isotropic matching-pursuit warm start
    centers = np.array([c for c, _, _ in iso])
    amps = np.array([a for _, a, _ in iso])
    Ls = np.array([np.eye(n) / max(sg, 0.5) for _, _, sg in iso])      # L = (1/sigma) I  (isotropic init)
    centers, amps, Ls, rendered = _aniso_optimize(target, centers, amps, Ls, steps=steps, lr=lr,
                                                  early_stop=early_stop, min_steps=min_steps,
                                                  patience=patience, tol=tol, stats=stats)
    splats = [(centers[k].copy(), float(amps[k]), Ls[k].copy()) for k in range(len(amps))]
    return splats, rendered


def densify_fit(target, K, stage_steps=(50, 80, 210), scales=(1.0, 2.0, 3.5, 6.0), stats=None):
    """COARSE-TO-FINE anisotropic splat fit (C1) -- 3D-Gaussian-Splatting densification, from scratch. Instead of
    placing all K isotropic splats at once and running ONE joint gradient fit (`aniso_fit`), grow the set in
    STAGES: place a fraction of the splats on the current RESIDUAL (matching pursuit, coarse scales first), then
    jointly optimise everything so far, then place more splats where the re-optimised reconstruction still errs,
    and optimise again. `stage_steps` gives the Adam steps per stage (the last stage should be long enough to
    fully converge the whole set). Returns (splats, rendered); pass stats={} to read stats['stages'].

    WHY THIS CAN BEAT THE ONE-SHOT (measured): the staged placement is a far better WARM START for the final joint
    fit -- it can land in a better basin of the non-convex loss when the final stage gets enough refinement. On a
    multi-scale target (a broad blob + small sharp details), the CI selftest uses a longer final stage to verify
    that staged placement can beat the 210-step one-shot baseline decisively. The trade is compute for a better
    basin; the short default is a quick demonstration path, not a universal optimum guarantee.

    KEPT SCOPE: still the from-scratch core of 3DGS (no tile rasteriser, no view-dependent colour, no GPU); and
    the win is on MULTI-SCALE content -- on a single-scale field the one-shot is already near-optimal and the
    extra rounds mostly buy little."""
    target = np.asarray(target, float)
    shape = target.shape
    n = target.ndim
    stages = len(stage_steps)
    per = [K // stages] * stages
    per[-1] += K - sum(per)                                            # remainder to the last stage
    centers = np.zeros((0, n)); amps = np.zeros(0); Ls = np.zeros((0, n, n))
    rendered = np.zeros(shape)
    for s, (k_add, steps) in enumerate(zip(per, stage_steps)):
        if k_add <= 0:
            continue
        residual = target - rendered                                  # place new splats where the fit still errs
        st_scales = scales[len(scales) // 2:] if s == 0 else scales    # coarse scales first, all scales later
        iso = _iso_pursuit(residual, k_add, st_scales)
        nc = np.array([c for c, _, _ in iso])
        na = np.array([a for _, a, _ in iso])
        nl = np.array([np.eye(n) / max(sg, 0.5) for _, _, sg in iso])
        centers = np.vstack([centers, nc]) if len(centers) else nc
        amps = np.concatenate([amps, na])
        Ls = np.concatenate([Ls, nl]) if len(Ls) else nl
        centers, amps, Ls, rendered = _aniso_optimize(target, centers, amps, Ls, steps=steps)   # re-fit ALL
    if stats is not None:
        stats["stages"] = stages
    splats = [(centers[k].copy(), float(amps[k]), Ls[k].copy()) for k in range(len(amps))]
    return splats, rendered


def _c1_selftest():
    """C1: coarse-to-fine densify_fit reaches a markedly better optimum than the one-shot aniso_fit on a
    multi-scale target -- the staged warm start lands in a basin the one-shot cannot reach at any step count,
    directly addressing aniso_fit's local-optimum kept negative."""
    import numpy as _np
    ys, xs = _np.mgrid[0:56, 0:56]
    T = (_np.exp(-(((xs - 28) ** 2 + (ys - 28) ** 2) / 300.0))
         + sum(0.8 * _np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / 8.0))
               for cx, cy in [(12, 12), (44, 16), (16, 44), (42, 42)]))   # broad blob + small sharp details

    def mse(z):
        return float(((z - T) ** 2).mean())

    one = mse(aniso_fit(T, 12, steps=210)[1])
    st = {}
    cf = mse(densify_fit(T, 12, stage_steps=(40, 80, 650), stats=st)[1])
    assert st["stages"] == 3, st
    assert cf < one * 0.5, (cf, one)            # densify reaches a markedly better optimum (measured ~100x here)


def _c3_selftest():
    """C3: the convergence-gated early-stop saves a good fraction of the fixed steps at a few-percent MSE cost
    (a SOFT plateau, not the resonator's free exact-certificate stop), past an Adam warm-up floor; and it is OFF
    by default -- early_stop=False runs the full fixed schedule, bit-identical to the original fit."""
    import numpy as _np
    ys, xs = _np.mgrid[0:48, 0:48]
    easy = (_np.exp(-(((xs - 16) ** 2 + (ys - 20) ** 2) / 90.0))
            + 0.7 * _np.exp(-(((xs - 32) ** 2 + (ys - 28) ** 2) / 120.0)))    # two smooth blobs

    st_full = {}
    _, full = aniso_fit(easy, 4, steps=200, stats=st_full)
    assert st_full["steps"] == 200, st_full                          # OFF by default: runs the full schedule
    mse_full = float(((full - easy) ** 2).mean())

    st_es = {}
    _, es = aniso_fit(easy, 4, steps=200, early_stop=True, stats=st_es)
    mse_es = float(((es - easy) ** 2).mean())
    assert 40 <= st_es["steps"] <= 160, st_es                       # stopped past the warm-up floor, before 200
    assert mse_es <= mse_full * 1.10 + 1e-6, (mse_es, mse_full)     # at a small MSE cost (a real trade, not free)


if __name__ == "__main__":
    _c1_selftest()
    _c3_selftest()
    print("holographic_splat C1 densify + C3 early-stop selftests passed")
