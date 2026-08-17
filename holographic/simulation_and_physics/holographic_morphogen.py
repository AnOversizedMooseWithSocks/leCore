"""Cell-aggregate morphogenesis: soft volumetric particles relaxed by ANALYTIC gradients.

WHAT THIS IS (backlog Workstream F, item F1). Stage one of the energy-based morphogenesis
pipeline: cells as soft spheres (center + radius) that proliferate and relax under a pairwise
potential, producing the compact genus-0 aggregate that later stages sculpt into a body plan.

SOTA CHECK (searched 2026-08-16 before building):
  * The differential adhesion hypothesis (Steinberg 1962/63) and the Cellular Potts Model
    (Graner & Glazier, PRL 1992 / PRE 1993) are the field's classics; CPM is LATTICE-based
    Monte Carlo. We take the LATTICE-FREE PARTICLE route (as in Palachanis, Szabo & Merks),
    which is a recognised paradigm precisely because each representation has its own
    artifacts -- and which matches this engine's existing particle machinery.
  * The closest current work to our source document is "Engineering morphogenesis of cell
    clusters with differentiable programming" (Nature Computational Science, Aug 2025):
    same energy-minimisation framing, driven by AUTODIFF.
  * HOUSE DEPARTURE, stated: autodiff is forbidden here (hard constraint), and for pairwise
    potentials it is not needed. The gradient of a radial pair potential is exact in closed
    form -- dE/dx_i = sum_j phi'(d_ij) * (x_i - x_j)/d_ij -- which is one line, faster than
    a tape, and EXACTLY verifiable against finite differences. The finite-difference check is
    this module's planted truth; it is not a weaker method, it is the same method with the
    derivative done by hand and then PROVED right.

RULE-0 AUDIT (2026-08-16) -- what this module REUSES rather than rebuilds:
  * `holographic_fields.spatial_hash_pairs` -- the O(N) uniform-grid neighbour cull. No new
    neighbour search is written here; "cull, don't batch" is already the house primitive.
  * `holographic_optimize.fd_gradient` -- the central finite-difference gradient, used as the
    VERIFICATION instrument in the selftest. The checker already existed; we only had to
    point it at the new energy.
  * `blue_noise_sample` was audited for seeding initial positions and NOT used: Poisson-disk
    gives a maximal set at a fixed radius, whereas proliferation needs division from a seed
    with radii that vary. Recorded so the next session does not re-audit it.

TURING'S QUESTION (the workstream's standing gate, from his 1952 morphogenesis paper): does
the pattern come from the DYNAMICS or from hand-placed initial conditions? ANSWER FOR F1,
stated plainly: the sphere here is DYNAMICS -- it is the minimum of an isotropic pair
potential from ANY seeded start, and the selftest measures that from a deliberately
non-spherical (flat slab) initial condition. No body-plan structure is present at this
stage; anything limb-like appearing later comes from F2's morphogen fields, not from here.

KEPT NEGATIVES:
  * No autodiff, no learned weights, no scipy optimiser -- plain gradient descent with
    backtracking. Measured: it reaches the sphericity plateau in a few hundred steps at
    N<=512, which is all F3 needs from it.
  * Cells are SPHERES, not the two-particle shape/orientation model (Wang & Nakano 2025).
    Orientation matters for elongated-cell phenomena (vasculogenesis); it does not matter
    for producing a compact aggregate, so it is deliberately out of scope. Adding it later
    is a second particle per cell, not a rewrite.
  * The potential is a soft-core repulsion plus a finite-range attractive well, NOT
    Lennard-Jones: LJ's r^-12 core is stiff enough to demand tiny steps, and the biology
    does not need a hard core. This was chosen for step size, on purpose.
"""

import numpy as np


def pair_energy_and_grad(positions, radii, k_rep=1.0, k_att=0.35, cutoff_scale=1.5,
                        core="inverse"):
    """Total pair-potential energy and its EXACT analytic gradient (N,3).

    Per pair, with d = |x_i - x_j| and rest distance r0 = r_i + r_j:
        overlap  (d < r0):  E = k_rep/2 * (d - r0)^2                 [soft core]
        contact  (r0 <= d < r_cut): E = -k_att * w(d)                [attractive well]
            w(d) = (1 - u^2)^2,  u = (d - r0) / (r_cut - r0)         [smooth to 0 at r_cut]
        beyond r_cut: no interaction (this is what makes the cull sound)
    The core branch carries a -k_att offset so both branches agree in VALUE and SLOPE at
    d = r0 (w(r0)=1, w'(r0)=0), and the well reaches 0 with zero slope at r_cut -- E is C^1
    across the whole range. WHY THIS MATTERS: relax() compares energies between configurations
    in which pairs cross branch boundaries; a jump there corrupts the line search, and a kink
    makes descent chatter.

    dE/dx_i = sum_j phi'(d) * (x_i - x_j)/d, assembled with np.add.at so each pair contributes
    once with opposite signs (Newton's third law holds by construction, not by hope).
    """
    from holographic.misc.holographic_fields import spatial_hash_pairs
    positions = np.asarray(positions, float)
    radii = np.asarray(radii, float)
    n = positions.shape[0]
    grad = np.zeros_like(positions)
    if n < 2:
        return 0.0, grad
    # cull radius: the largest possible interaction distance in this population
    rmax = float(radii.max())
    r_cut_max = cutoff_scale * 2.0 * rmax
    pairs = spatial_hash_pairs(positions, r_cut_max)
    if len(pairs) == 0:
        return 0.0, grad
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    i, j = pairs[:, 0], pairs[:, 1]
    dvec = positions[i] - positions[j]
    d = np.linalg.norm(dvec, axis=1)
    # coincident cells would divide by zero; nudge deterministically along +x (no rng in a
    # gradient -- a random tie-break would make the energy non-deterministic)
    zero = d < 1e-12
    if np.any(zero):
        dvec[zero] = np.array([1e-9, 0.0, 0.0])
        d = np.linalg.norm(dvec, axis=1)
    r0 = radii[i] + radii[j]
    r_cut = cutoff_scale * r0
    e = np.zeros_like(d)
    dphi = np.zeros_like(d)          # dE/dd, per pair
    core_m = d < r0
    # the -k_att offset is LOAD-BEARING, not cosmetic: without it the core branch is 0 at
    # contact while the well branch is -k_att, so E jumps by k_att exactly where pairs cross
    # r0 -- and relax()'s backtracking line search COMPARES ENERGIES across configurations
    # where pairs switch branches, so a discontinuity makes that comparison meaningless.
    # (Caught by this module's own C^1 selftest before any result was believed.)
    #
    # CORE SHAPE -- MEASURED BUG, and the whole reason this is a parameter:
    # the first version used a QUADRATIC core, which is FINITE at d=0. A finite core cannot
    # exclude volume: each of a cell's ~N neighbours contributes up to k_att of inward pull
    # while the core resists only linearly, so a dense aggregate COLLAPSES. Measured on the
    # shipped F1 code: 200 cells packed into 1.3 cell-diameters, mean nearest-neighbour
    # distance 0.15 against an ideal of 1.0, mean degree 199/200 -- every cell overlapping
    # every other. F1's SPHERICITY test passed the whole time, because a collapsed blob is
    # perfectly spherical: the gate was real but it was not the gate that catches this.
    # The "inverse" core k_rep*(r0/d - 1)^2 DIVERGES as d->0, so exclusion holds at any
    # density, and it still matches the well in value AND slope at d=r0. "quadratic" is
    # retained only to reproduce pre-fix numbers.
    if core == "quadratic":
        e[core_m] = 0.5 * k_rep * (d[core_m] - r0[core_m]) ** 2 - k_att
        dphi[core_m] = k_rep * (d[core_m] - r0[core_m])
    else:
        ratio = r0[core_m] / d[core_m]
        e[core_m] = k_rep * (ratio - 1.0) ** 2 - k_att
        # d/dd [ (r0/d - 1)^2 ] = 2 (r0/d - 1) * (-r0/d^2)
        dphi[core_m] = k_rep * 2.0 * (ratio - 1.0) * (-r0[core_m] / d[core_m] ** 2)
    well = (~core_m) & (d < r_cut)
    u = np.zeros_like(d)
    u[well] = (d[well] - r0[well]) / (r_cut[well] - r0[well])
    w = (1.0 - u[well] ** 2) ** 2
    e[well] = -k_att * w
    # dw/dd = 2(1-u^2)(-2u) * du/dd
    dwdu = 2.0 * (1.0 - u[well] ** 2) * (-2.0 * u[well])
    dphi[well] = -k_att * dwdu / (r_cut[well] - r0[well])
    dirv = dvec / d[:, None]
    contrib = dphi[:, None] * dirv
    np.add.at(grad, i, contrib)
    np.add.at(grad, j, -contrib)
    return float(e.sum()), grad


def relax(positions, radii, steps=300, step0=0.05, k_rep=1.0, k_att=0.35,
          cutoff_scale=1.5, tol=1e-9, core="inverse"):
    """Gradient descent with backtracking line search to the aggregate's energy minimum.

    Backtracking (halve the step until the energy actually decreases) rather than a fixed
    rate: with a soft core the curvature varies by orders of magnitude between a crowded
    interior and a loose surface, and a fixed rate either crawls or explodes. Deterministic --
    no rng anywhere in this loop. Returns (positions, history) with per-step energies, so a
    caller can SEE convergence rather than trust it."""
    x = np.array(positions, float, copy=True)
    radii = np.asarray(radii, float)
    e, g = pair_energy_and_grad(x, radii, k_rep, k_att, cutoff_scale, core)
    hist = [e]
    step = step0
    for _ in range(int(steps)):
        gn = float(np.linalg.norm(g))
        if gn < tol:
            break
        trial = step
        for _ in range(24):                       # backtracking
            y = x - trial * g
            e2, g2 = pair_energy_and_grad(y, radii, k_rep, k_att, cutoff_scale, core)
            if e2 <= e:
                break
            trial *= 0.5
        else:
            break                                  # no downhill step exists; stop honestly
        x, e, g = y, e2, g2
        hist.append(e)
        step = min(trial * 1.6, step0 * 8.0)       # grow again after a successful step
    return x, hist


def proliferate(positions, radii, n_new, rng, sep_frac=0.85):
    """Cell division: pick existing cells, place each daughter JUST INSIDE contact distance.

    `sep_frac` is a fraction of the contact distance 2r, and its default is NOT arbitrary --
    it is the second half of the collapse fix. MEASURED: with the divergent "inverse" core,
    placing a daughter at 0.25r (the first version) sits it where the core energy is ~49x
    k_rep, so the line search takes a violent step that flings the pair BEYOND the attraction
    cutoff, where nothing pulls them back -- the aggregate exploded to a mean neighbour
    distance of 36 diameters. Placing at 0.85 * 2r leaves a mild overlap that relaxation
    resolves in a few steps. Divergent cores and near-coincident spawns are incompatible;
    that lesson generalises to any spawn-into-a-potential code.

    Deterministic given `rng` (dedicated generator, per the house rule that planted truths
    own their seeds)."""
    x = list(np.asarray(positions, float))
    r = list(np.asarray(radii, float))
    for _ in range(int(n_new)):
        i = int(rng.integers(len(x)))
        d = rng.normal(size=3)
        d /= (np.linalg.norm(d) + 1e-12)
        x.append(x[i] + d * (2.0 * r[i]) * sep_frac)
        r.append(r[i])
    return np.array(x), np.array(r)


def packing_quality(positions):
    """Mean nearest-neighbour distance -- the gate that SPHERICITY MISSED.

    A collapsed aggregate (every cell sitting on top of every other) is perfectly spherical,
    so sphericity alone passed a physically broken body for a whole session. This measure
    catches it: for cells of radius r the ideal nearest-neighbour distance is 2r, so
    mean_nn/(2r) near 1.0 means real packing, and << 1 means collapse. Reported alongside
    sphericity everywhere, because ONE shape statistic is never enough."""
    p = np.asarray(positions, float)
    if len(p) < 2:
        return 0.0
    nn = []
    for i in range(len(p)):
        d = np.linalg.norm(p - p[i], axis=1)
        d[i] = np.inf
        nn.append(d.min())
    return float(np.mean(nn))


def sphericity(positions):
    """How ball-like is this point set? Ratio of the smallest to largest eigenvalue of the
    covariance (1.0 = isotropic, 0 = flat/linear).

    WHY THIS MEASURE and not a surface-area formula: the classic sphericity index needs a
    surface mesh, which does not exist until F3 tetrahedralises. The covariance ratio is
    computable on the raw point set, is rotation-invariant, and is exactly what "the
    isotropic minimum is a ball" predicts should approach 1."""
    p = np.asarray(positions, float)
    c = p - p.mean(axis=0)
    ev = np.linalg.eigvalsh((c.T @ c) / max(len(p), 1))
    ev = np.clip(ev, 0.0, None)
    return float(ev.min() / (ev.max() + 1e-12))


def grow_aggregate(n_cells=64, radius=0.5, seed=0, steps=300, relax_every=16,
                   k_rep=1.0, k_att=0.35, start="slab", cutoff_scale=1.5,
                   core="quadratic", anneal=True):
    """Grow a cell aggregate from a seed by alternating proliferation and relaxation.

    start="slab" begins from a deliberately FLAT (non-spherical) slab so that any sphericity
    in the result is produced by the DYNAMICS, not smuggled in by the initial condition --
    Turing's question, answered by construction rather than by assertion. start="point"
    begins from a single cell. steps=0 runs proliferation with NO relaxation, which is the
    honest control the selftest measures against.

    MEASURED NEGATIVE, kept because it is a real property of gradient descent and will bite
    again: a PERFECTLY planar slab is a CRITICAL POINT of this energy -- every z-gradient is
    zero by symmetry -- so relaxation alone cannot thicken it (measured: 0.000 sphericity
    after 600 steps, energy falling the whole time as it packs IN-PLANE). The symmetry must
    be broken by something; here it is proliferation's 3D division jitter. This is why the
    control is "proliferation without relaxation" and not "slab without relaxation".

    Returns {"positions", "radii", "energy", "sphericity", "history"}. Deterministic:
    same seed, same aggregate, bit for bit."""
    rng = np.random.default_rng(int(seed))
    if start == "slab":
        m = max(4, int(np.ceil(np.sqrt(max(n_cells // 4, 4)))))
        gx, gy = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
        pos = np.stack([gx.ravel() * radius * 1.8, gy.ravel() * radius * 1.8,
                        np.zeros(gx.size)], axis=1).astype(float)
    else:
        pos = np.zeros((1, 3))
    rad = np.full(len(pos), float(radius))
    hist = []
    while len(pos) < n_cells:
        add = min(relax_every, n_cells - len(pos))
        pos, rad = proliferate(pos, rad, add, rng)
        if steps > 0:                      # steps=0 is the HONEST CONTROL: proliferation
            pos, h = relax(pos, rad, steps=max(steps // 8, 20),   # only, zero relaxation,
                           k_rep=k_rep, k_att=k_att,             # so the contrast measures
                           cutoff_scale=cutoff_scale, core=core)
            hist.extend(h)                                        # the dynamics and nothing else
    if steps > 0:
        pos, h = relax(pos, rad, steps=steps, k_rep=k_rep, k_att=k_att,
                       cutoff_scale=cutoff_scale, core=core)
        hist.extend(h)
    if anneal and steps > 0:
        # SOFT-THEN-INFLATE, the fix for the jamming negative documented above and the
        # standard schedule in the packing literature (Lubachevsky-Stillinger-style
        # inflation): grow and round up under a SOFT core where cells may pass through one
        # another and therefore rearrange, then stiffen the core in stages so exclusion is
        # imposed gradually and the aggregate has a chance to accommodate it. MEASURED on
        # N=120: soft alone gives sphericity 1.000 at packing 0.202 (round but COLLAPSED);
        # a single hard relax gives 0.926 / 0.505; the full ladder gives 0.803 / 0.959 --
        # both properties real at once, which neither endpoint achieves.
        for stage_k in (0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            pos, h = relax(pos, rad, steps=max(steps, 200), k_rep=stage_k * k_rep,
                           k_att=k_att, cutoff_scale=cutoff_scale, core="inverse",
                           step0=0.02)
            hist.extend(h)
    e, _ = pair_energy_and_grad(pos, rad, k_rep, k_att, cutoff_scale,
                                "inverse" if anneal else core)
    return {"positions": pos, "radii": rad, "energy": e,
            "sphericity": sphericity(pos),
            "packing": packing_quality(pos) / (2.0 * float(np.mean(rad))),
            "history": hist}


# ---------------------------------------------------------------------------
# F2: MORPHOGENS AND DIFFERENTIAL ADHESION.
#
# SOTA CHECK (searched 2026-08-16, literature current to July 2026): the field's consensus
# for LIMB patterning specifically is that BOTH classical mechanisms run IN PARALLEL --
# a self-organising Turing reaction-diffusion network (periodic pattern: where digits go)
# MODULATED BY a Wolpert positional-information gradient (identity: which digit is which).
# Raspopovic et al. (Science 2014) identified the Bmp-Sox9-Wnt Turing network modulated by
# morphogen gradients; Green & Sharpe (Development 2015) named this "Mode 2"; the coordination
# of growth with self-organisation is still active work (PLOS Comput Biol, 2026). We
# therefore implement BOTH and let the caller weight them, rather than the source document's
# simpler "prescribed fixed spatial patterns" (which is PI alone).
#
# RULE-0 AUDIT: the engine's existing `reaction_diffusion` is a GRID HyperCA (size x size
# lattice) and our cells are an OFF-LATTICE point cloud, so it cannot be called directly --
# a genuine partial mismatch, recorded here so this is not re-audited. What IS reused:
# spatial_hash_pairs (the neighbour graph comes free from the same cull the energy uses) and
# graph_connected_components (lobe counting is the generic flood fill, not a new algorithm).
# ---------------------------------------------------------------------------

def cell_graph(positions, radii, cutoff_scale=1.5):
    """Neighbour graph over cells as (pairs, degree) -- the same uniform-grid cull the pair
    energy uses, so the chemistry and the mechanics agree about who is adjacent to whom
    (they must: an adhesion term between non-neighbours would be a force from nowhere)."""
    from holographic.misc.holographic_fields import spatial_hash_pairs
    positions = np.asarray(positions, float)
    radii = np.asarray(radii, float)
    r_cut = cutoff_scale * 2.0 * float(radii.max())
    pairs = np.asarray(spatial_hash_pairs(positions, r_cut), int).reshape(-1, 2)
    deg = np.zeros(len(positions))
    if len(pairs):
        np.add.at(deg, pairs[:, 0], 1.0)
        np.add.at(deg, pairs[:, 1], 1.0)
    return pairs, deg


def reaction_diffusion_cells(positions, radii, steps=400, feed=0.037, kill=0.06,
                             du=0.16, dv=0.08, dt=1.0, seed=0, cutoff_scale=1.5):
    """Gray-Scott reaction-diffusion ON THE CELL GRAPH (off-lattice), not on a grid.

    U + 2V -> 3V,  V -> P:   du/dt = Du L u - u v^2 + F(1-u)
                             dv/dt = Dv L v + u v^2 - (F+k) v
    where L is the graph Laplacian of the neighbour graph (sum over neighbours minus degree
    times self) -- the off-lattice analogue of the 5-point stencil. Gray-Scott rather than a
    generic activator-inhibitor because its (F,k) plane is the best-charted parameter space
    in the pattern-formation literature, so a caller can look up regimes instead of guessing.

    THIS IS THE TURING HALF (emergent, symmetry-breaking from a seeded perturbation).
    Deterministic given `seed`: the initial V perturbation owns its own generator.
    Returns (u, v) per cell."""
    pairs, deg = cell_graph(positions, radii, cutoff_scale)
    n = len(np.asarray(positions))
    rng = np.random.default_rng(int(seed))
    u = np.ones(n)
    v = np.zeros(n)
    # seed the instability in a compact patch: Gray-Scott needs a finite perturbation, a
    # uniform state is a fixed point (the same lesson as F1's planar critical point)
    if n:
        c = np.asarray(positions, float)
        centre = c[int(rng.integers(n))]
        d = np.linalg.norm(c - centre, axis=1)
        seedmask = d < (np.median(d) * 0.4 + 1e-9)
        v[seedmask] = 0.5
        u[seedmask] = 0.25
    if len(pairs) == 0:
        return u, v
    i, j = pairs[:, 0], pairs[:, 1]
    for _ in range(int(steps)):
        lu = np.zeros(n)
        lv = np.zeros(n)
        np.add.at(lu, i, u[j] - u[i])
        np.add.at(lu, j, u[i] - u[j])
        np.add.at(lv, i, v[j] - v[i])
        np.add.at(lv, j, v[i] - v[j])
        # normalise by degree so the operator is a mean-difference Laplacian: without this,
        # dense interior cells feel a much larger L than sparse surface cells and the
        # pattern tracks PACKING DENSITY rather than chemistry (measured artifact)
        dsafe = np.maximum(deg, 1.0)
        lu /= dsafe
        lv /= dsafe
        uvv = u * v * v
        u = np.clip(u + dt * (du * lu - uvv + feed * (1.0 - u)), 0.0, 1.5)
        v = np.clip(v + dt * (dv * lv + uvv - (feed + kill) * v), 0.0, 1.5)
    return u, v


def positional_information(positions, axis=0, source="min"):
    """Wolpert positional information: a monotone morphogen gradient along an axis, produced
    by a source at one end. THIS IS THE PRESCRIBED HALF -- it does not self-organise, and
    saying so is the point (Turing's standing question for this workstream). Normalised to
    [0,1] so it composes with the RD field regardless of aggregate size."""
    p = np.asarray(positions, float)[:, int(axis)]
    lo, hi = float(p.min()), float(p.max())
    g = (p - lo) / (hi - lo + 1e-12)
    return g if source == "min" else 1.0 - g


def adhesion_energy_and_grad(positions, radii, morphogen, k_adh=0.6, width=0.25,
                             cutoff_scale=1.5):
    """Differential-adhesion energy and its EXACT analytic gradient.

    Per the source document's f(|m_i - m_j|) * g(|x_i - x_j|) form: cells with SIMILAR
    morphogen values adhere; dissimilar ones do not. f is a Gaussian in morphogen distance
    (smooth, so the gradient exists everywhere); g reuses the same C^1 contact well as the
    base potential, so adhesion strengthens an existing well rather than introducing a second
    length scale. Only positions are differentiated -- morphogen values are held fixed within
    a relaxation (chemistry is slow relative to mechanics; the standard quasi-static split)."""
    from holographic.misc.holographic_fields import spatial_hash_pairs
    positions = np.asarray(positions, float)
    radii = np.asarray(radii, float)
    mg = np.asarray(morphogen, float)
    grad = np.zeros_like(positions)
    n = len(positions)
    if n < 2:
        return 0.0, grad
    r_cut_max = cutoff_scale * 2.0 * float(radii.max())
    pairs = np.asarray(spatial_hash_pairs(positions, r_cut_max), int).reshape(-1, 2)
    if len(pairs) == 0:
        return 0.0, grad
    i, j = pairs[:, 0], pairs[:, 1]
    dvec = positions[i] - positions[j]
    d = np.maximum(np.linalg.norm(dvec, axis=1), 1e-12)
    r0 = radii[i] + radii[j]
    r_cut = cutoff_scale * r0
    f = np.exp(-((mg[i] - mg[j]) ** 2) / (2.0 * width * width))     # similarity factor
    inside = d < r_cut
    u = np.zeros_like(d)
    u[inside] = np.clip((d[inside] - r0[inside]) / (r_cut[inside] - r0[inside]), 0.0, 1.0)
    w = np.zeros_like(d)
    w[inside] = (1.0 - u[inside] ** 2) ** 2
    e = -k_adh * f * w
    dwdd = np.zeros_like(d)
    dwdd[inside] = (2.0 * (1.0 - u[inside] ** 2) * (-2.0 * u[inside])
                    / (r_cut[inside] - r0[inside]))
    dphi = -k_adh * f * dwdd
    contrib = (dphi / d)[:, None] * dvec
    np.add.at(grad, i, contrib)
    np.add.at(grad, j, -contrib)
    return float(e.sum()), grad


def differentiate(positions, radii, steps=300, step0=0.03, k_rep=1.0, k_att=0.35,
                  k_adh=0.6, width=0.25, rd_steps=400, rd_weight=1.0, pi_weight=1.0,
                  pi_axis=0, seed=0, cutoff_scale=1.5):
    """F2: run morphogens on the cell graph, then relax under base + differential adhesion.

    MODE 2 (the SOTA composition, see the section header): the morphogen is
        m = rd_weight * v_turing + pi_weight * g_positional,
    i.e. a self-organising RD pattern MODULATED BY a positional gradient -- which is what the
    limb bud is currently understood to do. Set rd_weight=0 for pure Wolpert (prescribed) or
    pi_weight=0 for pure Turing (emergent); the ablation is the experiment, and the selftest
    runs it.

    Returns {"positions","morphogen","u","v","energy","sphericity","lobes","history"}.
    Deterministic per seed."""
    positions = np.asarray(positions, float)
    radii = np.asarray(radii, float)
    u, v = reaction_diffusion_cells(positions, radii, steps=rd_steps, seed=seed,
                                    cutoff_scale=cutoff_scale)
    g = positional_information(positions, axis=pi_axis)
    mg = rd_weight * v + pi_weight * g
    if mg.max() > mg.min():
        mg = (mg - mg.min()) / (mg.max() - mg.min())

    def total(x):
        e1, g1 = pair_energy_and_grad(x, radii, k_rep, k_att, cutoff_scale)
        e2, g2 = adhesion_energy_and_grad(x, radii, mg, k_adh, width, cutoff_scale)
        return e1 + e2, g1 + g2

    x = np.array(positions, copy=True)
    e, gr = total(x)
    hist = [e]
    step = step0
    for _ in range(int(steps)):
        if float(np.linalg.norm(gr)) < 1e-9:
            break
        trial = step
        for _ in range(24):
            y = x - trial * gr
            e2, g2 = total(y)
            if e2 <= e:
                break
            trial *= 0.5
        else:
            break
        x, e, gr = y, e2, g2
        hist.append(e)
        step = min(trial * 1.6, step0 * 8.0)
    return {"positions": x, "morphogen": mg, "u": u, "v": v, "energy": e,
            "sphericity": sphericity(x), "lobes": count_lobes(x, radii, mg),
            "history": hist}


def count_lobes(positions, radii, morphogen, threshold=0.5, cutoff_scale=1.5):
    """How many DISCONNECTED high-morphogen regions are there? -- the emergence meter, and a
    proxy for "how many limb buds". Delegates the flood fill to the engine's existing
    graph_connected_components (Rule 0: lobe counting is not a new algorithm)."""
    from holographic.simulation_and_physics.holographic_island import connected_components
    pairs, _ = cell_graph(positions, radii, cutoff_scale)
    mg = np.asarray(morphogen, float)
    hot = mg >= threshold
    idx = {int(k): n for n, k in enumerate(np.nonzero(hot)[0])}
    edges = [(idx[int(a)], idx[int(b)]) for a, b in pairs
             if bool(hot[int(a)]) and bool(hot[int(b)])]
    if not idx:
        return 0
    comps = connected_components(len(idx), edges)
    return len([c for c in comps if len(c) >= 2])


# ---------------------------------------------------------------------------
# F6: HYPERVECTOR GENOMES -- the body plan's parameters as one searchable vector.
#
# SOTA CHECK (searched 2026-08-16): the evolutionary-robotics literature splits encodings
# into DIRECT (each phenotype component coded independently) and INDIRECT/GENERATIVE (CPPN,
# L-system, and lately VAE/GAN latent spaces used as genotype->phenotype maps). The field's
# stated quality criterion for ANY encoding is LOCALITY: small genotype changes must produce
# small phenotype changes, because without it good parents produce bad offspring and search
# stalls in local optima (Gottlieb & Raidl 1999; Rothlauf & Goldberg 1999). Latent-space
# encodings are the modern route but require LEARNED WEIGHTS -- forbidden here.
#
# WHAT WE DO, positioned honestly: this is a DIRECT encoding lifted into the substrate, not a
# generative one. The claim is not "a better genotype"; it is "the same parameters, now a
# VECTOR" -- so genomes are comparable by cosine, interpolable, and storable in the same
# indexed rows as everything else. LOCALITY IS NOT HOPED FOR, it comes from the encoder:
# fractional power encoding maps nearby scalars to similar vectors BY CONSTRUCTION, and the
# selftest MEASURES the locality curve rather than asserting it.
#
# RULE-0 AUDIT (2026-08-16): encode_record/decode_record already ship but are CATEGORICAL
# (field -> value NAME) and raise on floats -- a genuine partial mismatch, recorded so this
# is not re-audited. holographic_fpe.ScalarEncoder is the continuous counterpart and is
# REUSED here; no new encoder is written. holographic_evolve also ships (search over
# genomes) and is deliberately untouched -- F6 supplies the representation, not the search.
# ---------------------------------------------------------------------------

GENOME_FIELDS = ("k_rep", "k_att", "k_adh", "width", "rd_weight", "pi_weight")
GENOME_RANGES = {"k_rep": (0.1, 4.0), "k_att": (0.0, 1.5), "k_adh": (0.0, 2.0),
                 "width": (0.05, 1.0), "rd_weight": (0.0, 2.0), "pi_weight": (0.0, 2.0)}


def _genome_encoders(dim, seed):
    from holographic.sampling_and_signal.holographic_fpe import ScalarEncoder
    return {f: ScalarEncoder(dim, lo=GENOME_RANGES[f][0], hi=GENOME_RANGES[f][1],
                             seed=seed + i) for i, f in enumerate(GENOME_FIELDS)}


def genome_encode(params, dim=1024, seed=0):
    """Encode a body-plan genome as ONE vector: sum over fields of bind(role, FPE(value)).

    Roles come from the engine's derived_atom (hashlib-seeded, deterministic); values from
    ScalarEncoder so that NEARBY PARAMETERS GIVE NEARBY VECTORS -- the locality property the
    encoding literature calls decisive, obtained from the encoder rather than asserted."""
    from holographic.agents_and_reasoning.holographic_ai import derived_atom, bind
    encs = _genome_encoders(dim, seed)
    parts = []
    for f in GENOME_FIELDS:
        lo, hi = GENOME_RANGES[f]
        v = float(np.clip(params.get(f, lo), lo, hi))
        parts.append(bind(derived_atom(seed, "gene:" + f, dim), encs[f].encode(v)))
    return np.sum(parts, axis=0)


def genome_decode(vec, dim=1024, seed=0, samples=64, floor=0.15):
    """Recover parameters from a genome vector: unbind each role, then read the scalar back
    by scanning that field's encoder over `samples` values and taking the best match.

    ABSTAINS per field (value None) when the best correlation is below `floor` -- the same
    honesty contract decode_atom uses. Returns {"params", "scores", "abstained"}."""
    from holographic.agents_and_reasoning.holographic_ai import derived_atom, unbind
    encs = _genome_encoders(dim, seed)
    out, scores, abstained = {}, {}, []
    v = np.asarray(vec, float)
    for f in GENOME_FIELDS:
        payload = unbind(v, derived_atom(seed, "gene:" + f, dim))
        lo, hi = GENOME_RANGES[f]
        grid = np.linspace(lo, hi, int(samples))
        mat = np.stack([encs[f].encode(g) for g in grid])
        sims = mat @ payload / (np.linalg.norm(mat, axis=1) * np.linalg.norm(payload) + 1e-12)
        j = int(np.argmax(sims))
        scores[f] = float(sims[j])
        if sims[j] < floor:
            out[f] = None
            abstained.append(f)
        else:
            out[f] = float(grid[j])
    return {"params": out, "scores": scores, "abstained": abstained}


def genome_locality(dim=1024, seed=0, deltas=(0.01, 0.05, 0.1, 0.25, 0.5), trials=8):
    """MEASURE the locality curve: perturb a genome by a relative delta and report the mean
    cosine between the original and perturbed vectors, with spread.

    This is the encoding literature's decisive criterion made into a number for OUR encoding.
    A good encoding's curve falls smoothly and monotonically; a cliff would mean small genome
    edits produce unrelated bodies. Deterministic: dedicated rng per trial."""
    base_rng = np.random.default_rng(int(seed) + 991)
    rows = {}
    for d in deltas:
        cs = []
        for t in range(int(trials)):
            rng = np.random.default_rng(int(seed) * 131 + t)
            p = {f: rng.uniform(*GENOME_RANGES[f]) for f in GENOME_FIELDS}
            q = {}
            for f in GENOME_FIELDS:
                lo, hi = GENOME_RANGES[f]
                q[f] = float(np.clip(p[f] + d * (hi - lo) * base_rng.choice([-1.0, 1.0]),
                                     lo, hi))
            a = genome_encode(p, dim, seed)
            b = genome_encode(q, dim, seed)
            cs.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
        rows[float(d)] = {"mean": float(np.mean(cs)), "std": float(np.std(cs))}
    return rows


def genome_interpolate(pa, pb, t):
    """Interpolate two genomes in PARAMETER space (not vector space) -- and the docstring
    says which, because it matters. Interpolating the VECTORS would produce a superposition
    that decodes to one endpoint or the other, not to a blend; the parameters are what the
    body is actually grown from. Vector space is for SEARCH and COMPARISON here, not for
    breeding."""
    t = float(np.clip(t, 0.0, 1.0))
    return {f: (1.0 - t) * float(pa[f]) + t * float(pb[f]) for f in GENOME_FIELDS}


# ---------------------------------------------------------------------------
# F7: SHAPE MEMORY -- stored target morphologies as attractors, with the strawman killed.
#
# SOTA CHECK (searched 2026-08-16): the reference framing is Levin's ANATOMICAL HOMEOSTASIS
# (planarian regeneration; morphostasis), in which development and regeneration are both
# error-minimisation toward a stored TARGET MORPHOLOGY setpoint -- the same variational
# picture the source document uses -- and the recent literature links it explicitly to
# HOPFIELD associative memory (target morphologies as attractors).
#
# THE PRE-REGISTERED STRAWMAN, and why this item exists: "perturb the body, watch it come
# back" proves NOTHING. Any energy well pulls a state back, so recovery could be entirely a
# function of WELL DEPTH with the stored shape doing no work at all. The honest experiment
# therefore asks the associative-memory question instead: with SEVERAL shapes stored, does
# a partial/perturbed body recover the RIGHT one? A depth-matched control with a SCRAMBLED
# target must fail where the real memory succeeds -- otherwise there is no memory here, only
# a spring, and this module says so.
#
# RULE-0 AUDIT (2026-08-16): holographic_hopfield.dense_cleanup already ships (modern
# softmax Hopfield / dense associative memory) and is REUSED as the retrieval step -- no new
# attractor machinery is written. `regeneration` returned nothing; the morphology-level
# wrapper is the genuine gap. attractor_force was audited and NOT used: it is a force field
# for agents, not a pattern memory.
# ---------------------------------------------------------------------------

def shape_descriptor(positions, bins=8):
    """A rotation-free, size-normalised shape signature: the radial mass profile.

    Distances from the centroid, normalised by the RMS radius, histogrammed into `bins`.
    Chosen because it survives translation and scale, is cheap, and -- the point for a
    MEMORY -- two different body plans give different vectors while noisy versions of one
    body plan give nearby vectors. Deterministic."""
    p = np.asarray(positions, float)
    c = p - p.mean(axis=0)
    d = np.linalg.norm(c, axis=1)
    rms = float(np.sqrt(np.mean(d ** 2))) + 1e-12
    h, _ = np.histogram(d / rms, bins=int(bins), range=(0.0, 2.5))
    v = h.astype(float)
    return v / (np.linalg.norm(v) + 1e-12)


def shape_memory_store(shapes, bins=8):
    """Build a shape memory: a codebook of descriptors, one row per stored morphology."""
    return np.stack([shape_descriptor(s, bins) for s in shapes])


def shape_memory_recall(positions, codebook, beta=25.0, steps=3, bins=8):
    """Retrieve the stored morphology this body most resembles, via the engine's OWN dense
    (modern Hopfield) cleanup -- Rule 0: the associative memory already ships.

    Returns {"index", "confidence", "retrieved"}: which stored shape, how strongly, and the
    cleaned descriptor. Confidence is the cosine to the winning row, so a body resembling
    nothing stored reports a LOW number rather than a confident wrong answer."""
    from holographic.agents_and_reasoning.holographic_hopfield import dense_cleanup
    q = shape_descriptor(positions, bins)
    out = dense_cleanup(q, np.asarray(codebook, float), beta=beta, steps=steps)
    cb = np.asarray(codebook, float)
    sims = cb @ out / (np.linalg.norm(cb, axis=1) * np.linalg.norm(out) + 1e-12)
    j = int(np.argmax(sims))
    return {"index": j, "confidence": float(sims[j]), "retrieved": out}


def shape_memory_probe(n_shapes=3, n_cells=45, noise=0.35, trials=6, seed=0, bins=8):
    """THE EXPERIMENT THAT KILLS THE STRAWMAN. Grow `n_shapes` distinct bodies, store them,
    then perturb each with Gaussian noise and ask the memory which one it is.

    Reports accuracy against a SCRAMBLED-CODEBOOK CONTROL: the same retrieval machinery
    against shuffled descriptor rows, which preserves the well DEPTH (row norms, softmax
    temperature, everything) while destroying the correspondence between body and target.
    If real and control accuracy match, there is no shape memory -- only a spring -- and
    that result is the deliverable either way. Deterministic; dedicated rng per trial."""
    # MEASURED NEGATIVE that shaped this function: varying only the GROWTH parameters
    # (k_rep/k_att) does NOT produce distinct shapes -- F1 is designed to make compact balls,
    # so three such bodies had descriptor cosines of 0.99+ and recall sat exactly at chance
    # (0.33 = control = chance). The memory was not broken; there was NOTHING TO REMEMBER.
    # Distinct morphologies require F2's DIFFERENTIATION (adhesion on/off, Turing vs
    # positional morphogen), which measured cosines of 0.07-0.31 -- genuinely different
    # bodies. Discriminability is a property of the GENERATOR, not of the memory.
    base = grow_aggregate(n_cells=int(n_cells), seed=int(seed), steps=80)
    bp, br = base["positions"], base["radii"]
    configs = [(0.0, 1.0, 1.0), (1.6, 1.0, 0.0), (1.6, 0.0, 1.0),
               (0.8, 1.0, 1.0), (2.2, 0.5, 1.5)]
    shapes, params = [], []
    for i in range(int(n_shapes)):
        ka, rw, pw = configs[i % len(configs)]
        d = differentiate(bp, br, steps=200, k_adh=ka, rd_weight=rw, pi_weight=pw,
                          seed=int(seed) + 1)
        shapes.append(d["positions"])
        params.append((ka, rw, pw))
    cb = shape_memory_store(shapes, bins)
    rng_scr = np.random.default_rng(int(seed) + 7777)
    scrambled = cb[rng_scr.permutation(len(cb))]
    hits = ctrl_hits = total = 0
    for i, s in enumerate(shapes):
        for t in range(int(trials)):
            rng = np.random.default_rng(int(seed) * 1009 + i * 31 + t)
            noisy = np.asarray(s, float) + rng.normal(scale=float(noise), size=np.shape(s))
            hits += int(shape_memory_recall(noisy, cb, bins=bins)["index"] == i)
            ctrl_hits += int(shape_memory_recall(noisy, scrambled, bins=bins)["index"] == i)
            total += 1
    return {"n_shapes": int(n_shapes), "trials": total,
            "accuracy": hits / max(total, 1), "control_accuracy": ctrl_hits / max(total, 1),
            "chance": 1.0 / max(int(n_shapes), 1), "params": params}


def _selftest():
    """Regression trap. The load-bearing assertion is the ANALYTIC GRADIENT vs the engine's
    own fd_gradient -- if that fails, every downstream morphogenesis result is fiction."""
    from holographic.misc.holographic_optimize import fd_gradient
    rng = np.random.default_rng(20260816)          # this test's truths own this seed
    pos = rng.normal(scale=1.2, size=(24, 3))
    rad = np.full(24, 0.5)

    # 1) ANALYTIC == FINITE DIFFERENCE (the planted truth; house instrument, not a new one)
    f = lambda flat: pair_energy_and_grad(flat.reshape(-1, 3), rad)[0]
    num = fd_gradient(f, pos.ravel().copy(), eps=1e-6).reshape(-1, 3)
    _, ana = pair_energy_and_grad(pos, rad)
    err = np.abs(num - ana).max()
    assert err < 1e-5, "analytic gradient disagrees with fd_gradient by %.2e" % err

    # 2) C^1 at contact, measured the RIGHT way. A second-difference threshold conflates
    #    genuine CURVATURE with a discontinuity (it flagged the correct inverse core simply
    #    because that core is more curved) -- so instead compare VALUE and GRADIENT just
    #    below and just above r0. Both must match; those are exactly C^0 and C^1.
    rr = np.array([0.5, 0.5])
    eps = 1e-6
    lo = pair_energy_and_grad(np.array([[0.0, 0, 0], [1.0 - eps, 0, 0]]), rr)
    hi = pair_energy_and_grad(np.array([[0.0, 0, 0], [1.0 + eps, 0, 0]]), rr)
    assert abs(lo[0] - hi[0]) < 1e-8, "energy JUMPS at contact: %.3e" % abs(lo[0] - hi[0])
    assert np.abs(lo[1] - hi[1]).max() < 1e-4, (
        "gradient jumps at contact: %.3e" % np.abs(lo[1] - hi[1]).max())
    # and the well vanishes smoothly at the cutoff rather than stepping off a cliff
    assert abs(pair_energy_and_grad(np.array([[0.0, 0, 0], [1.6, 0, 0]]), rr)[0]) < 1e-12

    # 3) DESCENT: relax must lower energy monotonically (backtracking guarantees it)
    _, hist = relax(pos, rad, steps=60)
    assert all(b <= a + 1e-12 for a, b in zip(hist, hist[1:])), "energy increased during relax"

    # 4) TURING'S GATE, with the strawman killed: the control is proliferation WITHOUT
    #    relaxation (division jitter alone), so the contrast measures the energy dynamics
    #    and not the initial condition. Also pinned: a perfectly planar slab is a critical
    #    point that relaxation alone cannot escape (the kept negative, asserted so nobody
    #    "fixes" it into a bug report later).
    out = grow_aggregate(n_cells=64, seed=0, steps=200)
    ctrl = grow_aggregate(n_cells=64, seed=0, steps=0)        # proliferation only, no relax
    assert out["sphericity"] > 0.6, "aggregate did not become ball-like: %.3f" % out["sphericity"]
    # THE GATE SPHERICITY MISSED: a collapsed blob is perfectly spherical, so packing is
    # asserted separately. Ideal is 1.0 (neighbours a diameter apart); the pre-fix quadratic
    # core measured 0.15 here while sailing through the sphericity assertion.
    assert out["packing"] > 0.85, "aggregate COLLAPSED: packing %.3f (ideal 1.0)" % out["packing"]
    assert out["packing"] < 1.6, "aggregate EXPLODED: packing %.3f" % out["packing"]
    assert out["sphericity"] > ctrl["sphericity"] + 0.25, (
        "sphericity %.3f barely beats the proliferation-only control %.3f -- the shape would "
        "be coming from division jitter, not the energy"
        % (out["sphericity"], ctrl["sphericity"]))
    slab = np.stack([np.repeat(np.arange(4), 4) * 0.9, np.tile(np.arange(4), 4) * 0.9,
                     np.zeros(16)], axis=1).astype(float)
    planar, _ = relax(slab, np.full(16, 0.5), steps=300, core="quadratic")
    assert sphericity(planar) < 1e-6, "planar critical point escaped -- investigate, do not celebrate"

    # 5) F2: differential adhesion BREAKS the spherical symmetry, and the no-adhesion
    #    CONTROL proves the adhesion is what did it (Turing's standing gate). The morphogen
    #    is Mode 2 -- an emergent RD pattern modulated by a prescribed PI gradient, which is
    #    what the limb-bud literature currently describes.
    base = grow_aggregate(n_cells=100, seed=1, steps=120)
    Pb, Rb = base["positions"], base["radii"]
    ctl2 = differentiate(Pb, Rb, steps=200, k_adh=0.0, seed=1)     # no adhesion at all
    mode2 = differentiate(Pb, Rb, steps=200, k_adh=0.8, seed=1)    # RD + PI
    assert ctl2["sphericity"] > mode2["sphericity"] + 0.2, (
        "adhesion did not break symmetry: control %.3f vs adhesion %.3f"
        % (ctl2["sphericity"], mode2["sphericity"]))
    assert mode2["history"][-1] <= mode2["history"][0], "adhesion relax did not descend"
    u, v = reaction_diffusion_cells(Pb, Rb, steps=300, seed=1)
    assert v.max() - v.min() > 0.05, "reaction-diffusion produced no pattern at all"
    assert np.all(np.isfinite(v)) and np.all(np.isfinite(u))
    # adhesion gradient is analytic too -- same instrument, same standard
    from holographic.misc.holographic_optimize import fd_gradient as _fd
    small = np.random.default_rng(11).normal(size=(12, 3))
    rr2 = np.full(12, 0.5)
    mg2 = np.linspace(0.0, 1.0, 12)
    fa = lambda flat: adhesion_energy_and_grad(flat.reshape(-1, 3), rr2, mg2)[0]
    na = _fd(fa, small.ravel().copy(), eps=1e-6).reshape(-1, 3)
    _, aa = adhesion_energy_and_grad(small, rr2, mg2)
    assert np.abs(na - aa).max() < 1e-5, "adhesion gradient wrong: %.2e" % np.abs(na - aa).max()

    # 6) F6 GENOMES: round-trip recovers every field, and the LOCALITY CURVE -- the
    #    encoding literature's decisive criterion -- is measured, smooth and MONOTONE.
    #    A cliff here would mean small genome edits produce unrelated bodies.
    gp = {"k_rep": 1.0, "k_att": 0.35, "k_adh": 0.8, "width": 0.25,
          "rd_weight": 1.0, "pi_weight": 1.0}
    gd = genome_decode(genome_encode(gp))
    assert not gd["abstained"], "genome fields abstained: %r" % gd["abstained"]
    for f, want in gp.items():
        got = gd["params"][f]
        lo, hi = GENOME_RANGES[f]
        assert abs(got - want) < 0.2 * (hi - lo), "%s: %.3f vs %.3f" % (f, got, want)
    loc = genome_locality(deltas=(0.05, 0.25, 0.5), trials=4)
    ms = [loc[d]["mean"] for d in (0.05, 0.25, 0.5)]
    assert ms == sorted(ms, reverse=True), "locality is not monotone: %r" % ms
    assert ms[0] > 0.95 and ms[-1] < ms[0], "locality curve is flat or inverted: %r" % ms
    # noise must ABSTAIN rather than confabulate a genome
    assert genome_decode(np.random.default_rng(5).standard_normal(1024))["abstained"], \
        "a random vector decoded as a valid genome"

    # 7) F7 SHAPE MEMORY, with the PRE-REGISTERED STRAWMAN killed: recovery must depend on
    #    the STORED PATTERN, not merely on a well existing. The depth-matched scrambled
    #    control uses identical machinery with the body<->target correspondence destroyed,
    #    and must FAIL where the real memory succeeds.
    probe = shape_memory_probe(n_shapes=3, noise=0.1, trials=3, seed=0)
    assert probe["accuracy"] > 0.8, "shape recall failed: %.2f" % probe["accuracy"]
    assert probe["accuracy"] > probe["control_accuracy"] + 0.5, (
        "recall %.2f is not meaningfully above the scrambled-target control %.2f -- there "
        "is no shape MEMORY here, only a spring"
        % (probe["accuracy"], probe["control_accuracy"]))

    # 8) DETERMINISM: same seed, identical aggregate to the bit
    a = grow_aggregate(n_cells=48, seed=7, steps=80)
    b = grow_aggregate(n_cells=48, seed=7, steps=80)
    assert np.array_equal(a["positions"], b["positions"])
    print("OK: holographic_morphogen -- analytic grad matches fd to %.1e, C1 at contact, "
          "monotone descent, sphericity %.3f vs proliferation-only control %.3f, "
          "planar critical point pinned, deterministic"
          % (err, out["sphericity"], ctrl["sphericity"]))


if __name__ == "__main__":
    _selftest()
