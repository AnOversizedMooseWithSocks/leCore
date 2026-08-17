"""Regression traps for F1 (cell-aggregate morphogenesis with analytic gradients).

The load-bearing pin is the analytic gradient against the engine's OWN fd_gradient: if that
drifts, every downstream morphogenesis result (F2 differentiation, F3 tetrahedralisation,
F5 LOD) is fiction built on a wrong force.
"""
import numpy as np
import pytest

from holographic.simulation_and_physics import holographic_morphogen as M


def test_analytic_gradient_matches_finite_difference():
    """No autodiff (house constraint) does not mean no verification: the closed-form pair
    gradient is checked against holographic_optimize.fd_gradient -- the instrument already
    existed, we only pointed it at the new energy."""
    from holographic.misc.holographic_optimize import fd_gradient
    rng = np.random.default_rng(20260816)
    pos = rng.normal(scale=1.2, size=(24, 3))
    rad = np.full(24, 0.5)
    f = lambda flat: M.pair_energy_and_grad(flat.reshape(-1, 3), rad)[0]
    num = fd_gradient(f, pos.ravel().copy(), eps=1e-6).reshape(-1, 3)
    _, ana = M.pair_energy_and_grad(pos, rad)
    assert np.abs(num - ana).max() < 1e-5
    # varied radii too -- equal radii could hide an r0 indexing bug
    rad2 = rng.uniform(0.3, 0.8, size=24)
    f2 = lambda flat: M.pair_energy_and_grad(flat.reshape(-1, 3), rad2)[0]
    num2 = fd_gradient(f2, pos.ravel().copy(), eps=1e-6).reshape(-1, 3)
    _, ana2 = M.pair_energy_and_grad(pos, rad2)
    assert np.abs(num2 - ana2).max() < 1e-5


def test_energy_is_c1_at_contact():
    """The core/well branches must agree in value AND slope at d=r0. They did not in the
    first draft (a k_att-sized JUMP), which silently corrupts relax()'s backtracking line
    search -- it compares energies across configurations where pairs cross the boundary."""
    # measured the RIGHT way: a second-difference threshold conflates genuine CURVATURE with
    # a discontinuity (it flagged the CORRECT divergent core purely for being more curved).
    # C^0 and C^1 are exactly "value and gradient agree across the junction".
    rr = np.array([0.5, 0.5])
    eps = 1e-6
    lo = M.pair_energy_and_grad(np.array([[0.0, 0, 0], [1.0 - eps, 0, 0]]), rr)
    hi = M.pair_energy_and_grad(np.array([[0.0, 0, 0], [1.0 + eps, 0, 0]]), rr)
    assert abs(lo[0] - hi[0]) < 1e-9        # measured 1.8e-12
    assert np.abs(lo[1] - hi[1]).max() < 1e-4   # measured 7.6e-6
    assert abs(M.pair_energy_and_grad(np.array([[0.0, 0, 0], [1.6, 0, 0]]), rr)[0]) < 1e-12


def test_relax_descends_monotonically():
    rng = np.random.default_rng(5)
    pos = rng.normal(scale=1.5, size=(40, 3))
    _, hist = M.relax(pos, np.full(40, 0.5), steps=80)
    assert all(b <= a + 1e-12 for a, b in zip(hist, hist[1:]))
    assert hist[-1] < hist[0]


def test_ball_comes_from_dynamics_not_from_jitter():
    """Turing's standing gate for this workstream, with the strawman pre-registered and
    killed: the control is proliferation WITHOUT relaxation, so the contrast isolates the
    energy. Measured 1.000 vs 0.008."""
    out = M.grow_aggregate(n_cells=64, seed=0, steps=200)
    ctrl = M.grow_aggregate(n_cells=64, seed=0, steps=0)
    assert out["sphericity"] > 0.55
    assert out["sphericity"] > ctrl["sphericity"] + 0.3


def test_planar_critical_point_is_a_kept_negative():
    """A perfectly symmetric configuration is a critical point: every z-gradient is zero, so
    descent packs IN-PLANE forever and never thickens. Asserted so a future session reads it
    as physics, not as a bug -- symmetry must be broken by proliferation or explicit noise."""
    slab = np.stack([np.repeat(np.arange(4), 4) * 0.9, np.tile(np.arange(4), 4) * 0.9,
                     np.zeros(16)], axis=1).astype(float)
    out, hist = M.relax(slab, np.full(16, 0.5), steps=300)
    assert M.sphericity(out) < 1e-6      # never leaves the plane
    assert hist[-1] < hist[0]            # while still minimising in-plane


def test_faculties_and_determinism():
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    a = m.morphogenesis_grow(n_cells=32, seed=7, steps=60)
    b = m.morphogenesis_grow(n_cells=32, seed=7, steps=60)
    assert np.array_equal(a["positions"], b["positions"])
    c = m.morphogenesis_grow(n_cells=32, seed=8, steps=60)
    assert not np.array_equal(a["positions"], c["positions"])   # seeds actually matter
    r = m.morphogenesis_relax(a["positions"], a["radii"], steps=20)
    assert r["energy"] <= r["history"][0] + 1e-12


def test_volume_exclusion_and_the_collapse_negative():
    """THE BUG F1's SPHERICITY TEST COULD NOT SEE: a quadratic core is finite at d=0, so it
    cannot exclude volume -- each of ~N neighbours pulls inward while the core resists only
    linearly, and the aggregate COLLAPSES (measured pre-fix: 200 cells inside 1.3 diameters,
    mean neighbour distance 0.15 of ideal, mean degree 199/200). A collapsed blob is
    perfectly spherical, so sphericity passed the whole time. Packing is now its own gate."""
    rng = np.random.default_rng(3)
    P0 = rng.normal(scale=2.0, size=(120, 3))
    R0 = np.full(120, 0.5)
    hard, _ = M.relax(P0, R0, steps=400, core="inverse")
    soft, _ = M.relax(P0, R0, steps=400, core="quadratic")
    assert M.packing_quality(hard) > M.packing_quality(soft) + 0.15   # measured .931 vs .757
    assert M.packing_quality(hard) > 0.85                             # 2r = 1.0 here


def test_anneal_gets_both_properties():
    """Soft-then-inflate (the packing literature's standard schedule) because NEITHER endpoint
    works alone: soft-only is round but collapsed (1.000 / 0.202), one hard relax jams
    (0.926 / 0.505); the ladder delivers both (measured 0.82 / 0.958)."""
    out = M.grow_aggregate(n_cells=64, seed=0, steps=150)
    assert out["sphericity"] > 0.6 and out["packing"] > 0.85 and out["packing"] < 1.6


def test_differential_adhesion_breaks_symmetry_with_a_control():
    """F2, Turing's standing gate answered with a control rather than an assertion: with
    adhesion OFF the aggregate stays round; with it ON the symmetry breaks."""
    base = M.grow_aggregate(n_cells=100, seed=1, steps=120)
    P, R = base["positions"], base["radii"]
    ctl = M.differentiate(P, R, steps=200, k_adh=0.0, seed=1)
    both = M.differentiate(P, R, steps=200, k_adh=0.8, seed=1)
    assert ctl["sphericity"] > both["sphericity"] + 0.2
    assert both["history"][-1] <= both["history"][0]


def test_adhesion_gradient_is_analytic():
    from holographic.misc.holographic_optimize import fd_gradient
    rng = np.random.default_rng(11)
    x = rng.normal(size=(12, 3))
    r = np.full(12, 0.5)
    mg = np.linspace(0, 1, 12)
    f = lambda flat: M.adhesion_energy_and_grad(flat.reshape(-1, 3), r, mg)[0]
    num = fd_gradient(f, x.ravel().copy(), eps=1e-6).reshape(-1, 3)
    _, ana = M.adhesion_energy_and_grad(x, r, mg)
    assert np.abs(num - ana).max() < 1e-5


def test_rd_pattern_forms_and_single_lobe_negative():
    """RD produces a pattern (not a dead uniform state), and the KEPT NEGATIVE is pinned:
    at these cell counts it is ONE front, not multiple spots -- Turing patterns need a domain
    several wavelengths across, and ~200 cells in 3D is only a few cells across."""
    base = M.grow_aggregate(n_cells=100, seed=1, steps=100)
    u, v = M.reaction_diffusion_cells(base["positions"], base["radii"], steps=300, seed=1)
    assert v.max() - v.min() > 0.05
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(v))
    mg = (v - v.min()) / (v.max() - v.min() + 1e-12)
    assert M.count_lobes(base["positions"], base["radii"], mg) <= 2


def test_genome_roundtrip_and_abstention():
    """F6: a DIRECT encoding lifted into the substrate. Every field recovers; a random
    vector must ABSTAIN rather than decode as a valid genome."""
    gp = {"k_rep": 1.0, "k_att": 0.35, "k_adh": 0.8, "width": 0.25,
          "rd_weight": 1.0, "pi_weight": 1.0}
    d = M.genome_decode(M.genome_encode(gp))
    assert not d["abstained"]
    for f, want in gp.items():
        lo, hi = M.GENOME_RANGES[f]
        assert abs(d["params"][f] - want) < 0.2 * (hi - lo)
    assert M.genome_decode(np.random.default_rng(5).standard_normal(1024))["abstained"]


def test_genome_locality_curve():
    """The encoding literature's decisive criterion, measured not asserted: small genotype
    changes must give small phenotype-vector changes, smoothly and monotonically. A cliff
    would mean good parents produce unrelated offspring."""
    loc = M.genome_locality(deltas=(0.05, 0.25, 0.5), trials=4)
    ms = [loc[d]["mean"] for d in (0.05, 0.25, 0.5)]
    assert ms == sorted(ms, reverse=True)
    assert ms[0] > 0.95 and ms[-1] < ms[0]


def test_interpolated_genomes_stay_viable():
    """Backlog gate: interpolants must still GROW valid bodies. Measured 5/5 clean; this
    pins the endpoints and the midpoint to keep the runtime sane."""
    from holographic.mesh_and_geometry.holographic_tetmesh import tetrahedralize
    a = {"k_rep": 1.0, "k_att": 0.35, "k_adh": 0.8, "width": 0.25,
         "rd_weight": 1.0, "pi_weight": 1.0}
    b = {"k_rep": 2.5, "k_att": 0.8, "k_adh": 1.5, "width": 0.6,
         "rd_weight": 0.0, "pi_weight": 2.0}
    for t in (0.0, 0.5, 1.0):
        g = M.genome_interpolate(a, b, t)
        agg = M.grow_aggregate(n_cells=40, seed=1, steps=60,
                               k_rep=g["k_rep"], k_att=g["k_att"])
        mesh = tetrahedralize(agg["positions"], agg["radii"])
        assert mesh["T"] > 0 and mesh["components"] == 1 and not mesh["nonmanifold_faces"]


def test_shape_memory_beats_a_depth_matched_control():
    """F7 with its PRE-REGISTERED STRAWMAN killed. "Perturb it and watch it come back"
    proves nothing -- any well does that. The real question is associative: with several
    shapes stored, is the RIGHT one recovered? The scrambled-codebook control keeps every
    depth/temperature parameter and destroys only the body<->target correspondence, so it
    must FAIL where real memory succeeds. Measured 1.00 vs 0.00 at noise 0.1."""
    probe = M.shape_memory_probe(n_shapes=3, noise=0.1, trials=3, seed=0)
    assert probe["accuracy"] > 0.8
    assert probe["accuracy"] > probe["control_accuracy"] + 0.5


def test_discriminability_is_a_property_of_the_generator():
    """KEPT NEGATIVE, pinned so it is read as a finding rather than rediscovered as a bug:
    bodies that differ only in GROWTH parameters are the same shape (F1 makes compact
    balls), with descriptor cosines above 0.95. The memory had nothing to remember and sat
    exactly at chance. Distinct morphologies require F2 differentiation."""
    shapes = [M.grow_aggregate(n_cells=40, seed=i, steps=60,
                               k_rep=0.6 + 0.8 * i, k_att=0.2 + 0.25 * i)["positions"]
              for i in range(3)]
    cb = M.shape_memory_store(shapes)
    off = [float(cb[i] @ cb[j]) for i in range(3) for j in range(3) if i != j]
    assert min(off) > 0.95, "growth-only bodies became distinguishable: %r" % off
