"""Stable neo-Hookean tetrahedral elasticity with HAND-DERIVED gradients, plus muscle fibers.

BACKLOG F4. The continuum half of the morphogenesis pipeline: the F3 tet mesh becomes a
deformable body whose bulk response is volumetric (per-tet hyperelastic energy) and whose
actuation is sparse (activation-dependent springs on selected edges) -- exactly the source
document's separation, and the reason one control policy can drive bodies of any topology.

SOTA CHECK (searched 2026-08-16) -- AND IT CHANGED THE MODEL:
  * The source document specifies the CLASSICAL neo-Hookean
        Psi = mu/2 (I_C - 3) - mu log J + lambda/2 (log J)^2.
    That form has a fatal property for our pipeline: log J is UNDEFINED for J <= 0, so the
    instant any tet inverts the energy is NaN and the whole solve dies. Morphogenesis meshes
    are generated, not authored, and they DO produce near-degenerate tets.
  * The standard since Smith, De Goes & Kim, "Stable Neo-Hookean Flesh Simulation"
    (ACM TOG 37(2), 2018) removes the log-J term entirely:
        Psi = mu/2 (I_C - 3) + lambda/2 (J - alpha)^2 - mu/2 log(I_C + 1),   alpha = 1 + mu/lambda
    It is finite and smooth for EVERY F including inverted ones (I_C + 1 >= 1 always), gives
    superior volume preservation near Poisson 0.5 -- which is the biological-tissue regime the
    paper was written for -- and is robust to extreme rotations. We implement THIS, not the
    document's version, and say so.
  * Follow-on work (Chen et al., "Stabler Neo-Hookean Simulation: Absolute Eigenvalue
    Filtering for Projected Newton", SIGGRAPH 2024) improves the HESSIAN projection for
    Newton solvers. We descend on gradients, so that machinery is out of scope -- noted here
    so a future session with a Newton solver knows where to look rather than re-deriving.

NO AUTODIFF (hard constraint). The first Piola-Kirchhoff stress P = dPsi/dF is derived by
hand below with each term justified, and the selftest checks it against the engine's own
fd_gradient. Deriving it is three lines of matrix calculus; verifying it is one call.

RULE-0 AUDIT (2026-08-16): `neo hookean` and `piola kirchhoff stress` returned nothing --
genuine gaps. Audited and NOT duplicated: `soft_body` is PBD/XPBD with DISTANCE constraints
(a different discretisation -- this module is a second constitutive model beside it, never a
replacement), and `tissue_fields` is nested SDF anatomy classification, not mechanics.
fd_gradient is reused as the verification instrument, as in F1/F2.

KEPT NEGATIVES:
  * Gradient descent only. No implicit integrator, no inertia, no contact -- this is the
    QUASISTATIC energy and its exact gradient. Dynamics belong to the existing XPBD path
    until a measurement says otherwise.
  * Muscle fibers use the document's activation form directly; that part needed no upgrade.
  * Rest shapes are taken from the input configuration, so a mesh born inverted stays
    inverted-at-rest. Detected and reported by rest_quality(), never silently accepted.
"""

import numpy as np


def _shape_matrices(points, tets):
    """Per-tet rest shape matrix Dm = [X1-X0, X2-X0, X3-X0], its inverse, and rest volume.

    Returns (Dm_inv, vol, ok) where `ok` flags tets with usable (non-degenerate, positively
    oriented) rest shape. Degenerate rest tets are EXCLUDED rather than regularised: a
    zero-volume element has no meaningful deformation gradient, and quietly stiffening it
    would fabricate forces from nothing."""
    pts = np.asarray(points, float)
    tets = np.asarray(tets, int)
    d = np.stack([pts[tets[:, 1]] - pts[tets[:, 0]],
                  pts[tets[:, 2]] - pts[tets[:, 0]],
                  pts[tets[:, 3]] - pts[tets[:, 0]]], axis=2)   # (M,3,3), columns are edges
    det = np.linalg.det(d)
    vol = det / 6.0
    ok = np.abs(det) > 1e-12
    dinv = np.zeros_like(d)
    if np.any(ok):
        dinv[ok] = np.linalg.inv(d[ok])
    return dinv, vol, ok


def _cofactor(f):
    """dJ/dF for 3x3: the cofactor matrix, assembled from cross products of F's columns.
    Written explicitly rather than via det*inv(F) because inv(F) does not exist for an
    inverted or singular element -- and surviving those is the whole point of this model."""
    f0, f1, f2 = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([np.cross(f1, f2), np.cross(f2, f0), np.cross(f0, f1)], axis=-1)


def neohookean_energy_and_grad(points, tets, mu=1.0, lam=10.0, rest=None):
    """Stable neo-Hookean energy over all tets and its EXACT gradient w.r.t. vertex positions.

    Psi = mu/2 (I_C - 3) + lam/2 (J - alpha)^2 - mu/2 log(I_C + 1),  alpha = 1 + mu/lam
    (Smith, De Goes & Kim 2018 -- see module docstring for why not the classical log-J form.)

    HAND DERIVATION, term by term:
        dI_C/dF = 2F                       (I_C = tr(F^T F) = sum of squared entries)
        dJ/dF   = cofactor(F)              (Jacobi's formula, written without inv(F))
        P = dPsi/dF = mu F - mu F/(I_C+1) + lam (J - alpha) cofactor(F)
                    = mu F (1 - 1/(I_C+1)) + lam (J - alpha) cofactor(F)
    Vertex forces follow the standard FEM assembly: H = -vol * P * Dm^{-T} holds the force
    contributions for vertices 1,2,3 as its columns, and vertex 0 takes the negative sum
    (the element exerts no net force on itself -- momentum conservation by construction,
    not by hope). The GRADIENT is the negative of the force, which is what we return.
    """
    pts = np.asarray(points, float)
    tets = np.asarray(tets, int)
    grad = np.zeros_like(pts)
    if len(tets) == 0:
        return 0.0, grad
    dm_inv, vol, ok = _shape_matrices(pts if rest is None else np.asarray(rest, float), tets)
    if not np.any(ok):
        return 0.0, grad
    t = tets[ok]
    dm_inv = dm_inv[ok]
    vol = np.abs(vol[ok])
    ds = np.stack([pts[t[:, 1]] - pts[t[:, 0]],
                   pts[t[:, 2]] - pts[t[:, 0]],
                   pts[t[:, 3]] - pts[t[:, 0]]], axis=2)
    f = ds @ dm_inv                                    # deformation gradient per tet
    ic = np.einsum("mij,mij->m", f, f)                 # I_C = ||F||_F^2
    j = np.linalg.det(f)
    # alpha = 1 + mu/lam - mu/(4 lam) = 1 + 3mu/(4 lam) -- the REST-STABILITY correction, and
    # it is not cosmetic. Derivation: at F = I we have I_C = 3, J = 1, cofactor(I) = I, so
    #     P(I) = mu(1 - 1/4) I + lam(1 - alpha) I = (3mu/4 + lam(1 - alpha)) I,
    # which vanishes only for alpha = 1 + 3mu/(4 lam). Using the uncorrected alpha = 1 + mu/lam
    # (the form quoted in many summaries) leaves a residual stress of -mu/4 at rest: MEASURED
    # here as a 4.17e-2 force on an undeformed tet before the fix -- a body that shrinks the
    # moment you press play. Caught by this module's rest-force assertion, which exists
    # precisely because "no force at rest" is a planted truth with a known answer.
    alpha = 1.0 + 3.0 * mu / (4.0 * lam)
    psi = 0.5 * mu * (ic - 3.0) + 0.5 * lam * (j - alpha) ** 2 - 0.5 * mu * np.log(ic + 1.0)
    energy = float(np.sum(psi * vol))
    p_stress = (mu * (1.0 - 1.0 / (ic + 1.0)))[:, None, None] * f \
        + (lam * (j - alpha))[:, None, None] * _cofactor(f)
    h = -vol[:, None, None] * (p_stress @ np.transpose(dm_inv, (0, 2, 1)))
    # h[:, :, k] is the FORCE on vertex k+1; gradient is -force
    np.add.at(grad, t[:, 1], -h[:, :, 0])
    np.add.at(grad, t[:, 2], -h[:, :, 1])
    np.add.at(grad, t[:, 3], -h[:, :, 2])
    np.add.at(grad, t[:, 0], h[:, :, 0] + h[:, :, 1] + h[:, :, 2])
    return energy, grad


def muscle_energy_and_grad(points, fibers, rest_lengths, activation, k=10.0):
    """Activation-dependent fiber springs, straight from the source document:

        E = k/2 * (l / (a * l0) - 1)^2

    a = 1 leaves the fiber relaxed at its rest length; a < 1 shortens the preferred length
    and the fiber CONTRACTS. One functional form for every fiber, so a single control policy
    drives bodies of any topology -- the document's design point, and the reason muscles are
    edges rather than a second continuum.

    dE/dl = k (l/(a l0) - 1) / (a l0), and dl/dx_i = (x_i - x_j)/l."""
    pts = np.asarray(points, float)
    fib = np.asarray(fibers, int).reshape(-1, 2)
    grad = np.zeros_like(pts)
    if len(fib) == 0:
        return 0.0, grad
    l0 = np.asarray(rest_lengths, float)
    a = np.asarray(activation, float)
    a = np.clip(a, 1e-3, None)                 # a -> 0 is an infinite contraction; refuse it
    dvec = pts[fib[:, 0]] - pts[fib[:, 1]]
    l = np.maximum(np.linalg.norm(dvec, axis=1), 1e-12)
    target = a * l0
    ratio = l / target
    e = 0.5 * k * (ratio - 1.0) ** 2
    dedl = k * (ratio - 1.0) / target
    contrib = (dedl / l)[:, None] * dvec
    np.add.at(grad, fib[:, 0], contrib)
    np.add.at(grad, fib[:, 1], -contrib)
    return float(e.sum()), grad


def select_fibers(points, tets, axis=0, fraction=0.25):
    """Choose which tet edges become muscle fibers: the `fraction` of edges best aligned with
    `axis`. Deterministic (sorted by alignment then by index). Alignment rather than random
    selection because a muscle that pulls in every direction at once does no work -- and that
    was worth stating, since 'pick some edges' is the obvious wrong first implementation."""
    pts = np.asarray(points, float)
    edges = set()
    for t in np.asarray(tets, int):
        ti = [int(x) for x in t]
        for a in range(4):
            for b in range(a + 1, 4):
                edges.add((min(ti[a], ti[b]), max(ti[a], ti[b])))
    edges = sorted(edges)
    if not edges:
        return np.zeros((0, 2), int), np.zeros(0)
    e = np.array(edges, int)
    d = pts[e[:, 0]] - pts[e[:, 1]]
    l = np.linalg.norm(d, axis=1)
    align = np.abs(d[:, int(axis)]) / np.maximum(l, 1e-12)
    order = np.lexsort((np.arange(len(e)), -align))
    keep = order[:max(1, int(len(e) * float(fraction)))]
    keep = np.sort(keep)
    return e[keep], l[keep]


def rest_quality(points, tets):
    """Report the rest mesh's element quality BEFORE anyone simulates it: how many tets are
    degenerate, how many are inverted (negative volume), and the volume extremes.

    Exists because a generated mesh can be born inverted, and a simulator that silently
    accepts that produces confident nonsense. Reported, never repaired in place."""
    _, vol, ok = _shape_matrices(points, tets)
    return {"n": int(len(vol)), "degenerate": int(np.sum(~ok)),
            "inverted": int(np.sum(vol < 0)), "min_vol": float(np.min(vol)) if len(vol) else 0.0,
            "max_vol": float(np.max(vol)) if len(vol) else 0.0}


def simulate(points, tets, steps=200, mu=1.0, lam=10.0, fibers=None, rest_lengths=None,
             activation=1.0, k_muscle=10.0, gravity=0.0, pinned=None, step0=0.01, rest=None):
    """Quasistatic solve: minimise (elastic + muscle + gravity) over vertex positions by
    gradient descent with backtracking, exactly as F1/F2 do. `pinned` indices are held fixed
    (their gradient is zeroed), which is how a body gets an anchor without a constraint solver.

    Returns {"positions","energy","history","rest_quality"}. Deterministic; no rng at all."""
    x = np.array(points, float, copy=True)
    tets = np.asarray(tets, int)
    rest_x = x.copy() if rest is None else np.asarray(rest, float)
    pin = np.zeros(len(x), bool)
    if pinned is not None:
        pin[np.asarray(pinned, int)] = True
    act = np.full(len(fibers) if fibers is not None else 0, float(activation)) \
        if np.isscalar(activation) else np.asarray(activation, float)

    def total(y):
        e, g = neohookean_energy_and_grad(y, tets, mu, lam, rest=rest_x)
        if fibers is not None and len(fibers):
            e2, g2 = muscle_energy_and_grad(y, fibers, rest_lengths, act, k_muscle)
            e += e2
            g = g + g2
        if gravity:
            e += float(gravity) * float(np.sum(y[:, 2]))
            g = g.copy()
            g[:, 2] += float(gravity)
        g[pin] = 0.0
        return e, g

    e, g = total(x)
    hist = [e]
    step = step0
    for _ in range(int(steps)):
        if float(np.linalg.norm(g)) < 1e-12:
            break
        trial = step
        for _ in range(30):
            y = x - trial * g
            e2, g2 = total(y)
            if e2 <= e:
                break
            trial *= 0.5
        else:
            break
        x, e, g = y, e2, g2
        hist.append(e)
        step = min(trial * 1.6, step0 * 8.0)
    return {"positions": x, "energy": e, "history": hist,
            "rest_quality": rest_quality(rest_x, tets)}


def _selftest():
    """Regression trap. The load-bearing pins are the ANALYTIC GRADIENT against fd_gradient
    (both energies), and the INVERSION test that motivated choosing the stable model: a tet
    turned inside out must give a FINITE energy and a finite gradient, where the classical
    log-J form would return NaN."""
    from holographic.misc.holographic_optimize import fd_gradient
    rng = np.random.default_rng(20260816)

    # 1) rest configuration has (near) zero stress: with F = I, I_C = 3 and J = 1, the energy
    #    is a small constant, and the FORCE must vanish -- a body at rest must not move
    ref = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t = np.array([[0, 1, 2, 3]])
    _, g0 = neohookean_energy_and_grad(ref, t, mu=1.0, lam=10.0)
    assert np.abs(g0).max() < 1e-9, "rest state exerts force: %.2e" % np.abs(g0).max()

    # 2) ANALYTIC == FINITE DIFFERENCE, on a deformed random tet cluster
    pts = ref + rng.normal(scale=0.25, size=(4, 3))
    f = lambda flat: neohookean_energy_and_grad(flat.reshape(-1, 3), t, 1.0, 10.0, rest=ref)[0]
    num = fd_gradient(f, pts.ravel().copy(), eps=1e-6).reshape(-1, 3)
    _, ana = neohookean_energy_and_grad(pts, t, 1.0, 10.0, rest=ref)
    err = np.abs(num - ana).max()
    assert err < 1e-4, "neo-Hookean gradient disagrees with fd by %.2e" % err

    # 3) THE REASON FOR THE STABLE MODEL: an INVERTED element stays finite. The classical
    #    log(J) form is NaN here, which would kill the whole solve on one bad tet.
    inv = ref.copy()
    inv[3, 2] = -1.0                      # flip the fourth vertex through the base plane
    e_inv, g_inv = neohookean_energy_and_grad(inv, t, 1.0, 10.0, rest=ref)
    assert np.isfinite(e_inv) and np.all(np.isfinite(g_inv)), "inverted element is not finite"
    assert e_inv > 0.0
    fi = lambda flat: neohookean_energy_and_grad(flat.reshape(-1, 3), t, 1.0, 10.0, rest=ref)[0]
    ni = fd_gradient(fi, inv.ravel().copy(), eps=1e-6).reshape(-1, 3)
    assert np.abs(ni - g_inv).max() < 1e-4, "gradient wrong where it matters most (inverted)"

    # 4) muscle: analytic gradient, and contraction actually SHORTENS (sign check -- a sign
    #    error here produces a muscle that pushes, which looks plausible in motion)
    fib = np.array([[0, 1]])
    l0 = np.array([1.0])
    fm = lambda flat: muscle_energy_and_grad(flat.reshape(-1, 3), fib, l0, np.array([0.5]))[0]
    nm = fd_gradient(fm, pts.ravel().copy(), eps=1e-6).reshape(-1, 3)
    _, am = muscle_energy_and_grad(pts, fib, l0, np.array([0.5]))
    assert np.abs(nm - am).max() < 1e-5, "muscle gradient off by %.2e" % np.abs(nm - am).max()
    # test the BEHAVIOUR, not the sign convention: take a descent step and check the fiber
    # actually got SHORTER. (An assertion on the gradient's sign is a test of which way the
    # author was thinking; this one is a test of what the muscle does.)
    stretched = np.array([[0., 0, 0], [1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    _, gm = muscle_energy_and_grad(stretched, fib, l0, np.array([0.5]))
    moved = stretched - 0.01 * gm
    before = np.linalg.norm(stretched[0] - stretched[1])
    after = np.linalg.norm(moved[0] - moved[1])
    assert after < before, ("an activated fiber must CONTRACT: %.4f -> %.4f" % (before, after))
    # and a relaxed fiber (a = 1) at its rest length must do nothing at all
    _, grelax = muscle_energy_and_grad(stretched, fib, l0, np.array([1.0]))
    assert np.abs(grelax).max() < 1e-12, "a relaxed fiber at rest length is exerting force"

    # 5) end-to-end on a real morphogenesis mesh: descent, finiteness, and honest reporting
    from holographic.simulation_and_physics.holographic_morphogen import grow_aggregate
    from holographic.mesh_and_geometry.holographic_tetmesh import tetrahedralize
    agg = grow_aggregate(n_cells=30, seed=0, steps=60)
    mesh = tetrahedralize(agg["positions"], agg["radii"])
    fibers, rl = select_fibers(agg["positions"], mesh["tets"], axis=0, fraction=0.2)
    out = simulate(agg["positions"], mesh["tets"], steps=60, fibers=fibers,
                   rest_lengths=rl, activation=0.7, pinned=[0])
    assert np.all(np.isfinite(out["positions"]))
    assert out["history"][-1] <= out["history"][0] + 1e-9, "energy increased"
    assert out["rest_quality"]["n"] == len(mesh["tets"])
    print("OK: holographic_fem -- rest force %.1e, grad vs fd %.1e, INVERTED element finite "
          "(E=%.3f), muscle pulls, %d tets solved (%.1f -> %.1f), rest: %d inverted / %d "
          "degenerate" % (np.abs(g0).max(), err, e_inv, len(mesh["tets"]),
                          out["history"][0], out["history"][-1],
                          out["rest_quality"]["inverted"], out["rest_quality"]["degenerate"]))


if __name__ == "__main__":
    _selftest()
