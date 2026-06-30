"""Gradient-field navigation with caustic detection -- a 'gravitational lens' over a cloud of attractors,
extracted from leOS (lvm/gravitational_lens.py) and decoupled from its nutrient-field so it works on any set of
weighted points.

The picture: treat a set of stored points as MASSES on the hypersphere. A query feels a force toward them --
each attractor pulls along the geodesic (log_map direction) with a strength that is its mass times a Gaussian
falloff in geodesic distance. DEFLECTING the query along that force (via exp_map) slides it toward the nearby
mass concentration -- a SOFT, continuous cousin of cleanup: where cleanup hard-snaps to the single nearest atom,
this drifts toward the weighted local centre of mass, and NAVIGATE iterates it to climb the field to an attractor.

The genuinely useful part is CAUSTIC DETECTION. In optics a caustic is a fold where light rays focus and the
mapping becomes singular -- the cusp of a coffee-cup reflection. Here it is the same idea in embedding space: a
point where two attractors pull in OPPOSITE directions with similar strength is a routing FOLD -- the query is
torn between them, and a tiny move flips which one wins. The caustic score flags exactly those ambiguous,
decision-boundary regions. This is complementary to RecallNull (which asks 'is this a match at all?'): caustic asks
'is this AMBIGUOUS between matches?'.

Pure NumPy, reusing holostuff's own log_map / exp_map / geodesic. The force is a DIRECT O(N) sum over attractors --
exact and clear; leOS used a Barnes-Hut tree to make it O(N log N), which is the acceleration for very large clouds
(holostuff's HoloForest would supply the spatial structure if needed) and a deliberate simplification here.

HONEST: deflection is a heuristic drift, not a guaranteed descent to the global nearest cluster -- with a wide
sigma it can over-smooth (pull toward the global centroid), with a narrow sigma it only feels very close mass and
barely moves. sigma is the scale knob and there is no free lunch; measured below.
"""

import numpy as np
from holographic_ai import log_map, exp_map, geodesic


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _forces(query, attractors, masses, sigma):
    """Per-attractor pull at `query`: magnitudes (N,) = mass * Gaussian(geodesic^2 / 2 sigma^2), and unit tangent
    DIRECTIONS (N, D) pointing along the geodesic toward each attractor. Vectorised. Attractors at the query (or
    antipodal, where the tangent is undefined) get zero magnitude."""
    q = _normalize(np.asarray(query, float))
    A = np.asarray(attractors, float)
    m = np.ones(len(A)) if masses is None else np.asarray(masses, float)
    sims = np.clip(A @ q, -1.0, 1.0)
    theta = np.arccos(sims)                                   # geodesic distance to each attractor
    perp = A - sims[:, None] * q                             # component of each attractor perpendicular to q
    pn = np.linalg.norm(perp, axis=1, keepdims=True)
    dirs = perp / (pn + 1e-12)                               # unit tangent directions toward the attractors
    gaussian = np.exp(-(theta ** 2) / (2.0 * sigma ** 2))
    mags = m * gaussian
    mags = np.where((theta > 1e-8) & (pn[:, 0] > 1e-10), mags, 0.0)   # kill undefined-direction contributions
    return mags, dirs, q


def field_force(query, attractors, masses=None, sigma=0.5):
    """The net force on `query` from the field of attractors -- a tangent vector at the query, sum of each
    attractor's (unit direction * mass * Gaussian falloff). Points toward the nearby mass concentration."""
    mags, dirs, _ = _forces(query, attractors, masses, sigma)
    return (mags[:, None] * dirs).sum(axis=0)


def deflect(query, attractors, masses=None, sigma=0.5, strength=0.1):
    """Slide the query toward the local mass concentration by one lensing step (move along the force via exp_map).
    Returns (lensed, deflection_magnitude, force_magnitude) -- deflection in radians of geodesic travel."""
    q = _normalize(np.asarray(query, float))
    force = field_force(q, attractors, masses, sigma)
    fmag = float(np.linalg.norm(force))
    lensed = exp_map(q, strength * force)
    return lensed, float(geodesic(q, lensed)), fmag


def detect_caustic(query, attractors, masses=None, sigma=0.5, significant=0.5):
    """Routing-ambiguity (caustic) score at `query`: high when the two strongest attractors pull in OPPOSITE
    directions with similar magnitude (a fold). Returns (caustic_score in [0,1], n_significant_attractors). 0 = a
    clear winner, ~1 = a perfect tie pulling apart -- a decision boundary the navigation is unstable on."""
    mags, dirs, _ = _forces(query, attractors, masses, sigma)
    if len(mags) < 2 or mags.max() < 1e-10:
        return 0.0, int((mags > 1e-10).sum())
    order = np.argsort(mags)[::-1]
    strongest = mags[order[0]]
    n_sig = int(np.sum(mags > strongest * significant))
    if n_sig < 2:
        return 0.0, n_sig
    d1, d2 = dirs[order[0]], dirs[order[1]]
    dir_sim = float(np.dot(d1, d2))                          # +1 = same way (no caustic), -1 = opposed (caustic)
    mag_ratio = float(mags[order[1]] / max(mags[order[0]], 1e-10))
    score = max(0.0, (1.0 - dir_sim) * mag_ratio / 2.0)
    return float(score), n_sig


def navigate(query, attractors, masses=None, sigma=0.5, strength=0.6, steps=40, tol=1e-4, decay=0.1):
    """Climb the field from `query` toward an attractor: iterate deflect with a decaying step (anneal
    strength/(1+decay*t)) so it settles instead of orbiting the well. Returns dict{final, path, n_steps,
    max_caustic (strongest ambiguity met en route)}. The continuous, field-following cousin of iterating cleanup,
    and it reports whether the route crossed a caustic. HONEST: a heuristic drift -- it APPROACHES an attractor
    (the decay schedule trades final precision against speed), it does not exactly solve for the nearest one."""
    q = _normalize(np.asarray(query, float))
    path = [q]
    max_caustic = 0.0
    n = 0
    for t in range(steps):
        n = t + 1
        max_caustic = max(max_caustic, detect_caustic(q, attractors, masses, sigma)[0])
        stg = strength / (1.0 + decay * t)
        force = field_force(q, attractors, masses, sigma)
        step = stg * force
        q = exp_map(q, step)
        path.append(q)
        if float(np.linalg.norm(step)) < tol:
            break
    return {"final": q, "path": path, "n_steps": n, "max_caustic": round(float(max_caustic), 4)}


def _selftest():
    rng = np.random.default_rng(0)
    D = 32
    a1 = _normalize(rng.standard_normal(D))
    # a clearly-separated second attractor
    a2 = _normalize(rng.standard_normal(D))
    while abs(np.dot(a1, a2)) > 0.3:
        a2 = _normalize(rng.standard_normal(D))
    attractors = np.stack([a1, a2])

    # 1. deflection moves a query NEARER its closest attractor
    q = _normalize(a1 + 0.5 * rng.standard_normal(D))
    before = geodesic(q, a1)
    lensed, dmag, _ = deflect(q, attractors, sigma=0.8, strength=0.5)
    assert geodesic(lensed, a1) < before, (before, geodesic(lensed, a1))
    # 2. navigate climbs to an attractor (ends much closer than it started)
    nav = navigate(q, attractors, sigma=0.8, strength=0.6)
    assert geodesic(nav["final"], a1) < 0.15, geodesic(nav["final"], a1)   # approaches the attractor closely
    # 3. caustic: a query on the MIDPOINT between two attractors is ambiguous; one near a single attractor is not
    mid = _normalize(a1 + a2)
    near = _normalize(a1 + 0.05 * rng.standard_normal(D))
    c_mid = detect_caustic(mid, attractors, sigma=0.8)[0]
    c_near = detect_caustic(near, attractors, sigma=0.8)[0]
    assert c_mid > c_near, (c_mid, c_near)                   # the fold between attractors scores higher
    print(f"lens selftest ok: deflect closed {before:.3f}->{geodesic(lensed, a1):.3f} rad to attractor; "
          f"navigate approached to {geodesic(nav['final'], a1):.3f} rad in {nav['n_steps']} steps; "
          f"caustic midpoint {c_mid:.3f} > near-attractor {c_near:.3f}")


if __name__ == "__main__":
    _selftest()
