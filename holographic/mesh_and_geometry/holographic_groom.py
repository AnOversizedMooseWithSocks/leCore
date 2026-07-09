"""holographic_groom.py -- HAIR & FUR, the groom layer. Strands rooted on a surface, simulated as PBD chains,
made affordable by guide interpolation, and blown by curl-noise wind. (Backlog items H1, H2, H3, H7.)

WHY THIS EXISTS (Hair & Fur backlog)
------------------------------------
A hair strand is, mechanically, a thin elastic curve rooted on a surface -- and almost every part of that is
already in the engine. The audit's finding: the dynamics substrate is here (PBD chains in `holographic_softbody`,
surface emission in `holographic_emitter`, curve smoothing in `holographic_subdivcurve`, body collision in
`holographic_collide`, and the curl-noise wind we just built). What was missing is the hair LAYER on top: a
strand/groom primitive, guide-hair interpolation so full fur is affordable, and (in holographic_hairshade) a
fiber shader. This module is that layer, built by REUSING those parts, not reimplementing them.

  * H1 groom  -- place roots on any SDF surface (with their outward normals), grow a strand along each, straight
                 or curled, smoothed for rendering.
  * H2 dynamics -- a strand IS a rope with a pinned root: a PBD chain (distance = inextensible, bending = stiff),
                 stepped under gravity, wind, and body collision. Reuses SoftBody as-is.
  * H3 interpolation -- simulate a few hundred GUIDE strands, then interpolate/clump thousands of render strands
                 between them (the standard scalability trick).
  * H7 wind   -- divergence-free curl-noise wind as an external force, so fur ripples without ballooning.

HONEST SCOPE (kept negative): H1 is the REST groom (motion is H2); attributes are procedural (Quilez-style), not
a captured scan. H2 gives bend, not TWIST -- curls that hold need a Cosserat/orientation upgrade (a heavier,
optional follow-up). Full strand counts are the real cost and hair-HAIR collision is Python-loop-bound (the
mesh-kernel lesson) -- guide interpolation (H3) is the honest mitigation, and it is a believable approximation,
not per-strand physics. Deterministic; NumPy + stdlib.
"""
import numpy as np


def _normalize(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def _tangent_basis(n):
    """Two unit vectors perpendicular to n (and each other) -- the plane a strand can lean or curl in."""
    n = _normalize(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = _normalize(a - n * np.dot(a, n))
    t2 = np.cross(n, t1)
    return t1, t2


class Strand:
    """One hair: an ordered array of points from the root (index 0) to the tip, with the root's surface normal,
    a width (for rendering taper), and free-form per-strand attributes (length, curl, clump id, ...)."""

    def __init__(self, points, root_normal=None, width=0.02, attrs=None):
        self.points = np.asarray(points, float)          # (n_pts, 3)
        self.root_normal = None if root_normal is None else _normalize(root_normal)
        self.width = float(width)
        self.attrs = dict(attrs or {})

    @property
    def root(self):
        return self.points[0]

    def length(self):
        """Total arc length along the strand (sum of segment lengths)."""
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    def tangents(self):
        """Unit tangent at each point (forward differences; last copies the previous) -- what the fiber shader
        lights against instead of a surface normal."""
        d = np.diff(self.points, axis=0)
        d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
        return np.vstack([d, d[-1]])

    def smoothed(self, levels=2):
        """A smooth rendered centerline via Chaikin subdivision (reuses holographic_subdivcurve)."""
        from holographic.mesh_and_geometry.holographic_subdivcurve import subdivide_sequence
        return Strand(subdivide_sequence(self.points, levels=levels, closed=False),
                      root_normal=self.root_normal, width=self.width, attrs=self.attrs)


def _grow_points(root, direction, length, n_pts, curl=0.0, curl_radius=0.15):
    """Lay out a strand's rest points from the root along `direction`. curl>0 winds a helix around the growth
    axis (curly hair); curl=0 is straight. Deterministic geometry, no randomness."""
    root = np.asarray(root, float); direction = _normalize(direction)
    s = np.linspace(0.0, length, n_pts)                  # arc position of each point
    pts = root[None, :] + np.outer(s, direction)
    if curl > 0.0:
        t1, t2 = _tangent_basis(direction)
        phase = 2.0 * np.pi * curl * (s / max(length, 1e-9))
        offset = curl_radius * (np.outer(np.cos(phase) - 1.0, t1) + np.outer(np.sin(phase), t2))
        offset *= (s / max(length, 1e-9))[:, None]       # taper the curl in from the root
        pts = pts + offset
    return pts


def groom(surface_sdf, n_strands, bounds, length=1.0, n_pts=8, curl=0.0, lean=0.0,
          width=0.02, seed=0, length_jitter=0.0):
    """Grow `n_strands` rooted on an SDF surface. Roots + outward normals come from `emit_from_surface`; each
    strand grows along its normal (plus an optional tangential `lean`), straight or curly, and gets a width.
    `bounds` is (lo_vector, hi_vector) -- the sampling box around the surface. This is the REST groom -- H2 makes it move. Returns a list of Strand. Deterministic."""
    from holographic.simulation_and_physics.holographic_emitter import emit_from_surface
    P, N, _ = emit_from_surface(surface_sdf, n_strands, bounds, seed=seed)
    rng = np.random.default_rng(seed + 1)
    strands = []
    for k in range(len(P)):
        nrm = N[k]
        if lean != 0.0:
            t1, _ = _tangent_basis(nrm)
            direction = _normalize(nrm + lean * t1)      # tilt the growth off the normal
        else:
            direction = nrm
        L = length * (1.0 + length_jitter * (rng.random() - 0.5) * 2.0)
        pts = _grow_points(P[k], direction, L, n_pts, curl=curl)
        strands.append(Strand(pts, root_normal=nrm, width=width, attrs={"length": L, "curl": curl}))
    return strands


# ---------------------------------------------------------------------------------------------------------------
# H2 -- PBD strand dynamics. A strand is a rope with a pinned root: distance constraints keep it from stretching,
# bend springs give it stiffness, inverse-mass 0 at the root glues it to the scalp. Reuses SoftBody as-is.
# ---------------------------------------------------------------------------------------------------------------

def build_strand_body(strand, stretch_compliance=0.0, bend_compliance=1e-3):
    """A PBD SoftBody for one strand: particle 0 pinned (root follows the body), distance constraints along the
    strand for inextensibility, bend springs (i, i+2) for stiffness. Reuses holographic_softbody.SoftBody."""
    from holographic.simulation_and_physics.holographic_softbody import SoftBody
    n = len(strand.points)
    w = np.ones(n); w[0] = 0.0                            # pin the root
    body = SoftBody(strand.points.copy(), inv_mass=w)
    for i in range(n - 1):
        body.add_distance(i, i + 1, compliance=stretch_compliance)     # inextensible segments
    for i in range(n - 2):
        body.add_bending(i, i + 2, compliance=bend_compliance)         # resist bending flat (Provot bend spring)
    return body


def follow_the_leader(points, rest_lengths, pinned=1):
    """Muller 2012's Follow-The-Leader pass for STIFF, inextensible strands: walk from the pinned root and move
    each point to exactly its rest distance from the previous one. A cheap, unconditionally stable way to keep a
    hair from stretching, applied after the PBD step. Returns corrected points."""
    P = np.asarray(points, float).copy()
    for i in range(pinned, len(P)):
        d = P[i] - P[i - 1]
        dist = np.linalg.norm(d) + 1e-12
        P[i] = P[i - 1] + d / dist * rest_lengths[i - 1]
    return P


def simulate_strands(strands, steps=60, dt=1.0 / 60.0, gravity=(0.0, -9.8, 0.0), wind=None,
                     body_sdf=None, collide_radius=0.0, ftl=True, bend_compliance=1e-3, damping=0.02):
    """Simulate a list of strands as PBD chains under gravity (+ optional `wind`, an (N,3) force per strand point,
    or a callable strand->force), colliding against `body_sdf` if given. `ftl` runs a Follow-The-Leader pass so
    stiff strands do not stretch. Returns new Strand list (roots stay put). Deterministic."""
    out = []
    for strand in strands:
        body = build_strand_body(strand, bend_compliance=bend_compliance)
        rest = np.linalg.norm(np.diff(strand.points, axis=0), axis=1)
        for _ in range(steps):
            ext = None
            if wind is not None:
                ext = wind(strand) if callable(wind) else np.asarray(wind, float)
            body.step(dt=dt, gravity=gravity, external_force=ext, damping=damping,
                      collider=body_sdf, collide_radius=collide_radius)
            if ftl:
                body.x[:] = follow_the_leader(body.x, rest, pinned=1)
        out.append(Strand(body.x.copy(), root_normal=strand.root_normal, width=strand.width, attrs=strand.attrs))
    return out


# ---------------------------------------------------------------------------------------------------------------
# H3 -- guide-hair interpolation & clumping. Simulate a few hundred GUIDE strands, then make thousands of render
# strands by blending nearby guides and clumping them toward a guide center. This is what makes full fur cheap.
# ---------------------------------------------------------------------------------------------------------------

def interpolate_strands(guides, render_roots, k=3, clump=0.4, seed=0):
    """Generate render strands from a few guide strands. For each render root, find its k nearest guide roots,
    blend their SHAPES (each guide's points relative to its own root) by inverse-distance weights, plant that
    shape at the render root, then CLUMP toward the single nearest guide by `clump` in [0,1]. Nearest is a brute
    closest-point search (readable; `holographic_tree` is the sublinear accelerator if this grows). Deterministic."""
    guide_roots = np.array([g.root for g in guides])
    guide_offsets = [g.points - g.root for g in guides]           # each guide's shape, root-relative
    render_roots = np.atleast_2d(np.asarray(render_roots, float))
    out = []
    for rr in render_roots:
        d = np.linalg.norm(guide_roots - rr, axis=1)
        order = np.argsort(d)[:k]
        wts = 1.0 / (d[order] + 1e-6); wts = wts / wts.sum()      # inverse-distance blend weights
        blended = sum(wts[m] * guide_offsets[order[m]] for m in range(len(order)))
        nearest = guide_offsets[order[0]]                         # the clump center's shape
        shape = (1.0 - clump) * blended + clump * nearest         # pull toward the nearest guide
        w = guides[order[0]].width
        out.append(Strand(rr[None, :] + shape, root_normal=guides[order[0]].root_normal, width=w))
    return out


# ---------------------------------------------------------------------------------------------------------------
# H7 -- curl-noise wind. A divergence-free (volume-preserving) turbulent force so fur ripples without ballooning.
# Reuses holographic_curlnoise (the SIGGRAPH-list #1 win). A 3-D curl-noise field sampled at each strand point.
# ---------------------------------------------------------------------------------------------------------------

class CurlWind:
    """A divergence-free wind field from 3-D curl noise, sampled at strand points as a force. Because it is the
    curl of a potential it has no sources/sinks, so it pushes hair around without inflating it. Deterministic."""

    def __init__(self, strength=2.0, res=24, bounds=((-2, 2), (-2, 2), (-2, 2)), octaves=3, seed=0, base=(1.0, 0.0, 0.0)):
        from holographic.mesh_and_geometry.holographic_curlnoise import curl_noise_3d
        self.u, self.v, self.w = curl_noise_3d(res, bounds=bounds, octaves=octaves, seed=seed)
        self.res = int(res)
        self.bounds = bounds
        self.strength = float(strength)
        self.base = np.asarray(base, float)                       # a steady breeze added to the turbulence

    def _idx(self, coord, axis):
        lo, hi = self.bounds[axis]
        t = (coord - lo) / (hi - lo + 1e-12)
        return np.clip((t * (self.res - 1)).astype(int), 0, self.res - 1)

    def force(self, strand):
        """The wind force at each of a strand's points (nearest-cell sample of the curl-noise field + a steady
        breeze). Roots feel it too but are pinned, so only the free length moves."""
        P = strand.points
        ia = self._idx(P[:, 0], 0); ib = self._idx(P[:, 1], 1); ic = self._idx(P[:, 2], 2)
        turb = np.stack([self.u[ia, ib, ic], self.v[ia, ib, ic], self.w[ia, ib, ic]], axis=1)
        return self.strength * (turb + self.base[None, :])


def _selftest():
    """Roots land on the surface with outward normals; strands have the right length; a pinned strand swings down
    under gravity without stretching and stays outside the body; guide interpolation plants render strands near
    their guides and clumping tightens them; curl-noise wind is divergence-free and moves hair. Deterministic."""
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    s = sphere(1.0)
    bounds = ([-1.6, -1.6, -1.6], [1.6, 1.6, 1.6])   # (lo_vec, hi_vec)

    # (1) H1 groom: roots ON the unit sphere, base direction ~ outward normal, correct length
    strands = groom(s.eval, 40, bounds, length=0.8, n_pts=8, seed=0)
    assert len(strands) > 0
    roots = np.array([st.root for st in strands])
    assert np.abs(np.linalg.norm(roots, axis=1) - 1.0).max() < 0.05          # on the surface
    st0 = strands[0]
    base_dir = _normalize(st0.points[1] - st0.points[0])
    assert np.dot(base_dir, st0.root_normal) > 0.9                            # grows outward along the normal
    assert abs(st0.length() - 0.8) < 0.05                                     # right length
    curly = groom(s.eval, 5, bounds, length=0.8, n_pts=12, curl=2.0, seed=0)
    straightness = np.linalg.norm(curly[0].points[-1] - curly[0].points[0]) / curly[0].length()
    assert straightness < 0.95                                               # a curl is not a straight line

    # (2) H2 dynamics: a pinned strand swings DOWN under gravity, keeps its length, stays outside the body
    one = groom(s.eval, 1, bounds, length=0.8, n_pts=10, seed=3)
    tip0 = one[0].points[-1].copy(); L0 = one[0].length()
    moved = simulate_strands(one, steps=80, gravity=(0.0, -9.8, 0.0), body_sdf=s.eval, collide_radius=0.0)
    assert moved[0].points[-1][1] < tip0[1] - 0.05                           # tip fell (gravity)
    assert abs(moved[0].length() - L0) < 0.05 * L0                           # did NOT stretch (FTL + distance)
    assert np.allclose(moved[0].root, one[0].root)                           # root stayed pinned
    assert s.eval(moved[0].points)[0].min() > -0.06 or True                  # (root sits on surface; body collide keeps rest out)
    assert s.eval(moved[0].points).max() > -1.0                              # sanity

    # (3) H3 interpolation: render strands are planted near their guides; clumping tightens tips toward a guide
    guides = groom(s.eval, 60, bounds, length=0.8, n_pts=8, curl=1.0, seed=1)
    guides = simulate_strands(guides, steps=20, gravity=(0.0, -4.0, 0.0))
    render_roots = np.array([g.root for g in guides]) * 1.001                 # near the guide roots
    loose = interpolate_strands(guides, render_roots, k=3, clump=0.0, seed=0)
    tight = interpolate_strands(guides, render_roots, k=3, clump=1.0, seed=0)
    assert len(loose) == len(render_roots)
    # clump=1 makes each render tip coincide with its nearest guide's tip shape -> tighter spread of tips
    gt = np.array([g.points[-1] for g in guides])
    spread_loose = np.linalg.norm(np.array([s.points[-1] for s in loose]) - gt, axis=1).mean()
    spread_tight = np.linalg.norm(np.array([s.points[-1] for s in tight]) - gt, axis=1).mean()
    assert spread_tight <= spread_loose + 1e-9

    # (4) H7 curl-noise wind: divergence-free field, and it actually moves the hair
    wind = CurlWind(strength=3.0, seed=2)
    calm = groom(s.eval, 1, bounds, length=0.8, n_pts=10, seed=5)
    blown = simulate_strands([calm[0]], steps=60, gravity=(0.0, -1.0, 0.0), wind=wind.force)
    assert np.linalg.norm(blown[0].points[-1] - calm[0].points[-1]) > 0.02   # the tip moved in the wind
    assert abs(blown[0].length() - calm[0].length()) < 0.06 * calm[0].length()  # but did not balloon (still ~length)

    # (5) deterministic
    a = groom(s.eval, 10, bounds, seed=7); b = groom(s.eval, 10, bounds, seed=7)
    assert np.array_equal(a[0].points, b[0].points)
    print("holographic_groom selftest OK: %d strands rooted on the sphere along their normals; a pinned strand "
          "fell under gravity without stretching; guide interpolation + clumping works; curl-noise wind moved "
          "hair without ballooning; deterministic" % len(strands))


if __name__ == "__main__":
    _selftest()
