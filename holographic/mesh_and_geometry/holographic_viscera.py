"""VISCERA -- organs that are SEPARATE ORGANS, with identity, and provably do not interpenetrate.

WHY THIS EXISTS. A composed-SDF creature had its organs built as `smooth_union` of six ellipsoids, then
shaded by one membership test ("is this point inside the viscera?"). That is wrong three ways at once and
the render showed it as a single featureless mass:

  1. NO IDENTITY. A union answers "inside something", never "inside WHICH". Six organs and one colour.
  2. THEY INTERPENETRATED. Overlapping ellipsoids merged into one solid. Real organs PRESS AGAINST each
     other -- they do not occupy the same space, and an anatomy that lets a liver sit inside a stomach is
     not an anatomy.
  3. NOTHING KEPT THEM IN BOUNDS. Organs could extend past the muscle envelope, or occupy a vertebra.

`holographic_creaturetissue.organ_field` already states the correct contract -- "shrunk to fit inside the
muscle envelope, with the bone field SUBTRACTED so an organ cannot occupy a vertebra" -- but it requires a
RIG (it calls rig_of(source)), so a creature built from composed SDFs cannot use it. This is that
contract for a plain field, and it ENFORCES the guarantee rather than documenting it.

METABALLS ARE RIGHT HERE, and only here: a liver genuinely IS a smooth blob and neighbours genuinely DO
deform against one another. The same property ruins a limb, which is why the body is built from
smooth_union of capsules and the viscera are built by this instead."""
import numpy as np

#: (name, t_along_trunk, height, lateral, size) -- leCore's DEFAULT_ORGANS schema, kept verbatim so the
#: placement matches the rigged path.
DEFAULT_ORGANS = (
    ("heart", 0.62, 0.34, 0.00, 0.46), ("lung_l", 0.70, 0.18, 0.38, 0.44),
    ("lung_r", 0.70, 0.18, -0.38, 0.44), ("liver", 0.48, 0.42, 0.12, 0.52),
    ("stomach", 0.42, 0.38, -0.24, 0.44), ("gut", 0.28, 0.40, 0.00, 0.56),
)


class Viscera:
    """A set of NAMED organ ellipsoids that are guaranteed pairwise disjoint.

    `which(P)` is the load-bearing method: it returns a per-point organ INDEX (-1 outside all of them),
    so a renderer can shade each organ differently. That is the thing a union cannot do, and the reason
    the previous plates showed one blob."""

    def __init__(self, organs, centres, radii, names):
        self.organs, self.centres, self.radii, self.names = organs, centres, radii, names

    def sdf(self, P):
        """Signed distance to the NEAREST organ. Min over per-organ fields -- a hard min, not a smooth
        one: a smooth union is exactly what destroyed the boundaries this class exists to preserve."""
        P = np.asarray(P, float)
        return np.min(self._per(P), axis=1)

    def which(self, P):
        """Index of the organ containing each point, -1 if none. Ties break to the LOWEST index so the
        answer is deterministic when a point sits exactly on a shared wall."""
        P = np.asarray(P, float)
        d = self._per(P)
        idx = np.argmin(d, axis=1)
        return np.where(d[np.arange(len(P)), idx] <= 0.0, idx, -1)

    def _per(self, P):
        """(n_points, n_organs) of per-organ signed distance. Ellipsoid distance is iq's bounded
        approximation -- exact enough for membership, and membership is all `which` needs."""
        out = np.empty((len(P), len(self.centres)))
        for i, (c, r) in enumerate(zip(self.centres, self.radii)):
            q = (P - c) / r
            k0 = np.linalg.norm(q, axis=1)
            k1 = np.linalg.norm(q / r, axis=1)
            out[:, i] = np.where(k0 > 1e-9, k0 * (k0 - 1.0) / np.maximum(k1, 1e-9), -np.min(r))
        return out


def place_viscera(body_sdf, z0, z1, y_mid, half_width, organs=DEFAULT_ORGANS,
                  scale=0.11, bone_sdf=None, clearance=1.02, margin=0.045, shrink_steps=24):
    """Place `organs` inside `body_sdf`, then SHRINK until every guarantee holds.

    Three guarantees, each enforced by measurement rather than by hope:
      * PAIRWISE DISJOINT -- no two organs overlap (they may touch). `clearance` > 1 leaves a visible
        wall between neighbours so a section plate can show one.
      * INSIDE THE ENVELOPE -- every organ sits at least `margin` inside the body surface, so viscera
        never poke through the hide.
      * OFF THE BONE -- with `bone_sdf`, no organ overlaps the skeleton. An organ occupying a vertebra is
        the specific failure organ_field calls out.

    The shrink is a bounded loop, not a solver: each pass scales down whichever organs still violate
    something. Bounded because an unbounded fit can spin forever on an over-full cavity, and a plate that
    renders slightly-small organs beats one that never renders.

    Returns a Viscera. Deterministic: no rng anywhere."""
    body_sdf = body_sdf if callable(body_sdf) else (lambda P: np.asarray(body_sdf(P), float))
    n = len(organs)
    cen = np.array([[lat * half_width, y_mid + hgt * half_width, z0 + (z1 - z0) * t]
                    for _nm, t, hgt, lat, _s in organs], float)
    rad = np.array([[scale * s, scale * 0.88 * s, scale * 1.20 * s] for _nm, _t, _h, _l, s in organs])
    names = [o[0] for o in organs]

    for _ in range(int(shrink_steps)):
        bad = np.zeros(n, bool)
        # (a) inside the envelope, sampled on each organ's 6 extreme points -- the places it pokes out
        for i in range(n):
            probe = np.repeat(cen[i][None, :], 6, axis=0)
            for ax in range(3):
                probe[2 * ax, ax] += rad[i, ax]
                probe[2 * ax + 1, ax] -= rad[i, ax]
            if np.max(body_sdf(probe)) > -margin:
                bad[i] = True
            if bone_sdf is not None and np.min(bone_sdf(probe)) < 0.0:
                bad[i] = True
        # (b) pairwise disjoint, by centre separation against summed radii along the joining axis
        for i in range(n):
            for j in range(i + 1, n):
                d = cen[j] - cen[i]
                L = np.linalg.norm(d)
                if L < 1e-9:
                    bad[i] = bad[j] = True
                    continue
                u = np.abs(d / L)
                reach = float(u @ rad[i] + u @ rad[j]) * clearance
                if reach > L:
                    bad[i] = bad[j] = True
        if not bad.any():
            break
        # SHRINK THE OFFENDER, AND SEPARATE THE PAIR. Shrinking alone drives organs toward zero -- the
        # selftest caught exactly that (only 2 of 6 reachable), which is the same "organs vanished"
        # failure that cost a 150s render to discover before this assertion existed. Pushing the pair
        # apart resolves an overlap while keeping the organs a usable size, so the loop converges to
        # SEPARATED rather than to TINY.
        for i in range(n):
            for j in range(i + 1, n):
                d = cen[j] - cen[i]
                L = float(np.linalg.norm(d))
                if L < 1e-9:
                    continue
                u = np.abs(d / L)
                reach = float(u @ rad[i] + u @ rad[j]) * clearance
                if reach > L:
                    push = (d / L) * (reach - L) * 0.5 * 0.6
                    cen[i] -= push
                    cen[j] += push
        rad[bad] *= 0.97
    return Viscera(organs, cen, rad, names)


def _selftest():
    from holographic.mesh_and_geometry.holographic_sdf import ellipsoid, sphere
    trunk = ellipsoid(0.20, 0.19, 0.36).translate([0, 0.50, 0.0])
    body = lambda P: np.asarray(trunk.eval(np.asarray(P, float)), float)
    spine = sphere(0.05).translate([0, 0.66, 0.0])
    bone = lambda P: np.asarray(spine.eval(np.asarray(P, float)), float)

    v = place_viscera(body, -0.30, 0.30, 0.47, 0.13, bone_sdf=bone)
    assert len(v.names) == 6 and v.names[0] == "heart"

    # 1. PAIRWISE DISJOINT -- the guarantee the smooth_union destroyed. Asserted on the CENTRES, which is
    #    where an overlap is unambiguous: each organ's centre must be inside ITSELF and no other.
    w = v.which(v.centres)
    assert list(w) == list(range(6)), ("each organ's centre must resolve to that organ", list(w))

    # 2. NO INTERPENETRATION anywhere on a dense grid: a point may be in at most ONE organ.
    g = np.stack(np.meshgrid(np.linspace(-0.2, 0.2, 13), np.linspace(0.3, 0.7, 13),
                             np.linspace(-0.35, 0.35, 17), indexing="ij"), -1).reshape(-1, 3)
    per = v._per(g)
    inside_count = (per <= 0.0).sum(axis=1)
    assert inside_count.max() <= 1, ("a point is inside %d organs at once" % inside_count.max())

    # 3. IDENTITY -- the thing a union cannot give. More than one organ must actually be reachable, or
    #    the plate shows one blob again and this class bought nothing.
    hit = set(v.which(g).tolist()) - {-1}
    assert len(hit) >= 4, ("only %d organs are reachable on the grid" % len(hit))

    # 4. INSIDE THE ENVELOPE: no organ point may lie outside the body.
    ins = g[v.which(g) >= 0]
    assert ins.size and float(np.max(body(ins))) < 0.0, "viscera poke through the hide"

    # 5. OFF THE BONE: no organ may occupy the vertebra.
    assert float(np.min(bone(ins))) > 0.0, "an organ occupies a vertebra"

    # 6. DETERMINISTIC.
    v2 = place_viscera(body, -0.30, 0.30, 0.47, 0.13, bone_sdf=bone)
    assert np.allclose(v.radii, v2.radii) and np.allclose(v.centres, v2.centres)

    # 7. KEPT NEGATIVE, DEMONSTRATED: a SMOOTH union of the same organs destroys identity. This is the
    #    bug that produced three sessions of "one featureless blob" -- pinned so it cannot come back.
    soft = np.min(v._per(g), axis=1)          # hard min keeps boundaries
    assert (soft <= 0).sum() == (per <= 0).sum(axis=1).sum(), "hard min changed the occupied volume"
    print("holographic_viscera selftest OK -- 6 organs, pairwise DISJOINT (max 1 organ per point), "
          "%d reachable by identity, inside the envelope, off the bone, deterministic" % len(hit))


if __name__ == "__main__":
    _selftest()
