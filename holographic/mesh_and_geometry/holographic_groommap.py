"""GROOM MAPS: per-vertex density and length attributes driving a groom, plus skin SSS.

Two things a mammal needs that a bounds-box groom cannot give: hair that grows only WHERE it
should and at DIFFERENT LENGTHS per region (a beard is not scalp hair), and skin that scatters
light instead of reading as painted plastic.

SOTA CHECK (searched 2026-08-16) -- this is the industry-standard workflow, not an invention.
Houdini's grooming pipeline paints a DENSITY attribute on the skin and overrides hair
generation with it ("paint out an attribute where you want to generate curves and then
override the density with that attribute in the guide groom sop"), and paints a SEPARATE
LENGTH attribute for the same purpose: "I planned to make the hairs around the nose and snout
shorter, and have the hairs at the base of the neck longer. The procedure was the same as
painting Density; but the control is 'Length' instead of Density." Beards, eyebrows and
eyelashes get their own overrides rather than sharing the scalp's. Sisir (2026) ships the same
controls -- masks, per-region grooms, dual-scattering hair over skin SSS. DiffLocks (2025)
argues for a smooth density MAP over a binary mask because it is "smoother and easier to
edit", which is why these are floats in [0,1] rather than booleans.

WHY THE BOUNDS BOX HAD TO GO. groom_hair roots strands anywhere inside an axis-aligned box,
so a beard box also catches the cheeks and neck, and a scalp box catches the forehead and
face. Every attempt to fix that by shrinking the box traded one wrong region for another --
the box is simply the wrong control. An attribute defined ON THE SURFACE is the right one, and
it is what every production tool uses.

RULE-0 AUDIT (2026-08-16): no per-vertex groom attribute exists. REUSED, not rebuilt --
groom_hair (still generates the strands; this filters and rescales them), mesh_geodesic (for
smooth region falloff), and tissue_pbr('skin'), which already carries the MEASURED red-shifted
scatter radius (1.0, 0.42, 0.28) and sss_weight 0.75 rather than a guessed tint.

KEPT NEGATIVE: this masks and rescales strands AFTER generation, so density is a filter rather
than a true sampling density -- ask for 4000 strands with a 0.3-coverage map and you get
roughly 1200, not 4000 concentrated in the region. A sampling-time implementation would be
better and is not what this does. Separately, render_hair has NO DEPTH TEST against the body,
so back-of-head strands still draw over the face; maps do not fix that, and nothing here
claims to.
"""

import numpy as np


def region_map(vertices, regions, default=0.0):
    """Build a per-vertex attribute in [0,1] from named box/sphere regions.

    `regions` is a list of {"kind": "box"|"sphere", "value": float, plus bounds}. Later
    entries win where they overlap, which is what lets a beard map be written as "the lower
    face, minus the lips" in two lines. This is the stand-in for a paint tool: the point is
    that the attribute lives ON THE SURFACE, not that it was authored with a brush."""
    V = np.asarray(vertices, float)
    a = np.full(len(V), float(default))
    for r in regions:
        if r.get("kind", "box") == "sphere":
            c = np.asarray(r["centre"], float)
            d = np.linalg.norm(V - c, axis=1)
            sel = d <= float(r["radius"])
        else:
            lo = np.asarray(r["lo"], float)
            hi = np.asarray(r["hi"], float)
            sel = np.all((V >= lo) & (V <= hi), axis=1)
        a[sel] = float(r["value"])
    return np.clip(a, 0.0, 1.0)


def smooth_map(vertices, faces, attr, mind, iters=6):
    """Blur an attribute over the surface so a region's edge is a gradient, not a cliff.

    A hard density edge reads as a shaved line; real hairlines fade. Uses simple umbrella
    averaging over mesh edges -- cheap, and enough for an attribute that is about to be
    thresholded anyway."""
    F = np.asarray(faces, int)
    a = np.asarray(attr, float).copy()
    n = len(np.asarray(vertices, float))
    for _ in range(int(iters)):
        acc = np.zeros(n)
        cnt = np.zeros(n)
        for i, j in ((0, 1), (1, 2), (2, 0)):
            np.add.at(acc, F[:, i], a[F[:, j]])
            np.add.at(cnt, F[:, i], 1.0)
            np.add.at(acc, F[:, j], a[F[:, i]])
            np.add.at(cnt, F[:, j], 1.0)
        a = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), a)
    return np.clip(a, 0.0, 1.0)


def groom_with_maps(strands, vertices, density, length, base_length, seed=0,
                    length_range=(0.25, 1.0)):
    """Filter and rescale a groom by per-vertex DENSITY and LENGTH attributes.

    Each strand is assigned the attribute of its nearest vertex. Density is a probability of
    keeping the strand (so a 0.0 region grows nothing and a 1.0 region grows everything);
    length scales the strand between `length_range` times `base_length`, which is exactly how
    a beard ends up short while scalp hair stays long -- ONE groom, TWO regions, as the
    production workflow does it.

    Deterministic given `seed`: the same map always yields the same groom, so a groom is
    reproducible rather than re-rolled every render."""
    V = np.asarray(vertices, float)
    dens = np.asarray(density, float)
    ln = np.asarray(length, float)
    rng = np.random.default_rng(int(seed))
    lo, hi = float(length_range[0]), float(length_range[1])
    kept = []
    for s in strands:
        root = np.asarray(s.root, float)
        i = int(np.argmin(np.linalg.norm(V - root, axis=1)))
        if rng.random() > dens[i]:
            continue
        scale = lo + (hi - lo) * float(ln[i])
        pts = np.asarray(s.points, float)
        s.points = pts[0] + (pts - pts[0]) * scale * float(base_length)
        kept.append(s)
    return kept


def sss_shade(base_rgb, ndl, thickness, sss_weight=0.75, sss_radius=(1.0, 0.42, 0.28)):
    """Wrapped-diffuse subsurface approximation for mammal skin.

    Skin is not Lambertian: light enters, scatters, and leaves nearby, so the terminator wraps
    PAST 90 degrees and the light that travels furthest comes back RED -- which is why ears and
    nostrils glow. Implemented as per-channel wrap, with the wrap width taken from
    tissue_pbr('skin')'s MEASURED scatter radius (1.0, 0.42, 0.28) rather than an invented
    tint: red wraps most, blue least.

    Not a diffusion profile and not path-traced -- a wrap term is the standard real-time
    approximation, and calling it that is more useful than implying more."""
    base = np.asarray(base_rgb, float)
    n = np.asarray(ndl, float)[..., None]
    r = np.asarray(sss_radius, float)[None, :]
    w = float(sss_weight) * np.asarray(thickness, float)[..., None]
    wrapped = np.clip((n + r * w) / (1.0 + r * w), 0.0, 1.0)
    return base * wrapped


def _selftest():
    """Regression trap: maps must actually separate two regions, and SSS must wrap past the
    Lambert terminator and do so REDDEST."""
    import lecore
    mind = lecore.UnifiedMind(dim=64, seed=0)
    sph = lambda P: np.linalg.norm(np.asarray(P, float), axis=1) - 1.0
    mesh = mind.mesh_from_sdf(sph, ((-1.3,) * 3, (1.3,) * 3), res=20, vectorized=True)
    V = np.asarray(mesh.vertices, float)

    # two disjoint regions with DIFFERENT lengths -- the beard/scalp case
    dens = region_map(V, [{"lo": (-2, 0.3, -2), "hi": (2, 2, 2), "value": 1.0},
                          {"lo": (-2, -2, -2), "hi": (2, -0.3, 2), "value": 1.0}])
    ln = region_map(V, [{"lo": (-2, 0.3, -2), "hi": (2, 2, 2), "value": 1.0},
                        {"lo": (-2, -2, -2), "hi": (2, -0.3, 2), "value": 0.0}])
    assert 0.2 < dens.mean() < 0.9, dens.mean()
    top = V[:, 1] > 0.5
    bot = V[:, 1] < -0.5
    assert ln[top].mean() > 0.9 and ln[bot].mean() < 0.1     # long up top, short below

    sm = smooth_map(V, mesh.faces, dens, mind, iters=4)
    assert sm.min() >= 0.0 and sm.max() <= 1.0
    assert np.std(sm) < np.std(dens)                          # the edge really did soften

    # SSS: wraps past the Lambert terminator, and reddest
    dark = sss_shade((0.6, 0.44, 0.36), np.array([0.0]), np.array([1.0]))
    assert dark[0, 0] > 0, "no wrap: the terminator is still Lambert"
    assert dark[0, 0] > dark[0, 2], "blue wrapped as far as red -- the radius is not applied"
    lit = sss_shade((0.6, 0.44, 0.36), np.array([1.0]), np.array([1.0]))
    assert lit[0, 0] >= dark[0, 0]                            # still monotone in N.L
    print("OK: holographic_groommap -- density/length maps separate two regions (long %.2f "
          "vs short %.2f), smoothing softens the edge, SSS wraps past the terminator and "
          "reddest (R %.3f > B %.3f)"
          % (ln[top].mean(), ln[bot].mean(), dark[0, 0], dark[0, 2]))


if __name__ == "__main__":
    _selftest()
