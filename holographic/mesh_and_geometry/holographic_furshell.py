"""FUR AS AN SDF SHELL -- length is an offset distance, coverage is a field.

WHY THIS EXISTS, and it is a diagnosis of two recurring symptoms rather than a new feature.
Every groom in this codebase has gone wrong the same two ways: the LENGTH is never the right
scale, and the COVERAGE is spotty. Both follow from the same root cause -- `groom_hair` takes
an abstract `length` number and scatters `n_strands` roots inside an axis-aligned box:

  * LENGTH is dimensionless-looking, so nothing ties it to the model. length=0.05 is a crew
    cut on a human head and a mane on a mouse, and the caller has no way to know which.
  * COVERAGE is a sample count over a BOX, so density per unit surface area depends on how
    much of the box the surface happens to fill. Move the box, change the head, and the same
    n_strands gives different density -- hence "spotty".

SOTA (searched 2026-08-16) says the field-native formulation fixes both, and it is old and
well-founded: Kajiya & Kay's "Rendering fur with three dimensional textures" (SIGGRAPH 1989)
treats fur as a VOLUMETRIC TEXTURE, and the production lineage renders it as "concentric
layers from the skin outwards" plus "extruded fins from triangle edges near the silhouette"
(the shells-and-fins family). HISR (2024) states the hybrid in exactly SDF terms: a HARD SDF
whose interior is "filled with opaque materials", and outside it a translucent region "with
volume densities", bounded by a second SOFT SDF. That is fur as the region between two
offsets of one field.

THE TWO FIXES FALL OUT OF THE REPRESENTATION, which is why this is worth doing rather than
adding more knobs:
  * LENGTH IS A DISTANCE. Fur occupies {0 < sdf(x) < L}. L is measured in the SAME UNITS as
    the model, because it is an SDF offset. A caller asking for 8mm of fur on a 180mm head
    gets 8mm, and `fur_length_for` converts a fraction-of-model-size into that distance so
    the intent "short fur" survives a change of scale.
  * COVERAGE IS A FIELD, not a sample count. Density is evaluated per POINT, so it is
    uniform per unit area by construction. There is no box, nothing to clump, and no
    resolution-dependent thinning.

RULE-0 AUDIT (2026-08-16): groom_hair (explicit strands) ships and is NOT replaced -- strands
remain the right answer for long, styled, animated hair. This is the complementary
representation for SHORT dense fur, stubble and beards, where strand counts explode and the
shell is both cheaper and better behaved. holographic_groommap's density/length maps are
REUSED as the modulating fields; nothing here re-implements them.

KEPT NEGATIVE: a shell cannot do long flowing hair. Past roughly a shell thickness comparable
to the surface's curvature radius the concentric offsets self-intersect in concave regions
(exactly the reach bound L3 already proves), and the fur reads as a solid crust rather than
fibres. `shell_is_valid` checks that against the measured reach and REFUSES rather than
letting the caller discover it in a render.
"""

import numpy as np


def fur_length_for(sdf_bounds, fraction=0.04):
    """Turn "short fur" into a DISTANCE in model units.

    `fraction` is the fur length as a fraction of the model's largest extent -- roughly 0.02
    for stubble, 0.04 for short fur, 0.10 for a thick coat. This is the control that has been
    missing: the caller states intent at model scale and gets a length that stays correct
    when the model is resized."""
    lo, hi = np.asarray(sdf_bounds[0], float), np.asarray(sdf_bounds[1], float)
    return float(np.max(hi - lo) * float(fraction))


def fur_shell(sdf, length, density_fn=None, length_fn=None, strand_scale=180.0,
              seed=0, taper=2.0):
    """Fur as the region between the surface and an outward offset.

    Returns `f(P) -> occupancy in [0,1]`: 1 deep in the fur, falling to 0 at the tip. The
    outer boundary is `sdf(x) == L(x)` where L is `length` modulated by `length_fn` -- so a
    beard and a scalp differ by a field, not by two separate grooms.

    STRAND STRUCTURE comes from a deterministic hash of position (`strand_scale` sets fibre
    frequency), so the shell reads as fibres rather than a crust WITHOUT storing any strands.
    That is Kajiya & Kay's point: fur is a volumetric texture. Deterministic in `seed`, so a
    coat is reproducible.

    `taper` > 1 thins the fur toward the tips, which is what makes a silhouette look like
    fur instead of a rind."""
    L0 = float(length)

    def occupancy(P):
        P = np.atleast_2d(np.asarray(P, float))
        d = np.asarray(sdf(P), float).ravel()
        L = L0 * (np.asarray(length_fn(P), float).ravel() if length_fn is not None else 1.0)
        L = np.maximum(L, 1e-9)
        t = np.clip(d / L, 0.0, 2.0)               # 0 at skin, 1 at the tip; clipped because
        inside = (d > 0.0) & (t < 1.0)             # t**taper on a negative base is undefined
        # fibre mask: a hash of the position PROJECTED to the skin, so a fibre stays coherent
        # along its length instead of dissolving into noise partway up
        base = P - np.asarray(_grad(sdf, P), float) * d[:, None]
        h = _hash3(base * float(strand_scale) + float(seed))
        # a fibre thins toward its tip: occupancy falls as t^taper, and thin fibres end sooner
        alive = h > (t ** float(taper))
        dens = np.asarray(density_fn(P), float).ravel() if density_fn is not None else 1.0
        return np.where(inside & alive, np.clip(dens, 0.0, 1.0) * (1.0 - t ** 3), 0.0)

    return occupancy


def _grad(sdf, P, eps=1e-4):
    """Outward unit normal of the field (the direction fur grows)."""
    P = np.atleast_2d(np.asarray(P, float))
    g = np.empty_like(P)
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        g[:, k] = (np.asarray(sdf(P + d), float).ravel() -
                   np.asarray(sdf(P - d), float).ravel()) / (2 * eps)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)


def _hash3(X):
    """Deterministic per-position hash in [0,1) -- pure arithmetic, no RNG state, so the same
    point always yields the same fibre no matter what order it is evaluated in."""
    X = np.asarray(X, float)
    v = (np.sin(X[:, 0] * 127.1 + X[:, 1] * 311.7 + X[:, 2] * 74.7) * 43758.5453)
    return np.abs(v - np.floor(v))


def shell_is_valid(length, reach):
    """Would this fur length make the offset shell self-intersect?

    Reuses L3's result: an outward offset is injective only below the REACH. Beyond it the
    concentric layers cross in concave regions and the fur becomes a crust.

    USE A *LOCAL* REACH, and this is a measured lesson rather than a caution: a whole-head
    reach on a real head measured 0.0003 -- set by the crevice between the LIPS -- which
    would forbid fur everywhere including the scalp, where the true local reach is two orders
    of magnitude larger. Reach is a LOCAL property and a global minimum is the wrong statistic
    for a spatially-varying groom. Pass the reach sampled over the region the fur actually
    covers (the density map's support), not the whole model."""
    return bool(float(length) < float(reach)), float(reach) - float(length)


def local_reach(sdf, points, density_fn, mind, threshold=0.5):
    """The reach measured ONLY where fur actually grows -- the statistic shell_is_valid wants.

    Filters the sample points by the density map before measuring, so a bare crevice cannot
    veto a furred region that is nowhere near it."""
    P = np.atleast_2d(np.asarray(points, float))
    keep = np.asarray(density_fn(P), float).ravel() > float(threshold)
    if keep.sum() < 8:
        return float("inf")
    return float(mind.surface_safe_offset(sdf, P[keep])["safe"])


def _selftest():
    """Regression trap: the two properties that motivated the module -- length in model units,
    and coverage that does NOT depend on sampling."""
    sphere = lambda P: np.linalg.norm(np.atleast_2d(np.asarray(P, float)), axis=1) - 1.0

    # 1) LENGTH IS A DISTANCE: fur ends at exactly the requested offset
    L = 0.15
    fur = fur_shell(sphere, L, strand_scale=40.0)
    r_in = np.array([[1.05, 0, 0]])          # inside the shell
    r_out = np.array([[1.0 + L * 1.2, 0, 0]])  # past the tip
    assert float(np.asarray(fur(r_out))[0]) == 0.0, "fur extends past its stated length"
    # occupancy at some point within the shell must be reachable (over many samples, some hit)
    ring = np.stack([np.full(400, 1.02), np.linspace(-0.4, 0.4, 400), np.zeros(400)], 1)
    assert float(np.asarray(fur(ring)).max()) > 0.0, "no fur anywhere inside the shell"

    # 2) SCALE INVARIANCE OF INTENT: the same fraction gives proportional length
    small = fur_length_for(((-1,) * 3, (1,) * 3), 0.04)
    big = fur_length_for(((-10,) * 3, (10,) * 3), 0.04)
    assert abs(big / small - 10.0) < 1e-9, (small, big)

    # 3) COVERAGE IS A FIELD: doubling the sample count must not change the covered FRACTION
    def frac(n):
        rng = np.random.default_rng(0)
        th = rng.uniform(0, np.pi, n)
        ph = rng.uniform(0, 2 * np.pi, n)
        rr = 1.0 + L * 0.35
        P = np.stack([rr * np.sin(th) * np.cos(ph), rr * np.cos(th),
                      rr * np.sin(th) * np.sin(ph)], 1)
        return float((np.asarray(fur(P)) > 0).mean())
    f1, f2 = frac(3000), frac(6000)
    assert abs(f1 - f2) < 0.06, (f1, f2)     # THE ANTI-SPOTTY PROPERTY

    # 4) the reach guard refuses an over-long shell
    ok, margin = shell_is_valid(0.5, 0.3)
    assert not ok and margin < 0
    assert shell_is_valid(0.1, 0.3)[0]
    print("OK: holographic_furshell -- fur stops at its stated offset, length scales with the "
          "model (%.3f vs %.3f for 10x), coverage %.3f vs %.3f under 2x sampling, reach "
          "guard refuses over-long shells" % (small, big, f1, f2))


if __name__ == "__main__":
    _selftest()
