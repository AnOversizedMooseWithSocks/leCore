"""holographic_shadowhome.py -- the SHADOW / VISIBILITY home (consolidation backlog R8): one place to ask "can light
(or the environment) reach this point?", with the engine's visibility strategies behind one door.

WHY THIS EXISTS
---------------
Visibility is recomputed in several render paths: the SDF SOFT shadow (Quilez penumbra march) and SDF AMBIENT
OCCLUSION both live in holographic_raymarch and are re-imported by holographic_raycoherence and holographic_semantic;
the HARD shadow-ray test (next-event estimation) is embedded in holographic_lights.direct_lighting; and PRT bakes
visibility into its transfer (holographic_prt). Same question, four spellings.

`Shadow` is the one door, with each spelling as a named strategy:

    Shadow.soft(sdf, P, Ldir, ...)               -> SDF soft shadow, [0,1] penumbra (raymarch.soft_shadow)
    Shadow.ambient_occlusion(sdf, P, N, ...)     -> SDF ambient occlusion, [0,1]   (raymarch.ambient_occlusion)
    Shadow.hard(sdf, P, N, light_dir, dist, ...) -> shadow-RAY visibility, 1 clear / 0 blocked (the NEE test)

Route, don't rewrite: the soft-shadow and AO marches stay in holographic_raymarch; PRT's baked visibility stays in
holographic_prt (cross-referenced, `Shadow.prt_visibility_note`). What is unified is the ENTRY POINT a render path
calls to test visibility.
"""
import numpy as np


class Shadow:
    """A namespace of staticmethods over the engine's visibility tests. Soft / AO / hard shadow-ray strategies."""

    @staticmethod
    def soft(sdf, P, Ldir, k=12.0, mint=0.02, maxt=12.0, steps=48):
        """SDF SOFT shadow (Quilez): march from P toward the light; the closest the ray passes to any surface,
        scaled by distance, is the penumbra. Returns per-point visibility in [0,1] (1 clear, 0 fully shadowed).
        Routes to holographic_raymarch.soft_shadow."""
        from holographic_raymarch import soft_shadow
        return soft_shadow(sdf, P, Ldir, k=k, mint=mint, maxt=maxt, steps=steps)

    @staticmethod
    def ambient_occlusion(sdf, P, N, samples=6, step=0.06, k=1.6):
        """SDF AMBIENT OCCLUSION (Quilez): march a short way along the normal; nearby geometry darkens the point.
        Returns per-point openness in [0,1] (1 open, 0 occluded). Routes to holographic_raymarch.ambient_occlusion.
        """
        from holographic_raymarch import ambient_occlusion
        return ambient_occlusion(sdf, P, N, samples=samples, step=step, k=k)

    @staticmethod
    def hard(sdf, P, N, light_dir, dist, shadow_eps=3e-3):
        """The HARD shadow-RAY visibility test (next-event estimation): 1 where the ray from P toward the light is
        clear (or its first hit is beyond the light), 0 where blocked. `light_dir` is the unit direction to the
        light per point (M,3); `dist` the distance to the light per point (M,). This is the test
        direct_lighting does per light, on its own so any render path can call it. Routes to sphere_trace."""
        from holographic_raymarch import sphere_trace
        P = np.asarray(P, float); N = np.asarray(N, float)
        O = P + N * shadow_eps                                    # start just off the surface (avoid self-hit)
        hit, t, _ = sphere_trace(sdf, O, np.asarray(light_dir, float))
        return ((~hit) | (t > np.asarray(dist, float) - shadow_eps)).astype(float)

    @staticmethod
    def prt_visibility_note():
        """PRT precomputes visibility (self-occlusion) INTO its transfer -- the fourth, baked strategy. There is no
        per-point call: build it with holographic_prt.precompute_transfer, then relight with Lighting.prt. Returned
        as a string so the home documents the strategy without importing prt eagerly."""
        return "PRT bakes visibility into the transfer: holographic_prt.precompute_transfer -> Lighting.prt"


def shadow_strategies():
    """The visibility strategies the home exposes (for the catalog / discovery)."""
    return ("soft", "ambient_occlusion", "hard", "prt(baked)")


def _selftest():
    from holographic_sdf import sphere, box
    # a ball above a floor whose top sits at y=0; floor points just above it (P + N*eps), as render paths offset them.
    scene = sphere(0.4).translate((0, 0.6, 0)).smooth_union(box(3.0, 0.1, 3.0).translate((0, -0.1, 0)), k=0.02)
    eps = 3e-3
    up = np.array([0.0, 1.0, 0.0])
    under = np.array([[0.0, eps, 0.0]])                          # floor point under the ball, lifted off the surface
    away = np.array([[1.4, eps, 0.0]])                           # floor point off to the side

    s_under = float(Shadow.soft(scene, under, up)[0])
    s_away = float(Shadow.soft(scene, away, up)[0])
    assert s_under < s_away                                       # under the ball is more shadowed

    # soft routes bit-identically to raymarch.soft_shadow
    from holographic_raymarch import soft_shadow, ambient_occlusion
    assert np.array_equal(Shadow.soft(scene, under, up), soft_shadow(scene, under, up))

    # AO: openness in [0,1], routes bit-identically
    Nfloor = np.array([[0.0, 1.0, 0.0]])
    ao = Shadow.ambient_occlusion(scene, away, Nfloor)
    assert np.array_equal(ao, ambient_occlusion(scene, away, Nfloor)) and (0.0 <= ao).all() and (ao <= 1.0).all()

    # hard shadow ray: the floor point under the ball is blocked from an overhead light, the side point is clear
    Ldir = np.array([[0.0, 1.0, 0.0]])
    dist = np.array([5.0])
    vis_under = Shadow.hard(scene, under, Nfloor, Ldir, dist)
    vis_away = Shadow.hard(scene, away, Nfloor, Ldir, dist)
    assert vis_under[0] == 0.0 and vis_away[0] == 1.0
    print("OK: holographic_shadowhome self-test passed (soft shadow darker under the occluder + routes bit-identical; "
          "AO in [0,1]; hard shadow-ray blocks under / clears beside; strategies %s)" % ", ".join(shadow_strategies()))


if __name__ == "__main__":
    _selftest()
