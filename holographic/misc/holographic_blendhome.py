"""holographic_blendhome.py -- the BLEND home (consolidation backlog H4): one place for "combine these into one",
promoting the canonical combine operations the engine already ships.

WHY THIS EXISTS
---------------
Combining shows up everywhere under different names: superposition (holographic_ai.bundle), spherical interpolation
(holographic_ai.slerp), the Frechet / Riemannian mean on the sphere (holographic_sphere.frechet_mean), front-to-back
alpha compositing (the splat / occlusion readout), and dict/scene MERGE with a conflict policy (the workspace
discipline). A soft weighted blend -- normalize(sum w_i v_i) -- is re-derived in blend skinning (blendpose), the
matter model (mixture), and elsewhere. One home names them all.

`Blend` promotes the canonical ops (route, don't rewrite):

    Blend.bundle(vectors, weights=None) -> superposition; weighted = normalize(sum w_i v_i) (the soft mixture)
    Blend.lerp(a, b, t)                 -> straight-line interpolation (1-t)a + t b
    Blend.slerp(a, b, t)                -> shortest-arc spherical interpolation (holographic_ai.slerp)
    Blend.mean(vectors, weights=None)   -> Frechet / Riemannian mean on the unit sphere (holographic_sphere)
    Blend.alpha_composite(colors, alphas) -> front-to-back OVER compositing (the splat / occlusion blend)
    Blend.merge(a, b, policy=...)       -> merge two dicts/records with a conflict policy (scene / workspace combine)

NOT everything called "blend" is this operation, and those are kept distinct on purpose (don't over-consolidate):
phasemorph's PHASE-domain shortest arc (angle interpolation on unit phasors), mixture's solvent-base weighted
DENSITY (a physical blend with a baseline term), and occlusion's alpha-composited bundle READOUT (a decompression,
not a forward blend) each stay in their own module.
"""
import numpy as np


class Blend:
    """A namespace of staticmethods over the engine's combine operations. Superpose / interpolate / average / merge."""

    @staticmethod
    def bundle(vectors, weights=None):
        """Superpose `vectors` into one (a set / a soft mixture). With `weights`, the WEIGHTED blend
        normalize(sum_i w_i v_i) -- the soft mixture blend skinning (blendpose) and the matter model use. Without,
        the plain equal-weight bundle (routes to holographic_ai.bundle). `vectors` is (m, dim), `weights` is (m,)."""
        V = np.asarray(vectors, float)
        if weights is None:
            from holographic.agents_and_reasoning.holographic_ai import bundle as _bundle
            return _bundle(V)
        w = np.asarray(weights, float)
        v = V.T @ w                                              # weighted superposition ...
        n = np.linalg.norm(v)
        return v / n if n > 0 else v                             # ... then renormalise (matches blend_pose exactly)

    @staticmethod
    def lerp(a, b, t):
        """Linear interpolation: the straight-line blend (1-t)*a + t*b. The chord, not the arc -- fine for small
        steps or when the vectors aren't meant to stay on a sphere."""
        return (1 - t) * np.asarray(a) + t * np.asarray(b)

    @staticmethod
    def slerp(a, b, t):
        """Spherical interpolation: travel the shortest ARC on the unit sphere from a to b (constant angular speed,
        stays on the sphere). Routes to holographic_ai.slerp."""
        from holographic.agents_and_reasoning.holographic_ai import slerp as _slerp
        return _slerp(a, b, t)

    @staticmethod
    def mean(vectors, weights=None, max_iters=12, tol=1e-8):
        """The Frechet / Riemannian mean on the unit sphere -- the average of DIRECTIONS that respects the manifold's
        curvature (not the chord midpoint of the raw vectors). Routes to holographic_sphere.frechet_mean."""
        from holographic.mesh_and_geometry.holographic_sphere import frechet_mean
        return frechet_mean(vectors, weights=weights, max_iters=max_iters, tol=tol)

    @staticmethod
    def alpha_composite(colors, alphas):
        """Front-to-back OVER compositing: C = sum_i c_i a_i prod_{j<i}(1 - a_j) -- the splat / occlusion blend.
        `colors` is (n, k), `alphas` is (n,), assumed already sorted front-to-back. Returns (composited (k,),
        accumulated alpha)."""
        colors = np.asarray(colors, float); alphas = np.asarray(alphas, float)
        out = np.zeros(colors.shape[1] if colors.ndim > 1 else (), dtype=float)
        trans = 1.0                                              # remaining transmittance in front of the current layer
        for c, a in zip(colors, alphas):
            out = out + trans * a * c
            trans *= (1.0 - a)
        return out, 1.0 - trans

    @staticmethod
    def merge(a, b, policy="prefer_a"):
        """Merge two dicts/records with a CONFLICT POLICY -- the scene / workspace combine discipline made explicit.
        Non-clashing keys always merge; on a key clash, 'prefer_a' keeps a's value, 'prefer_b' keeps b's, 'average'
        averages numeric values. Returns a new dict (inputs untouched)."""
        out = dict(a)
        for k, vb in b.items():
            if k not in out:
                out[k] = vb
            elif policy == "prefer_b":
                out[k] = vb
            elif policy == "average":
                try:
                    out[k] = 0.5 * (out[k] + vb)
                except TypeError:
                    pass                                        # non-numeric clash under 'average' -> keep a's
            # 'prefer_a' (default): keep out[k]
        return out


def blend_backends():
    """The blend facilities the home exposes (for the catalog / discovery)."""
    return ("bundle", "lerp", "slerp", "mean", "alpha_composite", "merge")


def _selftest():
    # weighted bundle == blendpose.blend_pose (the delegate it will route to), bit-identical
    from holographic.misc.holographic_blendpose import blend_pose
    targets = np.random.default_rng(0).standard_normal((3, 32))
    w = np.array([0.5, 0.3, 0.2])
    assert np.array_equal(Blend.bundle(targets, w), blend_pose(targets, w))

    # slerp == ai.slerp (the delegate morph_images will route to)
    from holographic.agents_and_reasoning.holographic_ai import slerp as _slerp
    a = np.zeros(4); a[0] = 1.0
    b = np.zeros(4); b[1] = 1.0
    assert np.array_equal(Blend.slerp(a, b, 0.5), _slerp(a, b, 0.5))

    # lerp is the straight chord; slerp stays on the sphere (longer than the chord midpoint's norm)
    mid_lerp = Blend.lerp(a, b, 0.5); mid_slerp = Blend.slerp(a, b, 0.5)
    assert np.linalg.norm(mid_slerp) > np.linalg.norm(mid_lerp) - 1e-9

    # alpha composite: an opaque front layer hides the back one
    col, acc = Blend.alpha_composite(np.array([[1., 0, 0], [0, 1., 0]]), np.array([1.0, 1.0]))
    assert np.allclose(col, [1., 0, 0]) and abs(acc - 1.0) < 1e-9

    # merge conflict policies
    assert Blend.merge({"a": 1, "b": 2}, {"b": 9, "c": 3}, policy="prefer_a") == {"a": 1, "b": 2, "c": 3}
    assert Blend.merge({"b": 2}, {"b": 9}, policy="prefer_b") == {"b": 9}
    assert Blend.merge({"b": 2.0}, {"b": 4.0}, policy="average") == {"b": 3.0}

    # Frechet mean of two unit vectors lands between them on the sphere
    m = Blend.mean(np.array([a, b]))
    assert abs(np.linalg.norm(m) - 1.0) < 1e-6
    print("OK: holographic_blendhome self-test passed (weighted bundle == blend_pose; slerp == ai.slerp; alpha "
          "composite; merge policies; frechet mean on the sphere; over %s)" % ", ".join(blend_backends()))


if __name__ == "__main__":
    _selftest()
