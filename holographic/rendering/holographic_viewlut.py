"""holographic_viewlut.py -- VIEW LUT for view-dependent specular (fluids/matter backlog, performance item MC3).

MC2 baked the view-INDEPENDENT channels (a procedural texture -> a grid lookup). What is left is the genuinely
view-DEPENDENT part: specular reflectance depends on the VIEW direction (and the surface roughness), so it can't be
baked over position alone. The backlog's move is "add a dimension": bake over (position, view) -- here the reusable
slice is (view_cos, roughness), which is exactly the pre-integrated BRDF / split-sum LUT the offline-render world uses.

`brdf.directional_albedo(metallic, roughness, view_cos)` is a 4096-sample Monte Carlo integral of the BRDF over all
incoming light for a fixed view -- expensive to call per pixel. We evaluate it ONCE on a small (view_cos x roughness)
grid, then per pixel BILINEARLY look it up: the hairy hemispherical integral becomes a table read. One more axis
turned into a query, so more of the material is O(1).

KEPT NEGATIVE (loud): the table is per (metallic, base_color) -- a different metalness needs its own bake (or a third
axis). The baked values carry the Monte-Carlo noise of the bake (raise the sample count to reduce it) plus bilinear
interpolation error between grid nodes. It pays for itself only when you shade MANY pixels/frames (the bake costs
res*res integrals up front). Deterministic given the seeds.
"""
import numpy as np

from holographic.rendering.holographic_brdf import directional_albedo


class ViewLUT:
    """A pre-integrated specular reflectance table over (view_cos in [0,1], roughness in [rmin,1]). sample(view_cos,
    roughness) does a bilinear read -- O(1), no per-pixel hemisphere integral."""

    def __init__(self, grid, rough_min, rough_max):
        self.grid = np.asarray(grid, float)                  # (res_view, res_rough)
        self.res_v, self.res_r = self.grid.shape
        self.rough_min = float(rough_min)
        self.rough_max = float(rough_max)

    def sample(self, view_cos, roughness):
        """Bilinearly read the reflectance for per-pixel arrays (or scalars) of view_cos and roughness."""
        vc = np.clip(np.atleast_1d(np.asarray(view_cos, float)), 0.0, 1.0)
        rg = np.clip(np.atleast_1d(np.asarray(roughness, float)), self.rough_min, self.rough_max)
        fv = vc * (self.res_v - 1)                            # continuous grid coords on each axis
        fr = (rg - self.rough_min) / max(self.rough_max - self.rough_min, 1e-12) * (self.res_r - 1)
        v0 = np.floor(fv).astype(int); v1 = np.minimum(v0 + 1, self.res_v - 1)
        r0 = np.floor(fr).astype(int); r1 = np.minimum(r0 + 1, self.res_r - 1)
        wv = fv - v0; wr = fr - r0
        g = self.grid
        top = g[v0, r0] * (1 - wr) + g[v0, r1] * wr          # interpolate along roughness at each view row...
        bot = g[v1, r0] * (1 - wr) + g[v1, r1] * wr
        return top * (1 - wv) + bot * wv                     # ...then along view


def bake_view_lut(metallic=1.0, base_color=(1.0, 1.0, 1.0), res_view=16, res_rough=16,
                  rough_min=0.05, rough_max=1.0, samples=8192, seed=0):
    """Evaluate directional_albedo on a (view_cos x roughness) grid -> a ViewLUT. This is the ONE precompute (res^2
    hemisphere integrals); every downstream shade is a bilinear read."""
    view = np.linspace(0.0, 1.0, res_view)
    rough = np.linspace(rough_min, rough_max, res_rough)
    grid = np.zeros((res_view, res_rough))
    for i, vc in enumerate(view):
        for j, rg in enumerate(rough):
            vc_clamped = max(vc, 1e-3)                        # a grazing view_cos of 0 is degenerate; nudge it
            grid[i, j] = directional_albedo(metallic, rg, base_color=base_color,
                                            n=samples, view_cos=vc_clamped, seed=seed)
    return ViewLUT(grid, rough_min, rough_max)


def _selftest():
    """Bake the view LUT and confirm a lookup matches a fresh (high-sample) directional_albedo across the well-behaved
    roughness range, that reflectance falls as roughness rises (energy loss), and that the lookup is far cheaper than
    the integral it replaces. The very-smooth corner is a kept negative (see below)."""
    import time

    lut = bake_view_lut(metallic=1.0, res_view=24, res_rough=24, samples=16384, seed=0)

    # a lookup matches a fresh high-sample integral across the STABLE roughness range (>=0.3)
    worst = 0.0
    for vc in (0.3, 0.5, 0.7, 0.9):
        for rg in (0.3, 0.5, 0.7, 0.9):
            ref = directional_albedo(1.0, rg, n=65536, view_cos=vc, seed=1)      # accurate reference
            got = float(lut.sample(vc, rg)[0])
            worst = max(worst, abs(got - ref))
    assert worst < 0.08, worst                              # within the estimator's own Monte-Carlo variance

    # physically sensible: rougher -> more single-scatter energy loss -> lower directional albedo (head-on view)
    smooth = float(lut.sample(0.9, 0.3)[0])
    rough = float(lut.sample(0.9, 0.9)[0])
    assert smooth > rough, (smooth, rough)

    # the lookup is dramatically cheaper than the integral it replaces
    vcs = np.random.default_rng(0).uniform(0.1, 1.0, 4000)
    rgs = np.random.default_rng(1).uniform(0.3, 1.0, 4000)
    t0 = time.time(); lut.sample(vcs, rgs); t_lut = time.time() - t0
    t0 = time.time(); directional_albedo(1.0, 0.5, n=16384, view_cos=0.7, seed=0); t_one = time.time() - t0
    speedup = (t_one * 4000) / max(t_lut, 1e-9)             # 4000 integrals vs 4000 lookups

    print("holographic_viewlut selftest OK: pre-integrated specular LUT (view_cos x roughness) matches a fresh "
          "65k-sample directional_albedo to <%.3f across roughness>=0.3; reflectance falls with roughness (smooth "
          "%.2f > rough %.2f, the GGX energy loss); 4000 bilinear lookups replace 4000 hemisphere integrals ~%.0fx "
          "cheaper -- the view dimension turned into a query. KEPT NEGATIVE: very-smooth surfaces (roughness<0.2) are "
          "a HIGH-VARIANCE estimate under uniform-hemisphere sampling, so both the LUT and a fresh integral are noisy "
          "there (~0.2); the LUT faithfully stores directional_albedo -- importance sampling would fix the estimator, "
          "not the table" % (worst, smooth, rough, speedup))


if __name__ == "__main__":
    _selftest()
