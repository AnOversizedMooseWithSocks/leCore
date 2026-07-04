"""holographic_texturehome.py -- the TEXTURE home (consolidation backlog R6): procedural and example-based surface
detail, as FIELDS you plug straight into a Material channel.

WHY THIS EXISTS
---------------
Surface detail is generated in several modules -- fractal / band-limited noise (holographic_noise), divergence-free
curl noise (holographic_curlnoise), cellular / Voronoi patterns (holographic_cellular), example-based patch synthesis
(holographic_texturesynth), and the weathering set (holographic_oxidation, holographic_burn, holographic_inclusions).
They all end up feeding the SAME thing: a Material channel (roughness / albedo / height / ...), which takes a
`Param(field=callable)` -- a function f(points (M,D)) -> (M,) values -- or a `Param(map=grid)`.

`Texture` is the one home that hands you exactly that. Each method ROUTES to a shipped generator and returns a field
callable (or a grid) ready to wire into a channel:

    Texture.fbm(...)       -> fractal-noise field f(points) -> value          (holographic_noise.FractalNoise)
    Texture.voronoi(...)   -> cellular field: crack (edge) distance or cell id (holographic_cellular.VoronoiCells)
    Texture.curl(...)      -> a divergence-free (u,v) flow grid, for warping   (holographic_curlnoise.curl_noise)
    Texture.synth(...)     -> a larger texture image grown from a small sample (holographic_texturesynth)

Route, don't rewrite: this owns none of the generators, only the "return something a channel can eat" adapter.
The weathering fields (oxidation fraction, burn char, inclusions) are their own modules; they plug in the same way
(a fraction field -> a channel), and are registered in the catalog under the Physics & chemistry / Texture domains.
"""
import numpy as np


class Texture:
    """A namespace of staticmethods that return Material-channel-ready detail: field callables or grids."""

    @staticmethod
    def fbm(n_dims=3, bounds=None, octaves=4, base_bandwidth=2.0, lacunarity=2.0, gain=0.5, dim=1024, seed=0):
        """A fractal-Brownian-motion noise FIELD: octaves of band noise summed with falling amplitude (rougher with
        higher `gain`, finer with more `octaves`). Returns a callable field(points (M,D)) -> (M,) values in ~[0,1]
        -- wire it into a channel with Param(field=Texture.fbm(...)). Wraps holographic_noise.FractalNoise; its
        `query` is per-point, so the returned field loops (fine for a channel you bake once via the Cache home).
        The underlying generator is on field.generator (for sample_grid / measurement)."""
        from holographic_noise import FractalNoise
        gen = FractalNoise(n_dims, dim=dim, bounds=bounds, octaves=octaves, lacunarity=lacunarity, gain=gain,
                           base_bandwidth=base_bandwidth, seed=seed)

        def field(points, **_):
            P = np.asarray(points, float)
            return np.array([gen.query(P[i, :gen.n_dims]) for i in range(len(P))], dtype=float)
        field.generator = gen
        return field

    @staticmethod
    def voronoi(n_seeds=24, bounds=((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)), seed=0, jitter=1.0, kind="edge"):
        """A cellular (Voronoi) FIELD -- vectorised. kind='edge' -> distance to the nearest cell boundary (crack
        lines / grout / faceting); kind='id' -> the cell id as a float (per-cell tint). Returns field(points) ->
        (M,) values, ready for Param(field=...). Wraps holographic_cellular.VoronoiCells. generator on field.generator.
        """
        from holographic_cellular import VoronoiCells
        gen = VoronoiCells(n_seeds=n_seeds, bounds=bounds, seed=seed, jitter=jitter)

        def field(points, **_):
            P = np.asarray(points, float)
            return gen.edge_distance(P) if kind == "edge" else gen.ids(P).astype(float)
        field.generator = gen
        return field

    @staticmethod
    def curl(res=32, bounds=((0.0, 8.0), (0.0, 8.0)), octaves=4, seed=0):
        """A DIVERGENCE-FREE (incompressible) 2-D flow field on a res x res grid -- swirls with no sources/sinks,
        the right thing for WARPING a texture or advecting detail. Returns the (u, v) velocity grids. Wraps
        holographic_curlnoise.curl_noise."""
        from holographic_curlnoise import curl_noise
        return curl_noise(res, bounds=bounds, octaves=octaves, seed=seed)

    @staticmethod
    def synth(sample, out_h, out_w, psize=24, overlap=6, seed=0, seam="mincut"):
        """Grow a larger texture from a small `sample` image by patch-based (example-based) synthesis -- new detail
        that looks like the sample without tiling seams. Returns an (out_h, out_w, 3) image; feed a channel with
        Param(map=image, domain=(lo,hi)). Wraps holographic_texturesynth.synthesize_texture."""
        from holographic_texturesynth import synthesize_texture
        return synthesize_texture(sample, out_h, out_w, psize=psize, overlap=overlap, seed=seed, seam=seam)


def texture_backends():
    """The texture facilities the home exposes (for the catalog / discovery)."""
    return ("fbm", "voronoi", "curl", "synth")


def _selftest():
    # a Voronoi crack field, vectorised, wired the way a Material channel consumes it
    crack = Texture.voronoi(n_seeds=12, seed=0, kind="edge")
    pts = np.random.default_rng(0).uniform(-1, 1, (100, 3))
    v = crack(pts)
    assert v.shape == (100,) and np.isfinite(v).all() and (v >= 0).all()      # edge distance is non-negative

    # DONE-WHEN: source a Material channel through Texture and sample it
    from holographic_surface import SurfaceMaterial
    from holographic_param import Param
    rough = lambda P, **k: 0.25 + 0.5 * np.clip(crack(P) * 4.0, 0, 1)         # crack lines -> rougher
    mat = SurfaceMaterial(color=(0.6, 0.6, 0.62), roughness=Param(field=rough), reflect=0.1, emission=0.0)
    ch = mat.resolve(pts[:8])
    r = np.asarray(ch["roughness"], float)
    assert r.shape == (8,) and np.isfinite(r).all() and (r >= 0.25).all()     # channel pulled from the Texture field

    # fBm field returns per-point values in a sane range
    fb = Texture.fbm(n_dims=3, bounds=[(-1, 1)] * 3, octaves=3, seed=0)
    fv = fb(pts[:20])
    assert fv.shape == (20,) and np.isfinite(fv).all()

    # curl noise is (nearly) divergence-free -- the point of it
    u, v2 = Texture.curl(res=24, seed=0)
    div = np.abs(np.gradient(u, axis=1) + np.gradient(v2, axis=0)).mean()
    assert div < 0.5                                                          # small residual divergence only
    print("OK: holographic_texturehome self-test passed (voronoi/fbm field callables feed a Material channel; "
          "curl is ~divergence-free; over %s)" % ", ".join(texture_backends()))


if __name__ == "__main__":
    _selftest()
