"""holographic_lightinghome.py -- the LIGHTING home (consolidation backlog R7): one place for the light TYPES and
the shade INTEGRAL, so render methods ask for lighting here instead of reaching into lights/prt/domecache directly.

WHY THIS EXISTS
---------------
"How much light reaches this surface point?" is answered in a few different modes, in a few different modules:
direct next-event estimation over the placed light types (holographic_lights.direct_lighting), a baked
precomputed-radiance-transfer relight under an environment (holographic_prt.shade_prt), and a dome/sky projected to
spherical harmonics (holographic_domecache.dome_light_sh). The light TYPES themselves
(point/directional/spot/area/dome/IES) live in holographic_lights. A render method that wants lighting shouldn't
have to know which module each piece is in.

`Lighting` is the one door. It RE-EXPORTS the light types (so `from holographic_lightinghome import Lighting,
RectLight` is all you need) and exposes the shade integral in each mode:

    Lighting.direct(sdf, P, N, V, albedo, metallic, roughness, lights, rng, ...)   # NEE from placed lights + shadows
    Lighting.prt(transfer, light_sh, albedo)                                       # relight a baked transfer under SH
    Lighting.environment_sh(dome, order, n)                                        # project a dome/sky to SH

Route, don't rewrite: every method delegates to the shipped function. The light integral (Cook-Torrance per light,
area sampling, shadow rays) stays in holographic_lights; PRT stays in holographic_prt. What is unified is the
ENTRY POINT the render methods call.
"""
# re-export the light types so this home is the single lighting import
from holographic_lights import (PointLight, DirectionalLight, AmbientLight, SpotLight, RectLight, DiskLight,
                                 SphereLight, DomeLight, MeshLight, IESLight, direct_lighting)

_LIGHT_TYPES = (PointLight, DirectionalLight, AmbientLight, SpotLight, RectLight, DiskLight, SphereLight,
                DomeLight, MeshLight, IESLight)


class Lighting:
    """A namespace of staticmethods over the engine's lighting. The shade integral, in each mode, plus the types."""

    @staticmethod
    def direct(sdf, P, N, V, albedo, metallic, roughness, lights, rng, shadow_eps=3e-3, area_samples=2,
               dome_samples=4):
        """The DIRECT light reaching shade points P from all `lights` -- next-event estimation with shadow rays,
        summed over the placed light types (each evaluated with Cook-Torrance). Area/dome lights are multi-sampled
        (area_samples / dome_samples). Routes to holographic_lights.direct_lighting -- the shade integral itself
        stays there."""
        return direct_lighting(sdf, P, N, V, albedo, metallic, roughness, lights, rng, shadow_eps=shadow_eps,
                               area_samples=area_samples, dome_samples=dome_samples)

    @staticmethod
    def prt(transfer, light_sh, albedo=None):
        """Relight a baked PRT `transfer` under an environment `light_sh` (spherical-harmonic coefficients) by a
        dot product -- soft environment lighting with self-shadowing, at lookup cost. Routes to
        holographic_prt.shade_prt."""
        from holographic_prt import shade_prt
        return shade_prt(transfer, light_sh, albedo=albedo)

    @staticmethod
    def environment_sh(dome, order=3, n=1024):
        """Project a dome / sky light to spherical harmonics (the environment `light_sh` PRT and the cached-dome
        pass consume). Routes to holographic_domecache.dome_light_sh."""
        from holographic_domecache import dome_light_sh
        return dome_light_sh(dome, order=order, n=n)

    @staticmethod
    def light_types():
        """The placed light-type classes this home re-exports (for discovery / construction)."""
        return _LIGHT_TYPES

    @staticmethod
    def split_cached(lights):
        """Partition `lights` into the ones the cached passes serve well (dome + soft area) vs the cheap hard lights
        (point/directional/spot/IES) the tracer samples per pixel. A convenience over holographic_lightcache /
        the is_dome flag, so a render method can route its lights without knowing the flags."""
        lights = list(lights or [])
        domes = [L for L in lights if getattr(L, "is_dome", False)]
        rest = [L for L in lights if not getattr(L, "is_dome", False)]
        from holographic_lightcache import split_soft_lights
        soft, hard = split_soft_lights(rest)
        return domes, soft, hard


def lighting_modes():
    """The lighting evaluation modes the home exposes (for the catalog / discovery)."""
    return ("direct", "prt", "environment_sh")


def _selftest():
    import numpy as np
    from holographic_sdf import sphere, box
    scene = sphere(0.5).smooth_union(box(2.0, 0.1, 2.0).translate((0, -0.55, 0)), k=0.03)
    P = np.array([[0.0, -0.4, 0.0], [0.3, -0.4, 0.2]])
    N = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    V = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]); V /= np.linalg.norm(V, axis=1, keepdims=True)
    alb = np.full((2, 3), 0.8); met = np.zeros(2); rough = np.full(2, 0.5)
    rng = np.random.default_rng(0)
    lights = [DirectionalLight(direction=(0.2, -1.0, -0.1), intensity=3.0),
              RectLight(position=(0.5, 1.5, 0.5), u_vec=(0.4, 0, 0), v_vec=(0, 0.3, 0.2), intensity=20.0)]

    # the direct integral routes bit-identically to holographic_lights.direct_lighting
    got = Lighting.direct(scene, P, N, V, alb, met, rough, lights, np.random.default_rng(0))
    ref = direct_lighting(scene, P, N, V, alb, met, rough, lights, np.random.default_rng(0))
    assert np.array_equal(got, ref) and np.isfinite(got).all()

    # split routes lights by how they're best served
    domes, soft, hard = Lighting.split_cached(lights + [DomeLight(intensity=1.0)])
    assert len(domes) == 1 and len(soft) == 1 and len(hard) == 1

    # PRT relight of a trivial transfer under a constant environment is finite
    from holographic_prt import precompute_transfer, project_env_to_sh
    T = precompute_transfer(scene, P, N, order=3, n=48)
    sh = project_env_to_sh(lambda d: np.tile([0.5, 0.6, 0.8], (len(d), 1)), order=3, n=256)
    lit = Lighting.prt(T, sh, alb)
    assert np.isfinite(lit).all()
    print("OK: holographic_lightinghome self-test passed (direct integral bit-identical to lights.direct_lighting; "
          "split routes lights; prt relights; %d light types, modes %s)"
          % (len(Lighting.light_types()), ", ".join(lighting_modes())))


if __name__ == "__main__":
    _selftest()
