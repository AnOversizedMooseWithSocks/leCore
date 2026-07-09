"""Tests for holographic_lights -- the full light set + next-event estimation (direct light sampling)."""
import numpy as np
from holographic.rendering.holographic_lights import PointLight, DirectionalLight, AmbientLight, SpotLight, RectLight, SphereLight, MeshLight, IESLight, load_ies, direct_lighting
from holographic.mesh_and_geometry.holographic_sdf import sphere


def test_point_light_inverse_square_falloff():
    pl = PointLight(position=(0, 0, 0), intensity=1.0)
    rng = np.random.default_rng(0)
    _, _, near = pl.sample(np.array([[0.0, 0.0, 1.0]]), rng)
    _, _, far = pl.sample(np.array([[0.0, 0.0, 2.0]]), rng)
    assert abs(float(near.mean()) / float(far.mean()) - 4.0) < 0.1     # twice as far -> a quarter as bright


def test_directional_light_uniform_no_falloff():
    dl = DirectionalLight(direction=(0, 1, 0), intensity=2.0)
    rng = np.random.default_rng(0)
    L, dist, rad = dl.sample(np.random.default_rng(1).standard_normal((6, 3)), rng)
    assert np.allclose(L, L[0]) and np.allclose(rad, rad[0]) and np.all(dist > 1e6)


def test_nee_lights_clear_point_and_shadows_occluded():
    scene = sphere(0.4)
    light = PointLight(position=(0, 3, 0), intensity=20.0)
    rng = np.random.default_rng(0)
    N = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]); V = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    P = np.array([[1.5, 0.0, 0.0], [0.0, -0.5, 0.0]])
    alb = np.full((2, 3), 0.8)
    lit = direct_lighting(scene, P, N, V, alb, np.zeros(2), np.full(2, 0.5), [light], rng)
    assert lit[0].sum() > 1e-3 and lit[1].sum() < lit[0].sum() * 0.2


def test_backfacing_point_receives_no_light():
    scene = sphere(0.4); light = PointLight(position=(0, 3, 0), intensity=20.0); rng = np.random.default_rng(0)
    N = np.array([[0.0, -1.0, 0.0]]); V = np.array([[0.0, 0.0, 1.0]]); P = np.array([[2.0, 0.0, 0.0]])
    lit = direct_lighting(scene, P, N, V, np.full((1, 3), 0.8), np.zeros(1), np.full(1, 0.5), [light], rng)
    assert lit.sum() < 1e-6


def test_no_lights_returns_zero():
    scene = sphere(0.4); rng = np.random.default_rng(0)
    P = np.zeros((3, 3)); N = np.tile([0, 1.0, 0], (3, 1)); V = np.tile([0, 0, 1.0], (3, 1))
    out = direct_lighting(scene, P, N, V, np.full((3, 3), 0.8), np.zeros(3), np.full(3, 0.5), [], rng)
    assert np.all(out == 0.0)


def test_ambient_fills_without_shadow():
    # ambient lights a point regardless of occlusion (it comes from everywhere) -> albedo * colour * intensity
    rng = np.random.default_rng(0)
    amb = AmbientLight(color=(1, 1, 1), intensity=0.2)
    P = np.array([[0.0, -0.5, 0.0]]); N = np.array([[0.0, -1.0, 0.0]]); V = np.array([[0.0, 0.0, 1.0]])
    out = direct_lighting(sphere(0.4), P, N, V, np.full((1, 3), 0.5), np.zeros(1), np.full(1, 0.5), [amb], rng)
    assert np.allclose(out, 0.5 * 0.2)                                 # albedo(0.5) * colour(1) * intensity(0.2)


def test_spot_cone_bright_on_axis_dark_outside():
    rng = np.random.default_rng(0)
    spot = SpotLight(position=(0, 3, 0), direction=(0, -1, 0), inner_deg=10, outer_deg=20, intensity=40.0)
    _, _, on = spot.sample(np.array([[0.0, 0.0, 0.0]]), rng)
    _, _, off = spot.sample(np.array([[3.0, 0.0, 0.0]]), rng)
    assert on.max() > 1e-3 and off.max() < 1e-6


def test_spot_gobo_projects_pattern():
    # a gobo that only passes the +u half of the cone darkens the -u side
    rng = np.random.default_rng(0)
    def half(uv):
        return (uv[:, 0] > 0).astype(float)
    spot = SpotLight(position=(0, 3, 0), direction=(0, -1, 0), inner_deg=30, outer_deg=40, intensity=40.0, gobo=half)
    _, _, passed = spot.sample(np.array([[0.4, 0.0, 0.0]]), rng)
    _, _, blocked = spot.sample(np.array([[-0.4, 0.0, 0.0]]), rng)
    assert passed.max() > blocked.max()


def test_rect_light_is_one_sided():
    rng = np.random.default_rng(0)
    rect = RectLight(position=(0, 3, 0), u_vec=(1, 0, 0), v_vec=(0, 0, 1), intensity=30.0)  # emits along +/-y
    _, _, below = rect.sample(np.array([[0.0, 0.0, 0.0]]), rng)        # in front of the emitting face -> lit
    _, _, above = rect.sample(np.array([[0.0, 6.0, 0.0]]), rng)       # behind the panel -> dark
    assert below.max() > 1e-4 and above.max() < 1e-6


def test_sphere_light_samples_vary():
    sl = SphereLight(position=(0, 3, 0), radius=1.0); rng = np.random.default_rng(0)
    dirs = np.array([sl.sample(np.array([[0.0, 0.0, 0.0]]), rng)[0][0] for _ in range(60)])
    assert dirs.std(axis=0).mean() > 1e-3


def test_mesh_light_emits_finite():
    rng = np.random.default_rng(0)
    verts = np.array([[-1, 3, -1.0], [1, 3, -1], [1, 3, 1], [-1, 3, 1]]); faces = np.array([[0, 1, 2], [0, 2, 3]])
    ml = MeshLight(verts, faces, intensity=20.0)
    L, dist, rad = ml.sample(np.zeros((4, 3)), rng)
    assert L.shape == (4, 3) and np.isfinite(rad).all()
    # samples land inside the panel's xz extent (y == 3)
    assert np.all(np.abs(ml._area.sum() - 4.0) < 1e-6)                # two unit-ish triangles -> total area 4


def test_ies_profile_shapes_the_beam():
    rng = np.random.default_rng(0)
    prof = np.cos(np.linspace(0, np.pi / 2, 20)) ** 4                 # narrow downlight
    ies = IESLight(position=(0, 3, 0), direction=(0, -1, 0), profile=prof, profile_max_deg=90, intensity=40.0)
    _, _, below = ies.sample(np.array([[0.0, 0.0, 0.0]]), rng)        # on-axis -> peak
    _, _, side = ies.sample(np.array([[3.0, 2.9, 0.0]]), rng)        # off-axis -> dim
    assert below.max() > side.max()


def test_load_ies_parses_a_minimal_file():
    # a tiny synthetic IES file: 3 vertical angles (0,45,90), 1 horizontal plane, candela 100/50/0
    text = (
        "IESNA:LM-63-2002\n"
        "TILT=NONE\n"
        "1 1000 1 3 1 1 2 0 0 0\n"
        "1 1 100\n"
        "0 45 90\n"
        "0\n"
        "100 50 0\n"
    )
    candela, maxdeg = load_ies(text)
    assert list(candela) == [100.0, 50.0, 0.0] and maxdeg == 90.0
    # and it drives an IESLight
    ies = IESLight(profile=candela, profile_max_deg=maxdeg, direction=(0, -1, 0), position=(0, 3, 0), intensity=10.0)
    rng = np.random.default_rng(0)
    _, _, r = ies.sample(np.array([[0.0, 0.0, 0.0]]), rng)
    assert np.isfinite(r).all()


def test_color_field_varies_across_scene():
    # colour may be a FIELD: a callable f(P)->rgb that varies the lamp colour over space
    rng = np.random.default_rng(0)
    def red_right(P):
        x = np.atleast_2d(P)[:, 0]
        return np.stack([np.clip(x, 0, 1), np.zeros_like(x), np.zeros_like(x)], axis=1)
    pl = PointLight(position=(0, 3, 0), color=red_right, intensity=1.0)
    _, _, rad = pl.sample(np.array([[0.2, 0, 0], [0.9, 0, 0]]), rng)
    assert rad[1, 0] > rad[0, 0]                                      # redder to the right


def test_intensity_field_varies_across_scene():
    rng = np.random.default_rng(0)
    def bright_far(P):
        return np.abs(np.atleast_2d(P)[:, 0]) * 10.0                  # brighter the farther out in x
    pl = PointLight(position=(0, 3, 0), color=(1, 1, 1), intensity=bright_far)
    _, _, rad = pl.sample(np.array([[0.1, 0, 0], [1.0, 0, 0]]), rng)
    assert rad[1].sum() > rad[0].sum()


def test_original_three_lights_backward_compatible():
    # point/directional/sphere keep their old constructor + sample contract (the tracer depends on it)
    rng = np.random.default_rng(0)
    for lt in (PointLight(), DirectionalLight(), SphereLight()):
        L, dist, rad = lt.sample(np.zeros((5, 3)), rng)
        assert L.shape == (5, 3) and dist.shape == (5,) and rad.shape == (5, 3)


def test_disk_light_is_one_sided_and_round():
    rng = np.random.default_rng(0)
    from holographic.rendering.holographic_lights import DiskLight
    disk = DiskLight(position=(0, 3, 0), normal=(0, -1, 0), radius=0.5, intensity=30.0)
    _, _, below = disk.sample(np.array([[0.0, 0.0, 0.0]]), rng)        # in front of the face -> lit
    _, _, above = disk.sample(np.array([[0.0, 6.0, 0.0]]), rng)       # behind -> dark (one-sided)
    assert below.max() > 1e-4 and above.max() < 1e-6
    # sampled points stay within the disk radius of the centre
    P0 = np.array([[0.0, 0.0, 0.0]])
    r = []
    for _ in range(200):
        L, dist, _ = disk.sample(P0, rng)
        pt = P0[0] + L[0] * dist[0]                                   # sampled light point = P + L*dist
        r.append(np.linalg.norm(pt - disk.position))
    assert max(r) <= 0.5 + 1e-6                                       # all within the radius


def test_dome_light_is_shadowed_ambient():
    from holographic.rendering.holographic_lights import DomeLight
    from holographic.mesh_and_geometry.holographic_sdf import box
    rng = np.random.default_rng(0)
    dome = DomeLight(color=(0.6, 0.7, 0.9), intensity=1.0)
    V = np.array([[0.0, 0.0, 1.0]])
    # an open point sees the sky
    lit_open = direct_lighting(sphere(0.3), np.array([[3.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]), V,
                               np.full((1, 3), 0.8), np.zeros(1), np.full(1, 0.5), [dome], rng, dome_samples=32)
    # a point right under a big wall sees much less sky (ambient occlusion)
    wall = box(6.0, 6.0, 0.3).translate((0, 0, -0.2))
    lit_occl = direct_lighting(wall, np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]]), V,
                               np.full((1, 3), 0.8), np.zeros(1), np.full(1, 0.5), [dome], rng, dome_samples=32)
    assert lit_open.sum() > 1e-3 and lit_occl.sum() < lit_open.sum()


def test_dome_color_field_by_direction():
    # the dome colour may be a field of DIRECTION (a gradient sky / environment map)
    from holographic.rendering.holographic_lights import DomeLight
    def sky_gradient(dirs):
        up = np.clip(dirs[:, 1], 0, 1)                                # bluer overhead, warmer at the horizon
        return np.stack([0.8 - 0.4 * up, 0.7 - 0.1 * up, 0.5 + 0.4 * up], axis=1)
    dome = DomeLight(color=sky_gradient, intensity=1.0)
    up_rad = dome.radiance(np.array([[0.0, 1.0, 0.0]]))              # straight up -> bluer
    horiz_rad = dome.radiance(np.array([[1.0, 0.0, 0.0]]))          # horizon -> warmer
    assert up_rad[0, 2] > horiz_rad[0, 2]                            # more blue overhead


def test_area_light_multisampling_reduces_variance():
    # in a penumbra, averaging MORE area-light samples lowers the run-to-run variance of the estimate. NOTE: a
    # SINGLE sample is degenerate -- it always takes the light centre, a biased HARD shadow with zero variance --
    # so the meaningful test compares two MULTI-sample counts, both unbiased, where more samples => less variance.
    from holographic.rendering.holographic_lights import RectLight
    rect = RectLight(position=(0, 3, 0), u_vec=(0.6, 0, 0), v_vec=(0, 0, 0.6), intensity=40.0)
    occl = sphere(0.5).translate((0.0, 1.5, 0.0))
    P = np.array([[0.35, 0.0, 0.0]]); N = np.array([[0.0, 1.0, 0.0]]); V = np.array([[0.0, 0.0, 1.0]])
    a4 = np.array([direct_lighting(occl, P, N, V, np.full((1, 3), 0.8), np.zeros(1), np.full(1, 0.5),
                                   [rect], np.random.default_rng(s), area_samples=4).sum() for s in range(24)])
    a32 = np.array([direct_lighting(occl, P, N, V, np.full((1, 3), 0.8), np.zeros(1), np.full(1, 0.5),
                                    [rect], np.random.default_rng(s), area_samples=32).sum() for s in range(24)])
    assert a32.std() <= a4.std()
