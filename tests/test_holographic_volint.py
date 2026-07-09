"""Tests for holographic_volint: the closed-form line integral over an FPE density field."""
import numpy as np
from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
from holographic.misc.holographic_volint import HolographicVolume, render_fog


def _vol(dim=1024, bw=2.0, seed=0):
    enc = VectorFunctionEncoder(3, dim=dim, bounds=[(-2, 2)] * 3, kernel="rbf", bandwidth=bw, seed=seed)
    return HolographicVolume.from_blobs(enc, [(-0.5, 0, 0), (0.7, 0.3, -0.4), (0.0, -0.5, 0.4)], [1.0, 0.8, 0.6])


def test_closed_form_matches_marched_line_integral():
    """The closed-form optical depth equals the marched integral of the field density (the whole point)."""
    vol = _vol()
    rng = np.random.default_rng(1)
    O = rng.uniform(-2, -1.8, (16, 3)); D = rng.normal(0, 1, (16, 3)); D /= np.linalg.norm(D, axis=1, keepdims=True)
    L = 3.5
    cf = vol.optical_depth(O, D, L)
    M = 200
    march = np.zeros(16)
    for m in range(M):
        t = (m + 0.5) / M * L
        march += np.clip(vol.density(O + t * D), 0.0, None) * (L / M)
    assert np.corrcoef(cf, march)[0, 1] > 0.99                 # exact shape
    assert np.abs(cf - march).mean() / (march.mean() + 1e-9) < 0.1   # calibrated to within 10%


def test_empty_space_is_known_without_marching():
    """A ray far from all field content reads ~0 optical depth -- emptiness is a property of the field, not marched."""
    vol = _vol()
    tau = vol.optical_depth(np.array([[6.0, 6.0, 6.0]]), np.array([[1.0, 0.0, 0.0]]), 1.0)[0]
    assert abs(tau) < 0.05


def test_optical_depth_is_vectorized_and_chunks():
    """Image-scale ray counts run in one call (chunked) without building giant arrays."""
    vol = _vol(dim=512)
    rng = np.random.default_rng(2)
    O = rng.uniform(-2, -1.8, (5000, 3)); D = rng.normal(0, 1, (5000, 3)); D /= np.linalg.norm(D, axis=1, keepdims=True)
    tau = vol.optical_depth(O, D, 3.0, chunk=1024)
    assert tau.shape == (5000,) and np.all(tau >= 0.0)


def test_render_fog_fades_with_distance():
    """Fog transmittance composites over a background: a far ray fogs more than a near one."""
    from holographic.rendering.holographic_render import Camera
    vol = _vol()
    cam = Camera(eye=(0, 0, 4.0), target=(0, 0, 0), fov_deg=45.0)
    bg = np.ones((32, 32, 3)) * np.array([0.2, 0.5, 0.9])
    near = render_fog(cam, 32, 32, vol, density_scale=0.1, background=bg, depth=np.full((32, 32), 1.0), max_dist=8.0)
    far = render_fog(cam, 32, 32, vol, density_scale=0.1, background=bg, depth=np.full((32, 32), 8.0), max_dist=8.0)
    # far rays accumulate more fog -> move further from the (blue) background toward the (grey) fog colour
    assert np.abs(far - bg).mean() > np.abs(near - bg).mean()


def test_fog_wired_into_render_scene():
    """render_scene(fog=volume) composites closed-form holographic fog in the real render path."""
    import numpy as np
    from holographic.simulation_and_physics.holographic_semantic import parse_description, render_scene
    from holographic.misc.holographic_volint import HolographicVolume
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
    from holographic.rendering.holographic_render import Camera
    objs = parse_description("a red ball beside a blue box")["objects"]
    cam = Camera(eye=(0.3, 1.4, 5.2), target=(0, 0.1, 0), fov_deg=48.0)
    enc = VectorFunctionEncoder(3, dim=1024, bounds=[(-4, 4)] * 3, kernel="rbf", bandwidth=1.2, seed=0)
    vol = HolographicVolume.from_blobs(enc, [(0, 0, -1), (1, 0, -2)], [1.0, 1.0])
    clear = render_scene(objs, cam, width=64, height=64, ss=1, dither=0.0)
    foggy = render_scene(objs, cam, width=64, height=64, ss=1, dither=0.0, fog=vol, fog_density=0.2)
    assert foggy.shape == clear.shape
    assert np.abs(foggy - clear).mean() > 1e-4                  # fog actually changed the frame
