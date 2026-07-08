"""Tests for holographic_scene_render -- rendering the canonical Scene document (backlog H7)."""
import numpy as np
from holographic.scene_and_pipeline.holographic_scene_doc import Scene
from holographic.rendering.holographic_scene_render import scene_to_render, render_scene_document, _place, _decompose
from holographic.mesh_and_geometry.holographic_sdf import sphere, plane, box


def _T(t, s=1.0):
    M = np.eye(4) * 1.0
    M[:3, :3] = np.eye(3) * s
    M[3, 3] = 1.0
    M[:3, 3] = t
    return M


def _scene():
    sc = Scene(seed=0)
    sc.add(name="floor", geometry=plane(-0.9), material="matte_white")
    sc.add(name="red", geometry=sphere(0.5), transform=_T((-0.8, 0, 0)), material="plastic_red")
    sc.add(name="gold", geometry=sphere(0.5), transform=_T((0.8, 0, 0)), material="gold")
    return sc


def test_flatten_sdf_is_nearest_object():
    sc = _scene()
    sdf, _ = scene_to_render(sc)
    P = np.array([[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, -0.9, 0.0], [0.0, 3.0, 0.0]])
    d = sdf.eval(P)
    assert d[0] < 0.02 and d[1] < 0.02 and abs(d[2]) < 0.05    # on the red / gold / floor surfaces
    assert d[3] > 2.0                                          # far above everything -> large positive distance


def test_material_fn_picks_the_owning_objects_material():
    sc = _scene()
    _, material_fn = scene_to_render(sc)
    P = np.array([[-0.8 - 0.5, 0.0, 0.0], [0.8 + 0.5, 0.0, 0.0]])  # points ON the red / gold sphere surfaces
    alb, met, rough, emis, ior = material_fn(P)
    assert met[1] == 1.0 and met[0] == 0.0                     # gold is metal, red plastic is not
    assert alb[0][0] > alb[0][2]                               # red point reads reddish (R > B)
    assert alb.shape == (2, 3) and ior.shape == (2,)


def test_transform_places_geometry():
    # an object's transform (translation + uniform scale) actually moves/sizes its SDF
    g = sphere(0.5)
    placed = _place(g, _T((2.0, 0.0, 0.0)))                    # translate +2 in x
    assert placed.eval(np.array([[2.0, 0.0, 0.0]]))[0] < 0.01  # centre of the moved sphere is on the surface? no:
    assert placed.eval(np.array([[2.0, 0.0, 0.0]]))[0] < 0.0   # inside the moved sphere (distance negative)
    assert placed.eval(np.array([[0.0, 0.0, 0.0]]))[0] > 0.0   # the origin is now OUTSIDE it
    t, s = _decompose(_T((1.0, 2.0, 3.0), s=2.0))
    assert np.allclose(t, [1, 2, 3]) and abs(s - 2.0) < 1e-9


def test_empty_scene_raises():
    sc = Scene(seed=0)
    sc.add(name="cam_only")                                    # an object with no geometry -> nothing to render
    try:
        scene_to_render(sc); assert False, "empty scene should raise"
    except ValueError:
        pass


def test_render_scene_document_end_to_end():
    # the whole path: document -> flatten -> render a small image, deterministically
    sc = _scene()
    class Cam:
        eye = np.array([0.0, 0.4, 3.2])
        def ray_dirs(self, w, h, jitter=None):
            ys, xs = np.mgrid[0:h, 0:w]
            jx, jy = (0.0, 0.0) if jitter is None else (jitter[0], jitter[1])
            u = ((xs + jx) / (w - 1) - 0.5) * 1.2; v = -((ys + jy) / (h - 1) - 0.5) * 1.2
            d = np.stack([u, v, -np.ones_like(u)], -1); return self.eye, d / np.linalg.norm(d, axis=-1, keepdims=True)
    img = render_scene_document(sc, Cam(), width=40, height=30, quality="draft", max_bounce=3, seed=0)
    assert img.shape == (30, 40, 3) and np.isfinite(img).all() and img.min() >= 0
    img2 = render_scene_document(sc, Cam(), width=40, height=30, quality="draft", max_bounce=3, seed=0)
    assert np.array_equal(img, img2)                           # deterministic


def test_albedo_socket_drives_per_point_colour():
    # backlog H2: an object can carry a spatially-varying albedo SOCKET (crystal grains / inclusions) that the
    # renderer samples per hit, instead of the material's flat base colour.
    import numpy as np
    from holographic.scene_and_pipeline.holographic_scene_doc import Scene
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    from holographic.simulation_and_physics.holographic_cellular import VoronoiCells, cell_albedo
    cells = VoronoiCells(n_seeds=20, bounds=((-1.2, -1.2, -1.2), (1.2, 1.2, 1.2)), seed=0, jitter=1.0)
    socket = cell_albedo(cells, base=(0.4, 0.5, 0.75), spread=0.25, seed=0)
    sc = Scene(seed=0)
    sc.add(name="crystal", geometry=sphere(0.8), material="matte_white", overrides={"albedo_socket": socket})
    _, material_fn = scene_to_render(sc)
    pts = np.array([[0.8, 0, 0], [0, 0.8, 0], [0, 0, 0.8], [-0.8, 0, 0], [0, -0.8, 0]])
    alb = material_fn(pts)[0]
    assert alb.shape == (5, 3)
    assert float(alb.std(0).mean()) > 0.01                       # colour VARIES across points (different cells)


def test_no_socket_uses_flat_material_colour():
    # without a socket, albedo is the material's flat base colour (constant across the object)
    import numpy as np
    from holographic.scene_and_pipeline.holographic_scene_doc import Scene
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    sc = Scene(seed=0)
    sc.add(name="plain", geometry=sphere(0.8), material="plastic_red")
    _, material_fn = scene_to_render(sc)
    pts = np.array([[0.8, 0, 0], [0, 0.8, 0], [-0.8, 0, 0]])
    alb = material_fn(pts)[0]
    assert np.allclose(alb, alb[0])                              # all points share the one flat colour


def test_sss_depth_sigma_thread_through():
    # the SSS tuning knobs reach the tracer through the scene-document path (they were stuck at defaults)
    import numpy as np
    from holographic.scene_and_pipeline.holographic_scene_doc import Scene
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    import holographic.materials_and_texture.holographic_matlib as ML
    sc = Scene(seed=0)
    honey = ML.material("honey"); honey.sss = 1.0
    sc.add(name="blob", geometry=sphere(0.8), material=honey)
    class Cam:
        eye = np.array([0.0, 0.0, 3.0])
        def ray_dirs(self, w, h, jitter=None):
            ys, xs = np.mgrid[0:h, 0:w]
            jx, jy = (0.0, 0.0) if jitter is None else (jitter[0], jitter[1])
            u = ((xs + jx) / (w - 1) - 0.5) * 1.1; v = -((ys + jy) / (h - 1) - 0.5) * 1.1
            d = np.stack([u, v, -np.ones_like(u)], -1); return self.eye, d / np.linalg.norm(d, axis=-1, keepdims=True)
    dark = lambda D: np.tile([0.01, 0.01, 0.015], (len(D), 1))
    soft = render_scene_document(sc, Cam(), width=32, height=32, quality="draft", max_bounce=2, seed=0,
                                 sky=dark, sss_dir=(0.5, 0.3, -0.8), sss_depth=1.3, sss_sigma=1.5)
    hard = render_scene_document(sc, Cam(), width=32, height=32, quality="draft", max_bounce=2, seed=0,
                                 sky=dark, sss_dir=(0.5, 0.3, -0.8), sss_depth=1.3, sss_sigma=8.0)
    assert soft.mean() > hard.mean()                             # softer absorption transmits more -> brighter
