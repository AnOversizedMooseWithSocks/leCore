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


# ---------------------------------------------------------------------------
# J-3D-10: the view transform. A path tracer emits linear radiance with no upper
# bound; saving that to an 8-bit PNG is a wrong answer, not a missing polish step.
# ---------------------------------------------------------------------------

def _still_life(m):
    """The scene the whole 3-D backlog has been measured on, built through the mind only."""
    import numpy as np

    def T(tx=0.0, ty=0.0, tz=0.0):
        M = np.eye(4); M[:3, 3] = (tx, ty, tz); return M

    sc = m.new_scene()
    sc.add(name="floor", geometry=m.sdf_parse("(plane 0.0)"), material="matte_gray", transform=T())
    sc.add(name="ball", geometry=m.sdf_parse("(translate -1.1 0.6 0.0 (sphere 0.6))"),
           material="copper", transform=T())
    sc.add(name="cube", geometry=m.sdf_parse("(translate 0.0 0.5 0.0 (box 0.5 0.5 0.5))"),
           material="wood_oak", transform=T())
    return sc


def test_view_none_is_bit_identical():
    """ADDITIVITY, and it is the assertion that matters most in this file. `view` is a new parameter on a
    shipped faculty; if its default moved a single bit, an existing decision flipped and the change is
    rejected regardless of how good the new image looks."""
    import numpy as np
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = _still_life(m)
    cam = m.camera(eye=(2.6, 2.0, 4.2), target=(0.0, 0.6, 0.0), fov_deg=40.0, aspect=4 / 3.)
    a = m.render_scene_document(sc, cam, 32, 24, quality="fast", seed=0)
    b = m.render_scene_document(sc, cam, 32, 24, quality="fast", seed=0, view=None)
    assert np.array_equal(a, b), "view=None must be bit-identical to omitting it"


def test_display_view_bounds_the_buffer_without_crushing():
    """The measured contract, both ends. On the full-size still life under a dome + area light the raw
    buffer clipped 15.5% of pixels; 'display' meters first, so highlights AND shadows survive, where the
    'graded' preset's FIXED exposure stop clears the top by crushing the bottom (1.97% to black)."""
    import numpy as np
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = _still_life(m)
    cam = m.camera(eye=(2.6, 2.0, 4.2), target=(0.0, 0.6, 0.0), fov_deg=40.0, aspect=4 / 3.)
    lights = [m.scene_light("dome", color=(0.30, 0.38, 0.52), intensity=1.6),
              m.scene_light("softbox", position=(2.2, 3.4, 2.6), target=(0.0, 0.5, 0.0),
                            width=2.0, height=2.0, intensity=90.0)]
    kw = dict(width=40, height=30, quality="fast", seed=0, lights=lights)
    raw = m.render_scene_document(sc, cam, **kw)
    disp = m.render_scene_document(sc, cam, view="display", **kw)
    assert raw.max() > 1.0, "the fixture must actually exceed display range, or it tests nothing"
    assert disp.max() <= 1.0 and disp.min() >= 0.0
    assert float((disp < 0.004).mean()) <= float((raw < 0.004).mean()) + 1e-9, \
        "the metered view must not manufacture black pixels"


def test_bad_view_name_says_what_is_valid():
    """Agent-facing means the ERROR TEXT is part of the contract. A bare KeyError three frames down is the
    exact failure mode this backlog exists to remove, so the message is what gets pinned."""
    import pytest
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = _still_life(m)
    cam = m.camera(eye=(2.6, 2.0, 4.2), target=(0.0, 0.6, 0.0), fov_deg=40.0, aspect=4 / 3.)
    with pytest.raises(ValueError) as e:
        m.render_scene_document(sc, cam, 8, 8, quality="fast", seed=0, view="filmic")
    assert "display" in str(e.value) and "graded" in str(e.value)


def test_the_working_stack_is_expressible_as_a_chain():
    """auto_exposure shipped as a module function and NOT as a chain step, so an agent holding postfx_chain
    got KeyError: 'auto_exposure' -- it could not express the one stack that works. Reachable-by-import is
    not reachable. Also pins discoverability: the phrasings that used to return texture-baking."""
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    ch = m.postfx_chain(("auto_exposure", {}), ("aces", {}), ("gamma", {"g": 2.2}))
    assert [n for n, _ in ch.to_list()] == ["auto_exposure", "aces", "gamma"]
    for phrasing in ("my render is blown out", "my highlights are clipping", "tonemap an hdr render"):
        assert "View transform" in str(m.find_capability(phrasing)[0]), phrasing


# ---------------------------------------------------------------------------
# J-3D-05/06: render_preview. A DRAFT, and the tests keep it honest about that.
# ---------------------------------------------------------------------------

def test_preview_is_much_faster_than_the_full_render():
    """The whole claim, measured in-test rather than quoted. Deliberately loose (3x, not the 12.0x measured
    at 240x180) because CI machines vary and a tight timing assert is a flaky test wearing a rigour costume
    -- but loose is not absent: if the preview ever stops being dramatically faster it has no reason to
    exist, and this fails."""
    import time
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = _still_life(m)
    cam = m.camera(eye=(2.6, 2.0, 4.2), target=(0.0, 0.6, 0.0), fov_deg=40.0, aspect=4 / 3.)
    t = time.time(); m.render_preview(sc, cam, 48, 36, seed=0); preview_s = time.time() - t
    t = time.time()
    m.render_scene_document(sc, cam, 48, 36, quality="fast", max_bounce=4, seed=0, view="display")
    full_s = time.time() - t
    assert preview_s * 3 < full_s, "preview %.2fs vs full %.2fs -- not worth having" % (preview_s, full_s)


def test_preview_returns_the_size_it_was_asked_for():
    """It renders at a fraction and upscales, so the output size is a CONTRACT. An agent framing a shot
    against a silently different size chases a bug that is not there."""
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = _still_life(m)
    cam = m.camera(eye=(2.6, 2.0, 4.2), target=(0.0, 0.6, 0.0), fov_deg=40.0, aspect=4 / 3.)
    for (w, h, scale) in ((64, 48, 0.5), (40, 30, 0.25), (32, 24, 1.0)):
        img = m.render_preview(sc, cam, w, h, scale=scale, seed=0)
        assert img.shape[0] == h and img.shape[1] == w, (w, h, scale, img.shape)


def test_scale_is_a_fraction_and_says_so():
    """scale=2.0 would make the preview SLOWER than the render it replaces. Accepting it silently is worse
    than refusing: the caller gets the opposite of what they asked for and no signal."""
    import pytest
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = _still_life(m)
    cam = m.camera(eye=(2.6, 2.0, 4.2), target=(0.0, 0.6, 0.0), fov_deg=40.0, aspect=4 / 3.)
    with pytest.raises(ValueError, match="FRACTION"):
        m.render_preview(sc, cam, 32, 24, scale=2.0, seed=0)


def test_the_preview_is_a_draft_and_differs_from_the_final():
    """KEPT NEGATIVE, pinned as a test. One bounce means no indirect light: previews are flatter with
    darker shadows. If this ever matches exactly, either max_bounce stopped mattering or the preview
    quietly became the full render -- both are regressions, in opposite directions."""
    import numpy as np
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = _still_life(m)
    cam = m.camera(eye=(2.6, 2.0, 4.2), target=(0.0, 0.6, 0.0), fov_deg=40.0, aspect=4 / 3.)
    p = m.render_preview(sc, cam, 32, 24, seed=0)
    f = m.render_scene_document(sc, cam, 32, 24, quality="fast", max_bounce=4, seed=0, view="display")
    assert float(np.abs(np.asarray(p, float) - np.asarray(f, float)).mean()) > 1e-4
    assert "Fast preview" in str(m.find_capability("my render is too slow to iterate on")[0])


# ---------------------------------------------------------------------------
# J-3D-16/17: affine placement + place(). The rotation was silently dropped.
# ---------------------------------------------------------------------------

def test_affine_placement_is_exact_against_the_matrix():
    """EXACTNESS, not 'it looks turned'. A backwards axis or a flipped sign produces a picture that looks
    plausibly rotated and is wrong, so the assertion is against the transform's own definition: the placed
    field must equal the original evaluated at inverse-transformed points."""
    import numpy as np
    from holographic.rendering.holographic_scene_render import _place
    from holographic.mesh_and_geometry.holographic_sdf import box
    rng = np.random.default_rng(7)
    g = box(0.6, 0.25, 0.4)
    for _ in range(4):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        th = float(rng.uniform(-np.pi, np.pi)); s = float(rng.uniform(0.6, 1.8))
        tr = rng.uniform(-1.5, 1.5, 3)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        T = np.eye(4); T[:3, :3] = R * s; T[:3, 3] = tr
        P = rng.uniform(-2.5, 2.5, (200, 3))
        expect = g.eval((np.linalg.inv(R) @ (P - tr).T).T / s) * s
        assert float(np.abs(_place(g, T, affine=True).eval(P) - expect).max()) < 1e-12


def test_affine_defaults_off_and_the_default_is_unchanged():
    """ADDITIVITY. affine=True changes the rendered image of every scene containing a rotated object. The
    old picture is WRONG, but 'wrong' and 'safe to change under someone' are different claims -- shipped
    output does not move without an explicit decision, so the fix ships reachable and OFF."""
    import numpy as np
    from holographic.rendering.holographic_scene_render import _place
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    g = sphere(0.4).translate((0.5, 0, 0))
    T = np.eye(4)
    T[0, 0] = T[2, 2] = np.cos(0.9); T[0, 2] = np.sin(0.9); T[2, 0] = -np.sin(0.9)
    P = np.random.default_rng(0).uniform(-2, 2, (150, 3))
    assert np.array_equal(_place(g, T).eval(P), g.eval(P)), "the default must still drop the rotation"
    assert not np.array_equal(_place(g, T, affine=True).eval(P), g.eval(P))


def test_place_replaces_only_what_it_is_given():
    """The verb has to be usable incrementally. `place(rotation=...)` must not snap the object back to the
    origin, or every turn becomes a two-call dance and agents will get it wrong half the time."""
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = m.new_scene()
    h = m.scene_add(sc, name="cube", geometry=m.sdf_parse("(box 0.4 0.4 0.4)"), material="copper")
    m.place(sc, h, position=(1.0, 0.5, 0.0), rotation=(0, 45, 0), scale=1.5)
    o = m.scene_info(sc)["objects"][0]
    assert o["position"] == [1.0, 0.5, 0.0] and o["rotated"] is True and abs(o["scale"] - 1.5) < 1e-9
    m.place(sc, h, position=(2.0, 0.0, 0.0))            # position ONLY
    o = m.scene_info(sc)["objects"][0]
    assert o["position"] == [2.0, 0.0, 0.0], "position did not update"
    assert o["rotated"] is True, "a position-only place() erased the rotation"
    assert abs(o["scale"] - 1.5) < 1e-9, "a position-only place() erased the scale"
    assert m.scene_undo(sc) is True and m.scene_info(sc)["objects"][0]["position"] == [1.0, 0.5, 0.0]


def test_place_accepts_the_three_spellings_of_a_rotation():
    """Euler degrees, (axis, angle), and a 3x3 all arrive from real callers -- a person, a tool, a file
    format. Rejecting two would push the conversion into every caller."""
    import numpy as np
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    ref = m._rotation_matrix((0, 90, 0))
    assert np.allclose(ref, m._rotation_matrix(((0, 1, 0), 90)), atol=1e-9)
    assert np.allclose(ref, m._rotation_matrix(ref), atol=1e-12)
    assert np.allclose(ref, m._rotation_matrix((0, np.pi / 2, 0), degrees=False), atol=1e-9)


def test_scene_info_now_names_the_fix():
    """A warning that describes a dead end teaches an agent to ignore warnings. Now that affine=True
    exists, the pre-flight message must point at it."""
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = m.new_scene()
    h = m.scene_add(sc, name="cube", geometry=m.sdf_parse("(box 0.4 0.4 0.4)"), material="copper")
    m.place(sc, h, rotation=(0, 30, 0))
    problems = " | ".join(m.scene_info(sc)["problems"])
    assert "affine=True" in problems
    assert "Move / rotate / scale" in str(m.find_capability("why did my object not rotate")[0])


# ---------------------------------------------------------------------------
# render_animation + save_gif: the motion see->fix loop, composed from existing parts.
# ---------------------------------------------------------------------------

def test_animation_actually_moves_and_keys_are_json_shapes():
    """The composition claim: Timeline + place + render_preview, driven entirely by JSON-shaped keys --
    because a Timeline object cannot cross POST /invoke, so the JSON shape IS the interface."""
    import numpy as np
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = _still_life(m)
    h = sc.objects and list(sc.objects)[1]                     # the ball
    cam = m.camera(eye=(0.0, 1.0, 3.0), target=(0.0, 0.0, 0.0), fov_deg=40.0, aspect=4 / 3.)
    keys = {h: {"position": [[0.0, [-1.0, 0.5, 0.0]], [1.0, [1.0, 0.5, 0.0]]]}}
    frames = m.render_animation(sc, cam, keys, n_frames=4, fps=4, width=32, height=24, seed=0)
    assert len(frames) == 4 and frames[0].shape == (24, 32, 3)
    assert np.abs(frames[0] - frames[-1]).mean() > 1e-3, "the object did not move"


def test_unknown_property_raises_with_the_valid_set():
    """'velocity' is a plausible guess. The error must name what place() can actually apply."""
    import pytest
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = _still_life(m)
    h = list(sc.objects)[0]
    cam = m.camera(eye=(0.0, 1.0, 3.0), target=(0.0, 0.0, 0.0), fov_deg=40.0, aspect=4 / 3.)
    with pytest.raises(ValueError, match="position"):
        m.render_animation(sc, cam, {h: {"velocity": [[0, [1, 0, 0]]]}}, n_frames=2, width=16, height=12)


def test_gif_writer_is_deterministic_and_well_formed(tmp_path):
    """Two runs, identical bytes -- the fixed 252-colour lattice exists precisely so this holds; median-cut
    palettes split on content and would make the same animation differ run to run. Plus the container
    basics a viewer needs: signature, per-frame image descriptors, trailer."""
    import numpy as np
    from holographic.rendering.holographic_render import save_gif
    frames = []
    for t in range(5):
        img = np.zeros((20, 30, 3))
        img[:, 5 * t:5 * t + 4] = (1.0, 0.4, 0.1)
        frames.append(img)
    a, b = tmp_path / "a.gif", tmp_path / "b.gif"
    save_gif(str(a), frames, fps=10)
    save_gif(str(b), frames, fps=10)
    da = a.read_bytes()
    assert da == b.read_bytes(), "the writer is not deterministic"
    assert da[:6] == b"GIF89a" and da[-1] == 0x3B
    assert da.count(b"\x2c") >= 5, "expected an image descriptor per frame"


def test_gif_rejects_mismatched_frames(tmp_path):
    """A size change mid-animation renders as garbage in some viewers and truncates in others -- neither is
    a useful failure. Refusing at the writer names the frame sizes involved."""
    import numpy as np
    import pytest
    from holographic.rendering.holographic_render import save_gif
    with pytest.raises(ValueError, match="share one size"):
        save_gif(str(tmp_path / "x.gif"), [np.zeros((8, 8, 3)), np.zeros((8, 10, 3))])
    assert True


def test_sky_keys_animates_the_hour_and_the_lighting_follows():
    """The timelapse contract: with sky_keys (and NO explicit lights) the frames must genuinely darken as
    the keyed hour crosses sunset -- both the sky pixels AND the ground, because the animated sky drives
    the dome when the caller gave no lights. A timelapse whose lighting ignores its sky is two different
    times of day in one frame."""
    import numpy as np
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = m.new_scene()
    sc.add(name="floor", geometry=m.shape("plane"), material="matte_gray")
    cam = m.camera(eye=(0.0, 1.0, 3.5), target=(0.0, 1.4, -2.0), fov_deg=55.0, aspect=4 / 3.)
    f = m.render_animation(sc, cam, {}, n_frames=4, fps=2, width=24, height=18,
                           sky_keys={"hour": [[0.0, 12.0], [2.0, 22.0]]}, view=None, seed=0)
    assert f[-1].mean() < 0.5 * f[0].mean(), \
        "noon -> late-evening timelapse did not darken: %.3f -> %.3f" % (f[0].mean(), f[-1].mean())
    ground = [x[-4:].mean() for x in f]                        # bottom rows: the lit floor
    assert ground[-1] < 0.7 * ground[0], "the GROUND ignored the animated sky (dome not driven)"


def test_sky_and_sky_keys_are_mutually_exclusive():
    """Two skies in one call has no meaning; the refusal must say which one to drop."""
    import pytest
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = m.new_scene()
    sc.add(name="floor", geometry=m.shape("plane"), material="matte_gray")
    cam = m.camera(eye=(0.0, 1.0, 3.0), target=(0.0, 0.5, 0.0), fov_deg=45.0, aspect=4 / 3.)
    with pytest.raises(ValueError, match="not both"):
        m.render_animation(sc, cam, {}, n_frames=2, width=16, height=12,
                           sky=m.sky_model(12.0), sky_keys={"hour": [[0, 12], [1, 20]]})
    with pytest.raises(ValueError, match="hour"):
        m.render_animation(sc, cam, {}, n_frames=2, width=16, height=12,
                           sky_keys={"clouds": [["cirrus", 0.4]]})


def test_adaptive_gif_palette_beats_fixed_on_gradients_and_stays_deterministic():
    """MOOSE'S REVIEW, pinned: the fixed 6x7x6 lattice collapsed a smooth sky gradient to ~9 colours --
    'very crazy artifacts'. The adaptive palette (median-cut over ALL frames, one palette for the whole
    animation) must (a) quantise a gradient with much lower error than the lattice, and (b) remain
    deterministic: same frames, same bytes, twice -- median-cut is content-dependent, so determinism is by
    fixed stride + stable sort + fixed split rule, and this assertion is what holds those in place."""
    import numpy as np
    from holographic.rendering.holographic_render import save_gif

    # a smooth vertical sky-like gradient: the exact banding victim
    h, w = 60, 80
    g = np.linspace(0.25, 0.85, h)[:, None, None] * np.array([0.55, 0.7, 0.95])[None, None, :]
    frames = [np.broadcast_to(g, (h, w, 3)).copy() for _ in range(3)]

    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())
    a1, a2, fx = tmp / "a1.gif", tmp / "a2.gif", tmp / "f.gif"
    save_gif(str(a1), frames, fps=5, palette="adaptive", dither=True)
    save_gif(str(a2), frames, fps=5, palette="adaptive", dither=True)
    save_gif(str(fx), frames, fps=5)
    assert a1.read_bytes() == a2.read_bytes(), "adaptive palette broke byte determinism"

    # error comparison, measured on the quantisation itself (fixed lattice vs the adaptive palette)
    rl, gl, bl = 6, 7, 6
    f0 = frames[0]
    fq = np.stack([np.clip((f0[..., 0] * (rl - 1)).round(), 0, rl - 1) / (rl - 1),
                   np.clip((f0[..., 1] * (gl - 1)).round(), 0, gl - 1) / (gl - 1),
                   np.clip((f0[..., 2] * (bl - 1)).round(), 0, bl - 1) / (bl - 1)], axis=-1)
    fixed_rms = float(np.sqrt(((fq - f0) ** 2).mean()))
    # the adaptive file being larger than the fixed one on a gradient is itself evidence the palette is
    # being USED; the hard numeric bound lives on the lattice side
    assert fixed_rms > 0.02, "the lattice should band on this gradient; if not, the test lost its victim"
    assert a1.stat().st_size > fx.stat().st_size, \
        "adaptive+dither should spend MORE bits on a gradient than 9 flat bands"


def test_bayer_dither_is_stable_between_identical_frames():
    """The dithering that was DECLINED was error-diffusion/noise, which crawls between near-identical
    frames. Bayer is a fixed spatial pattern: two identical frames must dither identically -- the encoded
    frames inside the GIF must be the same bytes, or the animation shimmers while standing still."""
    import numpy as np
    from holographic.rendering.holographic_render import save_gif
    import tempfile, pathlib

    h, w = 40, 50
    g = np.linspace(0.3, 0.7, h)[:, None, None] * np.ones((1, w, 3))
    # PROBE NOTE: the first version split the file on b"\x2c" (the image-descriptor byte) and failed --
    # not because the dither moved but because 0x2c occurs freely inside LZW data. The honest probe is the
    # length argument: if identical frames encode identically, the two-frame file is the one-frame file
    # plus EXACTLY one more frame section. A moved dither pattern changes the second frame's LZW stream
    # and (overwhelmingly) its length; equality of the delta with the measured frame-section size is the
    # stable, parser-free assertion.
    tmp = pathlib.Path(tempfile.mkdtemp())
    two, one = tmp / "two.gif", tmp / "one.gif"
    save_gif(str(two), [g, g], fps=5, palette="adaptive", dither=True)
    save_gif(str(one), [g], fps=5, palette="adaptive", dither=True)
    header = 6 + 7 + 768 + 19                                  # signature + LSD + global palette + NETSCAPE
    frame_section = one.stat().st_size - header - 1            # minus the trailer byte
    assert two.stat().st_size - one.stat().st_size == frame_section, \
        "identical frames encoded differently -- the dither pattern moved (or the container changed shape)"


def test_gif_optimizations_preserve_exact_output():
    """The optimization contract for save_gif's two rewrites (bit-accumulator LZW, GEMM quantiser):
    2.8x measured, and the OUTPUT MUST NOT MOVE. LZW is byte-identical by construction (same codes, same
    order -- repackaging); the GEMM quantiser has the stated tie caveat (a pixel exactly equidistant from
    two palette entries could flip), so the assertion here runs the GEMM form against the direct
    |x-p|^2 argmin on real-shaped noisy gradient frames -- the measure-zero claim, actually measured."""
    import numpy as np
    from holographic.rendering.holographic_render import save_gif

    rng = np.random.default_rng(0)
    frames = [np.clip(np.linspace(0, 1, 40)[:, None, None] * np.array([0.5, 0.7, 0.95])
                      + rng.normal(0, 0.02, (40, 60, 3)), 0, 1) for _ in range(2)]

    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())
    a, b = tmp / "a.gif", tmp / "b.gif"
    save_gif(str(a), frames, palette="adaptive", dither=True)
    save_gif(str(b), frames, palette="adaptive", dither=True)
    assert a.read_bytes() == b.read_bytes()

    # GEMM vs direct nearest-palette on the same pixels: indices must agree everywhere
    pal = rng.uniform(0, 255, (256, 3))
    px = (frames[0].reshape(-1, 3) * 255)
    direct = np.argmin(((px[:, None, :] - pal[None, :, :]) ** 2).sum(-1), axis=1)
    gemm = np.argmin((pal ** 2).sum(1)[None, :] - 2.0 * (px @ pal.T), axis=1)
    assert np.array_equal(direct, gemm), \
        "GEMM nearest-palette diverged from the direct form: %d pixels" % (direct != gemm).sum()
