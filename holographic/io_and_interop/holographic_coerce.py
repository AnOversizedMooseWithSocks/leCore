"""holographic_coerce.py -- accept PLAIN JSON where a faculty wants a live object.

WHY THIS EXISTS (a downstream audit, not a hypothesis). A ComfyUI node pack auditing leCore reported that
`render_mesh` -- the flagship mesh->image path -- "cannot be called by ANY JSON client today, including your own
POST /invoke", because it needs live `Mesh` and `Camera` objects and a node can only do
`getattr(mind, name)(**json_args)`. Reproduced exactly: `render_mesh({'vertices':...,'faces':...})` raises
`AttributeError: 'dict' object has no attribute 'faces'`.

WHAT RULE 0 FOUND, and why this module is small. The audit also claimed the constructors were missing
("make_box, box_mesh, cube, primitive -- all absent"). They are not: `mind.mesh_box()` returns a real Mesh and
`mind.camera(...)` returns a real Camera WITH `projection_matrix`, so
`m.render_mesh(m.mesh_box(), m.camera(...))` already works in-process today. What was actually missing was
(a) DISCOVERABILITY -- `find_capability("make a box")` surfaced Catmull-Clark subdivision, never `mesh_box`
(fixed with aliases, the D1 pattern), and (b) this: the coercion at the boundary.

The two-camera trap is the same story. `CameraController` has no `projection_matrix()`, so it fails DEEP inside
the rasteriser -- the auditor "fell in it". But it already carries `to_camera()`, the exact bridge; nothing ever
called it. So this module does not add a protocol or a second Camera: it CALLS THE BRIDGE THAT EXISTS.

DESIGN: coercion lives at the FACULTY boundary, never in the renderer. holographic_render keeps taking real
objects and stays free of dict-sniffing; only the JSON-facing edge is permissive. Passing a real Mesh/Camera is
byte-identical to before -- these helpers return their input untouched when it is already the right type, so no
existing decision can flip.
"""


def as_mesh(obj):
    """A live `Mesh` from either a Mesh (returned untouched) or a JSON dict {'vertices', 'faces', ...}.

    Accepts the optional `normals` / `uvs` / `colours` Mesh fields when present, so a JSON client can round-trip
    a textured mesh, not just naked geometry. Anything already exposing `.faces` is passed straight through --
    duck-typed on purpose, so a subclass or a future Mesh-alike keeps working.

        as_mesh({'vertices': V, 'faces': F})   -> Mesh
        as_mesh(existing_mesh)                 -> the SAME object (identity, not a copy)
    """
    if hasattr(obj, "faces"):
        return obj
    if isinstance(obj, dict):
        from holographic.mesh_and_geometry.holographic_mesh import Mesh
        try:
            v, f = obj["vertices"], obj["faces"]
        except KeyError as e:
            raise TypeError("a mesh dict needs 'vertices' and 'faces'; missing %s. Got keys: %s"
                            % (e, sorted(obj))) from None
        return Mesh(v, f, normals=obj.get("normals"), uvs=obj.get("uvs"), colours=obj.get("colours"))
    raise TypeError("cannot read a mesh from %r -- pass a Mesh (e.g. mind.mesh_box()) or "
                    "{'vertices': [...], 'faces': [...]}" % type(obj).__name__)


def as_camera(obj):
    """A live render `Camera` from a Camera, a `CameraController`, or a JSON dict {'eye', 'target', ...}.

    The CameraController branch is the fix for a real trap: it has `view_matrix()` but NOT
    `projection_matrix()`, so handing one to the rasteriser fails deep inside the MVP build with an error that
    names neither the caller nor the cause. It has always carried `to_camera()`; this calls it rather than
    inventing a second protocol or a third camera class.

        as_camera({'eye': [2,2,2], 'target': [0,0,0]})   -> Camera
        as_camera(mind.camera_controller(...))           -> Camera (via its own to_camera bridge)
        as_camera(existing_camera)                       -> the SAME object
    """
    if hasattr(obj, "projection_matrix"):
        return obj                                        # already satisfies the RASTERISER's protocol
    if hasattr(obj, "ray_dirs"):
        # THE RAY TRACER'S PROTOCOL IS ray_dirs, NOT projection_matrix, and
        # they are different interfaces for different renderers. This function
        # was written for the rasteriser; I then called it from
        # render_scene_document, which is RAY-TRACED -- so four integration
        # tests that had always passed a duck-typed camera (eye + ray_dirs, no
        # matrices at all) started failing with "cannot read a camera from
        # 'Cam'".
        # THE COERCION WAS RIGHT TO EXIST AND WRONG ABOUT WHAT COUNTS AS A
        # CAMERA. Anything the target renderer can already use must pass
        # through untouched; a coercion that rejects a working input has become
        # a gate, and a gate nobody asked for is a regression.
        return obj
    if hasattr(obj, "to_camera"):
        return obj.to_camera()                            # CameraController: the bridge it already had
    if isinstance(obj, dict):
        from holographic.rendering.holographic_render import Camera
        d = dict(obj)
        if "look" in d and "target" not in d:
            d["target"] = d.pop("look")                   # the audit's own wording; accept it rather than punish
        allowed = ("eye", "target", "up", "fov_deg", "aspect", "near", "far")
        bad = [k for k in d if k not in allowed]
        if bad:
            raise TypeError("unknown camera field(s) %s; expected any of %s" % (bad, list(allowed)))
        return Camera(**d)
    raise TypeError("cannot read a camera from %r -- pass a Camera (e.g. mind.camera(eye=..., target=...)), a "
                    "CameraController, or {'eye': [...], 'target': [...]}" % type(obj).__name__)


def as_scene(obj):
    """A Scene document from a Scene, or from a dict that CONTAINS one.

    `scene_from_image` returns {objects, regions, roles, scene} -- a report
    whose `scene` value is a real scene -- and every renderer wants the scene
    itself, so the composition raised "'dict' object has no attribute
    'objects'". Two honest fixes existed: change what scene_from_image returns
    (breaking anyone reading `regions` or `roles`, which are the interesting
    parts of that report) or COERCE AT THE CONSUMER. The report is not wrong to
    be a report; the renderers were wrong to accept only one shape.
    Anything with `.objects` passes through untouched, so a real Scene costs
    nothing here."""
    if hasattr(obj, "objects"):
        # A SEMANTIC SCENE IS NOT A RENDERABLE ONE, and it took three layers to
        # find that out: dict-vs-Scene, then list-vs-dict objects, then objects
        # that are DICTS with {label, shape, position, colour} and no geometry
        # at all. SemanticScene DESCRIBES a scene ("a red cube on the left");
        # Scene CONTAINS one (SDF geometry the renderer can .eval()).
        # Refusing here names the gap in one place instead of failing deep in
        # the tracer with "'dict' object has no attribute 'geometry'", which is
        # the same accept-then-crash shape scene_add had.
        # CHECK EVERY OBJECT, NOT THE FIRST. A scene can be MIXED -- the first
        # object carrying geometry and a later one being a bare dict -- and
        # sampling one is how a guard passes the case it was written for and
        # lets through the case that motivated it.
        _require_renderable(obj)
        return obj
    if isinstance(obj, dict):
        for k in ("scene", "document", "doc"):
            v = obj.get(k)
            if hasattr(v, "objects"):
                _require_renderable(v)     # the SAME check, both entry points
                return v
        raise TypeError(
            "cannot read a scene from a dict with keys %s -- expected a Scene, "
            "or a report carrying one under 'scene' (as scene_from_image "
            "returns)" % sorted(obj))
    raise TypeError("cannot read a scene from %r -- pass a Scene document or a "
                    "report containing one" % type(obj).__name__)


def semantic_to_scene(semantic, scene=None):
    """A SEMANTIC scene -> a RENDERABLE Scene document. The missing converter.

    scene_from_image and the description parser both produce a SemanticScene:
    objects as dicts of {label, shape, position, colour, material} that DESCRIBE
    a scene rather than carrying geometry. Every renderer wants a Scene whose
    objects have an SDF the tracer can .eval(), so the composition
    render_scene_document(scene_from_image(img), camera) raised three different
    errors at three different depths and there was no supported way to bridge.
    RULE 0 FOUND THE BRIDGE ALREADY BUILT. `realize_scene` (holographic_semantic)
    turns parsed objects into renderables -- dicts with an `sdf` that has .eval,
    a colour and a material -- and describe_to_scene has been using it all
    along. The converter was never missing; the DOOR from the image side to it
    was. This is that door, and it adds no geometry logic of its own."""
    from holographic.scene_and_pipeline.holographic_scene_doc import Scene
    from holographic.simulation_and_physics.holographic_semantic import (
        realize_scene)

    # UNWRAP A REPORT TOO. scene_from_image returns {objects, regions, roles,
    # scene}, and `getattr(report, "objects", report)` fell through to the
    # REPORT ITSELF -- a dict, which realize_scene then indexed as a list of
    # objects. Accepting the report is the whole point of this door, so it
    # handles all three shapes: a report, a SemanticScene, or a bare list.
    if isinstance(semantic, dict):
        for k in ("scene", "document", "doc"):
            if hasattr(semantic.get(k), "objects"):
                semantic = semantic[k]
                break
    objs = getattr(semantic, "objects", semantic)
    if hasattr(objs, "values"):
        objs = list(objs.values())
    out = scene if scene is not None else Scene()
    # MATERIAL NAMES ARE SEMANTIC, NOT MATLIB KEYS. realize_scene returns
    # mat_name="matte" (the word a person says) while the library holds
    # "matte_white"/"matte_gray"/"matte_black", and its `material` field is a
    # loose dict like {"reflect": 0.0} that the shader cannot read -- it wants
    # an object with .base_color. Passing either through unchanged crashed deep
    # in the tracer, which is the accept-then-crash shape this file already
    # guards against elsewhere.
    # So resolve against the library and FALL BACK rather than guess: a name
    # that does not resolve leaves the material unset, and the renderer's own
    # default_material applies. A wrong material renders; a missing attribute
    # does not.
    import holographic.materials_and_texture.holographic_matlib as ML

    for r in realize_scene(list(objs)):
        mat = None
        for cand in (r.get("mat_name"), "%s_gray" % r.get("mat_name"),
                     "%s_white" % r.get("mat_name")):
            if not cand:
                continue
            try:
                mat = ML.material(cand)
                break
            except Exception:
                continue
        out.add(name=r.get("name"), geometry=r.get("sdf"), material=mat)
    return out


def _require_renderable(scene):
    """Refuse a SEMANTIC scene before a renderer touches it.

    CHECK EVERY OBJECT, NOT THE FIRST: a scene can be MIXED, and sampling one is
    how a guard passes the case it was written for and lets through the case
    that motivated it."""
    objs = getattr(scene, "objects", ())
    for o in (objs.values() if hasattr(objs, "values") else objs):
        if not hasattr(o, "geometry"):
            raise TypeError(
                "this is a SEMANTIC scene (objects carry %s), not a renderable "
                "Scene document -- it describes what is in a scene rather than "
                "carrying SDF geometry. There is no semantic->renderable "
                "Scene document. Convert it first: semantic_to_scene(x) (or "
                "mind.semantic_to_scene), which realizes each described object "
                "into SDF geometry the tracer can .eval()."
                % (sorted(o)[:4] if isinstance(o, dict)
                   else type(o).__name__))


def _selftest():
    import numpy as np

    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    from holographic.rendering.holographic_camera import CameraController
    from holographic.rendering.holographic_render import Camera

    V = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    F = [[0, 1, 2]]

    # dict -> Mesh, and the fields actually survive.
    m = as_mesh({"vertices": V, "faces": F})
    assert isinstance(m, Mesh) and len(m.faces) == 1 and len(m.vertices) == 3

    # IDENTITY, not a copy: an existing object must pass through untouched or a caller's edits would vanish.
    real = Mesh(np.array(V), np.array(F))
    assert as_mesh(real) is real, "a real Mesh must pass through by identity"

    cam = Camera(eye=(1.0, 1.0, 1.0), target=(0.0, 0.0, 0.0))
    assert as_camera(cam) is cam, "a real Camera must pass through by identity"

    # dict -> Camera, including the audit's 'look' spelling.
    c = as_camera({"eye": [2.0, 2.0, 2.0], "target": [0.0, 0.0, 0.0]})
    assert isinstance(c, Camera) and hasattr(c, "projection_matrix")
    c2 = as_camera({"eye": [2.0, 2.0, 2.0], "look": [0.0, 0.0, 0.0]})
    assert np.allclose(c2.target, [0.0, 0.0, 0.0]), "the 'look' alias must map to target"

    # THE TRAP THIS EXISTS FOR: a CameraController must come out renderable.
    cc = CameraController(eye=(2.0, 2.0, 2.0), target=(0.0, 0.0, 0.0))
    assert not hasattr(cc, "projection_matrix"), "premise changed: CameraController now has the attr directly"
    got = as_camera(cc)
    assert hasattr(got, "projection_matrix"), "CameraController must coerce to something the rasteriser accepts"
    assert np.allclose(got.eye, cc.eye), "the coerced camera must keep the controller's pose"

    # KEPT NEGATIVE (loud): a bad payload must fail with a message naming the FIX, not an AttributeError from
    # 200 lines deeper. The whole complaint was an error that named neither the caller nor the cause.
    for bad, kind in (({"vertices": V}, "mesh"), (42, "mesh"), ({"eye": [0, 0, 1], "zoom": 2}, "camera"), (42, "camera")):
        try:
            as_mesh(bad) if kind == "mesh" else as_camera(bad)
            raise AssertionError("should have raised for %r" % (bad,))
        except TypeError as e:
            assert "mesh" in str(e).lower() or "camera" in str(e).lower(), e

    print("holographic_coerce selftest OK (dict->Mesh/Camera; real objects pass by IDENTITY; CameraController "
          "coerces via its own to_camera bridge; 'look' aliases target; bad payloads raise a TypeError that "
          "names the fix instead of an AttributeError from inside the rasteriser)")


if __name__ == "__main__":
    _selftest()
