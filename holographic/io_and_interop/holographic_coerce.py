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
        return obj                                        # already satisfies the renderer's protocol
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
