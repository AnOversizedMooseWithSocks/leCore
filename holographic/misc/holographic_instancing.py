"""holographic_instancing.py -- CMP4: type-correct scene binding + shared-definition instancing.

Two things real scenes need that a flat list of objects doesn't give you:

  1. TYPE-CORRECT BINDING. A material has a KIND -- a SURFACE material (paint, metal, glass) or a VOLUMETRIC one
     (fog, smoke, fire, which are participating media, not skins). Geometry has a kind too -- a SURFACE (a mesh) or a
     VOLUME. You may only bind like to like: a surface material onto a mesh, a volumetric material into a volume.
     Binding smoke onto a solid mesh, or paint onto a volume, is a mistake -- so we REFUSE it when the Definition is
     built (compose time), with a clear message, rather than letting it render wrong.

  2. SHARED-DEFINITION INSTANCING (edit-once). A Definition is a named, shared unit: geometry + material. An Instance
     is a placement of ONE Definition through a transform. Many instances share one Definition, so editing the
     Definition (repaint it, swap its mesh) updates EVERY instance at once, because an instance holds a REFERENCE, not
     a copy. That is the difference between "100 chairs" you can recolour in a single edit and 100 independent copies
     you'd have to edit one at a time.

KEPT NEGATIVE (loud): the sharing is at the GRAPH level (edit-once). flatten() is where instances become concrete
geometry -- it materialises each surface instance's mesh through its transform and MERGES them into one mesh (reusing
scenegraph.flatten_scene). After you flatten you hold a concrete pile of triangles, not instances; edit-once lives on
the graph you flattened FROM, not on the flattened mesh.

Reuses: holographic_scenegraph (SceneNode + flatten_scene for the geometry view; translation/scaling for placing an
instance), holographic_semantic._VOLUMETRIC (the surface/volumetric material split, kept in one place). Plain NumPy;
the graph stays a readable object tree you can print.
"""
import numpy as np

# The two KINDS, named once so geometry, materials, and the binding rule all agree.
SURFACE, VOLUME = "surface", "volume"


def material_kind(material):
    """The kind of a material by NAME: VOLUME for participating media (fog / smoke / fire), SURFACE otherwise. Reuses
    the one list in holographic_semantic so 'what counts as volumetric' is defined in exactly one place."""
    try:
        from holographic.simulation_and_physics.holographic_semantic import _VOLUMETRIC
    except Exception:
        _VOLUMETRIC = {"fog", "smoke", "fire"}
    return VOLUME if str(material).lower() in _VOLUMETRIC else SURFACE


def geometry_kind_of(geometry):
    """Best-guess the kind of a geometry object: SURFACE if it looks like a mesh (has vertices + faces), else VOLUME.
    You can always state the kind explicitly when building a Definition; this is just the sensible default."""
    if hasattr(geometry, "vertices") and hasattr(geometry, "faces"):
        return SURFACE
    return VOLUME


class Definition:
    """A shared, editable scene DEFINITION -- geometry bound to a material, with the binding TYPE-CHECKED: a surface
    material only binds to surface geometry (a mesh), a volumetric material only to a volume. The check runs when you
    build the Definition and again whenever you edit it, so a bad binding is refused up front. Edit this once (its
    material or geometry) and every Instance that references it updates."""

    def __init__(self, name, geometry, material, geometry_kind=None):
        self.name = name
        self.geometry = geometry
        self.geometry_kind = geometry_kind or geometry_kind_of(geometry)
        self._bind(material)                                   # sets self.material + self.material_kind, type-checked

    def _bind(self, material):
        mk = material_kind(material)
        if self.geometry_kind not in (SURFACE, VOLUME):
            raise ValueError("geometry_kind must be 'surface' or 'volume', got %r" % (self.geometry_kind,))
        if mk != self.geometry_kind:
            raise TypeError("cannot bind a %s material (%r) to %s geometry: a surface material (paint/metal/glass) "
                            "needs a mesh, a volumetric material (fog/smoke/fire) needs a volume"
                            % (mk, material, self.geometry_kind))
        self.material = material
        self.material_kind = mk

    def set_material(self, material):
        """Repaint the definition -- re-checks the binding, and updates every Instance of it (they share this object)."""
        self._bind(material)
        return self

    def set_geometry(self, geometry, geometry_kind=None):
        """Swap the definition's geometry -- re-checks the binding against the (existing) material. Updates every
        Instance of it."""
        self.geometry = geometry
        self.geometry_kind = geometry_kind or geometry_kind_of(geometry)
        self._bind(self.material)                              # the material must still be valid for the new geometry
        return self

    def __repr__(self):
        return "Definition(%r, %s geometry + %s material %r)" % (self.name, self.geometry_kind, self.material_kind,
                                                                 self.material)


class Instance:
    """A placement of ONE shared Definition through a transform (a 4x4 matrix; identity if omitted). Reads through to
    the shared definition, so instance.material / instance.geometry always reflect the LATEST edit to the definition."""

    def __init__(self, definition, transform=None, name=None):
        if not isinstance(definition, Definition):
            raise TypeError("an Instance references a Definition, got %r" % type(definition).__name__)
        self.definition = definition
        self.transform = np.eye(4) if transform is None else np.asarray(transform, float)
        self.name = name

    @property
    def material(self):
        return self.definition.material                       # always the shared definition's current material

    @property
    def geometry(self):
        return self.definition.geometry

    @property
    def kind(self):
        return self.definition.geometry_kind

    def __repr__(self):
        return "Instance(of %r%s)" % (self.definition.name, "" if self.name is None else ", " + self.name)


class InstancedScene:
    """A scene as a list of Instances over shared Definitions. Add instances; edit a Definition once and all of its
    instances change with it. flatten_surface() materialises the SURFACE instances into one mesh (via scenegraph);
    volume instances are listed separately (they aren't triangles). The kept negative made concrete: sharing is on
    this graph, flattening is where surface instances become geometry."""

    def __init__(self):
        self.instances = []

    def add(self, instance):
        """Place an instance in the scene. Returns the instance (so you can keep a handle to it)."""
        if not isinstance(instance, Instance):
            raise TypeError("add() takes an Instance, got %r" % type(instance).__name__)
        self.instances.append(instance)
        return instance

    def place(self, definition, transform=None, name=None):
        """Convenience: build an Instance of `definition` and add it in one call."""
        return self.add(Instance(definition, transform=transform, name=name))

    def instances_of(self, definition):
        """Every instance that shares a given Definition -- the set that a single edit to that Definition changes."""
        return [i for i in self.instances if i.definition is definition]

    def definitions(self):
        """The distinct Definitions referenced, in first-seen order (the shared, editable units in the scene)."""
        seen, out = set(), []
        for i in self.instances:
            if id(i.definition) not in seen:
                seen.add(id(i.definition))
                out.append(i.definition)
        return out

    def surface_instances(self):
        return [i for i in self.instances if i.kind == SURFACE]

    def volume_instances(self):
        return [i for i in self.instances if i.kind == VOLUME]

    def flatten_surface(self):
        """Materialise the SURFACE instances into ONE merged mesh: each instance's shared mesh placed through its
        transform, then merged (reusing scenegraph). Returns a Mesh (empty if there are no surface instances). This is
        where edit-once instances become concrete geometry -- the kept negative in action."""
        from holographic.scene_and_pipeline.holographic_scenegraph import SceneNode, flatten_scene
        children = [SceneNode(transform=i.transform, mesh=i.definition.geometry)
                    for i in self.surface_instances() if i.definition.geometry is not None]
        return flatten_scene(SceneNode(children=children))

    def __repr__(self):
        return "InstancedScene(%d instances over %d definitions)" % (len(self.instances), len(self.definitions()))


def _selftest():
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.scene_and_pipeline.holographic_scenegraph import translation

    # a shared DEFINITION: a cube mesh painted 'metal' (a surface material on surface geometry -- valid)
    chair = Definition("chair", box(1.0, 1.0, 1.0), "metal")
    assert chair.geometry_kind == SURFACE and chair.material_kind == SURFACE

    # TYPE-CORRECT BINDING refused at compose time -----------------------------------------------------------
    try:
        Definition("bad", box(1.0, 1.0, 1.0), "smoke")        # a volumetric material on a mesh -> refused
        raise AssertionError("should refuse smoke on a mesh")
    except TypeError as e:
        assert "volumetric material" in str(e)
    # a volumetric material DOES bind to a volume (geometry_kind='volume', geometry is not a mesh)
    haze = Definition("haze", object(), "fog", geometry_kind=VOLUME)
    assert haze.material_kind == VOLUME
    try:
        Definition("bad2", object(), "metal", geometry_kind=VOLUME)   # a surface material on a volume -> refused
        raise AssertionError("should refuse metal on a volume")
    except TypeError:
        pass

    # SHARED-DEFINITION INSTANCING (edit-once) ---------------------------------------------------------------
    scene = InstancedScene()
    a = scene.place(chair, translation([-2, 0, 0]), name="left")
    b = scene.place(chair, translation([2, 0, 0]), name="right")
    scene.place(chair, translation([0, 0, 2]), name="back")
    assert len(scene.instances) == 3 and len(scene.definitions()) == 1
    assert len(scene.instances_of(chair)) == 3
    # editing the ONE definition changes ALL instances (they hold a reference, not a copy)
    assert a.material == "metal" and b.material == "metal"
    chair.set_material("glass")
    assert a.material == "glass" and b.material == "glass", "edit-once should update every instance"

    # editing to an INVALID material is refused, and leaves the definition unchanged
    try:
        chair.set_material("smoke")                           # surface geometry can't take a volumetric material
        raise AssertionError("should refuse")
    except TypeError:
        pass
    assert a.material == "glass"                              # unchanged after the refused edit

    # FLATTEN materialises the surface instances into one mesh (the kept negative: instances -> concrete triangles)
    scene.place(haze, name="fog")                            # a volume instance -- not triangles, listed separately
    merged = scene.flatten_surface()
    one_cube = box(1.0, 1.0, 1.0)
    assert merged.n_vertices == 3 * one_cube.n_vertices      # 3 chair instances merged
    assert len(scene.volume_instances()) == 1 and len(scene.surface_instances()) == 3

    print("OK: holographic_instancing self-test passed (binding type-checked: smoke-on-mesh and metal-on-volume "
          "refused at compose time; 3 instances share one definition, editing it once repaints all, an invalid "
          "repaint is refused and leaves them unchanged; flatten merges the 3 surface instances into one mesh with "
          "%d vertices while the volume instance is listed separately)" % merged.n_vertices)


if __name__ == "__main__":
    _selftest()
