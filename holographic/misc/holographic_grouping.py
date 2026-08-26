"""holographic_grouping.py -- GROUPING (a bundle) and INSTANCING (a bind) over the Scene (modeling-app feature layer).

Two more features that fall straight out of the VSA reframe:

  * GROUPING is a BUNDLE. A group is a null parent object that owns its members (through the Scene's hierarchy),
    and its identity is the SUPERPOSITION of the members' identity atoms -- a member reads as present in the group,
    which is exactly bundle + cleanup. Moving the group's transform moves every member (transform composition up
    the hierarchy), and ungrouping just re-parents the members and drops the null.

  * INSTANCING is a BIND. An instance shares ONE source geometry but carries its OWN transform -- the instance is
    the source geometry "bound to" a placement. Nothing is copied: every instance reads the source's geometry
    through its handle, so editing the source updates all instances at once (the whole point of instancing, and
    what makes a forest of 10,000 trees cost one tree plus 10,000 transforms).

Both write through the canonical Scene (scene.add / set_parent / remove), wrapped in a scene.group(...) undo
transaction so each is a SINGLE undo. Deterministic; NumPy + stdlib only.
"""


# ---- grouping (a bundle) -------------------------------------------------------------------------------------

def group_objects(scene, handles, name="Group"):
    """Create a GROUP: a null parent object owning `handles` (via the hierarchy). Returns the group's handle. One
    undo step. (Note: scene.group(...) here is the Scene's undo TRANSACTION, not this grouping -- different thing,
    same word.)"""
    with scene.group("Group " + name):
        g = scene.add(name=name, geometry=None, params={"is_group": True})
        for h in handles:
            scene.set_parent(h, g)
    return g


def ungroup(scene, group_handle):
    """Dissolve a group: re-parent its members to the group's own parent (or to the top level), then remove the
    null. One undo step. Returns the freed member handles."""
    members = group_members(scene, group_handle)
    parent = scene.parent_of(group_handle)
    with scene.group("Ungroup"):
        for child in members:
            scene.set_parent(child, parent)                  # re-parent to the group's parent (None = top level)
        scene.remove(group_handle)
    return members


def group_members(scene, group_handle):
    """The direct members of a group (its children in the hierarchy)."""
    return scene.children_of(group_handle)


def is_group(scene, handle):
    return bool(scene.get(handle).params.get("is_group"))


def group_bundle(scene, group_handle):
    """The group's identity as a BUNDLE of its members' identity atoms -- the holographic representation of the
    group (a member reads as present by high cosine). Returns None for an empty group."""
    from holographic.agents_and_reasoning.holographic_ai import bundle
    vecs = [scene.handle_vector(h) for h in group_members(scene, group_handle)]
    return bundle(vecs) if vecs else None


# ---- instancing (a bind) -------------------------------------------------------------------------------------

def instance(scene, source_handle, transform=None, name=None):
    """Create an INSTANCE of a source object: it SHARES the source's geometry (read through the source handle) but
    has its OWN transform. Editing the source geometry updates every instance. Returns the instance handle."""
    src = scene.get(source_handle)
    with scene.group("Instance " + src.name):
        h = scene.add(name=name or (src.name + "_inst"), transform=transform, geometry=None,
                      material=src.material, params={"instance_of": source_handle})
    return h


def instance_source(scene, handle):
    """The source an object is an instance OF, or None if it is a normal object."""
    return scene.get(handle).params.get("instance_of")


def resolve_geometry(scene, handle):
    """The geometry the renderer should USE for an object: an instance reads its SOURCE's geometry (shared -- so
    editing the source updates all instances); a normal object uses its own. This is the 'unbind the geometry
    filler' step -- the transform is the bound placement, the geometry is the shared filler."""
    src = instance_source(scene, handle)
    return scene.get(src).geometry if src is not None else scene.get(handle).geometry


def instances_of(scene, source_handle):
    """Every instance that points at `source_handle`."""
    return [h for h, o in scene.objects.items() if o.params.get("instance_of") == source_handle]


def _selftest():
    """Grouping parents members under a null (its bundle recognizes a member) and ungrouping frees them; both are
    ONE undo step. Instancing shares the source geometry (editing the source updates the instance) while keeping
    its own transform; deterministic."""
    import numpy as np
    from holographic.scene_and_pipeline.holographic_scene_doc import Scene
    from holographic.agents_and_reasoning.holographic_ai import cosine

    scene = Scene(dim=256, seed=0)
    a = scene.add(name="wheel_l", geometry=np.zeros((4, 3)))
    b = scene.add(name="wheel_r", geometry=np.ones((4, 3)))
    c = scene.add(name="body", geometry=np.full((4, 3), 2.0))

    # (1) GROUP a and b -> a null parent owns them; it's one undo step
    steps_before = len(scene._undo)
    g = group_objects(scene, [a, b], name="wheels")
    assert set(group_members(scene, g)) == {a, b}
    assert is_group(scene, g) and scene.parent_of(a) == g and scene.parent_of(b) == g
    assert len(scene._undo) == steps_before + 1              # ONE undo step for the whole grouping

    # (2) the group's identity is a BUNDLE: a member reads as present, a non-member less so
    gb = group_bundle(scene, g)
    assert cosine(gb, scene.handle_vector(a)) > cosine(gb, scene.handle_vector(c))

    # (3) one undo dissolves the group... wait, grouping was the last step -> undo removes the group
    scene.undo()
    assert g not in scene.objects and scene.parent_of(a) is None   # members back at top level, null gone
    scene.redo()
    assert set(group_members(scene, g)) == {a, b}

    # (4) UNGROUP frees the members and removes the null
    ungroup(scene, g)
    assert g not in scene.objects and scene.parent_of(a) is None and scene.parent_of(b) is None

    # (5) INSTANCE of c: shares c's geometry, own transform
    T = np.eye(4); T[0, 3] = 5.0
    inst = instance(scene, c, transform=T, name="body_copy")
    assert instance_source(scene, inst) == c
    assert np.allclose(resolve_geometry(scene, inst), scene.get(c).geometry)   # shared geometry
    assert np.allclose(scene.get(inst).transform, T)                          # its own placement
    assert inst in instances_of(scene, c)

    # (6) editing the SOURCE geometry updates the instance (nothing was copied)
    scene.edit(c, geometry=np.full((4, 3), 9.0))
    assert np.allclose(resolve_geometry(scene, inst), 9.0)    # the instance follows the source

    # (7) deterministic
    s2 = Scene(dim=256, seed=0); x = s2.add(name="x"); y = s2.add(name="y")
    g1 = group_objects(s2, [x, y]); assert set(group_members(s2, g1)) == {x, y}

    print("holographic_grouping selftest OK: grouping parents members under a null (its bundle recognizes a "
          "member over a non-member) and is ONE undo step; ungrouping frees them; an instance SHARES the source "
          "geometry (editing the source updates the instance) while keeping its own transform; deterministic")


if __name__ == "__main__":
    _selftest()
