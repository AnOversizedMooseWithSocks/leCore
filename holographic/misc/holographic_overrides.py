"""holographic_overrides.py -- RENDER OVERRIDES: a bound role with fallback (modeling-app feature layer).

Thinking holographically: a render setting an object CAN override (its sample count, denoise mode, visibility, a
material tweak) is a BOUND ROLE on that object, and resolving it is bind-with-fallback -- read the object's bound
value if present, else fall back to the scene default. Only the DELTAS are stored per object; everything else is
INHERITED, so a scene of thousands with a couple of special objects costs a couple of entries, not thousands of
copies. That is exactly how a DCC's per-object render settings and material overrides work.

The fallback chain is: the object's own override -> (its material's override, if the material carries a dict of
them) -> the scene default -> a bare default. A cleanup that finds "no bound value here" simply hands control to
the next tier down. Overrides are written through scene.edit / scene.clear_override, so they get undo and change
events for free. NumPy-free; stdlib only; deterministic.
"""


def set_override(scene, handle, prop, value):
    """Bind a render override on an object (through scene.edit, so undo + change events come for free)."""
    scene.edit(handle, overrides={prop: value})


def clear_override(scene, handle, prop):
    """Remove an object's override for `prop` so it falls back to the default."""
    scene.clear_override(handle, prop)


def resolve(scene, handle, prop, defaults=None, default=None):
    """Resolve a render property for one object, the bound-role-with-fallback way:
        1. the object's OWN override wins;
        2. else its material's override, if the material is a dict carrying overrides;
        3. else the scene `defaults` (a dict of prop -> value);
        4. else the bare `default`.
    Returns the effective value the renderer should use for this object."""
    obj = scene.get(handle)
    if prop in obj.overrides:
        return obj.overrides[prop]                          # the object's bound value wins
    mat = obj.material
    if isinstance(mat, dict) and prop in mat:               # a material-level override tier
        return mat[prop]
    if defaults is not None and prop in defaults:
        return defaults[prop]
    return default                                          # the final fallback


def effective_settings(scene, handle, props, defaults=None):
    """Resolve a whole set of props for an object into a dict of effective values -- what the renderer actually
    uses, after the fallback chain is applied to each."""
    return {p: resolve(scene, handle, p, defaults=defaults) for p in props}


def overridden_props(scene, handle):
    """The props this object actually overrides (the deltas it carries) -- what a property panel shows in bold."""
    return dict(scene.get(handle).overrides)


def _selftest():
    """An object with no override inherits the scene default; setting one makes it win; a material dict is the
    middle tier; clearing falls back; overrides are undoable; only deltas are stored."""
    from holographic.scene_and_pipeline.holographic_scene_doc import Scene

    scene = Scene(dim=256, seed=0)
    defaults = {"samples": 64, "denoise": "svgf", "visible": True}

    a = scene.add(name="hero")                              # no overrides -> inherits everything
    b = scene.add(name="prop", overrides={"samples": 256})  # a hero object that wants more samples

    # (1) fallback: `a` inherits all defaults; `b` overrides only samples, inherits the rest
    assert resolve(scene, a, "samples", defaults) == 64      # inherited
    assert resolve(scene, b, "samples", defaults) == 256     # overridden -- the bound value wins
    assert resolve(scene, b, "denoise", defaults) == "svgf"  # not overridden -> inherited

    # (2) only DELTAS are stored (not a full copy of every setting)
    assert overridden_props(scene, a) == {}
    assert overridden_props(scene, b) == {"samples": 256}

    # (3) set/clear through the module -> undoable, and clearing falls back
    set_override(scene, a, "samples", 512)
    assert resolve(scene, a, "samples", defaults) == 512
    clear_override(scene, a, "samples")
    assert resolve(scene, a, "samples", defaults) == 64      # back to the default
    scene.undo()                                            # the clear was recorded
    assert resolve(scene, a, "samples", defaults) == 512     # the override is back

    # (4) a MATERIAL-level tier sits between object and scene default
    c = scene.add(name="glass", material={"denoise": "bilateral"})   # material carries an override
    assert resolve(scene, c, "denoise", defaults) == "bilateral"     # material beats the scene default
    scene.edit(c, overrides={"denoise": "off"})
    assert resolve(scene, c, "denoise", defaults) == "off"           # ...but the object beats the material

    # (5) effective settings resolves a whole set at once
    eff = effective_settings(scene, b, ["samples", "denoise", "visible"], defaults)
    assert eff == {"samples": 256, "denoise": "svgf", "visible": True}

    print("holographic_overrides selftest OK: an object with no override inherits the scene default; an override "
          "is a bound value that wins; a material dict is the middle tier (object > material > scene default); "
          "clearing falls back and is undoable; only the deltas are stored, not a per-object copy of every setting")


if __name__ == "__main__":
    _selftest()
