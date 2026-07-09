"""Scene component delta + dedup measurement (holographic_scenedelta).

WHY THIS MODULE EXISTS -- AND THE HONEST SCOPE
----------------------------------------------
From the geometry->stack backlog (reverse item R6, "cluster/scene delta"): store scene variants as a shared base
plus per-variant deltas. The investigation measured this on the real scene-graph and found the honest scope:

  THE SAVING IS AUTOMATIC. scene_to_recipe names every component (mesh, transform) by its CONTENT HASH, so two
  scenes that share a subtree already share the identical atom. Stored in any content-addressed table (atom by hash),
  the common components dedup for FREE -- measured 5.79x fewer stored components across a base + 8 variants that each
  change one of four subtrees. There is no new delta ALGEBRA to invent here; content-addressing already does the
  sharing. (This is the same reason a content-addressed blob store dedups a repo.)

So this module does NOT claim a novel mechanism. It ships the two genuinely-useful operations that are NOT automatic:

  * the explicit DIFF between two scenes -- which components were added and removed -- so a variant can be
    TRANSMITTED as its delta (send the base once, then small deltas) rather than re-sent whole; and
  * the MEASUREMENT of the dedup saving across a scene set, so the sharing is visible and quantified.

The full scene tree is reconstructed by the existing recipe (the wiring is small); the heavy data -- the mesh and
transform components -- is what dedups, and what the delta moves.

WHAT IT PROVIDES
  * scene_components(scene) -- the set of content-hashed component ids a scene uses (the handle the sharing keys on).
  * scene_delta(base, variant) -- {'added', 'removed'}: the component diff, for delta transmission / versioning.
  * apply_scene_delta(base_components, delta) -- the variant's component set rebuilt from base + delta.
  * scene_dedup_saving(scenes) -- {'naive', 'unique', 'saving_x'}: the content-addressed dedup measurement.

THE MEASUREMENT BAR (checked exactly in the self-test)
  * a variant that changes one subtree yields a SMALL delta (a couple of components) vs its full component count.
  * apply_scene_delta(base, scene_delta(base, variant)) reconstructs the variant's component set EXACTLY.
  * an identical scene yields an EMPTY delta.
  * dedup across a base + variants gives a saving well above 1x (measured ~5-6x).

DETERMINISM (per ISA.md)
  Content hashes are deterministic (hashlib); the diffs are set operations -- same scenes give the same delta and
  the same saving (asserted).

KEPT NEGATIVES (loud)
  * The dedup saving is AUTOMATIC from content-addressed atoms, NOT a contribution of this module -- this exposes and
    measures it and adds the transmittable diff. Stated, not dressed up as a new mechanism.
  * The delta is over COMPONENTS (the heavy mesh/transform atoms). The scene TREE wiring is reconstructed by the
    recipe, not by this; a delta that only re-wires shared components (no new component) reads as an empty component
    delta though the scene changed. Component-level versioning, not full structural diff.
  * Sharing requires BIT-IDENTICAL components (the hash is exact) -- a near-but-not-identical mesh does not dedup.
    That is the content-addressing contract, and the reason geometric quantization (which makes near-identical
    things identical) matters upstream.
"""

import hashlib

import numpy as np

from holographic.scene_and_pipeline.holographic_scenegraph import scene_to_recipe


def _component_ids(scene, dim=512, seed=0):
    """The content-hashed component (atom) ids a scene's recipe uses -- mesh and transform atoms, named by content
    hash so shared subtrees share ids. Returns a list (with multiplicity as the recipe emits them)."""
    r = scene_to_recipe(scene, dim=dim, seed=seed)
    return [op[1] for op in r._ops if op[0] == "atom"]


def scene_components(scene):
    """The SET of content-hashed component ids a scene uses -- the handle the content-addressed sharing keys on."""
    return frozenset(_component_ids(scene))


def scene_delta(base, variant):
    """The component DIFF turning `base` into `variant`: {'added', 'removed'} (frozensets of component ids). A
    variant is transmitted as this delta -- send the base once, then the small deltas -- instead of re-sent whole.
    An identical scene gives empty added/removed."""
    b, v = scene_components(base), scene_components(variant)
    return {"added": frozenset(v - b), "removed": frozenset(b - v)}


def apply_scene_delta(base_components, delta):
    """Rebuild a variant's component set from a base component set and a delta -- (base - removed) | added. The
    inverse of scene_delta at the component level."""
    base = frozenset(base_components)
    return (base - frozenset(delta["removed"])) | frozenset(delta["added"])


def scene_dedup_saving(scenes):
    """The content-addressed dedup saving across a set of scenes: {'naive' (sum of per-scene component counts),
    'unique' (distinct components stored once), 'saving_x'}. Quantifies what the automatic sharing buys."""
    naive = 0
    union = set()
    for s in scenes:
        comps = scene_components(s)
        naive += len(comps)
        union |= set(comps)
    unique = len(union)
    return {"naive": naive, "unique": unique, "saving_x": (naive / unique) if unique else 1.0}


# =====================================================================================================
# Self-test -- small deltas, exact reconstruction, empty delta on identity, measured dedup saving.
# =====================================================================================================
def _selftest():
    from holographic.scene_and_pipeline.holographic_scenegraph import SceneNode, translation
    from holographic.mesh_and_geometry.holographic_mesh import box

    cube = box()
    other = box(2, 1, 1)

    def variant(i, changed=True):
        children = [SceneNode(translation([0, 0, 0]), mesh=cube),
                    SceneNode(translation([2, 0, 0]), mesh=cube),
                    SceneNode(translation([0, 2, 0]), mesh=other),
                    SceneNode(translation([2, 2, 0]), mesh=cube)]
        if changed:
            children[i % 4] = SceneNode(translation([float(i), 5, 0]), mesh=cube)
        return SceneNode(children=children)

    base = variant(0, changed=False)

    # --- a one-subtree change yields a small delta vs the full component count ---
    var = variant(1)
    d = scene_delta(base, var)
    full = len(scene_components(var))
    assert len(d["added"]) + len(d["removed"]) < full, "a one-subtree change must be a small delta"

    # --- reconstruction is exact at the component level ---
    rebuilt = apply_scene_delta(scene_components(base), d)
    assert rebuilt == scene_components(var), "base + delta must rebuild the variant's components exactly"

    # --- an identical scene yields an empty delta ---
    same = scene_delta(base, variant(0, changed=False))
    assert not same["added"] and not same["removed"], "an identical scene must give an empty delta"

    # --- measured dedup saving across a base + variants ---
    scenes = [base] + [variant(i) for i in range(8)]
    sav = scene_dedup_saving(scenes)
    assert sav["saving_x"] > 2.0, f"content-addressed sharing should save well above 1x, got {sav['saving_x']:.2f}"

    # --- determinism ---
    assert scene_delta(base, var) == scene_delta(base, var)
    assert scene_dedup_saving(scenes) == scene_dedup_saving(scenes)

    print(f"holographic_scenedelta selftest: ok (one-subtree change -> delta {len(d['added'])}+{len(d['removed'])} vs "
          f"full {full} components; base+delta rebuilds the variant exactly; identical scene -> empty delta; dedup "
          f"across {len(scenes)} scenes saves {sav['saving_x']:.2f}x ({sav['naive']} -> {sav['unique']} components); "
          f"deterministic. NOTE: the dedup is automatic from content-addressing; this adds the diff + measurement)")


if __name__ == "__main__":
    _selftest()
