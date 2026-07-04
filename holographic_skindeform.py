"""holographic_skindeform.py -- make an imported rig actually MOVE.

The importer (holographic_assetimport) reads a glTF's skeleton, per-vertex skin weights, animation clips, and morph
targets, but it stops at reading them -- it hands you the DATA, not a deformed mesh. This module is the deformer that
closes the loop: given a LoadedMesh and a time t, it produces the deformed vertex positions.

Two standard operations, both classic and readable:

  * LINEAR-BLEND SKINNING (the skeleton). Each vertex is bound to a few joints with weights that sum to one. The
    deformed position is the weighted blend of what each joint would do to it. glTF's recipe for one joint's transform
    is  global_joint_transform @ inverse_bind_matrix  -- the inverse-bind moves the vertex from mesh space into the
    joint's bind-pose local space, then the joint's CURRENT global transform moves it to where the joint is now. We
    reuse the engine's own linear_blend_skin (holographic_meshskin) for the final blend; the work here is turning a
    glTF rig into its inputs (compose the node hierarchy into global joint transforms, expand the sparse per-vertex
    JOINTS_0/WEIGHTS_0 into the dense weight matrix that kernel wants).

  * MORPH-TARGET BLENDING (blend shapes). The mesh has a base shape plus a few "target" shapes (position deltas); a
    per-target weight blends them in:  v = base + sum_i weight_i * delta_i.  The weights can be animated (a glTF
    'weights' channel), so a face can smile over time.

`deform(loaded, clip, t)` does morph first (on the base shape), then skinning (the skeleton moves the morphed shape),
which is the order glTF specifies.

Readable + NumPy only. Reuses holographic_meshskin.linear_blend_skin for the blend and Mesh for the result.
"""
import numpy as np


def global_transforms(node_graph, overrides=None):
    """Walk the node hierarchy and return the GLOBAL 4x4 transform of every node: global = parent_global @ local,
    starting from the roots with identity. `overrides` is {node_index: local 4x4} -- the animated local transforms
    from an AnimationClip.sample(t); any node not overridden uses its REST local. Nodes are a list of dicts with
    'local' (4x4) and 'children' (indices), exactly what the importer's node_graph provides."""
    n = len(node_graph)
    overrides = overrides or {}

    # a node is a ROOT if nobody lists it as a child
    is_child = set()
    for nd in node_graph:
        is_child.update(nd["children"])
    roots = [i for i in range(n) if i not in is_child]

    G = [None] * n

    def visit(i, parent_global):
        local = overrides.get(i, node_graph[i]["local"])
        G[i] = parent_global @ local
        for c in node_graph[i]["children"]:
            visit(c, G[i])

    for r in roots:
        visit(r, np.eye(4))
    return [g if g is not None else np.eye(4) for g in G]      # any orphan (shouldn't happen) stays at identity


def _dense_weights(joints, weights, n_joints):
    """Expand the sparse per-vertex skin binding -- JOINTS_0 (V,4) joint indices + WEIGHTS_0 (V,4) weights -- into the
    dense (V, n_joints) weight matrix that linear_blend_skin wants. Each of a vertex's up-to-4 bindings scatters its
    weight into the column of the joint it names."""
    V = len(joints)
    W = np.zeros((V, n_joints), float)
    jj = np.asarray(joints, int)                               # (V, 4) -- indices into the skin's joint list
    ww = np.asarray(weights, float)                            # (V, 4)
    rows = np.repeat(np.arange(V), jj.shape[1])
    cols = jj.reshape(-1)
    np.add.at(W, (rows, np.clip(cols, 0, n_joints - 1)), ww.reshape(-1))
    return W


def skin_positions(loaded, clip=None, t=0.0, base=None):
    """Deform positions by the FIRST skin at time t via linear-blend skinning. `base` overrides loaded.positions (so a
    morphed shape can be skinned). Returns (V,3). If the mesh isn't skinned, returns the base unchanged."""
    from holographic_meshskin import linear_blend_skin

    pts = loaded.positions if base is None else np.asarray(base, float)
    if not loaded.skins or loaded.joints is None or loaded.weights is None or not loaded.node_graph:
        return pts                                             # nothing to skin -> pass the shape through

    overrides = clip.sample(t) if clip is not None else {}     # animated joint local transforms at time t
    G = global_transforms(loaded.node_graph, overrides)        # global transform per NODE

    skin = loaded.skins[0]
    joint_nodes = skin["joints"]                               # node index of each skin-joint
    ibm = skin["inverse_bind"]                                 # (J,4,4) or None (None -> identity bind)
    J = len(joint_nodes)

    # one transform per joint: where the joint is now (global) composed with its inverse-bind
    transforms = np.zeros((J, 4, 4))
    for b in range(J):
        gj = G[joint_nodes[b]] if joint_nodes[b] < len(G) else np.eye(4)
        transforms[b] = gj @ (ibm[b] if ibm is not None else np.eye(4))

    W = _dense_weights(loaded.joints, loaded.weights, J)       # (V, J) dense weights aligned with `transforms`
    return linear_blend_skin(pts, transforms, W)


def apply_morph(base, targets, weights):
    """Blend morph targets onto a base shape: base + sum_i weights[i] * targets[i]. `targets` is (T,V,3), `weights`
    is (T,). Missing targets/weights -> the base is returned unchanged."""
    out = np.array(base, float)
    if targets is None or weights is None:
        return out
    for i, w in enumerate(weights):
        if i < len(targets) and w != 0.0:
            out += float(w) * np.asarray(targets[i], float)
    return out


def morph_weights_at(loaded, clip, t):
    """The morph weights to use at time t: an animated 'weights' channel if the clip drives one, else the mesh's
    default morph weights."""
    if clip is not None and loaded.morph_targets is not None:
        for node in clip.nodes():
            w = clip.sample_channel(node, "weights", t)
            if w is not None:
                return np.asarray(w, float)
    return loaded.morph_weights


def deform(loaded, clip=None, t=0.0):
    """The full deform at time t: morph-blend the base shape (if the mesh has morph targets), then skin it (if the mesh
    has a skeleton). Returns a deformed Mesh (positions moved, faces unchanged). With clip=None you get the REST pose
    (skinning by the rest hierarchy, default morph weights) -- a no-op for an unrigged mesh."""
    from holographic_mesh import Mesh

    shape = apply_morph(loaded.positions, loaded.morph_targets, morph_weights_at(loaded, clip, t))   # blend shapes
    shape = skin_positions(loaded, clip=clip, t=t, base=shape)                                        # then the skeleton
    return Mesh(shape, [tuple(f) for f in loaded.faces])


def deformed_positions(loaded, clip=None, t=0.0):
    """Just the deformed (V,3) positions at time t (deform() without wrapping a Mesh)."""
    shape = apply_morph(loaded.positions, loaded.morph_targets, morph_weights_at(loaded, clip, t))
    return skin_positions(loaded, clip=clip, t=t, base=shape)


def _selftest():
    from holographic_assetimport import LoadedMesh, AnimationClip

    # ---- SKINNING: one bone (node 1) that translates; a vertex fully bound to it must follow ----------------
    node_graph = [
        {"name": "root", "local": np.eye(4), "children": [1]},          # node 0: the mesh's node
        {"name": "bone", "local": np.eye(4), "children": []},           # node 1: the joint (rest at origin)
    ]
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    joints = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])                     # both verts -> skin-joint 0
    weights = np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    skins = [{"joints": [1], "inverse_bind": np.eye(4)[None]}]          # skin-joint 0 == node 1, identity bind
    lm = LoadedMesh(positions, [(0, 1, 0)], joints=joints, weights=weights, skins=skins, node_graph=node_graph)

    # a clip translating node 1 by +2 on Y over 1s
    clip = AnimationClip("wave", {1: {"translation": (np.array([0.0, 1.0]),
                                                      np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))}})
    rest = deformed_positions(lm, clip=None)                           # rest pose = unchanged
    assert np.allclose(rest, positions), "rest pose should not move the mesh"
    moved = deformed_positions(lm, clip=clip, t=1.0)                   # at t=1 the bone is +2 on Y
    assert np.allclose(moved, positions + np.array([0.0, 2.0, 0.0])), moved
    half = deformed_positions(lm, clip=clip, t=0.5)                    # halfway
    assert np.allclose(half, positions + np.array([0.0, 1.0, 0.0])), half

    # ---- global_transforms: a child inherits its parent's transform -----------------------------------------
    G = global_transforms(node_graph, overrides={0: _translate(1.0, 0.0, 0.0)})
    assert np.allclose(G[1][:3, 3], [1.0, 0.0, 0.0]), "child should inherit the parent's translation"

    # ---- MORPH: base + weight*delta -------------------------------------------------------------------------
    targets = np.array([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])          # one target: lift both verts by 1 on Y
    lm2 = LoadedMesh(positions, [(0, 1, 0)], morph_targets=targets, morph_weights=np.array([0.5]))
    m = deformed_positions(lm2, clip=None)                            # weight 0.5 -> lifted by 0.5
    assert np.allclose(m, positions + np.array([0.0, 0.5, 0.0])), m

    print("OK: holographic_skindeform self-test passed (linear-blend skinning follows an animated bone: rest "
          "unchanged, +2 at t=1, +1 at t=0.5; a child node inherits its parent's transform; morph target blends "
          "base + weight*delta)")


def _translate(x, y, z):
    M = np.eye(4); M[:3, 3] = [x, y, z]; return M


if __name__ == "__main__":
    _selftest()
