"""Skinning / rigging (FWD-9): linear blend skinning as a SOFT mixture of expert bone-transforms.

WHY THIS MODULE EXISTS
----------------------
Tier 2, the last core item -- and the one whose "reuse" claim needed the most honesty. Skinning deforms a mesh by
attaching each vertex to one or more bones: the deformed position is a WEIGHTED COMBINATION of what each bone's
transform would do to that vertex, with the per-vertex skin weights summing to one. Structurally that is a mixture
of experts -- each bone is an expert transform, the skin weights are the gate.

THE HONEST REUSE PICTURE (the finding, reported not buried)
  holostuff already has a mixture of experts: `holographic_moe.GatedMixture`. But it is the HARD, SPARSE, LEARNED
  kind -- a top-1 router whose gate is the creature brain, trained from outcomes, where only the chosen expert
  runs. Linear blend skinning is the OPPOSITE regime: a SOFT, DENSE, FIXED mixture -- every bone contributes,
  weighted by painted weights that form a partition of unity, with no learning and no winner-take-all. So the moe
  connection is real but CONCEPTUAL, not a literal call: skinning is the soft/dense cousin of the engine's
  hard/sparse GatedMixture. Same "experts + gating" skeleton, different gating regime. Naming that difference is
  more useful than pretending LBS routes through a top-1 gate it does not.

WHAT IT PROVIDES
  * linear_blend_skin(vertices, transforms, weights) -- the classic LBS: v' = sum_b w_b (M_b v), weights row-
    normalised to a partition of unity. Returns deformed (V,3).
  * skin_mesh(mesh, transforms, weights) -- the same, returning a new Mesh (deformed vertices, faces untouched).
  * make_transform(...) / rotation(axis, angle) -- build the 4x4 bone transforms (Rodrigues rotation + translation).

THE MEASUREMENT BAR (checked exactly in the self-test)
  * RIGID REPRODUCTION (the partition-of-unity guarantee, LBS's analogue of subdivision's affine reproduction):
    if every bone shares the same rigid transform M, LBS reproduces M EXACTLY on every vertex, for ANY weights.
  * A single-bone (weight 1) vertex gets exactly that bone's transform; identity transforms leave the mesh fixed.

THE KEPT NEGATIVE, MEASURED EXACTLY (this is the point of the module)
  LBS averages the bone MATRICES, not the rotations -- so a vertex blended 50/50 between two bones with a large
  relative TWIST collapses toward the bone axis (the infamous "candy-wrapper" artifact). It is not vague: for a
  unit ring twisted by angle theta, the blended radius is EXACTLY |cos(theta/2)| of the original -- 0.5 at 120
  degrees, 0 (full collapse) at 180. The self-test asserts that closed form. Dual-quaternion skinning fixes this
  by blending rotations properly; that is the honest next step, not shipped here.

DETERMINISM (per ISA.md)
  Pure linear algebra, no RNG; weights normalised deterministically. Same inputs -> byte-identical output.
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import Mesh


def rotation(axis, angle):
    """A 3x3 rotation matrix about `axis` by `angle` radians (Rodrigues' formula).

    KEPT SEPARATE, on purpose (rev. 9 organization audit): the engine's canonical builder is
    `holographic_transform.rotation_axis_angle` (quaternion-based), and `holographic_scenegraph.rotation` keeps a
    third, 4x4 Rodrigues. Measured: this one differs from BOTH by up to ~9.0e-12 (the `+1e-12` in the axis
    normalization plus the quaternion round-trip). Bit-identity is the merge gate, it fails, and skinning weights
    baked against this exact matrix must not move -- so the copy stays, DECLARED, with the number."""
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def make_transform(rot=None, translation=(0.0, 0.0, 0.0), axis=None, angle=0.0):
    """A 4x4 homogeneous bone transform from a rotation (a 3x3 matrix `rot`, or an `axis`/`angle` pair) and a
    translation. Identity by default."""
    M = np.eye(4)
    if rot is not None:
        M[:3, :3] = np.asarray(rot, float)
    elif axis is not None:
        M[:3, :3] = rotation(axis, angle)
    M[:3, 3] = np.asarray(translation, float)
    return M


def linear_blend_skin(vertices, transforms, weights):
    """Linear blend skinning: deform each vertex as the weighted combination of what each bone transform would do
    to it -- v' = sum_b w_b (M_b v), with weights row-normalised to a partition of unity (the gate). `transforms`
    is (B,4,4), `weights` is (V,B). Returns deformed (V,3). This is the SOFT/DENSE mixture of expert transforms
    (the soft cousin of holographic_moe's hard top-1 GatedMixture)."""
    V = np.asarray(vertices, float)
    T = np.asarray(transforms, float)
    W = np.asarray(weights, float)
    W = W / (W.sum(axis=1, keepdims=True) + 1e-12)         # partition of unity (the gate)
    homog = np.hstack([V, np.ones((len(V), 1))])           # (V,4)
    out = np.zeros((len(V), 3))
    for b in range(T.shape[0]):
        transformed = homog @ T[b].T                       # what bone b would do to every vertex
        out += W[:, b:b + 1] * transformed[:, :3]          # gated contribution
    return out


def skin_mesh(mesh, transforms, weights):
    """Linear-blend-skin a mesh, returning a new Mesh with the deformed vertices and the same faces."""
    return Mesh(linear_blend_skin(mesh.vertices, transforms, weights), [tuple(f) for f in mesh.faces])


# =====================================================================================================
# Self-test -- rigid reproduction (partition of unity), single-bone exactness, and the candy-wrapper collapse.
# =====================================================================================================
def _selftest():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((20, 3))

    # --- RIGID REPRODUCTION: all bones share one rigid transform -> LBS reproduces it exactly, for ANY weights ---
    M = make_transform(rot=rotation([0.3, 1.0, 0.2], 0.7), translation=[0.5, -0.2, 1.0])
    transforms = np.stack([M, M, M])                       # three bones, same transform
    weights = rng.uniform(0.1, 1.0, (20, 3))               # arbitrary weights
    skinned = linear_blend_skin(pts, transforms, weights)
    expected = (np.hstack([pts, np.ones((20, 1))]) @ M.T)[:, :3]
    assert np.allclose(skinned, expected, atol=1e-12), "shared transform must be reproduced exactly (partition of unity)"

    # --- identity transforms leave the mesh fixed; a single-bone vertex gets exactly that bone's transform ---
    ident = np.stack([np.eye(4), np.eye(4)])
    assert np.allclose(linear_blend_skin(pts, ident, np.ones((20, 2))), pts, atol=1e-12)
    two = np.stack([M, make_transform(translation=[10, 0, 0])])
    w_first = np.zeros((20, 2)); w_first[:, 0] = 1.0       # 100% bone 0
    assert np.allclose(linear_blend_skin(pts, two, w_first), expected, atol=1e-12)

    # --- THE CANDY-WRAPPER NEGATIVE: a unit ring twisted theta, blended 50/50, has radius EXACTLY |cos(theta/2)| ---
    phi = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    ring = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=1)   # unit ring in the z=0 plane
    half = np.full((32, 2), 0.5)
    for theta in (np.pi / 2, 2 * np.pi / 3, np.pi):        # 90, 120, 180 degrees of twist about z
        bones = np.stack([np.eye(4), make_transform(axis=[0, 0, 1], angle=theta)])
        twisted = linear_blend_skin(ring, bones, half)
        radius = float(np.mean(np.linalg.norm(twisted[:, :2], axis=1)))
        assert abs(radius - abs(np.cos(theta / 2))) < 1e-9, f"LBS collapse radius must be cos(theta/2): {radius}"
    collapse_180 = float(np.mean(np.linalg.norm(
        linear_blend_skin(ring, np.stack([np.eye(4), make_transform(axis=[0, 0, 1], angle=np.pi)]), half)[:, :2], axis=1)))

    # --- determinism ---
    assert np.array_equal(linear_blend_skin(pts, transforms, weights), linear_blend_skin(pts, transforms, weights))

    print(f"holographic_meshskin selftest: ok (linear blend skinning as a soft mixture of expert bone-transforms: "
          f"shared rigid transform reproduced EXACTLY for any weights (partition of unity); single-bone exact; "
          f"CANDY-WRAPPER negative measured to closed form -- a 50/50 twist collapses the radius to cos(theta/2), "
          f"reaching {collapse_180:.3f} at 180 degrees; deterministic)")


if __name__ == "__main__":
    _selftest()
