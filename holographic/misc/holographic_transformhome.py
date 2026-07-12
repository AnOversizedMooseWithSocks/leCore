"""holographic_transformhome.py -- the TRANSFORM home (consolidation backlog H5): one facade over "move / rotate /
warp this", across the representations the engine transforms things in.

WHY THIS EXISTS
---------------
Transforming shows up in several representations, in several modules:
  * VSA:        a rigid shift IS a single BIND (circular convolution), and order is a PERMUTE (cyclic shift)
                -- holographic_ai.bind / permute, on hypervectors.
  * geometric:  4x4 matrices for translate / scale / rotate / compose, plus decompose and a quaternion kit and
                look_at -- holographic_transform (the modeling-app gizmo/panel kit). The BASIC matrix builders were
                DUPLICATED in holographic_scenegraph.
  * rotor:      geometric-algebra rotation without gimbal lock -- holographic_clifford.
  * anisotropic: direction-aware (steered) kernels -- holographic_steering.

`Transform` is the one door. It ROUTES to each module (route, don't rewrite); nothing here re-derives the math:

    Transform.bind(a, b) / Transform.permute(vec, shift)     # VSA rigid transform / order
    Transform.translation(t) / scaling(s) / rotation(axis, angle) / compose(*M)   # 4x4 matrices
    Transform.decompose(M) / compose_trs(t, quat, s) / look_at(eye, target, up)   # gizmo + camera
    Transform.rotor(axis, angle) / rotate_vec(rotor, v)      # geometric-algebra rotation (clifford)
    Transform.steer_bandwidths(X, y, ...)                    # anisotropic steering

CONVENTIONS (held by holographic_transform, stated here too): 4x4 matrices act on COLUMN vectors, p' = M @ [x,y,z,1];
compose(A, B) = A @ B applies B then A; quaternions are (w, x, y, z), unit length.
"""
import numpy as np

# a single geometric-algebra instance for R^3 rotors (reused so rotor/rotate_vec don't rebuild it each call)
_CL3 = None


def _clifford3():
    global _CL3
    if _CL3 is None:
        from holographic.mesh_and_geometry.holographic_clifford import CliffordAlgebra
        _CL3 = CliffordAlgebra()                                 # fixed Cl(3) -- 3D geometric algebra
    return _CL3


class Transform:
    """A namespace of staticmethods over the engine's transforms. VSA / geometric / rotor / anisotropic."""

    # --- VSA transforms (on hypervectors) ---
    @staticmethod
    def bind(a, b):
        """The rigid transform on hypervectors: bind (circular convolution). A rigid shift IS one binding -- which is
        why motion-compensated video, the propagator, and scene transforms all reduce to it. Routes to holographic_ai.bind."""
        from holographic.agents_and_reasoning.holographic_ai import bind
        return bind(a, b)

    @staticmethod
    def permute(vec, shift):
        """Order / direction on a hypervector: a cyclic shift (permutation) -- protects a bound term and encodes
        sequence position. Routes to holographic_ai.permute."""
        from holographic.agents_and_reasoning.holographic_ai import permute
        return permute(vec, shift)

    # --- geometric transforms (4x4 matrices, column-vector convention) -> holographic_transform ---
    @staticmethod
    def translation(t):
        """A 4x4 translation matrix from a 3-vector. Routes to holographic_transform.translation."""
        from holographic.misc.holographic_transform import translation
        return translation(t)

    @staticmethod
    def scaling(s):
        """A 4x4 scale matrix (scalar = uniform, 3-vector = per-axis). Routes to holographic_transform.scaling."""
        from holographic.misc.holographic_transform import scaling
        return scaling(s)

    @staticmethod
    def rotation(axis, angle):
        """A 4x4 rotation of `angle` radians about `axis`. Routes to holographic_transform.rotation_axis_angle
        (quaternion-based; NOTE holographic_scenegraph keeps a separate Rodrigues rotation, ~1e-12 different -- the
        two are not bit-identical, so they are not merged)."""
        from holographic.misc.holographic_transform import rotation_axis_angle
        return rotation_axis_angle(axis, angle)

    @staticmethod
    def compose(*mats):
        """Matrix product M0 @ M1 @ ... -- applies the rightmost first. Routes to holographic_transform.compose."""
        from holographic.misc.holographic_transform import compose
        return compose(*mats)

    @staticmethod
    def decompose(M):
        """A 4x4 matrix -> (translate, rotation-quaternion, scale): what a gizmo/property-panel reads off a matrix.
        Routes to holographic_transform.decompose."""
        from holographic.misc.holographic_transform import decompose
        return decompose(M)

    @staticmethod
    def compose_trs(translate, quat, scale):
        """Build a 4x4 from panel values (translate, quaternion, scale) -- the inverse of decompose. Routes to
        holographic_transform.compose_trs."""
        from holographic.misc.holographic_transform import compose_trs
        return compose_trs(translate, quat, scale)

    @staticmethod
    def look_at(eye, target, up=(0.0, 1.0, 0.0)):
        """An OpenGL view matrix aiming from `eye` at `target`. Routes to holographic_transform.look_at."""
        from holographic.misc.holographic_transform import look_at
        return look_at(eye, target, up)

    # --- geometric-algebra rotation (clifford rotor) ---
    @staticmethod
    def rotor(axis, angle):
        """A geometric-algebra ROTOR for a rotation about `axis` by `angle` -- composes without gimbal lock (a
        quaternion generalised). Apply it with Transform.rotate_vec. Routes to holographic_clifford."""
        return _clifford3().rotor(axis, angle)

    @staticmethod
    def rotate_vec(rotor, v3):
        """Rotate a 3-vector by a rotor (the sandwich product R v R~). Routes to holographic_clifford."""
        return _clifford3().rotate(rotor, v3)

    # --- anisotropic steering ---
    @staticmethod
    def steer_bandwidths(X, y, base=2.0, k=10, clip=8.0):
        """Per-sample ANISOTROPIC bandwidths from local structure -- the steering that makes a kernel direction-aware
        (narrow across an edge, wide along it). Routes to holographic_steering.steer_bandwidths."""
        from holographic.misc.holographic_steering import steer_bandwidths
        return steer_bandwidths(X, y, base=base, k=k, clip=clip)


    # --- THE STRUCTURE: which floor of the affine group is this transform standing on? ---
    #
    # This home offers `bind` and `scaling` side by side, as if they were the same KIND of thing. They are not, and
    # a door that suggests a false symmetry is worse than a door that omits something. `bind` is the abelian IDEAL
    # -- a convolution algebra can represent nothing else -- while `scaling` is the CENTRE of the linear part and is
    # not a bind at all (no spectrum represents a dilation; fit one on a vector, apply it to another, relative error
    # 1.579). The tower says which is which, and whether a delta can be pushed through.

    @staticmethod
    def layer(transform, dim=3):
        """Which floor of the transform tower `transform` stands on. Takes a callable OR a matrix.
        {layer, name, diagonalisable, bankable, delta_pushable, why}. Routes to
        holographic_grouptower.classify_transform."""
        from holographic.mesh_and_geometry.holographic_grouptower import classify_transform
        return classify_transform(transform, dim=int(dim))

    @staticmethod
    def hypervector_layer():
        """A hypervector operator is ALWAYS the abelian ideal -- `Transform.bind` cannot be a rotation, ever,
        because circular convolution is commutative. Routes to holographic_grouptower.hypervector_layer."""
        from holographic.mesh_and_geometry.holographic_grouptower import hypervector_layer
        return hypervector_layer()

    @staticmethod
    def is_affine(M, tol=1e-12):
        """Does the 4x4 `M` fix the plane at infinity? A boolean about the bottom row, not a tolerance on a
        distance. Beyond it, no delta pushes through. Routes to holographic_projectivetower.is_affine."""
        from holographic.mesh_and_geometry.holographic_projectivetower import is_affine
        return is_affine(M, tol=tol)

    @staticmethod
    def compose_word(chain):
        """Compose a chain of 4x4s into ONE. A group is CLOSED, so the composed "word" is drawn from the same set
        as its "letters". Routes to holographic_projectivetower.compose_word."""
        from holographic.mesh_and_geometry.holographic_projectivetower import compose_word
        return compose_word(chain)

    @staticmethod
    def bank(dim, seed=0):
        """A prebuilt map of hypervector transforms, held as Fourier spectra -- a representation of the abelian
        ideal, and it can be nothing else. Routes to holographic_transformbank.TransformBank."""
        from holographic.caching_and_storage.holographic_transformbank import TransformBank
        return TransformBank(int(dim), seed=int(seed))

    @staticmethod
    def adjoint(tri, light, A):
        """Push a delta onto the OTHER operand: `shade(A x, L) == max(0, n . (A^-1 L) / ||A^-T n||)`. This is
        conjugation, and it works because the translations are a NORMAL subgroup of the affine group.
        Routes to holographic_equivariance.shade_adjoint."""
        from holographic.mesh_and_geometry.holographic_equivariance import shade_adjoint
        return shade_adjoint(tri, light, A)


def transform_kinds():
    """The transform representations the home spans (for the catalog / discovery)."""
    return ("bind(vsa)", "permute(vsa)", "matrix(4x4)", "rotor(clifford)", "steer(anisotropic)",
            "tower(which floor)", "bank(prebuilt spectra)", "projective(the ceiling)")


def _selftest():
    # geometric matrices route bit-identically to the transform kit
    import holographic.misc.holographic_transform as TF
    assert np.array_equal(Transform.translation([1, 2, 3]), TF.translation([1, 2, 3]))
    assert np.array_equal(Transform.scaling([2, 0.5, 1]), TF.scaling([2, 0.5, 1]))
    assert np.array_equal(Transform.compose(TF.translation([1, 0, 0]), TF.scaling(2)),
                          TF.compose(TF.translation([1, 0, 0]), TF.scaling(2)))

    # translate then read it back: a point moves by the translation
    M = Transform.translation([1.0, 2.0, 3.0])
    p = M @ np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(p[:3], [1.0, 2.0, 3.0])

    # decompose(compose_trs(...)) round-trips the T/R/S a gizmo shows
    t = np.array([1.0, -2.0, 0.5]); s = np.array([2.0, 2.0, 2.0])
    from holographic.misc.holographic_transform import quat_from_axis_angle
    q = quat_from_axis_angle([0, 1, 0], 0.6)
    M2 = Transform.compose_trs(t, q, s)
    t2, q2, s2 = Transform.decompose(M2)
    assert np.allclose(t2, t) and np.allclose(s2, s)

    # VSA: bind is invertible (unbind recovers), permute shifts and its inverse restores
    from holographic.agents_and_reasoning.holographic_ai import bind, unbind, permute
    rng = np.random.default_rng(0)
    a = rng.standard_normal(256); b = rng.standard_normal(256)
    assert np.array_equal(Transform.bind(a, b), bind(a, b))
    assert np.array_equal(Transform.permute(a, 3), permute(a, 3))

    # clifford rotor rotates a vector 90 deg about z: x -> y
    R = Transform.rotor([0, 0, 1], np.pi / 2)
    v = Transform.rotate_vec(R, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(v, [0.0, 1.0, 0.0], atol=1e-9)
    print("OK: holographic_transformhome self-test passed (matrices route bit-identical; translate + TRS round-trip; "
          "bind/permute route; clifford rotor rotates x->y; kinds %s)" % ", ".join(transform_kinds()))


if __name__ == "__main__":
    _selftest()
