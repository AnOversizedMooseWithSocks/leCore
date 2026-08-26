"""holographic_guide.py -- guide a state toward a goal by ITERATING A PROJECTION (L10).

WHY THIS MODULE EXISTS
----------------------
The NOTES recorded the deep identity long ago: IK (reach a target), PBD (preserve edge lengths), PnP/RED (data
fidelity + denoiser), and the SBC resonator (factor against codebooks) are ALL the same move -- repeatedly project
a state onto a constraint set until it settles (Macklin's position-based-dynamics observation). The engine already
has that iterator (holographic_meshik.project_onto_constraints). What it did NOT have is the LEVEL-GENERIC name
for it: "move this thing legally toward a goal", discoverable as such, so an agent reaching for constrained
movement finds one faculty instead of an animation-specific one. This module is that name. It delegates entirely --
NO new solver.

THE ONE PATTERN, FOUR COSTUMES
  * skeleton joints -> reach a target under bone-length limits (classic IK)
  * mesh verts -> preserve edge lengths (PBD)
  * a noisy signal -> the manifold of clean signals (PnP / RED denoise)
  * a mystery vector -> the product of per-factor codebooks (the resonator)
Each is `project_onto_constraints(state, [proj_1, proj_2, ...])`: a list of projection callables, applied in turn
until the state stops moving. The constraints ARE the structure of the space; iterating them is guided movement.

Deterministic (the underlying iterator is). NumPy only. Additive.
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_meshik import project_onto_constraints


def guide_structure(state, constraints, iters=50, tol=1e-6, omega=1.0):
    """Guide `state` toward satisfying a set of `constraints` by ITERATING A PROJECTION -- the level-generic form
    of IK / PBD / denoise / resonator (all 'iterate a projection'). `state` is a NumPy array (any shape the
    constraints understand); `constraints` is a list of callables each taking the state and returning it projected
    onto that one constraint (pin a root to a target, clamp a link length, snap to a codebook, ...). They are
    applied in turn, sweep after sweep, until the state settles or `iters` runs out. Returns a dict: `state` (the
    settled result), `iters_used`, `converged` (True if it stopped moving within `tol`), and `residual` (how far
    the final sweep still moved -- 0 at a fixed point). The constraints are the STRUCTURE of the space; iterating
    them is legal movement through it. Delegates to holographic_meshik.project_onto_constraints -- one solver."""
    state = np.asarray(state, float)
    settled, used, converged = project_onto_constraints(state, list(constraints), iters=iters, tol=tol, omega=omega)
    settled = np.asarray(settled, float)
    # residual: how far one more projection sweep would move the settled state (0 at a true fixed point).
    probe = settled.copy()
    for proj in constraints:
        probe = np.asarray(proj(probe), float)
    residual = float(np.linalg.norm(probe - settled))
    return {"state": settled, "iters_used": int(used), "converged": bool(converged), "residual": residual}


# --- ready-made constraint builders (so callers don't hand-roll the common ones) ------------------------------

def pin(index, value):
    """A projection that pins state[index] to a fixed `value` (an IK end-effector target, a Dirichlet boundary)."""
    def proj(x):
        x = np.asarray(x, float).copy()
        x[index] = value
        return x
    return proj


def clamp_link(i, j, length):
    """A projection that clamps the distance between state[i] and state[j] to at most `length` (a bone/edge length
    limit -- the PBD distance constraint on a 1-D chain of scalar joints)."""
    def proj(x):
        x = np.asarray(x, float).copy()
        d = x[j] - x[i]
        dist = abs(d)
        if dist > length:
            x[j] = x[i] + np.sign(d) * length
        return x
    return proj


def snap_to_codebook(book):
    """A projection that snaps every entry of the state to its nearest value in `book` (the resonator's alternating
    cleanup -- project onto a factor's codebook). `book` is a 1-D array of allowed values."""
    book = np.asarray(book, float)
    def proj(x):
        x = np.asarray(x, float)
        return book[np.argmin(np.abs(x[:, None] - book[None, :]), axis=1)].astype(float)
    return proj


def _selftest():
    """Contracts -- ONE iterator, several costumes, each settling correctly:

    1. IK/PBD costume: a 3-scalar chain with the root pinned to a target and each link clamped to length 1 settles
       to the reachable configuration (root at target, no link longer than 1), and converges.
    2. RESONATOR costume: the SAME guide_structure, with a codebook-snap constraint, snaps a noisy vector to its
       nearest codebook entries -- a fixed point (residual 0).
    3. Convergence + residual: a satisfiable system reaches residual ~0; the result reports converged True.
    4. Determinism: same inputs -> same settled state.
    """
    # (1) IK/PBD chain.
    x0 = np.array([0.0, 5.0, 9.0])
    r = guide_structure(x0, [pin(0, 3.0), clamp_link(0, 1, 1.0), clamp_link(1, 2, 1.0)], iters=100)
    s = r["state"]
    assert abs(s[0] - 3.0) < 1e-6                              # root reached the target
    assert abs(s[1] - s[0]) <= 1.0 + 1e-6 and abs(s[2] - s[1]) <= 1.0 + 1e-6   # links within length
    assert r["converged"]

    # (2) resonator costume: same solver, codebook-snap constraint.
    book = np.array([0.0, 2.0, 4.0, 6.0])
    r2 = guide_structure(np.array([0.3, 3.9, 5.2]), [snap_to_codebook(book)], iters=10)
    assert np.allclose(r2["state"], [0.0, 4.0, 6.0])          # each snapped to nearest codebook entry
    assert r2["residual"] < 1e-9                              # a fixed point

    # (3) the chain result is a near-fixed point too.
    assert r["residual"] < 1e-3

    # (4) determinism.
    assert np.array_equal(guide_structure(x0, [pin(0, 3.0), clamp_link(0, 1, 1.0)], iters=20)["state"],
                          guide_structure(x0, [pin(0, 3.0), clamp_link(0, 1, 1.0)], iters=20)["state"])

    print("holographic_guide selftest OK (ONE iterator, two costumes: IK/PBD chain settles root->target with links "
          "<=1 and converges; the SAME solver snaps a vector to a codebook as the resonator does, residual %.0e; "
          "deterministic)" % (r2["residual"],))


if __name__ == "__main__":
    _selftest()
