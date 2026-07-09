"""D-2 -- the unifier adoption lint. This is the DURABLE fix for the engine's biggest structural debt.

The debt: a unifier can exist, be excellent, and be wired to none of the clients its own docstring names -- every one
of them hand-rolling the thing it unifies. `holographic_iterate` was wired to 0 of its 7 named clients while each
re-implemented "iterate a linear operator" as a python loop (one of them overflowing silently to nan).

Being wired is not a property you fix once. Without a lint, the next module to need a k-step loop writes a k-step
loop, and the silo re-forms. So:

  * every client the registry records as WIRED must stay wired -- un-wiring one fails CI;
  * the PENDING set (known not-yet-wired clients) must match reality EXACTLY. Wire one and the test tells you to
    delete its line, so progress is recorded in the registry rather than in someone's memory. Let one rot and the
    test tells you that too.

The registry (`tools/unifiers.py`) is the single source of truth, and `python tools/unifiers.py --markdown` turns it
into the UNIFIERS.md table. Adding a unifier here is how "promote and generalize" becomes enforceable instead of
aspirational.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

from unifiers import DEFERRED, NOT_APPLICABLE, PENDING, REGISTRY, cites, status   # noqa: E402


def test_every_registered_client_module_exists():
    """A registry that names a module which isn't there is a stale registry."""
    missing = [(u, c) for u, c, wired in status(REPO) if wired is None]
    assert not missing, "registry names modules that don't exist: %s" % missing


def test_wired_clients_stay_wired():
    """The regression guard: anything currently wired to its unifier may never quietly go back to hand-rolling."""
    should_be_wired = [(u, c) for u, c, _w in status(REPO) if (u, c) not in PENDING]
    broken = [(u, c) for u, c in should_be_wired if not cites(c, u, REPO)]
    assert not broken, (
        "these clients no longer cite their unifier (a silo is re-forming):\n"
        + "\n".join("  %s should cite %s" % (c, u) for u, c in broken))


def test_pending_list_is_exactly_the_unwired_set():
    """The pending list is the backlog made executable: it must match reality, so it can only ever shrink."""
    actually_unwired = {(u, c) for u, c, wired in status(REPO) if wired is False}
    newly_wired = PENDING - actually_unwired
    newly_broken = actually_unwired - PENDING
    assert not newly_wired, (
        "these are now wired -- delete them from PENDING in tools/unifiers.py so the progress is recorded:\n"
        + "\n".join("  %s -> %s" % (u, c) for u, c in sorted(newly_wired)))
    assert not newly_broken, (
        "these clients stopped citing their unifier and are not in PENDING:\n"
        + "\n".join("  %s -> %s" % (u, c) for u, c in sorted(newly_broken)))


def test_the_determinism_contract_is_fully_adopted():
    """P3: the tie-break contract is the one whose violation is a correctness hazard (bit-unstable scores flip a
    winner), so it gets its own explicit assertion: all four decision modules cite it, and none does a bare argmax."""
    import re
    for client in REGISTRY["determinism.argmax_tiebreak"]["clients"]:
        assert cites(client, "determinism.argmax_tiebreak", REPO), "%s must cite argmax_tiebreak" % client
        from unifiers import _find_module
        src = open(_find_module(client, REPO), "r", encoding="utf-8", errors="ignore").read()
        code = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))
        assert not re.search(r"\bnp\.argmax\s*\(", code), (
            "%s still calls np.argmax directly -- use argmax_tiebreak (the named ISA-1 rule)" % client)


def test_iterate_is_wired_into_the_propagator():
    """P1: the headline case. Propagator must reach the closed-form path, and must not silently overflow."""
    import numpy as np
    from holographic.simulation_and_physics.holographic_dynamics import Propagator

    rng = np.random.default_rng(0)
    dim = 256
    U = rng.standard_normal(dim) / np.sqrt(dim)
    U /= np.abs(np.fft.rfft(U)).max() * 1.02                  # a comfortably stable operator
    H = np.fft.rfft(U)
    U_inv = np.fft.irfft(np.conj(H) / (np.abs(H) ** 2 + 1e-9), n=dim)
    p = Propagator(U, U_inv)
    s = rng.standard_normal(dim)

    # jump(k) == k sequential steps, in closed form
    loop = s.copy()
    for _ in range(32):
        loop = p.step(loop)
    fast = p.jump(s, 32)
    assert np.allclose(loop, fast, atol=1e-8)

    # rollout still returns the WHOLE trajectory (the contract), matching the old per-step loop
    traj = p.rollout(s, 8)
    assert traj.shape == (8, dim)
    assert np.allclose(traj[-1], p.jump(s, 8), atol=1e-8)

    # recall_at inverts it. NOTE: U_inv is a RIDGE-REGULARIZED pseudo-inverse, so the round-trip is approximate by
    # construction (measured: cos 1.000000, abs diff ~2e-6 on a vector of norm 16). Direction is the contract.
    back = p.recall_at(p.jump(s, 5), 5)
    cos = float(np.dot(back, s) / (np.linalg.norm(back) * np.linalg.norm(s)))
    assert cos > 0.999999, cos

    # a divergent operator raises instead of silently overflowing to nan
    bad = Propagator(rng.standard_normal(dim) * 3.0, U_inv)
    try:
        bad.jump(s, 200)
        assert False, "jump() must raise on a divergent operator"
    except ValueError:
        pass


def test_retracted_claims_are_not_silently_re_added():
    """A unifier's docstring once named clients the MATH cannot serve (a nonlinear cleanup has no operator to raise
    to a power). Those retractions are recorded with a measured reason in NOT_APPLICABLE. This asserts nobody quietly
    puts them back in `clients` -- a wrong backlog item is worse than a missing one, because it looks like work."""
    from unifiers import NOT_APPLICABLE
    assert NOT_APPLICABLE, "the retraction ledger should not be empty"
    for (unifier, client), reason in NOT_APPLICABLE.items():
        assert unifier in REGISTRY, unifier
        assert len(reason) > 40, "a retraction must state WHY, measurably: %r" % ((unifier, client),)
        for name in client.split("/"):
            assert name not in REGISTRY[unifier]["clients"], (
                "%s was retracted from %s (%s) but is listed as a client again" % (name, unifier, reason[:60]))


def test_iterate_scope_is_linear_and_circular():
    """The measurement behind the retraction, pinned: only a LINEAR, CIRCULAR (bind) operator can be exponentiated."""
    import numpy as np
    from holographic.agents_and_reasoning.holographic_ai import bind
    from holographic.misc.holographic_diffuse import manifold_denoise_step
    from holographic.simulation_and_physics.holographic_laplacian import laplacian, is_circular

    rng = np.random.default_rng(0)
    dim = 64
    a, b = rng.standard_normal(dim), rng.standard_normal(dim)

    U = rng.standard_normal(dim) / np.sqrt(dim)
    assert np.allclose(bind(U, a + b), bind(U, a) + bind(U, b))          # dynamics' step IS linear

    M = rng.standard_normal((12, dim))
    assert not np.allclose(manifold_denoise_step(a + b, M, 18.0),        # diffuse's step is NOT
                           manifold_denoise_step(a, M, 18.0) + manifold_denoise_step(b, M, 18.0))

    # heat/wave are linear but Neumann -> not circular -> not a bind -> not exponentiable by step_k
    assert np.allclose(laplacian(a + b, "neumann"), laplacian(a, "neumann") + laplacian(b, "neumann"))
    assert not is_circular("neumann") and is_circular("periodic")


def test_no_design_client_is_silently_dropped():
    """The anti-cheat: the lint above would go green if someone simply DELETED clients from the registry. So every
    client a unifier's design ever named must end up either wired, tracked in PENDING, or retracted in
    NOT_APPLICABLE with a measured reason. Narrowing the scope is not progress."""
    from unifiers import unaccounted
    gaps = unaccounted(REPO)
    assert not gaps, (
        "these clients were named by a unifier's design but are neither wired, pending, nor retracted:\n"
        + "\n".join("  %s -> %s" % (u, c) for u, c in gaps))


def test_retraction_reasons_are_verifiable_against_live_code():
    """A retraction is only as good as its evidence. Spot-check the two most load-bearing ones against the source,
    so a reason can't quietly become false as the code moves underneath it."""
    from unifiers import _find_module

    # octnormal: retracted from mesh/render/splat/gbuffer/meshcurvature BECAUSE none of them quantize normals.
    for client in ("holographic_mesh", "holographic_render", "holographic_splat", "holographic_gbuffer",
                   "holographic_meshcurvature"):
        src = open(_find_module(client, REPO), "r", encoding="utf-8", errors="ignore").read()
        assert "octnormal" not in src, "%s now references octnormal -- the retraction reason is stale" % client
    # ...and its real client does.
    autobump = open(_find_module("holographic_autobump", REPO), "r", encoding="utf-8", errors="ignore").read()
    assert "octnormal" in autobump

    # dynamics: retracted from project_onto_constraints because bind(U,x) is not idempotent (a projection is).
    import numpy as np
    from holographic.agents_and_reasoning.holographic_ai import bind
    rng = np.random.default_rng(0)
    dim = 64
    U = rng.standard_normal(dim) / np.sqrt(dim)
    x = rng.standard_normal(dim)
    once, twice = bind(U, x), bind(U, bind(U, x))
    assert not np.allclose(once, twice), "bind would be idempotent -- then it WOULD be a projection"


def test_the_nonlinear_iterate_is_a_bind_only_near_its_fixed_point():
    """P2, refined. A nonlinear update induces a LINEAR operator on observables (Koopman; the algebra-primary
    stance) -- so "nonlinear" is not by itself a proof that no closed form exists. The measurement that settles it:

      * GLOBALLY a single learned bind cannot reproduce the softmax cleanup (cos ~0.08): one linear operator has one
        fixed point, and the cleanup has one per codebook atom.
      * LOCALLY, inside a small neighbourhood of a fixed point, it CAN (cos > 0.9), with error second order in the
        radius -- exactly the "corrections enter at O(eps^2)" structure.
      * ...and that neighbourhood is precisely where the answer is already known (one argmax gives the attractor),
        while the loop spends its iterations OUTSIDE it. So the closed form is valid where it is useless.

    This is why `iterate.step_k` stays retracted for the nonlinear clients: not because a linearisation cannot
    exist, but because the regime where it holds is the regime that costs nothing.
    """
    import numpy as np
    from holographic.agents_and_reasoning.holographic_ai import cosine
    from holographic.misc.holographic_diffuse import manifold_denoise_step
    from holographic.simulation_and_physics.holographic_dynamics import Propagator

    rng = np.random.default_rng(0)
    dim = 128
    M = rng.standard_normal((6, dim))
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    f = lambda x: manifold_denoise_step(x, M, 18.0)          # noqa: E731

    # GLOBAL: a bind learned over the whole space fails
    X0 = rng.standard_normal((300, dim)) * 0.7
    P_global = Propagator.learn_pairs(X0, np.stack([f(x) for x in X0]), ridge=1e-3)
    test = rng.standard_normal((10, dim)) * 0.7
    assert np.mean([cosine(f(x), P_global.step(x)) for x in test]) < 0.4

    # LOCAL: near one fixed point it succeeds
    atom = M[0]
    Xl = atom + 0.05 * rng.standard_normal((300, dim)) * 0.3
    P_local = Propagator.learn_pairs(Xl, np.stack([f(x) for x in Xl]), ridge=1e-5)
    tl = atom + 0.05 * rng.standard_normal((10, dim)) * 0.3
    assert np.mean([cosine(f(x), P_local.step(x)) for x in tl]) > 0.9

    # ...and the neighbourhood where the linearisation holds is the one where the answer is already free.
    # Measured at dim=128: cosine-to-limit along a settle() run from noise is 0.17, 0.69, 0.95, 1.00, ...
    # so the walk is OUTSIDE the eps-ball for its first two steps -- exactly where the closed form does not apply --
    # and a single argmax after ONE step already returns the limit exactly.
    walk = np.random.default_rng(7)                          # its own stream, so the claim does not depend on order
    traj = [walk.standard_normal(dim) / np.sqrt(dim)]
    for _ in range(7):
        traj.append(f(traj[-1]))
    limit = traj[-1]
    # The bind was fitted inside a ball of radius eps=0.05, i.e. 1 - cos ~ 1e-3. After one step the walk is still
    # an order of magnitude outside it -- the closed form does not apply where the iterations are being spent.
    assert 1.0 - cosine(traj[1], limit) > 0.01
    nearest = M[int(np.argmax(M @ (traj[1] / np.linalg.norm(traj[1]))))]
    assert cosine(nearest, limit) > 0.99                     # ...yet one argmax already gives the attractor exactly


def test_impossible_and_merely_deferred_are_not_confused():
    """The distinction I got wrong, and it matters. NOT_APPLICABLE means CLOSED BY MATHEMATICS (a nonlinear map has
    no operator power; XOR is already exact; a module has no sampling path at all). DEFERRED means the construction
    EXISTS and measurement says it does not pay. Filing the second under the first reads as "impossible" and stops
    people from trying -- which is exactly what happened to `meshsubdiv`, whose closed form turned out to be real.
    """
    assert NOT_APPLICABLE and DEFERRED
    overlap = set(NOT_APPLICABLE) & set(DEFERRED)
    assert not overlap, "an entry cannot be both impossible and merely deferred: %s" % overlap
    for reason in list(NOT_APPLICABLE.values()) + list(DEFERRED.values()):
        assert len(reason) > 60                                  # a verdict must carry its evidence
    # every DEFERRED reason must say why it does not pay, not that it cannot be done
    for (u, c), reason in DEFERRED.items():
        low = reason.lower()
        assert any(w in low for w in ("exists", "buys nothing", "useless", "not built", "wireable", "follow-up",
                                      "is the lift", "linearises", "remains deferred",
                                      "built and wired")), (u, c)


def test_subdivision_has_a_closed_form_and_subdivcurve_uses_it():
    """P2 CORRECTED. I claimed subdivision could have no closed form because 'the operator changes size each level'.
    Wrong: it is a REFINEMENT operator, and on a closed uniform curve it is still diagonal in the Fourier basis, so
    the levels compose analytically (the cascade formula behind Stam's exact evaluation). Exact to 2.7e-15."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_subdivcurve import (CHAIKIN_TAPS, chaikin_subdivide,
                                                                       subdivide_sequence)
    from holographic.misc.holographic_iterate import refine_k

    rng = np.random.default_rng(0)
    P = rng.standard_normal((8, 3))
    for k in (1, 3, 5):
        loop = P.copy()
        for _ in range(k):
            loop = chaikin_subdivide(loop, closed=True)
        assert np.max(np.abs(refine_k(P, CHAIKIN_TAPS, k) - loop)) < 1e-12
        assert np.max(np.abs(subdivide_sequence(P, levels=k, closed=True) - loop)) < 1e-12   # it delegates

    # the honest boundary: an OPEN curve is not shift-invariant, so it keeps the literal loop
    openP = rng.standard_normal((6, 2))
    ref = openP.copy()
    for _ in range(2):
        ref = chaikin_subdivide(ref, closed=False)
    assert np.allclose(subdivide_sequence(openP, levels=2, closed=False), ref)
