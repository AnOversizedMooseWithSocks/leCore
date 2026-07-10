"""The MODAL JUMP SOLVER (Box3D lesson B1, backlog item X1) -- the measured headline of the Box3D read-through.

Catto's soft constraint is a damped harmonic oscillator integrated implicitly over many small substeps: robust,
unconditionally stable, and paid PER SUBSTEP, EVERY FRAME. But look at what an implicit substep of a linear
island actually is:

    s <- A s + b            (s = [positions; velocities], A from the implicit solve, b the constant forcing)

That is an affine recurrence -- exactly the object `iterate.affine_step_k` evaluates in closed form. Within a
CONTACT MODE (a fixed set of active constraints) the system is linear, so N substeps are one eigendecomposition,
and the cost of reaching t = 1 s and t = 10 s is the same.

WHERE THIS IS TRUE, STATED FIRST. Physics is nonlinear at contact EVENTS: constraints switch on and off, and
that is where Catto's per-substep iteration earns its keep (a pile of boxes has a constantly changing contact
set). The jump wins where the contact topology is STABLE -- machinery, ragdolls at rest, spring networks,
suspensions -- and it must re-diagonalize at every mode switch. So the mode-switch RATE is the whole economics,
and `should_jump` below is a measured gate, not a formality. No claim is made to beat sequential impulses at
Catto's own benchmark.

Two hard limits are enforced rather than hidden (see `iterate.affine_step_k`):
  * an island with a marginal-but-diagonalizable mode accumulates a RAMP, not a geometric sum (free fall);
  * an island that is genuinely DEFECTIVE (the free-body Jordan block) has no eigenbasis at all and is stepped.
"""

import numpy as np

from holographic.misc.holographic_iterate import affine_step_k, affine_transfer, affine_limit


def soft_chain_matrices(n_bodies, hertz=15.0, zeta=0.7, mass=1.0, gravity=-9.81, dt=1.0 / 60.0, substeps=64):
    """Build the implicit-Euler substep of an anchored soft-constraint chain as the affine map `s <- A s + b`,
    with s = [positions; velocities] of length 2*n_bodies. Returns (A, b, h) where h = dt/substeps.

    Constraints are parameterized the way Catto parameterizes them and the way `denoise.soft_relaxation` does --
    a natural frequency `hertz` and a damping ratio `zeta` -- rather than by raw stiffness, so the same numbers
    mean the same physics at any substep count. Stiffness k = m*w^2 and damping c = 2*zeta*m*w with w = 2*pi*f.

    Body 0 is anchored to the origin by one spring; consecutive bodies are joined by identical springs. `gravity`
    is the constant forcing (the `b` term). One-dimensional (the sag axis), which is all the mode structure
    needs -- the algebra is identical per axis."""
    n = int(n_bodies)
    if n < 1:
        raise ValueError("n_bodies must be >= 1")
    h = float(dt) / int(substeps)
    w = 2.0 * np.pi * float(hertz)
    k, c = mass * w * w, 2.0 * float(zeta) * mass * w

    K = np.zeros((n, n))
    C = np.zeros((n, n))
    K[0, 0] += k
    C[0, 0] += c                                      # the anchor spring: without it the island is defective
    for i in range(n - 1):
        K[i, i] += k; K[i + 1, i + 1] += k
        K[i, i + 1] -= k; K[i + 1, i] -= k
        C[i, i] += c; C[i + 1, i + 1] += c
        C[i, i + 1] -= c; C[i + 1, i] -= c
    f = np.full(n, mass * float(gravity))
    return _implicit_euler_affine(K, C, f, mass, h) + (h,)


def _implicit_euler_affine(K, C, f, mass, h):
    """One IMPLICIT (backward-Euler) substep of `M x'' + C x' + K x = f` as an affine map on s = [x; v].

    Implicit rather than explicit because that is what makes Catto's solver unconditionally stable, and because
    the whole point is that the substep is a LINEAR map whose power we can take. The velocity solve
    (I + h M^-1 C + h^2 M^-1 K) v' = v - h M^-1 K x + h M^-1 f is done once, symbolically, into the matrix."""
    n = K.shape[0]
    I = np.eye(n)
    Minv = I / float(mass)
    S = np.linalg.inv(I + h * (Minv @ C) + h * h * (Minv @ K))
    Avx = -h * (S @ (Minv @ K))
    Avv = S
    bv = h * (S @ (Minv @ f))

    A = np.zeros((2 * n, 2 * n))
    b = np.zeros(2 * n)
    A[n:, :n] = Avx
    A[n:, n:] = Avv
    b[n:] = bv
    A[:n, :n] = I + h * Avx                            # x' = x + h*v'  (the implicit position update)
    A[:n, n:] = h * Avv
    b[:n] = h * bv
    return A, b


def soft_chain_bank(n_bodies, hertz, zeta, mass=1.0, gravity=-9.81, dt=1.0 / 60.0, substeps=64):
    """Build a TUNING BANK: M variants of one soft-constraint chain, differing in stiffness and/or damping.

    `hertz` and `zeta` are scalars or length-M sequences (broadcast against each other). Returns (As:(M,2n,2n),
    bs:(M,2n), h) ready for `iterate.affine_step_k_batch` -- one batched eigendecomposition advances every variant
    to any horizon. This is the designer's friction/stiffness dial, evaluated in one pass (backlog X8, Box3D B7's
    data-oriented half).

    NOT a superposition. See `iterate`'s batched-recurrence note: a trajectory is linear in the forcing and
    nonlinear in the operator, so stiffness variants cannot be summed into one vector at any dimension. They are
    batched as arrays, which is exact and has no capacity budget."""
    hz = np.atleast_1d(np.asarray(hertz, float))
    ze = np.atleast_1d(np.asarray(zeta, float))
    hz, ze = np.broadcast_arrays(hz, ze)
    As, bs, h = [], [], None
    for f, z in zip(hz.ravel(), ze.ravel()):
        A, b, h = soft_chain_matrices(n_bodies, hertz=float(f), zeta=float(z), mass=mass,
                                      gravity=gravity, dt=dt, substeps=substeps)
        As.append(A)
        bs.append(b)
    return np.stack(As), np.stack(bs), h


def advance_bank(S0, As, bs, k, transfer=None):
    """Advance M island variants k substeps in ONE batched closed form. `S0`:(M,2n) -> (M,2n).

    MEASURED (M=32 variants, 1,920 substeps each): 13.20 ms substepping the batch vs 3.10 ms here -- 4.3x -- with
    max|diff| 1.86e-12 against the substepped reference, and the horizon free as always. Pass a cached `transfer`
    from `iterate.affine_transfer_batch` to reuse the factorization across frames within a contact mode.

    Raises, naming the variant, if any member of the bank is defective (no eigenbasis): one bad variant would
    otherwise poison exactly one row of the answer, silently."""
    from holographic.misc.holographic_iterate import affine_step_k_batch
    return affine_step_k_batch(S0, As, bs, k, transfer=transfer)


def blend_forcings(s0, A, forcings, weights, k, transfer=None):
    """The one thing that DOES superpose exactly: variants that differ only in their FORCING.

    `s_k = A^k s0 + G(A,k) b` is LINEAR in b, so a weighted blend of forcings is the weighted blend of their
    trajectories, to machine precision (measured 1.11e-16 over 600 substeps). One eigendecomposition serves every
    forcing variant AND every blend of them -- gravity, wind, a designer's load dial, a per-variant external push.

    Contrast `soft_chain_bank`, whose variants differ in the OPERATOR (stiffness, damping): blending those is
    nonsense (measured max diff 2.94e-01). The dividing line is exactly where the map stops being linear."""
    from holographic.misc.holographic_iterate import affine_step_k, affine_transfer
    s0 = np.asarray(s0, float)
    F = np.asarray(forcings, float)
    w = np.asarray(weights, float)
    if F.shape[0] != w.shape[0]:
        raise ValueError("need one weight per forcing, got %d and %d" % (F.shape[0], w.shape[0]))
    tr = affine_transfer(np.asarray(A, float)) if transfer is None else transfer
    # blend FIRST -- one solve, not M. Exact by linearity; the test pins it against solving each and blending.
    return affine_step_k(s0, A, np.einsum("m,mi->i", w, F), k, transfer=tr)


def mode_key(active_constraints):
    """A deterministic signature for a CONTACT MODE -- the set of currently active constraints. Any hashable ids
    work (indices, (body_a, body_b) pairs). Sorted, so the same mode reached by different orders is the same key,
    which is what lets a solver recognise 'nothing switched' and reuse its factorization."""
    return tuple(sorted(active_constraints))


# Break-even between one O(n^3) eigendecomposition and k O(n^2) matvecs, MEASURED on this box
# (see _selftest and NOTES): the jump is a loss below roughly 20*dim substeps per mode. It is a gate,
# not a formality -- at dim=24 (12 bodies) the closed form loses 11x at k=16 and wins 14x at k=3840.
JUMP_BREAKEVEN = 20.0


def should_jump(dim, k, breakeven=JUMP_BREAKEVEN):
    """Is jumping `k` substeps of a `dim`-dimensional island cheaper than stepping them? True when
    k >= breakeven*dim. MEASURED, not assumed: an eigendecomposition is O(dim^3) and a substep is O(dim^2), so
    the jump only pays once enough substeps amortize the factorization. The descriptor calls this to route
    jump-vs-step per island per mode -- the same regime-gate pattern as everywhere else in the engine."""
    return int(k) >= float(breakeven) * int(dim)


def escalation_plan(dim, k, energy=None, sleep_energy=0.0, diagonalizable=True, breakeven=JUMP_BREAKEVEN):
    """THE ESCALATION LADDER (backlog X11). Catto's manual suggests "4 substeps" as a quality dial; leCore's closed
    form and his substep count are the two ends of ONE axis, and the descriptor picks the rung per island per frame:

        sleep    -- the island is at its fixed point. Do nothing; its state is already the answer (X3).
        jump     -- the island is linear in this contact mode and k is large enough to amortize an O(dim^3)
                    eigendecomposition. One solve, any horizon (X1).
        substep  -- everything else: contacts churning, k small, or a DEFECTIVE operator with no eigenbasis.
                    Catto's iteration, which is the right tool here and is not being beaten.

    Returns {rung, why, substeps}. Pure decision, no state, no solving -- inspectable before anything runs, exactly
    like `plan_render` and `plan_waves`. The order of the tests is the order of the ladder: an asleep island is not
    asked whether it is diagonalizable, and a defective one is never asked whether the jump would pay."""
    k = int(k)
    if energy is not None and float(energy) <= float(sleep_energy):
        return {"rung": "sleep", "why": "energy %.3g <= sleep bar %.3g: already at the fixed point"
                                        % (float(energy), float(sleep_energy)), "substeps": 0}
    if not diagonalizable:
        return {"rung": "substep", "why": "operator is defective (no eigenbasis): the closed form does not exist",
                "substeps": k}
    if should_jump(dim, k, breakeven):
        return {"rung": "jump", "why": "k=%d >= %g*dim=%g: one eigendecomposition beats %d matvecs"
                                       % (k, breakeven, breakeven * int(dim), k), "substeps": 0}
    return {"rung": "substep", "why": "k=%d < %g*dim=%g: the eigendecomposition would not amortize"
                                      % (k, breakeven, breakeven * int(dim)), "substeps": k}


class ModalSolver:
    """Advance a linear island in CLOSED FORM within a contact mode, re-diagonalizing only at mode SWITCHES.

    Usage: `set_mode(key, A, b)` whenever the active-constraint set changes; `advance(k)` to jump k substeps.
    The eigendecomposition is cached per mode key, so a scene that revisits modes (a ragdoll settling, a
    machine cycling) pays for each mode once, ever.

    `switches` and `substeps` are counted, because the win IS their ratio and the backlog demands it be reported
    rather than asserted. `fallbacks` counts islands routed to stepping because they were defective or below the
    break-even -- an honest solver reports how often its fast path did not apply."""

    def __init__(self, state, breakeven=JUMP_BREAKEVEN, cache_modes=True):
        self.state = np.asarray(state, float).copy()
        self.breakeven = float(breakeven)
        self.cache_modes = bool(cache_modes)
        self._mode = None
        self._A = self._b = self._transfer = None
        self._cache = {}
        self.switches = 0
        self.substeps = 0
        self.fallbacks = 0

    def set_mode(self, key, A, b):
        """Declare the active contact mode. If `key` differs from the current one this is a mode SWITCH. Returns
        True if a switch happened.

        The factorization is LAZY, and that is a measured decision, not tidiness. Diagonalizing here would make
        every switch pay O(dim^3) -- including the switches whose mode is about to be STEPPED because it sits
        below the break-even. Measured with eager factorization: 16 switches x 240 substeps ran 13.15 ms against
        6.37 ms of pure substepping, i.e. the solver paid for eigendecompositions it then declined to use, and
        turned a 2x loss into a 2x-worse loss. A gate must not pay for the thing it declines."""
        if key == self._mode:
            return False
        self._mode = key
        self._A = np.asarray(A, float)
        self._b = np.asarray(b, float)
        self._transfer = None
        self._factorized = False
        self.switches += 1
        return True

    def _get_transfer(self):
        """Factorize the current mode ON FIRST USE, reusing the per-mode cache. Returns None for a DEFECTIVE
        island (no eigenbasis) -- a correct answer meaning 'step this one', not an error to swallow."""
        if self._factorized:
            return self._transfer
        self._factorized = True
        if self.cache_modes and self._mode in self._cache:
            self._transfer = self._cache[self._mode]
        else:
            try:
                self._transfer = affine_transfer(self._A)
            except ValueError:
                self._transfer = None                   # defective: this mode gets stepped, forever
                self.fallbacks += 1
            if self.cache_modes:
                self._cache[self._mode] = self._transfer
        return self._transfer

    def advance(self, k):
        """Advance `k` substeps within the current mode and return the new state. Jumps in closed form when
        `should_jump` says it pays AND the operator is diagonalizable; otherwise steps. Either path is exact to
        floating point -- the gate trades cost, never correctness.

        Order matters: `should_jump` is consulted BEFORE the operator is factorized, so a mode that will be
        stepped never pays for an eigendecomposition."""
        k = int(k)
        if self._A is None:
            raise RuntimeError("set_mode(...) before advance(...)")
        self.substeps += k
        dim = self._A.shape[0]
        transfer = self._get_transfer() if should_jump(dim, k, self.breakeven) else None
        if transfer is not None:
            self.state = affine_step_k(self.state, self._A, self._b, k, transfer=transfer)
        else:
            s = self.state
            for _ in range(k):
                s = self._A @ s + self._b
            self.state = s
        return self.state

    def report(self):
        """{switches, substeps, substeps_per_switch, fallbacks, modes_cached} -- the economics of the jump, so a
        caller can see whether the mode-switch rate ate the win. substeps_per_switch is the number that decides."""
        return {"switches": self.switches, "substeps": self.substeps,
                "substeps_per_switch": (self.substeps / self.switches) if self.switches else 0.0,
                "fallbacks": self.fallbacks, "modes_cached": len(self._cache)}


def _selftest():
    """Regression trap for X1. The BAR from the backlog: match a substepped reference to 1e-10, and count mode
    switches against substeps. Both are asserted numerically, plus the two kept negatives."""
    import time

    n = 12
    A, b, h = soft_chain_matrices(n, hertz=15.0, zeta=0.7, dt=1 / 60.0, substeps=64)
    s0 = np.zeros(2 * n)
    N = 60 * 64                                        # one second of Catto's parameterization: 3,840 substeps

    ref = s0.copy()
    t0 = time.perf_counter()
    for _ in range(N):
        ref = A @ ref + b
    t_sub = time.perf_counter() - t0

    t0 = time.perf_counter()
    got = affine_step_k(s0, A, b, N)
    t_cf = time.perf_counter() - t0

    err = float(np.abs(ref - got).max())
    assert err < 1e-10, ("THE BAR: closed form must match the substepped reference to 1e-10", err)
    assert ref[n - 1] < -0.01                          # the chain really did sag; not a trivial zero match

    # HORIZON INDEPENDENCE: ten seconds costs one more elementwise power, not 10x the substeps.
    tr = affine_transfer(A)
    s1 = affine_step_k(s0, A, b, N, transfer=tr)
    s10 = affine_step_k(s0, A, b, 10 * N, transfer=tr)
    # NOT "the sag keeps growing": at zeta=0.7 the chain is UNDERDAMPED, so at t=1s it has overshot equilibrium
    # and by t=10s it has settled back toward it. The invariant is convergence to the fixed point.
    fixed = affine_limit(A, b, transfer=tr)
    assert np.abs(s10 - fixed).max() < np.abs(s1 - fixed).max()
    ref10 = s0.copy()
    for _ in range(10 * N):
        ref10 = A @ ref10 + b
    assert float(np.abs(ref10 - s10).max()) < 1e-10

    # KEPT NEGATIVE 1 -- an unanchored (free-body) island is DEFECTIVE: no eigenbasis, so it must be stepped.
    Afree = np.eye(4)
    Afree[:2, 2:] = np.eye(2) * h
    try:
        affine_transfer(Afree)
    except ValueError:
        pass
    else:
        raise AssertionError("a free-body Jordan block must be refused, not silently jumped")

    # ... and the solver ROUTES it to stepping rather than failing
    ms = ModalSolver(np.zeros(4))
    ms.set_mode(mode_key(["free"]), Afree, np.array([0.0, 0.0, -0.001, -0.001]))
    stepped = ms.advance(300)
    manual = np.zeros(4)
    for _ in range(300):
        manual = Afree @ manual + np.array([0.0, 0.0, -0.001, -0.001])
    assert np.allclose(stepped, manual) and ms.report()["fallbacks"] == 1

    # KEPT NEGATIVE 2 -- the jump LOSES below break-even; should_jump must say so.
    assert not should_jump(2 * n, 16)                  # 16 substeps on a 24-dim island: step it
    assert should_jump(2 * n, 3840)                    # 3,840: jump it

    # X8 -- the tuning bank. M stiffness variants batch EXACTLY; they do not superpose.
    from holographic.misc.holographic_iterate import affine_transfer as _at
    M = 8
    As, bs, _ = soft_chain_bank(6, hertz=np.linspace(5.0, 40.0, M), zeta=0.7, substeps=32)
    assert As.shape == (M, 12, 12) and bs.shape == (M, 12)
    S0 = np.zeros((M, 12))
    ref = S0.copy()
    for _ in range(600):
        ref = np.einsum("mij,mj->mi", As, ref) + bs
    assert np.abs(advance_bank(S0, As, bs, 600) - ref).max() < 1e-10   # batched closed form == substepping

    # variants in the FORCING superpose exactly; variants in the OPERATOR do not. The dividing line.
    A0, b0, _ = soft_chain_matrices(6, hertz=15.0, zeta=0.7, substeps=32)
    tr0 = _at(A0)
    f1, f2 = b0.copy(), b0 * 0.25
    w = np.array([0.3, 0.7])
    blended = blend_forcings(np.zeros(12), A0, np.stack([f1, f2]), w, 600, transfer=tr0)
    separate = sum(w[i] * affine_step_k(np.zeros(12), A0, f, 600, transfer=tr0)
                   for i, f in enumerate((f1, f2)))
    assert np.abs(blended - separate).max() < 1e-10                    # exact by linearity in b
    mixA = 0.5 * As[0] + 0.5 * As[-1]                                  # blending OPERATORS is nonsense ...
    lhs = affine_step_k(np.zeros(12), mixA, bs[0], 600)
    rhs = 0.5 * affine_step_k(np.zeros(12), As[0], bs[0], 600) + 0.5 * affine_step_k(np.zeros(12), As[-1], bs[0], 600)
    assert np.abs(lhs - rhs).max() > 1e-3                              # ... and the selftest pins that it is

    # X11 -- the escalation ladder, in ladder order: sleep beats everything, defect beats break-even.
    assert escalation_plan(24, 3840, energy=0.0, sleep_energy=1e-8)["rung"] == "sleep"
    assert escalation_plan(24, 3840, energy=1.0, sleep_energy=1e-8)["rung"] == "jump"
    assert escalation_plan(24, 3840, diagonalizable=False)["rung"] == "substep"
    assert escalation_plan(24, 16)["rung"] == "substep"
    assert escalation_plan(24, 16)["substeps"] == 16   # the substep rung reports Catto's count

    # MODE SWITCHING: a solver that switches every few substeps pays the eigendecomposition every time.
    ms2 = ModalSolver(s0)
    A2, b2, _ = soft_chain_matrices(n, hertz=30.0, zeta=0.7)
    for i in range(6):                                 # alternate two modes, revisiting each (cache pays)
        ms2.set_mode(mode_key([i % 2]), A if i % 2 == 0 else A2, b if i % 2 == 0 else b2)
        ms2.advance(640)
    rep = ms2.report()
    assert rep["switches"] == 6 and rep["substeps"] == 3840
    assert rep["modes_cached"] == 2                    # two distinct modes, factorized once each
    assert abs(rep["substeps_per_switch"] - 640.0) < 1e-9

    print("OK: holographic_modal self-test passed (12-body soft chain, hertz=15 zeta=0.7, 3,840 substeps: "
          "closed form matches the substepped reference to %.1e -- the 1e-10 bar -- at %.2f ms vs %.2f ms, and "
          "t=10s is the same one solve; a free-body island is a Jordan block and is REFUSED then stepped; "
          "should_jump gates the loss below break-even; mode cache factorizes each of 2 modes once)"
          % (err, t_cf * 1e3, t_sub * 1e3))


if __name__ == "__main__":
    _selftest()
