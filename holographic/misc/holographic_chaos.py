"""Nonlinear dynamics -- learning a chaotic flow the linear propagator structurally cannot.

WHY THIS EXISTS
---------------
holographic_dynamics.Propagator learns ONE per-frequency complex transfer: the Koopman operator in
Fourier coordinates. That is exact for linear or linearisable flow (advection-diffusion is recovered
almost perfectly), but it is a single FIXED LINEAR map, so it cannot follow a state-dependent
nonlinearity. The record already carries the consequence: on a shock-forming Burgers field the linear
propagator does WORSE than persistence (kept negative in holographic_dynamics), and on a chaotic system
no linear map -- not even the best least-squares one -- can turn the flow.

The unlocked LEARNING gives the missing companion. The reservoir (holographic_reservoir) is a FIXED
nonlinear feature expansion (echo-state dynamics) read out by a TRAINED linear map; that composition
learns the nonlinear ONE-STEP evolution operator a linear transfer cannot represent. This module is the
"learned lift" the dynamics negative explicitly called for -- and it DELEGATES to HolographicESN rather
than re-implementing a learner. It adds two things on top: the dynamics framing (fit on a trajectory,
one-step `predict_sequence`, closed-loop `free_run`) and the honest measurement the project insists on
(one-step error against persistence AND against the strongest linear baseline, not a strawman).

MEASURED (honest picture; Lorenz '63, the canonical reservoir-computing test, RK4 dt=0.02)
  * ONE-STEP is a clean WIN. The reservoir predicts the chaotic next state at ~0.0014 relative error
    -- about 40x better than the BEST linear map (full Dynamic Mode Decomposition, ~0.059) and ~50x
    better than persistence (~0.071). The engine's own circulant propagator only ties persistence.
    A linear map sits at the persistence floor because the Lorenz flow is state-dependent; the
    nonlinear reservoir genuinely learned the local evolution operator. Deterministic (seed in ->
    identical readout out).
  * CLOSED-LOOP free-run tracks the attractor about 10x longer than persistence before diverging.

KEPT NEGATIVES (loud, on the record -- these are the boundaries, not footnotes)
  * The closed-loop horizon is only about ONE Lyapunov time -- far short of what the one-step error
    implies. A 0.0014 one-step error under clean chaotic growth predicts ~5 Lyapunov times; getting ~1
    means the autonomous system is diverging FASTER than chaos alone. That is the well-known reservoir
    free-run STABILITY problem: teacher-forced one-step is excellent, but the closed loop has spurious
    modes. The state-noise trick helps marginally (noise=1e-2 is the sweet spot; more hurts), bigger
    reservoirs help only a little, and -- importantly -- the recurrence MIXING is not the lever:
    cyclic-shift, random-permutation, and unitary-bind recurrences all cap at ~1 Lyapunov time. The
    wall is closed-loop stability, not mixing. Cracking it is a research problem of its own; this
    module does not claim to have.
  * HIGH-DIMENSIONAL PDE FIELDS are out of reach for a single global reservoir. Forecasting a 48-D
    Burgers field one-step lands ~0.27 relative error -- far worse than persistence. The literature
    (Pathak et al. 2018, forecasting the chaotic Kuramoto-Sivashinsky equation) needs LOCAL / parallel
    reservoirs for spatially-extended systems; one global readout cannot. Equilibrium Propagation is
    also weak here: a 48-D field regression is outside its low-output sweet spot (it shines on
    classification-shaped targets). So the win above is a LOW-dimensional dynamics result, honestly.
  * On MILD dissipative Burgers the per-step change is tiny, making persistence a punishing baseline
    with almost nothing to win -- independent of which learner is used.

DESIGN NOTES
  * Pure delegation to the reservoir; pure NumPy; deterministic. The normalisation is per-coordinate
    and captured at fit time so predict / free_run denormalise consistently.
  * This is the nonlinear COMPANION to holographic_dynamics.Propagator, not a replacement: linear flow
    -> the exact, content-addressable Propagator; nonlinear / chaotic flow -> this learned operator.
"""

import numpy as np
from holographic.rendering.holographic_reservoir import HolographicESN


def lorenz_trajectory(n, dt=0.02, sigma=10.0, rho=28.0, beta=8.0 / 3.0, seed=0):
    """A chaotic Lorenz '63 trajectory by RK4 -- the canonical reservoir-computing benchmark.

    Returns an (n + 1, 3) array. WHY it lives here: the selftest needs a known chaotic system to show
    the nonlinear learner winning where every linear map cannot, with no external data dependency.
    The standard parameters (sigma=10, rho=28, beta=8/3) put the system on the strange attractor with
    a largest Lyapunov exponent of about 0.906 per time unit.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(3) * 0.1 + 1.0

    def f(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    out = np.empty((n + 1, 3))
    out[0] = v
    for t in range(n):
        k1 = f(v)
        k2 = f(v + 0.5 * dt * k1)
        k3 = f(v + 0.5 * dt * k2)
        k4 = f(v + dt * k3)
        v = v + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[t + 1] = v
    return out


class NonlinearPropagator:
    """A learned NONLINEAR dynamics operator: the reservoir's fixed nonlinear expansion plus a trained
    linear readout predict state(t+1) from the trajectory so far. The nonlinear companion to the
    linear holographic_dynamics.Propagator -- use the Propagator for linearisable flow (it is exact and
    invertible), this for state-dependent / chaotic flow a single transfer cannot capture.
    """

    def __init__(self, reservoir, mu, sd):
        self.res = reservoir          # the trained HolographicESN -- the delegated learner
        self.mu = np.asarray(mu, float)
        self.sd = np.asarray(sd, float)

    @classmethod
    def learn(cls, states, dim=600, rho=0.9, leak=1.0, in_scale=0.5,
              ridge=1e-6, washout=200, noise=1e-2, seed=0):
        """Fit a nonlinear one-step forecaster on a trajectory (rows = states over time).

        `noise` is the echo-state state-noise trick: it perturbs the reservoir states before the ridge
        solve so the readout is robust to its own drift. It is the single most important knob for
        closed-loop stability, and 1e-2 is the empirical sweet spot here -- more noise actually shortens
        the free-run horizon (see the module's kept negatives). Normalisation is per-coordinate and
        captured so predict / free_run can denormalise.
        """
        X = np.asarray(states, float)
        mu = X.mean(0)
        sd = X.std(0) + 1e-12                              # per-coordinate; the +eps guards a flat axis
        Z = (X - mu) / sd
        res = HolographicESN(n_in=Z.shape[1], dim=dim, rho=rho, leak=leak,
                             in_scale=in_scale, seed=seed)
        # Train ONLY the readout (ridge) to map reservoir-state(t) -> next normalised state. The
        # reservoir's recurrence integrates the past, so the state at t already encodes the history --
        # this is one-step-ahead supervised learning, gradient-free and deterministic.
        res.fit(Z[:-1], Z[1:], ridge=ridge, washout=washout, noise=noise)
        return cls(res, mu, sd)

    def _readout(self, x):
        # The trained linear readout with the reservoir's bias column, in normalised coordinates.
        return np.hstack([x, [1.0]]) @ self.res.W_out

    def predict_sequence(self, states):
        """One-step-ahead predictions for a whole trajectory. Returns an array whose row t is the
        predicted state(t+1) given states[:t+1]; length is len(states) - 1. The first `washout`-ish
        rows carry the reservoir's start-up transient, so score after a short warm-up.
        """
        Z = (np.asarray(states, float) - self.mu) / self.sd
        P = self.res.predict(Z[:-1])                        # P[t] is the readout after consuming Z[t]
        return P * self.sd + self.mu                        # denormalise back to state space

    def free_run(self, warmup, k):
        """Closed-loop autonomous forecast: prime the reservoir on `warmup`, then feed each prediction
        back as the next input for k steps. Returns the (k, dim) predicted states (row i = i+1 steps
        after the end of warmup). HONEST: accurate for roughly one Lyapunov time on a chaotic system,
        then it diverges -- the reservoir free-run stability limit documented at module scope.
        """
        Zw = (np.asarray(warmup, float) - self.mu) / self.sd
        x = self.res.run(Zw)[-1].copy()                     # reservoir state after the warm-up
        y = self._readout(x)
        out = np.empty((k, self.mu.shape[0]))
        for i in range(k):
            out[i] = y * self.sd + self.mu
            x = self.res._step(x, y)                         # feed the (normalised) prediction back
            y = self._readout(x)
        return out


def _rel(p, t):
    return float(np.linalg.norm(np.asarray(p) - np.asarray(t)) / (np.linalg.norm(t) + 1e-12))


def _selftest():
    """CI-fast: prove the nonlinear learner beats EVERY linear baseline on the chaotic one-step map
    (the clean win), that it is deterministic, and that its free-run beats persistence -- while NOT
    overclaiming the closed-loop horizon, which the docstring keeps as a loud negative."""
    dt = 0.02
    tr = lorenz_trajectory(4000, dt=dt, seed=0)
    ntr = 2600
    train, test = tr[:ntr], tr[ntr:]

    # --- baselines, all scored as one-step relative error on the held-out chaotic trajectory ---
    pers = np.mean([_rel(test[i], test[i + 1]) for i in range(len(test) - 1)])
    # strongest linear map: full DMD (best least-squares 3x3 operator). If the reservoir beats THIS,
    # the win is not a strawman against the engine's circulant propagator.
    A, _, _, _ = np.linalg.lstsq(train[:-1], train[1:], rcond=None)
    dmd = np.mean([_rel(test[i] @ A, test[i + 1]) for i in range(len(test) - 1)])

    # --- the nonlinear forecaster ---
    prop = NonlinearPropagator.learn(train, dim=400, ridge=1e-6, washout=200, noise=1e-2, seed=0)
    pred = prop.predict_sequence(test)
    res = np.mean([_rel(pred[i], test[i + 1]) for i in range(200, len(test) - 1)])

    # The win: nonlinear << best-linear and << persistence (chaos pins any linear map at the floor).
    assert res < 0.01, f"reservoir one-step should be tiny on Lorenz, got {res:.4f}"
    assert dmd > 0.03 and pers > 0.03, f"linear/persistence should sit at the chaos floor: dmd={dmd:.4f} pers={pers:.4f}"
    assert res < dmd / 10.0, f"reservoir should beat best-linear by >10x, got {dmd / res:.0f}x"

    # Determinism: same seed -> identical learned readout.
    prop2 = NonlinearPropagator.learn(train, dim=400, ridge=1e-6, washout=200, noise=1e-2, seed=0)
    assert np.allclose(prop.res.W_out, prop2.res.W_out), "fit must be deterministic for a fixed seed"

    # Closed-loop beats persistence (modest but real); we do NOT assert literature-scale horizons.
    lyap = 0.906
    warm = test[:300]
    truth = test[300:300 + 600]
    gen = prop.free_run(warm, len(truth))
    rerr = np.array([_rel(gen[i], truth[i]) for i in range(len(truth))])
    perr = np.array([_rel(warm[-1], truth[i]) for i in range(len(truth))])
    res_h = int(np.argmax(rerr > 0.3)) if (rerr > 0.3).any() else len(rerr)
    per_h = int(np.argmax(perr > 0.3)) if (perr > 0.3).any() else len(rerr)
    assert res_h > 3 * per_h, f"free-run should track longer than persistence: res={res_h} per={per_h}"

    print("holographic_chaos selftest OK")
    print(f"  one-step rel-error: reservoir {res:.4f}  vs  full-DMD {dmd:.4f}  vs  persistence {pers:.4f}"
          f"   ({dmd / res:.0f}x better than best-linear)")
    print(f"  closed-loop valid horizon: reservoir {res_h * dt * lyap:.1f} vs persistence "
          f"{per_h * dt * lyap:.2f} Lyapunov times (capped ~1 Lyapunov time -- kept negative)")


if __name__ == "__main__":
    _selftest()
