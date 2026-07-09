"""holographic_horizon.py -- MULTI-HORIZON forecasting with a TRUSTED-HORIZON gate: forecast N steps ahead,
widen the interval honestly as error compounds, and report the horizon at which confidence drops below the
caller's tolerance -- so a sim or renderer can SUBSTITUTE a cheap forecast for real compute out to the trusted
horizon and fall back to full compute beyond it.

WHY THIS EXISTS (Forecasting & Prediction backlog, F6)
------------------------------------------------------
The concrete "drive simulation/render if confident" use case. A closed-loop producer (Propagator.rollout,
chaos.free_run, a reservoir) can roll a trajectory forward cheaply, but error compounds with depth, so a single
interval is wrong -- the interval must GROW with the horizon. Calibrate a separate conformal quantile PER
horizon step (reusing the split-conformal machinery from holographic_conformal), and the first horizon whose
width exceeds the caller's tolerance is the `trusted_horizon`: predict up to there, recompute past it. This is
the forecasting sibling of dirtyfield (recompute only what changed) and the reflection-interpolation move.

KEPT NEGATIVE (loud): for CHAOTIC systems the trusted horizon is short BY NATURE (a Lyapunov-time fact) -- even
a perfect one-step learner diverges over multiple steps. The honest result there is "predict a few steps, then
recompute," not "predict forever." The gate makes that boundary mechanical instead of hoping.

Real basis: Stankeviciute et al. (2021), multi-horizon conformal; Pathak et al. (2018), closed-loop reservoir
forecasting. Seat: Stam/Macklin (predict a field/step forward cheaply, verify). Deterministic; NumPy + stdlib.
"""
import numpy as np

from holographic.mesh_and_geometry.holographic_conformal import conformal_quantile


class MultiHorizonForecaster:
    """Calibrate a per-horizon prediction interval for a closed-loop `rollout_fn(state, H) -> trajectory` and
    report how far ahead it can be trusted. `rollout_fn` returns an (H,) scalar path or an (H, dim) vector path.
    Scalar residuals use |error|; vector residuals use 1 - cosine (the engine's metric)."""

    def __init__(self, rollout_fn, alpha=0.1, kind="scalar"):
        self.rollout_fn = rollout_fn
        self.alpha = float(alpha)
        self.kind = kind
        self.q = None                                           # q[h] = calibrated half-width at horizon h

    def _residual(self, pred, actual):
        if self.kind == "scalar":
            return float(abs(float(pred) - float(actual)))
        p = np.asarray(pred, float); a = np.asarray(actual, float)
        cos = float(p @ a / ((np.linalg.norm(p) + 1e-12) * (np.linalg.norm(a) + 1e-12)))
        return 1.0 - cos

    def calibrate(self, states, true_futures, horizon):
        """For each calibration `state`, roll forward `horizon` steps and compare to the KNOWN future trajectory;
        collect residuals per horizon step and take the conformal quantile at each. The quantile GROWS with the
        horizon on a real system -- that growth is the honest compounding of error."""
        per_h = [[] for _ in range(horizon)]
        for s, fut in zip(states, true_futures):
            pred = self.rollout_fn(s, horizon)
            for h in range(horizon):
                per_h[h].append(self._residual(pred[h], fut[h]))
        self.q = np.array([conformal_quantile(per_h[h], self.alpha) for h in range(horizon)])
        return self.q

    def forecast(self, state, tolerance):
        """Roll `state` forward and return, per horizon step, the point and its calibrated half-width, plus the
        `trusted_horizon`: the number of leading steps whose width stays within `tolerance` (predict these,
        recompute the rest). tolerance=inf trusts the whole roll; a tight tolerance trusts only the near steps."""
        if self.q is None:
            raise RuntimeError("call calibrate() first")
        H = len(self.q)
        pred = self.rollout_fn(state, H)
        # the trusted horizon is the first point the width crosses the tolerance; count the leading trusted steps
        within = self.q <= tolerance
        trusted = 0
        for ok in within:
            if ok:
                trusted += 1
            else:
                break                                          # once it exceeds tolerance, everything past is untrusted
        steps = [{"horizon": h + 1, "point": pred[h], "half_width": float(self.q[h]),
                  "trusted": bool(within[h])} for h in range(H)]
        return {"steps": steps, "trusted_horizon": trusted}


def _selftest():
    """On a SMOOTH learnable system the interval widens with horizon and a tolerance gate trusts a long lead; on
    a CHAOTIC system the same gate honestly trusts only a few steps (the Lyapunov-time boundary), reproducing the
    kept negative mechanically."""
    rng = np.random.default_rng(0)

    # (1) SMOOTH system: a slowly-decaying oscillation x_{t+1} = 0.99 * R(theta) x_t. Predictable far ahead.
    theta = 0.15
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) * 0.99

    def smooth_true(state, H):
        traj = []; x = np.asarray(state, float)
        for _ in range(H):
            x = R @ x; traj.append(x[0])                        # observe the first coordinate
        return np.array(traj)

    # the learned rollout is the same operator with a tiny error -> error compounds slowly
    def smooth_roll(state, H):
        traj = []; x = np.asarray(state, float)
        Rn = R * (1.0 + 1e-3)                                  # a slightly-off learned operator
        for _ in range(H):
            x = Rn @ x; traj.append(x[0])
        return np.array(traj)

    H = 30
    states = [rng.standard_normal(2) for _ in range(300)]
    futures = [smooth_true(s, H) for s in states]
    mh = MultiHorizonForecaster(smooth_roll, alpha=0.1, kind="scalar")
    q = mh.calibrate(states, futures, H)
    assert q[-1] >= q[0]                                        # width grows (or holds) with horizon
    smooth_trusted = mh.forecast(rng.standard_normal(2), tolerance=float(np.median(q)))["trusted_horizon"]
    assert smooth_trusted >= 3                                  # a smooth system is trusted several steps out

    # (2) CHAOTIC system: the logistic map r=3.9. A tiny operator error diverges fast -> short trusted horizon.
    def logistic_true(state, H):
        traj = []; x = float(state)
        for _ in range(H):
            x = 3.9 * x * (1 - x); traj.append(x)
        return np.array(traj)

    def logistic_roll(state, H):
        traj = []; x = float(state) + 1e-4                     # a tiny initial error -> chaotic divergence
        for _ in range(H):
            x = 3.9 * x * (1 - x); traj.append(x)
        return np.array(traj)

    cstates = [rng.uniform(0.1, 0.9) for _ in range(300)]
    cfutures = [logistic_true(s, H) for s in cstates]
    mhc = MultiHorizonForecaster(logistic_roll, alpha=0.1, kind="scalar")
    qc = mhc.calibrate(cstates, cfutures, H)
    chaos_trusted = mhc.forecast(0.5, tolerance=0.05)["trusted_horizon"]
    # the chaotic width blows up, so at a tight tolerance only a few steps are trusted, far fewer than the smooth case
    assert chaos_trusted < smooth_trusted or qc[-1] > q[-1] * 5

    print("holographic_horizon selftest OK: smooth system interval widens %.4f->%.4f, trusted %d steps at median "
          "tolerance; chaotic system trusted only %d steps at tol=0.05 (Lyapunov boundary, width ends at %.3f) -- "
          "predict-a-little-recompute-often kept mechanical"
          % (q[0], q[-1], smooth_trusted, chaos_trusted, qc[-1]))


if __name__ == "__main__":
    _selftest()
