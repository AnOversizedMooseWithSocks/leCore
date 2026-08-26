"""Propagator binding (B4): dynamics as an algebra of binds."""
import numpy as np
from holographic.simulation_and_physics.holographic_dynamics import Propagator
from holographic.agents_and_reasoning.holographic_ai import bind, cosine, random_vector


def _bind_trajectory(D=256, T=400, seed=0):
    rng = np.random.default_rng(seed)
    U = random_vector(D, rng)
    s = random_vector(D, rng)
    traj = [s]
    for _ in range(T):
        s = bind(U, s) + 0.01 * rng.standard_normal(D)
        s /= np.linalg.norm(s)
        traj.append(s)
    return np.array(traj)


def test_step_is_literally_a_bind():
    # "dynamics as an algebra of binds" is exact, not a metaphor: step == bind(U, state).
    traj = _bind_trajectory()
    prop = Propagator.learn(traj[:300])
    x = traj[310]
    assert np.allclose(prop.step(x), bind(prop.U, x))


def test_propagator_predicts_bind_shaped_dynamics():
    # when the dynamics ARE a bind, the propagator recovers the operator and predicts full states,
    # far better than persistence (binding scrambles, so the last state poorly predicts the next).
    traj = _bind_trajectory()
    prop = Propagator.learn(traj[:300])
    pred = np.mean([cosine(prop.step(traj[300 + i]), traj[301 + i]) for i in range(80)])
    persist = np.mean([cosine(traj[300 + i], traj[301 + i]) for i in range(80)])
    assert pred > 0.9 and pred > persist + 0.2


def test_trajectory_is_content_addressable():
    # the durable win: forward k then back k returns the start -- past states are recoverable.
    traj = _bind_trajectory()
    prop = Propagator.learn(traj[:300])
    x = traj[350]
    fwd = prop.rollout(x, 4)[-1]
    back = prop.recall_at(fwd, 4)
    assert cosine(x, back) > 0.99


def test_rollout_shape():
    traj = _bind_trajectory()
    prop = Propagator.learn(traj[:300])
    assert prop.rollout(traj[10], 5).shape == (5, traj.shape[1])


def test_reanchoring_a_rollout_does_not_help_kept_negative():
    # R1 (chunking-transfer): re-anchoring rescues DECODE-via-cleanup chains (a route's per-hop cleanup
    # accumulates crosstalk), but a learned linear propagator rollout is an EVALUATION. Where the operator
    # tracks its model class there is NO drift to fix, and projecting the state onto the training-state
    # manifold only discards valid forward signal. Pinned so nobody "fixes" a non-problem.
    rng = np.random.default_rng(0)
    dim, K = 256, 8
    f = rng.integers(2, dim // 4, K)
    a = rng.uniform(0.5, 1.0, K)
    dphi = rng.uniform(0.1, 0.4, K)
    n = np.arange(dim)
    # a trajectory whose dynamics ARE a per-frequency phase advance == a circular convolution (the propagator's
    # exact model class): state(t)[n] = sum_k a_k cos(2*pi*f_k*n/dim + t*dphi_k)
    state = lambda t: np.sum([a[k] * np.cos(2 * np.pi * f[k] * n / dim + t * dphi[k]) for k in range(K)], axis=0)
    T = 100
    X = np.array([state(t) for t in range(T)])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    train, horizon = 40, 50
    prop = Propagator.learn(X[:train], ridge=1e-3)
    # consolidation manifold = top singular directions of the training states (99% energy)
    U_, S_, Vt = np.linalg.svd(X[:train], full_matrices=False)
    r = int(np.searchsorted(np.cumsum(S_ ** 2) / np.sum(S_ ** 2), 0.99)) + 1
    basis = Vt[:r].T

    def rollout_drift(reanchor):
        st = X[train - 1].copy()
        errs = []
        for i in range(horizon):
            st = prop.step(st)
            if reanchor and (i + 1) % reanchor == 0:
                p = basis @ (basis.T @ st)
                st = p / (np.linalg.norm(p) + 1e-12) * np.linalg.norm(st)
            j = train - 1 + i + 1
            if j < T:
                errs.append(1 - cosine(st, X[j]))
        return float(np.mean(errs))

    free = rollout_drift(0)
    anchored = rollout_drift(4)
    assert free < 0.05         # the propagator TRACKS its model class: essentially no drift to fix
    assert anchored > free     # re-anchoring only HURTS -- it discards valid forward signal
