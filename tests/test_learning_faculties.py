"""Integration test: the gradient-free learning faculties (Path D's learning program) wired into
UnifiedMind actually work END-TO-END through the mind, not merely import. Per the project's rule that
wiring is proven by a cross-faculty test, not an import check."""
import numpy as np
from holographic.misc.holographic_unified import UnifiedMind

_MIND = UnifiedMind(dim=1024, seed=0)   # built once; both faculties run on the mind's own dim/seed


def test_reservoir_faculty_through_mind():
    esn = _MIND.reservoir(n_in=1, rho=0.95, leak=0.5)          # gradient-free sequence learner
    t = np.arange(1300); s = np.sin(t / 4.0) + 0.3 * np.sin(t / 9.0)
    esn.fit(s[:900, None], s[1:901], ridge=1e-6, washout=100)  # one ridge solve -- the only training
    pr = esn.predict(s[900:1150, None]).ravel(); tgt = s[901:1151]
    nrmse = float(np.sqrt(np.mean((tgt[60:] - pr[60:]) ** 2) / (np.var(tgt[60:]) + 1e-12)))
    assert nrmse < 0.4, f"reservoir-through-mind next-step NRMSE too high: {nrmse:.3f}"
    assert esn.W_out is not None


def test_prototype_classifier_faculty_through_mind():
    rng = np.random.default_rng(0)
    C, d, per = 3, 8, 80
    centers = rng.standard_normal((C, d)) * 2.0
    Xtr = np.vstack([centers[c] + rng.standard_normal((per, d)) for c in range(C)]); ytr = np.repeat(np.arange(C), per)
    Xte = np.vstack([centers[c] + rng.standard_normal((40, d)) for c in range(C)]); yte = np.repeat(np.arange(C), 40)
    clf = _MIND.prototype_classifier(levels=16)                # gradient-free classifier
    clf.fit(Xtr, ytr, epochs=15)
    assert float(np.mean(clf.predict(Xte) == yte)) > 0.70


def test_equilibrium_faculty_through_mind():
    from holographic.simulation_and_physics.holographic_equilibrium import _moons
    rng = np.random.default_rng(0)
    X, yy = _moons(360, 0.10, rng)
    perm = rng.permutation(len(X)); X, yy = X[perm], yy[perm]   # shuffle -> class-balanced split
    Y = np.eye(2)[yy]; ntr = 260
    net = _MIND.equilibrium_net(n_in=2, n_hidden=48, n_out=2, beta=0.35, dt=0.35, t_free=45, t_nudge=12)
    net.fit(X[:ntr], Y[:ntr], epochs=100, lr=0.3, batch=90, seed=0)   # local-gradient, no backprop
    assert float(np.mean(net.predict(X[ntr:]) == yy[ntr:])) > 0.88    # learns the nonlinear task


def test_forward_forward_faculty_through_mind():
    from holographic.misc.holographic_forward import _blobs
    rng = np.random.default_rng(0)
    X, y = _blobs(560, 16, 4, sep=2.2, rng=rng)
    p = rng.permutation(len(X)); X, y = X[p], y[p]; ntr = 420
    net = _MIND.forward_forward(n_in=16, layer_sizes=(100, 100), n_classes=4, theta=0.05, label_scale=4.0)
    net.fit(X[:ntr], y[:ntr], epochs=60, lr=0.1, batch=100, seed=0)   # local goodness objectives, no backprop
    assert float(np.mean(net.predict(X[ntr:]) == y[ntr:])) > 0.85     # classifies the separable task


def test_chaos_faculty_through_mind():
    from holographic.misc.holographic_chaos import lorenz_trajectory
    tr = lorenz_trajectory(3000, seed=0); ntr = 2000
    prop = _MIND.learn_chaos(tr[:ntr], dim=400, noise=1e-2)        # nonlinear dynamics operator, no backprop
    test = tr[ntr:]; pred = prop.predict_sequence(test)
    rel = lambda p, t: float(np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-12))
    res = np.mean([rel(pred[i], test[i + 1]) for i in range(200, len(test) - 1)])
    pers = np.mean([rel(test[i], test[i + 1]) for i in range(len(test) - 1)])
    assert res < pers / 5.0, f"nonlinear forecaster should beat persistence by >5x on chaotic flow, got {pers / res:.0f}x"


def test_learned_energy_faculty_through_mind():
    from holographic.simulation_and_physics.holographic_energy import torus_bump_manifold
    from holographic.agents_and_reasoning.holographic_hopfield import dense_cleanup
    clean, noisy, D = torus_bump_manifold(n_grid=6, latent_dim=2, sigma=0.13, n_samples=1500, noise=0.30, seed=0)
    mem = _MIND.learn_cleanup(clean, noise=0.30, n_hidden=24, epochs=70)        # EP trains the cleanup's attractors
    cte, nte, _ = torus_bump_manifold(n_grid=6, latent_dim=2, sigma=0.13, n_samples=200, noise=0.30, seed=5)
    rel = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
    ep = np.mean([rel(mem.cleanup(nte[i]), cte[i]) for i in range(len(nte))])
    cb = clean[:64]                                                            # fixed soft energy cleanup over stored samples
    soft = np.mean([rel(dense_cleanup(nte[i], cb, beta=25.0, steps=3), cte[i]) for i in range(len(nte))])
    assert ep < soft, f"learned energy should beat the fixed soft cleanup on a continuous manifold, got ep={ep:.3f} soft={soft:.3f}"
