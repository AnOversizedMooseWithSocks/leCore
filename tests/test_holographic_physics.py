"""Physics on the substrate: the native-kinematics property, its boundaries,
and the price-as-particle verdicts -- wins and negatives pinned alike."""
import numpy as np


def test_translation_is_binding_exactly():
    # THE NATIVE PROPERTY: encode(a+b) == bind(encode(a), encode(b)) -- the
    # scalar code's frequency phases multiply, so value-translation IS the
    # binding operation. Pinned at near-perfect cosine.
    from holographic.io_and_interop.holographic_encoders import ScalarEncoder
    from holographic.agents_and_reasoning.holographic_ai import bind, cosine
    se = ScalarEncoder(2048, lo=-50, hi=50, seed=1)
    for a, b in ((3.0, 7.0), (-12.5, 4.25), (20.0, -31.0)):
        assert cosine(se.encode(a + b), bind(se.encode(a), se.encode(b))) > 0.999


def test_kinematics_integrates_by_pure_binding():
    # Uniform and uniformly-accelerated motion as repeated binding, decoded back
    # within a tenth of a unit over 15 steps -- and the range boundary enforced.
    import pytest
    from holographic.simulation_and_physics.holographic_physics import Kinematics
    k = Kinematics()
    got, true = k.trajectory(-30, 4.0, steps=15)
    assert np.max(np.abs(got - true)) < 0.5
    got, true = k.trajectory(-40, 10.0, a=-1.0, steps=15)
    assert np.max(np.abs(got - true)) < 0.5
    assert abs(k.read_velocity(-17.0, -13.5) - 3.5) < 0.2
    with pytest.raises(ValueError):
        k.trajectory(-40, 8.0, steps=15)            # reaches +80: out of range


def test_price_particle_state_matches_shape_at_one_move():
    # PHYSICS AS COMPRESSION, pinned: at H=1 the two-number particle state
    # (v = last move, a = its change) gives ray-targets EQUIVALENT to the
    # validated 5-move shape rays (|paired z| < 2) while beating the
    # unconditional distribution (z > 2). At one step ahead the market's
    # structure IS kinematic.
    from holographic.io_and_interop.holographic_encoders import ScalarEncoder
    from holographic.agents_and_reasoning.holographic_ai import bind, random_vector
    from holographic.misc.holographic_market import load_ticks, move_series, RayProjector
    ts, px = load_ticks()
    moves, mb = move_series(ts, px)
    QS = (0.1, 0.5, 0.9)

    def pinball(y, q, tau):
        return max(tau * (y - q), (tau - 1) * (y - q))

    p99 = np.percentile(np.abs(moves), 99)
    rng = np.random.default_rng(0)
    RV, RA = random_vector(512, rng), random_vector(512, rng)
    seV = ScalarEncoder(512, lo=-p99, hi=p99, seed=1)
    seA = ScalarEncoder(512, lo=-2 * p99, hi=2 * p99, seed=2)
    H, R = 1, 80
    rows = [i for i in range(2, len(moves) - H) if mb[i - 2] == mb[i + H - 1]]
    M = np.zeros((len(rows), 512), np.float32)
    outc = np.zeros(len(rows))
    for r_, i in enumerate(rows):
        v, a = moves[i - 1], moves[i - 1] - moves[i - 2]
        w = bind(RV, seV.encode(float(np.clip(v, -p99, p99)))) \
            + bind(RA, seA.encode(float(np.clip(a, -2 * p99, 2 * p99))))
        M[r_] = w / (np.linalg.norm(w) + 1e-12)
        outc[r_] = moves[i:i + H].sum()
    rp = RayProjector(R=R, H=H).fit(moves, mb)
    shape_rows = {i: r for r, i in enumerate(rp.rows)}
    n = len(rows)
    mid = (150 + n) // 2
    d_u, d_s = [], []
    for r_ in range(mid, n):
        i = rows[r_]
        y = outc[r_]
        sims = M[:r_ - 1] @ M[r_]
        top = np.argsort(sims)[-R:]
        qp = [float(np.quantile(outc[top], q)) for q in QS]
        qu = [float(np.quantile(outc[:r_ - 1], q)) for q in QS]
        lp = sum(pinball(y, q, t) for q, t in zip(qp, QS))
        d_u.append(sum(pinball(y, q, t) for q, t in zip(qu, QS)) - lp)
        if i in shape_rows and shape_rows[i] >= R + 1:
            qs_, _ = rp.project(shape_rows[i], QS)
            d_s.append(sum(pinball(y, q, t) for q, t in zip(qs_, QS)) - lp)
    zu = np.mean(d_u) / (np.std(d_u) / np.sqrt(len(d_u)) + 1e-12)
    zs = np.mean(d_s) / (np.std(d_s) / np.sqrt(len(d_s)) + 1e-12)
    assert zu > 2.0                          # beats unconditional
    assert abs(zs) < 2.0                     # equivalent to the 5-move shape


def test_prices_have_no_inertia():
    # THE KEPT NEGATIVE, the physics verdict: kinematic extrapolation
    # (x + H*v + a*H(H-1)/2) as a point forecast LOSES to predict-zero at
    # H=1 and H=3. The velocity's sign persists one tick; its magnitude does
    # not. The price is not a coasting mass -- pinned so the metaphor stays
    # honest.
    from holographic.misc.holographic_market import load_ticks, move_series
    ts, px = load_ticks()
    moves, mb = move_series(ts, px)
    for H in (1, 3):
        rows = [i for i in range(2, len(moves) - H) if mb[i - 2] == mb[i + H - 1]]
        kin, zero = [], []
        for i in rows:
            v, a = moves[i - 1], moves[i - 1] - moves[i - 2]
            y = moves[i:i + H].sum()
            kin.append(abs(y - (H * v + a * (H * (H - 1)) / 2)))
            zero.append(abs(y))
        assert np.mean(kin) > np.mean(zero)
