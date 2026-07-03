"""Wave field (A3): d'Alembert split at c, CFL stability, speed scales with c, absorbing border, determinism."""
import numpy as np
from holographic_wave import WaveField


def test_dalembert_split_at_c():
    N = 400; c = 1.0
    w = WaveField((N,), c=c, dx=1.0)
    w.pulse((N // 2,), amp=1.0, radius=4.0)
    T = 120.0
    w.step(dt=T)
    peak_r = N // 2 + int(np.argmax(w.p[N // 2:]))
    peak_l = int(np.argmax(w.p[:N // 2]))
    assert abs((peak_r - N // 2) - c * T) < 12                     # right mover at c*T
    assert abs((N // 2 - peak_l) - c * T) < 12                     # left mover at c*T
    assert np.isfinite(w.p).all()


def test_cfl_autosubstep_keeps_energy_bounded():
    w = WaveField((120, 120), c=2.0, dx=1.0)
    w.pulse((60, 60), amp=1.0, radius=3.0)
    e0 = w.energy()
    w.step(dt=200.0)                                               # far past one CFL step -> must auto-substep
    assert np.isfinite(w.p).all() and w.energy() < 5.0 * e0


def test_speed_scales_with_c():
    def front(cval):
        wf = WaveField((300,), c=cval, dx=1.0); wf.pulse((150,), radius=4.0); wf.step(dt=50.0)
        return int(np.argmax(wf.p[150:]))
    assert front(2.0) > front(1.0) * 1.5


def test_absorbing_border_soaks_energy():
    hard = WaveField((160,), c=1.0); hard.pulse((80,), radius=3.0)
    soft = WaveField((160,), c=1.0, absorb_border=30); soft.pulse((80,), radius=3.0)
    for _ in range(6):
        hard.step(dt=20.0); soft.step(dt=20.0)
    assert soft.energy() < hard.energy() * 0.6


def test_deterministic():
    a = WaveField((80,), c=1.0); a.pulse((40,)); a.step(dt=20.0)
    b = WaveField((80,), c=1.0); b.pulse((40,)); b.step(dt=20.0)
    assert np.array_equal(a.p, b.p)
