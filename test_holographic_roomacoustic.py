"""Room acoustics (A6): direct sound at d/c, reflections later at geometric delays, RT60 drops with absorption."""
import numpy as np
from holographic_roomacoustic import ShoeboxRoom


def test_direct_and_reflection_delays():
    room = ShoeboxRoom(size=(6.0, 4.0, 3.0), absorption=0.05, c=343.0)
    src = (1.0, 2.0, 1.5); lis = (5.0, 2.0, 1.5)
    taps = room.reflections(src, lis, max_order=1)
    assert taps[0]["order"] == 0 and abs(taps[0]["delay"] - 4.0 / 343.0) < 1e-9      # direct = |s-r|/c
    assert all(t["delay"] >= taps[0]["delay"] - 1e-12 for t in taps)                 # reflections arrive later
    floor = np.sqrt(4.0 ** 2 + 3.0 ** 2) / 343.0                                     # floor bounce geometry
    assert any(abs(t["delay"] - floor) < 1e-6 for t in taps if t["order"] == 1)
    assert all(t["amplitude"] < taps[0]["amplitude"] for t in taps if t["order"] == 1)


def test_rt60_drops_with_absorption():
    live = ShoeboxRoom(size=(6, 4, 3), absorption=0.03)
    dead = ShoeboxRoom(size=(6, 4, 3), absorption=0.45)
    assert live.rt60() > dead.rt60() * 5
    assert live.rt60() > 1.0 and dead.rt60() < 0.5
    refl_live = sum(t["amplitude"] ** 2 for t in live.reflections((1, 2, 1.5), (5, 2, 1.5)) if t["order"] >= 1)
    refl_dead = sum(t["amplitude"] ** 2 for t in dead.reflections((1, 2, 1.5), (5, 2, 1.5)) if t["order"] >= 1)
    assert refl_live > refl_dead


def test_named_materials_and_rir():
    assert ShoeboxRoom(size=(6, 4, 3), material="concrete").rt60() > ShoeboxRoom(size=(6, 4, 3), material="carpet").rt60()
    rir, fs = ShoeboxRoom(size=(5, 4, 3), absorption=0.1).impulse_response((1, 1, 1), (4, 3, 2))
    assert np.isfinite(rir).all() and rir.max() > 0.0 and fs > 0
