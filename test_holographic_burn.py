"""Burn/decay (M7): a lit object loses mass, appearance base->char->ash, ends as ash; evaporation drains a puddle."""
import numpy as np
from holographic_burn import BurningObject, char_color, evaporate, _base_albedo


def test_lit_object_burns_down_to_ash():
    cold = BurningObject("wood", 1.0)
    assert not cold.step(0.5)["burning"] and cold.mass == 1.0       # unlit at room temp
    obj = BurningObject("wood", 1.0).light()
    masses = []
    for _ in range(80):
        masses.append(obj.step(0.5)["mass"])
    assert masses[0] < 1.0 and masses[-1] < 0.05                    # burned down
    assert all(masses[i + 1] <= masses[i] + 1e-12 for i in range(len(masses) - 1))   # monotonic
    assert obj.is_ash()


def test_appearance_marches_base_char_ash():
    base = _base_albedo("wood")
    mid = char_color("wood", 0.5); end = char_color("wood", 1.0)
    assert mid.mean() < base.mean()                                # char darkens
    assert end.mean() > mid.mean()                                 # ash paler than char
    assert np.allclose(char_color("wood", 0.0), base)              # unburned pristine


def test_emits_smoke_while_burning():
    obj = BurningObject("wood", 1.0).light()
    saw = any(obj.step(0.5)["smoke_color"].mean() > 0.3 and obj.step(0.5) for _ in range(10))
    obj2 = BurningObject("pvc_plastic", 1.0).light()
    s = obj2.step(0.5)
    assert s["smoke_color"].mean() < 0.2                           # PVC black smoke while burning


def test_evaporation_drains_puddle():
    liq = evaporate("water", 1.0, temp_K=355.0, energy_per_step=2.0e5, steps=15)
    assert liq[-1] < liq[0]


def test_deterministic():
    a = BurningObject("wood", 1.0).light(); b = BurningObject("wood", 1.0).light()
    ra, rb = a.step(0.5), b.step(0.5)
    assert ra["mass"] == rb["mass"] and np.array_equal(ra["appearance"], rb["appearance"])
