"""Oxidation (M4): corrosion nucleates at exposed faces and spreads inward as a front; monotonic; base->oxide blend."""
import numpy as np
from holographic.simulation_and_physics.holographic_oxidation import OxidationField, oxide_color, OXIDATION


def test_front_spreads_from_exposed_faces_inward():
    f = OxidationField((21, 21))                                    # border exposed by default
    edge, centre = [], []
    for _ in range(30):
        f.step("steel", dt=1.0)
        edge.append(f.ox[0, 10]); centre.append(f.ox[10, 10])
    assert edge[-1] > centre[-1]                                    # a front: edge ahead of centre
    assert centre[-1] > 0.0                                         # but it reached the centre
    assert all(centre[i + 1] >= centre[i] - 1e-12 for i in range(len(centre) - 1))   # monotonic


def test_material_rate_and_moisture_matter():
    s = OxidationField((15, 15)); cu = OxidationField((15, 15))
    for _ in range(20):
        s.step("steel"); cu.step("copper")
    assert s.fraction() > cu.fraction()                            # steel rusts faster than copper patinas
    wet = OxidationField((15, 15), moisture=1.0); dry = OxidationField((15, 15), moisture=0.1)
    for _ in range(20):
        wet.step("steel"); dry.step("steel")
    assert wet.fraction() > dry.fraction()                         # wet corrodes faster


def test_base_to_oxide_blend():
    rust = oxide_color("steel", 1.0); patina = oxide_color("copper", 1.0)
    assert rust[0] > rust[2]                                       # rust orange-brown
    assert patina[1] > patina[0]                                   # patina green
    f = OxidationField((10, 10)); alb = f.albedo("steel")
    assert alb.shape == (10, 10, 3) and alb.min() >= 0 and alb.max() <= 1


def test_deterministic():
    a = OxidationField((10, 10)); b = OxidationField((10, 10))
    for _ in range(10):
        a.step("iron"); b.step("iron")
    assert np.array_equal(a.ox, b.ox)
