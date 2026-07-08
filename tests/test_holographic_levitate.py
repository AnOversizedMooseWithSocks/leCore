"""Acoustic levitation (A7): Gor'kov force traps beads at pressure nodes against gravity; falls when off."""
import numpy as np
from holographic.simulation_and_physics.holographic_levitate import pressure_nodes, gorkov_force_y, LevitationChamber

LAM = 0.0086


def test_node_spacing_is_half_wavelength():
    nodes = pressure_nodes(LAM, 0.10)
    assert np.allclose(np.diff(nodes), LAM / 2.0, atol=1e-9)


def test_force_traps_toward_nodes():
    node = pressure_nodes(LAM, 0.10)[2]
    assert abs(gorkov_force_y(node, LAM, 4000.0)) < abs(gorkov_force_y(node + LAM / 8, LAM, 4000.0))
    assert gorkov_force_y(node - LAM / 8, LAM, 4000.0) > 0        # pushed up from below the node
    assert gorkov_force_y(node + LAM / 8, LAM, 4000.0) < 0        # pushed down from above -> stable trap


def test_field_on_levitates_field_off_falls():
    on = LevitationChamber(height=0.05, wavelength=LAM, amplitude=5000.0, n_beads=30, seed=0)
    on.settle(steps=6000, field_on=True)
    h_on = on.heights()
    assert (h_on > 0.002).mean() > 0.8                           # most beads aloft
    nd = pressure_nodes(LAM, 0.05)
    nearest = np.min(np.abs(h_on[:, None] - nd[None, :]), axis=1)
    assert np.median(nearest) < LAM / 8                          # trapped near the nodes

    off = LevitationChamber(height=0.05, wavelength=LAM, amplitude=5000.0, n_beads=30, seed=0)
    off.settle(steps=8000, field_on=False)
    assert off.heights().mean() < h_on.mean() * 0.5              # gravity wins when the field is off


def test_deterministic():
    a = LevitationChamber(wavelength=LAM, n_beads=10, seed=1); a.settle(500)
    b = LevitationChamber(wavelength=LAM, n_beads=10, seed=1); b.settle(500)
    assert np.array_equal(a.heights(), b.heights())
