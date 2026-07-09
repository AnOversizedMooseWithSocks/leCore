"""Physics backlog N8: the wave-packet field -- reflection, group velocity, shoaling, surface-as-bundle."""
import numpy as np
from holographic.simulation_and_physics.holographic_wavepacket import WavePacketField, packet_record, surface_bundle
from holographic.agents_and_reasoning.holographic_ai import cosine


def test_reflects_off_wall():
    f = WavePacketField(size=64.0, seed=0)
    f.add_packet(pos=[60.0, 32.0], wavevector=[1.2, 0.0])
    kx0 = f.k[0, 0]
    for _ in range(200):
        f.advance(0.1)
    assert f.k[0, 0] == -kx0 and f.pos[0, 0] < 60.0            # mirrored and heading back


def test_group_velocity_is_half_phase_velocity():
    g = WavePacketField(size=200.0, seed=0)
    kmag = 0.5
    cg = g._group_speed(kmag); cp = g._omega(kmag) / kmag
    assert abs(cg - 0.5 * cp) < 1e-3


def test_packet_moves_at_group_speed():
    g = WavePacketField(size=200.0, seed=0)
    g.add_packet(pos=[10.0, 100.0], wavevector=[0.5, 0.0])
    cg = g._group_speed(0.5)
    x0 = g.pos[0, 0]; g.advance(1.0)
    assert abs((g.pos[0, 0] - x0) - cg) < 1e-2


def test_shoaling_slows_in_shallow_water():
    g = WavePacketField(size=64.0, seed=0)
    assert g._group_speed(0.5, depth=0.5) < g._group_speed(0.5, depth=None)


def test_obstacle_reflects_packet():
    f = WavePacketField(size=64.0, seed=0)
    f.add_packet(pos=[10.0, 32.0], wavevector=[1.0, 0.0])
    box = (28.0, 20.0, 36.0, 44.0)                             # a rock in the middle
    ky_seen = []
    for _ in range(300):
        f.advance(0.2, obstacles=[box])
    # the packet never ends up inside the obstacle
    px, py = f.pos[0]
    assert not (28.0 < px < 36.0 and 20.0 < py < 44.0)


def test_render_is_localized():
    r = WavePacketField(size=64.0, seed=0)
    r.add_packet(pos=[32.0, 32.0], wavevector=[1.5, 0.0], amplitude=1.0)
    h = r.render(res=64)
    row = np.abs(h[32])
    assert row[32] > row[5] and row[32] > row[60]


def test_surface_is_content_addressable_bundle():
    s = WavePacketField(size=64.0, seed=1)
    s.add_packet(pos=[20.0, 20.0], wavevector=[1.0, 0.3])
    s.add_packet(pos=[45.0, 40.0], wavevector=[0.4, 1.1])
    bv, records, roles, enc = surface_bundle(s, dim=4096, seed=0)
    stranger = WavePacketField(size=64.0, seed=9); stranger.add_packet([5.0, 55.0], [2.0, -1.5])
    strec = packet_record(stranger, 0, roles, enc)
    assert cosine(records[0], bv) > cosine(strec, bv)


def test_deterministic():
    def run():
        f = WavePacketField(size=64.0, seed=0); f.add_packet([60.0, 32.0], [1.2, 0.0])
        for _ in range(50):
            f.advance(0.1)
        return f.pos.copy(), f.k.copy()
    p1, k1 = run(); p2, k2 = run()
    assert np.array_equal(p1, p2) and np.array_equal(k1, k2)
