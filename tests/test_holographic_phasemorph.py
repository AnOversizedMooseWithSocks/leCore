"""CI wrapper for the FHRR phase-domain morph (PHASE-1). The module ships its asserts in `_selftest`: an
FHRR-encoded position morphed across a large change moves at constant velocity under the phase morph (tracks the
ideal trajectory; the amplitude blend eases and collapses in magnitude), with the kept negative that under extreme
change the shortest-arc morph wraps and loses tracking. This collects that check into the suite."""
from holographic.simulation_and_physics.holographic_phasemorph import _selftest, _c2_selftest


def test_holographic_phasemorph_selftest():
    _selftest()


def test_holographic_phasemorph_image_c2_selftest():
    _c2_selftest()
