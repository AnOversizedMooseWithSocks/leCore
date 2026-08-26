"""CI wrapper for the anisotropic-splat-fit adaptive stop (C3). The module ships its asserts in `_c3_selftest`:
the convergence-gated early-stop saves a good fraction of the fixed Adam steps at a few-percent MSE cost (a soft
plateau, not the resonator's free exact-certificate stop), and is OFF by default. This collects that check."""
from holographic.rendering.holographic_splat import _c3_selftest


def test_holographic_aniso_earlystop_selftest():
    _c3_selftest()
