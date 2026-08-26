"""CI wrapper for looping denoise / diffusion on an arbitrary manifold (XDATA-2). The module ships its asserts in
`_selftest`: on a curved manifold (a unit ring in R^D) the looping denoiser settles points onto the ring
(idempotent), settles an off-manifold interpolation midpoint back on (beating interpolation), and generates
novel-but-valid samples from noise where bare-codebook generation would be degenerate. This collects that check
into the suite."""
from holographic.misc.holographic_diffuse import _selftest


def test_holographic_diffuse_selftest():
    _selftest()
