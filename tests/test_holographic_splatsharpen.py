"""CI wrapper for the C4 negative (cross-cutting XDATA-3 -> splat reconstruction). The module ships its asserts
in `_selftest`: the 2-D Van Cittert sharpener recovers detail from a GENUINE blur (control -- machinery works),
but sharpening a SPLAT RENDER does not help at any setting (the render is sum-of-Gaussians(centres), not
blur(truth), so you cannot recover the detail the lossy splat basis discarded). A kept negative. This collects
that check."""
from holographic.rendering.holographic_splatsharpen import _selftest


def test_holographic_splatsharpen_negative_selftest():
    _selftest()
