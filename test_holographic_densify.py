"""CI wrapper for coarse-to-fine splat densification (C1). The module ships its asserts in `_c1_selftest`:
coarse-to-fine densify_fit reaches a markedly better optimum than the one-shot aniso_fit on a multi-scale target,
because the staged placement is a far better warm start -- it lands in a basin the one-shot cannot reach at any
step count (directly addressing aniso_fit's local-optimum kept negative). This collects that check."""
from holographic_splat import _c1_selftest


def test_holographic_densify_selftest():
    _c1_selftest()
