"""CI wrapper for the gradient-cached decode (Ward's irradiance gradients). The module ships its asserts in
`_selftest`: on a smooth splat / Gaussian-mixture field, first-order gradient interpolation beats the
nearest-neighbour baseline at fixed anchors, roughly matches double the anchors (gradients ~halve the count),
and FAILS badly with global weights -- the validity-radius locality guard is required. This collects that check
into the suite."""
from holographic.caching_and_storage.holographic_cache import _selftest


def test_holographic_cache_selftest():
    _selftest()
