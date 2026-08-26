"""CI wrapper for adaptive curvature-driven anchor placement (CACHE-3). The module ships its asserts in `_selftest`:
on a field with non-uniform smoothness, placing anchors denser where the field bends (density ~ |curvature|^1/2)
matches uniform-placement quality at materially fewer anchors and is far better at a fixed count; on a
uniformly-smooth field the two are ~tied (the honest control). This collects that check into the suite."""
from holographic.caching_and_storage.holographic_adaptive_cache import _selftest


def test_holographic_adaptive_cache_selftest():
    _selftest()
