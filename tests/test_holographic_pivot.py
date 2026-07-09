"""CI wrapper for the recursive pivot-tree sublinear-index module (Path D). Its _selftest asserts the core
finding -- greedy top-1 routing matches the exhaustive scan at a fraction of the comparisons, and a beam lands
the true leaf in the candidate set nearly always -- on a fixed well-separated hierarchical leaf set."""
from holographic.misc.holographic_pivot import _selftest


def test_holographic_pivot_selftest():
    _selftest()
