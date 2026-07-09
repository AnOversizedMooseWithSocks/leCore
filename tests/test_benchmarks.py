"""Tests for the external-baseline benchmark harness (BLD-2): the comparisons run reproducibly and the honest
directional findings hold -- INCLUDING where the standard tool wins."""
from benchmarks.bench_compression import compare_compression
from benchmarks.bench_recall import compare_recall


def test_compression_rd_wins_on_structure_loses_on_random():
    rows = {(r["dataset"], r["N"]): r for r in compare_compression(Ns=(2000,))}
    # rd exploits low-rank structure -> far fewer bits than int8+zlib at matched fidelity
    s = rows[("structured", 2000)]
    assert s["rd_bits"] < s["int8_zlib_bits"] and s["rd_cos"] >= 0.999
    # the KEPT NEGATIVE: full-rank random data has no structure to exploit, so the standard tool wins
    r = rows[("random", 2000)]
    assert r["rd_bits"] > r["int8_zlib_bits"]


def test_compression_is_reproducible():
    a = compare_compression(Ns=(200,))[0]["rd_bits"]
    b = compare_compression(Ns=(200,))[0]["rd_bits"]
    assert a == b                                        # deterministic: same seed -> identical bits/vector


def test_recall_forest_is_sublinear_and_near_exact():
    rows = {r["N"]: r for r in compare_recall(Ns=(500, 2000), Q=40)}
    for N in (500, 2000):
        r = rows[N]
        assert r["forest_cmp"] < r["brute_cmp"]          # sublinear: fewer comparisons than the exact scan
        assert r["forest_recall1"] >= 0.95               # and recall stays near-exact at this scale
