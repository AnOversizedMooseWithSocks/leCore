"""Factoring a composite into its parts by searching in superposition: the
resonator recovers the bound factors from a combinatorial space far larger than it
enumerates, with random restarts, and reports honestly when it can't."""
import numpy as np
import pytest

from holographic.misc.holographic_resonator import ResonatorNetwork, map_codebook, map_bind


def test_factors_a_three_way_binding():
    books = [map_codebook(40, 1500, s) for s in range(3)]
    rn = ResonatorNetwork(books)
    rng = np.random.default_rng(0)
    true = [int(rng.integers(40)) for _ in range(3)]
    c = map_bind(books[0][true[0]], books[1][true[1]], books[2][true[2]])
    r = rn.factor(c, restarts=30)
    assert r["solved"]
    assert r["factors"] == tuple(true)


def test_searches_more_than_it_enumerates():
    books = [map_codebook(40, 1500, s) for s in range(3)]
    rn = ResonatorNetwork(books)
    rng = np.random.default_rng(1)
    true = [int(rng.integers(40)) for _ in range(3)]
    c = map_bind(*[books[f][true[f]] for f in range(3)])
    r = rn.factor(c, restarts=30)
    assert r["search_space"] == 40 ** 3            # 64,000 combinations
    assert r["solved"]


def test_map_bind_is_self_inverse():
    a = map_codebook(1, 500, 0)[0]
    b = map_codebook(1, 500, 1)[0]
    # binding b in and then again cancels it out
    assert np.array_equal(map_bind(map_bind(a, b), b), a)


def test_two_factor_problem():
    books = [map_codebook(60, 1200, s) for s in range(2)]
    rn = ResonatorNetwork(books)
    rng = np.random.default_rng(2)
    true = [int(rng.integers(60)) for _ in range(2)]
    c = map_bind(books[0][true[0]], books[1][true[1]])
    r = rn.factor(c, restarts=30)
    assert r["solved"] and r["factors"] == tuple(true)


def test_solved_flag_reports_honestly():
    # too few restarts on a hard problem may not solve; the flag must say so rather
    # than return a wrong answer as if correct.
    books = [map_codebook(120, 800, s) for s in range(3)]   # hard: small dim, big books
    rn = ResonatorNetwork(books)
    rng = np.random.default_rng(3)
    true = [int(rng.integers(120)) for _ in range(3)]
    c = map_bind(*[books[f][true[f]] for f in range(3)])
    r = rn.factor(c, restarts=1, iters=50)
    # whatever it returns, solved is only True if the factors actually re-bind to c
    rec = map_bind(*[books[f][r["factors"][f]] for f in range(3)])
    assert r["solved"] == bool(np.array_equal(rec, c))


def test_brain_factor_composite():
    from holographic.misc.holographic_unified import UnifiedMind
    books = [map_codebook(40, 1500, s) for s in range(3)]
    rng = np.random.default_rng(4)
    true = [int(rng.integers(40)) for _ in range(3)]
    c = map_bind(*[books[f][true[f]] for f in range(3)])
    m = UnifiedMind(dim=256, seed=0)
    # the dense MAP path is the LEGACY route (the SBC resonator cannot factor this algebra), so it is
    # kept working but deprecated -- assert both: it still solves, and it warns.
    with pytest.warns(DeprecationWarning):
        r = m.factor_composite(c, books, restarts=30)
    assert r["solved"] and r["factors"] == tuple(true)


# ---------------------------------------------------------------------------------------------------------------
# THE NOISE-TOLERANT EXIT (CLM backlog E3.2). The exact exit solved 0/20 products with 5% flipped components (757 ms
# each) and took 1,858 ms to refuse a random composite (panel, exp_a_resonator.py). tolerant=True keeps the update
# rule and adds an agreement-based exit with a procedure-matched p-value; the default path must not move.
# Small shapes so each test stays fast; the full before/after is tools/bench_rolecall.py 'resonator'.
def _perm_books(dim=1024, nv=8, nc=10):
    vals = map_codebook(nc, dim, 12)
    return [map_codebook(nv, dim, 11), np.roll(vals, 1, axis=1), np.roll(vals, 2, axis=1)]


_TOL = dict(restarts=10, iters=80, tolerant=True, patience=5, accept_p=0.05, m_null=30)


def test_tolerant_exit_recovers_noisy_products_that_the_exact_exit_cannot():
    books = _perm_books()
    rn = ResonatorNetwork(books)
    rng = np.random.default_rng(7)
    right = accepted = accepted_wrong = exact_solved = 0
    for t in range(6):
        true = (int(rng.integers(8)), int(rng.integers(10)), int(rng.integers(10)))
        c = map_bind(*[books[f][true[f]] for f in range(3)])
        c = np.where(rng.random(1024) < 0.05, -c, c)                  # 5% flipped components
        exact_solved += rn.factor(c, restarts=2, iters=40)["solved"]   # the old exit: never exact on a noisy input
        r = rn.factor(c, **_TOL)
        ok = r["factors"] == true
        right += ok
        accepted += r["accepted"]
        accepted_wrong += r["accepted"] and not ok
        assert r["solved"] is False and 0.0 < r["p_value"] <= 1.0      # 'solved' keeps meaning EXACT only
    assert exact_solved == 0
    assert right >= 5 and accepted >= 5 and accepted_wrong == 0


def test_tolerant_exit_refuses_a_random_composite_with_a_large_p():
    rn = ResonatorNetwork(_perm_books())
    r = rn.factor(map_codebook(1, 1024, 99)[0], **_TOL)
    assert r["accepted"] is False and r["p_value"] > 0.05 and r["agreement"] < 0.6


def test_tolerant_exit_on_an_exact_product_is_the_exact_answer():
    books = _perm_books()
    rn = ResonatorNetwork(books)
    c = map_bind(books[0][3], books[1][4], books[2][7])
    r = rn.factor(c, **_TOL)
    assert r["factors"] == (3, 4, 7) and r["solved"] and r["exit"] == "exact" and r["agreement"] == 1.0
    assert r["p_value"] == pytest.approx(1.0 / 31)                     # nothing in the null reaches 1.0


def test_default_path_is_unchanged_by_the_tolerant_option():
    # the exact-exit contract other tests pin: same factors, same restart count, no new keys
    books = [map_codebook(40, 1500, s) for s in range(3)]
    rn = ResonatorNetwork(books)
    c = map_bind(books[0][5], books[1][17], books[2][30])
    r = rn.factor(c, restarts=30)
    assert set(r) == {"factors", "solved", "restarts", "iterations", "search_space"}
    assert r["solved"] and r["factors"] == (5, 17, 30)


def test_tolerant_null_is_a_property_of_the_shape_not_the_content():
    from holographic.misc import holographic_resonator as R
    a = ResonatorNetwork(_perm_books()).noise_null(restarts=4, iters=40, patience=4, m=20)
    R._TOLERANT_NULL_CACHE.clear()
    vals = map_codebook(10, 1024, 99)
    b = ResonatorNetwork([map_codebook(8, 1024, 98), np.roll(vals, 1, axis=1), np.roll(vals, 2, axis=1)]
                         ).noise_null(restarts=4, iters=40, patience=4, m=20)
    assert abs(float(np.median(a)) - float(np.median(b))) < 0.01 and b.max() < 0.6


def test_accept_threshold_matches_the_p_value_rule():
    null = np.array([0.50, 0.51, 0.52, 0.53, 0.54])
    # p = (1 + #null >= a) / 6 <= 0.2  <=>  #null >= a <= 0.2  ->  a must exceed the max
    assert ResonatorNetwork.accept_threshold(null, 0.2) == 0.54
    assert ResonatorNetwork.accept_threshold(null, 0.34) == 0.53     # one null value may reach it
    assert ResonatorNetwork.accept_threshold(null, 0.1) is None      # m=5 can never testify to p <= 0.1
