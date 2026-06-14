"""Factoring a composite into its parts by searching in superposition: the
resonator recovers the bound factors from a combinatorial space far larger than it
enumerates, with random restarts, and reports honestly when it can't."""
import numpy as np

from holographic_resonator import ResonatorNetwork, map_codebook, map_bind


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
    from holographic_unified import UnifiedMind
    books = [map_codebook(40, 1500, s) for s in range(3)]
    rng = np.random.default_rng(4)
    true = [int(rng.integers(40)) for _ in range(3)]
    c = map_bind(*[books[f][true[f]] for f in range(3)])
    m = UnifiedMind(dim=256, seed=0)
    r = m.factor_composite(c, books, restarts=30)
    assert r["solved"] and r["factors"] == tuple(true)
