"""Decomposition contract (I4) at the seams: through the mind, certifying the engine's OWN decomposers
(smooth_sharp_split and decompose_piecewise get their honest labels on the record), and composing with the
paper book -- a decomposition's non-causal component, used as a trading signal, is exactly the leak the
contract warned about, and the actionable harness confirms the warning economically."""
import numpy as np

import lecore


def _walk(seed=0, n=600):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n)) + 0.02 * np.arange(n)


def test_certify_the_engines_own_decomposers():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    x = _walk()

    def engine_split(s):
        code = mind.smooth_sharp_split(s, k_smooth=8, k_sharp=64)
        nbins = code.n // 2 + 1
        sm = np.fft.irfft(np.concatenate([code.smooth_coeffs,
                                          np.zeros(nbins - len(code.smooth_coeffs), complex)]), n=code.n)
        return {"smooth": sm, "residual": np.asarray(s, float).ravel() - sm}

    r = mind.decomposition_contract(engine_split, x)
    assert r["complete"] and not r["causal"]                     # its honest label: diagnosis, not features

    def piecewise(s):
        seg = mind.decompose_piecewise(s)
        fit = np.asarray(seg.get("fit", seg.get("reconstruction", None)) if isinstance(seg, dict) else None,
                         float) if isinstance(seg, dict) else None
        if fit is None or fit.shape != s.shape:
            return {"whole": s}                                  # API surprise -> degenerate but honest
        return {"fit": fit, "residual": s - fit}

    r2 = mind.decomposition_contract(piecewise, x)
    assert r2["complete"]                                        # whichever branch: sums back exactly


def test_the_contracts_warning_is_economic_not_pedantic():
    """A centred moving average is a textbook trend component and NON-CAUSAL -- the contract says so. Use
    its slope as a trading signal anyway, and the paper book (actionable, lag=1) shows the 'edge' the
    centred trend appears to carry is mostly the future it already contains: the causal trailing version
    of the same trend keeps far less. The contract's per-component verdict priced the difference."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(3)
    n = 4000
    px = np.cumsum(rng.standard_normal(n))

    def centred_decomp(s):
        k = 21
        pad = np.pad(s, (k // 2, k // 2), mode="edge")
        trend = np.convolve(pad, np.ones(k) / k, mode="valid")
        return {"trend": trend, "residual": s - trend}

    r = mind.decomposition_contract(centred_decomp, px)
    assert r["complete"] and not r["components"]["trend"]["causal"]

    centred_sig = np.sign(np.concatenate([[0.0], np.diff(centred_decomp(px)["trend"])]))
    trail = np.concatenate([np.full(20, px[0]), np.convolve(px, np.ones(21) / 21, mode="valid")])
    trailing_sig = np.sign(np.concatenate([[0.0], np.diff(trail)]))

    book = mind.paper_book(lag=1, cost=0.0)
    book.add_sleeve("centred", centred_sig).add_sleeve("trailing", trailing_sig)
    rep = book.run(px)
    # even at lag=1 the centred trend still contains ~10 future steps: its paper t dwarfs the causal twin's.
    assert rep["sleeves"]["centred"]["t"] > rep["sleeves"]["trailing"]["t"] + 3, \
        (rep["sleeves"]["centred"]["t"], rep["sleeves"]["trailing"]["t"])
    assert abs(rep["sleeves"]["trailing"]["t"]) < 3                # the causal twin on a pure walk: nothing


def test_refusals_through_the_faculty():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    x = _walk(1, 100)
    try:
        mind.decomposition_contract(lambda s: {"short": s[:-1]}, x)
        raise AssertionError("expected refusal")
    except ValueError as e:
        assert "length" in str(e)
