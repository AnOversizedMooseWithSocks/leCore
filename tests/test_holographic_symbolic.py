"""MDL-gated symbolic regression: decompose foreign data into a compact, extrapolating law."""
import numpy as np
from holographic.agents_and_reasoning.holographic_symbolic import symbolic_regress, full_fit


def _rms(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))


def test_recovers_known_law_and_extrapolates():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 6, 240); xe = np.linspace(6, 9, 120)
    true = lambda t: 2.0 * np.sin(1.5 * t) + 0.5 * t
    y = true(x) + 0.05 * rng.standard_normal(len(x))
    f, info = symbolic_regress(x, y)
    assert info["n_terms"] <= 3                              # parsimonious
    assert _rms(f.generate(xe), true(xe)) < 0.1             # extrapolates (true law recovered)
    kinds = {(k, round(p, 1)) for (k, p), _ in f.terms}
    assert ("sin", 1.5) in kinds                            # found the oscillation


def test_mdl_gate_beats_maxfit_on_extrapolation():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 6, 240); xe = np.linspace(6, 9, 120)
    true = lambda t: 2.0 * np.sin(1.5 * t) + 0.5 * t
    y = true(x) + 0.05 * rng.standard_normal(len(x))
    f, _ = symbolic_regress(x, y)
    ff = full_fit(x, y)
    mdl_ex = _rms(f.generate(xe), true(xe))
    max_ex = _rms(ff.generate(xe), true(xe))
    assert mdl_ex < max_ex / 100                            # parsimony extrapolates; max-fit explodes


def test_refuses_to_fit_pure_noise():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 6, 240)
    f, info = symbolic_regress(x, rng.standard_normal(240))
    assert info["n_terms"] <= 2                             # no spurious law manufactured
    # and whatever it returns must not blow up out of range (honest refusal, not overfit)
    assert np.max(np.abs(f.generate(np.linspace(6, 9, 60)))) < 10


def test_formula_is_a_compact_seed():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 6, 240)
    y = np.sin(2.0 * x) + 0.05 * rng.standard_normal(240)
    f, info = symbolic_regress(x, y)
    assert f.model_bits(info["dict_size"]) < len(x) * 20    # the seed is far smaller than the data


def test_generate_matches_within_window():
    rng = np.random.default_rng(4)
    x = np.linspace(0, 6, 240)
    true = lambda t: 1.0 + 0.5 * np.cos(1.0 * t)
    y = true(x) + 0.02 * rng.standard_normal(240)
    f, _ = symbolic_regress(x, y)
    assert _rms(f.generate(x), true(x)) < 0.05


def test_formula_is_a_saveable_seed_roundtrip():
    from holographic.agents_and_reasoning.holographic_symbolic import Formula
    rng = np.random.default_rng(7)
    x = np.linspace(0, 6, 240)
    y = 2.0 * np.sin(1.5 * x) + 0.5 * x + 0.05 * rng.standard_normal(240)
    f, _ = symbolic_regress(x, y)
    f.save("/tmp/_law.seed")
    g = Formula.load("/tmp/_law.seed")
    xe = np.linspace(6, 9, 60)
    assert np.allclose(f.generate(xe), g.generate(xe))           # the seed reloads and regenerates exactly
    assert Formula.from_recipe(f.to_recipe()).to_recipe() == f.to_recipe()


def test_compress_signal_one_call_end_to_end():
    from holographic.agents_and_reasoning.holographic_symbolic import compress_signal
    rng = np.random.default_rng(8)
    x = np.linspace(0, 6, 240); xe = np.linspace(6, 9, 120)
    true = lambda t: 2.0 * np.sin(1.5 * t) + 0.5 * t
    y = true(x) + 0.05 * rng.standard_normal(240)
    seed, info = compress_signal(x, y, path="/tmp/_e2e.seed")
    assert "compression_ratio" in info
    assert _rms(seed.generate(xe), true(xe)) < 0.1              # decompose->seed->extrapolate, one call


def test_multiplicative_mode_recovers_a_product_law():
    rng = np.random.default_rng(0)
    x = np.linspace(0.2, 4, 240); xe = np.linspace(4, 5.5, 80)
    true = lambda t: 2.0 * t ** 1.5 * np.exp(0.3 * t)
    y = true(x) * np.exp(0.03 * rng.standard_normal(len(x)))
    f, info = symbolic_regress(x, y, multiplicative=True)
    assert info["multiplicative"] and f.log_space
    kinds = {k for (k, p), _ in f.terms}
    assert "log" in kinds and "pow" in kinds                 # power-law + exponential factors
    rel = np.sqrt(np.mean((f.generate(xe) - true(xe)) ** 2)) / np.mean(np.abs(true(xe)))
    assert rel < 0.05                                        # the product law extrapolates


def test_multiplicative_requires_positive_y():
    x = np.linspace(0.2, 4, 50); y = np.sin(x)               # has negatives
    try:
        symbolic_regress(x, y, multiplicative=True)
        assert False, "should have raised"
    except ValueError:
        pass


def test_log_space_formula_roundtrips_and_exponentiates():
    from holographic.agents_and_reasoning.holographic_symbolic import Formula
    f = Formula(0.7, [(("log", 1), 1.5), (("pow", 1), 0.3)], log_space=True)
    g = Formula.from_recipe(f.to_recipe())
    assert g.log_space
    x = np.linspace(0.5, 3, 20)
    assert np.allclose(f.generate(x), g.generate(x))
    assert np.all(f.generate(x) > 0)                         # exp(...) is positive


def test_auto_never_false_positives_on_additive_data():
    from holographic.agents_and_reasoning.holographic_symbolic import compress_signal
    x = np.linspace(0.2, 4, 240)
    for s in range(4):
        rng = np.random.default_rng(s)
        y = 3.0 + np.sin(1.5 * x) + 0.5 * x + 0.03 * rng.standard_normal(len(x))   # additive, y>0
        _, info = compress_signal(x, y, mode="auto")
        assert info["mode"] == "additive"                   # conservative: doesn't mis-pick multiplicative


def test_multiplicative_beats_additive_extrapolation_on_product_law():
    rng = np.random.default_rng(1)
    x = np.linspace(0.2, 4, 240); xe = np.linspace(4, 5.5, 80)
    true = lambda t: 2.0 * t ** 1.5 * np.exp(0.3 * t)
    y = true(x) * np.exp(0.03 * rng.standard_normal(len(x)))
    fa, _ = symbolic_regress(x, y, multiplicative=False)
    fm, _ = symbolic_regress(x, y, multiplicative=True)
    rel = lambda f: np.sqrt(np.mean((f.generate(xe) - true(xe)) ** 2)) / np.mean(np.abs(true(xe)))
    assert rel(fm) <= rel(fa)                               # the true family extrapolates at least as well
