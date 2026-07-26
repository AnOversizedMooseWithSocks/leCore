"""Traps for two capabilities that WORKED and were TESTED but were exposed nowhere until now.

Both were surfaced by the function-granularity reachability audit's TEST-ONLY bucket -- code that passes
every module-level audit while being unreachable from find_capability or /invoke, and therefore (by this
repo's own governing rule) code that formally did not exist.
"""
import numpy as np
import pytest

import lecore


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


def test_antiperiodic_fraction_reads_the_three_reference_cases(mind):
    """The numeric contract, not "no exception". Two periods in; the halves ARE the periods.

    A signal that INVERTS across the half-way point is pure antiperiodic (1.0); one that REPEATS is pure
    periodic (0.0); their sum splits the energy exactly evenly (0.5). The 0.5 case is the real trap -- an
    implementation that merely thresholded would still pass the first two."""
    t = np.arange(256)
    anti = np.cos(np.pi * t / 128)        # f(t+128) = -f(t)
    per = np.cos(2 * np.pi * t / 128)     # f(t+128) = +f(t)
    assert mind.antiperiodic_fraction(anti) == pytest.approx(1.0, abs=1e-9)
    assert mind.antiperiodic_fraction(per) == pytest.approx(0.0, abs=1e-9)
    assert mind.antiperiodic_fraction(anti + per) == pytest.approx(0.5, abs=1e-9)


def test_antiperiodic_split_is_exact_and_reconstructs(mind):
    """The split is orthogonal and lossless: the parts sum back to the first period, and each part recovers
    its own generator exactly. allclose at default tolerance would hide a bin-parity bug, so this asserts
    against the generators themselves rather than against a round trip alone."""
    t = np.arange(256)
    anti = np.cos(np.pi * t / 128)
    per = np.cos(2 * np.pi * t / 128)
    p, a = mind.antiperiodic_split(anti + per)
    assert np.allclose(p + a, (anti + per)[:128], atol=1e-12), "the parts do not sum back to the first period"
    assert np.allclose(p, per[:128], atol=1e-12), "the periodic part did not recover its generator"
    assert np.allclose(a, anti[:128], atol=1e-12), "the antiperiodic part did not recover its generator"


def test_a_pure_circle_signal_is_reported_as_safe_for_a_circular_encoder(mind):
    """The decision the diagnostic exists to make: low fraction -> a circle can carry this."""
    t = np.arange(512)
    assert mind.antiperiodic_fraction(np.sin(2 * np.pi * t / 256)) < 1e-9


def test_load_ies_parses_a_minimal_lm63_file(mind):
    """A real LM-63 header shape, parsed to (candela profile, max vertical angle)."""
    text = ("IESNA:LM-63-2002\nTILT=NONE\n1 1000 1 3 1 1 -1 0 0 0\n1.0 1.0 0.0\n"
            "0 45 90\n0\n1000 500 0\n")
    profile, max_angle = mind.load_ies(text)
    assert max_angle == pytest.approx(90.0)
    assert np.allclose(profile, [1000.0, 500.0, 0.0]), "the candela values were not read in order"


def test_both_capabilities_are_discoverable(mind):
    """The whole point of the exercise: before this, neither could be found by any phrasing. A capability
    find_capability cannot surface does not exist, however well it works."""
    assert "Antiperiodic" in str(mind.find_capability("mobius strip or circle")[0])
    assert "Antiperiodic" in str(mind.find_capability("does this repeat or invert")[0])
    assert "IES" in str(mind.find_capability("photometric file")[0])
    assert "IES" in str(mind.find_capability("real world light falloff")[0])
