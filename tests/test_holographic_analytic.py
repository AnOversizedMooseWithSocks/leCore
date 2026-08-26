"""Tests for holographic_analytic: sign as rotation, and the clockwise-only cost.

These pin the framework's exact numeric contracts:
  * the Hilbert transform is a real quadrature (H[cos] = sin);
  * the (amplitude, phase) form reconstructs a signed series EXACTLY (lossless);
  * THE REAL-SIGNAL THEOREM: a real scalar signal is already a one-way rotation, so
    clockwise-only costs ~0 on it (a single real channel cannot carry a reversal);
  * THE GROUP-VS-MONOID PRICE: a true complex / I-Q rotation that reverses pays a
    large, well-defined cost when clamped one-way -- the quadrature encoder with both
    channels present.

Determinism: everything is FFT + arithmetic on fixed inputs, no RNG, so every
assertion is a hard numeric contract.
"""

import numpy as np
import pytest

from holographic.sampling_and_signal.holographic_analytic import (
    hilbert, analytic_signal, rotary_encode, enforce_monotone,
    reversal_fraction, monotone_cost, phasor_monotone_cost, _interior,
)


# ---------------------------------------------------------------------------
# The Hilbert transform and the lossless round-trip
# ---------------------------------------------------------------------------

def test_hilbert_of_cosine_is_sine():
    # The canonical quadrature check: H[cos] = sin, so the analytic signal of a
    # cosine is a unit phasor exp(i w t) -- constant amplitude, linear phase.
    t = np.linspace(0, 8 * np.pi, 1024, endpoint=False)
    z = hilbert(np.cos(t))
    assert np.max(np.abs(_interior(np.imag(z)) - _interior(np.sin(t)))) < 1e-2
    assert np.max(np.abs(_interior(np.abs(z)) - 1.0)) < 1e-2


def test_real_part_of_analytic_signal_is_the_input():
    # z = x + i H[x], so Re(z) = x exactly. This is why the amplitude/phase form is
    # a lossless re-coordinatisation, not a model.
    t = np.linspace(0, 6 * np.pi, 777, endpoint=False)
    x = 0.4 * np.cos(t) - 0.9 * np.cos(2 * t + 1.0)
    z = hilbert(x)
    assert np.max(np.abs(np.real(z) - x)) < 1e-10


def test_reversible_reconstruction_is_exact():
    # rotary_encode in reversible mode: A*cos(phi) == x to float precision.
    t = np.linspace(0, 10 * np.pi, 2048, endpoint=False)
    x = np.cos(t) + 0.3 * np.cos(3 * t)
    enc = rotary_encode(x, monotonic=False)
    assert np.max(np.abs(enc["reconstruction"] - x)) < 1e-10
    assert enc["monotonic"] is False


def test_amplitude_tracks_the_envelope():
    # For an AM signal A(t)cos(wt) with A slowly varying, the recovered amplitude
    # should track A(t). Compare on the interior (edge effects at the ends).
    t = np.linspace(0, 1, 4096)
    env = 1.0 + 0.5 * np.cos(2 * np.pi * 2 * t)     # slow, positive envelope
    x = env * np.cos(2 * np.pi * 60 * t)            # fast carrier
    amp = analytic_signal(x)["amplitude"]
    err = np.abs(_interior(amp) - _interior(env))
    assert np.mean(err) < 0.05                       # envelope recovered


# ---------------------------------------------------------------------------
# THE REAL-SIGNAL THEOREM: a real scalar series is already one-way
# ---------------------------------------------------------------------------

def test_real_signal_has_near_zero_reversal_fraction():
    # A real signal's spectrum is Hermitian-symmetric, so its analytic signal
    # rotates one way: the instantaneous frequency is essentially non-negative.
    # This is the surprising, sharp fact -- a real scalar series cannot itself carry
    # a reversal, so clockwise-only is (almost) free on it.
    t = np.linspace(0, 1, 4096)
    multitone = (np.cos(2 * np.pi * 6 * t) + 0.6 * np.cos(2 * np.pi * 11 * t + 0.3)
                 + 0.3 * np.cos(2 * np.pi * 17 * t))
    # A few percent at most, and those are envelope-null glitches, not true
    # reversals -- contrast the complex case which reverses ~50% of the time.
    assert reversal_fraction(multitone) < 0.08


def test_real_monotone_cost_is_small_and_from_nulls_not_reversals():
    # Because a real signal is already one-way, monotone_cost reads a small excess,
    # and its reversal fraction is ~0. Any residual cost is envelope-null glitches,
    # NOT a direction reversal -- pinned to keep the theorem honest.
    t = np.linspace(0, 1, 4096)
    x = np.cos(2 * np.pi * 8 * t) + 0.5 * np.cos(2 * np.pi * 15 * t)
    cost = monotone_cost(x)
    assert cost["reversible_rmse"] < 1e-9        # baseline exact
    assert cost["reversal_fraction"] < 0.02      # already one-way
    assert cost["excess"] < 0.3                  # small, bounded


# ---------------------------------------------------------------------------
# THE GROUP-VS-MONOID PRICE: a true complex rotation that reverses
# ---------------------------------------------------------------------------

def test_complex_phasor_reversal_pays_the_monoid_price():
    # A genuine I/Q rotation: run the phase forward for half the series, backward for
    # the other half. Half the steps reverse; clamping clockwise-only loses them at a
    # large, well-defined cost. This is where the group-vs-monoid price actually
    # lives (unlike the real case, which cannot carry a reversal at all).
    steps = np.concatenate([np.full(1024, 0.2), np.full(1024, -0.2)])
    z = np.exp(1j * np.cumsum(steps))
    cost = phasor_monotone_cost(z)
    assert cost["reversal_fraction"] > 0.3       # about half the steps reverse
    assert cost["excess"] > 0.5                  # the monoid price, loud
    assert cost["max_local_error"] > 0.5


def test_forward_only_complex_phasor_is_free_to_clamp():
    # A complex phasor that only ever advances pays ~nothing to clamp -- it was
    # already clockwise-only. Confirms the cost is specifically about reversal, not
    # about being complex.
    z = np.exp(1j * np.cumsum(np.full(2048, 0.1)))   # monotone forward
    cost = phasor_monotone_cost(z)
    assert cost["reversal_fraction"] < 0.02
    assert cost["excess"] < 1e-9


# ---------------------------------------------------------------------------
# enforce_monotone mechanics + determinism
# ---------------------------------------------------------------------------

def test_enforce_monotone_never_decreases():
    # The clockwise-only clamp: the output phase must be non-decreasing, and it must
    # stall (not back up) exactly where the input decreased.
    phase = np.array([0.0, 0.5, 0.3, 0.9, 0.8, 1.5])  # decreases at idx 2 and 4
    mono = enforce_monotone(phase, direction=+1)
    assert np.all(np.diff(mono) >= -1e-15)
    assert mono[2] == mono[1]                          # stalled at the reversal
    assert mono[4] == mono[3]


def test_enforce_monotone_leaves_a_monotone_input_unchanged():
    phase = np.cumsum(np.full(20, 0.1))                # already increasing
    mono = enforce_monotone(phase)
    assert np.max(np.abs(mono - phase)) < 1e-12


def test_analytic_costs_are_deterministic():
    t = np.linspace(0, 1, 2048)
    x = np.cos(2 * np.pi * 9 * t) + 0.4 * np.cos(2 * np.pi * 14 * t)
    a = monotone_cost(x)
    b = monotone_cost(x)
    for k in a:
        assert a[k] == b[k]


def test_selftest_runs():
    from holographic.sampling_and_signal.holographic_analytic import _selftest
    _selftest()
