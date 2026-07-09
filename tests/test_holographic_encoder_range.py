"""A6/E1 -- the scalar encoder's working range, and the silent failure when it's wrong.

The encoder's [lo, hi] is what makes nearby numbers similar. Values outside it don't raise, don't clamp, and don't
warn -- they simply stop being distinguishable from one another. Measured on data spanning [0, 100] with the
UniversalEncoder's default (-4, 4) range: normalized decode error 0.5422, against a RANDOM-VECTOR baseline of
0.5436. The code carried no information at all. Ranged to the data: 0.0014.

These tests pin the fix: an auto-ranging constructor, and a one-shot warning so it can never fail quietly again.
"""
import warnings

import numpy as np

from holographic.io_and_interop.holographic_encoders import ScalarEncoder


def _norm_decode_err(enc, vals):
    return float(np.mean([abs(enc.decode(enc.encode(v)) - v) for v in vals])) / (vals.max() - vals.min())


def _vals():
    return np.random.default_rng(0).uniform(0, 100, 120)


def test_out_of_range_encoding_is_no_better_than_random():
    """The kept negative, pinned: this is WHY the warning and for_values() exist."""
    vals = _vals()
    rng = np.random.default_rng(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bad = ScalarEncoder(1024, lo=-4.0, hi=4.0, seed=0)
        err = _norm_decode_err(bad, vals)
        chance = float(np.mean([abs(bad.decode(rng.standard_normal(1024)) - v) for v in vals])) / (
            vals.max() - vals.min())
    assert err > 0.4 and abs(err - chance) < 0.15, (err, chance)   # at chance: the code holds no information


def test_for_values_ranges_to_the_data_and_is_far_better():
    vals = _vals()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bad = ScalarEncoder(1024, lo=-4.0, hi=4.0, seed=0)
        bad_err = _norm_decode_err(bad, vals)
    good = ScalarEncoder.for_values(vals, 1024, seed=0)
    good_err = _norm_decode_err(good, vals)
    assert good.lo < vals.min() and good.hi > vals.max()          # a margin, so endpoints aren't on the boundary
    assert good_err < 0.01
    assert bad_err / good_err > 50.0                              # measured ~385x


def test_encode_warns_once_when_out_of_range():
    enc = ScalarEncoder(256, lo=0.0, hi=1.0, seed=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        enc.encode(5.0)                                           # outside [0, 1]
        enc.encode(9.0)                                           # still outside -- but we warn only once
    assert len(w) == 1 and "outside the encoder's range" in str(w[0].message)


def test_in_range_encoding_never_warns():
    enc = ScalarEncoder(256, lo=0.0, hi=1.0, seed=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for v in (0.0, 0.25, 1.0):
            enc.encode(v)
    assert not w


def test_for_values_handles_a_constant_column():
    enc = ScalarEncoder.for_values([3.0, 3.0, 3.0], 128, seed=0)
    assert enc.lo < 3.0 < enc.hi                                  # a degenerate range gets a usable window
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        enc.encode(3.0)
    assert not w


def test_organizer_range_is_configurable_and_default_unchanged():
    from holographic.scene_and_pipeline.holographic_organizer import SelfOrganizingMind
    assert (SelfOrganizingMind(dim=128, seed=0).encoder._scalar.lo,
            SelfOrganizingMind(dim=128, seed=0).encoder._scalar.hi) == (-4.0, 4.0)   # backward compatible
    m = SelfOrganizingMind(dim=128, seed=0, number_range=(0.0, 100.0))
    assert (m.encoder._scalar.lo, m.encoder._scalar.hi) == (0.0, 100.0)
