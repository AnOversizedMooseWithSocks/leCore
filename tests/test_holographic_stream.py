"""F8 -- the brain/muscle format contract, and the norm the guarantee lives in.

The backlog: *"leCore ships baked fields and descriptors -- the descriptor bake, TT cores in rank order =
progressive LOD streams, SOG splats -- the front end consumes our formats rather than recomputing."*

That is a claim about TT cores, and it is measured here. It is TRUE, with a precision the sentence does not carry:
**the monotonicity is in the RMS (Frobenius) norm, not in max-abs.** TT-SVD truncation is Frobenius-optimal, so
adding a rank always lowers the L2 error and can still make one voxel worse. On white noise the max-abs error rises
at 4 of 15 levels while the RMS falls at every one.

*The low-rank field the format was designed for is monotone in BOTH norms. Testing only that case would have
shipped the wrong guarantee.*
"""

import numpy as np
import pytest

from holographic.caching_and_storage.holographic_tucker import tt_bytes, tt_reconstruct
from holographic.io_and_interop.holographic_stream import (
    stream_decode, stream_encode, stream_prefix, stream_report, truncate_tt)


def _low_rank(n=16, modes=(1.0, 0.5, 0.25, 0.12, 0.06, 0.03)):
    """A genuinely intermediate-rank field: several separable modes with decaying weights.

    NOT a single outer product -- that is TT-rank 1, would compress 300x, and would make every ladder look perfect.
    NOT white noise either. The interesting case is the one in between, and it is the only one that tests a ladder.
    """
    g = np.linspace(0, 1, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    return sum(w * np.sin((k + 1) * np.pi * X) * np.cos((k + 1) * np.pi * Y) * np.exp(-(k + 1) * Z)
               for k, w in enumerate(modes))


def _noise(n=16, seed=0):
    return np.random.default_rng(seed).normal(size=(n, n, n))


def test_selftest_runs():
    from holographic.io_and_interop import holographic_stream as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# the fixtures have the ranks they claim
# ---------------------------------------------------------------------------------------------------------

def test_the_fixtures_are_genuinely_low_and_high_rank():
    lo = stream_encode(_low_rank())["descriptor"]
    hi = stream_encode(_noise())["descriptor"]
    assert max(lo["full_ranks"]) <= 8            # a handful of modes
    assert max(hi["full_ranks"]) >= 15           # near-full
    assert 3 <= lo["n_levels"] <= 10


# ---------------------------------------------------------------------------------------------------------
# the contract: every prefix is a field
# ---------------------------------------------------------------------------------------------------------

def test_every_prefix_decodes_to_the_full_shape():
    F = _low_rank()
    p = stream_encode(F)
    for i in range(p["descriptor"]["n_levels"]):
        assert stream_decode(p, i).shape == F.shape          # a coarse level is smoothed, not small


def test_the_tail_reconstructs_the_field():
    F = _low_rank()
    p = stream_encode(F, tol=1e-12)
    assert np.abs(stream_decode(p, -1) - F).max() < 1e-9
    assert p["descriptor"]["rel_rms"][-1] < 1e-9


def test_bytes_rise_with_rank_and_the_descriptor_tells_the_truth():
    F = _low_rank()
    p = stream_encode(F)
    d = p["descriptor"]
    assert d["bytes"] == sorted(d["bytes"])
    for i, lvl in enumerate(p["levels"]):
        assert d["bytes"][i] == tt_bytes(lvl)                # published bytes ARE the bytes


def test_the_muscle_chooses_from_the_descriptor_alone():
    F = _low_rank()
    p = stream_encode(F)
    d = p["descriptor"]
    assert stream_prefix(p, d["bytes"][0] - 1) is None       # cannot afford even level 0
    assert stream_prefix(p, d["bytes"][0]) == 0
    assert stream_prefix(p, d["bytes"][2]) == 2
    assert stream_prefix(p, 10 ** 9) == d["n_levels"] - 1


# ---------------------------------------------------------------------------------------------------------
# THE GUARANTEE, and the norm it lives in
# ---------------------------------------------------------------------------------------------------------

def test_the_rms_error_is_monotone_on_both_a_low_rank_field_and_on_noise():
    for F in (_low_rank(), _noise()):
        rep = stream_report(F, stream_encode(F))
        assert rep["monotone_rms"] is True


def test_kept_negative_the_max_abs_error_is_NOT_monotone_on_noise():
    # THE FINDING. TT truncation is Frobenius-optimal, not uniform-optimal: a new rank always lowers the L2 error
    # and can raise one voxel's error. If this test ever passes, the counterexample has stopped being one.
    F = _noise()
    p = stream_encode(F)
    rep = stream_report(F, p)
    assert rep["monotone_rms"] is True
    assert rep["monotone_max"] is False

    d = p["descriptor"]
    rises = [i for i in range(d["n_levels"] - 1) if d["rel_max"][i + 1] > d["rel_max"][i]]
    assert len(rises) >= 1


def test_the_low_rank_field_is_monotone_in_both_which_is_how_the_wrong_guarantee_gets_shipped():
    # Testing only the case the format was designed for would have hidden the distinction entirely.
    rep = stream_report(_low_rank(), stream_encode(_low_rank()))
    assert rep["monotone_rms"] is True and rep["monotone_max"] is True


def test_the_descriptor_names_the_norm_its_guarantee_is_in():
    d = stream_encode(_low_rank())["descriptor"]
    assert d["monotone_in"] == "rel_rms"
    assert "rel_rms" in d and "rel_max" in d                 # both published, one guaranteed


# ---------------------------------------------------------------------------------------------------------
# where the ladder pays, and where it does not
# ---------------------------------------------------------------------------------------------------------

def test_the_ladder_pays_on_a_low_rank_field():
    F = _low_rank()
    rep = stream_report(F, stream_encode(F))
    assert rep["best_ratio_at_10pct"] is not None
    assert rep["best_ratio_at_10pct"] > 5.0                  # measured 20.4x at 20^3


def test_kept_negative_white_noise_gets_no_ladder():
    # The ladder is a property of the FIELD's rank, not of the format. A format cannot make a field compressible.
    F = _noise()
    p = stream_encode(F)
    rep = stream_report(F, p)
    assert rep["best_ratio_at_10pct"] is None                # never reaches 10% below full rank
    assert p["descriptor"]["rel_rms"][0] > 0.9               # the first level carries almost nothing
    assert rep["levels"][-1]["vs_dense"] < 3.0               # ... and even the tail barely compresses


def test_kept_negative_a_coarse_level_is_smoothed_not_small():
    # Rank is not resolution. A front end wanting fewer SAMPLES needs a mip chain: a different object.
    F = _low_rank()
    p = stream_encode(F)
    coarse = stream_decode(p, 0)
    assert coarse.shape == F.shape
    assert np.abs(np.diff(coarse, axis=0)).max() < np.abs(np.diff(F, axis=0)).max()


# ---------------------------------------------------------------------------------------------------------
# truncation mechanics
# ---------------------------------------------------------------------------------------------------------

def test_truncating_to_full_rank_is_the_identity():
    F = _low_rank()
    p = stream_encode(F, tol=1e-12)
    full = p["levels"][-1]
    same = truncate_tt(full, [np.shape(c)[2] for c in full["cores"][:-1]])
    assert np.abs(tt_reconstruct(same) - tt_reconstruct(full)).max() < 1e-12


def test_truncating_clamps_rather_than_overreaching():
    F = _low_rank()
    full = stream_encode(F, tol=1e-12)["levels"][-1]
    over = truncate_tt(full, [999] * (len(full["cores"]) - 1))
    assert np.abs(tt_reconstruct(over) - tt_reconstruct(full)).max() < 1e-12
    under = truncate_tt(full, [0] * (len(full["cores"]) - 1))     # 0 clamps up to 1, not to an empty core
    assert tt_reconstruct(under).shape == F.shape


def test_a_bad_rank_list_raises():
    full = stream_encode(_low_rank())["levels"][-1]
    with pytest.raises(ValueError):
        truncate_tt(full, [1])                               # a 3-core TT needs 2 bond ranks


def test_truncation_does_not_mutate_its_input():
    full = stream_encode(_low_rank())["levels"][-1]
    before = [np.array(c, copy=True) for c in full["cores"]]
    truncate_tt(full, [1, 1])
    for a, b in zip(before, full["cores"]):
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------------------------------------

def test_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    F = _low_rank(n=12, modes=(1.0, 0.5, 0.25, 0.12))
    p = m.stream_encode(F)
    rep = m.stream_report(F, p)
    assert rep["monotone_rms"] is True
    assert m.stream_decode(p, 0).shape == F.shape
    assert m.stream_prefix(p, p["descriptor"]["bytes"][1]) == 1

    for phrase in ("progressive level of detail stream", "truncate to a byte budget",
                   "format contract for the front end"):
        assert "Progressive LOD" in str(m.find_capability(phrase)[:3]), phrase
