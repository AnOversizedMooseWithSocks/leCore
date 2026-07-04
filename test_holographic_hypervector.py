"""Tests for holographic_hypervector -- the first-class Hypervector datatype (D1: thin wrapper, five verbs)."""
import numpy as np
from holographic_hypervector import Hypervector


def _enc():
    from holographic_encoders import ScalarEncoder
    return ScalarEncoder(dim=1024, seed=0)


def test_encode_from_data_carries_metadata():
    enc = _enc()
    a = Hypervector.encode(enc, 0.3, tag="a")
    assert a.dim == 1024 and a.encoder is enc and a.tag == "a"


def test_five_verbs_match_bare_ops():
    from holographic_ai import bind, unbind, bundle, permute
    enc = _enc()
    a = Hypervector.encode(enc, 0.3); b = Hypervector.encode(enc, 0.7)
    assert np.array_equal(a.bind(b).array, bind(a.array, b.array))
    assert np.array_equal(a.bind(b).unbind(b).array, unbind(bind(a.array, b.array), b.array))
    assert np.array_equal(a.bundle(b).array, bundle(np.stack([a.array, b.array])))
    assert np.array_equal(a.permute(3).array, permute(a.array, 3))


def test_verbs_accept_raw_array_too():
    from holographic_ai import bind
    enc = _enc()
    a = Hypervector.encode(enc, 0.3); b = Hypervector.encode(enc, 0.7)
    assert np.array_equal(a.bind(b.array).array, bind(a.array, b.array))       # mixes with bare arrays


def test_raw_array_is_cheap_no_copy():
    a = Hypervector.wrap(np.arange(16.0), tag="x")
    assert a.raw() is a.array                                                  # no copy
    assert np.asarray(a) is a.array                                            # np.asarray -> raw, no copy
    assert len(a) == 16


def test_cleanup_dict_and_array_and_vocab():
    enc = _enc()
    a = Hypervector.encode(enc, 0.3, tag="a"); b = Hypervector.encode(enc, 0.7, tag="b")
    noisy = Hypervector.wrap(a.array + 0.05 * np.random.default_rng(0).standard_normal(1024))
    assert noisy.cleanup({"a": a, "b": b}).tag == "a"                          # dict codebook
    assert noisy.cleanup(np.stack([a.array, b.array])).tag == "atom0"          # array codebook
    from holographic_ai import Vocabulary
    v = Vocabulary(dim=256, seed=0); v.get("x"); v.get("y")
    hx = Hypervector.wrap(v.vectors["x"] + 0.01 * np.random.default_rng(1).standard_normal(256))
    assert hx.cleanup(v).tag == "x"                                            # Vocabulary codebook


def test_cosine_roundtrip_recovers_b():
    from holographic_ai import cosine
    enc = _enc()
    a = Hypervector.encode(enc, 0.3); b = Hypervector.encode(enc, 0.7)
    assert a.bind(b).unbind(a).cosine(b) > cosine(a.array, b.array)


def test_mind_hypervector_faculty():
    from holographic_unified import UnifiedMind
    m = UnifiedMind(dim=512, seed=0)
    a = m.hypervector(0.3, tag="a")
    assert isinstance(a, Hypervector) and a.dim == 512 and a.tag == "a"
    assert a.cleanup({"a": a, "b": m.hypervector(0.7)}).tag == "a"


def test_repr_readable():
    a = Hypervector.wrap(np.zeros(64), tag="hello")
    assert "dim=64" in repr(a) and "hello" in repr(a)
