"""Tests for holographic_provenance.py -- source roles and origin-tagging."""
import numpy as np
from holographic.caching_and_storage.holographic_provenance import source_role, from_external, of_source


def test_same_name_same_role():
    assert np.allclose(source_role("alice", 512), source_role("alice", 512))


def test_different_names_orthogonal():
    a, b = source_role("alice", 1024), source_role("bob", 1024)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert abs(cos) < 0.15


def test_tag_and_recover_with_right_source():
    v = np.random.default_rng(0).standard_normal(1024); v /= np.linalg.norm(v)
    tagged = from_external(v, "nomic", 1024)
    rec = of_source(tagged, "nomic", 1024)
    cos = float(np.dot(rec, v) / (np.linalg.norm(rec) * np.linalg.norm(v)))
    assert cos > 0.6


def test_wrong_source_recovers_noise():
    v = np.random.default_rng(1).standard_normal(1024); v /= np.linalg.norm(v)
    tagged = from_external(v, "nomic", 1024)
    good = of_source(tagged, "nomic", 1024)
    bad = of_source(tagged, "qwen", 1024)
    cg = float(np.dot(good, v) / (np.linalg.norm(good) * np.linalg.norm(v)))
    cb = float(np.dot(bad, v) / (np.linalg.norm(bad) * np.linalg.norm(v)))
    assert cg > 0.6 and cg > 2 * abs(cb)
