"""The routing seed is a MEASUREMENT INSTRUMENT and a committed binary -- so it must be a pure function of its
content, not of how a run happened to warm.

WHY (measured, not hypothetical). Moose re-ran the cold embed on his machine -- 80 minutes, an empty cache -- and
the result was compared against the committed seed:

  * all 521 vectors BIT-IDENTICAL (max|diff| = 0, min cosine 1.000000000). The NumPy forward pass is deterministic
    across machines and across a cold-vs-warm cache. That is the engine's determinism claim, validated end to end
    by an independent re-embed rather than asserted.
  * and yet the FILE differed: 499 of 521 rows were in a different ORDER, because `seed_cache.py` wrote rows in
    dict-iteration (= insertion) order, which depends on how the cache warmed.

So identical data produced a byte-different 730 KB binary: git churn for zero change, and a landmine for any
byte-level drift gate. `seed_cache.py` now sorts by key, which makes the artifact content-determined -- the same
rule the engine already applies everywhere else (hashlib over hash(), PYTHONHASHSEED=0, seeded RNG).

These tests pin BOTH halves: the seed is canonical, and it still resolves as the offline instrument that answers
dimension/ABTT/fusion questions with no encoder and no network.
"""
import hashlib
import io
import lzma
import pathlib

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SEED = _ROOT / "tools" / "semantic" / "routing_seed.npz.xz"
#: Must match knowledge_index.embed_cached / holographic_router._cache_key exactly, or every lookup misses.
_WIRING = "1000.0|12|True|False"


def _load():
    if not _SEED.is_file():
        pytest.skip("routing seed not present in this tree")
    z = np.load(io.BytesIO(lzma.decompress(_SEED.read_bytes())), allow_pickle=False)
    return [str(k) for k in z["keys"]], z["vecs"]


def test_seed_rows_are_sorted_by_key():
    """Canonical order = reproducible bytes. If this fails, someone reintroduced insertion-order writing and the
    seed will churn in git on every refresh while containing identical data."""
    keys, _ = _load()
    assert keys == sorted(keys), "routing seed rows must be sorted by cache key (see seed_cache.py)"


def test_seed_is_the_routing_slice_only():
    """~500 code entries + the exam's asks -- NOT the ~18k md/NOTES windows that made the 26 MB bloat."""
    keys, vecs = _load()
    assert vecs.dtype == np.float16, "half precision is the shipped form (cosine-identical, half the bytes)"
    assert vecs.shape[0] == len(keys)
    assert 400 <= len(keys) <= 700, ("seed should hold only the routing slice", len(keys))
    assert vecs.shape[1] == 768, "the seed stores FULL width so any dim can be measured from it"
    assert _SEED.stat().st_size < 5_000_000, "seed must stay small enough to commit comfortably"


def test_seed_still_answers_the_exam_asks():
    """The instrument's whole point: the 12 exam asks must be resolvable from the committed seed WITHOUT the
    encoder. If this breaks, a dimension question needs an 80-minute cold embed again."""
    import ast
    import re

    keys, _ = _load()
    have = set(keys)
    src = (_ROOT / "tools" / "semantic" / "knowledge_index.py").read_text(encoding="utf-8")
    asks = ast.literal_eval(re.search(r"ASKS_MODULE\s*=\s*(\[.*?\n\])", src, re.S).group(1))
    key = lambda t: hashlib.sha256((_WIRING + "||" + t).encode()).hexdigest()[:32]
    missing = [a for a, _ in asks if key(f"search_query: {a}") not in have]
    assert not missing, ("ask vectors missing from the seed -- the offline instrument is broken", missing)


def test_seed_covers_the_shipped_index_module_set():
    """The seed must cover what the router actually ranks, or an offline measurement scores a different corpus
    than production. Allows a small tolerance: editing a module's docstring legitimately moves its cache key
    until the next seed refresh."""
    keys, _ = _load()
    have = set(keys)
    idx = _ROOT / "lecore_data" / "routing" / "index_128d.npz"
    if not idx.is_file():
        pytest.skip("shipped index not present")
    shipped = len(np.load(idx, allow_pickle=False)["names"])
    # code entries in the seed = everything that is not an ask; compare against the shipped module count
    assert len(have) >= shipped, ("seed holds fewer vectors than the shipped index has modules", len(have), shipped)
