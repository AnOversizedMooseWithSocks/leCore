"""The shared-build contracts that keep the first teach and the first route of a process cheap (backlog G2).

WHY THIS FILE EXISTS
--------------------
Wave 2 put the typed learning guard in front of every teach and the capability router in front of every serve. Both
were correct and both were slow in the one place CI feels it: the FIRST use in each process. MEASURED on the 2-core box
(one pytest process per file, PYTHONHASHSEED=0): tests/test_external_abstention.py 1.4 s at base 965fdb1 -> 32-38 s
(62 fresh minds, each rebuilding a ~4,000-card catalog to route once), and the guard's base build 2.9 s quiet, 26-39 s
under load, paid by every process's first teach -- over CI's 15 s per-test budget, where a test is SKIPPED, not failed.

The fixes share work across minds and processes WITHOUT changing a single decision. These tests pin that STRUCTURE,
never a wall-clock number (a timing assertion flakes under load; "built once and shared" does not):
  * the guard's base rows are built once per process, shared read-only, and the temp-dir cache reloads them BIT-EXACT;
  * default_catalog() hands out independent clones of a once-per-process template;
  * seed_from_mind's per-class card memo misses when the class attribute changes (no stale card is ever served);
  * a card edited after construction is still re-tokenised by the bake (the skip only skips unchanged cards);
  * hashed_ngram_encode's atom cache is ONE bounded cache per dimension, and a cap far below the n-gram count
    changes no encoding (measured on all of CLINC150: peak RSS 1,513 MB unbounded -> 301 MB, same sha256).
"""
import numpy as np

from holographic.agents_and_reasoning import holographic_learnguard as lg
from holographic.caching_and_storage import holographic_catalog as C


# ---------------------------------------------------------------------------------------------- the typed guard
def _small_examples():
    """A tiny example list in the guard's shape -- enough rows and types to exercise the build, fast to build."""
    ex = []
    for i, t in enumerate(["what is my password", "the admin password", "my api key", "the wifi password"]):
        ex.append((t, "credential", "c%d" % (i % 2)))
    for i, t in enumerate(["sol price", "weather now", "how much is bitcoin", "gas fees now"]):
        ex.append((t, "live_value", "l%d" % (i % 2)))
    for i, t in enumerate(["ada?", "the code?", "eth?"]):
        ex.append((t, "unclear", "u0"))
    for i, t in enumerate(["what is the boiling point of water", "how many legs does a spider have",
                           "how do i reset my password", "what port does ssh use"]):
        ex.append((t, "ordinary", "o%d" % (i % 2)))
    return ex


def test_the_guard_base_is_built_once_per_process_and_shared():
    """Every mind's guard reads THE SAME base rows (they deep-copy them only to add their own learned rows)."""
    b1 = lg.SemanticGuard._base()
    g1, g2 = lg.SemanticGuard(), lg.SemanticGuard()
    assert g1._base() is b1 and g2._base() is b1
    assert lg.SemanticGuard._BASE_HOW in ("loaded", "built", "built (cache off)")
    # a mind's own rows are a COPY: learning in one guard never moves the shared base or another guard
    before = b1[1].A.copy()
    g1.learn("what's the pin for the shed", "credential")
    g1._rows()
    assert np.array_equal(b1[1].A, before) and len(g2._rows()) == len(b1[1])


def test_the_guard_cache_reloads_bit_exact(tmp_path, monkeypatch):
    monkeypatch.delenv("LECORE_GUARD_CACHE", raising=False)
    ex = _small_examples()
    enc1, st1, how1 = lg.cached_typed_store(ex, cache_dir=str(tmp_path))
    enc2, st2, how2 = lg.cached_typed_store(ex, cache_dir=str(tmp_path))
    assert (how1, how2) == ("built", "loaded")
    assert st1.labels == st2.labels and st1.key_of == st2.key_of and st1.count == st2.count
    assert st1.A.tobytes() == st2.A.tobytes() and st1.P.tobytes() == st2.P.tobytes()   # bit for bit, not allclose
    assert enc1.idf == enc2.idf and enc1.unseen == enc2.unseen
    q = "whats the pw for the router"
    assert enc1(q).tobytes() == enc2(q).tobytes()
    assert lg.type_scores(st1, enc1(q)) == lg.type_scores(st2, enc2(q))
    # and both equal the plain build (the path bench_learnguard's cross-validation uses)
    enc0, st0 = lg.build_typed_store(ex)
    assert st0.A.tobytes() == st1.A.tobytes() and st0.P.tobytes() == st1.P.tobytes()


def test_a_different_example_list_never_reads_another_lists_cache(tmp_path):
    ex = _small_examples()
    lg.cached_typed_store(ex, cache_dir=str(tmp_path))
    ex2 = ex + [("what is the melting point of iron", "ordinary", "o0")]
    _enc, st, how = lg.cached_typed_store(ex2, cache_dir=str(tmp_path))
    assert how == "built"                                   # a new key, so a fresh build -- never the old rows
    assert st.A.tobytes() == lg.build_typed_store(ex2)[1].A.tobytes()


def test_a_corrupt_cache_file_is_rebuilt_not_trusted(tmp_path):
    ex = _small_examples()
    lg.cached_typed_store(ex, cache_dir=str(tmp_path))
    (f,) = list(tmp_path.glob("lecore_guard_*.npz"))
    f.write_bytes(b"not an npz")
    _enc, st, how = lg.cached_typed_store(ex, cache_dir=str(tmp_path))
    assert how == "built" and st.A.tobytes() == lg.build_typed_store(ex)[1].A.tobytes()


def test_the_cache_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv("LECORE_GUARD_CACHE", "0")
    _enc, _st, how = lg.cached_typed_store(_small_examples(), cache_dir=str(tmp_path))
    assert how == "built (cache off)" and not list(tmp_path.iterdir())


# ---------------------------------------------------------------------------------------------- the catalog
def _card(c):
    return (c.name, c.does, c.example, c.native, tuple(c.aliases), c.semantic, tuple(c.consumes),
            tuple(c.produces), c.module, c.method, c.polymorphic, sorted(c._nw), sorted(c._hay), c._al)


def test_default_catalog_hands_out_independent_clones_of_one_template():
    a, b = C.default_catalog(), C.default_catalog()
    fresh = C._build_default_catalog()                      # what every call used to build from scratch
    assert list(a._by_name) == list(fresh._by_name) and [_card(x) for x in a.all()] == [_card(x) for x in fresh.all()]
    assert a._version == fresh._version
    name = next(iter(a._by_name))
    assert a.get(name) is not b.get(name)                   # separate card objects
    a.get(name).does = "edited in one catalog only"
    a.get(name).method = "nope"
    a.register_capability("ZZ only in a", "a card only catalog a has")
    a.unregister(list(a._by_name)[1])
    c = C.default_catalog()
    assert c.get(name).does == fresh.get(name).does and c.get(name).method == fresh.get(name).method
    assert c.get("ZZ only in a") is None and len(c) == len(fresh) and len(b) == len(fresh)


def test_seed_from_mind_memo_misses_when_the_faculty_changes():
    """The per-class card memo is keyed on the class attribute OBJECT: a redefined faculty gets a new card."""
    import lecore

    class _Mind(lecore.UnifiedMind):
        def g2_probe_faculty(self, x, y=2):
            """First doc line of the original probe."""
            return x

    m1 = _Mind(dim=64, seed=0)
    c1 = C.seed_from_mind(C.default_catalog(), m1)
    assert c1.get("g2_probe_faculty").does == "First doc line of the original probe."
    c1b = C.seed_from_mind(C.default_catalog(), _Mind(dim=64, seed=0))     # the memo path
    assert _card(c1b.get("g2_probe_faculty")) == _card(c1.get("g2_probe_faculty"))

    def g2_probe_faculty(self, z):
        """A different doc line after a redefinition."""
        return z
    _Mind.g2_probe_faculty = g2_probe_faculty
    c2 = C.seed_from_mind(C.default_catalog(), _Mind(dim=64, seed=0))
    assert c2.get("g2_probe_faculty").does == "A different doc line after a redefinition."
    assert c2.get("g2_probe_faculty").example == "mind.g2_probe_faculty(z)"
    # an INSTANCE attribute shadowing the class never takes the memo path
    m3 = _Mind(dim=64, seed=0)
    m3.g2_probe_faculty = lambda w, v=1: w
    c3 = C.seed_from_mind(C.default_catalog(), m3)
    assert c3.get("g2_probe_faculty").example == "mind.g2_probe_faculty(w, v=1)"


def test_the_bake_still_retokenises_a_card_edited_after_construction():
    cat = C.Catalog()
    cap = cat.register_capability("Widget smoother", "smooth widgets gently", aliases=("widget polish",))
    cap.does = "sharpen gizmos quickly"                     # edited AFTER its token sets were made
    cat._ensure_fc_baked()
    assert "gizmos" in cap._hay and "widgets" not in cap._hay
    assert cat.find_capability("sharpen gizmos")[0].name == "Widget smoother"


def test_the_token_memo_hands_out_fresh_lists():
    t1 = C._tokens("smooth a bumpy mesh")
    t1.append("poison")
    assert C._tokens("smooth a bumpy mesh") == ["smooth", "bumpy", "mesh"]
    s1 = C._alias_tokens(("denoise mesh",))
    s1.add("poison")
    assert "poison" not in C._alias_tokens(("denoise mesh",))


# ---------------------------------------------------------------------------------------------- the n-gram atoms
def test_the_ngram_atom_cache_is_bounded_shared_and_changes_no_encoding(monkeypatch):
    """hashed_ngram_encode's atoms come from ONE bounded cache per dimension. A cap far below the number of distinct
    n-grams must change nothing but how often an atom is regenerated (the atoms are a pure function of the n-gram)."""
    import hashlib
    from holographic.agents_and_reasoning import holographic_systemone as so
    dim = 96                                                 # a dimension no other test uses: its cache is ours
    monkeypatch.setattr(so, "NGRAM_CACHE_MB", 64 * 8 * dim / float(1 << 20))    # exactly the 64-atom floor
    texts = ["the quick brown fox jumps over the lazy dog", "pack my box with five dozen liquor jugs",
             "sphinx of black quartz judge my vow", "the quick brown fox again"]

    def reference(text):                                     # the definition, with no cache at all
        t = " " + text.lower() + " "
        v = np.zeros(dim)
        for n in range(3, 6):
            for i in range(len(t) - n + 1):
                g = t[i:i + n]
                seed = int.from_bytes(hashlib.sha256(g.encode()).digest()[:8], "big") % (2 ** 32)
                v += np.random.default_rng(seed).standard_normal(dim)
        return v

    e1, e2 = so.hashed_ngram_encode(dim=dim), so.hashed_ngram_encode(dim=dim)
    assert e1.cache is e2.cache and e1.cap == 64            # one shared cache per dimension, at the 64-atom floor
    for t in texts + texts[::-1]:
        assert e1(t).tobytes() == reference(t).tobytes()     # bit for bit, through evictions
        assert len(e1.cache) <= 64
    assert e1.stats["misses"] > 64                           # the cap really bit: atoms were regenerated
    a = next(iter(e1.cache.values()))
    assert not a.flags.writeable                             # a shared atom cannot be edited in place by a caller
    # cap=N: a PRIVATE LRU (the rank door's _bounded_ngram semantics) -- same vectors, its own memory
    p = so.hashed_ngram_encode(dim=dim, cap=8)
    assert p.cache is not e1.cache and p.cap == 8
    for t in texts:
        assert p(t).tobytes() == reference(t).tobytes() and len(p.cache) <= 8


def test_one_builder_at_a_time_and_a_dead_builders_lock_never_hangs(tmp_path, monkeypatch):
    """A second process finding another's FRESH lock waits for that build instead of competing for the CPU (measured:
    8 cold guide-check processes on 2 cores, 26.6 s); a lock nobody releases costs at most GUARD_CACHE_WAIT_S, and a
    STALE lock (a builder that died) is ignored at once. Either way the caller gets the same rows."""
    import os
    import time
    monkeypatch.delenv("LECORE_GUARD_CACHE", raising=False)
    ex = _small_examples()
    ref = lg.build_typed_store(ex)[1].A.tobytes()
    key = lg._guard_cache_key(ex, lg.GUARD_EPOCHS, lg.GUARD_DIM)
    lock = tmp_path / ("lecore_guard_%s.npz.lock" % key)
    lock.write_text("")                                     # a fresh lock whose builder never finishes
    monkeypatch.setattr(lg, "GUARD_CACHE_WAIT_S", 0.3)
    t0 = time.time()
    _e, st, how = lg.cached_typed_store(ex, cache_dir=str(tmp_path))
    assert how == "built" and st.A.tobytes() == ref and time.time() - t0 >= 0.3
    (tmp_path / ("lecore_guard_%s.npz" % key)).unlink()
    old = time.time() - 3600
    os.utime(str(lock), (old, old))                         # a STALE lock: its builder died an hour ago
    monkeypatch.setattr(lg, "GUARD_CACHE_WAIT_S", 60.0)
    t0 = time.time()
    _e, st, how = lg.cached_typed_store(ex, cache_dir=str(tmp_path))
    assert how == "built" and st.A.tobytes() == ref and time.time() - t0 < 30
    _e, st, how = lg.cached_typed_store(ex, cache_dir=str(tmp_path))
    assert how == "loaded" and st.A.tobytes() == ref        # the file exists now: no lock is even looked at
