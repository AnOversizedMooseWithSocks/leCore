"""Sweep 3 item 7: the storage spine -- uri keys + content dedup + fountain erasure robustness."""
from holographic.caching_and_storage.holographic_storage import StorageSpine


def test_roundtrip_and_dedup():
    s = StorageSpine(block_size=16)
    p = b"freelance dev since 1997, readable code" * 3
    s.put(("db", "u", "alice"), p)
    s.put(("db", "u", "bob"), b"other" * 5)
    assert s.get(("db", "u", "alice")) == p
    before = s.distinct_payloads()
    s.put(("cache", "alice2"), p)                     # identical content, new key
    assert s.distinct_payloads() == before            # deduped
    assert s.get(("cache", "alice2")) == p


def test_erasure_recovery_and_honest_none():
    s = StorageSpine(block_size=16)
    p = b"payload that must survive droplet loss" * 6
    s.put("k", p)
    assert s.get("k", loss=0.3) == p                  # recovered under 30% loss
    lost = s.get("k", loss=0.97, overhead=2.0)
    assert lost is None or lost == p                  # honest None, never wrong bytes
    assert s.get("missing") is None


def test_deterministic():
    s = StorageSpine(block_size=16); s.put("k", b"abc" * 20)
    assert s.get("k", loss=0.2, seed=3) == s.get("k", loss=0.2, seed=3)
