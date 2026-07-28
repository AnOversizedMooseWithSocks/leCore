"""Regression traps for the external asset fetcher -- the design decision made testable.

The claim under guard: the network meets the determinism rule BY PINNING, the same way randomness meets it
by seeding. Every test runs against a loopback server; none touches the real internet, because CI must not
depend on anyone else's uptime to prove OUR contract.
"""
import hashlib
import http.server
import pathlib
import threading

import pytest

from holographic.io_and_interop.holographic_assetfetch import fetch_asset


@pytest.fixture()
def served(tmp_path):
    root = tmp_path / "www"
    root.mkdir()
    payload = b"FAKE GLB BYTES " * 100
    (root / "chair.glb").write_bytes(payload)

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield ("http://127.0.0.1:%d/chair.glb" % srv.server_address[1], payload,
           hashlib.sha256(payload).hexdigest(), srv)
    srv.shutdown()


def test_first_fetch_returns_the_pin(served, tmp_path):
    """The workflow's first half: browse once, get the hash to record. If the hash were not returned the
    caller would have to fetch twice to pin, and nobody would."""
    url, payload, digest, _ = served
    r = fetch_asset(url, cache_dir=str(tmp_path / "cache"))
    assert r["sha256"] == digest and r["cached"] is False
    assert pathlib.Path(r["path"]).read_bytes() == payload
    assert pathlib.Path(r["path"]).name.startswith(digest), "the cache must be content-addressed"


def test_pinned_and_cached_needs_no_network(served, tmp_path):
    """THE DETERMINISM CLAIM ITSELF: kill the server, and a pinned fetch of a cached asset must still
    succeed. This is what makes a (url, sha256) recipe replayable offline forever -- and it is the exact
    property that separates this design from every download-on-demand integration."""
    url, _, digest, srv = served
    cache = str(tmp_path / "cache")
    fetch_asset(url, cache_dir=cache)
    srv.shutdown()
    r = fetch_asset(url, cache_dir=cache, sha256=digest)
    assert r["cached"] is True


def test_mismatch_refuses_and_keeps_nothing(served, tmp_path):
    """A silently-different asset is the supply-chain version of a flipped decision. The error must name
    BOTH hashes (so the caller can re-pin deliberately), and the cache must stay empty (so a poisoned
    download cannot be picked up later by an unpinned call)."""
    url, _, digest, _ = served
    cold = tmp_path / "cold"
    with pytest.raises(ValueError, match="MISMATCH"):
        fetch_asset(url, cache_dir=str(cold), sha256="ab" * 32)
    assert not any(cold.iterdir())


def test_only_http_schemes(tmp_path):
    """file:// would alias the local filesystem into a function whose name promises the network."""
    with pytest.raises(ValueError, match="http"):
        fetch_asset("file:///etc/passwd", cache_dir=str(tmp_path))


def test_the_fetched_asset_feeds_the_pipeline(tmp_path):
    """Cross-faculty: a fetched .hdr must flow straight into load_hdr -- the reason the fetcher exists is
    that load_hdr had nothing to load. Served bytes are a REAL Radiance file so the whole chain is honest."""
    import numpy as np
    import lecore

    h, w = 4, 8
    rgbe = np.zeros((h, w, 4), np.uint8)
    rgbe[..., :3] = 128
    rgbe[..., 3] = 129                                          # exponent for values around 1.0
    payload = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y %d +X %d\n" % (h, w) + rgbe.tobytes()
    root = tmp_path / "www"
    root.mkdir()
    (root / "sky.hdr").write_bytes(payload)

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        m = lecore.UnifiedMind(dim=128, seed=0)
        r = m.fetch_asset("http://127.0.0.1:%d/sky.hdr" % srv.server_address[1],
                          cache_dir=str(tmp_path / "cache"))
        env = m.load_hdr(r["path"])
        assert env.shape == (h, w, 3) and env.dtype.name == "float32"
        assert "Fetch an external asset" in str(m.find_capability("download an hdri")[:3])
    finally:
        srv.shutdown()
