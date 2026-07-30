"""holographic_assetfetch.py -- fetch an external asset (HDRI, model, texture) ONCE, then never again.

THE DESIGN QUESTION THIS ANSWERS. The engine's constitution says deterministic; the network is not. Every
other integration in this space (the reference Blender one included) just downloads on demand and hopes --
same query, different day, different asset, and a scene that rendered yesterday renders differently today.
The resolution here is the same one the repo already uses for randomness: DETERMINISM COMES FROM PINNING.
A seeded RNG is replayable because the seed is recorded; a fetched asset is replayable because its
CONTENT HASH is recorded. Concretely:

  * The cache is CONTENT-ADDRESSED: a fetched file lives at <cache>/<sha256><ext>. Two URLs serving the
    same bytes share one entry; a URL that changes its bytes gets a NEW entry rather than silently
    replacing the old one under the same name.
  * `sha256=` pins a fetch. A pinned fetch that is already cached is served FROM DISK WITHOUT TOUCHING THE
    NETWORK -- so a scene recipe of (url, sha256) pairs replays bit-identically offline, forever, which is
    the property a downloaded-on-demand asset can never have.
  * A pinned fetch whose downloaded bytes do NOT match the pin is DELETED and raises. A silently-different
    asset is the supply-chain version of a flipped decision, and this repo does not ship those. The error
    names both hashes so the caller can decide whether the upstream legitimately changed.
  * An UNPINNED fetch computes and RETURNS the hash, so the first exploratory fetch hands you exactly the
    pin to record. The workflow is: browse once, pin, replay forever.

WHAT THIS DELIBERATELY IS NOT:
  * Not imported by any core path. The engine renders, simulates, and tests with zero network access; this
    module is reached only when a caller explicitly asks to fetch. `import holographic_assetfetch` itself
    performs no I/O.
  * Not a scraper or a search client. It takes a URL. Site-specific search APIs (PolyHaven's, Sketchfab's)
    churn, need keys, and belong in userland glue -- the stable contract is "give me bytes at a URL,
    verified"; everything above that is fashion.
  * Not a package manager: no resolution, no versions, no metadata store. asset_library (hash / track /
    relink, already shipped) is the downstream bookkeeping half; this is only the missing network half.
"""
import hashlib
import os
import pathlib
import urllib.request

DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".lecore_assets")
MAX_BYTES = 512 * 1024 * 1024          # a 512 MB ceiling: an HDRI or a mesh, not a mistake


def fetch_asset(url, cache_dir=None, sha256=None, timeout=30.0, max_bytes=MAX_BYTES):
    """Fetch `url` into the content-addressed cache and return {path, sha256, bytes, cached}.

    `sha256=` pins the fetch (hex string). Pinned + already cached = served from disk, NO network I/O --
    the deterministic-replay path. Pinned + mismatch after download = the file is removed and ValueError
    raised naming both hashes. Unpinned = the computed hash is returned; record it to pin the recipe.

    Only http(s) URLs are accepted: file:// would silently alias the local filesystem into a function whose
    name promises the network, and stranger schemes (ftp, data:) are attack surface with no user."""
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ValueError("fetch_asset takes an http(s) URL; got %r" % (url,))
    cache = pathlib.Path(cache_dir or DEFAULT_CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    ext = pathlib.Path(url.split("?")[0]).suffix.lower() or ".bin"

    if sha256 is not None:
        sha256 = sha256.lower().strip()
        pinned = cache / (sha256 + ext)
        if pinned.exists():
            # THE REPLAY PATH: the whole point of pinning. No network, no freshness check -- content
            # addressing means the bytes cannot be stale, because different bytes are a different address.
            return {"path": str(pinned), "sha256": sha256, "bytes": pinned.stat().st_size, "cached": True}

    req = urllib.request.Request(url, headers={"User-Agent": "leCore-assetfetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        declared = resp.headers.get("Content-Length")
        if declared and int(declared) > max_bytes:
            raise ValueError("refusing %s: Content-Length %s exceeds the %d-byte ceiling"
                             % (url, declared, max_bytes))
        data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("refusing %s: body exceeds the %d-byte ceiling (lied about or missing "
                         "Content-Length)" % (url, max_bytes))

    got = hashlib.sha256(data).hexdigest()
    if sha256 is not None and got != sha256:
        raise ValueError("HASH MISMATCH for %s: pinned %s, downloaded %s -- the upstream content changed "
                         "(or the transfer was tampered with). Nothing was kept. If the change is "
                         "legitimate, re-pin to the new hash deliberately." % (url, sha256, got))

    out = cache / (got + ext)
    if not out.exists():                                       # same bytes from another URL: already have it
        tmp = out.with_suffix(out.suffix + ".part")
        tmp.write_bytes(data)                                  # write-then-rename: no torn files on a crash
        tmp.rename(out)
    return {"path": str(out), "sha256": got, "bytes": len(data), "cached": False}


def _selftest():
    """Round trip against a loopback server -- a REAL socket, because the failure modes worth pinning
    (mismatch handling, the no-network replay path) live at the boundary, not in the hashing."""
    import http.server
    import shutil
    import tempfile
    import threading

    root = tempfile.mkdtemp()
    payload = b"#?RADIANCE\nFAKE HDR PAYLOAD\n" * 40
    (pathlib.Path(root) / "sky.hdr").write_bytes(payload)
    want = hashlib.sha256(payload).hexdigest()

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=root, **kw)

        def log_message(self, *a):                              # a selftest that chats is a selftest nobody reads
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d/sky.hdr" % srv.server_address[1]
    cache = tempfile.mkdtemp()

    try:
        # 1. unpinned fetch RETURNS the pin
        r1 = fetch_asset(url, cache_dir=cache)
        assert r1["sha256"] == want and r1["cached"] is False
        assert pathlib.Path(r1["path"]).read_bytes() == payload
        assert pathlib.Path(r1["path"]).name.startswith(want), "the cache must be content-addressed"

        # 2. THE REPLAY PATH: pinned + cached = served with the network GONE. This is the determinism
        #    story in one assertion -- kill the server, and the pinned fetch must still succeed.
        srv.shutdown()
        r2 = fetch_asset(url, cache_dir=cache, sha256=want)
        assert r2["cached"] is True and r2["path"] == r1["path"], \
            "a pinned, cached fetch must not need the network"

        # 3. a WRONG pin on a cold cache must refuse and keep nothing
        cold = tempfile.mkdtemp()
        srv2 = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
        threading.Thread(target=srv2.serve_forever, daemon=True).start()
        url2 = "http://127.0.0.1:%d/sky.hdr" % srv2.server_address[1]
        try:
            fetch_asset(url2, cache_dir=cold, sha256="ab" * 32)
            raise AssertionError("a hash mismatch must raise")
        except ValueError as e:
            assert "MISMATCH" in str(e) and want in str(e), "the error must name BOTH hashes"
        assert not any(pathlib.Path(cold).iterdir()), "a mismatched download must not be kept"
        srv2.shutdown()

        # 4. scheme discipline
        try:
            fetch_asset("file:///etc/passwd")
            raise AssertionError("file:// must be refused")
        except ValueError:
            pass
        print("assetfetch selftest OK -- content-addressed, pinned replay works with the server DOWN, "
              "mismatch refuses and keeps nothing, file:// refused")
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    _selftest()
