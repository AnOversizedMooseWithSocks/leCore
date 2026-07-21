"""holographic_framesource.py -- the FRAME-SOURCE protocol for temporal media (leStudio backlog item 12).

WHY THIS EXISTS (and where the line is drawn)
---------------------------------------------
leStudio built capture threads, sequence-numbered frames, play/pause/seek, and signature-based cache invalidation
entirely app-side. That is a lot of machinery, but the REUSABLE kernel is tiny and has nothing to do with decoding:
a consumer only ever needs to pull "the current frame and a number that changes when the frame changes". If leCore
wants first-class video -- temporal NCA, optical-flow doors, video colour transfer -- THIS is the seam.

THE CONSTITUTIONAL LINE: the engine must NOT own decoding. cv2 / ffmpeg / yt-dlp are host concerns and stay host-side
(and are banned from the core anyway). leCore owns only the CONTRACT and the consumers of it. A leCore door takes a
host-provided FrameSource and calls `.get()`; it never imports a decoder. This module imports numpy + hashlib + stdlib
only, and a test confirms it stays that way.

THE CONTRACT (duck-typed -- a host source need not subclass anything)
--------------------------------------------------------------------
A FrameSource is any object with:
  * get() -> (frame, seq)
        `frame` is an ndarray (H,W) or (H,W,C) float, or None if no frame is available yet (connecting / error).
        `seq` is a non-negative int that changes IFF the frame content changes -- it is the SIGNATURE a consumer
        memoises on, so per-frame work runs once per distinct frame and is skipped while the frame is held. `get()`
        is IDEMPOTENT: it reads the current state, it does not advance it (advancing is the source's job -- a capture
        thread for a live stream, an explicit advance()/seek() for a deterministic one).
  * seekable : bool     -- if True, seek(pos in [0,1]) is supported (files); a live stream is not seekable.
  * pausable : bool     -- if True, pause(flag) holds/resumes advancing.
Optional, gated by the flags: seek(pos), pause(flag). A consumer checks the flag before calling.

WHY seq AND NOT a hash of the pixels: hashing every frame to detect change is O(pixels) per pull; the source already
knows when it advanced, so it just counts. seq is the cheap, honest invalidation signal (leStudio's `media:src:seq`).
"""

import hashlib

import numpy as np


class FrameSource:
    """Base class DOCUMENTING the frame-source contract (see the module docstring). A host source may subclass this
    or merely duck-type it -- the engine only calls get() and reads seekable/pausable. Subclasses override get()
    (required) and, if capable, set seekable/pausable and override seek()/pause()."""

    seekable = False
    pausable = False

    def get(self):
        """Return (frame, seq). Override this. Idempotent: reads current state, does not advance it."""
        raise NotImplementedError("a FrameSource must implement get() -> (frame, seq)")

    def seek(self, pos):
        """Move to normalised position pos in [0,1] (only if seekable)."""
        raise NotImplementedError("this FrameSource is not seekable")

    def pause(self, flag=True):
        """Hold (flag=True) or resume (flag=False) advancing (only if pausable)."""
        raise NotImplementedError("this FrameSource is not pausable")


def is_frame_source(obj):
    """Duck-check: True if obj honours the FrameSource contract well enough to consume -- a callable get() that
    returns a 2-tuple. Deliberately structural (not isinstance), so a HOST source that never imported leCore's base
    class still qualifies."""
    if not hasattr(obj, "get") or not callable(obj.get):
        return False
    try:
        out = obj.get()
    except Exception:
        return False
    return isinstance(out, tuple) and len(out) == 2


def frame_key(source, prefix=""):
    """A deterministic cache key for the source's CURRENT frame -- `prefix` + the current seq, hashed with hashlib
    (never Python's salted hash(); the engine's determinism rule). This is the signature-based invalidation seam:
    key it once per pull, and a consumer's per-frame result is valid exactly while the key is unchanged."""
    _frame, seq = source.get()
    raw = ("%s|%d" % (str(prefix), int(seq))).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def map_frames(source, fn, cache=None):
    """The consuming DOOR (and the acceptance case): pull the current frame from a host-provided FrameSource, apply
    `fn(frame)`, and MEMOISE the result by seq so it recomputes only when the frame changes. Returns (out, seq);
    out is None when the source has no frame yet.

    Two cache shapes, both caller-kept between calls (additive -- backlog A2):
      - a plain DICT (default): a SINGLE-SLOT memo (keys 'seq'/'out') -- it holds only the latest frame, so it is
        already O(1) in RAM (the A2 "unbounded {seq:out} dict" worry was a misread of this code). Perfect for
        streaming, where seq only moves forward and you never look back.
      - a ColdStore (holographic_coldstore): a BOUNDED multi-frame LRU keyed by seq -- for SCRUB / SEEK / LOOP work
        (a leStudio timeline dragged back and forth), where several recent frames' outputs should stay warm without
        letting RAM grow without bound. Reuses the engine's ColdStore rather than a hand-rolled growing dict.

    This imports NO decoder: it works on ANY object honouring get() -> (frame, seq), which is the whole point --
    cv2/ffmpeg live in the host's FrameSource, the temporal DOOR lives here."""
    frame, seq = source.get()
    if frame is None:
        return None, seq
    # ColdStore path: a bounded, multi-frame LRU (scrub/seek/loop). Detected by its put/get, and it is NOT a dict.
    if cache is not None and hasattr(cache, "put") and hasattr(cache, "get") and not isinstance(cache, dict):
        try:
            return cache.get(seq), seq                 # hit: a recently-processed frame (e.g. scrubbed back to)
        except KeyError:
            out = fn(np.asarray(frame)); cache.put(seq, out); return out, seq
    # default single-slot dict memo (streaming) -- unchanged, byte-for-byte behaviour as before.
    if cache is not None and cache.get("seq") == seq and "out" in cache:
        return cache["out"], seq                       # same frame -> skip the per-frame work
    out = fn(np.asarray(frame))
    if cache is not None:
        cache["seq"] = seq
        cache["out"] = out
    return out, seq


def collect_frames(source, n, advance=True):
    """Pull `n` frames into a list of (frame, seq) -- the batch/temporal entry point (a window for temporal NCA,
    optical flow, or video colour transfer). When advance=True and the source exposes advance(), the source is
    stepped between pulls (deterministic sources); a live source ignores it and returns whatever is current."""
    out = []
    for _ in range(int(n)):
        out.append(source.get())
        if advance and hasattr(source, "advance"):
            source.advance()
    return out


class SyntheticFrameSource(FrameSource):
    """A pure-NumPy, DECODER-FREE reference FrameSource -- a deterministic synthetic clip, so leCore has a self-
    contained example and a testable one (a host's real cv2/ffmpeg source honours the same contract). seekable and
    pausable. Advancing is explicit (advance()/seek()), NOT a wall clock, so a frame is a deterministic function of
    its index -- reproducible run to run (the engine's determinism rule).

    kinds: 'clock' (a sweeping hand), 'gradient' (a scrolling ramp), 'bars' (moving colour bars). `seq` == the frame
    index, so it changes exactly when the frame does."""

    seekable = True
    pausable = True

    def __init__(self, kind="clock", size=(64, 64), frames=120, seed=0):
        self.kind = kind
        self.h, self.w = int(size[0]), int(size[1])
        self.frames = int(frames)
        self.seed = int(seed)
        self.i = 0                                       # current frame index == seq
        self.paused = False

    def advance(self, k=1):
        """Step forward k frames (wrapping), unless paused. Returns the new index."""
        if not self.paused:
            self.i = (self.i + int(k)) % max(self.frames, 1)
        return self.i

    def seek(self, pos):
        self.i = int(round(float(np.clip(pos, 0.0, 1.0)) * (self.frames - 1)))
        return self.i

    def pause(self, flag=True):
        self.paused = bool(flag)

    def _render(self, i):
        """Deterministic frame i -> (H,W,3) float in [0,1]. Pure arithmetic on a lattice; no RNG in the pixels so it
        is bit-reproducible, with `seed` perturbing the phase."""
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(float)
        u = xx / max(self.w - 1, 1)
        v = yy / max(self.h - 1, 1)
        t = (i + self.seed) / max(self.frames, 1)
        if self.kind == "gradient":
            g = np.clip(u + t, 0.0, 1.0)                 # a ramp that scrolls with the frame
            return np.stack([g, np.clip(v + t, 0, 1), np.full_like(g, t)], axis=-1)
        if self.kind == "bars":
            phase = np.floor((u + t) * 6.0) % 3.0        # three colour bars sliding across
            return np.stack([(phase == 0).astype(float), (phase == 1).astype(float),
                             (phase == 2).astype(float)], axis=-1)
        # 'clock': a sweeping hand -- angle of each pixel vs the current hand angle
        ang = np.arctan2(v - 0.5, u - 0.5)
        hand = 2.0 * np.pi * t
        d = np.abs(((ang - hand + np.pi) % (2.0 * np.pi)) - np.pi)
        m = np.clip(1.0 - d * 3.0, 0.0, 1.0)
        return np.stack([m, m * 0.6, 1.0 - m], axis=-1)

    def get(self):
        return self._render(self.i), self.i


def _selftest():
    """The contract's real behaviour, as asserts (not a smoke test): get() is idempotent and returns (ndarray, int);
    advancing changes seq and the frame; seek lands on an index; frame_key changes IFF seq changes; map_frames
    MEMOISES by seq (fn runs once per distinct frame); a FOREIGN host source (duck-typed, not subclassing) is
    consumed; and the module imports NO decoder (the constitutional line)."""
    src = SyntheticFrameSource(kind="clock", size=(32, 32), frames=10, seed=0)

    # 1. get() -> (ndarray (H,W,3) float, int seq); idempotent (two pulls without advancing are identical + same seq)
    f0, s0 = src.get()
    assert isinstance(f0, np.ndarray) and f0.shape == (32, 32, 3) and 0.0 <= f0.min() and f0.max() <= 1.0
    f0b, s0b = src.get()
    assert s0 == s0b == 0 and np.array_equal(f0, f0b), "get() must be idempotent (reads, does not advance)"

    # 2. advance changes seq AND the frame; determinism: frame is a function of the index alone
    src.advance()
    f1, s1 = src.get()
    assert s1 == 1 and not np.array_equal(f0, f1), "advance must change seq and the frame"
    assert np.array_equal(SyntheticFrameSource(size=(32, 32), frames=10)._render(1), f1)   # deterministic per index

    # 3. seek lands on an index; capability flags are honest
    assert src.seekable and src.pausable
    src.seek(1.0); assert src.get()[1] == 9                            # last frame of a 10-frame clip
    src.pause(True); before = src.get()[1]; src.advance(); assert src.get()[1] == before   # paused holds

    # 4. frame_key changes IFF seq changes (signature-based invalidation), and is deterministic
    src2 = SyntheticFrameSource(frames=10)
    k_a = frame_key(src2, "img")
    assert frame_key(src2, "img") == k_a                              # same seq -> same key
    src2.advance()
    assert frame_key(src2, "img") != k_a                             # new seq -> new key

    # 5. map_frames MEMOISES by seq: fn runs once per distinct frame, skipped while the frame is held
    src3 = SyntheticFrameSource(frames=10)
    calls = [0]
    def _fn(fr):
        calls[0] += 1
        return fr.mean()
    cache = {}
    out_a, sa = map_frames(src3, _fn, cache); out_b, sb = map_frames(src3, _fn, cache)   # same frame twice
    assert sa == sb == 0 and out_a == out_b and calls[0] == 1, "map_frames must skip recompute at the same seq"
    src3.advance(); map_frames(src3, _fn, cache)
    assert calls[0] == 2, "a new frame must recompute"

    # 5b. A2: a ColdStore cache is a BOUNDED MULTI-FRAME LRU (scrub/seek/loop) -- retains every processed frame's
    # output (cools, never evicts) while holding at most keep_warm live, so RAM is bounded AND scrub-back never
    # recomputes. Reuses the engine's ColdStore rather than a hand-rolled growing dict. The dict path is unchanged.
    from holographic.caching_and_storage.holographic_coldstore import ColdStore
    src3b = SyntheticFrameSource(kind="gradient", size=(16, 16)); cs = ColdStore(keep_warm=3); calls2 = [0]
    def _fn2(fr):
        calls2[0] += 1; return float(np.asarray(fr).mean())
    seen = {}
    for i in range(6):
        src3b.i = i; o, s = map_frames(src3b, _fn2, cache=cs); seen[s] = o
    assert calls2[0] == 6                                              # six distinct frames, one compute each
    assert sum(1 for _, c in cs._items.items() if not c.is_cold()) <= 3   # RAM bounded: <= keep_warm live
    for j in (0, 2, 5):                                               # scrub back to ANY earlier frame -> HIT
        src3b.i = j; o, s = map_frames(src3b, _fn2, cache=cs); assert o == seen[j]
    assert calls2[0] == 6, "ColdStore path must retain all frames -- scrub-back must not recompute"

    # 6. a FOREIGN host source -- duck-typed, does NOT subclass FrameSource -- is consumed by the same door.
    class HostSource:                                                 # pretend this wraps cv2 in the app; leCore never sees cv2
        def __init__(self): self._i = 0
        def get(self): return np.full((8, 8, 3), self._i / 10.0), self._i
        def advance(self): self._i += 1
    hs = HostSource()
    assert is_frame_source(hs)
    out, seq = map_frames(hs, lambda fr: float(fr.mean()))
    assert seq == 0 and abs(out - 0.0) < 1e-9
    frs = collect_frames(HostSource(), 3)
    assert [s for _f, s in frs] == [0, 1, 2]                          # collected a 3-frame window

    # 7. THE CONSTITUTIONAL LINE: this module imports no decoder. Checked via the AST's actual import statements
    #    (not a substring scan, which would false-positive on the names written in this very comment/test).
    import ast
    import holographic.io_and_interop.holographic_framesource as _mod
    tree = ast.parse(open(_mod.__file__).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    decoders = {"cv2", "ffmpeg", "av", "imageio", "PIL", "moviepy", "decord"}
    assert not (imported & decoders), "the frame-source protocol imported a decoder: %s" % (imported & decoders)

    print("OK: holographic_framesource self-test passed (get() idempotent -> (ndarray, seq); advance/seek/pause honest; "
          "frame_key invalidates IFF seq changes; map_frames memoises by seq; a FOREIGN duck-typed host source is "
          "consumed; and the module imports NO decoder -- the engine owns the contract, the host owns decoding)")


if __name__ == "__main__":
    _selftest()
