"""Tests for the frame-source protocol (leStudio backlog item 12).

Pins the acceptance criterion: a leCore door (map_frames) consumes a HOST-provided FrameSource -- one that never
imported leCore's base class and pretends to wrap a decoder -- without leCore importing any decoder, memoising
per-frame work by seq.
"""
import numpy as np

from holographic.io_and_interop.holographic_framesource import (
    FrameSource, SyntheticFrameSource, is_frame_source, frame_key, map_frames, collect_frames)
from holographic.misc.holographic_unified import UnifiedMind


class _HostVideoSource:
    """A stand-in for the app's real source (which would wrap cv2/ffmpeg). It duck-types the contract only -- it does
    NOT subclass FrameSource -- to prove the engine consumes any host source."""
    seekable = True
    pausable = False

    def __init__(self, n=5):
        self._i = 0
        self._n = n

    def get(self):
        return np.full((8, 8, 3), self._i / float(self._n), dtype=float), self._i

    def advance(self):
        self._i = (self._i + 1) % self._n

    def seek(self, pos):
        self._i = int(round(float(np.clip(pos, 0, 1)) * (self._n - 1)))


def test_contract_shape_and_idempotence():
    src = SyntheticFrameSource(kind="clock", size=(24, 24), frames=6)
    f, s = src.get()
    assert isinstance(f, np.ndarray) and f.shape == (24, 24, 3) and 0.0 <= f.min() <= f.max() <= 1.0
    assert s == 0
    # idempotent: a second pull without advancing is identical, same seq
    f2, s2 = src.get()
    assert s2 == 0 and np.array_equal(f, f2)
    # advancing changes seq and the frame deterministically
    src.advance()
    f3, s3 = src.get()
    assert s3 == 1 and not np.array_equal(f, f3)
    assert np.array_equal(SyntheticFrameSource(kind="clock", size=(24, 24), frames=6)._render(1), f3)


def test_seek_pause_and_flags():
    src = SyntheticFrameSource(frames=10)
    assert src.seekable and src.pausable
    src.seek(1.0)
    assert src.get()[1] == 9
    src.seek(0.0)
    assert src.get()[1] == 0
    src.pause(True)
    before = src.get()[1]
    src.advance()
    assert src.get()[1] == before          # paused holds the frame
    src.pause(False)
    src.advance()
    assert src.get()[1] == before + 1


def test_frame_key_invalidates_iff_seq_changes():
    src = SyntheticFrameSource(frames=6)
    k0 = frame_key(src, "grade")
    assert frame_key(src, "grade") == k0           # same seq -> same key (deterministic, hashlib)
    src.advance()
    assert frame_key(src, "grade") != k0           # new seq -> new key
    # prefix namespaces the key
    assert frame_key(src, "a") != frame_key(src, "b")


def test_map_frames_memoises_by_seq():
    src = SyntheticFrameSource(frames=6)
    calls = [0]

    def fn(fr):
        calls[0] += 1
        return fr.mean()

    cache = {}
    a, sa = map_frames(src, fn, cache)
    b, sb = map_frames(src, fn, cache)             # same frame -> skip recompute
    assert sa == sb == 0 and a == b and calls[0] == 1
    src.advance()
    map_frames(src, fn, cache)                     # new frame -> recompute
    assert calls[0] == 2


def test_host_source_consumed_without_a_decoder():
    """THE ACCEPTANCE: a host source (duck-typed, decoder in the app) is consumed by leCore's door, and leCore
    imports no decoder to do it. Also verified: the door drives a REAL leCore faculty (color_transfer) per frame."""
    m = UnifiedMind(dim=64, seed=0)
    host = _HostVideoSource(n=5)
    assert is_frame_source(host) and not isinstance(host, FrameSource)   # honoured by structure, not inheritance
    ref = np.zeros((6, 6, 3)); ref[..., 0] = 0.8

    cache = {}
    calls = [0]

    def grade(fr):
        calls[0] += 1
        return m.color_transfer(fr, ref, mode="meanstd")   # a real per-frame leCore operation (video colour transfer)

    out0, s0 = m.map_frames(host, grade, cache)
    out0b, _ = m.map_frames(host, grade, cache)
    assert s0 == 0 and calls[0] == 1 and np.array_equal(out0, out0b)     # memoised
    host.advance()
    _out1, s1 = m.map_frames(host, grade, cache)
    assert s1 == 1 and calls[0] == 2

    # a window of frames for a temporal op
    frames = collect_frames(_HostVideoSource(n=5), 3)
    assert [s for _f, s in frames] == [0, 1, 2]

    # the module itself imports no decoder (AST check, robust to the names appearing in strings)
    import ast
    import holographic.io_and_interop.holographic_framesource as mod
    tree = ast.parse(open(mod.__file__).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & {"cv2", "ffmpeg", "av", "imageio", "PIL", "moviepy", "decord"})
