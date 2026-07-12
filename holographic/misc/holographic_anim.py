"""Animation over time: a keyframe Timeline and a tiered delta FrameCache (ANIM-2).

WHY THIS MODULE EXISTS
----------------------
A modeling / simulation tool plays geometry back over time and lets the user scrub the timeline. Two pieces
were missing: a way to KEY values over time and sample a deformed state at any t, and a way to CACHE evaluated
frames so playback / scrubbing does not re-simulate. Both are here, vectorised and built on the engine's own
delta philosophy.

ON "L1/L2/L3/L4 CACHING" -- HONESTLY
  Python and NumPy cannot control the CPU's actual L1-L4 caches; that is the hardware's job and no library-level
  code touches it. What DOES map to the user's mental model -- and is genuinely useful -- is a TIERED frame
  cache: the most-recently-touched frames kept FULL in RAM for instant scrubbing (the "hot" tier), older frames
  kept as compact DELTAS versus a base frame (the "warm" tier, O(change) memory -- the engine's delta protocol
  applied to time), and anything beyond that recomputed from the base + the animation function (the "cold"
  tier). So "L1..L4" here is an honest ANALOGY for hot/warm/cold frame storage, not literal cache-line control.

VSA-NATIVE WHERE TRUE
  The delta cache IS the engine's O(change) patch idea on the time axis: a deforming mesh where a wave passes
  through changes only the wave-front vertices per frame, so the per-frame delta is small and playback memory is
  O(total change), not O(frames * full_state). Measured below.
"""

import numpy as np


class Timeline:
    """Keyframed channels with interpolated sampling. `key(channel, t, value, interp='linear')` adds a keyframe;
    `sample(channel, t)` reads the interpolated value at time t (t may be a scalar OR an array of times --
    vectorised), clamped at the ends. `value` can be a scalar, a vector (e.g. blendshape weights), or any array; the
    interpolation broadcasts. Rotations should be keyed as quaternions and slerped by the caller (holographic_ai.slerp).

    EASING (E1): each key carries the interpolation used to reach it FROM the previous key -- 'linear' (default,
    the old behaviour, byte-identical), 'step' (hold the previous value until the next key, no blend), 'smooth'
    (smoothstep ease-in-out), 'ease_in' (slow start), or 'ease_out' (slow stop). The mode only reshapes the [0,1]
    fraction before the same lerp, so a channel with all-linear keys is bit-identical to before -- easing is
    additive, not a behaviour change."""

    # the easing shapes: each maps a linear fraction f in [0,1] to an eased fraction. 'linear' is the identity.
    _EASE = {
        "linear":  lambda f: f,
        "smooth":  lambda f: f * f * (3.0 - 2.0 * f),          # smoothstep (ease in AND out)
        "ease_in": lambda f: f * f,                            # quadratic slow start
        "ease_out": lambda f: 1.0 - (1.0 - f) ** 2,            # quadratic slow stop
        # 'step' is handled specially in sample (it holds v0, i.e. fraction forced to 0).
    }

    def __init__(self):
        self.channels = {}                                    # name -> (times sorted, values stacked, interp modes)

    def key(self, channel, t, value, interp="linear"):
        if interp not in ("linear", "smooth", "ease_in", "ease_out", "step"):
            raise ValueError("interp must be linear/smooth/ease_in/ease_out/step; got %r" % interp)
        ts, vs, modes = self.channels.get(channel, ([], [], []))
        ts = list(ts) + [float(t)]
        vs = list(vs) + [np.asarray(value, float)]
        modes = list(modes) + [interp]
        order = np.argsort(ts)
        self.channels[channel] = (np.asarray(ts)[order], [vs[i] for i in order], [modes[i] for i in order])
        return self

    def sample(self, channel, t):
        """Interpolate the channel at time(s) t (scalar or array), applying each segment's easing (the interp mode
        of the key being approached). Clamps before the first / after the last key. Vectorised over t."""
        ts, vs, modes = self.channels[channel]
        V = np.stack(vs)                                      # (K, ...) keyframe values
        tq = np.atleast_1d(np.asarray(t, float))
        if len(ts) == 1:
            out = np.broadcast_to(V[0], (len(tq),) + V[0].shape).copy()
            return out[0] if np.ndim(t) == 0 else out
        idx = np.clip(np.searchsorted(ts, tq) - 1, 0, len(ts) - 2)
        t0 = ts[idx]; t1 = ts[idx + 1]
        frac = np.clip((tq - t0) / np.where(t1 - t0 > 1e-12, t1 - t0, 1.0), 0.0, 1.0)
        # apply the easing of the key being APPROACHED (segment idx -> idx+1 uses key idx+1's interp mode).
        seg_mode = np.array(modes, dtype=object)[idx + 1]
        eased = frac.copy()
        for mode in set(seg_mode):
            sel = seg_mode == mode
            if mode == "step":
                # step holds the PREVIOUS value across the segment, but once time reaches the next key exactly
                # (frac == 1) we have arrived, so deliver that key's value -- otherwise the final key never shows.
                eased[sel] = np.where(frac[sel] >= 1.0, 1.0, 0.0)
            else:
                eased[sel] = Timeline._EASE[mode](frac[sel])
        v0 = V[idx]; v1 = V[idx + 1]
        extra = (1,) * (V.ndim - 1)
        f = eased.reshape((len(tq),) + extra)
        out = (1 - f) * v0 + f * v1
        return out[0] if np.ndim(t) == 0 else out


class FrameCache:
    """A tiered frame cache for animation playback. Stores each frame's state (an (N, D) array -- mesh vertices,
    a particle cloud, or a flattened volume) as a DELTA versus a base frame, so memory is O(total change), and
    keeps the `hot` most-recently-accessed frames reconstructed in full for instant scrubbing.

      put(frame, state)  -- record a frame (stored as a sparse delta vs base; rows that differ by > tol).
      get(frame)         -- reconstruct base + delta (served from the hot tier if recently touched).
      memory_bytes(), full_bytes() -- the actual vs naive (store-every-frame-full) cost, i.e. the saving.

    The hot tier is the "L1/L2" analogy (full, fast); the delta store is "L3" (compact); recompute-from-base is
    "L4". A frame whose whole state changed is just stored full as its delta -- no worse than naive."""

    def __init__(self, base, hot=8, tol=1e-9):
        self.base = np.asarray(base, float).copy()
        self.hot = int(hot)
        self.tol = float(tol)
        self.deltas = {}                                      # frame -> (changed_row_idx, changed_rows)
        self._hotframes = {}                                  # frame -> full state (LRU)
        self._order = []                                      # recency of hot frames

    def put(self, frame, state):
        state = np.asarray(state, float)
        diff = np.abs(state - self.base).max(axis=1) > self.tol   # which rows changed vs base
        idx = np.where(diff)[0]
        self.deltas[int(frame)] = (idx, state[idx].copy())
        self._touch(int(frame), state)
        return self

    def get(self, frame):
        frame = int(frame)
        if frame in self._hotframes:
            return self._hotframes[frame]
        idx, rows = self.deltas[frame]
        state = self.base.copy()
        state[idx] = rows                                     # reconstruct: base + the sparse delta
        self._touch(frame, state)
        return state

    def _touch(self, frame, state):
        self._hotframes[frame] = state
        if frame in self._order:
            self._order.remove(frame)
        self._order.append(frame)
        while len(self._order) > self.hot:                    # evict the coldest from the hot tier
            old = self._order.pop(0)
            self._hotframes.pop(old, None)

    def memory_bytes(self):
        """Actual bytes held: the base + every frame's sparse delta (row indices + changed rows)."""
        b = self.base.nbytes
        for idx, rows in self.deltas.values():
            b += idx.nbytes + rows.nbytes
        return b

    def full_bytes(self):
        """What storing every frame in full would cost (the naive baseline)."""
        return self.base.nbytes * (len(self.deltas) + 1)


def bake_deformation(base, n_frames, frame_fn):
    """Evaluate an animation into a FrameCache: for frame f in [0, n_frames), call frame_fn(base, f) -> (N,D)
    deformed state, and cache it as a delta. Returns the FrameCache. `frame_fn` is any vectorised deformer
    (bend/twist/lattice/blendshapes with a time-varying parameter). The whole bake never loops in Python over
    vertices -- only over frames."""
    cache = FrameCache(base)
    for f in range(n_frames):
        cache.put(f, frame_fn(base, f))
    return cache


class Transport:
    """A PLAYHEAD over a frame function -- the start / pause / step / seek / scrub / rewind / fast-forward the Timeline
    (keyframes) and FrameCache (per-frame storage) did not provide. It holds a CURRENT FRAME and a play state, and
    computes each frame on demand from `frame_fn(frame) -> state`, caching results in a FrameCache so revisiting a
    frame (rewind, scrub back, replay) is O(1) and never recomputes. This is what makes an animation SCRUBBABLE.

    `frame_fn(frame)` returns the state at an integer frame (a deformed mesh's vertices, a sim field, a parameter dict
    packed as an array, ...). `n_frames` bounds the range; `fps` is metadata for time<->frame. Determinism: because
    frame_fn is a pure function of the frame index, seeking to frame f gives the SAME state whether you got there by
    playing, rewinding, or jumping -- no hidden accumulator drift (the classic scrub bug). A stateful sim that CANNOT
    be evaluated at an arbitrary frame should be baked with bake_deformation first, then driven through a Transport
    over the cache."""

    def __init__(self, frame_fn, n_frames, fps=24.0, base=None):
        self._fn = frame_fn
        self.n_frames = int(n_frames)
        self.fps = float(fps)
        self.frame = 0
        self.playing = False
        self.speed = 1.0                                       # frames advanced per tick (negative = reverse)
        self._acc = 0.0                                        # fractional-frame accumulator for non-integer speeds
        # cache the frame states for O(1) revisits. The FrameCache stores (N,D) states as sparse deltas off a base
        # (ideal for a deforming mesh -- most vertices are unchanged frame to frame); for any other shape (a param
        # vector, a scalar) fall back to a plain dict, which is still O(1) and correct.
        self._base = base if base is not None else np.asarray(self._fn(0))
        self._twod = (self._base.ndim == 2)
        self._cache = FrameCache(self._base) if self._twod else {}

    def at(self, frame):
        """The state at integer `frame` (clamped to [0, n_frames-1]). Cached: the first call computes via frame_fn,
        every later call for the same frame is O(1). This is the scrub primitive."""
        f = int(max(0, min(self.n_frames - 1, frame)))
        if self._twod:
            try:
                return self._cache.get(f)                      # O(1) if already computed
            except KeyError:
                got = np.asarray(self._fn(f))                  # first visit: compute and cache
                self._cache.put(f, got)
                return got
        if f not in self._cache:
            self._cache[f] = np.asarray(self._fn(f))
        return self._cache[f]

    def seek(self, frame):
        """Jump the playhead to `frame` and return its state. The seek/scrub operation -- same state no matter how
        you arrived (deterministic)."""
        self.frame = int(max(0, min(self.n_frames - 1, frame)))
        return self.at(self.frame)

    def seek_time(self, seconds):
        """Seek by TIME (seconds) instead of frame index, via fps."""
        return self.seek(int(round(seconds * self.fps)))

    def play(self, speed=1.0):
        """Start playback at `speed` frames/tick (1.0 = forward, -1.0 = reverse/rewind, 2.0 = fast-forward, 0.5 =
        slow-mo). Call tick() to advance."""
        self.playing = True
        self.speed = float(speed)
        return self

    def pause(self):
        """Pause -- the playhead holds its current frame."""
        self.playing = False
        return self

    def rewind(self, speed=1.0):
        """Play in REVERSE at `speed` (a convenience for play(-speed))."""
        return self.play(-abs(speed))

    def stop(self):
        """Pause and rewind the playhead to frame 0."""
        self.playing = False
        self.frame = 0
        self._acc = 0.0
        return self

    def tick(self, n=1):
        """Advance playback by `n` ticks at the current speed (does nothing if paused). Wraps at the ends (loops).
        Returns the state at the new frame. The play-loop step: call once per rendered frame."""
        if self.playing:
            self._acc += self.speed * n
            step = int(self._acc)                              # integer frames to move; keep the remainder
            self._acc -= step
            self.frame = (self.frame + step) % self.n_frames   # loop at both ends (reverse wraps to the tail)
        return self.at(self.frame)

    def step(self, n=1):
        """Advance the playhead by `n` frames REGARDLESS of play state (frame-by-frame stepping, e.g. the '.'/',' keys
        in a DCC). Negative steps back. Returns the new frame's state."""
        self.frame = int(max(0, min(self.n_frames - 1, self.frame + n)))
        return self.at(self.frame)

    @property
    def time(self):
        """The current playhead position in seconds (frame / fps)."""
        return self.frame / self.fps


def _selftest():
    from holographic.mesh_and_geometry.holographic_deform import twist
    # Timeline: lerp between two vector keys
    tl = Timeline().key("w", 0.0, [0.0, 0.0]).key("w", 1.0, [1.0, 2.0])
    assert np.allclose(tl.sample("w", 0.5), [0.5, 1.0])
    assert np.allclose(tl.sample("w", 0.0), [0.0, 0.0]) and np.allclose(tl.sample("w", 2.0), [1.0, 2.0])
    # E1 EASING: the interp mode reshapes the fraction. At the midpoint, smooth == linear (0.5), but ease_in < 0.5
    # (slow start) and step holds the previous value; linear is unchanged (backward compatible).
    te = Timeline().key("x", 0.0, 0.0).key("x", 1.0, 1.0, interp="ease_in")
    assert abs(float(te.sample("x", 0.5)) - 0.25) < 1e-9              # ease_in: f^2 at 0.5 -> 0.25
    ts_ = Timeline().key("x", 0.0, 0.0).key("x", 1.0, 1.0, interp="step")
    assert float(ts_.sample("x", 0.5)) == 0.0 and float(ts_.sample("x", 1.0)) == 1.0  # step holds then jumps
    tsm = Timeline().key("x", 0.0, 0.0).key("x", 1.0, 1.0, interp="smooth")
    assert abs(float(tsm.sample("x", 0.5)) - 0.5) < 1e-9             # smoothstep symmetric at the midpoint
    lin = Timeline().key("x", 0.0, 0.0).key("x", 1.0, 1.0)           # default linear unchanged
    assert abs(float(lin.sample("x", 0.5)) - 0.5) < 1e-9
    # FrameCache: a traveling local change -> small deltas, reconstruct exactly
    base = np.zeros((100, 3))
    def fr(b, f):
        s = b.copy(); s[f:f + 5, 2] = 1.0; return s          # a 5-row bump moving along the array
    cache = bake_deformation(base, 20, fr)
    for f in range(20):
        assert np.allclose(cache.get(f), fr(base, f))         # exact reconstruction
    saving = cache.full_bytes() / cache.memory_bytes()
    assert saving > 1.5, saving                               # delta cache is smaller than store-every-frame

    # TRANSPORT: play / pause / step / seek / rewind / fast-forward over a frame function, deterministically.
    calls = [0]
    def frame_state(f):
        calls[0] += 1
        return np.array([float(f), float(f) * 2.0])           # a simple ramp so we can check the exact frame
    tr = Transport(frame_state, n_frames=10, fps=24.0)
    assert np.allclose(tr.seek(5), [5.0, 10.0])               # seek jumps to a frame
    assert tr.frame == 5
    # PLAY forward: tick advances the playhead
    tr.stop().play(1.0)
    tr.tick(); tr.tick()
    assert tr.frame == 2 and np.allclose(tr.at(2), [2.0, 4.0])
    # PAUSE holds the frame
    tr.pause(); held = tr.frame
    tr.tick(); tr.tick()
    assert tr.frame == held, "pause holds the playhead"
    # STEP works regardless of play state (frame-by-frame)
    tr.step(3); assert tr.frame == held + 3
    tr.step(-1); assert tr.frame == held + 2
    # REWIND (reverse play) and FAST-FORWARD (speed 2)
    tr.seek(8); tr.play(-1.0); tr.tick(); assert tr.frame == 7        # reverse
    tr.seek(0); tr.play(2.0); tr.tick(); assert tr.frame == 2         # fast-forward 2x
    # DETERMINISM + CACHE: revisiting a frame gives the same state and does NOT recompute (O(1) scrub)
    before = calls[0]
    a = tr.seek(2); b = tr.seek(2)
    assert np.array_equal(a, b) and calls[0] == before, "revisiting a cached frame is O(1), no recompute"

    print(f"anim selftest ok: timeline lerp exact; frame cache reconstructs exactly, "
          f"{saving:.1f}x smaller than full-frame storage; Transport plays/pauses/steps/seeks/rewinds/fast-forwards "
          f"with O(1) cached scrub and deterministic frame states")


if __name__ == "__main__":
    _selftest()
