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
    """Keyframed channels with interpolated sampling. `key(channel, t, value)` adds a keyframe; `sample(channel,
    t)` reads the linearly-interpolated value at time t (t may be a scalar OR an array of times -- vectorised),
    clamped at the ends. `value` can be a scalar, a vector (e.g. blendshape weights), or any array; the lerp
    broadcasts. Rotations should be keyed as quaternions and slerped by the caller (holographic_ai.slerp)."""

    def __init__(self):
        self.channels = {}                                    # name -> (times sorted, values stacked)

    def key(self, channel, t, value):
        ts, vs = self.channels.get(channel, ([], []))
        ts = list(ts) + [float(t)]
        vs = list(vs) + [np.asarray(value, float)]
        order = np.argsort(ts)
        self.channels[channel] = (np.asarray(ts)[order], [vs[i] for i in order])
        return self

    def sample(self, channel, t):
        """Linearly interpolate the channel at time(s) t (scalar or array). Clamps before the first / after the
        last key. Vectorised over t for scalar/vector channels."""
        ts, vs = self.channels[channel]
        V = np.stack(vs)                                      # (K, ...) keyframe values
        tq = np.atleast_1d(np.asarray(t, float))
        idx = np.clip(np.searchsorted(ts, tq) - 1, 0, len(ts) - 2) if len(ts) > 1 else np.zeros(len(tq), int)
        if len(ts) == 1:
            out = np.broadcast_to(V[0], (len(tq),) + V[0].shape).copy()
            return out[0] if np.ndim(t) == 0 else out
        t0 = ts[idx]; t1 = ts[idx + 1]
        frac = np.clip((tq - t0) / np.where(t1 - t0 > 1e-12, t1 - t0, 1.0), 0.0, 1.0)
        v0 = V[idx]; v1 = V[idx + 1]
        extra = (1,) * (V.ndim - 1)
        f = frac.reshape((len(tq),) + extra)
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


def _selftest():
    from holographic.mesh_and_geometry.holographic_deform import twist
    # Timeline: lerp between two vector keys
    tl = Timeline().key("w", 0.0, [0.0, 0.0]).key("w", 1.0, [1.0, 2.0])
    assert np.allclose(tl.sample("w", 0.5), [0.5, 1.0])
    assert np.allclose(tl.sample("w", 0.0), [0.0, 0.0]) and np.allclose(tl.sample("w", 2.0), [1.0, 2.0])
    # FrameCache: a traveling local change -> small deltas, reconstruct exactly
    base = np.zeros((100, 3))
    def fr(b, f):
        s = b.copy(); s[f:f + 5, 2] = 1.0; return s          # a 5-row bump moving along the array
    cache = bake_deformation(base, 20, fr)
    for f in range(20):
        assert np.allclose(cache.get(f), fr(base, f))         # exact reconstruction
    saving = cache.full_bytes() / cache.memory_bytes()
    assert saving > 1.5, saving                               # delta cache is smaller than store-every-frame
    print(f"anim selftest ok: timeline lerp exact; frame cache reconstructs exactly, "
          f"{saving:.1f}x smaller than full-frame storage")


if __name__ == "__main__":
    _selftest()
