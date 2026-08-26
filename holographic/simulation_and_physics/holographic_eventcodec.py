"""The PHYSICS EVENT CODEC (Box3D lesson B8, backlog item X7).

Catto makes contact begin/end and sleep notifications first-class OUTPUTS. Read through this engine's primitives,
that becomes a compression statement: **between events, physics is a deterministic function of the state.** So a
recorded trace does not need to store the states at all. It needs the base state and the EVENTS -- the impulses,
the contacts, the moments where the deterministic flow was interrupted -- and a replayer regenerates the rest.
This is the seed+delta model (composability L6, the NMS world) applied to dynamics, and it is exactly what
deterministic-lockstep netcode does.

THE BAR, from the backlog: beat delta-compression of raw state. MEASURED on a 600-frame, 16-body bouncing-box
trace (663 contact events), against the strongest honest baseline in the original space:

    raw float64                  460,800 bytes
    zlib(raw)                    308,090
    zlib(frame deltas)            87,057    <- the baseline the bar names
    EVENT CODEC                    6,360    <- 13.7x, and LOSSLESS: replay is bit-identical

WHAT THE WIN ACTUALLY IS, because the backlog attributes it to the wrong mechanism. X7 says "impulses/contacts as
recognized edits over the edit codebook; the edit codec (6.3x) compresses multiplayer physics streams." Measured,
the codebook is not where the compression lives:

    event codec, impulses stored as LITERAL float64      6,360 bytes   (lossless)
    ... with a quantized impulse codebook, q=1e-3          4,010 bytes   (1.59x more, LOSSY)
    ... q=1e-2                                             3,516         (1.81x, LOSSY)
    ... q=1e-1                                             3,099         (2.05x, LOSSY)

**Event SPARSITY is the 13.7x. The codebook is a further 2x at best, and it is lossy.** 663 events replace
600x16 = 9,600 state rows; that ratio is the compression, and it needs no codebook at all.

KEPT NEGATIVE, and it kills the codebook for replay: a quantized impulse is NOT a small error in the output,
because an event changes WHICH events happen next. Measured trajectory error at the final frame, in a box of
half-extent 2.0:

    q=1e-3 -> 5.71e-02      q=1e-2 -> 4.36e-01      q=1e-1 -> 4.47e+00

At q=1e-1 the reconstruction has left the box. The codebook's 2x is paid for with a trajectory that is not the one
you recorded. Store impulses literally, or store the residual (which erases the gain). This is W2's Lyapunov
negative wearing a codec's clothes: a contact sequence is a chaotic map, and lossy compression of its inputs does
not degrade gracefully.

SECOND KEPT NEGATIVE -- the obvious reuse is the wrong tool, and it was measured before being rejected.
`holographic_deltachain.DeltaChain` stores base + per-chunk deltas and skips rows that did not change. On a
physics trace EVERY body moves EVERY frame, so it stores every row plus the index bookkeeping: **memory_bytes()
614,144 vs full_bytes() 460,800 -- a 33% LOSS.** Row-delta compression is for sparse mutation (a scene graph, an
asset table); a sim is dense mutation with sparse *causes*, which is a different structure and wants a different
codec.
"""

import struct
import zlib

import numpy as np

_MAGIC = b"LEEV1"                                # a version tag, so a future format change is detectable not silent


class EventTrace:
    """A recorded trace as (base state, sparse events). `events` is an (E, 4) int array of (frame, body, axis,
    wall) plus an (E,) float64 array of impulse magnitudes -- the shape the reference bouncing-box replayer wants.

    `encode()` -> bytes; `EventTrace.decode(blob)` -> EventTrace. Deterministic: no RNG, and the byte layout is
    fixed little-endian, so a blob written on one machine decodes identically on another."""

    def __init__(self, base, keys, impulses, frames, dt=1.0 / 120.0, gravity=-9.81, box=2.0):
        self.base = np.ascontiguousarray(np.asarray(base, float))
        self.keys = np.ascontiguousarray(np.asarray(keys, np.int32).reshape(-1, 4))
        self.impulses = np.ascontiguousarray(np.asarray(impulses, float).ravel())
        self.frames = int(frames)
        # THE PARAMETERS TRAVEL WITH THE BLOB, and that is not a nicety. A trace is base + events + THE LAW THAT
        # CONNECTS THEM; a decoder that guesses dt or gravity replays a different universe. The first version of
        # this class omitted them and a cross-faculty test caught it immediately -- encoding a zero-gravity trace
        # and replaying it under the default -9.81 reconstructed a completely different simulation, silently.
        self.dt = float(dt)
        self.gravity = float(gravity)
        self.box = float(box)
        if len(self.keys) != len(self.impulses):
            raise ValueError("need one impulse per event key, got %d and %d" % (len(self.keys), len(self.impulses)))

    def encode(self, level=9):
        """Serialize to a compressed blob: base, events, AND the integrator parameters that make them replayable.
        Impulses are stored as LITERAL float64 -- see the module docstring for the measurement that says a
        quantized codebook buys 2x and costs you the trajectory."""
        head = struct.pack("<5sIIIIddd", _MAGIC, self.frames, self.base.shape[0], self.base.shape[1],
                           len(self.keys), self.dt, self.gravity, self.box)
        body = head + self.base.tobytes() + self.keys.tobytes() + self.impulses.tobytes()
        return zlib.compress(body, int(level))

    @staticmethod
    def decode(blob):
        """Inverse of `encode`. Raises on a bad magic rather than silently misreading a stale format."""
        body = zlib.decompress(blob)
        n_head = struct.calcsize("<5sIIIIddd")
        if len(body) < n_head:                                    # a truncated blob is not a struct.error, it is bad data
            raise ValueError("not an event-trace blob (only %d bytes)" % len(body))
        magic, frames, n, d, n_ev, dt, gravity, box = struct.unpack("<5sIIIIddd", body[:n_head])
        if magic != _MAGIC:
            raise ValueError("not an event-trace blob (magic %r)" % (magic,))
        off = n_head
        base = np.frombuffer(body, np.float64, n * d, off).reshape(n, d).copy()
        off += n * d * 8
        keys = np.frombuffer(body, np.int32, n_ev * 4, off).reshape(n_ev, 4).copy()
        off += n_ev * 4 * 4
        impulses = np.frombuffer(body, np.float64, n_ev, off).copy()
        return EventTrace(base, keys, impulses, frames, dt=dt, gravity=gravity, box=box)

    def nbytes(self, level=9):
        """The encoded size. The number the bar is measured against."""
        return len(self.encode(level=level))


def bouncing_box_trace(n=16, frames=600, seed=0, dt=1.0 / 120.0, gravity=-9.81,
                       restitution=0.8, box=2.0, speed=2.0):
    """The reference workload: `n` bodies under gravity in a box with restitution walls. Returns (trace, EventTrace)
    where `trace` is (frames, n, 6) of [position, velocity].

    Deterministic by construction: fixed integrator, fixed clamp, events applied in a fixed axis-then-wall-then-body
    order. That order is part of the format -- `replay_bouncing_box` reapplies it exactly, which is what makes the
    reconstruction bit-identical rather than merely close."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-box * 0.8, box * 0.8, size=(n, 3))
    V = rng.normal(size=(n, 3)) * float(speed)   # speed=0 with gravity=0 is a genuinely at-rest island: no events
    trace, keys, imps = [], [], []
    for f in range(frames):
        V[:, 1] += gravity * dt
        X += V * dt
        for k in range(3):                                   # axis, then wall, then body: the fixed order
            for w, (mask, lim) in enumerate(((X[:, k] < -box, -box), (X[:, k] > box, box))):
                for i in np.flatnonzero(mask):
                    X[i, k] = lim
                    j = -(1.0 + restitution) * V[i, k]
                    V[i, k] += j
                    keys.append((f, int(i), k, w))
                    imps.append(float(j))
        trace.append(np.concatenate([X, V], axis=1).copy())
    T = np.stack(trace)
    return T, EventTrace(T[0], np.array(keys, np.int32).reshape(-1, 4), np.array(imps, float), frames,
                         dt=dt, gravity=gravity, box=box)


def replay_bouncing_box(ev, dt=None, gravity=None, box=None):
    """Regenerate the full (frames, n, 6) trace from base + events. BIT-IDENTICAL to the recorded trace (measured:
    max|diff| exactly 0.0), because the integrator, the clamp and the event order are the same and the impulses were
    stored literally.

    This is the whole codec: the states were never data, they were a deterministic function of the base and the
    interruptions. Bit-exactness is REQUIRED, not a nicety -- a contact sequence is a chaotic map (see the module
    docstring's quantization measurement), so an approximate replay is a different simulation, not a noisy one."""
    # default to the parameters the TRACE was recorded with, not to this function's own defaults
    dt = ev.dt if dt is None else float(dt)
    gravity = ev.gravity if gravity is None else float(gravity)
    box = ev.box if box is None else float(box)
    base = ev.base
    X = base[:, :3].copy()
    V = base[:, 3:].copy()
    by_frame = {}
    for (f, i, k, w), j in zip(ev.keys, ev.impulses):
        by_frame.setdefault(int(f), []).append((int(i), int(k), int(w), float(j)))
    out = [base.copy()]
    for f in range(1, ev.frames):
        V[:, 1] += gravity * dt
        X += V * dt
        for i, k, w, j in by_frame.get(f, ()):               # insertion order == the recorded order
            X[i, k] = -box if w == 0 else box
            V[i, k] += j
        out.append(np.concatenate([X, V], axis=1).copy())
    return np.stack(out)


def compression_report(trace, ev, level=9):
    """The codec's size AND every baseline it claims to beat, computed together so the comparison travels with the
    capability and an agent can re-run it rather than take the number on trust (the `navigator_benchmark` pattern).

    Returns {raw, zlib_raw, zlib_frame_deltas, deltachain, event_codec, ratio_vs_frame_deltas, events, rows}.
    `zlib_frame_deltas` is the bar the backlog names; `deltachain` is the obvious reuse, reported so its failure on
    this workload is on the record rather than in a footnote."""
    T = np.ascontiguousarray(np.asarray(trace, float))
    raw = T.tobytes()
    deltas = np.concatenate([T[:1], np.diff(T, axis=0)]).tobytes()
    from holographic.agents_and_reasoning.holographic_deltachain import DeltaChain
    dc = DeltaChain(T[0])
    for f in range(1, len(T)):
        dc.append(T[f])
    codec = ev.nbytes(level=level)
    zdel = len(zlib.compress(deltas, level))
    return {"raw": len(raw),
            "zlib_raw": len(zlib.compress(raw, level)),
            "zlib_frame_deltas": zdel,
            "deltachain": int(dc.memory_bytes()),
            "event_codec": codec,
            "ratio_vs_frame_deltas": zdel / codec,
            "events": int(len(ev.impulses)),
            "rows": int(T.shape[0] * T.shape[1])}


def _selftest():
    """Regression trap for X7: the bar (beat delta-compression of raw state), bit-exact replay, the round-trip,
    and both kept negatives -- the codebook is lossy-and-chaotic, and DeltaChain is the wrong tool here."""
    T, ev = bouncing_box_trace()
    assert T.shape == (600, 16, 6) and len(ev.impulses) == 663

    # 1. BIT-EXACT replay. Not 'close' -- exactly equal, because a contact sequence does not degrade gracefully.
    assert np.array_equal(replay_bouncing_box(ev), T)

    # 2. the blob round-trips, and a bad magic is refused rather than misread
    blob = ev.encode()
    back = EventTrace.decode(blob)
    assert np.array_equal(back.base, ev.base) and np.array_equal(back.keys, ev.keys)
    assert np.array_equal(back.impulses, ev.impulses) and back.frames == ev.frames
    assert (back.dt, back.gravity, back.box) == (ev.dt, ev.gravity, ev.box)   # the LAW travels with the data
    assert np.array_equal(replay_bouncing_box(back), T)               # decode -> replay is still bit-exact
    try:
        EventTrace.decode(zlib.compress(b"NOPE" + b"\x00" * 40))
    except ValueError:
        pass
    else:
        raise AssertionError("a bad magic must be refused")

    # 3. THE BAR: beat delta-compression of raw state, against the strongest honest baseline.
    rep = compression_report(T, ev)
    assert rep["event_codec"] < rep["zlib_frame_deltas"]
    assert rep["ratio_vs_frame_deltas"] > 10.0
    assert rep["rows"] > 14 * rep["events"]                           # 663 events replace 9,600 rows: the sparsity

    # 4. KEPT NEGATIVE: DeltaChain LOSES on a dense-mutation trace. Measured, not assumed.
    assert rep["deltachain"] > rep["raw"]

    # 5. KEPT NEGATIVE: a quantized impulse codebook is lossy, and the loss AMPLIFIES -- events decide which
    #    events happen next. At q=0.1 the reconstruction leaves a box of half-extent 2.0.
    q = 0.1
    coarse = EventTrace(ev.base, ev.keys, np.round(ev.impulses / q) * q, ev.frames,
                        dt=ev.dt, gravity=ev.gravity, box=ev.box)
    err = float(np.abs(replay_bouncing_box(coarse) - T).max())
    assert err > 1.0, ("a coarse impulse codebook is supposed to wreck the trajectory", err)
    assert coarse.nbytes() < ev.nbytes()                              # ... which is what its 2x buys

    print("OK: holographic_eventcodec self-test passed (600 frames x 16 bodies, 663 events: replay is BIT-IDENTICAL; "
          "codec %d bytes vs zlib(frame deltas) %d -- %.1fx, the bar; DeltaChain would take %d, MORE than the raw %d, "
          "because a sim is dense mutation with sparse causes; and a q=0.1 impulse codebook saves bytes while moving "
          "the final state by %.2f in a box of half-extent 2.0)"
          % (rep["event_codec"], rep["zlib_frame_deltas"], rep["ratio_vs_frame_deltas"],
             rep["deltachain"], rep["raw"], err))


if __name__ == "__main__":
    _selftest()
