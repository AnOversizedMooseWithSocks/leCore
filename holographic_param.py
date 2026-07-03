"""Connectable parameters -- a value that can be a CONSTANT or WIRED to something else.

Every DCC app (Blender, Houdini, Nuke, Substance) has this affordance: a parameter field with a little dot beside it,
and instead of typing a number you can plug in a texture MAP, a procedural FIELD, or the OUTPUT of another node. A
roughness that is a texture; an emit-rate driven by a density field; a scale wired to a curvature map. holostuff had
only bare numbers everywhere -- so this module adds the socket.

The design is deliberately tiny and additive: one `Param` class (a socket that holds a constant OR a connection) and
one `resolve_param(...)` function that turns ANY of {scalar, ndarray of per-point values, callable field, `Param`}
into concrete values at the points you ask about. Existing code keeps passing bare numbers (a scalar resolves to
itself); new code can accept a `Param` and call `resolve_param` to get 'more than a number' for free. This is the
data-oriented / node-graph composability the panel put first: a parameter is just another edge in the VSA program.

WHY a single resolver rather than per-call-site handling: so the rule for "what can drive a parameter" lives in ONE
place and every faculty gets the same set of connection kinds (const / map / field / named source) the moment it opts
in. Deterministic: no randomness here; sampling a map is nearest-cell by construction.
"""
import numpy as np


class Param:
    """A parameter SOCKET: it holds EITHER a constant OR a CONNECTION to something that produces values.

    Exactly one channel is 'live'; the rest are None:
      * `value`  -- a plain constant (the default, backward-compatible case).
      * `field`  -- a callable f(points (M,D)) -> (M,) values: a procedural / another faculty's output as a field.
      * `map`    -- an ndarray sampled over `domain`=(lo, hi): a TEXTURE / baked field indexed by position.
      * `source` -- a name looked up in the resolve `ctx` dict: a wire to another node's named output.

    `default` is used when a `source` isn't found, so a dangling connection degrades gracefully instead of crashing.
    """

    def __init__(self, value=None, field=None, map=None, domain=None, source=None, default=0.0):
        self.value = value
        self.field = field
        self.map = None if map is None else np.asarray(map, float)
        self.domain = domain
        self.source = source
        self.default = default

    def connected(self):
        """True if this socket is wired to a map / field / named source rather than holding a bare constant."""
        return self.field is not None or self.map is not None or self.source is not None

    def resolve(self, points=None, ctx=None, n=None):
        """Concrete value(s) at `points` (or length `n`), following the connection. See module `resolve_param`."""
        return resolve_param(self, points=points, ctx=ctx, n=n)

    def __repr__(self):
        kind = ("field" if self.field is not None else "map" if self.map is not None
                else "source:%s" % self.source if self.source is not None else "const:%s" % (self.value,))
        return "Param(%s)" % kind


def _broadcast(scalar, points, n):
    """A constant -> an array of the right length (or the bare scalar if no length is implied)."""
    m = (len(points) if points is not None else n)
    if m is None:
        return float(scalar)
    return np.full(int(m), float(scalar))


def _as_values(vals, points, n):
    """Coerce a field/callable result to a 1-D per-point array of the expected length."""
    a = np.asarray(vals, float).ravel()
    m = (len(points) if points is not None else n)
    if m is not None and a.size == 1:
        return np.full(int(m), float(a))
    return a


def _sample_map(mp, points, domain):
    """Sample an N-D `map` at continuous `points` via nearest-cell indexing over `domain`=(lo, hi). A texture read:
    the map's shape defines the resolution; a point's position picks the cell. Nearest (not bilinear) to stay exact
    and dependency-free -- callers wanting smoothness can pre-blur the map."""
    if points is None:
        raise ValueError("sampling a map needs points to sample at")
    P = np.asarray(points, float)
    D = mp.ndim
    if domain is None:
        lo = np.zeros(D); hi = np.array(mp.shape, float)
    else:
        lo = np.asarray(domain[0], float); hi = np.asarray(domain[1], float)
    frac = (P[:, :D] - lo) / np.maximum(hi - lo, 1e-9)
    idx = np.clip((frac * np.array(mp.shape, float)).astype(int), 0, np.array(mp.shape) - 1)
    return mp[tuple(idx[:, k] for k in range(D))]


def resolve_param(p, points=None, ctx=None, n=None):
    """Resolve ANY parameter spec to concrete value(s) at `points` (shape (M, D)).

    Accepts, in order: a `Param` socket (dispatched on its live channel, following `source` wires through `ctx`); a
    callable field f(points)->values; an ndarray of per-point values (used as-is when its length matches); or a bare
    scalar (broadcast). `ctx` is a dict of other named outputs a `source=` connection can reach. `n` is a fallback
    length for broadcasting a constant when no `points` are given.

    This is the one place that decides 'what can drive a parameter', so every faculty that calls it inherits the same
    connection kinds -- the affordance Moose asked for: a parameter that takes a map or another output, not just a
    number.
    """
    if isinstance(p, Param):
        if p.field is not None:
            return _as_values(p.field(points), points, n)
        if p.map is not None:
            return _sample_map(p.map, points, p.domain)
        if p.source is not None:
            if ctx is not None and p.source in ctx:
                return resolve_param(ctx[p.source], points=points, ctx=ctx, n=n)   # follow the wire
            return _broadcast(p.default, points, n)                                # dangling -> default
        return _broadcast(p.value if p.value is not None else p.default, points, n)
    if callable(p):
        return _as_values(p(points), points, n)
    if isinstance(p, np.ndarray):
        if points is not None and p.ndim == 1 and p.size == len(points):
            return p.astype(float)                                                  # already per-point values
        return _sample_map(np.asarray(p, float), points, None)                      # else treat as a spatial map
    return _broadcast(p, points, n)


def _selftest():
    """A parameter resolves the same whether it's a constant, a field, a map, or a wired source -- one socket, many
    drivers."""
    pts = np.array([[0.2, 0.0], [0.8, 0.0], [0.5, 0.0]])
    # constant
    assert np.allclose(resolve_param(0.3, pts), [0.3, 0.3, 0.3])
    assert np.allclose(resolve_param(Param(value=0.3), pts), 0.3)
    # field (callable): roughness rises with x
    assert np.allclose(resolve_param(Param(field=lambda P: P[:, 0]), pts), [0.2, 0.8, 0.5])
    # map (texture): a 1-D map over domain [0,1] sampled at x
    mp = np.array([0.0, 0.5, 1.0, 1.0])
    got = resolve_param(Param(map=mp, domain=(np.array([0.0, 0.0]), np.array([1.0, 1.0]))), pts)
    assert got.shape == (3,)
    # source: wire 'roughness' to a curvature output living in ctx
    ctx = {"curv": Param(field=lambda P: P[:, 0] * 2.0)}
    assert np.allclose(resolve_param(Param(source="curv"), pts, ctx=ctx), [0.4, 1.6, 1.0])
    # dangling source -> default, no crash
    assert np.allclose(resolve_param(Param(source="missing", default=0.7), pts), [0.7, 0.7, 0.7])
    print("param selftest ok: const/field/map/source all resolve; dangling wire falls back to default")


if __name__ == "__main__":
    _selftest()
