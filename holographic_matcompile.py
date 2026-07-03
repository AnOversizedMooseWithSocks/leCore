"""holographic_matcompile.py -- COMPILE + FUSE MATERIALS (fluids/matter backlog, performance item MC1).

Do some materials render slower? Yes -- `surface` resolves EVERY channel PER HIT, EVERY frame, and that recomputation
is the slowness. The fix the backlog calls for is "a material is a program (a socket graph), compiled once and reused":
compile+fuse it ONCE per (material, options), key it by a content hash, and hand the SAME kernel to every hit, every
instance, every frame. This reuses the content-addressed compile cache that already ships (holographic_compile).

What the compile actually buys here, concretely and measurably:
  * the kernel is BUILT ONCE -- N frames of the same material cost one compile and N-1 cache hits (no rebuild);
  * CONSTANT channels (a flat roughness, a fixed reflect) are resolved ONCE at compile and just broadcast per hit,
    so only the genuinely view/position-dependent SOCKET channels (a procedural pattern) re-resolve. A mostly-flat
    material -- the common case -- skips most of its per-hit work.

The heavier baking (turn a procedural texture into a lookup, a view LUT for specular) is MC2/MC3; this is the cheap,
always-safe first move that helps every material.

KEPT NEGATIVE: this folds CONSTANT channels and caches the build; a fully-procedural material (every channel a
position-dependent field) still resolves every channel per hit -- the compile saves the rebuild, not the field
evaluation. That evaluation is what MC2's sdfbake/prt turns into a lookup. Determinism: constant detection probes the
socket at two point sets (position independence), deterministic.
"""
import numpy as np

from holographic_compile import compiled, DEFAULT_CACHE
from holographic_param import resolve_param
from holographic_surface import _rgb


CHANNELS = ("color", "roughness", "reflect", "emission", "opacity")


def _resolve(name, ch, points, n):
    """Resolve one channel socket at `points`. `color` -> (n,3) via surface._rgb (handles triples, grey, colour
    fields); every other channel -> (n,) scalar via resolve_param."""
    if name == "color":
        return _rgb(ch, points, n)
    v = np.asarray(resolve_param(ch, points=points, n=n), float)
    return np.full(n, float(v)) if v.ndim == 0 else v.reshape(n)


def _is_const(name, ch):
    """Is a channel position-INDEPENDENT (a constant)? Probe it at two different point sets and compare -- a robust,
    general test that works for scalars, rgb triples, Params, and callable fields alike."""
    a = _resolve(name, ch, np.zeros((2, 3)), 2)
    b = _resolve(name, ch, np.ones((2, 3)) * 7.0, 2)
    return a.shape == b.shape and np.allclose(a, b)


def material_spec(material):
    """A canonical, hashable spec for a material: constant channels by VALUE, socket channels by a stable id (so the
    SAME material reuses its compiled kernel across frames, and two identical-constant materials share one)."""
    spec = []
    for name in CHANNELS:
        ch = getattr(material, name)
        if _is_const(name, ch):
            val = _resolve(name, ch, np.zeros((2, 3)), 2)[0]    # constant value (2 pts avoids a 1-pt resolve_param edge case)
            spec.append((name, "const", tuple(np.atleast_1d(val).ravel().tolist())))
        else:
            spec.append((name, "socket", id(ch)))
    return tuple(spec)


def compiled_shader(material, cache=None):
    """Return a compiled shade(points)->channel-dict for `material`, BUILT ONCE and cached by the material's spec.
    Constant channels are folded to precomputed values (broadcast per hit); only socket channels re-resolve. Reuses
    the content-addressed compile cache -- the same material hands back the same kernel with no rebuild."""
    cache = cache if cache is not None else DEFAULT_CACHE       # `or` would ignore an empty (falsy) cache -- see compiled()
    spec = material_spec(material)

    def _compile(_spec):
        # split channels once: constants get precomputed here; sockets stay as live channels to resolve per hit
        const_vals = {}
        socket_names = []
        for name in CHANNELS:
            ch = getattr(material, name)
            if _is_const(name, ch):
                const_vals[name] = _resolve(name, ch, np.zeros((2, 3)), 2)[0]   # resolved ONCE at compile (2 pts, see above)
            else:
                socket_names.append(name)

        def shade(points):
            points = np.asarray(points, float)
            n = len(points)
            out = {}
            for name in CHANNELS:
                if name in const_vals:                          # cheap broadcast of the folded constant, no re-resolve
                    cv = const_vals[name]
                    out[name] = np.tile(cv, (n, 1)) if np.ndim(cv) and np.size(cv) == 3 else np.full(n, float(cv))
                else:
                    out[name] = _resolve(name, getattr(material, name), points, n)   # only sockets pay the per-hit cost
            return out

        shade.socket_names = socket_names                       # expose what still resolves per hit (for measurement)
        shade.const_names = list(const_vals)
        return shade

    return compiled(spec, _compile, tag="shader", cache=cache)


def _selftest():
    """A compiled shader matches the naive per-hit resolve, is built once and reused (cache hits), and folds the
    constant channels so only the procedural ones re-resolve per hit."""
    from holographic_surface import SurfaceMaterial
    from holographic_param import Param
    from holographic_compile import CompileCache

    pts = np.random.default_rng(0).uniform(-1, 1, size=(50, 3))

    # a mostly-constant material with ONE procedural (position-dependent) channel: roughness varies with x
    rough_field = lambda P, **k: 0.2 + 0.3 * (np.asarray(P)[:, 0] > 0)
    mat = SurfaceMaterial(color=(0.8, 0.3, 0.2), roughness=Param(field=rough_field), reflect=0.1, emission=0.0)

    cache = CompileCache()
    shade = compiled_shader(mat, cache=cache)

    # correctness: compiled shade == the material's own resolve, channel for channel
    ref = mat.resolve(pts)
    got = shade(pts)
    for name in CHANNELS:
        assert np.allclose(got[name], ref[name]), name

    # only roughness re-resolves per hit; color/reflect/emission/opacity were folded to constants
    assert shade.socket_names == ["roughness"], shade.socket_names
    assert set(shade.const_names) == {"color", "reflect", "emission", "opacity"}

    # built once, reused: compiling the SAME material again is a cache hit (no second compile)
    compiled_shader(mat, cache=cache)
    compiled_shader(mat, cache=cache)
    assert cache.stats["compiles"] == 1 and cache.stats["hits"] >= 2, cache.stats

    # a different constant makes a different spec -> a fresh compile (the cache is content-addressed, not identity)
    mat2 = SurfaceMaterial(color=(0.1, 0.1, 0.9), roughness=Param(field=rough_field), reflect=0.1)
    compiled_shader(mat2, cache=cache)
    assert cache.stats["compiles"] == 2, cache.stats

    print("holographic_matcompile selftest OK: compiled shader matches resolve() exactly; only the procedural "
          "'roughness' re-resolves per hit while color/reflect/emission/opacity are folded to constants; the same "
          "material compiles ONCE and is reused (compiles=%d, hits=%d); a changed constant is a fresh content-addressed "
          "compile" % (cache.stats["compiles"], cache.stats["hits"]))


if __name__ == "__main__":
    _selftest()
