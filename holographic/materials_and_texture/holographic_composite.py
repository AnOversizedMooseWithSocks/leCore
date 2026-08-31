"""COMPOSITE -- the blend modes and the alpha-over loop, ONCE, for every app.

Layer compositing is the one operation every image-consuming app must perform
IDENTICALLY, and it lived only inside leStudio: ten modes defined in that app's
own __init__.py, with nothing in the engine. Verified before writing this --
the engine defined no BLEND_MODES and no composite_layers.

WHY THAT IS A CORRECTNESS BUG AND NOT AN ERGONOMICS ONE: any second app reading
a shared workspace must re-implement all ten plus the alpha-over loop, and TWO
COPIES OF THE SAME MATHS DRIFT. The same document then renders differently in
the modeller than in the painter -- exactly the failure the shared container
format was built to prevent. `normal` is easy and stays in agreement; the nine
others are where copies diverge, because each is a one-line formula that four
different people will round, clamp and order slightly differently.

THE FORMULAS ARE THE STANDARD ONES (PDF 1.7 blend modes / the W3C compositing
spec), written on PREMULTIPLIED-BY-NOTHING straight alpha in 0..1 float, which
is what the container format already carries. Every mode is a pure function of
(backdrop, source) per channel; alpha compositing is applied afterwards by the
same Porter-Duff over in every case, so a new mode is one line and cannot get
the alpha wrong.

THE SEPARABLE-MODE CONTRACT, worth stating because it is what makes this
shareable: B(cb, cs) operates per channel and ignores alpha. The result is then
    co = cs*as + cb*ab*(1-as)          [premultiplied out]
    ao =    as + ab*(1-as)
with the blended colour substituted for cs where ab > 0. That is the whole
model, and it is why matching leStudio needs no leStudio.
"""

import numpy as np


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _normal(cb, cs):
    return cs


def _multiply(cb, cs):
    return cb * cs


def _screen(cb, cs):
    return cb + cs - cb * cs


def _overlay(cb, cs):
    # overlay(cb, cs) == hardlight(cs, cb) -- the standard identity, written
    # explicitly so the two never drift apart in this file.
    return np.where(cb <= 0.5, 2.0 * cb * cs,
                    1.0 - 2.0 * (1.0 - cb) * (1.0 - cs))


def _add(cb, cs):
    return _clip01(cb + cs)


def _subtract(cb, cs):
    return _clip01(cb - cs)


def _difference(cb, cs):
    return np.abs(cb - cs)


def _darken(cb, cs):
    return np.minimum(cb, cs)


def _lighten(cb, cs):
    return np.maximum(cb, cs)


def _softlight(cb, cs):
    # THE W3C FORM, not the cheap approximation. The cheap one
    # (2*cb*cs + cb^2*(1-2*cs)) differs visibly in the dark end, and "visibly"
    # is precisely the drift this module exists to prevent -- a shared kernel
    # that is ALMOST the same is worse than no shared kernel, because the
    # difference shows up as a rendering discrepancy nobody can attribute.
    d = np.where(cb <= 0.25, ((16.0 * cb - 12.0) * cb + 4.0) * cb,
                 np.sqrt(np.maximum(cb, 0.0)))
    return np.where(cs <= 0.5,
                    cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb),
                    cb + (2.0 * cs - 1.0) * (d - cb))


#: The ten modes leStudio defines, by the names it already writes into a
#: container's layer records -- so a section round-trips without translation.
BLEND_MODES = {
    "normal": _normal,
    "multiply": _multiply,
    "screen": _screen,
    "overlay": _overlay,
    "add": _add,
    "subtract": _subtract,
    "difference": _difference,
    "darken": _darken,
    "lighten": _lighten,
    "softlight": _softlight,
}


def blend(name, backdrop, source):
    """Apply one separable blend mode to two straight-alpha colour arrays."""
    fn = BLEND_MODES.get(str(name or "normal"))
    if fn is None:
        raise KeyError("unknown blend mode %r -- known: %s"
                       % (name, sorted(BLEND_MODES)))
    return _clip01(fn(np.asarray(backdrop, np.float64),
                      np.asarray(source, np.float64)))


def composite_layers(layers, meta=None, background=None):
    """Composite a layer stack into one image. The engine-side of leStudio's display.

    `layers` maps a layer id to an (H, W, 3|4) float array in 0..1; `meta` is
    the list of layer records exactly as a container section carries them --
    {id, name, visible, opacity, blend, mask?} -- IN PAINT ORDER, first at the
    bottom. Returns (H, W, 4).

    THE RECORDS ARE READ, NOT REINTERPRETED: `visible` false skips, `opacity`
    scales the layer's alpha (not its colour -- scaling colour darkens instead
    of fading, which is the classic wrong version), `blend` names a mode, and an
    optional `mask` multiplies alpha. A layer with no alpha channel is treated
    as fully opaque, which is what an RGB texture means.
    """
    recs = list(meta if meta is not None else
                [{"id": k} for k in sorted(layers)])
    first = None
    for r in recs:
        a = layers.get(r.get("id"))
        if a is not None:
            first = np.asarray(a)
            break
    if first is None:
        raise ValueError("no layer arrays matched the records")
    h, w = first.shape[:2]

    if background is None:
        out_rgb = np.zeros((h, w, 3), np.float64)
        out_a = np.zeros((h, w), np.float64)
    else:
        bg = np.asarray(background, np.float64)
        out_rgb = _clip01(bg[..., :3])
        out_a = (bg[..., 3] if bg.shape[-1] == 4
                 else np.ones((h, w), np.float64))

    for r in recs:
        arr = layers.get(r.get("id"))
        if arr is None or not r.get("visible", True):
            continue
        src = np.asarray(arr, np.float64)
        cs = _clip01(src[..., :3])
        a_s = (src[..., 3] if src.shape[-1] == 4
               else np.ones(src.shape[:2], np.float64))
        a_s = _clip01(a_s * float(r.get("opacity", 1.0)))
        mask = r.get("mask")
        if mask is not None:
            a_s = a_s * _clip01(np.asarray(mask, np.float64))

        # BLEND AGAINST THE BACKDROP, THEN COMPOSITE. Doing it the other way
        # (compositing first, then blending) is the mistake that makes a
        # multiply layer over transparency go black: with ab == 0 there is no
        # backdrop to multiply, and the spec says the blend result must fade to
        # the source itself exactly there.
        blended = blend(r.get("blend", "normal"), out_rgb, cs)
        eff = out_a[..., None] * blended + (1.0 - out_a[..., None]) * cs

        a_out = a_s + out_a * (1.0 - a_s)
        num = eff * a_s[..., None] + out_rgb * out_a[..., None] * (1.0 - a_s[..., None])
        with np.errstate(invalid="ignore", divide="ignore"):
            out_rgb = np.where(a_out[..., None] > 0.0,
                               num / np.maximum(a_out[..., None], 1e-12), 0.0)
        out_a = a_out

    return np.concatenate([_clip01(out_rgb), _clip01(out_a)[..., None]], -1)


def _selftest():
    rng = np.random.default_rng(0)
    cb = rng.random((8, 8, 3))
    cs = rng.random((8, 8, 3))

    # ---- EVERY MODE MUST STAY IN RANGE AND BE PURE ----
    for name in BLEND_MODES:
        out = blend(name, cb, cs)
        assert out.shape == cb.shape, name
        assert out.min() >= -1e-12 and out.max() <= 1.0 + 1e-12, (name, out.min(), out.max())
        assert np.array_equal(out, blend(name, cb, cs)), name

    # ---- THE IDENTITIES THAT PIN THE FORMULAS ----
    z, o = np.zeros_like(cb), np.ones_like(cb)
    assert np.allclose(blend("multiply", cb, o), cb)          # x1 is identity
    assert np.allclose(blend("multiply", cb, z), 0.0)
    assert np.allclose(blend("screen", cb, z), cb)            # +0 is identity
    assert np.allclose(blend("screen", cb, o), 1.0)
    assert np.allclose(blend("difference", cb, cb), 0.0)
    assert np.allclose(blend("darken", cb, o), cb)
    assert np.allclose(blend("lighten", cb, z), cb)
    assert np.allclose(blend("normal", cb, cs), cs)
    # overlay(cb, cs) == hardlight(cs, cb): check the branch point behaves
    lo = blend("overlay", np.full_like(cb, 0.25), cs)
    assert np.allclose(lo, 2.0 * 0.25 * cs)

    # ---- A ONE-LAYER OPAQUE STACK IS THE LAYER ----
    lay = {"a": np.concatenate([cs, np.ones((8, 8, 1))], -1)}
    out = composite_layers(lay, [{"id": "a", "blend": "normal"}])
    assert np.allclose(out[..., :3], cs, atol=1e-9), np.abs(out[..., :3] - cs).max()
    assert np.allclose(out[..., 3], 1.0)

    # ---- INVISIBLE AND ZERO-OPACITY LAYERS CONTRIBUTE NOTHING ----
    two = {"a": np.concatenate([cs, np.ones((8, 8, 1))], -1),
           "b": np.concatenate([cb, np.ones((8, 8, 1))], -1)}
    hidden = composite_layers(two, [{"id": "a"}, {"id": "b", "visible": False}])
    assert np.allclose(hidden[..., :3], cs, atol=1e-9)
    faded = composite_layers(two, [{"id": "a"}, {"id": "b", "opacity": 0.0}])
    assert np.allclose(faded[..., :3], cs, atol=1e-9)

    # ---- OPACITY FADES, IT DOES NOT DARKEN ----
    # the classic wrong version scales COLOUR, which sends a half-opacity white
    # layer over white to grey instead of leaving it white.
    white = {"w": np.ones((8, 8, 4))}
    half = composite_layers(white, [{"id": "w", "opacity": 0.5}],
                            background=np.ones((8, 8, 4)))
    assert np.allclose(half[..., :3], 1.0), half[..., :3].min()

    # ---- A MULTIPLY LAYER OVER NOTHING IS ITSELF, NOT BLACK ----
    mult = composite_layers({"m": np.concatenate([cs, np.ones((8, 8, 1))], -1)},
                            [{"id": "m", "blend": "multiply"}])
    assert np.allclose(mult[..., :3], cs, atol=1e-9), np.abs(mult[..., :3] - cs).max()

    print("composite selftest OK -- %d blend modes and the alpha-over loop live "
          "in the ENGINE now, so a second app consuming a shared workspace does "
          "not re-implement them and drift. Identities pinned (multiply by 1, "
          "screen by 0, difference with self, darken/lighten limits); opacity "
          "FADES rather than darkens (a half-opacity white layer over white "
          "stays white, which the colour-scaling version gets wrong); and a "
          "multiply layer over transparency is ITSELF rather than black, which "
          "is what compositing before blending gets wrong"
          % len(BLEND_MODES))


if __name__ == "__main__":
    _selftest()
