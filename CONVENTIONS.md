# Conventions & Gotchas

*The handful of conventions that are load-bearing and easy to get wrong. Each one below cost real time on a
demo when it was gotten wrong the first time. Read this page before starting a new demo or app, so you meet
these once here instead of one bug at a time. (This is the "conventions & gotchas" section for the dev guide;
it lives as its own file so it is easy to find and to paste into the guide.)*

---

## 1. SDF sign — negative inside

A signed-distance function returns the distance to the nearest surface, **signed**:

- **negative inside** the surface,
- **zero** on it,
- **positive outside**.

Every SDF in the engine follows this, and `SDFScene.eval` returns the `min` over parts — so the scene's
zero-level-set is the union of its parts. If you write a primitive with the sign flipped, the ray-marcher walks
the wrong way and the surface turns inside-out. When in doubt, evaluate your SDF at a point you know is inside
and check it is negative.

## 2. Colour space — shade in linear, tonemap + gamma at the very end

The renderer hands back a **raw linear** radiance buffer. It is *not* display-ready — showing it directly looks
washed out or blown out. The display pipeline is, in order:

1. render → linear radiance (HDR, can exceed 1.0),
2. **exposure** (scale linear radiance; done in HDR, *before* tonemapping),
3. **tonemap** (e.g. Reinhard `x/(1+x)` or ACES — compress HDR → LDR),
4. **gamma / sRGB** encode (the last step, linear → display).

The gotcha: **mixing display-space and linear values silently washes things out.** Keep everything linear until
the tonemap+gamma at the very end (that is what `holographic_postfx` does). Don't gamma-encode twice, and don't
feed an already-tonemapped image back into a linear operation.

## 3. Camera handedness — `right = forward × up`, `view-up = right × forward`

The camera looks from `eye` toward `target`:

```
forward  = normalize(target - eye)
right    = forward × up          # (world up)
view-up  = right × forward
```

`holographic_camera` and `holographic_transform.look_at` both follow this (the camera looks down **−z** in view
space, y is up — the OpenGL convention). The gotcha: **a mirrored `right` vector renders the whole scene
flipped left-to-right**, and it is easy to do by writing `up × forward` instead of `forward × up`. If your scene
comes out mirrored, this cross-product order is the first thing to check.

## 4. Determinism — the trio that keeps outputs reproducible

Non-negotiable, engine-wide. Every output must be reproducible bit-for-bit across runs:

- **`PYTHONHASHSEED=0`** — set it in the environment; otherwise set/dict iteration order can vary run to run.
- **`hashlib`, never Python's `hash()`** — `hash()` is salted per-process and is *not* stable; use `hashlib`
  (e.g. sha256) for any content hash or content-addressed key.
- **stable sorts** — sort by an explicit key that fully determines order (add a tie-break like the index), so a
  tie never resolves differently on another run or platform.

The deeper lesson behind the trio (the `bind_batch` story): **a change that is bit-identical to 1e-12 can still
flip a downstream discrete decision** (it flipped a creature's maze trajectory once). So on any tie-sensitive
path — argmin, argmax, a sort tie, a threshold — preserve the *exact* arithmetic and ordering, and keep new
"equivalent" fast paths out of it unless they are bit-for-bit identical.

---

## Bonus gotcha — import-time `NameError`

A module-level constant that references a name defined **lower** in the same file raises a `NameError` the
moment the module is imported (before any function runs). This bit the garage demo repeatedly. It shows up
instantly as an import failure — which is exactly why `tools/demo_kit.smoke_test` catches it: it *imports* the
backend as its first step, so a use-before-def surfaces immediately rather than at request time.
