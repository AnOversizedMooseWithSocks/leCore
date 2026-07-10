"""holographic_postfx.py -- composable post-processing for the rasterized (H, W, 3) pixel output.

The renderer hands back a RAW linear buffer; modern game engines never show that directly. This module is the
"projection tail": an optional, ordered, named PROGRAM of effects -- a `PostChain` with the same shape as a
HoloMachine instruction sequence -- composed onto the frame as the last step of projection.

WHY this belongs to the VSA engine and isn't just an image library bolted on:
  * The CONVOLUTION FAMILY (gaussian blur, bloom, glare, depth-of-field, denoise, sharpen) runs on the engine's own
    core operator. bind(a, b) = irfft(rfft(a) * rfft(b)) is 1-D CIRCULAR CONVOLUTION; a 2-D blur is
    irfft2(rfft2(image) * G) -- the SAME operator, one dimension up. The blur kernel is a frequency-domain Gaussian,
    so there is no kernel truncation. "One operator, many costumes."
  * BLOOM and GLARE are a SUPERPOSITION (bundle) of blurred bright layers added back onto the frame -- bundling, the
    engine's other primitive, in image space.
  * The chain itself is a PROGRAM: an ordered, serializable, composable list of (effect, params), applied in sequence
    exactly like the machine runs an instruction list.

HONEST about what is NOT VSA magic: the per-pixel curves (exposure, tonemap, gamma, colour grade, vignette, film
grain, chromatic aberration, lens flare) are plain vectorized NumPy. They live in the same pipeline because that is
where a frame gets graded -- but they are tone/colour math, not bind/bundle. Calling them "VSA" would be dishonest;
they are the rest of the program. Everything here is deterministic (film grain is seeded) and dependency-free
(NumPy + its FFT only).
"""
import numpy as np


# --------------------------------------------------------------------------------------------------------------
# Convolution core -- the bind operator, one dimension up
# --------------------------------------------------------------------------------------------------------------
def _fft_blur(img, sigma):
    """Isotropic Gaussian blur by multiplying the image's 2-D real FFT by a Gaussian transfer function -- bind
    (multiply in the frequency domain) generalized to the image plane. Circular (wraps at the frame edges)."""
    if sigma <= 0:
        return np.asarray(img, float).copy()
    img = np.asarray(img, float)
    H, W = img.shape[:2]
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    G = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * (fy * fy + fx * fx))      # Gaussian in frequency = Gaussian
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = np.fft.irfft2(np.fft.rfft2(img[:, :, c]) * G, s=(H, W))
    return out


def _shift_clamp(img, oy, ox):
    """Integer image shift with edge clamping (no wrap), vectorized -- the tap used by directional blur."""
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    Y = np.clip(yy - oy, 0, H - 1)
    X = np.clip(xx - ox, 0, W - 1)
    return img[Y, X]


def _directional_blur(img, angle_deg, length):
    """Linear (motion / streak) blur: average the frame shifted along a direction -- a 1-D convolution with a line
    kernel, done by shift-and-add. Used by motion blur and anamorphic glare."""
    img = np.asarray(img, float)
    n = int(max(length, 1))
    if n <= 1:
        return img.copy()
    ang = np.radians(angle_deg)
    dx, dy = np.cos(ang), np.sin(ang)
    acc = np.zeros_like(img)
    for i in range(n):
        t = i - (n - 1) / 2.0
        acc += _shift_clamp(img, int(round(dy * t)), int(round(dx * t)))
    return acc / n


# --------------------------------------------------------------------------------------------------------------
# Tone / exposure / colour -- per-pixel curves (honest NumPy, not VSA)
# --------------------------------------------------------------------------------------------------------------
def exposure(img, ev=0.0):
    """Stops of exposure: scale linear radiance by 2**ev. Applied BEFORE tonemapping (works in HDR)."""
    return np.asarray(img, float) * (2.0 ** ev)


def auto_exposure(img, key=0.18):
    """AUTO-exposure: meter the frame so its LOG-AVERAGE luminance lands on the mid-grey `key` (0.18, the
    photographer's grey card -- Reinhard et al.'s photographic key). A scene lit by a bright HDR sun has no fixed
    'right' exposure; this makes every scene self-expose to a consistent mid-tone with no hand-set stop. Applied
    BEFORE tonemapping (works in HDR), exactly like exposure()."""
    x = np.asarray(img, float)
    lum = x @ np.array([0.2126, 0.7152, 0.0722])                 # perceptual luminance
    log_avg = np.exp(np.mean(np.log(np.clip(lum, 0, None) + 1e-4)))   # log-average = the scene's key luminance
    return x * (key / (log_avg + 1e-6))


def reinhard(img):
    """Reinhard HDR -> LDR tonemap: x / (1 + x). The simplest highlight compressor."""
    x = np.maximum(np.asarray(img, float), 0.0)
    return x / (1.0 + x)


def aces(img):
    """ACES filmic tonemap (Narkowicz fit) -- the modern game look: deep contrast, graceful highlight rolloff."""
    x = np.maximum(np.asarray(img, float), 0.0)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def gamma(img, g=2.2):
    """Encode linear -> display gamma. The raw buffer is LINEAR; without this it reads dark and muddy."""
    return np.clip(np.asarray(img, float), 0.0, 1.0) ** (1.0 / g)


def color_grade(img, contrast=1.0, saturation=1.0, temperature=0.0, tint=0.0, lift=0.0):
    """Contrast (pivot 0.5), saturation (about Rec.709 luma), temperature (warm/cool on R vs B), tint (G), lift
    (raise the black level). The colourist's basic knobs, in one pass."""
    x = np.asarray(img, float) + lift
    x = (x - 0.5) * contrast + 0.5
    luma = (x * np.array([0.2126, 0.7152, 0.0722])).sum(-1, keepdims=True)
    x = luma + (x - luma) * saturation
    x = x + np.array([temperature, tint, -temperature])
    return np.clip(x, 0.0, 1.0)


def vignette(img, strength=0.4, radius=1.0, softness=1.0):
    """Radial darkening toward the corners -- the classic lens falloff that pulls the eye to centre."""
    img = np.asarray(img, float)
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2) / max(radius, 1e-6)
    mask = np.clip(1.0 - strength * np.clip(r, 0, 1) ** (2.0 / max(softness, 1e-3)), 0.0, 1.0)
    return img * mask[:, :, None]


# --------------------------------------------------------------------------------------------------------------
# Convolution-family effects -- built on _fft_blur / _directional_blur (the bind operator in image space)
# --------------------------------------------------------------------------------------------------------------
def bloom(img, threshold=0.8, sigma=4.0, intensity=0.6):
    """Bright-pass -> FFT blur -> add back. The glow around bright regions is a SUPERPOSITION (bundle) of the blurred
    bright layer onto the frame; the blur itself is the bind operator in 2-D."""
    img = np.asarray(img, float)
    bright = np.maximum(img - threshold, 0.0)
    return img + intensity * _fft_blur(bright, sigma)


def glare(img, threshold=0.85, length=24, intensity=0.5, streaks=2):
    """Anamorphic / star glare: bright pixels smeared along `streaks` directional kernels, summed -- bright streaks
    radiating from highlights, a bundle of directional convolutions."""
    img = np.asarray(img, float)
    bright = np.maximum(img - threshold, 0.0)
    acc = np.zeros_like(img)
    for k in range(max(streaks, 1)):
        acc += _directional_blur(bright, 180.0 * k / max(streaks, 1), length)
    return img + intensity * acc / max(streaks, 1)


def lens_flare(img, threshold=0.9, intensity=0.4, ghosts=4):
    """Cheap ghost flare: bright spots mirrored through the frame centre at fractional offsets, tinted alternately
    warm/cool, blurred and added -- the lens-ghost artefact games fake for spectacle."""
    img = np.asarray(img, float)
    H, W = img.shape[:2]
    bright = np.maximum(img - threshold, 0.0)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    acc = np.zeros_like(img)
    for g in range(1, ghosts + 1):
        s = -0.6 + 1.2 * g / (ghosts + 1.0)                 # offset factor along the centre axis
        Y = np.clip((cy + (yy - cy) * (-s)).astype(int), 0, H - 1)
        X = np.clip((cx + (xx - cx) * (-s)).astype(int), 0, W - 1)
        tint = np.array([1.0, 0.7, 0.4]) if g % 2 else np.array([0.4, 0.7, 1.0])
        acc += bright[Y, X] * tint
    return img + intensity * _fft_blur(acc, 2.0) / ghosts


def chromatic_aberration(img, strength=0.004):
    """Radial channel split: sample R slightly outward and B slightly inward of G -- the colour fringing of a cheap
    lens, strongest at the frame edges."""
    img = np.asarray(img, float)
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

    def warp(ch, k):
        Y = np.clip((cy + (yy - cy) * (1 + k)).astype(int), 0, H - 1)
        X = np.clip((cx + (xx - cx) * (1 + k)).astype(int), 0, W - 1)
        return ch[Y, X]

    out = img.copy()
    out[:, :, 0] = warp(img[:, :, 0], strength)
    out[:, :, 2] = warp(img[:, :, 2], -strength)
    return out


def dof(img, depth=None, focus=None, aperture=2.0, max_sigma=6.0):
    """Depth of field: blend a sharp frame with an FFT-blurred copy by the circle of confusion CoC =
    clip(|depth - focus| * aperture). Needs the renderer's depth buffer (ray distance t at the hit). If `focus` is
    None it locks onto the median scene depth. The cheap, fast two-level game DOF."""
    img = np.asarray(img, float)
    if depth is None:
        return img
    d = np.asarray(depth, float)
    finite = d < 1e29
    if focus is None:
        focus = float(np.median(d[finite])) if np.any(finite) else 1.0
    coc = np.where(finite, np.clip(np.abs(d - focus) * aperture, 0.0, 1.0), 1.0)   # background fully defocused
    blurred = _fft_blur(img, max_sigma)
    w = coc[:, :, None]
    return img * (1.0 - w) + blurred * w


def motion_blur(img, angle=0.0, length=12):
    """Camera / linear motion blur: a directional (line-kernel) convolution at `angle` degrees over `length` px."""
    return _directional_blur(img, angle, length)


def denoise(img, sigma=1.0):
    """Light isotropic FFT denoise. For heavier structure-aware denoising the engine's holographic_denoise (SVD /
    manifold / sublinear forest recall) is the right tool; this is the cheap edge-preserving-enough smoother."""
    return _fft_blur(img, sigma)


def sharpen(img, amount=0.5, sigma=1.5):
    """Unsharp mask: img + amount * (img - blur(img)). Recovers crispness lost to blur, resample or denoise."""
    img = np.asarray(img, float)
    return np.clip(img + amount * (img - _fft_blur(img, sigma)), 0.0, 1.0)


# --------------------------------------------------------------------------------------------------------------
# Grain and sampling
# --------------------------------------------------------------------------------------------------------------
def film_grain(img, amount=0.04, seed=0, mono=True):
    """Deterministic seeded additive grain -- breaks up flat gradients the way real film / sensor noise does. Seeded
    so the same frame always grains identically (the engine's determinism rule)."""
    img = np.asarray(img, float)
    rng = np.random.default_rng(seed)
    H, W = img.shape[:2]
    n = rng.standard_normal((H, W, 1) if mono else (H, W, 3))
    return np.clip(img + amount * n, 0.0, 1.0)


def resample(img, scale=2.0):
    """Bilinear resample: scale > 1 upscales, < 1 downscales. resample(0.5) then resample(2.0) is the building block
    of supersample anti-aliasing; upscaling is the cheap spatial-upscaler stand-in."""
    img = np.asarray(img, float)
    H, W = img.shape[:2]
    nH, nW = max(1, int(round(H * scale))), max(1, int(round(W * scale)))
    ys = np.linspace(0, H - 1, nH)
    xs = np.linspace(0, W - 1, nW)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    fy = (ys - y0)[:, None, None]
    fx = (xs - x0)[None, :, None]
    a = img[y0][:, x0]; b = img[y0][:, x1]; c = img[y1][:, x0]; d = img[y1][:, x1]
    return a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx + c * fy * (1 - fx) + d * fy * fx


def supersample(img, factor=2):
    """Anti-alias by downsampling a higher-res frame: average factor x factor blocks. Pass the frame you rendered at
    `factor` times the target resolution. (Cheap SSAA when you can afford to over-render.)"""
    img = np.asarray(img, float)
    H, W = img.shape[:2]
    f = int(factor)
    Hc, Wc = (H // f) * f, (W // f) * f
    return img[:Hc, :Wc].reshape(Hc // f, f, Wc // f, f, img.shape[2]).mean(axis=(1, 3))


# --------------------------------------------------------------------------------------------------------------
# The program: an ordered, named, serializable chain of effects
# --------------------------------------------------------------------------------------------------------------
EFFECTS = {
    "exposure": exposure, "reinhard": reinhard, "aces": aces, "gamma": gamma,
    "color_grade": color_grade, "vignette": vignette, "bloom": bloom, "glare": glare,
    "lens_flare": lens_flare, "chromatic_aberration": chromatic_aberration, "dof": dof,
    "motion_blur": motion_blur, "denoise": denoise, "sharpen": sharpen, "film_grain": film_grain,
    "resample": resample, "supersample": supersample,
}
_NEEDS_DEPTH = {"dof"}                                       # effects that read the depth buffer


# --------------------------------------------------------------------------------------------------------------
# KERNEL FUSION over the LINEAR, SHIFT-INVARIANT effects (backlog B1: postfx on the shader algebra)
# --------------------------------------------------------------------------------------------------------------
# `_fft_blur` already runs the engine's core operator: a 2-D circular convolution is `bind`, one dimension up. What
# it does NOT do is compose. A run of linear stages pays one forward and one inverse FFT EACH, when the composed
# operator is just the elementwise PRODUCT of their transfers -- diagonal operators commute and multiply. This is
# `holographic_shader.Pipeline`, in image space: compose the graph, then evaluate once.
#
# MEASURED (256x256x3, blur(2) -> denoise(1) -> blur(0.7)):
#     sequential, 3 FFT pairs   13.66 ms
#     fused, 1 FFT pair          4.95 ms      2.8x, max|diff| 4.44e-16
#
# THREE KEPT NEGATIVES, each measured, and the first is the one that decides how this ships:
#
#   1. THE SHIPPED CHAINS HAVE NO ADJACENT LINEAR STAGES. `default_chain` is exposure -> bloom -> aces ->
#      chromatic_aberration -> vignette -> film_grain -> gamma; `cinematic_chain` interleaves dof/glare/grade the
#      same way. Every blur is separated by a nonlinear tone curve. So `fuse=True` is a capability for chains that
#      HAVE such runs (a user's denoise -> sharpen, a multi-blur stack), not a speedup of the default chains. It
#      correctly does nothing to them, and a test pins that.
#
#   2. `sharpen` CLIPS INTERNALLY. `img + a*(img - blur)` is linear; `np.clip(..., 0, 1)` is not. Its transfer
#      (1+a) - a*G matches the unclipped math to 7.77e-16, but on a noisy frame the lower clamp fires on 4.5% of
#      pixels. Fusing DEFERS that clamp to the end of the run -- and the effect of deferring it is subtler than it
#      first looks, measured:
#
#          run [denoise, sharpen]   max|sequential - fused| = 1.33e-15   (exact)
#          run [sharpen, denoise]   max|sequential - fused| = 2.81e-01   (differs)
#
#      A deferred clamp only matters when the clamped stage is FOLLOWED by another stage inside the run: with
#      `sharpen` last, the chain's own final clip does the same job. So fusion is exact for the common
#      denoise->sharpen ordering and lossy for sharpen->denoise. Opt-in, and stated, not hidden.
#
#   3. BATCHING THE THREE CHANNELS INTO ONE FFT IS SLOWER, not faster: `rfft2(img, axes=(0,1))` on an (H,W,3)
#      array walks a non-contiguous stride. Measured 0.66x at 256^2 and 0.61x at 512^2, bit-identical output. The
#      per-channel loop stays. (Filed so nobody "optimizes" it later.)

_LINEAR_EFFECTS = ("denoise", "sharpen")     # linear + shift-invariant; `motion_blur`/`glare` clamp their edges


def _gaussian_transfer(shape, sigma):
    """The frequency-domain Gaussian `_fft_blur` multiplies by. Exposed so a transfer can be composed instead of
    applied. A Gaussian in space is a Gaussian in frequency -- there is no kernel truncation."""
    H, W = int(shape[0]), int(shape[1])
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    return np.exp(-2.0 * (np.pi ** 2) * (float(sigma) ** 2) * (fy * fy + fx * fx))


def linear_transfer(shape, name, params):
    """The transfer of ONE linear effect, or None if it is not linear and shift-invariant.

    `denoise(sigma)` -> G_sigma. `sharpen(amount, sigma)` -> (1+amount) - amount*G_sigma, WITHOUT its internal
    clamp (see kept negative 2). Everything else returns None: `motion_blur` and `glare` clamp their edges, so they
    are not shift-equivariant and have no exact transfer -- the same boundary-condition gate as the periodic-vs-
    Neumann heat operator. Refusing is the honest answer."""
    if name == "denoise":
        return _gaussian_transfer(shape, params.get("sigma", 1.0))
    if name == "sharpen":
        a = float(params.get("amount", 0.5))
        return (1.0 + a) - a * _gaussian_transfer(shape, params.get("sigma", 1.5))
    return None


def fuse_transfers(shape, steps):
    """Compose a run of linear steps into ONE transfer: the elementwise product of theirs. Diagonal operators
    commute and multiply, so the order within a fusable run does not change the composed operator (it does change
    where a deferred clamp would have fired -- see kept negative 2). Returns None if any step is not fusable."""
    T = None
    for name, params in steps:
        t = linear_transfer(shape, name, params)
        if t is None:
            return None
        T = t if T is None else T * t
    return T


def apply_transfer(img, T):
    """Evaluate a composed transfer on an image: ONE forward and ONE inverse FFT per channel, whatever the run
    length was. The per-channel loop is deliberate -- batching the channels into one FFT is measured 0.66x.

    DELEGATES to `holographic_shader.Pipeline(shape, real=True)` (backlog G8). It did not, until Pipeline learned
    the half-spectrum: its transfer used to live on the full `fftn` grid, which is a measured 2.2x LOSS on a real
    image, so postfx hand-composed its own rfft2 transfer and the duplication was filed as a DEFERRED unifier
    adoption. With `real=True` the delegation is BIT-IDENTICAL (max|diff| exactly 0.0e+00) and the same speed
    (1.02 ms vs 1.04 ms per 128x128x3 image). The silo is closed by generalizing the primitive, not by paying for
    the wrong spectrum."""
    from holographic.rendering.holographic_shader import Pipeline
    img = np.asarray(img, float)
    H, W = img.shape[:2]
    pipe = Pipeline.from_transfer((H, W), T, real=True)   # zero-overhead: the transfer is already composed
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = pipe.apply(img[:, :, c])
    return out


def fusable_runs(steps):
    """Split `steps` into [(is_linear, [steps...])] maximal runs. A run of length 1 is not worth fusing (it would
    pay exactly the FFT pair it already pays), so it is reported as non-linear."""
    runs = []
    for name, params in steps:
        lin = name in _LINEAR_EFFECTS
        if runs and runs[-1][0] == lin:
            runs[-1][1].append((name, params))
        else:
            runs.append((lin, [(name, params)]))
    return [(lin and len(grp) > 1, grp) for lin, grp in runs]


class PostChain:
    """An ordered, named, serializable post-processing PROGRAM -- the same shape as a HoloMachine instruction
    sequence. `steps` is a list of (effect_name, params). Build it fluently with .then(name, **params), compose two
    chains with +, serialize with to_list / from_list, and run it with .apply(image, depth=...)."""

    def __init__(self, steps=None):
        self.steps = [(n, dict(p)) for n, p in (steps or [])]

    def then(self, name, **params):
        if name not in EFFECTS:
            raise KeyError("unknown effect %r; known: %s" % (name, sorted(EFFECTS)))
        self.steps.append((name, params))
        return self                                         # chainable

    def apply(self, img, depth=None, fuse=False):
        """Run the program. `fuse=True` composes maximal RUNS of linear, shift-invariant stages (denoise, sharpen)
        into one transfer and evaluates them with a single FFT pair instead of one per stage -- exact to 4.44e-16
        on a pure run, 2.8x at three stages.

        DEFAULT-OFF. The one thing it changes: `sharpen`'s internal clamp is deferred to the end of the run. That
        only matters when a clamped stage is FOLLOWED by another stage inside the run -- measured, `denoise ->
        sharpen` is exact (1.33e-15) while `sharpen -> denoise` differs by 2.81e-01, because with `sharpen` last
        the chain's own final clip does the same job. The shipped chains have NO adjacent linear stages, so
        `fuse=True` is a bit-identical no-op on them -- correctly, and a test pins it."""
        out = np.asarray(img, float)
        if not fuse:
            for name, params in self.steps:
                fn = EFFECTS[name]
                out = fn(out, depth=depth, **params) if name in _NEEDS_DEPTH else fn(out, **params)
            return np.clip(out, 0.0, 1.0)

        for is_fused, group in fusable_runs(self.steps):
            if is_fused:
                T = fuse_transfers(out.shape[:2], group)
                out = apply_transfer(out, T) if T is not None else out
                continue
            for name, params in group:
                fn = EFFECTS[name]
                out = fn(out, depth=depth, **params) if name in _NEEDS_DEPTH else fn(out, **params)
        return np.clip(out, 0.0, 1.0)

    def to_list(self):
        """Serialize to a plain [(name, params), ...] list -- a program you can store, diff or version-stamp."""
        return [(n, dict(p)) for n, p in self.steps]

    @classmethod
    def from_list(cls, data):
        return cls(data)

    def __add__(self, other):
        return PostChain(self.steps + other.steps)          # compose two programs

    def __repr__(self):
        return "PostChain(" + " -> ".join(n for n, _ in self.steps) + ")"


def default_chain(seed=0):
    """A tasteful default preset that turns the raw linear buffer into a graded frame: lift exposure, bloom the
    highlights, ACES tonemap, a touch of chromatic aberration, vignette, fine grain, then gamma. Effects that work in
    HDR (exposure, bloom) come BEFORE the tonemap; display-space effects come after."""
    return (PostChain()
            .then("exposure", ev=0.3)
            .then("bloom", threshold=0.7, sigma=4.0, intensity=0.5)
            .then("aces")
            .then("chromatic_aberration", strength=0.003)
            .then("vignette", strength=0.35)
            .then("film_grain", amount=0.02, seed=seed)
            .then("gamma", g=2.2))


def cinematic_chain(depth_focus=None, seed=0):
    """A heavier 'cinematic' preset: depth of field + glare + warm grade. Needs a depth buffer for the DOF step."""
    return (PostChain()
            .then("exposure", ev=0.2)
            .then("dof", focus=depth_focus, aperture=2.5, max_sigma=5.0)
            .then("bloom", threshold=0.65, sigma=5.0, intensity=0.6)
            .then("glare", threshold=0.8, length=20, intensity=0.35, streaks=2)
            .then("aces")
            .then("color_grade", contrast=1.08, saturation=1.12, temperature=0.02)
            .then("vignette", strength=0.4)
            .then("film_grain", amount=0.02, seed=seed)
            .then("gamma", g=2.2))


def _selftest():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 1, (64, 64, 3))
    img[28:36, 28:36] = 3.0                                 # a bright HDR patch to drive bloom/glare

    # tonemap compresses the bright patch below 1
    assert aces(img).max() <= 1.0 and reinhard(img).max() < 1.0
    # bloom adds energy around the bright patch (a glow ring), so the neighbourhood gets brighter
    bl = bloom(img, threshold=1.0, sigma=4.0, intensity=0.8)
    ring = bl[24:40, 24:40].sum() - img[24:40, 24:40].sum()
    assert ring > 0.0, "bloom should add glow energy"
    # vignette darkens the corners more than the centre
    v = vignette(np.ones((64, 64, 3)), strength=0.6)
    assert v[0, 0].mean() < v[32, 32].mean()
    # gamma on linear 0.5 brightens it (display-encode)
    assert gamma(np.full((2, 2, 3), 0.5))[0, 0, 0] > 0.5
    # film grain is deterministic for a fixed seed, and changes for another
    g0 = film_grain(img, seed=1); g0b = film_grain(img, seed=1); g1 = film_grain(img, seed=2)
    assert np.array_equal(g0, g0b) and not np.array_equal(g0, g1)
    # dof blurs far depths and leaves the focus plane sharp
    depth = np.zeros((64, 64)); depth[:, 32:] = 5.0
    df = dof(np.tile(np.linspace(0, 1, 64), (64, 1))[:, :, None].repeat(3, 2), depth=depth, focus=0.0, aperture=2.0)
    assert df.shape == (64, 64, 3)
    # resample changes size and supersample averages it back down
    up = resample(img, 2.0); assert up.shape[0] == 128
    ss = supersample(up, 2); assert ss.shape[0] == 64
    # the chain is a serializable program and runs end to end
    ch = default_chain()
    out = ch.apply(img)
    assert out.shape == img.shape and out.max() <= 1.0 and out.min() >= 0.0
    assert PostChain.from_list(ch.to_list()).to_list() == ch.to_list()
    assert (ch + PostChain().then("sharpen")).steps[-1][0] == "sharpen"
    print("postfx selftest ok: tonemap/bloom/vignette/gamma/grain/dof/resample + PostChain program all behave; "
          "default_chain ->", repr(ch))


if __name__ == "__main__":
    _selftest()
