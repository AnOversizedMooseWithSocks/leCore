"""holographic_shader.py -- N filter passes in ONE evaluation. Two things a GPU structurally cannot do.

WHY THIS EXISTS (H6)
--------------------
A GPU blurs by running the kernel again, and again. N passes cost N passes. But a circular convolution is a `bind`,
and a bind is DIAGONAL in the Fourier basis -- so applying it N times just raises each frequency's transfer to the
N-th power. Compute `H**N` once, multiply, inverse-transform. The cost is independent of N.

    filter_k(field, kernel, N)      # N passes, one evaluation

MEASURED (a 1024-sample field, 3-tap circular blur), against literally running the kernel N times:

    N          iterative      closed form     speedup      max abs diff
    16           637 us            61 us        10.4x         1.8e-15
    256        8,024 us           157 us        51.2x         5.5e-15
    4096     130,289 us            71 us     1,824.0x         2.3e-14
    1,000,000   (~9 hours)          65 us         --            --

Two consequences with no GPU analogue:

  * **N may be FRACTIONAL.** "Half a blur pass" is well defined for a diagonal operator: `H**0.5`. Compose it twice
    and you get exactly one pass (verified). A GPU cannot run half a pass.
  * **N may be INFINITE.** The steady state is a PROJECTION onto the modes with |transfer| >= 1: everything that
    decays is gone, everything that persists remains. `filter_limit` returns it in O(D), with no iteration at all.

HONEST SCOPE, and it matters:
  * The kernel must be a CIRCULAR convolution (a periodic domain). A clamped or mirrored border is not circular and
    is not diagonal in the DFT -- the same boundary condition that keeps `iterate` out of the Neumann PDE solvers.
  * A fractional N raises a complex number to a fractional power, which needs a branch cut. For a SMOOTHING kernel
    the transfer is real and non-negative and there is no ambiguity; for a kernel whose transfer goes negative (a
    sharpening kernel, say) the fractional power is genuinely ambiguous, so we RAISE rather than silently pick a
    branch. Integer N is always fine.
  * If any |transfer| > 1 the filter AMPLIFIES, and `H**N` overflows for large N exactly as the iterative loop
    would. We warn on real growth rather than return silent `inf`.

THE REST OF THE ALGEBRA
-----------------------
The same "a linear operator is diagonal in the phasor basis" argument, pushed in three more directions:

  * **H1, the compiler** (`Pipeline`). A whole post-process GRAPH -- blur, translate, gain, unsharp -- composes
    ALGEBRAICALLY before any data is touched, into one transfer applied in one FFT / multiply / inverse FFT.
    Compose the OPERATORS, not the images: materialising an intermediate image takes a real part, and a half-sample
    shift's (genuinely imaginary) Nyquist component vanishes with it -- 9.3e-2 of signal, silently.
  * **H3, the texture unit** (`bake_1d` / `fetch`). A sampled function lives in ONE hypervector; a fetch at any x is
    one dot product, with interpolation built into the algebra. THE ALGEBRA HAS A NYQUIST -- below the signal's
    maximum angular frequency the bake returns a confident, smooth-looking, wrong answer -- so the bandwidth is
    probed from the data. The raw fetch is a kernel SUM whose gain is the sample count; `normalize=True` makes it a
    kernel average.
  * **H2, the superposed gather** (`gather_rule` / `gather` / `translate_rule`). N weighted lookups compile into ONE
    query vector; applying it is one dot product, EXACT against running the lookups separately, with no sqrt(N/D)
    crosstalk wall (a gather never unbinds). Binding SLIDES the whole rule, at a cost independent of N -- a
    grid-free convolution.

numpy only; deterministic; any number of dimensions.
"""
import warnings

import numpy as np

_GROWTH_LIMIT = 1e12


def transfer(kernel, shape=None):
    """The kernel's eigenvalues: its (n-D) FFT. A circular convolution is diagonal in this basis, so this IS the
    eigendecomposition -- free, no dense O(n^3) work.

    `axes` is passed explicitly whenever `s` is: NumPy 2.0 deprecates `axes=None` with a non-None `s`, and in a
    future release `s[i]` will bind to `axes[i]` instead of to the leading axes. Silent re-binding of a transfer's
    axes is exactly the kind of change that would not raise and would not be caught."""
    k = np.asarray(kernel, float)
    if shape is None:
        return np.fft.fftn(k)
    axes = tuple(range(len(np.atleast_1d(shape))))
    return np.fft.fftn(k, s=tuple(shape), axes=axes)


def _check(H, n_passes):
    mag = np.abs(H)
    if mag.max() > 1.0 + 1e-9:
        with np.errstate(over="ignore"):
            growth = float(np.float_power(np.float64(mag.max()), np.float64(abs(n_passes))))
        if not np.isfinite(growth) or growth > _GROWTH_LIMIT:
            warnings.warn("filter_k: kernel amplifies (max|transfer| = %.4f > 1); %g passes grow the field by ~%.1e "
                          "and will overflow, exactly as the iterative loop would."
                          % (mag.max(), n_passes, growth), RuntimeWarning, stacklevel=3)


def filter_k(field, kernel, n_passes):
    """Apply `n_passes` of a circular convolution in ONE evaluation. `n_passes` may be any real number.

    Integer N reproduces N literal passes to FFT tolerance (measured 2.3e-14 at N=4096) and costs the same whether
    N is 1 or 1,000,000. Fractional N is well defined for a smoothing kernel (non-negative real transfer); for a
    kernel whose transfer changes sign the fractional power has no canonical branch and this raises."""
    f = np.asarray(field, float)
    H = transfer(kernel, shape=f.shape)
    n = float(n_passes)
    fractional = abs(n - round(n)) > 1e-12
    if fractional:
        # a real, non-negative transfer has an unambiguous fractional power; anything else does not
        if np.max(np.abs(H.imag)) > 1e-9 or np.min(H.real) < -1e-9:
            raise ValueError("fractional passes need a smoothing kernel (real, non-negative transfer); this kernel's "
                             "transfer changes sign or is complex, so H**%.3g has no canonical branch" % n)
        Hn = np.power(np.maximum(H.real, 0.0), n)
    else:
        _check(H, n)
        Hn = H ** int(round(n))
    return np.real(np.fft.ifftn(np.fft.fftn(f) * Hn))


def filter_limit(field, kernel, tol=1e-6):
    """The N -> infinity steady state, in closed form: a PROJECTION onto the modes the filter does not decay.

    Modes with |transfer| < 1 vanish; modes with |transfer| ~ 1 survive untouched. Idempotent by construction (it is
    a 0/1 mask), so it really is a projection -- unlike raising the transfer to a large power, which keeps rescaling
    the survivors. Raises if the filter amplifies any mode, where no finite limit exists."""
    f = np.asarray(field, float)
    H = transfer(kernel, shape=f.shape)
    mag = np.abs(H)
    if mag.max() > 1.0 + tol:
        raise ValueError("filter amplifies (max|transfer| = %.4f > 1): the infinite-pass limit does not exist"
                         % mag.max())
    keep = (mag >= 1.0 - tol).astype(float)              # a 0/1 mask -> idempotent -> a true projection
    return np.real(np.fft.ifftn(np.fft.fftn(f) * keep))


def blur_kernel(shape, taps=(0.25, 0.5, 0.25), axis=0):
    """A separable circular blur kernel of the given `shape`, laid out for `filter_k` (taps centred on index 0)."""
    k = np.zeros(shape, float)
    idx = [0] * len(k.shape) if isinstance(shape, tuple) else [0]
    mid = len(taps) // 2
    for t, c in enumerate(taps):
        off = t - mid
        pos = list(idx)
        pos[axis] = off % k.shape[axis]
        k[tuple(pos)] += c
    return k


def gauss_kernel(n, sigma):
    """A 1-D circular Gaussian blur kernel of length `n` and width `sigma`, centred on index 0.

    The natural way to build a variant STACK -- a ladder of sigmas is an LOD chain, a multi-scale filter, or a
    parameter sweep -- which `combine` then folds into a single transfer."""
    xs = np.arange(int(n))
    xs = np.minimum(xs, int(n) - xs)                        # circular distance from index 0
    k = np.exp(-0.5 * (xs / float(sigma)) ** 2)
    return k / k.sum()


def _phasor_key(n, seed):
    """A unit-modulus, conjugate-symmetric spectral key: FHRR binding, exact to unbind.

    Private on purpose. It exists ONLY to pin the H7 kept negative in the self-test -- superposing shader variants
    under distinct keys so they can be unbound back out. The measurement says do not build on it; see the H7 block
    below `Pipeline`."""
    r = np.random.default_rng(seed)
    ph = r.uniform(-np.pi, np.pi, int(n))
    ph[0] = 0.0
    for j in range(1, int(n) // 2 + 1):
        ph[int(n) - j] = -ph[j]
    if int(n) % 2 == 0:
        ph[int(n) // 2] = 0.0
    return np.exp(1j * ph)


# ==================================================================================================================
# H1 -- THE COMPILER. Every stage of a filter graph -- a blur, a translation, a gain, an unsharp blend -- is LINEAR
# and SHIFT-INVARIANT. In the Fourier basis each is a multiplication, so the WHOLE GRAPH collapses into ONE transfer
# function, computed before any data is touched. A GPU runs three passes over the image; we run one multiply.
#
#     pipe = Pipeline(shape).blur(k, 8).translate(3).unsharp(k_wide, 0.6)
#     out  = pipe.apply(field)          # one FFT, one multiply, one inverse FFT -- for a 3-stage graph
#
# MEASURED (1024-sample field, blur x8 -> translate 3 -> unsharp): the composed transfer reproduces the staged
# computation to 6.7e-16, compiles once in ~1.1 ms, and then each application costs 34 us against the staged 205 us
# -- 6.0x, and the gap grows with the number of stages, because the compiled cost does not depend on it at all.
#
# Bonus with no GPU analogue: `translate` accepts a FRACTIONAL shift (a phase ramp is exact at any offset), and
# `blur` accepts a fractional number of passes.
# ==================================================================================================================
class Pipeline:
    """A filter graph, composed ALGEBRAICALLY into one transfer before any query is issued.

    Every method returns `self`, so stages chain. `transfer` is the composed operator; `apply(field)` runs it. All
    stages must share the field's `shape`, because a transfer is defined on a fixed grid."""

    def __init__(self, shape, real=False):
        """`real=True` builds the pipeline on the HALF-SPECTRUM (`rfftn`), which is what a real-valued field wants
        (backlog G8). The transfer then has shape `shape[:-1] + (shape[-1]//2 + 1,)`, and `apply` uses
        `irfftn(rfftn(f) * H)`.

        WHY IT MATTERS. `Pipeline`'s original transfer lives on the FULL `fftn` grid, which costs ~2x on a real
        image: measured 3.48 ms vs 1.00 ms for one 256x256 round trip. That 2.2x is precisely why `postfx`
        hand-composed its own rfft2 transfer instead of delegating here. Both modes give identical results (7.8e-16
        on a real field); `real=True` is the one to use when the input is real, and it is default-OFF so every
        existing pipeline is bit-identical."""
        self.shape = tuple(int(n) for n in np.atleast_1d(shape))
        self.real = bool(real)
        self.axes = tuple(range(len(self.shape)))               # explicit: numpy deprecates axes=None with s=
        self.tshape = (self.shape[:-1] + (self.shape[-1] // 2 + 1,)) if self.real else self.shape
        self.transfer = np.ones(self.tshape, dtype=complex)     # identity

    @classmethod
    def from_transfer(cls, shape, H, real=False):
        """A Pipeline whose transfer IS `H` -- no identity array, no composing multiply. The zero-overhead entry
        point for a caller that already holds a composed transfer, which is what `postfx.apply_transfer` does. The
        long way round (`Pipeline(shape).stage(H)`) allocates a complex `ones` and multiplies it: measured, that
        turned a 1.04 ms call into 1.48 ms, a 42% regression paid for nothing."""
        p = cls(shape, real=real)
        H = np.asarray(H)
        if H.shape != p.tshape:
            raise ValueError("transfer shape %s does not match the pipeline's %s (real=%s)"
                             % (H.shape, p.tshape, p.real))
        p.transfer = H.astype(complex, copy=False)
        return p

    def _fwd(self, f):
        return np.fft.rfftn(f, axes=self.axes) if self.real else np.fft.fftn(f, axes=self.axes)

    def _inv(self, F):
        if self.real:
            return np.fft.irfftn(F, s=self.shape, axes=self.axes)
        return np.real(np.fft.ifftn(F, axes=self.axes))

    def _freq(self, axis):
        """The frequency grid for `axis` -- HALVED on the last axis when `real=True`, because that is where rfftn
        folds the conjugate-symmetric half away."""
        if self.real and axis == len(self.shape) - 1:
            return np.fft.rfftfreq(self.shape[axis])
        return np.fft.fftfreq(self.shape[axis])

    def _kernel_transfer(self, kernel, n_passes=1):
        H = (np.fft.rfftn(np.asarray(kernel, float), s=self.shape, axes=self.axes)
             if self.real else transfer(kernel, shape=self.shape))
        n = float(n_passes)
        if abs(n - round(n)) > 1e-12:
            if np.max(np.abs(H.imag)) > 1e-9 or np.min(H.real) < -1e-9:
                raise ValueError("fractional passes need a smoothing kernel (real, non-negative transfer)")
            return np.power(np.maximum(H.real, 0.0), n).astype(complex)
        return H ** int(round(n))

    def blur(self, kernel, n_passes=1):
        """Convolve with `kernel`, `n_passes` times (any real number -- see filter_k)."""
        self.transfer = self.transfer * self._kernel_transfer(kernel, n_passes)
        return self

    def translate(self, shift):
        """Circularly translate by `shift` samples along each axis. A phase ramp, so a FRACTIONAL shift is exact --
        sub-sample translation with no resampling filter and no GPU analogue.

        CAUTION, measured: a half-sample shift puts a genuinely IMAGINARY component in the Nyquist bin. Composing
        two half-shifts inside a Pipeline is exact (8.9e-16), but applying one, taking the real part, and applying
        the other loses 9.3e-2 of signal. Compose the operators; do not materialise the intermediate."""
        shifts = np.atleast_1d(np.asarray(shift, float))
        if shifts.size == 1:
            shifts = np.repeat(shifts, len(self.shape))
        ramp = np.ones(self.tshape, dtype=complex)
        for axis, d in enumerate(shifts):
            w = 2.0 * np.pi * self._freq(axis)
            shape = [1] * len(self.shape)
            shape[axis] = self.tshape[axis]
            ramp = ramp * np.exp(-1j * w * float(d)).reshape(shape)
        self.transfer = self.transfer * ramp
        return self

    def gain(self, scale):
        """Multiply by a constant."""
        self.transfer = self.transfer * float(scale)
        return self

    def unsharp(self, kernel, alpha=0.5):
        """Unsharp mask against `kernel`: out = (1 + a)*x - a*blur(x). A LINEAR COMBINATION of two branches, so it
        stays a single transfer -- which is why a whole graph, not just a chain, collapses to one multiply."""
        K = (np.fft.rfftn(np.asarray(kernel, float), s=self.shape, axes=self.axes)
             if self.real else transfer(kernel, shape=self.shape))
        self.transfer = (1.0 + alpha) * self.transfer - alpha * self.transfer * K
        return self

    def stage(self, transfer):
        """Compose an arbitrary PRECOMPUTED transfer into the graph. The general injection point: any operator that
        is diagonal in the Fourier basis has one, and multiplying transfers is how they compose.

        This is what lets a non-graphics operator join the algebra without pretending to be a blur kernel. The heat
        equation's exact solution on a periodic domain is `exp(-alpha |k|^2 t)` -- a transfer -- and
        `holographic_laplacian.diffusion_operator` builds a Pipeline from it. The transfer must match the field's
        shape, because a transfer is defined on a fixed grid."""
        H = np.asarray(transfer)
        if H.shape != self.tshape:
            raise ValueError("transfer shape %s does not match the pipeline's %s (real=%s)"
                             % (H.shape, self.tshape, self.real))
        self.transfer = self.transfer * H.astype(complex)
        return self

    def apply(self, field):
        """Run the compiled graph: one FFT, one multiply, one inverse FFT -- however many stages it has."""
        f = np.asarray(field, float)
        if f.shape != self.shape:
            raise ValueError("field shape %s does not match the pipeline's %s" % (f.shape, self.shape))
        return self._inv(self._fwd(f) * self.transfer)


def combine(pipelines, weights=None):
    """H7, the half that survived measurement: blend M shader variants into ONE transfer, exactly.

    An LOD stack, a multi-scale filter, an MIS-weighted combination, a parameter sweep you intend to average -- any
    FIXED linear combination sum_j w_j * pipe_j -- is itself linear and shift-invariant, so the transfers simply add.
    `combine` returns a Pipeline you can keep chaining. Measured against staging the M pipelines and blending their
    output images: identical to 2.2e-16, and the cost does not depend on M at all --

        M           staged        combined     speedup
        4           0.99 ms        0.229 ms       4.3x
        16          2.17 ms        0.233 ms       9.3x
        64          8.66 ms        0.289 ms      30.0x

    -- so it gets CHEAPER, relatively, the more variants you have. This is `unsharp` generalised: unsharp is the
    two-branch case, and the reason a filter GRAPH, not merely a chain, collapses to one multiply.

    KEPT NEGATIVE, and it is the rest of H7 (see the block below `filter_limit`): this works because the weights are
    FIXED and the variants are only ever SUMMED. Superposing M variants under distinct keys so you can unbind any
    one of them back out does NOT work, and the measurement is unambiguous. Do not build it."""
    pipes = list(pipelines)
    if not pipes:
        raise ValueError("combine: no pipelines given")
    shape = pipes[0].shape
    if any(p.shape != shape for p in pipes):
        raise ValueError("combine: all pipelines must share one shape, got %s" % ([p.shape for p in pipes],))
    w = np.ones(len(pipes)) if weights is None else np.atleast_1d(np.asarray(weights, float))
    if w.size != len(pipes):
        raise ValueError("combine: %d pipelines but %d weights" % (len(pipes), w.size))
    out = Pipeline(shape)
    out.transfer = sum(float(wj) * p.transfer for wj, p in zip(w, pipes))    # linearity: the transfers just add
    return out


# ==================================================================================================================
# H7 -- KEPT NEGATIVE: a superposed BANK of shader variants does not work, and would not pay if it did.
#
# The plan (backlog Tier 4) was: role-bind each of M shader variants, bundle them, apply the bundle to the field
# ONCE, then unbind to read any variant's output -- "a GPU runs M passes; we run one and unbind" -- with crosstalk
# budgeted by the familiar sqrt(M/D) law. All three parts of that are wrong, measured at D = 8192.
#
#   (1) FIDELITY FOLLOWS 1/sqrt(M), NOT sqrt(M/D). Unbinding a bundle of M keyed items returns the item plus M-1
#       random vectors, so the recovered vector's cosine with the truth is ~1/sqrt(M). Measured on UNCORRELATED
#       variants, it matches to three digits:
#
#           M                2       4       8      16      32
#           cos(rec, true)  .712    .507    .353    .249    .177
#           1/sqrt(M)       .707    .500    .354    .250    .177
#
#       sqrt(M/D) (0.016 at M=2) is a different quantity entirely -- the cosine with a WRONG item, i.e. a confusion
#       measure. It was read as a fidelity budget. The bank was never within two orders of magnitude of usable.
#
#   (2) REAL SHADER VARIANTS ARE CORRELATED, WHICH MAKES IT WORSE, NOT BETTER. M blurs of the SAME field are M
#       filtered copies of one signal, not M independent items: measured mean |cos| between distinct variants' true
#       outputs of 0.487 at M=2. Cleanup -- the discrete decision that normally RESETS crosstalk -- needs a
#       near-orthogonal codebook, and has none here. Selecting which variant produced a given output collapses:
#
#           M                     2      4      8     16     32     64
#           bank-select accuracy .60    .33    .20    .17    .07    .03
#           direct comparison   1.00   1.00   1.00   1.00   1.00   1.00
#
#   (3) THERE WAS NO COST TO SAVE. The bank still needs M inverse transforms to read M outputs -- the "M passes" a
#       GPU runs are the M READOUTS, not the M multiplies -- so it measured 0.83-0.87x the direct path (i.e.
#       slower) at M = 4, 16, 64. The cost model that motivated the item does not exist.
#
# THE RULE THIS LEAVES BEHIND, which is worth more than the item was: superposition buys width only when the items
# are near-orthogonal AND a cleanup follows the readout. Shader variants are neither. What you CAN do in one pass is
# any fixed linear combination of them -- `combine` above -- because that never unbinds anything: exact to 2.2e-16,
# and 30x faster at M = 64. This is the same dividing line the superposed GATHER (H2) sits on the good side of.
# See holographic_superposed for the regime where a keyed bundle is the right tool.
# ==================================================================================================================


# ==================================================================================================================
# H3 -- THE ALGEBRA HAS A NYQUIST.
#
# The "texture unit" bakes a function into ONE hypervector: F = sum_i f(x_i) * Z(x_i), and a fetch is a single dot
# product with Z(x). What decides whether that works is the phasor BANDWIDTH B of the encoder: it sets how fast the
# similarity kernel falls off, i.e. the finest detail the code can resolve. Set B too low and the bake does not
# merely blur -- it returns garbage, silently, with no error raised anywhere.
#
# MEASURED (a sine of frequency f baked from 240 samples, D=4096, scale-free RMS error of the fetch):
#
#       f      w_max = 2*pi*f     B = 0.5*w      B = 1.0*w     B = 1.5*w
#      2.0          12.6            0.298          0.046         0.018
#      5.0          31.4            0.198          0.047         0.034
#      8.0          50.3            0.085          0.042         0.042
#     12.0          75.4            0.102          0.060         0.042
#
# So the law is B >= w_max, and B ~ 1.5 * w_max is the sweet spot -- past that it stops improving and eventually
# costs capacity. `bandwidth_probe` measures w_max from the data (this is the descriptor's "variation" probe, in the
# one form that is actually checkable), and `bake_1d` sets B from it instead of leaving it to a default.
# ==================================================================================================================
def _is_uniform(x, tol=1e-6):
    """Are these sample positions evenly spaced? `bandwidth_probe` is an FFT and only means anything if they are.

    On scattered samples the FFT reads the spacing jitter as high-frequency content: measured w_max = 100.8 for a
    signal whose true w_max is 18.8. Silently baking with that number costs capacity and blames the data."""
    d = np.diff(np.asarray(x, float))
    if d.size == 0:
        return True
    span = float(np.abs(d).max())
    return span <= 0 or float(np.abs(d - d.mean()).max()) <= tol * span


def bandwidth_probe(xs, ys, energy=0.995):
    """The maximum ANGULAR frequency w_max carrying `energy` of the signal's power -- what the bake must resolve.

    Requires uniformly spaced `xs` (it is an FFT). Returns w_max in radians per unit of x, so a sine of frequency f
    returns ~2*pi*f. This is the checkable core of the descriptor's `variation` probe: not "how rough does this
    look" but "what is the highest frequency I must not throw away"."""
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    if x.size < 4:
        return 0.0
    span = float(x[-1] - x[0])
    if span <= 0:
        return 0.0
    spec = np.abs(np.fft.rfft(y - y.mean())) ** 2
    if spec.sum() <= 0:
        return 0.0
    freqs = np.fft.rfftfreq(y.size, d=span / (y.size - 1))       # cycles per unit x
    keep = np.searchsorted(np.cumsum(spec) / spec.sum(), energy)
    f_max = float(freqs[min(keep, freqs.size - 1)])
    return 2.0 * np.pi * f_max                                   # radians per unit x


# ==================================================================================================================
# H4 -- DETREND BEFORE YOU BAKE.  The backlog said "near-singular functions need domain warping" (a raw sqrt LUT
# measured 0.125). Measurement says the diagnosis was wrong, and so was the prescription.
#
# `bandwidth_probe` is a plain FFT, and an FFT treats its samples as PERIODIC. Any function whose endpoints disagree
# therefore carries an implicit JUMP at the wrap, and a jump has an unbounded spectrum. So the probe reports a huge
# w_max for functions that are not remotely high-frequency:
#
#       f(x) on [0,1], probed from 400 uniform samples    w_max raw    w_max after removing the endpoint line
#       sqrt(x)                                              789.70                     68.94
#       x               (a straight line!)                   607.95                      0.00
#       1/(x + 0.05)                                        1002.80                     56.41
#       sin(2*pi*2*x)   (already periodic)                    12.53                     12.53
#
# A STRAIGHT LINE probes at the same bandwidth as sqrt. It was never the singularity; it was the wrap. And a huge B
# does real damage: the RBF kernel collapses toward a delta, so the bake stops interpolating and starts reading back
# whichever sample happens to be nearest.
#
# THE FIX is the classical one from spectral methods: subtract the line joining the endpoints, bake the RESIDUAL
# (which is genuinely band-limited and periodic-looking), and add the line back at fetch time -- where it is exact,
# analytic, and costs no vector capacity at all.
#
# RE-MEASURED, with the protocol stated and the spread reported (the first table shipped here was a SINGLE SEED, and
# a single seed is not a measurement -- see the methods negative at the bottom). Protocol: 400 uniform samples on
# [0,1], D=4096, read at 61 points in [0.05,0.95] with normalize=True; ABSOLUTE relative error, no fitted constant;
# mean +- sd over 12 encoder seeds, with a bootstrap 95% CI on the mean.
#
#       f(x)              plain bake                    DETRENDED                    ratio
#       sqrt(x)           0.1105 +- 0.0379 [.090,.133]  0.0087 +- 0.0051 [.006,.012]  12.6x
#       x**(1/3)          0.1404 +- 0.0887 [.096,.195]  0.0170 +- 0.0062 [.014,.021]   8.3x
#       1/(x + 0.05)      1.8278 +- 4.2475 [.397,4.45]  0.1157 +- 0.0606 [.086,.152]  15.8x
#       exp(3x)           0.2225 +- 0.0638 [.188,.260]  0.0214 +- 0.0106 [.015,.028]  10.4x
#       x (a line)        0.1330 +- 0.0604 [.098,.167]  0.0000 +- 0.0000               exact
#       sin(2*pi*2*x)     0.2064 +- 0.0413 [.184,.230]  0.2064 +- 0.0413 [.184,.230]   1.0x
#
# Read that table carefully, because it says four separate things:
#   * DETRENDING IS THE LEVER, buying 8-16x across the board and turning a line into an EXACT reconstruction (the
#     residual is identically zero, so the analytic trend alone does the whole job). The backlog prescribed domain
#     warping instead, which buys ~1.9x -- and only when the warp happens to LINEARIZE the function. On
#     1/(x+0.05), where it does not, warping buys nothing over detrending.
#   * THE PLAIN BAKE IS NOT JUST WRONG, IT IS UNSTABLE. Look at 1/(x+0.05): 1.83 +- 4.25, a 95% CI spanning an
#     order of magnitude, occasionally 4x WORSE than predicting the mean. Its seed-to-seed coefficient of variation
#     is 2.32 against the detrended bake's 0.52. That instability IS the mechanism: an inflated B collapses the RBF
#     kernel toward a delta, so the fetch stops interpolating and starts returning whichever sample lands nearest,
#     and which sample that is depends on the random phases. Any single-seed number here is a lottery ticket.
#   * DETRENDING NEVER HURTS. On an already-periodic function the trend is zero and nothing changes (sin, 1.0x).
#     That is why it is safe to recommend, and still safe to leave default-off for backward compatibility.
#   * DIMENSION ONLY HELPS THE DETRENDED BAKE. sqrt at D = 1k / 4k / 16k: plain 0.199 / 0.113 / 0.069, detrended
#     0.026 / 0.006 / 0.006. You cannot buy your way out of a bad bandwidth with dimension.
#
# KEPT NEGATIVE, retired and replaced: "a raw baked LUT of sqrt measured 0.125 -- near-singular functions need
# DOMAIN WARPING". Wrong cause (the wrap, not the singularity), wrong fix (warping, not detrending). The new
# negative in its place: a plain bake of ANY non-periodic function is silently wrong, and the probe will not tell
# you, because the probe is the thing that is confused.
#
# METHODS NEGATIVE, kept loud because it was mine: the table above originally shipped as six single-seed numbers
# (sqrt 0.3280, cube root 4.4871). None of them reproduces -- they all sit outside the 95% CI of the quantity they
# claimed to measure. The DIRECTIONS were right and the conclusion was right, which is exactly why nobody caught it.
# The engine's own rule ("every claim has a baseline, a variance estimate, and its negatives kept loud") applies to
# the docstrings, not just to the results. A number without a spread is an anecdote.
# ==================================================================================================================
def bake_1d(xs, ys, dim=4096, seed=0, margin=1.5, bandwidth=None, detrend=False):
    """Bake a sampled 1-D function into ONE hypervector, with the bandwidth chosen FROM THE DATA.

    Returns {"encoder", "field", "density", "bandwidth", "omega_max", "trend"}. Fetch any point with
    `fetch(bake, x)` -- a single dot product, at any x, whether or not it was sampled.

    `bandwidth=None` (the default) sets B = margin * w_max from `bandwidth_probe`. Pass B yourself and you get a
    WARNING if it is below w_max, because that failure is silent otherwise: the fetch returns a confident,
    smooth-looking, wrong answer.

    `detrend=True` subtracts the straight line joining the endpoints, bakes the RESIDUAL, and restores the line
    analytically at fetch time. Turn it on for any function that is not already periodic -- see the block above:
    a plain bake of `sqrt` scores 0.328 absolute relative error and of `x**(1/3)` scores 4.49 (worse than predicting
    the mean), because the probe reads the endpoint mismatch as a jump discontinuity and returns a wildly inflated
    bandwidth. Detrended: 0.017 and 0.069. It is default-OFF only for backward compatibility, and it costs nothing
    when the endpoints already agree. A detrended bake must be read with `normalize=True` (the trend is an absolute
    offset, and a raw kernel sum has no absolute scale).

    "density" is the SAME bake of the constant function 1 -- sum Z(x_i) with no y weighting. It costs nothing extra
    (the encodings are already in hand) and it is what `fetch(..., normalize=True)` divides by. See `fetch` for why
    that division is not optional once your samples are not uniformly spaced.

    PRECONDITION, and it is easy to miss: `bandwidth_probe` is an FFT, so it only means anything on UNIFORMLY spaced
    `xs`. On scattered samples it reads the spacing jitter as high-frequency content and over-reports w_max (measured
    100.8 against a true 18.8), which then costs capacity. Bake scattered data with an explicit `bandwidth` taken
    from a uniform proxy of the same signal."""
    from holographic.io_and_interop.holographic_encoders import ScalarEncoder
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    trend = None
    if detrend:
        span_x = float(x[-1] - x[0])
        slope = (float(y[-1]) - float(y[0])) / span_x if span_x != 0 else 0.0
        trend = (slope, float(y[0]) - slope * float(x[0]))       # the line through the endpoints -- the wrap's jump
        y = y - (trend[0] * x + trend[1])                        # ...bake only what is left, which is band-limited
    uniform = _is_uniform(x)
    omega_max = bandwidth_probe(x, y) if uniform else 0.0    # an FFT on scattered xs measures the jitter, not f
    if bandwidth is None:
        if not uniform:
            warnings.warn("bake_1d: the samples are not uniformly spaced, so the bandwidth cannot be probed (the "
                          "probe is an FFT -- on scattered samples it reads the spacing jitter as high-frequency "
                          "content and over-reports w_max). Pass an explicit `bandwidth`, taken from a uniform "
                          "proxy of the same signal.", RuntimeWarning, stacklevel=2)
        bandwidth = max(1.0, float(margin) * omega_max)
    elif omega_max > 0 and bandwidth < omega_max:
        warnings.warn("bake_1d: bandwidth %.2f is below the signal's maximum angular frequency %.2f. The algebra "
                      "has a Nyquist: below it the bake does not blur, it returns garbage -- and raises nothing. "
                      "Leave bandwidth=None to have it chosen from the data." % (bandwidth, omega_max),
                      RuntimeWarning, stacklevel=2)
    enc = ScalarEncoder(int(dim), float(x.min()), float(x.max()), seed=seed, kernel="rbf", bandwidth=float(bandwidth))
    field = np.zeros(int(dim))
    density = np.zeros(int(dim))                  # the SAME bake of f(x) = 1: the kernel weight that landed at x
    for xi, yi in zip(x, y):
        z = enc.encode(xi)
        field += yi * z
        density += z
    return {"encoder": enc, "field": field, "density": density, "trend": trend,
            "bandwidth": float(bandwidth), "omega_max": float(omega_max)}


def fetch(bake, x, normalize=False):
    """Query a baked field at any point: ONE dot product. Scalar or array `x`.

    `normalize=False` (the default, and the historical behaviour) returns the raw kernel SUM,
    <F, Z(x)> = sum_i y_i k(x_i - x). That is proportional to f(x), but only PROPORTIONAL: the constant is the total
    kernel weight that landed near x, so it scales with how densely you sampled. MEASURED on a uniform bake of the
    same function: the implied scale is 7.8 / 15.6 / 31.2 / 62.4 for 100 / 200 / 400 / 800 samples -- exactly linear
    in the sample count. Read a raw fetch as f(x) and you are off by a factor nobody wrote down.

    `normalize=True` divides by the density bake, giving the kernel AVERAGE (Nadaraya-Watson):
    <F, Z(x)> / <D, Z(x)>. It needs no fitted constant and it is what you want whenever the samples are not evenly
    spaced. MEASURED, samples clumped 3:1 into the first third of the domain: the raw fetch (even rescaled by a
    constant fitted against the true function) lands at 1.229 RMS -- worse than predicting the mean -- while the
    normalized fetch lands at 0.283 with nothing fitted at all.

    KEPT NEGATIVE, loud: normalizing is not free. On a uniform bake the denominator carries its own bake error, and
    dividing two noisy numbers compounds them: 0.046 RMS with a constant fitted from the truth, 0.083 normalized. If
    your sampling is uniform AND you have an independent way to know the scale, the raw fetch is the more accurate
    of the two. It is the *scattered* case where normalizing stops being a nicety.

    On a DETRENDED bake (see `bake_1d(detrend=True)`) the field holds only the residual, and the endpoint line is
    added back here analytically -- exact, and costing no vector capacity. That requires `normalize=True`, because
    the line is an absolute offset and a raw kernel sum has no absolute scale; asking for a raw fetch raises."""
    enc, F = bake["encoder"], bake["field"]
    trend = bake.get("trend")
    if trend is not None and not normalize:
        raise ValueError("fetch: this bake was detrended, so the field holds only the residual. The endpoint line is "
                         "an ABSOLUTE offset and cannot be added to a raw kernel sum, whose scale is the sample "
                         "density. Call fetch(bake, x, normalize=True).")
    xs = np.atleast_1d(np.asarray(x, float))
    Z = np.stack([enc.encode(float(xi)) for xi in xs]) if xs.size else np.zeros((0, F.shape[0]))
    out = Z @ F
    if normalize:
        out = out / (Z @ bake["density"])
    if trend is not None:
        out = out + trend[0] * xs + trend[1]                  # ...and the line comes back exactly
    return out if np.ndim(x) else float(out[0])


# ==================================================================================================================
# H2 -- THE SUPERPOSED GATHER.  N weighted lookups in ONE dot product; and the rule itself is a vector you can move.
#
# A gather is sum_j w_j * f(u_j) -- a quadrature rule, a filter stencil, a light-sampling estimator. Classically it
# costs N lookups. Here the lookup is a dot product against the baked field, and a dot product is LINEAR, so the
# whole rule collapses into ONE query vector before the field is ever touched:
#
#       Q = sum_j w_j Z(u_j)                     <- compile the rule (once)
#       gather = <F, Q>                          <- apply it to any baked field (one dot product, forever)
#
# WHAT THE MEASUREMENT ACTUALLY SAID -- and it corrected the plan we came in with.
#
#   (1) The gather is EXACT, not approximate. <F, sum_j w_j Z(u_j)> == sum_j w_j <F, Z(u_j)> by linearity, and it
#       measures that way: |gather - sum-of-fetches| = 0.0 to 7e-15 at N = 4, 32, 128, 512. There is nothing here to
#       approximate. The gather's only error is the bake's own error, which it inherited.
#
#   (2) There is NO CAPACITY WALL, and we expected one. The backlog predicted crosstalk growing as sqrt(N/D) -- the
#       familiar law for a bundle of KEYED items you must later unbind. It does not apply, because a gather never
#       unbinds anything: it superposes N encodings and then reads ONE linear functional off the result. Measured on
#       an interpolation rule (weights summing to 1), the error FALLS with N as the bake's independent per-point
#       errors average each other down:
#
#             N          2       8      32     128     512
#             RMS/std  0.053   0.027   0.012   0.011   0.008        (sqrt(N/D) would have said 0.022 -> 0.354)
#
#       More lookups make a superposed gather MORE accurate, not less. The crosstalk law governs cleanup-gated
#       recall (holographic_superposed) and has no jurisdiction here. Naming which regime you are in is the whole
#       skill: superposing things you must TELL APART pays crosstalk; superposing things you only ever SUM does not.
#
#   (3) The compiled rule is a vector, so BINDING TRANSLATES IT -- the entire N-tap stencil slides to a new offset
#       for one bind, at a cost independent of N (measured: cos(bind-shifted, re-encoded-from-scratch) = 1.0000000000
#       and the gathered values agree to 1e-13). That is `translate_rule`. A GPU re-fetches N taps per offset.
#
#   (4) The win is REUSE, and only reuse. Compiling costs N encodings, the same N the naive path pays, so a
#       single-use gather saves only N-1 dot products. Apply the same rule to many baked fields and the picture
#       changes completely (N = 64 taps, 200 fields): naive 2,806 ms; compile once 14 ms; then 0.48 ms of dot
#       products -- 5,871x per application, 190x amortised including the compile. State it that way or it is a
#       benchmark, not a measurement.
#
#   (5) ...and reuse has ONE precondition, which the integration test found the hard way. A rule is a superposition
#       of a particular encoder's atoms. Apply it to a field baked with a different encoder -- and `bake_1d` picks
#       the bandwidth per function, so two different functions get different encoders by default -- and the dot
#       product is a finite, confident, WRONG number: measured 1.816 against a truth of -0.818, with nothing raised.
#       So a compiled rule carries the encoder's signature and `gather` refuses on a mismatch. To share a rule,
#       bake the fields with the same dim, seed and an explicit shared bandwidth.
# ==================================================================================================================
def _encoder_signature(bake):
    """What a compiled rule is only valid against: the exact placement rule that made Z(u).

    A gather rule is `sum_j w_j Z(u_j)` -- the encoder's atoms. Apply it to a field baked with a DIFFERENT encoder
    and the dot product is meaningless, but it is a perfectly finite number and nothing complains. MEASURED: a
    5-tap stencil compiled against one bake, applied to a second bake of a different function (whose bandwidth was
    therefore chosen differently), returned 1.816 where the truth was -0.818. That is the failure this signature
    exists to catch."""
    enc = bake["encoder"]
    return (int(enc.dim), float(enc.lo), float(enc.hi), float(enc.bandwidth), str(enc.kernel),
            float(np.asarray(enc.phases)[1]) if enc.dim > 1 else 0.0)     # phases pin the seed


def gather_rule(bake, points, weights=None):
    """Compile N weighted lookups into ONE query vector: Q = sum_j w_j Z(u_j).

    `points`  -- the N places to sample (a quadrature rule, a stencil, a set of light samples).
    `weights` -- the N coefficients; None means all ones (a plain sum).

    Returns {"rule": the hypervector, "signature": the encoder it was compiled against}. Apply it with
    `gather(bake, rule)`, slide it with `translate_rule`. Compiling costs the same N encodings a naive gather pays,
    so this only pays when the rule is reused -- across many fields, or at many offsets.

    THE RULE IS ONLY VALID AGAINST ITS OWN ENCODER. To reuse it across fields, bake them with the same `dim`,
    `seed` and an explicit shared `bandwidth`; otherwise `bake_1d` picks a bandwidth per function and the encoders
    differ. `gather` and `translate_rule` check the signature and raise; pass the bare vector to skip the check."""
    enc = bake["encoder"]
    u = np.atleast_1d(np.asarray(points, float))
    w = np.ones(u.size) if weights is None else np.atleast_1d(np.asarray(weights, float))
    if w.size != u.size:
        raise ValueError("gather_rule: %d points but %d weights" % (u.size, w.size))
    Z = np.stack([enc.encode(float(ui)) for ui in u])
    return {"rule": w @ Z, "signature": _encoder_signature(bake)}      # superposition IS the compiled rule


def _rule_vector(bake, rule):
    """Accept either a signed rule dict (checked) or a bare hypervector (the 'I know what I am doing' path)."""
    if isinstance(rule, dict):
        if rule.get("signature") != _encoder_signature(bake):
            raise ValueError("gather: this rule was compiled against a different encoder. A rule is a superposition "
                             "of that encoder's atoms; against another bake the dot product is a finite, confident, "
                             "WRONG number. Bake both fields with the same dim, seed and an explicit bandwidth.")
        return np.asarray(rule["rule"], float)
    return np.asarray(rule, float)


def gather(bake, rule, normalize=False):
    """Apply a compiled rule to a baked field: ONE dot product, whatever N was.

    Exact against running the N fetches and summing them (measured to 7e-15). With `normalize=True` this returns
    <F, Q> / <density, Q>, the DENSITY-WEIGHTED average over the rule -- which is what you want for an interpolation
    rule, and is NOT the same thing as averaging the individually normalized fetches (measured: 0.054351 for the
    ratio-of-gathers against a truth of 0.054149, where the mean-of-ratios gives 0.057631). The ratio is the better
    estimator AND the cheaper one; it is just not the estimator you would have written by hand.

    A DETRENDED bake is refused: the field holds only the residual, and restoring the endpoint line would need the
    rule's individual sample points, which a compiled rule deliberately no longer carries (that is what makes it one
    vector). Gather against a plain bake, or add sum_j w_j (a u_j + b) yourself."""
    if bake.get("trend") is not None:
        raise ValueError("gather: this bake was detrended. The compiled rule is one vector and no longer knows its "
                         "own sample points, so the endpoint line cannot be restored. Bake without detrend=True, or "
                         "add the trend term sum_j w_j*(a*u_j + b) yourself.")
    q = _rule_vector(bake, rule)
    num = float(np.dot(bake["field"], q))
    if not normalize:
        return num
    den = float(np.dot(bake["density"], q))
    if den == 0.0:
        raise ValueError("gather: the rule carries zero density -- nothing to normalize against")
    return num / den


def translate_rule(bake, rule, dx):
    """Slide an entire compiled rule by `dx`, for one bind. Cost independent of N.

    This is exactly `bind(rule, Z(dx))`: the encoder is a fractional power encoding, so Z(u) * Z(dx) = Z(u + dx) in
    the spectrum, and binding a superposition of encodings translates every term at once. We rotate the phases
    directly rather than calling `encode(dx)` because `dx` is an OFFSET, not a coordinate -- a negative or large one
    is perfectly legal and would trip the encoder's range warning. (A test pins the two against each other for a dx
    that does lie in range.)

    Measured against re-encoding all N taps at the shifted positions: cosine 1.0000000000, gathered values agree to
    1e-13. Returns a rule in the same form it was given (signed dict in, signed dict out)."""
    enc = bake["encoder"]
    q = _rule_vector(bake, rule)
    spec = np.exp(1j * enc.scale * float(dx) * enc.phases)      # the conjugate-symmetric generator: Z(dx)'s spectrum
    moved = np.real(np.fft.ifft(np.fft.fft(q) * spec))
    return {"rule": moved, "signature": rule["signature"]} if isinstance(rule, dict) else moved


def bake_nd(grids, values, dim=8192, seed=0, margin=1.5):
    """H5 -- the texture unit in N dimensions: a gridded function baked into ONE hypervector.

    KEPT NEGATIVE -- **"SAMPLE O(1)" MEANS O(1) IN THE NUMBER OF BAKED SAMPLES, NOT O(1) IN `dim`.** A `fetch_nd`
    costs one O(dim log dim) transform per point. On a CPU that is far more than re-evaluating a procedural field.
    Measured against a 5-octave fBm driving a material channel (24^3 grid, dim 2048), with the sample path fully
    vectorised:

        hits    direct resolve   fetch_nd   ratio
         512        1.25 ms       207.6 ms   166x SLOWER
        4,096       3.18 ms      2010.0 ms   632x SLOWER

    ... and the bake does not even hold the field: correlation 0.252, max error 0.296 on a [0.05, 0.95] channel.
    That is H5's bandwidth negative, not a tuning failure -- a 24^3 grid cannot carry five octaves.

    **So do NOT bake a procedural material into the texture unit for a CPU renderer.** The primitive earns its keep
    where the fetch is a parallel dot product and the field is genuinely expensive or unavailable in closed form --
    that is the muscle side, the browser, where `exp` and a dot are free and a pattern graph is not. Bake for the
    consumer that has the hardware, not for the one that has the source.


    `grids`  -- one 1-D array of UNIFORMLY spaced coordinates per axis.
    `values` -- the sampled function, shaped (len(grids[0]), len(grids[1]), ...).

    Same law as `bake_1d`, one dimension up: the per-axis bandwidths are probed FROM THE DATA (B_k = margin * w_max
    along axis k), because the underlying `VectorFunctionEncoder`'s default of 3.0 measures at 1.0019 scale-free RMS
    on a 2-D sine -- worse than predicting the mean, and it raises nothing.

    Returns {"encoder", "field", "density", "bandwidths"}; read it with `fetch_nd(bake, point)`.

    THERE IS NO CAPACITY BUDGET ON THE NUMBER OF POINTS -- a bundled function is only ever summed, never unbound, so
    it sits where the H2 gather sits. Measured (2-D sine, D=8192, bandwidth held fixed so the probe cannot confound
    it), scale-free RMS is essentially flat as the grid goes 400 -> 6,400 points: 0.098 / 0.111 / 0.116 / 0.118. It
    neither blows up nor averages down. Spend points freely.

    BANDWIDTH IS A BIAS-VARIANCE DIAL, AND `dim` IS THE VARIANCE BUDGET. This is the thing to understand before
    trusting a number, and an earlier version of this docstring got it backwards ("error falls with D"). Measured
    scale-free RMS on a 2-D sine, 40x40 grid:

          margin        D=4,096      D=16,384      D=65,536      amplitude gain (D=65,536)
            1.5          0.1179       0.1174        0.1191               0.66
            2.5          0.0715       0.0566        0.0315               0.84
            4.0          0.1256       0.0777        0.0320               0.91
            6.0          0.1972       0.1546        0.0568               0.88

    On THIS signal, at the DEFAULT margin of 1.5, the error is a BIAS floor: sixteen times the dimension buys
    nothing at all (0.1179 -> 0.1191). The kernel is simply too smooth to represent the function, and no amount of D
    fixes a bandwidth that is too low. Raise the margin and the bias falls -- but a narrower kernel has more
    crosstalk, and crosstalk is what D pays for, which is why margin 4.0 and 6.0 are WORSE than 1.5 at D=4,096 and
    better at D=65,536. Around margin 2.5 is the knee HERE. Raise the two together, or not at all.

    THE CAUSAL VARIABLE IS THE BANDWIDTH B, NOT `margin`, because B = margin * w_max. Held fixed at B = 18.8, a
    1-cycle and a 2-cycle sine behave identically (both variance-limited, D pays: 0.122 -> 0.043 and 0.180 ->
    0.055); the same 1-cycle sine at B = 9.4 is bias-limited and D buys nothing. So "the default margin" means a
    different regime on different data. THE DIAGNOSTIC COSTS ONE EXTRA BAKE: double D. If the error drops you are
    variance-limited -- keep spending dimension. If it does not move you are bias-limited -- raise the margin.

    KEPT NEGATIVE: at the default margin this is a SHAPE estimator, not a calibrated one -- measured amplitude gain
    0.66. Read shape, not amplitude, unless you have raised margin and dim together (2.5 at D=65,536 gives 0.84;
    4.0 gives 0.91) or calibrated the gain once against known samples. See holographic_fpe's H5 block."""
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
    gs = [np.asarray(g, float) for g in grids]
    V = np.asarray(values, float)
    if V.shape != tuple(g.size for g in gs):
        raise ValueError("bake_nd: values shape %s does not match the grids %s" % (V.shape, [g.size for g in gs]))
    enc = VectorFunctionEncoder.for_grid(gs, V, dim=int(dim), margin=float(margin), seed=int(seed))
    pts = np.stack(np.meshgrid(*gs, indexing="ij"), -1).reshape(-1, len(gs))
    field, density = enc.bundle_normalized(pts, V.reshape(-1))
    return {"encoder": enc, "field": field, "density": density, "bandwidths": list(enc.bandwidth)}


def fetch_nd(bake, point, normalize=True):
    """Query an n-D baked field at any point: ONE dot product (or two, normalized). One point, or an (M, n) array.

    `normalize=True` (the default here, unlike the 1-D `fetch`, which keeps its historical raw behaviour) divides by
    the density bake, so the answer does not scale with how densely you happened to sample. It still carries the
    kernel's smoothing attenuation -- see `bake_nd`'s kept negative. Read shape, not amplitude, unless you have
    raised the margin or calibrated the gain."""
    enc = bake["encoder"]
    P = np.atleast_2d(np.asarray(point, float))
    # "Bake once, sample O(1)" -- and the sample path was a Python loop over `encode`. `encode_many` is the same
    # arithmetic, batched. It is not bit-identical (5.6e-16: binding all axes' spectra at once reassociates the
    # products that pairwise `bind` performs in sequence), and it is only ~1.4x faster, because the cost of a
    # sample is one O(dim log dim) transform per point and the loop was never the bottleneck. See the kept negative
    # on `bake_nd`.
    Z = enc.encode_many(P) if hasattr(enc, "encode_many") else np.stack([enc.encode(p) for p in P])
    out = Z @ bake["field"]
    if normalize:
        out = out / (Z @ bake["density"])
    return out if np.ndim(point) > 1 else float(out[0])


def gather_samples(xs, ys, points, weights=None, dim=4096, seed=0, margin=1.5, bandwidth=None, normalize=True):
    """Bake, compile a rule, and gather -- all in one call, from plain numbers to a plain number.

    WHY THIS EXISTS. `bake_1d` and `gather_rule` hand back LIVE objects (an encoder, a hypervector). Over an HTTP
    /invoke boundary those serialise to dead dictionaries: an agent can read the bandwidth but cannot feed the bake
    back in. So the compile-once-reuse-often API is an in-process API, and this is its stateless one-shot twin --
    the same math, no handles, callable with nothing but JSON. It re-bakes every call, so it buys none of the reuse
    win (that is the whole point of the other API); it exists so the capability is reachable at all from outside."""
    bake = bake_1d(xs, ys, dim=dim, seed=seed, margin=margin, bandwidth=bandwidth)
    return gather(bake, gather_rule(bake, points, weights), normalize=normalize)


def _selftest():
    rng = np.random.default_rng(0)
    n = 256
    f = rng.standard_normal(n)
    k = blur_kernel((n,))

    def literal(x, passes):
        for _ in range(passes):
            x = np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(k)))
        return x

    # integer N matches the literal loop, at a cost independent of N
    for N in (1, 8, 64):
        assert np.max(np.abs(filter_k(f, k, N) - literal(f, N))) < 1e-9, N
    assert np.isfinite(filter_k(f, k, 1_000_000)).all()          # a million passes: same cost, no loop

    # FRACTIONAL passes: half a pass, twice, is one pass. No GPU can do this.
    half = filter_k(f, k, 0.5)
    assert np.max(np.abs(filter_k(half, k, 0.5) - filter_k(f, k, 1))) < 1e-9
    # ...and a kernel whose transfer CHANGES SIGN refuses, rather than silently picking a branch.
    # (Note k = [0, .5, .5] has transfer cos(w), which is negative for half the spectrum. A "sharpen" kernel like
    # 2 - cos(w) is strictly POSITIVE, so its fractional power is perfectly well defined -- and allowed.)
    signed = np.zeros(n); signed[1] = 0.5; signed[-1] = 0.5      # transfer = cos(w): real, changes sign
    try:
        filter_k(f, signed, 0.5)
        assert False, "fractional power of a sign-changing transfer must raise"
    except ValueError:
        pass
    sharpen = np.zeros(n); sharpen[0] = 2.0; sharpen[1] = -0.5; sharpen[-1] = -0.5   # transfer 2 - cos(w) > 0
    assert np.isfinite(filter_k(f, sharpen, 0.5)).all()          # ...this one is unambiguous, and permitted

    # INFINITE passes: the steady state is a projection (idempotent), and it is what the loop converges to
    lim = filter_limit(f, k)
    assert np.max(np.abs(filter_limit(lim, k) - lim)) < 1e-12    # idempotent -> a true projection
    # The loop DOES converge there -- but slowly, and that is the whole argument. This blur's slowest non-DC mode
    # has |transfer| = 0.999849, so a literal loop needs ~200,000 passes to arrive; measured error 1.8e-1 at 1,000
    # passes, 6.3e-3 at 20,000, 1.1e-14 at 200,000. `filter_limit` returns the answer in one O(D) evaluation.
    assert np.max(np.abs(literal(f, 300) - lim)) < np.max(np.abs(f - lim))     # it is heading there
    assert np.max(np.abs(filter_k(f, k, 200_000) - lim)) < 1e-9               # ...and arrives, in closed form
    # a blur's only non-decaying mode is the DC term, so the limit is the field's mean
    assert abs(lim.mean() - f.mean()) < 1e-9 and lim.std() < 1e-6

    # 2-D works the same way (the FFT is n-D)
    F = rng.standard_normal((32, 32))
    K = blur_kernel((32, 32), axis=0) * 0.0
    K[0, 0] = 0.5; K[0, 1] = 0.25; K[0, -1] = 0.25
    twice = np.real(np.fft.ifftn(np.fft.fftn(np.real(np.fft.ifftn(np.fft.fftn(F) * np.fft.fftn(K)))) * np.fft.fftn(K)))
    assert np.max(np.abs(filter_k(F, K, 2) - twice)) < 1e-9

    # ---- H1: the compiler -- a 3-stage graph collapses to one transfer, exactly ----------------------------
    kb = blur_kernel((n,))
    kw = np.zeros(n); kw[0] = 0.34; kw[1] = 0.33; kw[-1] = 0.33
    staged = filter_k(f, kb, 8)
    w_ = 2.0 * np.pi * np.fft.fftfreq(n)
    staged = np.real(np.fft.ifft(np.fft.fft(staged) * np.exp(-1j * w_ * 3)))          # translate 3
    staged = 1.6 * staged - 0.6 * np.real(np.fft.ifft(np.fft.fft(staged) * np.fft.fft(kw)))
    piped = Pipeline((n,)).blur(kb, 8).translate(3).unsharp(kw, 0.6).apply(f)
    assert np.max(np.abs(piped - staged)) < 1e-9
    # A fractional sub-sample translation is a phase ramp, so half-shifts compose EXACTLY -- but only in the
    # TRANSFER domain. Materialising the intermediate field takes a real part, which throws away the (genuinely
    # imaginary) Nyquist-bin component that a half-sample shift creates: measured 9.3e-2 of signal, silently lost.
    # That is the compiler's whole argument in one line: compose the operators, not the images.
    assert np.max(np.abs(Pipeline((n,)).translate(0.5).translate(0.5).apply(f)
                         - Pipeline((n,)).translate(1.0).apply(f))) < 1e-9          # composed: exact
    staged_half = Pipeline((n,)).translate(0.5).apply(Pipeline((n,)).translate(0.5).apply(f))
    assert np.max(np.abs(staged_half - Pipeline((n,)).translate(1.0).apply(f))) > 1e-3   # materialised: lossy

    # ---- H3: the algebra has a Nyquist -----------------------------------------------------------------
    xs = np.linspace(0.0, 1.0, 240)
    for freq in (2.0, 5.0):
        ys = np.sin(2 * np.pi * freq * xs)
        assert abs(bandwidth_probe(xs, ys) - 2 * np.pi * freq) < 0.15 * 2 * np.pi * freq   # the probe finds w_max
        b = bake_1d(xs, ys, dim=4096)                       # bandwidth chosen FROM THE DATA
        q = np.linspace(0.05, 0.95, 41)
        pred, true = fetch(b, q), np.sin(2 * np.pi * freq * q)
        scale = np.dot(pred, true) / np.dot(pred, pred)     # the bake has an arbitrary gain
        assert np.sqrt(np.mean((scale * pred - true) ** 2)) < 0.06

    # forcing a bandwidth below w_max warns -- the failure is silent otherwise
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bake_1d(xs, np.sin(2 * np.pi * 12 * xs), dim=1024, bandwidth=3.0)
        assert any("Nyquist" in str(w.message) for w in caught)

    # ---- H2: the superposed gather -- exact, wall-free, and slideable ----------------------------------
    g = lambda t: np.sin(2 * np.pi * 2.0 * t) + 0.4 * np.cos(2 * np.pi * 3.0 * t)
    gx = np.linspace(0.0, 1.0, 400)
    b = bake_1d(gx, g(gx), dim=4096, seed=0)
    r = np.random.default_rng(7)
    u, w = r.uniform(0.05, 0.95, 64), r.standard_normal(64)

    # (1) EXACT against the staged fetches. This is the load-bearing claim: a gather approximates nothing.
    Q = gather_rule(b, u, w)
    assert abs(gather(b, Q) - float(np.sum(w * fetch(b, u)))) < 1e-9

    # (2) NO CAPACITY WALL. An interpolation rule's error must FALL as taps are added, not grow as sqrt(N/D).
    #     (Guarding the direction, not the constants: this is where a crosstalk law would announce itself.)
    def _interp_err(N, seed):
        rr = np.random.default_rng(seed)
        uu = rr.uniform(0.1, 0.9, N)
        ww = rr.random(N); ww /= ww.sum()
        return abs(gather(b, gather_rule(b, uu, ww), normalize=True) - float(np.sum(ww * g(uu))))
    err_few = np.mean([_interp_err(2, 300 + s) for s in range(16)])
    err_many = np.mean([_interp_err(256, 300 + s) for s in range(16)])
    assert err_many < err_few, "a superposed gather must average its error DOWN, not pay sqrt(N/D) crosstalk"

    # (3) The rule TRANSLATES by a bind, at a cost independent of N -- and it really is `bind`, not a lookalike.
    from holographic.agents_and_reasoning.holographic_ai import bind as _bind, cosine as _cos
    dx = 0.05                                                    # in range, so encode(dx) will not warn
    assert _cos(translate_rule(b, Q, dx)["rule"], _bind(Q["rule"], b["encoder"].encode(dx))) > 1.0 - 1e-9
    rebuilt = gather_rule(b, u + dx, w)                          # re-encode every tap at the shifted position
    assert _cos(translate_rule(b, Q, dx)["rule"], rebuilt["rule"]) > 1.0 - 1e-9

    # (3b) A rule is a superposition of ONE encoder's atoms. Against another bake it is a confident wrong number
    #      (measured 1.816 vs a truth of -0.818), so the signature check refuses rather than answer.
    other = bake_1d(gx, np.cos(2 * np.pi * gx), dim=4096, seed=0)     # different function -> different bandwidth
    try:
        gather(other, Q)
        assert False, "a rule from a different encoder must raise, not return a number"
    except ValueError:
        pass
    shared = bake_1d(gx, np.cos(2 * np.pi * gx), dim=4096, seed=0, bandwidth=b["bandwidth"])   # ...share the encoder
    assert abs(gather(shared, Q) - float(np.sum(w * fetch(shared, u)))) < 1e-9                 # and it transfers

    # (5) The stateless one-shot form -- the same answer, from plain numbers, with no live handles to carry.
    tp = np.array([0.3, 0.4, 0.5, 0.6, 0.7]); tw = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    assert abs(gather_samples(gx, g(gx), tp, tw, dim=4096) - float(np.sum(tw * g(tp)))) < 0.05

    # ---- H4: detrend before you bake. It is the wrap, not the singularity. -----------------------------
    dx_ = np.linspace(0.0, 1.0, 400)
    dq_ = np.linspace(0.002, 0.998, 200)
    arel = lambda got, tru: float(np.sqrt(np.mean((got - tru) ** 2)) / np.std(tru))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # a STRAIGHT LINE probes at nearly the same bandwidth as sqrt -- because an FFT thinks its samples wrap.
        assert bandwidth_probe(dx_, dx_) > 500.0                       # measured 607.95 for f(x) = x
        assert bandwidth_probe(dx_, np.sqrt(dx_)) > 500.0              # measured 789.70
        assert bandwidth_probe(dx_, np.sin(4 * np.pi * dx_)) < 20.0    # measured 12.53 -- a REAL bandwidth
        # The contrast, asserted at EVERY seed. Absolute bars would be lottery tickets: the plain bake's error
        # ranges 0.268-4.487 across seeds for the cube root, because an inflated bandwidth collapses the RBF kernel
        # toward a delta and the fetch returns whichever sample lands nearest -- which the random phases decide.
        # What IS stable is the contrast: detrending wins at every seed (worst ratio measured 2.13, median 5-17x).
        for f_ in (np.sqrt, np.cbrt, lambda t: 1.0 / (t + 0.05)):
            ratios = []
            for seed_ in range(6):
                e_plain = arel(fetch(bake_1d(dx_, f_(dx_), dim=4096, seed=seed_), dq_, normalize=True), f_(dq_))
                e_detr = arel(fetch(bake_1d(dx_, f_(dx_), dim=4096, seed=seed_, detrend=True), dq_,
                                    normalize=True), f_(dq_))
                assert e_detr < e_plain, (f_, seed_, e_plain, e_detr)     # never worse, at any seed
                ratios.append(e_plain / max(e_detr, 1e-12))
            assert np.median(ratios) > 4.0, (f_, ratios)                  # and typically 5-17x better
        # a straight line detrends to EXACTLY zero residual: the analytic trend reconstructs it alone
        assert arel(fetch(bake_1d(dx_, dx_, dim=4096, seed=0, detrend=True), dq_, normalize=True), dq_) < 1e-9
        # ...and it never hurts: an already-periodic function has no trend to remove.
        sn = lambda t: np.sin(2 * np.pi * 2 * t)
        e_p = arel(fetch(bake_1d(dx_, sn(dx_), dim=4096, seed=0), dq_, normalize=True), sn(dq_))
        e_d = arel(fetch(bake_1d(dx_, sn(dx_), dim=4096, seed=0, detrend=True), dq_, normalize=True), sn(dq_))
        assert abs(e_p - e_d) < 1e-9, (e_p, e_d)
    # a detrended field holds only the residual, so a RAW kernel sum cannot carry the line back
    bd = bake_1d(dx_, np.sqrt(dx_), dim=1024, seed=0, detrend=True)
    try:
        fetch(bd, 0.5)
        assert False, "a raw fetch of a detrended bake must raise, not silently drop the trend"
    except ValueError:
        pass
    try:
        gather(bd, gather_rule(bd, [0.3, 0.6]))
        assert False, "a gather over a detrended bake must raise -- the rule no longer knows its points"
    except ValueError:
        pass

    # ---- H5: the N-D Nyquist. The library default carries no information at all. -----------------------
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder, axis_bandwidths
    gg = lambda P: np.sin(2 * np.pi * 2 * P[..., 0]) * np.cos(2 * np.pi * 2 * P[..., 1])
    ax = np.linspace(0.0, 1.0, 40)
    PP = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1)
    VV = gg(PP)
    qq = np.random.default_rng(0).uniform(0.15, 0.85, (120, 2))
    tt = gg(qq)

    def _sf(a):                                            # scale-free RMS: 1.0 means "no information"
        c = float(np.dot(a, tt) / np.dot(a, a)) if np.dot(a, a) > 0 else 0.0
        return float(np.sqrt(np.mean((c * a - tt) ** 2)) / np.std(tt))

    probed = axis_bandwidths([ax, ax], VV)                 # errs HIGH on a coarse grid (leakage), never low
    assert all(w >= 2 * np.pi * 2.0 * 0.95 for w in probed), probed

    bad = VectorFunctionEncoder(2, dim=8192, bounds=[(0, 1), (0, 1)], bandwidth=3.0, seed=0)
    Fb, Db = bad.bundle_normalized(PP.reshape(-1, 2), VV.reshape(-1))
    assert _sf(np.array([bad.query_normalized(Fb, Db, q) for q in qq])) > 0.9   # the default: measured 1.0015

    good = bake_nd([ax, ax], VV, dim=8192, seed=0)         # bandwidth from the data
    assert _sf(fetch_nd(good, qq)) < 0.20                  # measured 0.101

    # the n-D kernel is the PRODUCT of the 1-D kernels only in expectation: an O(1/sqrt(D)) floor is the budget
    r_ = np.random.default_rng(1)
    for D_, bar in ((1024, 0.06), (16384, 0.015)):
        e_ = VectorFunctionEncoder(2, dim=D_, bounds=[(0, 1), (0, 1)], bandwidth=[18.0, 18.0], seed=0)
        devs = [abs(float(np.dot(e_.encode(p), e_.encode(q))) - e_.kernel_at(p - q))
                for p, q in ((r_.uniform(0, 1, 2), r_.uniform(0, 1, 2)) for _ in range(40))]
        assert np.mean(devs) < bar, (D_, np.mean(devs))    # measured 0.031 at 1k, 0.006 at 16k

    # (4) The raw fetch's scale is the SAMPLE COUNT, not a constant of the function. Normalizing removes it.
    coarse = bake_1d(np.linspace(0, 1, 100), g(np.linspace(0, 1, 100)), dim=4096, seed=0)
    fine = bake_1d(np.linspace(0, 1, 800), g(np.linspace(0, 1, 800)), dim=4096, seed=0)
    q = np.linspace(0.1, 0.9, 41)
    raw_ratio = float(np.dot(fetch(fine, q), fetch(coarse, q)) / np.dot(fetch(coarse, q), fetch(coarse, q)))
    assert 7.0 < raw_ratio < 9.0, "the raw fetch scales with sample count (8x here); it is not f(x)"
    for bk in (coarse, fine):                                    # ...and normalized, both land on f with NO fit at
        e = np.sqrt(np.mean((fetch(bk, q, normalize=True) - g(q)) ** 2)) / np.std(g(q))   # all (measured 0.085 /
        assert e < 0.10, e                                       # 0.084 -- the density division carries them both)

    # ---- H7: `combine` is exact; the superposed variant BANK is a kept negative ------------------------
    _sigmas = (2.0, 6.0, 14.0, 30.0)
    _pipes = [Pipeline((n,)).blur(gauss_kernel(n, _s)) for _s in _sigmas]
    _w = np.array([0.4, 0.3, 0.2, 0.1])
    _staged = sum(wi * p.apply(f) for wi, p in zip(_w, _pipes))
    assert np.max(np.abs(combine(_pipes, _w).apply(f) - _staged)) < 1e-12   # M variants, ONE transfer, exact
    assert combine(_pipes, _w).blur(k, 2) is not None                       # ...and it keeps chaining
    try:
        combine(_pipes, [1.0, 2.0])
        assert False, "combine must reject a weight/pipeline count mismatch"
    except ValueError:
        pass

    # The BANK: unbinding M keyed variants recovers each at ~1/sqrt(M), NOT at 1 - sqrt(M/D). Pinned so nobody
    # re-derives it. (M=8 -> ~0.354; a usable bank would need ~0.97.)
    _D = 2048
    _rb = np.random.default_rng(1)
    _fb = _rb.standard_normal(_D)
    _Fh = np.fft.fft(_fb)
    _M = 8
    _Hs = [np.fft.fft(_rb.standard_normal(_D)) for _ in range(_M)]          # uncorrelated: the KINDEST case
    _Ks = [_phasor_key(_D, 100 + j) for j in range(_M)]
    _bank = sum(K * H for K, H in zip(_Ks, _Hs))
    _fid = []
    for j in range(_M):
        _rec = np.real(np.fft.ifft(np.conj(_Ks[j]) * (_bank * _Fh)))
        _tru = np.real(np.fft.ifft(_Hs[j] * _Fh))
        _fid.append(float(np.dot(_rec, _tru) / (np.linalg.norm(_rec) * np.linalg.norm(_tru))))
    assert abs(np.mean(_fid) - 1.0 / np.sqrt(_M)) < 0.05, np.mean(_fid)     # the law is 1/sqrt(M)
    assert np.mean(_fid) < 0.5, "a superposed variant bank cannot read back its variants -- KEPT NEGATIVE"

    print("OK: holographic_shader self-test passed (N integer passes in one evaluation, exact to 1e-9 and "
          "independent of N -- 1e6 passes cost the same as 1; FRACTIONAL passes compose (half twice == one) and a "
          "sign-changing kernel refuses rather than pick a branch; the INFINITE-pass limit is an idempotent "
          "projection onto the non-decaying modes -- reached instantly, where a literal loop needs ~200,000 passes "
          "because the slowest mode decays as 0.999849^N; n-D works unchanged. H3: the bandwidth probe recovers "
          "w_max from the data and the bake sets B from it, so a fetch lands within 0.06 at every frequency tried -- "
          "and forcing B below w_max WARNS, because that failure is otherwise silent. H1: a 3-stage graph "
          "(blur x8 -> translate -> unsharp) compiles to ONE transfer, exact to 1e-9; two half-sample shifts compose "
          "exactly in the TRANSFER domain, while materialising the intermediate silently loses 9.3e-2 at the Nyquist "
          "bin -- compose the operators, not the images. H2: N weighted lookups compile into ONE query vector and "
          "the gather is EXACT, not approximate; there is NO sqrt(N/D) crosstalk wall because a gather never "
          "unbinds -- its error AVERAGES DOWN with more taps; the compiled rule slides to any offset for one bind, "
          "at a cost independent of N; and the raw fetch's gain is the SAMPLE COUNT, which normalize=True removes. "
          "H7: M variants COMBINE into one transfer exactly (2e-16), 30x faster at M=64 -- but the superposed "
          "variant BANK is a KEPT NEGATIVE, pinned here: unbinding recovers a variant at 1/sqrt(M), not at "
          "1 - sqrt(M/D), and there was never a cost to save. H4: a STRAIGHT LINE probes at 607.95 where a real "
          "2-cycle sine probes at 12.53 -- an FFT thinks its samples wrap -- so the fix for a near-singular bake is "
          "DETRENDING, not domain warping: it wins at EVERY seed (median 5-17x), turns a line into an exact "
          "reconstruction, and costs nothing on an already-periodic function. Absolute bars here would be lottery "
          "tickets, because the plain bake's error ranges 0.27-4.49 across seeds -- an inflated bandwidth collapses "
          "the kernel toward a delta. H5: the n-D encoder's default bandwidth of 3.0 carries NO information "
          "(scale-free RMS 1.0015); "
          "probed from the data it lands at 0.101 -- and the n-D kernel is the product of the 1-D kernels only in "
          "EXPECTATION, with an O(1/sqrt(D)) floor that is where the crosstalk budget actually lives)")


if __name__ == "__main__":
    _selftest()
