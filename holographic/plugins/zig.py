"""ZIG -- bundled plugin (pip extra: pip install leos-core[zig]).

Every verb here exists to drive the `ziglang` wheel (a ~45 MB toolchain). Without it each already answered honestly ('ziglang not installed'); that answer is unchanged, the verbs just no longer live on the core class.

MOVED VERBATIM from the UnifiedMind parts by tools/migrate_to_plugins.py: bodies and
docstrings are the former methods with `self` dropped, so GET /tools, find_capability and
REFERENCE.md say what they said before. A move, not a rewrite.
"""
PLUGIN = {
    "name": 'zig',
    "version": "1.0",
    "does": 'Native Zig kernels: compile scalar kernels to shared libraries, race them against NumPy, and validate emitted C/Zig against the Python original.',
    "requires": ('ziglang',),
    "install": 'pip install leos-core[zig]',
}


def zig_batch_eval(kernel, arrays, dtype="f64", simd=0, opt="safe"):
    """Compile a scalar kernel to a native shared library (content-hash cached, `ziglang` wheel, OPT-IN like
    numba) and batch-evaluate it over P same-length arrays. `opt='safe'` is deterministic (f64 scalar measured
    BIT-IDENTICAL to the NumPy evaluation); `simd=8` with dtype='f32' is the measured throughput sweet spot.
    Returns the results as a list. First call pays ~1-2 s of compiler, then ~0 -- a one-shot small-n call is a
    LOSS and this method does not pretend otherwise. See holographic_zigrun.ZigKernel."""
    from holographic.io_and_interop.holographic_zigrun import ZigKernel
    import numpy as _np
    cols = [_np.asarray(a, dtype=float) for a in arrays]
    return [float(x) for x in ZigKernel(kernel, dtype=str(dtype), simd=int(simd), opt=str(opt))(*cols)]


def zig_regime_map(kernel, sizes=(1000, 100000, 1000000), repeats=5, seed=0, simd_width=8):
    """Z3's honest measurement: race numpy / zig scalar f64 / zig simd f32 across sizes. Every row carries the
    baseline, the spread, and a correctness max-abs-err -- a fast wrong answer is not a result. MEASURED verdict
    on the round-box SDF: a modest real 2-5x, peaking near n=1e5, compressing to ~2x at n=1e6 where everything
    goes memory-bandwidth bound. No order-of-magnitude win exists and none is claimed.
    See holographic_zigrun.regime_map."""
    from holographic.io_and_interop.holographic_zigrun import regime_map
    return regime_map(kernel, sizes=tuple(int(s) for s in sizes), repeats=int(repeats),
                      seed=int(seed), simd_width=int(simd_width))


def zig_dispatch_policy(n, calls_expected, min_calls_to_compile=3, min_n=4096, toolchain=None):
    """Z5: which backend (numpy | zig) the native-kernel dispatcher would choose for arrays of length `n`
    called `calls_expected` times, WITH the reason -- the policy is data, sized to the measured 2-5x regime
    and the ~1-2 s first-call compile. Compose with mind.zig_batch_eval to act on the answer.
    See holographic_zigrun.dispatch_policy (AutoKernel enforces the same policy in-process, identity-gated:
    a native result that is not bit-identical to numpy in safe mode refuses the substitution permanently)."""
    from holographic.io_and_interop.holographic_zigrun import dispatch_policy
    return dispatch_policy(int(n), int(calls_expected), int(min_calls_to_compile), int(min_n), toolchain=toolchain)


def zig_march_compare(kernel=None, width=96, height=72, max_steps=96, out_dir=None, opt='safe'):
    """Z4's executed bar: sphere-trace the SAME rays through the engine's Python marcher and a natively
    compiled Zig loop (same scene SDF text, shared dialect table), shade both with the SAME code, and return
    {t_max_abs_diff, hit_flips, bit_identical, frames_byte_identical}. MEASURED verdict on the demo scene:
    f64 BIT-IDENTICAL, frames byte-identical, zig 3.8x on 110k rays x 96 steps -- and safe-vs-fast is a wash,
    so determinism costs nothing here. Pass out_dir to also write both PPM frames.
    See holographic_zigmarch.render_compare."""
    from holographic.io_and_interop.holographic_zigmarch import DEMO_SCENE, render_compare
    return render_compare(kernel if kernel is not None else DEMO_SCENE, width=int(width),
                          height=int(height), max_steps=int(max_steps), out_dir=out_dir, opt=opt)

# -- C3: canonical element + delta chain (instancing, generalised) --------------------------------------


def validate_kernel(fn, calls, dialect="c_f64"):
    """Compile the emitted C with `cc`, RUN it on `calls`, and compare to the Python original: {dialect, n,
    max_abs_diff, max_rel_diff, bit_identical}. `c_f64` comes out BIT-IDENTICAL. `c_f32` cannot -- and its
    error (2.9e-07 on an SDF) IS the tolerance a WGSL port must be judged against, because WGSL is f32 and
    NumPy is f64. That is why `c_f32` exists: so the tolerance is MEASURED, not chosen.
    Zig dialects (`zig_f64` / `zig_f32`) route to validate_zig -- compiled `-O ReleaseSafe` with the OPT-IN
    `ziglang` wheel (exactly numba's contract: everything passes without it, absence reported loudly).
    Measured: zig_f64 BIT-IDENTICAL on builtin-intrinsic kernels; std.math.pow is a declared 1-ulp negative.
    See holographic_emit.validate_c / validate_zig."""
    from holographic.io_and_interop.holographic_emit import validate_c, validate_zig
    calls = [tuple(float(x) for x in c) for c in calls]
    if str(dialect).startswith("zig"):
        return validate_zig(fn, calls, dialect=str(dialect))
    return validate_c(fn, calls, dialect=str(dialect))


def register(mind, config=None):
    """Bind this plugin's verbs. `config` is accepted and unused."""
    return [
        {"name": 'zig_batch_eval', "fn": zig_batch_eval,
         "does": 'Compile a scalar kernel to a native shared library (content-hash cached, `ziglang` wheel, OPT-IN like',
         "example": "mind.zig_batch_eval(kernel, arrays, dtype='f64', simd=0, opt='safe')"},
        {"name": 'zig_regime_map', "fn": zig_regime_map,
         "does": "Z3's honest measurement: race numpy / zig scalar f64 / zig simd f32 across sizes. Every row carries the",
         "example": 'mind.zig_regime_map(kernel, sizes=(1000, 100000, 1000000), repeats=5, seed=0, simd_width=8)'},
        {"name": 'zig_dispatch_policy', "fn": zig_dispatch_policy,
         "does": 'Z5: which backend (numpy | zig) the native-kernel dispatcher would choose for arrays of length `n`',
         "example": 'mind.zig_dispatch_policy(n, calls_expected, min_calls_to_compile=3, min_n=4096, toolchain=None)'},
        {"name": 'zig_march_compare', "fn": zig_march_compare,
         "does": "Z4's executed bar: sphere-trace the SAME rays through the engine's Python marcher and a natively",
         "example": "mind.zig_march_compare(kernel=None, width=96, height=72, max_steps=96, out_dir=None, opt='safe')"},
        {"name": 'validate_kernel', "fn": validate_kernel,
         "does": 'Compile the emitted C with `cc`, RUN it on `calls`, and compare to the Python original: {dialect, n,',
         "example": "mind.validate_kernel(fn, calls, dialect='c_f64')"},
    ]


def _selftest():
    """Contract: the verbs bind on a default mind and are absent from a slim one; each
    is callable (its own honest failure without the dependency is the fallback that
    made this a plugin in the first place)."""
    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0)
    slim = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    for v in ['zig_batch_eval', 'zig_regime_map', 'zig_dispatch_policy', 'zig_march_compare', 'validate_kernel']:
        assert callable(getattr(m, v, None)), 'bundled zig did not bind %r' % v
        assert not hasattr(slim, v), 'plugins=() still carried %r' % v
    assert [p['name'] for p in m.plugin_list() if p['name'] == 'zig'] == ['zig']
    print('holographic.plugins.zig selftest OK')


if __name__ == "__main__":
    _selftest()
