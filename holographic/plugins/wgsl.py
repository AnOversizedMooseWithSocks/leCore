"""WGSL -- bundled plugin (pip extra: pip install leos-core[wgsl]).

The WGSL EMITTER stays in core (it is pure Python and used by the dialect emitters); only RUNNING the result needs the wgpu wheel.

MOVED VERBATIM from the UnifiedMind parts by tools/migrate_to_plugins.py: bodies and
docstrings are the former methods with `self` dropped, so GET /tools, find_capability and
REFERENCE.md say what they said before. A move, not a rewrite.
"""
PLUGIN = {
    "name": 'wgsl',
    "version": "1.0",
    "does": 'Run an annotated Python kernel on any GPU (Vulkan / Metal / DX12 / WebGPU, or a software adapter) via its WGSL projection.',
    "requires": ('wgpu',),
    "install": 'pip install leos-core[wgsl]',
}


def run_wgsl_kernel(fn, data, extra_args=(), workgroup=64):
    """RUN AN ANNOTATED PYTHON KERNEL ON ANY GPU via its own WGSL projection (holographic_wgpurun).
    emit_kernel already turned `fn` into WGSL; this wraps it in a @compute entry point with storage
    bindings and a bounds guard, dispatches it, and returns float32.
    SCOPE: elementwise maps over a 1-D array, f32 only (WGSL has no f64). A bounded `for range(N)` is
    fine; a CROSS-INVOCATION REDUCTION -- what bundle and cleanup need -- is not solved here.
    RAISES rather than falling back when wgpu is absent: a caller who explicitly asked for the device
    path deserves to know they did not get it. (use_gpu falls back silently, which is right for a
    transparent accelerator and wrong for an explicit request.)
    Use verify_wgsl_kernel to check the projection against the Python original on your own data --
    exactness holds for single-expression kernels and NOT for accumulating ones."""
    from holographic.io_and_interop.holographic_wgpurun import run_kernel
    from holographic.io_and_interop.holographic_emit import emit
    return run_kernel(emit(fn, "wgsl"), fn.__name__, data, extra_args=extra_args, workgroup=workgroup)


def register(mind, config=None):
    """Bind this plugin's verbs. `config` is accepted and unused."""
    return [
        {"name": 'run_wgsl_kernel', "fn": run_wgsl_kernel,
         "does": 'RUN AN ANNOTATED PYTHON KERNEL ON ANY GPU via its own WGSL projection (holographic_wgpurun).',
         "example": 'mind.run_wgsl_kernel(fn, data, extra_args=(), workgroup=64)'},
    ]


def _selftest():
    """Contract: the verbs bind on a default mind and are absent from a slim one; each
    is callable (its own honest failure without the dependency is the fallback that
    made this a plugin in the first place)."""
    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0)
    slim = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    for v in ['run_wgsl_kernel']:
        assert callable(getattr(m, v, None)), 'bundled wgsl did not bind %r' % v
        assert not hasattr(slim, v), 'plugins=() still carried %r' % v
    assert [p['name'] for p in m.plugin_list() if p['name'] == 'wgsl'] == ['wgsl']
    print('holographic.plugins.wgsl selftest OK')


if __name__ == "__main__":
    _selftest()
