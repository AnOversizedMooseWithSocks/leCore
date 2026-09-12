"""JIT -- bundled plugin (pip extra: pip install leos-core[jit,symbolic]).

These need BOTH numba and sympy and exist only to produce JIT'd kernels. NOT moved: signed_distance_field and every other faculty with a numba FAST PATH and a NumPy fallback -- that is the accelerator pattern, and it stays in core precisely because it works without numba.

MOVED VERBATIM from the UnifiedMind parts by tools/migrate_to_plugins.py: bodies and
docstrings are the former methods with `self` dropped, so GET /tools, find_capability and
REFERENCE.md say what they said before. A move, not a rewrite.
"""
PLUGIN = {
    "name": 'jit',
    "version": "1.0",
    "does": "Numba-compiled SDF kernels and the fully-JIT'd analytic SDF renderer (SymPy expression in, njit kernels out).",
    "requires": ('numba', 'sympy'),
    "install": 'pip install leos-core[jit,symbolic]',
}


def compiled_sdf_numba(expr, variables=("x", "y", "z")):
    """SymPy -> Numba, cached: compile a symbolic 3-D SDF to njit scalar+grid value/normal kernels ONCE and
    reuse them. The scalar njit SDF composes into other njit loops (a sphere-trace march) -- the closure barrier
    that blocked Numba from the raymarch is gone. Needs sympy + numba. See holographic_compile.compiled_sdf_numba."""
    from holographic.scene_and_pipeline.holographic_compile import compiled_sdf_numba
    return compiled_sdf_numba(expr, variables)


def render_sdf_fast(expr, camera, width=256, height=256, light_dir=(-0.4, 0.7, -0.3),
                    base_color=(0.85, 0.5, 0.35), ao=True, shadows=True, ambient=0.25, sky=None):
    """Render an analytic SDF (given as a symbolic expression) with the fully-JIT'd renderer: the whole march --
    primary ray, exact normal, AO, soft shadow -- compiles into one njit kernel (the closure barrier is gone),
    ~9-15x the numpy renderer for the field-native shading. Compiled renderer cached per SDF. Needs sympy+numba;
    falls back is the caller's (use render_sdf without jit_expr). See holographic_sdf_render.render_analytic."""
    from holographic.rendering.holographic_sdf_render import render_analytic
    return render_analytic(expr, camera, width=width, height=height, light_dir=light_dir,
                           base_color=base_color, ao=ao, shadows=shadows, ambient=ambient, sky=sky)


def register(mind, config=None):
    """Bind this plugin's verbs. `config` is accepted and unused."""
    return [
        {"name": 'compiled_sdf_numba', "fn": compiled_sdf_numba,
         "does": 'SymPy -> Numba, cached: compile a symbolic 3-D SDF to njit scalar+grid value/normal kernels ONCE and',
         "example": "mind.compiled_sdf_numba(expr, variables=('x', 'y', 'z'))"},
        {"name": 'render_sdf_fast', "fn": render_sdf_fast,
         "does": "Render an analytic SDF (given as a symbolic expression) with the fully-JIT'd renderer: the whole march --",
         "example": 'mind.render_sdf_fast(expr, camera, width=256, height=256, light_dir=(-0.4, 0.7, -0.3), base_color=(0.85, 0.5, 0.35), ao=True, shadows=True, ambient=0.25, sky=None)'},
    ]


def _selftest():
    """Contract: the verbs bind on a default mind and are absent from a slim one; each
    is callable (its own honest failure without the dependency is the fallback that
    made this a plugin in the first place)."""
    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0)
    slim = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    for v in ['compiled_sdf_numba', 'render_sdf_fast']:
        assert callable(getattr(m, v, None)), 'bundled jit did not bind %r' % v
        assert not hasattr(slim, v), 'plugins=() still carried %r' % v
    assert [p['name'] for p in m.plugin_list() if p['name'] == 'jit'] == ['jit']
    print('holographic.plugins.jit selftest OK')


if __name__ == "__main__":
    _selftest()
