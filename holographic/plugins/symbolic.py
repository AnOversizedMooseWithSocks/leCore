"""SYMBOLIC -- bundled plugin (pip extra: pip install leos-core[symbolic]).

SymPy is a design-time dependency: these derive an exact expression ONCE and hand a plain callable to the renderer. The renderer itself stays in core and never imports sympy.

MOVED VERBATIM from the UnifiedMind parts by tools/migrate_to_plugins.py: bodies and
docstrings are the former methods with `self` dropped, so GET /tools, find_capability and
REFERENCE.md say what they said before. A move, not a rewrite.
"""
PLUGIN = {
    "name": 'symbolic',
    "version": "1.0",
    "does": 'Design-time symbolic maths: exact SDF normals and Jacobians derived with SymPy and compiled once into the content-addressed cache.',
    "requires": ('sympy',),
    "install": 'pip install leos-core[symbolic]',
}


def compiled_sdf_normal(expr, variables=("x", "y", "z")):
    """Compile a symbolic SDF's exact normal ONCE and reuse it via the content-addressed compile cache: the
    same expr returns the cached (value_fn, normal_fn) instantly instead of re-running the ~140-390 ms sympy
    lambdify, and recompiles only when the expr changes. The runtime use of the codegen pipeline -- compile a
    spec, cache the compiled version, hand it out everywhere. See holographic_compile.compiled_sdf_normal."""
    from holographic.scene_and_pipeline.holographic_compile import compiled_sdf_normal
    return compiled_sdf_normal(expr, variables)


def exact_sdf_normal(expr, variables=("x", "y", "z")):
    """Derive an EXACT SDF surface normal from a symbolic SDF expression (SymPy, design-time) and return
    (value_fn, normal_fn) of pure NumPy -- no finite-difference step-size error, no autodiff. The Quilez-seat
    path: e.g. exact_sdf_normal('sqrt(x**2+y**2+z**2)-1.0'). Needs sympy (requirements-accel.txt); the returned
    functions are pure NumPy. See holographic_codegen.sdf_normal_fn."""
    from holographic.misc.holographic_codegen import sdf_normal_fn
    return sdf_normal_fn(expr, variables)


def gradient_cache_symbolic(expr, anchors, variables=("x", "y", "z")):
    """Build an irradiance/GI-style GradientCache with EXACT Jacobians from a symbolic field (SymPy) instead of
    finite differences -- no truncation error in the cached gradients, so first-order interpolation is more
    accurate at the same anchors. Needs sympy. See holographic_cache.gradient_cache_symbolic."""
    from holographic.caching_and_storage.holographic_cache import gradient_cache_symbolic
    return gradient_cache_symbolic(expr, anchors, variables)


def register(mind, config=None):
    """Bind this plugin's verbs. `config` is accepted and unused."""
    return [
        {"name": 'compiled_sdf_normal', "fn": compiled_sdf_normal,
         "does": "Compile a symbolic SDF's exact normal ONCE and reuse it via the content-addressed compile cache: the",
         "example": "mind.compiled_sdf_normal(expr, variables=('x', 'y', 'z'))"},
        {"name": 'exact_sdf_normal', "fn": exact_sdf_normal,
         "does": 'Derive an EXACT SDF surface normal from a symbolic SDF expression (SymPy, design-time) and return',
         "example": "mind.exact_sdf_normal(expr, variables=('x', 'y', 'z'))"},
        {"name": 'gradient_cache_symbolic', "fn": gradient_cache_symbolic,
         "does": 'Build an irradiance/GI-style GradientCache with EXACT Jacobians from a symbolic field (SymPy) instead of',
         "example": "mind.gradient_cache_symbolic(expr, anchors, variables=('x', 'y', 'z'))"},
    ]


def _selftest():
    """Contract: the verbs bind on a default mind and are absent from a slim one; each
    is callable (its own honest failure without the dependency is the fallback that
    made this a plugin in the first place)."""
    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0)
    slim = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    for v in ['compiled_sdf_normal', 'exact_sdf_normal', 'gradient_cache_symbolic']:
        assert callable(getattr(m, v, None)), 'bundled symbolic did not bind %r' % v
        assert not hasattr(slim, v), 'plugins=() still carried %r' % v
    assert [p['name'] for p in m.plugin_list() if p['name'] == 'symbolic'] == ['symbolic']
    print('holographic.plugins.symbolic selftest OK')


if __name__ == "__main__":
    _selftest()
