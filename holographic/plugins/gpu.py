"""GPU -- bundled plugin (pip extra: pip install cupy-cuda12x  (match your CUDA)).

CuPy is tied to the host's CUDA version and is deliberately left out of the `all` extra; a verb that only makes sense with it is the definition of a plugin. The transparent CuPy array backend (holographic_backend, mind.use_gpu) stays in core because it has a NumPy fallback.

MOVED VERBATIM from the UnifiedMind parts by tools/migrate_to_plugins.py: bodies and
docstrings are the former methods with `self` dropped, so GET /tools, find_capability and
REFERENCE.md say what they said before. A move, not a rewrite.
"""
PLUGIN = {
    "name": 'gpu',
    "version": "1.0",
    "does": 'Run the language model on the CuPy/CUDA backend and prove it agrees with the NumPy path.',
    "requires": ('cupy',),
    "install": 'pip install cupy-cuda12x  (match your CUDA)',
}


def unicron_device(runtime=None, want="auto", ids=None):
    """RUN THE MODEL ON WHATEVER HARDWARE IS THERE, AND PROVE IT AGREES.
    An LLM is usually run on a GPU. This runtime was pure host NumPy, so on a machine
    with a card it left the ENTIRE FORWARD PASS on the CPU -- the FLOPs are in the
    model, and leCore's WGSL path only covered leCore's OWN kernels.
    leCORE ALREADY HAD THE SWITCH and the runtime never asked for it:
    `array_module()` returns cupy when a device is present AND the policy allows and
    numpy otherwise, `gpu_available` / `backend_status` say what is there, and
    `resource_policy(gpu=...)` decides. So this is not a GPU port -- it is the missing
    WIRE between a switch that existed and a forward pass that ignored it.
    RESIDENCY IS THE POINT, and the backend's own docstring says why: every
    host-to-device transfer costs, and a small per-call op loses to the transfer that
    feeds it. WEIGHTS MOVE ONCE AND STAY; ids and logits are small and cross per call.
    A runtime that moved weights per layer would be SLOWER on a GPU than on a CPU and
    would look like the GPU was at fault.
    ASKING FOR A GPU THAT IS NOT THERE IS NOT AN ERROR -- it reports cpu and runs,
    because a pipeline that dies on a laptop is worse than one that is merely slower.
    TESTED WITHOUT A GPU, because an untested path rots: the selftest substitutes a
    fake device module and drives the whole dispatch, making 50 weight tensors
    resident and returning output BIT-IDENTICAL to the host path.
    WHAT IS NOT CLAIMED: no speedup, because none was measured on real hardware.
    `gpu_crossover` exists to find where a device starts winning and needs a real
    adapter to answer. The claim here is PARITY -- the same numbers either way -- which
    is what makes the speed question safe to ask later. See holographic_devicerun."""
    from holographic.io_and_interop.holographic_devicerun import (
        status, place, parity)
    if runtime is None:
        return status()
    if ids is not None:
        return parity(runtime, ids)
    return place(runtime, want=want)


def register(mind, config=None):
    """Bind this plugin's verbs. `config` is accepted and unused."""
    return [
        {"name": 'unicron_device', "fn": unicron_device,
         "does": 'RUN THE MODEL ON WHATEVER HARDWARE IS THERE, AND PROVE IT AGREES.',
         "example": "mind.unicron_device(runtime=None, want='auto', ids=None)"},
    ]


def _selftest():
    """Contract: the verbs bind on a default mind and are absent from a slim one; each
    is callable (its own honest failure without the dependency is the fallback that
    made this a plugin in the first place)."""
    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0)
    slim = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    for v in ['unicron_device']:
        assert callable(getattr(m, v, None)), 'bundled gpu did not bind %r' % v
        assert not hasattr(slim, v), 'plugins=() still carried %r' % v
    assert [p['name'] for p in m.plugin_list() if p['name'] == 'gpu'] == ['gpu']
    print('holographic.plugins.gpu selftest OK')


if __name__ == "__main__":
    _selftest()
