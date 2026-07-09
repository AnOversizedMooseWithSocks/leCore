"""holographic_computehome.py -- the COMPUTE home (consolidation backlog H7): stay VSA-native. Keep the hot middle
of a computation in the vector / frequency domain, with no Python hops between ops.

THE RULE
--------
PUSH decisions and cleanups to the BOUNDARIES; keep the linear middle in the vector domain. Every time a computation
leaves the frequency domain to make a Python-level decision (an if, a cleanup, a copy) it pays a round trip -- an
inverse FFT out and a forward FFT back in. So: transform in ONCE, do all the bind / bundle / permute algebra on the
spectra, decide / clean up ONCE at the end, transform out ONCE.

THE LEVERS (already shipped; this home is the single door -- route, don't rewrite)
  * FUSE      -- holographic_fuse: collapse a bind/bundle/permute EXPRESSION TREE into ~2 FFTs (one forward per
                 distinct leaf, all algebra on spectra, one inverse out) instead of an FFT round-trip per op. Equal
                 to the op-by-op result to FFT tolerance (~1e-15). fft_counts() MEASURES the win.
  * SCHEDULE  -- holographic_schedule: a cost model that fuses runs of linear ops and falls back to eager execution
                 where a decision/cleanup interrupts the run. run_recipe / run_scheduled.
  * WIDTH     -- holographic_superschedule: run independent work in parallel across the vector (breadth, not depth).
  * PROGRAM   -- holographic_machine: run logic as a VSA PROGRAM on the stored-program machine -- the fully native
                 path, with the interpreter itself in the vector domain (no Python loop between instructions).

    Compute.leaf / bind / unbind / bundle / sum / permute   # build a fusable expression tree
    Compute.fuse(expr, cache)                                # collapse it into ~2 FFTs
    Compute.fuse_record(keys, values, cache)                 # the common role/filler record, fused
    Compute.reset_fft_counts() / fft_counts()               # measure the FFT count (the fusion win)
    Compute.run_recipe(recipe, ...) / run_scheduled(ops, ...)  # the cost-model scheduler
    Compute.machine(dim, seed, ...)                          # a stored-program VSA machine (as_program path)
"""


class Compute:
    """A namespace of staticmethods over the VSA-native compute levers. Fuse / schedule / width / program."""

    # --- expression builders (build a tree, then fuse it) ---
    @staticmethod
    def leaf(v):
        """A leaf of a fusable expression tree: a raw vector. Routes to holographic_fuse.leaf."""
        from holographic.misc.holographic_fuse import leaf
        return leaf(v)

    @staticmethod
    def bind(x, y):
        from holographic.misc.holographic_fuse import fbind
        return fbind(x, y)

    @staticmethod
    def unbind(x, y):
        from holographic.misc.holographic_fuse import funbind
        return funbind(x, y)

    @staticmethod
    def bundle(children):
        from holographic.misc.holographic_fuse import fbundle
        return fbundle(children)

    @staticmethod
    def sum(children):
        from holographic.misc.holographic_fuse import fsum
        return fsum(children)

    @staticmethod
    def permute(x, shift):
        from holographic.misc.holographic_fuse import fpermute
        return fpermute(x, shift)

    # --- the fusion itself + its measurement ---
    @staticmethod
    def fuse(expr, spectrum_cache=None):
        """Evaluate a bind/bundle/permute expression TREE in the FFT domain: one forward transform per distinct
        leaf, all algebra on spectra, one inverse out -- ~2 FFTs total instead of a round-trip per op. Pass a
        Memory.spectrum_cache() to make known-atom leaf transforms free. Equal to the op-by-op result to FFT
        tolerance. Routes to holographic_fuse.fuse."""
        from holographic.misc.holographic_fuse import fuse
        return fuse(expr, spectrum_cache=spectrum_cache)

    @staticmethod
    def fuse_record(keys, values, spectrum_cache=None):
        """The most common fusable pattern: bundle([bind(k_i, v_i)]) -- a role/filler record -- fused, in
        2*len(keys)+2 FFTs instead of ~3*len. Routes to holographic_fuse.fuse_record."""
        from holographic.misc.holographic_fuse import fuse_record
        return fuse_record(keys, values, spectrum_cache=spectrum_cache)

    @staticmethod
    def reset_fft_counts():
        """Zero the FFT counters, so the next fuse/op run can be MEASURED. Routes to holographic_fuse."""
        from holographic.misc.holographic_fuse import reset_fft_counts
        reset_fft_counts()

    @staticmethod
    def fft_counts():
        """The FFT counters since the last reset: {'rfft': n, 'irfft': m}. The honest measure of a fusion win.
        Routes to holographic_fuse.fft_counts."""
        from holographic.misc.holographic_fuse import fft_counts
        return fft_counts()

    # --- the scheduler (fuse runs of linear ops; eager at the boundaries) ---
    @staticmethod
    def run_recipe(recipe, fused=True, min_run=2, spectrum_cache=None):
        """Run a recipe under the cost model: fuse runs of >= min_run linear ops, execute eagerly where a
        decision/cleanup interrupts the run. Routes to holographic_schedule.run_recipe."""
        from holographic.scene_and_pipeline.holographic_schedule import run_recipe
        return run_recipe(recipe, fused=fused, min_run=min_run, spectrum_cache=spectrum_cache)

    @staticmethod
    def run_scheduled(ops, min_run=2, spectrum_cache=None):
        """Run a flat op list with the fuse-runs cost model. Routes to holographic_schedule.run_scheduled."""
        from holographic.scene_and_pipeline.holographic_schedule import run_scheduled
        return run_scheduled(ops, min_run=min_run, spectrum_cache=spectrum_cache)

    # --- run logic as a VSA program (the fully-native path) ---
    @staticmethod
    def machine(dim=4096, seed=7, data=None, faculties=None):
        """A stored-program VSA MACHINE: logic encoded as a program vector and run with the interpreter itself in
        the vector domain (no Python loop between instructions). Routes to holographic_machine.HoloMachine."""
        from holographic.agents_and_reasoning.holographic_machine import HoloMachine
        return HoloMachine(dim=dim, seed=seed, data=data, faculties=faculties)

    @staticmethod
    def iterate(op, state, k, min_k=8, probes=3, seed=0):
        """Apply a linear operator `op` to `state` k times, RE-ENABLING the closed-form jump behind a detector: if op
        is a circular convolution (a bind) -- decidable by its impulse response -- evaluate step k in ONE FFT (EXACT,
        ~k-fold fewer transforms); otherwise step k times. The closed form is exact in regime, so this never does
        worse than stepping. Returns (result, info). Routes to holographic_iterate.iterate_gated. ('iterate is
        PRT-for-time': bake the operator's spectrum once, read any level for free.)"""
        from holographic.misc.holographic_iterate import iterate_gated
        return iterate_gated(op, state, k, min_k=min_k, probes=probes, seed=seed)


def compute_levers():
    """The VSA-native compute levers the home exposes (for the catalog / discovery)."""
    return ("fuse", "schedule", "width", "program")


def _selftest():
    import numpy as np
    from holographic.agents_and_reasoning.holographic_ai import bind, bundle

    rng = np.random.default_rng(0)
    n, d = 16, 512
    keys = [rng.standard_normal(d) for _ in range(n)]
    values = [rng.standard_normal(d) for _ in range(n)]

    # FUSE a record and MEASURE the FFT count (the done-when)
    Compute.reset_fft_counts()
    fused = Compute.fuse_record(keys, values)
    counts = Compute.fft_counts()
    total_fused = counts["rfft"] + counts["irfft"]
    naive = 3 * n                                                 # op-by-op: each bind is 2 rfft + 1 irfft
    assert total_fused < naive                                   # the fused chain uses measurably fewer FFTs
    assert total_fused == 2 * n + 2                              # exactly the module's stated 2*len(keys)+2

    # and it AGREES with the op-by-op result to FFT tolerance
    ref = bundle(np.stack([bind(keys[i], values[i]) for i in range(n)]))
    assert np.allclose(fused, ref, atol=1e-9)

    # fuse an arbitrary expression tree: it AGREES with the op-by-op result (the fusion is what's under test, not the
    # recovery quality -- unbind of a non-unitary key is noisy either way)
    a = rng.standard_normal(d); b = rng.standard_normal(d)
    from holographic.agents_and_reasoning.holographic_ai import unbind
    expr = Compute.unbind(Compute.bind(Compute.leaf(a), Compute.leaf(b)), Compute.leaf(a))
    got = Compute.fuse(expr)
    assert np.allclose(got, unbind(bind(a, b), a), atol=1e-9)     # fused == eager unbind(bind(a,b), a)

    # the program path is reachable (a machine can be built)
    mach = Compute.machine(dim=256, seed=1)
    assert hasattr(mach, "run")
    print("OK: holographic_computehome self-test passed (fuse_record %d FFTs vs %d naive -> %.0f%% fewer, agrees to "
          "tol; expression fuse recovers; machine reachable; levers %s)"
          % (total_fused, naive, 100 * (1 - total_fused / naive), ", ".join(compute_levers())))


if __name__ == "__main__":
    _selftest()
