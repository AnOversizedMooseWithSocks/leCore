"""Fill 4: the program scheduler -- fuses linear runs, keeps tie-sensitive bit-exact, crosses only at cleanups."""
import numpy as np
from holographic.scene_and_pipeline.holographic_schedule import leaf, op_bind, op_unbind, op_bundle, op_cleanup, run_sequential, run_scheduled, plan, plan_signature
from holographic.agents_and_reasoning.holographic_ai import cosine


def _atoms(rng, k, d):
    v = [rng.standard_normal(d) for _ in range(k)]
    return [x / np.linalg.norm(x) for x in v]


def _pipeline(rng, D):
    r0, r1, r2, f0, f1, f2 = _atoms(rng, 6, D)
    cb = np.stack([f0, f1, f2])
    ops = [leaf(r0), leaf(r1), leaf(r2), leaf(f0), leaf(f1), leaf(f2),
           op_bind(0, 3), op_bind(1, 4), op_bind(2, 5),
           op_bundle([6, 7, 8]), op_unbind(9, 0), op_cleanup(10, cb)]
    return ops, f0


def test_scheduler_fewer_ffts_same_result():
    rng = np.random.default_rng(0); D = 1024
    ops, f0 = _pipeline(rng, D)
    sv, seq = run_sequential(ops); cv, sch = run_scheduled(ops)
    assert sch["fft"] < seq["fft"]
    assert sch["kernel_calls"] < seq["kernel_calls"]
    assert sch["crossings"] == seq["crossings"] == 1
    assert np.array_equal(sv[11], cv[11])
    assert np.abs(sv[10] - cv[10]).max() < 1e-9
    assert cosine(cv[11], f0) > 0.99


def test_tie_sensitive_stays_bit_exact():
    rng = np.random.default_rng(1); D = 512
    a, b, c = _atoms(rng, 3, D)
    ops = [leaf(a), leaf(b), leaf(c),
           op_bind(0, 1, tie_sensitive=True), op_bind(3, 2, tie_sensitive=True)]
    sv, _ = run_sequential(ops); cv, _ = run_scheduled(ops)
    assert np.array_equal(sv[4], cv[4])           # bit-exact, not just tolerance
    assert not any(plan(ops))


def test_plan_deterministic():
    rng = np.random.default_rng(2); D = 256
    ops, _ = _pipeline(rng, D)
    ops2 = [leaf(ops[0].param), leaf(ops[1].param), op_bind(0, 1)]
    assert plan_signature(ops) == plan_signature(ops)
    assert plan_signature(ops) != plan_signature(ops2)


def test_recipe_bridge_fuses_and_matches():
    """Fill 4 integration: fusing a real Layer-4 recipe matches the exact build() to tolerance with fewer FFTs;
    a recipe with `repeat` falls back to the exact op-by-op build."""
    import numpy as np
    from holographic.misc.holographic_recipe import StructureRecipe
    from holographic.scene_and_pipeline.holographic_schedule import run_recipe
    r = StructureRecipe(dim=1024, seed=0)
    roles = [r.atom("role%d" % i, unitary=True) for i in range(4)]
    fills = [r.atom("fill%d" % i) for i in range(4)]
    binds = [r.bind(roles[i], fills[i]) for i in range(4)]
    r.mark_output(r.normalize(r.permute(r.bundle(binds), 3)))
    exact = r.outputs()
    fused, fs = run_recipe(r, fused=True)
    seq, ss = run_recipe(r, fused=False)
    assert np.abs(exact[0] - fused[0]).max() < 1e-9
    assert fs["fft"] < ss["fft"] and fs["fused"] is True
    # repeat -> fallback to exact build (bit-exact, no fusion)
    r2 = StructureRecipe(dim=256, seed=1)
    base = r2.atom("base")
    r2.repeat(3, [("atom", "x{i}", False), ("bind", 0, 0)])
    out2, st2 = run_recipe(r2, fused=True)
    assert st2["fused"] is False                            # fell back for the repeat template
    assert np.array_equal(out2[0], r2.outputs()[0])         # and it is the exact build
