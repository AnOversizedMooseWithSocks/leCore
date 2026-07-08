"""holographic_schedule.py -- Fill 4: the PROGRAM SCHEDULER / cost model. The capstone. Turn a VSA program into a
dependency DAG and schedule it: FUSE the straight-line spectral runs (Fill 2), keep the tie-sensitive ones
op-by-op, and cross to Python only at the genuine discrete commitments (the cleanups). This is the missing
compiler-and-scheduler rung between the program layer and the kernel -- the thing nvcc plus a warp scheduler do
on a GPU, in VSA-native form.

WHY THIS EXISTS (Compute Architecture plan, Fill 4 -- the capstone, built on Fills 1-3)
--------------------------------------------------------------------------------------
Today a VSA program goes to the kernel op-by-op: every bind materializes an intermediate vector (irfft) that the
next op immediately transforms back (rfft), and every step is a Python-level call. The scheduler reads the whole
DAG first and decides, per sub-graph:
  * a maximal LINEAR run (bind/bundle/permute/unbind) with no cleanup in the middle is ONE spectral kernel --
    FUSE it (Fill 2): one traversal, `leaves+1` transforms instead of `3*ops`, no materialized intermediates;
  * a CLEANUP / cosine / argmax is a boundary -- it collapses to a decision, the one place we cross to Python;
  * a TIE-SENSITIVE run (flagged by the caller) stays op-by-op on the frozen kernel, bit-exact -- the bind_batch
    lesson (a 1e-16 change flipped a maze trajectory), so fusion never touches those paths.
The cost model is deliberately small and readable -- a roofline "is this run long enough to be worth fusing?"
gate -- deterministic given the DAG, with NO learned policy and NO autotuner. Because a recurring workload has a
recurring DAG SHAPE, the schedule for a shape is computed once and content-addressed (so its cost amortizes).

This is the plain, in-Python scheduler the plan says to build FIRST. The SER-style "coherence key is a
hypervector, grouping is a cleanup" native variant is a fork to be measured AGAINST this one (its reorder cost
only pays above some DAG size), and is left as a documented next step, not the default.

HONEST SCOPE (kept loud): fusion is tolerance-not-bit-exact (~1e-15), so the scheduled result matches the
sequential one to tolerance on fused runs and BIT-EXACTLY on the unfused/tie-sensitive runs; the final discrete
commitment is a real Python crossing (the scheduler reduces the COUNT of crossings, it does not zero them); it is
a no-op speed-wise on compute-bound (~0% FFT) runs -- there it just avoids materialization. Deterministic;
NumPy + stdlib; delegates to holographic_fuse / _residency / _superschedule, never reimplements them.
"""
import hashlib

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, unbind, bundle, permute, nearest
import holographic.misc.holographic_fuse as _fuse

# op kinds
LEAF, BIND, UNBIND, BUNDLE, PERMUTE, CLEANUP = "leaf", "bind", "unbind", "bundle", "permute", "cleanup"
SUPERPOSE, NORMALIZE = "superpose", "normalize"     # recipe's plain-sum and normalize ops
_LINEAR = {BIND, UNBIND, BUNDLE, PERMUTE, SUPERPOSE}   # SUPERPOSE (plain add) fuses too; NORMALIZE is a boundary


class Op:
    """One node of a VSA program DAG. `kind` is a leaf/linear-op/cleanup; `inputs` are indices of the nodes it
    consumes; `param` carries a leaf's vector, a permute shift, or a cleanup codebook. `tie_sensitive` forces the
    node (and the run it belongs to) to stay op-by-op on the frozen kernel."""
    def __init__(self, kind, inputs=(), param=None, tie_sensitive=False):
        self.kind = kind
        self.inputs = tuple(inputs)
        self.param = param
        self.tie_sensitive = bool(tie_sensitive)


# --- program builders (read like the algebra) ------------------------------------------------------------------

def leaf(vector):
    return Op(LEAF, param=np.asarray(vector, float))


def op_bind(i, j, tie_sensitive=False):
    return Op(BIND, (i, j), tie_sensitive=tie_sensitive)


def op_unbind(i, j, tie_sensitive=False):
    return Op(UNBIND, (i, j), tie_sensitive=tie_sensitive)


def op_bundle(inputs, tie_sensitive=False):
    return Op(BUNDLE, tuple(inputs), tie_sensitive=tie_sensitive)


def op_permute(i, shift, tie_sensitive=False):
    return Op(PERMUTE, (i,), param=int(shift), tie_sensitive=tie_sensitive)


def op_cleanup(i, codebook):
    return Op(CLEANUP, (i,), param=np.asarray(codebook, float))


def op_superpose(inputs, tie_sensitive=False):
    return Op(SUPERPOSE, tuple(inputs), tie_sensitive=tie_sensitive)


def op_normalize(i, tie_sensitive=False):
    return Op(NORMALIZE, (i,), tie_sensitive=tie_sensitive)


# --- the schedule (analysis) -----------------------------------------------------------------------------------

def _consumers(ops):
    """How many nodes consume each node -- a node with >1 consumer must be MATERIALIZED (shared intermediate),
    so it becomes a fusion boundary rather than being pulled into two different fused runs."""
    counts = [0] * len(ops)
    for op in ops:
        for i in op.inputs:
            counts[i] += 1
    return counts


def plan(ops, min_run=2):
    """Analyse the DAG and return, per node, whether it is a FUSE-ROOT (the top of a linear run worth fusing).
    A node is a fuse-root iff it is a linear op, not tie-sensitive, and either it is an output, feeds a cleanup,
    or is shared -- and the linear run it heads is at least `min_run` ops long (the roofline cost gate). The plan
    is a pure function of the DAG structure -> deterministic and content-addressable."""
    cons = _consumers(ops)
    n = len(ops)
    is_output = [cons[i] == 0 for i in range(n)]

    def run_length(i):
        """Size of the maximal linear, single-consumer, non-tie-sensitive run feeding node i (i included)."""
        op = ops[i]
        if op.kind not in _LINEAR or op.tie_sensitive:
            return 0
        size = 1
        for j in op.inputs:
            cj = ops[j]
            if cj.kind in _LINEAR and not cj.tie_sensitive and cons[j] == 1:
                size += run_length(j)
        return size

    fuse_root = [False] * n
    for i in range(n):
        op = ops[i]
        if op.kind not in _LINEAR or op.tie_sensitive:
            continue
        # a linear node HEADS a run (is a fuse-root) when its consumer won't absorb it into a longer run:
        # it is an output, is shared (>1 consumer), or any consumer is non-linear (cleanup/normalize/...) or
        # tie-sensitive. Otherwise a single linear consumer swallows it into that consumer's run.
        heads_run = (is_output[i] or cons[i] > 1 or
                     any(ops[c].kind not in _LINEAR or ops[c].tie_sensitive
                         for c in _direct_consumers(ops, i)))
        if heads_run and run_length(i) >= min_run:
            fuse_root[i] = True
    return fuse_root


def _direct_consumers(ops, i):
    return [c for c, op in enumerate(ops) if i in op.inputs]


def plan_signature(ops, min_run=2):
    """A deterministic content hash of the DAG SHAPE + the resulting plan -- so a recurring program shape reuses
    its schedule (the 'compute the schedule once' amortization). Same DAG -> identical signature."""
    fr = plan(ops, min_run=min_run)
    parts = []
    for i, op in enumerate(ops):
        parts.append("%d:%s:%s:%d:%d" % (i, op.kind, ",".join(map(str, op.inputs)),
                                         int(op.tie_sensitive), int(fr[i])))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


# --- execution -------------------------------------------------------------------------------------------------

def _fuse_expr(i, ops, values, cons, spectrum_cache):
    """Build a holographic_fuse expression for the linear run headed at node i, pulling in linear, single-consumer,
    non-tie-sensitive ancestors; anything else (a leaf, a cleanup output, a shared/materialized node) enters as an
    already-computed Leaf value."""
    op = ops[i]
    def child(j):
        cj = ops[j]
        if cj.kind in _LINEAR and not cj.tie_sensitive and cons[j] == 1:
            return _fuse_expr(j, ops, values, cons, spectrum_cache)
        return _fuse.leaf(values[j])                       # boundary: use its materialized value
    if op.kind == BIND:
        return _fuse.fbind(child(op.inputs[0]), child(op.inputs[1]))
    if op.kind == UNBIND:
        return _fuse.funbind(child(op.inputs[0]), child(op.inputs[1]))
    if op.kind == PERMUTE:
        return _fuse.fpermute(child(op.inputs[0]), op.param)
    if op.kind == BUNDLE:
        return _fuse.fbundle([child(j) for j in op.inputs])
    if op.kind == SUPERPOSE:
        return _fuse.fsum([child(j) for j in op.inputs])
    raise ValueError(op.kind)


def _eval_opbyop(i, ops, values):
    """Evaluate one node on the frozen kernel, op-by-op (the sequential path and the tie-sensitive path)."""
    op = ops[i]
    if op.kind == BIND:
        return bind(values[op.inputs[0]], values[op.inputs[1]])
    if op.kind == UNBIND:
        return unbind(values[op.inputs[0]], values[op.inputs[1]])
    if op.kind == PERMUTE:
        return permute(values[op.inputs[0]], op.param)
    if op.kind == BUNDLE:
        return bundle([values[j] for j in op.inputs])
    if op.kind == SUPERPOSE:
        return np.sum([values[j] for j in op.inputs], axis=0)      # plain sum, no renormalization
    if op.kind == NORMALIZE:
        v = values[op.inputs[0]]; nrm = float(np.linalg.norm(v))
        return v / nrm if nrm > 0 else v
    raise ValueError(op.kind)


def run_sequential(ops):
    """The BASELINE: execute every op on the frozen kernel, op-by-op, materializing every intermediate. Returns
    (values, stats) where stats has the FFT count and the number of Python kernel-op calls (the crossing proxy)."""
    values = [None] * len(ops)
    fft = 0
    kernel_calls = 0
    crossings = 0
    for i, op in enumerate(ops):
        if op.kind == LEAF:
            values[i] = op.param
        elif op.kind == CLEANUP:
            idx, _ = nearest(values[op.inputs[0]], op.param)
            values[i] = op.param[idx]
            crossings += 1
        else:
            values[i] = _eval_opbyop(i, ops, values)
            kernel_calls += 1
            if op.kind in (BIND, UNBIND):
                fft += 3                                   # 2 rfft + 1 irfft per bind/unbind
    return values, {"fft": fft, "kernel_calls": kernel_calls, "crossings": crossings}


def run_scheduled(ops, min_run=2, spectrum_cache=None):
    """The SCHEDULED executor: fuse each linear run (Fill 2), keep tie-sensitive runs op-by-op, cross to Python
    only at cleanups. Returns (values, stats) with the reduced FFT count and kernel-op-call count. The result
    matches run_sequential to FFT tolerance on fused runs and bit-exactly on the unfused ones."""
    cons = _consumers(ops)
    fr = plan(ops, min_run=min_run)
    values = [None] * len(ops)
    consumed_by_fusion = [False] * len(ops)

    # mark nodes that will be absorbed into some fuse-root's expression (so we don't also evaluate them alone)
    def mark(i):
        for j in ops[i].inputs:
            cj = ops[j]
            if cj.kind in _LINEAR and not cj.tie_sensitive and cons[j] == 1:
                consumed_by_fusion[j] = True
                mark(j)
    for i in range(len(ops)):
        if fr[i]:
            mark(i)

    _fuse.reset_fft_counts()
    fft = 0
    kernel_calls = 0
    crossings = 0
    for i, op in enumerate(ops):
        if op.kind == LEAF:
            values[i] = op.param
        elif op.kind == CLEANUP:
            idx, _ = nearest(values[op.inputs[0]], op.param)
            values[i] = op.param[idx]
            crossings += 1
        elif consumed_by_fusion[i] and not fr[i]:
            continue                                       # absorbed into a fuse-root's expression; skip
        elif fr[i]:
            before = _fuse.fft_counts()
            values[i] = _fuse.fuse(_fuse_expr(i, ops, values, cons, spectrum_cache),
                                   spectrum_cache=spectrum_cache)
            after = _fuse.fft_counts()
            fft += (after["rfft"] - before["rfft"]) + (after["irfft"] - before["irfft"])
            kernel_calls += 1                              # the whole run is ONE fused call
        else:
            values[i] = _eval_opbyop(i, ops, values)       # short/tie-sensitive run: op-by-op, bit-exact
            kernel_calls += 1
            if op.kind in (BIND, UNBIND):
                fft += 3
    return values, {"fft": fft, "kernel_calls": kernel_calls, "crossings": crossings}


def from_recipe(recipe):
    """Convert a StructureRecipe's op list (its DAG, at Layer 4) into scheduler Ops, 1:1 by handle. Atoms are
    materialized from the seed exactly as recipe.build does. Returns (ops, output_handles), or raises for a
    recipe that uses `repeat` (whose templates are not lowered here -- run_recipe falls back to build() there)."""
    from holographic.agents_and_reasoning.holographic_ai import derived_atom
    ops = []
    for op in recipe._ops:
        kind = op[0]
        if kind == "atom":
            ops.append(leaf(derived_atom(recipe.seed, op[1], recipe.dim, unitary=op[2])))
        elif kind == "raw":
            ops.append(leaf(recipe._raws[op[1]].copy()))
        elif kind == "bind":
            ops.append(op_bind(op[1], op[2]))
        elif kind == "bundle":
            ops.append(op_bundle(list(op[1])))
        elif kind == "superpose":
            ops.append(op_superpose(list(op[1])))
        elif kind == "permute":
            ops.append(op_permute(op[1], op[2]))
        elif kind == "normalize":
            ops.append(op_normalize(op[1]))
        elif kind == "repeat":
            raise ValueError("recipe uses 'repeat' templates -- not lowered to the scheduler; use build()")
        else:
            raise ValueError("unknown recipe op %r" % (kind,))
    handles = recipe._outputs if recipe._outputs else list(range(len(ops)))
    return ops, handles


def run_recipe(recipe, fused=True, min_run=2, spectrum_cache=None):
    """Realize a recipe's OUTPUT vectors through the scheduler -- fusing its straight-line bind/bundle/permute
    runs (Fill 2) so a long structure build does far fewer FFTs than the op-by-op `recipe.build()`. Returns
    (outputs, stats). THROUGHPUT path: fusion is ~1e-15, not bit-exact, so this does NOT preserve the recipe's
    bit-exact-replay guarantee -- `build()` remains the exact default; opt in here only when you want speed and
    can tolerate FFT round-off. Recipes using `repeat` fall back to the exact op-by-op build."""
    try:
        ops, handles = from_recipe(recipe)
    except ValueError:
        return recipe.outputs(), {"fft": None, "kernel_calls": None, "crossings": 0, "fused": False}
    values, stats = (run_scheduled(ops, min_run=min_run, spectrum_cache=spectrum_cache)
                     if fused else run_sequential(ops))
    stats["fused"] = fused
    return [values[h] for h in handles], stats


def _selftest():
    """The scheduler fuses linear runs (fewer FFTs and fewer Python kernel-calls than sequential), produces the
    same result to tolerance, keeps a tie-sensitive run bit-exact and unfused, crosses to Python only at cleanups,
    and its plan is deterministic given the DAG."""
    from holographic.agents_and_reasoning.holographic_ai import cosine
    rng = np.random.default_rng(0)
    D = 1024
    atoms = [rng.standard_normal(D) for _ in range(8)]
    for a in atoms:
        a /= np.linalg.norm(a)

    # a realistic pipeline: build a role/filler record (K binds + a bundle), unbind a role, clean up to an atom.
    # program: leaves 0..5 (roles r0,r1,r2 ; fillers f0,f1,f2), then binds, bundle, unbind, cleanup.
    r0, r1, r2, f0, f1, f2 = atoms[:6]
    codebook = np.stack([f0, f1, f2])
    ops = [
        leaf(r0), leaf(r1), leaf(r2), leaf(f0), leaf(f1), leaf(f2),   # 0..5
        op_bind(0, 3), op_bind(1, 4), op_bind(2, 5),                  # 6,7,8  role_i * filler_i
        op_bundle([6, 7, 8]),                                        # 9      the record
        op_unbind(9, 0),                                             # 10     free filler bound to r0
        op_cleanup(10, codebook),                                    # 11     -> nearest atom (the crossing)
    ]

    seq_vals, seq = run_sequential(ops)
    sch_vals, sch = run_scheduled(ops, min_run=2)

    # (1) fewer FFTs and fewer Python kernel-calls than the sequential baseline
    assert sch["fft"] < seq["fft"], (sch["fft"], seq["fft"])
    assert sch["kernel_calls"] < seq["kernel_calls"]
    # (2) same number of Python crossings = cleanups only (the scheduler crosses ONLY at the discrete commit)
    assert sch["crossings"] == seq["crossings"] == 1
    # (3) the recovered filler is the same atom in both, and the pre-cleanup vector matches to FFT tolerance
    assert np.array_equal(seq_vals[11], sch_vals[11])               # same cleanup winner
    assert np.abs(seq_vals[10] - sch_vals[10]).max() < 1e-9         # fused run == sequential to tolerance
    assert cosine(sch_vals[11], f0) > 0.99                          # it recovered the right filler

    # (4) a TIE-SENSITIVE run stays op-by-op (bit-exact) and unfused
    ops_ts = [
        leaf(r0), leaf(f0), leaf(r1),
        op_bind(0, 1, tie_sensitive=True),                          # 3  tie-sensitive bind
        op_bind(3, 2, tie_sensitive=True),                          # 4  tie-sensitive bind
    ]
    seq_ts, _ = run_sequential(ops_ts)
    sch_ts, st = run_scheduled(ops_ts)
    assert np.array_equal(seq_ts[4], sch_ts[4])                     # BIT-EXACT (not just tolerance) -- unfused
    fr = plan(ops_ts)
    assert not any(fr)                                             # nothing tie-sensitive was made a fuse-root

    # (5) deterministic plan: same DAG -> identical signature
    assert plan_signature(ops) == plan_signature(ops)
    # a structurally different program has a different signature
    assert plan_signature(ops) != plan_signature(ops_ts)

    # (6) the recipe bridge: fusing a real Layer-4 recipe's linear run does fewer FFTs and matches build() to tol
    from holographic.misc.holographic_recipe import StructureRecipe
    rc = StructureRecipe(dim=1024, seed=0)
    _roles = [rc.atom("role%d" % i, unitary=True) for i in range(4)]
    _fills = [rc.atom("fill%d" % i) for i in range(4)]
    _binds = [rc.bind(_roles[i], _fills[i]) for i in range(4)]
    rc.mark_output(rc.normalize(rc.permute(rc.bundle(_binds), 3)))
    _exact = rc.outputs()
    _fused, _fs = run_recipe(rc, fused=True)
    _seqr, _ss = run_recipe(rc, fused=False)
    assert np.abs(_exact[0] - _fused[0]).max() < 1e-9      # fused recipe == exact build to tolerance
    assert _fs["fft"] < _ss["fft"]                         # and it fused the linear run

    print("holographic_schedule selftest OK: scheduled %d FFTs / %d kernel-calls vs sequential %d / %d; result "
          "matches to tolerance, cleanup winner identical; tie-sensitive run stays bit-exact & unfused; crosses "
          "only at cleanups; recipe bridge fuses a real Layer-4 build (%d vs %d FFTs); plan deterministic"
          % (sch["fft"], sch["kernel_calls"], seq["fft"], seq["kernel_calls"], _fs["fft"], _ss["fft"]))


if __name__ == "__main__":
    _selftest()
