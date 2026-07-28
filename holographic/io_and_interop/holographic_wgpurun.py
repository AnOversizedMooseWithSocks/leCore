"""WGPU-1 -- run an emitted WGSL kernel on ANY device (holographic_wgpurun).

WHAT THIS CLOSES
----------------
`emit_kernel` already projects an annotated Python kernel into WGSL, and nothing ran it. This dispatches it.

WHY IT MATTERS MORE THAN THE CuPy PATH: `holographic_backend` is CuPy, which is NVIDIA/CUDA ONLY. On Apple
silicon, AMD, Intel Arc, or in a browser, `use_gpu(True)` returns False and falls back. WGSL runs on Vulkan,
Metal, DX12 and WebGPU, so THIS is the path that matches the project's actual constraint -- run on a wide
variety of devices.

And it carries a guarantee CuPy cannot: THE SHADER IS A PROJECTION OF THE AUTHORITATIVE PYTHON. The same
function is the CPU implementation and the source of the GPU one, so they cannot silently diverge -- and
`verify_against_numpy` below differentially tests them on every call if you ask it to.

MEASURED, and it is the reason this shipped before any CuPy work:
    adapter llvmpipe (Mesa, adapter_type='CPU'), NO GPU PRESENT
    an emit_kernel body dispatched and read back, max abs deviation vs NumPy float32: 0.0 -- BIT-EXACT
Software adapters (llvmpipe, WARP) implement Vulkan/DX12, so CORRECTNESS is verifiable on an ordinary CI
runner with no hardware budget. The CuPy path can be tested NOWHERE without an NVIDIA machine.

SCOPE, honestly
  * ELEMENTWISE MAPS ONLY. `emit_kernel` produces a scalar straight-line `fn` (bounded `for range(N)` with a
    literal bound is allowed; `while` and data-dependent bounds are refused). That is exactly the shape a
    compute-shader invocation runs, so a per-element map over a big buffer is the natural fit. A
    CROSS-INVOCATION REDUCTION -- which is what bundle and cleanup need -- is a different problem and is NOT
    solved here.
  * f32 ONLY. WGSL has no f64. A kernel needing double precision must stay on CPU, and that is a
    determinism statement as much as a precision one.
  * NOT FOR TIE-SENSITIVE PATHS. Same rule as the CuPy backend: this is throughput.

BIT-EXACTNESS IS DATA-DEPENDENT AND IS NOT A GUARANTEE. This was got wrong three times before it was got
right, so the measurements are recorded rather than summarised:
    x*1.7 + 0.5 over arange(64 / 128 / 256 / 1024)   ->  EXACT, max_rel 0.0
    the SAME kernel over linspace(-3, 3, 128)        ->  NOT exact, max_rel ~8e-07
    a bounded loop accumulating 4 terms              ->  NOT exact, max_rel ~3e-06
So `exact` is an OBSERVATION ABOUT ONE RUN, not a property of a kernel shape. The honest contract is a
TOLERANCE: single-expression kernels stay within f32 rounding, accumulating ones drift measurably further.

TWO HARNESS BUGS FOUND WHILE ESTABLISHING THAT, both the same root cause, both worth knowing because they
make a correct kernel look broken:
  1. The array was cast to float32 for dispatch while Python evaluated the ORIGINAL float64 values.
  2. `extra_args` were emitted as f32 LITERALS in the shader while Python received float64 -- and 1.7 is not
     exactly representable, so the two sides multiplied by different constants.
Both reported the INPUT CONVERSION as kernel deviation. A DIFFERENTIAL TEST THAT DOES NOT FEED BOTH SIDES
IDENTICAL INPUTS MEASURES ITS OWN CONVERSIONS. Fixed by casting both the array and the scalars before either
side runs -- and that is precisely why this function exists: MEASURE THE DEVIATION ON YOUR OWN KERNEL AND
YOUR OWN DATA, never assume exactness and never assume drift.

`wgpu` is an OPTIONAL dependency, exactly like CuPy and numba. Absent, every entry point raises a clear
install message and nothing else in the engine changes.
"""

import numpy as np

_ENTRY = """
@group(0) @binding(0) var<storage, read> _in: array<f32>;
@group(0) @binding(1) var<storage, read_write> _out: array<f32>;

%(body)s

@compute @workgroup_size(%(wg)d)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&_in)) { _out[i] = %(call)s; }
}
"""


def available():
    """Is a WGSL compute device reachable? True on a software adapter too -- which is the point."""
    try:
        from wgpu.utils import get_default_device

        get_default_device()
        return True
    except Exception:
        return False


def device_info():
    """What adapter would run a kernel: {available, device, backend, type} or {available: False, why}."""
    try:
        from wgpu.utils import get_default_device

        info = dict(get_default_device().adapter.info)
        return {"available": True, "device": info.get("device", "?"),
                "backend": info.get("backend_type", "?"), "type": info.get("adapter_type", "?")}
    except ImportError:
        return {"available": False, "why": "wgpu is not installed (pip install wgpu)"}
    except Exception as exc:
        return {"available": False, "why": "no adapter: %s" % exc}


def wrap_kernel(body, name, extra_args=(), workgroup=64):
    """Wrap an emitted `fn` in a @compute entry point with storage bindings.

    THIS IS THE WHOLE MISSING PIECE. `emit_kernel` produces a bare function with no entry point, no
    bindings and no workgroup size -- correct WGSL, not a runnable shader. The wrapper owns exactly those
    three things plus the `arrayLength` bounds guard, which matters because the dispatch is rounded UP to
    whole workgroups and the tail invocations must not write past the buffer.

    `extra_args` are scalar f32 literals appended to the call, so a kernel like f(x, gain) can be dispatched
    with gain fixed -- uniforms would be the general answer and are deliberately not built until something
    needs them."""
    args = ", ".join(["_in[i]"] + ["%.8ef" % float(a) for a in extra_args])
    return _ENTRY % {"body": body, "wg": int(workgroup), "call": "%s(%s)" % (name, args)}


def run_kernel(body, name, data, extra_args=(), workgroup=64):
    """Dispatch an emitted kernel over `data` (1-D float array) and return the result as float32.

    Raises ImportError with an install hint when wgpu is absent -- never silently falls back to NumPy,
    because a caller who asked for the device path deserves to know they did not get it. (The CuPy backend
    falls back silently; that is right for a transparent accelerator and wrong for an explicit request.)"""
    try:
        from wgpu.utils.compute import compute_with_buffers
    except ImportError as exc:
        raise ImportError("wgpu is required to run WGSL kernels (pip install wgpu)") from exc

    arr = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
    if arr.ndim != 1:
        raise ValueError("run_kernel takes a 1-D array, got shape %r" % (arr.shape,))
    n = int(arr.size)
    groups = -(-n // int(workgroup))                  # round UP; the bounds guard handles the tail
    shader = wrap_kernel(body, name, extra_args=extra_args, workgroup=workgroup)
    out = compute_with_buffers({0: arr}, {1: (n, "f")}, shader, n=(groups, 1, 1))
    return np.frombuffer(out[1], dtype=np.float32).copy()


#: WGSL for a workgroup reduction: shared memory + barrier + tree halving, one partial per workgroup.
#: The %(init)s / %(combine)s slots are what make it sum-or-max rather than two near-identical shaders.
_REDUCE = """
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
var<workgroup> scratch: array<f32, %(wg)d>;

@compute @workgroup_size(%(wg)d)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let i = gid.x;
  // OUT-OF-RANGE LANES GET THE IDENTITY, not a skipped write: every lane must write scratch before the
  // barrier, or the tree below folds in whatever was left in workgroup memory from a previous dispatch.
  scratch[lid.x] = select(%(init)sf, inp[i], i < arrayLength(&inp));
  workgroupBarrier();
  var s: u32 = %(half)du;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) { scratch[lid.x] = %(combine)s; }
    workgroupBarrier();                      // EVERY lane reaches this, including those that skipped the
    s = s / 2u;                              // write above -- a barrier inside the `if` would deadlock.
  }
  if (lid.x == 0u) { outp[wid.x] = scratch[0]; }
}
"""

#: (identity, combine-expression) per op. `a` is scratch[lid.x], `b` is its partner scratch[lid.x + s].
_OPS = {
    "sum": ("0.0", "scratch[lid.x] + scratch[lid.x + s]"),
    "max": ("-3.4028235e38", "max(scratch[lid.x], scratch[lid.x + s])"),
    "min": ("3.4028235e38", "min(scratch[lid.x], scratch[lid.x + s])"),
}


def reduce_kernel(op, data, workgroup=64):
    """Reduce a 1-D float array on the device: 'sum' | 'max' | 'min'. Returns a Python float.

    TWO-STAGE BY DESIGN. Each workgroup reduces its own slice in shared memory and writes ONE partial; the
    host finishes the (much shorter) partial array. A single-pass whole-array reduction would need either a
    grid-wide barrier -- which WGSL does not have -- or atomics, which are float-nondeterministic and would
    put this on the wrong side of the engine's determinism rule. Two stages keeps the device work
    order-defined within a workgroup and leaves the cross-workgroup order to the host, where it is a plain
    NumPy reduction.

    WHY THIS UNLOCKS THE VSA HALF: `holographic_wgpurun` shipped elementwise maps only, which serves the
    rendering kernels and none of `bundle` / `cleanup` / `resonator` / `amp` / `htcodebook` -- every one of
    which is a cross-invocation reduction. This is that missing primitive.

    NOT BIT-EXACT WITH NUMPY IN GENERAL, and the reason is the same as everywhere else on this path: the
    device sums in tree order and f32 throughout, NumPy sums pairwise in float64 for a float64 input. Equal
    for exactly-representable data, close otherwise. Use a float32 input if you want to compare like for
    like, and never assume -- measure."""
    if op not in _OPS:
        raise ValueError("op must be one of %s, got %r" % (", ".join(sorted(_OPS)), op))
    try:
        from wgpu.utils.compute import compute_with_buffers
    except ImportError as exc:
        raise ImportError("wgpu is required to run WGSL kernels (pip install wgpu)") from exc

    arr = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
    if arr.ndim != 1:
        raise ValueError("reduce_kernel takes a 1-D array, got shape %r" % (arr.shape,))
    if arr.size == 0:
        raise ValueError("reduce_kernel needs at least one element")

    wg = int(workgroup)
    init, combine = _OPS[op]
    groups = -(-int(arr.size) // wg)
    shader = _REDUCE % {"wg": wg, "half": wg // 2, "init": init, "combine": combine}
    out = compute_with_buffers({0: arr}, {1: (groups, "f")}, shader, n=(groups, 1, 1))
    partials = np.frombuffer(out[1], dtype=np.float32)
    finish = {"sum": np.sum, "max": np.max, "min": np.min}[op]
    return float(finish(partials))


def argmax_kernel(data, workgroup=64):
    """Device argmax over a 1-D float array -> (index, value).

    ARGMAX IS A DECISION, NOT A VALUE, and that changes what this is allowed to do. The engine's rule is that
    existing decisions never flip, so this deliberately does the VALUE reduction on the device and resolves
    the INDEX on the host: it finds the max with `reduce_kernel`, then takes the FIRST host-side index
    attaining it. That makes ties resolve by lowest index -- the same canonical rule
    `determinism.argmax_tiebreak` uses -- rather than by whichever workgroup happened to finish first.

    A fully device-side argmax reducing (value, index) pairs is possible and is NOT built here: it would make
    tie resolution depend on reduction order, which is precisely the property that must not vary. If a
    measurement ever shows the host-side finish dominating, that is the moment to revisit -- with a tie test
    on ADVERSARIAL near-ties, not random data, because random data almost never ties."""
    arr = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
    peak = reduce_kernel("max", arr, workgroup=workgroup)
    hits = np.flatnonzero(arr == np.float32(peak))
    if hits.size == 0:                       # f32 rounding in the tree can land between representable values
        return int(np.argmax(arr)), float(arr.max())
    return int(hits[0]), float(peak)


#: One workgroup PER ROW: each reduces D products to a single dot, so an M x D matvec is M workgroup
#: reductions and needs no cross-workgroup communication at all. The rows are independent, which is what
#: makes this the shape a GPU actually wants.
_MATVEC = """
@group(0) @binding(0) var<storage, read> mat: array<f32>;
@group(0) @binding(1) var<storage, read> vec: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
var<workgroup> scratch: array<f32, %(wg)d>;

@compute @workgroup_size(%(wg)d)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let row = wid.x;
  let base = row * %(dim)du;
  // STRIDED WALK, not one element per lane: D is usually far larger than the workgroup, so each lane
  // accumulates its own partial across the row before the tree runs. This is why the kernel does not care
  // whether D is a multiple of the workgroup size.
  var acc: f32 = 0.0;
  var k: u32 = lid.x;
  loop {
    if (k >= %(dim)du) { break; }
    acc = acc + mat[base + k] * vec[k];
    k = k + %(wg)du;
  }
  scratch[lid.x] = acc;
  workgroupBarrier();
  var s: u32 = %(half)du;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  if (lid.x == 0u) { outp[row] = scratch[0]; }
}
"""


def matvec_kernel(matrix, vector, workgroup=64):
    """Compute `matrix @ vector` on the device for a 2-D (M, D) matrix. Returns a float32 array of length M.

    THIS IS THE KERNEL THAT ACTUALLY MATTERS FOR VSA. Measured on CPU, the codebook similarity is 98-100% of
    a cleanup's cost at any real codebook size (M=1024 D=512 -> 98%; M=4096 D=1024 -> 100%), while the argmax
    is single-digit microseconds. Offloading the argmax alone can never pay on ANY device -- NumPy's argmax
    does not reach a plausible dispatch floor until M ~ 500,000, and real codebooks are 64..4096. The
    similarity is the work.

    ONE WORKGROUP PER ROW, and the rows never talk to each other, so there is no cross-workgroup reduction
    and no grid-wide barrier (which WGSL does not have). Each lane walks its row with a stride and the tree
    folds the lanes -- so D need not be a multiple of the workgroup size."""
    try:
        from wgpu.utils.compute import compute_with_buffers
    except ImportError as exc:
        raise ImportError("wgpu is required to run WGSL kernels (pip install wgpu)") from exc

    mat = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
    vec = np.ascontiguousarray(np.asarray(vector, dtype=np.float32))
    if mat.ndim != 2:
        raise ValueError("matvec_kernel needs a 2-D matrix, got shape %r" % (mat.shape,))
    if vec.ndim != 1 or vec.size != mat.shape[1]:
        raise ValueError("vector length %r does not match matrix columns %r" % (vec.shape, mat.shape))
    rows, dim = int(mat.shape[0]), int(mat.shape[1])
    wg = int(workgroup)
    shader = _MATVEC % {"wg": wg, "half": wg // 2, "dim": dim}
    out = compute_with_buffers({0: mat.reshape(-1), 1: vec}, {2: (rows, "f")}, shader, n=(rows, 1, 1))
    return np.frombuffer(out[2], dtype=np.float32).copy()


#: One workgroup per OUTPUT ELEMENT (row r, query q): a direct extension of the matvec's one-per-row, and
#: still no cross-workgroup communication. Workgroups are dispatched as a 2-D grid (rows x queries).
_MATMUL = """
@group(0) @binding(0) var<storage, read> mat: array<f32>;
@group(0) @binding(1) var<storage, read> qs: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
var<workgroup> scratch: array<f32, %(wg)d>;

@compute @workgroup_size(%(wg)d)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let row = wid.x;
  let qi  = wid.y;
  var acc: f32 = 0.0;
  var k: u32 = lid.x;
  loop {
    if (k >= %(dim)du) { break; }
    acc = acc + mat[row * %(dim)du + k] * qs[qi * %(dim)du + k];
    k = k + %(wg)du;
  }
  scratch[lid.x] = acc;
  workgroupBarrier();
  var s: u32 = %(half)du;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  if (lid.x == 0u) { outp[qi * %(rows)du + row] = scratch[0]; }
}
"""


def matmul_kernel(matrix, queries, workgroup=64):
    """`queries @ matrix.T` for a (M, D) matrix and a (K, D) query stack -> (K, M) float32.

    THE BATCHED FORM, AND IT IS THE ONE THAT PAYS. Building the single-query matvec first was the SAME
    MISTAKE this path has now made three times: the natural unit of work is smaller than the dispatch floor.
    Measured on CPU at M=1024, D=512:
        ONE query      0.095 ms   -- below any plausible dispatch floor; can never pay
        256 queries    2.98 ms    -- comfortably above it
    So a caller cleaning up one cue should stay on the CPU, and a caller cleaning up a batch is the one with
    something to gain. `wgsl_matvec` remains the K=1 special case and is kept for exactly that.

    ONE WORKGROUP PER OUTPUT ELEMENT (row, query), dispatched as a 2-D grid -- a direct extension of the
    matvec's one-per-row, with still no cross-workgroup communication."""
    try:
        from wgpu.utils.compute import compute_with_buffers
    except ImportError as exc:
        raise ImportError("wgpu is required to run WGSL kernels (pip install wgpu)") from exc

    mat = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
    qs = np.ascontiguousarray(np.asarray(queries, dtype=np.float32))
    if mat.ndim != 2 or qs.ndim != 2:
        raise ValueError("matmul_kernel needs 2-D matrix and queries, got %r and %r" % (mat.shape, qs.shape))
    if mat.shape[1] != qs.shape[1]:
        raise ValueError("query width %d does not match matrix columns %d" % (qs.shape[1], mat.shape[1]))
    rows, dim, n_q = int(mat.shape[0]), int(mat.shape[1]), int(qs.shape[0])
    wg = int(workgroup)
    shader = _MATMUL % {"wg": wg, "half": wg // 2, "dim": dim, "rows": rows}
    out = compute_with_buffers({0: mat.reshape(-1), 1: qs.reshape(-1)}, {2: (rows * n_q, "f")},
                               shader, n=(rows, n_q, 1))
    return np.frombuffer(out[2], dtype=np.float32).reshape(n_q, rows).copy()


#: One workgroup per OUTPUT ELEMENT (batch row r, output index n), reducing over k. Identical shape to the
#: matmul kernel -- the third use of the same workgroup-reduction pattern, which is why this was buildable
#: as an M item where a Stockham FFT would have been an L one.
_CONV = """
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
var<workgroup> scratch: array<f32, %(wg)d>;

@compute @workgroup_size(%(wg)d)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let n   = wid.x;                       // output index within the row
  let row = wid.y;                       // which pair in the batch
  let base = row * %(dim)du;
  var acc: f32 = 0.0;
  var k: u32 = lid.x;
  loop {
    if (k >= %(dim)du) { break; }
    // (n - k) mod D, computed in unsigned arithmetic: adding D before subtracting avoids the wrap that
    // makes a u32 subtraction underflow into a huge index.
    let j = (n + %(dim)du - k) %% %(dim)du;
    acc = acc + a[base + k] * b[base + j];
    k = k + %(wg)du;
  }
  scratch[lid.x] = acc;
  workgroupBarrier();
  var s: u32 = %(half)du;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  if (lid.x == 0u) { outp[base + n] = scratch[0]; }
}
"""


def bind_batch_kernel(a_stack, b_stack, workgroup=64):
    """Circular-convolution BIND over stacks of shape (K, D) on any GPU -> (K, D) float32.

    WHY DIRECT CONVOLUTION AND NOT AN FFT. `bind` IS a plain circular convolution -- verified against the
    shipped bind() to 7e-15 -- so it can be computed either as rfft->multiply->irfft in O(D log D), or
    DIRECTLY in O(D^2). The direct form is ~100x more arithmetic at D=1024 (268M vs 2.6M ops for a batch of
    256) and is the RIGHT TRADE HERE for two reasons:
      * IT IS THE SAME WORKGROUP-REDUCTION SHAPE already used by the matvec and matmul kernels -- proven,
        tail-safe, and free of the bit-reversal, twiddle tables and multi-stage barriers a Stockham FFT
        needs. That turned an L item into an M one.
      * ARITHMETIC IS WHAT A GPU HAS. 100x more MACs on a device with thousands of lanes is a different
        proposition from 100x more work on a CPU.
    THAT TRADE IS NOT MEASURED. Whether 100x more arithmetic is recovered by parallelism depends entirely on
    the device, and llvmpipe is a CPU adapter -- so correctness is established here and the crossover is
    not. If a real device shows direct convolution losing to the CPU FFT, the Stockham route is the fallback
    and its design constraints are already recorded (one dispatch per row, D <= 4096 fits shared memory).

    BATCHED ON PURPOSE, for the third time on this path: a SINGLE bind costs ~0.03 ms on CPU at D=1024,
    below any plausible dispatch floor. The batch is the only shape that can pay."""
    try:
        from wgpu.utils.compute import compute_with_buffers
    except ImportError as exc:
        raise ImportError("wgpu is required to run WGSL kernels (pip install wgpu)") from exc

    a = np.ascontiguousarray(np.asarray(a_stack, dtype=np.float32))
    b = np.ascontiguousarray(np.asarray(b_stack, dtype=np.float32))
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError("bind_batch_kernel needs two matching (K, D) stacks, got %r and %r"
                         % (a.shape, b.shape))
    rows, dim = int(a.shape[0]), int(a.shape[1])
    wg = int(workgroup)
    shader = _CONV % {"wg": wg, "half": wg // 2, "dim": dim}
    out = compute_with_buffers({0: a.reshape(-1), 1: b.reshape(-1)}, {2: (rows * dim, "f")},
                               shader, n=(dim, rows, 1))
    return np.frombuffer(out[2], dtype=np.float32).reshape(rows, dim).copy()


def cleanup_batch_kernel(codebook, queries, workgroup=64):
    """Cleanup for a STACK of queries -> (indices, scores), one per query.

    The shape a device can actually win on (see matmul_kernel). Indices are resolved host-side by lowest
    index, exactly as the single-query path does, so the canonical tie rule is unchanged by batching."""
    sims = matmul_kernel(codebook, queries, workgroup=workgroup)
    idx = np.array([int(np.flatnonzero(row == np.float32(row.max()))[0]) for row in sims], dtype=int)
    return idx, sims[np.arange(len(idx)), idx]


def cleanup_kernel(codebook, query, workgroup=64):
    """A full VSA cleanup on the device -> (index, score): similarity THEN argmax, in one dispatch.

    FUSED ON PURPOSE. Running the matvec and the argmax as two dispatches would pay the submission cost
    twice and ship the M-length intermediate back across the bus in between -- the same 'fuse before you
    dispatch' rule that governs shader_pipeline and that should_offload's round_trips gate encodes. Here the
    fusion is free: the argmax finish is host-side anyway.

    THE INDEX IS RESOLVED ON THE HOST, DELIBERATELY. An argmax is a DECISION, not a value, and this engine's
    rule is that existing decisions never flip -- so ties break by LOWEST INDEX (the canonical
    determinism.argmax_tiebreak rule) rather than by whichever workgroup finished first. That costs
    microseconds at any real codebook size, which is precisely why it is not worth moving.

    THE RESIDUAL RISK, MEASURED, AND ITS MITIGATION. The matvec is NOT bit-exact -- the device sums each row
    in tree order in f32, NumPy sums in a different order -- so a similarity near-tie CAN flip the decision.
    The boundary was measured with independent rows tuned to a controlled gap:
        gap 1e-3, 1e-5, 1e-6   ->  0/150 disagreements with CPU
        gap 1e-7               ->  3/150 DISAGREE
    (Near-DUPLICATE atoms do NOT flip, 0/120, because the rounding error is CORRELATED across near-identical
    rows and largely cancels in their difference. It is INDEPENDENT rows landing coincidentally close that
    are dangerous, which is the opposite of the intuition.)
    A flip needs a gap at or below ~1e-7, which is FOUR ORDERS BELOW the smallest tie margin anyone would
    sensibly use (1e-3). So `tied_candidates` reports both candidates for every gap that could flip, and a
    caller using it never receives a silent difference -- it receives a declared ambiguity, which is exactly
    what candidate sets were built for. Use them when running cleanup on a device."""
    sims = matvec_kernel(codebook, query, workgroup=workgroup)
    peak = float(sims.max())
    hits = np.flatnonzero(sims == np.float32(peak))
    index = int(hits[0]) if hits.size else int(np.argmax(sims))
    return index, float(sims[index])


def verify_against_numpy(fn, data, extra_args=(), dialect="wgsl", workgroup=64):
    """Run `fn` BOTH ways -- Python on CPU and its WGSL projection on the device -- and report the deviation.

    THE POINT OF THE PROJECTION DESIGN, made executable. Because the shader is generated from the same
    function that runs on CPU, they can be differentially tested on real data instead of trusted. Returns
    {max_abs, max_rel, exact, n}. A CuPy kernel cannot be checked this way: there is no shared source, only
    two implementations that are supposed to agree."""
    from holographic.io_and_interop.holographic_emit import emit

    # THE REFERENCE MUST BE EVALUATED ON THE SAME INPUTS THE DEVICE SAW. A first version cast the array to
    # float32 for dispatch but ran Python on the ORIGINAL float64 values, so the "deviation" it reported was
    # the INPUT cast, not the kernel -- it made a bit-exact kernel look inexact and would have sent someone
    # hunting a device bug that did not exist. Casting first isolates the thing being tested.
    arr = np.asarray(data, dtype=np.float32)
    # THE SCALARS MUST BE CAST TOO, for the same reason the array is. `extra_args` are emitted as f32
    # LITERALS in the shader, so passing float64 to the Python side means the two sides evaluate DIFFERENT
    # CONSTANTS -- 1.7 is not exactly representable in f32, and that alone made an otherwise bit-exact
    # kernel report as inexact. This is the second instance of one root cause: A DIFFERENTIAL TEST MUST
    # FEED BOTH SIDES IDENTICAL INPUTS, or it measures its own conversions instead of the kernel.
    args = [float(np.float32(a)) for a in extra_args]
    body = emit(fn, dialect)
    got = run_kernel(body, fn.__name__, arr, extra_args=args, workgroup=workgroup)
    ref = np.asarray([fn(float(x), *args) for x in arr], dtype=np.float32)
    diff = np.abs(got - ref)
    denom = np.maximum(np.abs(ref), 1e-30)
    return {"max_abs": float(diff.max()), "max_rel": float((diff / denom).max()),
            "exact": bool(np.array_equal(got, ref)), "n": int(arr.size)}


def _selftest():
    info = device_info()
    if not info.get("available"):
        print("holographic_wgpurun: no adapter (%s) -- selftest skipped, which is the correct fallback"
              % info.get("why", "?"))
        return

    # 1. THE WRAPPER PRODUCES A RUNNABLE SHADER, not just valid WGSL.
    body = "fn scale(x: f32, g: f32) -> f32 { return ((x * g) + 0.5f); }"
    shader = wrap_kernel(body, "scale", extra_args=(2.0,))
    assert "@compute" in shader and "arrayLength" in shader

    # 2. END TO END, and against NumPy rather than against expectation.
    data = np.arange(256, dtype=np.float32)
    got = run_kernel(body, "scale", data, extra_args=(2.0,))
    ref = data * 2.0 + 0.5
    assert np.array_equal(got, ref), "max dev %r" % float(np.abs(got - ref).max())

    # 3. THE TAIL. A size that is not a whole number of workgroups must not lose or corrupt elements --
    #    this is what the arrayLength guard is for, and it is the easiest thing to get silently wrong.
    odd = np.arange(100, dtype=np.float32)
    assert np.array_equal(run_kernel(body, "scale", odd, extra_args=(2.0,)), odd * 2.0 + 0.5)

    # 4. Refusals.
    try:
        run_kernel(body, "scale", np.zeros((4, 4), dtype=np.float32))
        raise AssertionError("accepted a 2-D array")
    except ValueError:
        pass

    print("holographic_wgpurun: all selftests passed on %s (%s) -- bit-exact vs NumPy"
          % (info["device"], info["type"]))


if __name__ == "__main__":
    _selftest()
