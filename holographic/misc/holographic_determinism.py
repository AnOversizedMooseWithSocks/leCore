"""The determinism contract, made executable (ISA-1): the ONE place the engine's tie-break and sign
conventions live, so they are cited rather than re-invented per module.

WHY THIS EXISTS
---------------
A VSA instruction set, like any ISA, is only durable if the EXACT observable semantics of its base operations
are a frozen contract while the implementations (FFT, BLAS, the forest) vary underneath. The `bind_batch` bug
is the cautionary tale: a microarchitecture change (batched BLAS, bit-exact to 1e-12) flipped a creature's
maze trajectory because it changed a summation order the contract never pinned -- an under-specified tie-break
leaking through. The fix is not "never vectorize"; it is "write down the observable decision and pin it."

The audit (June 2026) found the cost of NOT having this one home: the determinism behaviour was specified four
different ways across modules --
  * `holographic_ai.cleanup` leans on numpy's implicit `argmax` (ties resolve to the LOWEST index -- correct,
    but written nowhere);
  * `holographic_spectral.sign_fix` invented "each eigenvector's largest-magnitude component is positive",
    explicitly citing "the same bit-exact-tie class as the bind_batch bug";
  * `holographic_flow` carries its own weighted Laplacian rather than share one (a different summation order
    "could flip a trajectory");
  * `holographic_chart._fix_signs` (RT-II1, just shipped) RE-invented the very same sign rule as a private copy
    rather than sharing it -- the fourth instance, added while the contract was still missing.
Same bug class, re-litigated four times, with code duplication as the price. This module ends that: `spectral`
and `chart` now CITE `fix_eigvec_signs` here; the argmax tie-break has a name; the rules are stated once.

THE ARCHITECTURE / MICROARCHITECTURE BOUNDARY (the whole point):
  * ARCHITECTURE = the observable decision a caller depends on -- WHICH atom `cleanup` returns (the argmax),
    that `unbind` inverts `bind` exactly, that an eigenbasis has a fixed sign. These are pinned EXACTLY.
  * MICROARCHITECTURE = how the continuous numbers are computed -- the FFT vs a direct convolution, a batched
    vs a looped reduction. These may vary within a stated numeric tolerance, because no caller can observe the
    last bit of a reduction -- only the decision it feeds.
The rules below pin the architecture. ISA.md states the full contract per instruction in prose.

Pure NumPy, no state, deterministic by construction.
"""

import numpy as np


# =================================================================================================
# THE SIGN RULE (reconciles spectral.sign_fix and chart._fix_signs into one cited implementation).
# =================================================================================================
def fix_eigvec_signs(V, copy=True):
    """Pin the sign of each column of `V` (an eigenvector / embedding-axis matrix, shape [n, k]) so that its
    largest-magnitude entry is non-negative. This removes the sign ambiguity `numpy.linalg.eigh` leaves (an
    eigenvector and its negation are both valid), making any eigenbasis or spectral embedding bit-stable run to
    run -- the same tie class the `bind_batch` bug lives in.

    THE RULE, stated once (this is the contract):
      * Operate COLUMN-WISE (each eigenvector independently).
      * The pivot entry is `argmax(|column|)` -- and ties in magnitude resolve to the LOWEST index (numpy's
        argmax convention; see `argmax_tiebreak`), so the pivot choice is itself deterministic.
      * If the pivot entry is negative, negate the whole column.

    `copy=True` (default) returns a new array and leaves the input untouched (what `chart` relies on);
    `copy=False` fixes signs in place and returns the same array (what `spectral.sign_fix` has always done).
    Both are bit-identical in their output. NOTE: this does NOT resolve the basis WITHIN a degenerate
    eigenspace (equal eigenvalues leave the basis rotation-ambiguous) -- that is a deeper, documented limit, not
    something a sign flip can fix."""
    V = np.asarray(V, float)
    if copy:
        V = V.copy()
    for j in range(V.shape[1]):
        i = int(np.argmax(np.abs(V[:, j])))          # lowest-index tie-break -> deterministic pivot
        if V[i, j] < 0:
            V[:, j] = -V[:, j]
    return V


# =================================================================================================
# THE ARGMAX TIE-BREAK (names the convention cleanup has always used implicitly, so it is citable).
# =================================================================================================
def argmax_tiebreak(a, axis=None):
    """The engine's argmax convention, named: the index of the maximum, with ties resolved to the LOWEST index.
    This is exactly what `numpy.argmax` does, but giving it a name makes the contract explicit and citable --
    `cleanup`'s `int(sims.argmax())` is THIS rule, and any conformance test for a cleanup-style decision pins it
    here rather than re-deriving "ties go to the lowest index" in prose each time.

    Why it matters: the argmax IS the observable architectural decision (which atom is recalled). Two
    implementations of the similarity scan may differ in the last bit of the dot products (microarchitecture),
    but they must agree on this index (architecture) -- and when two scores are EXACTLY equal, "lowest index"
    is the frozen rule that makes that agreement well-defined."""
    return int(np.argmax(a)) if axis is None else np.argmax(a, axis=axis)


# =================================================================================================
# ==================================================================================================================
# A4/D1 -- STATELESS, COORDINATE-KEYED RANDOMNESS.
#
# `np.random.default_rng(seed)` carries STATE: the n-th draw depends on every draw before it. That is fatal for work
# you want to split across a farm, because bucket order then changes the numbers, and it forces every node to agree
# on a seed and a draw count. The fix is to stop drawing and start LOOKING UP: make the random value a pure function
# of WHERE and WHICH -- hash_unit(x, y, walk, step, seed). Same inputs, same value, on any node, in any order, with
# no coordination at all. (This is what the shadertoy/renderer world calls "hash noise", and what PBRT's stateless
# sampler does.)
#
# Pure integer arithmetic (the same family as `pattern._hash01`), so it is reproducible to the bit and independent of
# PYTHONHASHSEED -- never Python's salted hash(). Floats are keyed by their exact BIT PATTERN, so two different
# coordinates never collide by rounding.
# ==================================================================================================================
_HASH_ODD = np.uint64(0x9E3779B97F4A7C15)          # golden-ratio odd constant (splitmix64's increment)


def _fold_str(text):
    """A deterministic uint64 for a STRING key (a domain separator like "sphere_z"). An FNV-1a byte fold in pure
    integer arithmetic -- NEVER Python's hash(), which is salted per process and would break reproducibility."""
    h = np.uint64(0xCBF29CE484222325)                                  # FNV offset basis
    with np.errstate(over="ignore"):
        for byte in text.encode("utf-8"):
            h = (h ^ np.uint64(byte)) * np.uint64(0x100000001B3)       # FNV prime
    return h


def _as_u64(key):
    """Turn one key into uint64 bits: a str is folded deterministically (a domain separator); an int keeps its value;
    a float is keyed by its IEEE-754 BIT PATTERN (so 0.1 and 0.100000000000001 are different keys, as they must be);
    an array keeps its shape so keys broadcast together."""
    if isinstance(key, str):
        return _fold_str(key)
    a = np.asarray(key)
    if a.dtype.kind in "US":                                           # an array of strings: fold each
        return np.array([_fold_str(str(x)) for x in a.ravel()], dtype=np.uint64).reshape(a.shape)
    if a.dtype.kind == "f":
        return a.astype(np.float64).view(np.uint64)
    return a.astype(np.int64).view(np.uint64)


def _mix64(x):
    """splitmix64's finalizer: an avalanche mixer -- one input bit flips ~half the output bits."""
    with np.errstate(over="ignore"):                                   # uint64 wraparound IS the arithmetic
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def hash_u64(*keys):
    """A deterministic uint64 hash of any tuple of keys (ints, floats, or numpy arrays -- broadcast together)."""
    with np.errstate(over="ignore"):
        h = np.uint64(0)
        for k in keys:
            h = _mix64(h + _HASH_ODD + _as_u64(k))                     # sequentially absorb each key
        return _mix64(h)


def hash_unit(*keys):
    """A uniform float in [0, 1), a PURE FUNCTION of the keys -- stateless randomness.

    hash_unit(x, y, walk, step, seed) gives the same number on every node, in any order, with no seed coordination
    and no draw counter. Use it wherever a sample is indexed by WHERE it is rather than by HOW MANY came before:
    walk-on-spheres steps, path-tracer bounces, per-pixel jitter, farm buckets. Returns a scalar for scalar keys and
    an array for array keys (they broadcast). 53 bits of mantissa, so the values are as fine-grained as a double."""
    h = hash_u64(*keys)
    out = (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)    # 53 bits -> [0,1)
    return float(out) if np.isscalar(out) or out.ndim == 0 else out


def hash_direction(*keys, dim=3):
    """A uniform direction on the unit sphere (dim=3) or circle (dim=2), keyed statelessly. Uses the area-preserving
    map for the sphere (uniform in cos(theta)), so directions are genuinely uniform, not clustered at the poles."""
    if dim == 2:
        a = 2.0 * np.pi * hash_unit("circle", *keys)
        return np.stack([np.cos(a), np.sin(a)], axis=-1)
    if dim != 3:
        raise ValueError("hash_direction supports dim=2 or dim=3, got %r" % dim)
    z = 2.0 * hash_unit("sphere_z", *keys) - 1.0                       # uniform in cos(theta) == equal area
    a = 2.0 * np.pi * hash_unit("sphere_a", *keys)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack([r * np.cos(a), r * np.sin(a), z], axis=-1)


def _selftest():
    """The sign rule is deterministic, idempotent, and obeys its stated convention; the argmax tie-break picks
    the lowest index on an exact tie."""
    rng = np.random.default_rng(0)
    V = rng.standard_normal((20, 5))

    fixed = fix_eigvec_signs(V)
    # every column's largest-magnitude entry is now non-negative
    for j in range(fixed.shape[1]):
        i = int(np.argmax(np.abs(fixed[:, j])))
        assert fixed[i, j] >= 0
    # idempotent: fixing an already-fixed matrix changes nothing
    assert np.array_equal(fix_eigvec_signs(fixed), fixed)
    # deterministic: same input -> same output, every time
    assert np.array_equal(fix_eigvec_signs(V), fix_eigvec_signs(V))
    # sign-invariant: V and -V map to the SAME fixed basis (the ambiguity is removed)
    assert np.allclose(fix_eigvec_signs(V), fix_eigvec_signs(-V))
    # copy=True leaves the input untouched; copy=False mutates in place (both same output)
    Vc = V.copy()
    out_copy = fix_eigvec_signs(Vc, copy=True)
    assert np.array_equal(Vc, V)                      # untouched
    out_inplace = fix_eigvec_signs(Vc, copy=False)
    assert out_inplace is Vc and np.array_equal(out_inplace, out_copy)

    # argmax tie-break: exact tie -> lowest index
    assert argmax_tiebreak(np.array([1.0, 3.0, 3.0, 2.0])) == 1

    print("holographic_determinism: ok")


if __name__ == "__main__":
    _selftest()
