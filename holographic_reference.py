"""Reference implementations + the conformance harness (ISA-2): the teeth of the ISA contract (ISA.md).

WHY THIS EXISTS
---------------
ISA.md is the written contract; this is its enforcement. For each base instruction there is a DEFINITIONAL
reference implementation -- the simplest, slowest, obviously-correct version (e.g. `bind` as a direct O(D^2)
circular convolution, not an FFT) -- and a conformance check that the production kernel matches it. The split
ISA-1 introduced is the whole point:

  * VALUE conformance (TOL): a CONTINUOUS output (bind, unbind, bundle, cosine) must match the reference within
    a numeric tolerance. The FFT and the direct convolution agree to machine epsilon; a future batched `bundle`
    would too. The last bit of a reduction is microarchitecture -- no caller observes it.
  * DECISION conformance (EXACT): an OBSERVABLE decision (which atom `cleanup` picks; an exact reindex like
    `permute`; the self-inverse `involution`) must match EXACTLY, ties resolved by the contract's rule
    (`argmax_tiebreak`, lowest index).

This is what makes §7's vectorization safe to pursue: a vectorized op is "conformant" iff it passes here, and
the bind_batch class -- a value-conformant change that flips a DECISION -- is caught by construction, because
the decision is checked separately and exactly. The `test_isa_conformance.py` regression proves it.

Pure NumPy, deterministic.
"""

import numpy as np

from holographic_determinism import argmax_tiebreak

# Numeric tolerance for CONTINUOUS (TOL) outputs. EXACT outputs/decisions are compared bit-for-bit (tol 0).
TOL = 1e-9


# =================================================================================================
# Definitional reference implementations -- the simplest, obviously-correct version of each base op.
# (Verified against the production kernel to machine epsilon; these are the "golden" definitions.)
# =================================================================================================
def ref_bind(a, b):
    """Circular convolution by its DEFINITION: (a * b)[n] = sum_k a[k] b[(n-k) mod D]. O(D^2), no FFT."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    D = len(a)
    out = np.zeros(D)
    for n in range(D):
        acc = 0.0
        for k in range(D):
            acc += a[k] * b[(n - k) % D]
        out[n] = acc
    return out


def ref_involution(a):
    """The reversal used to invert bind: inv[0] = a[0], inv[i] = a[D-i]. Exactly self-inverse."""
    a = np.asarray(a, float)
    D = len(a)
    inv = np.empty(D)
    inv[0] = a[0]
    for i in range(1, D):
        inv[i] = a[D - i]
    return inv


def ref_unbind(composite, a):
    """Unbind by its definition: bind the composite with the involution of the key."""
    return ref_bind(composite, ref_involution(a))


def ref_permute(vec, shift):
    """Cyclic shift by its definition (numpy's roll convention): out[i] = vec[(i - shift) mod D]. Exact."""
    vec = np.asarray(vec)
    D = len(vec)
    return np.array([vec[(i - shift) % D] for i in range(D)])


def ref_bundle(vectors):
    """Superpose then renormalize; a zero-sum bundle returns the zero vector (the pinned edge)."""
    total = np.sum(np.asarray(vectors, float), axis=0)
    norm = np.linalg.norm(total)
    return total / norm if norm > 0 else total


def ref_cosine(a, b):
    """dot / (|a| |b|); zero norm -> 0.0 (the pinned edge)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# =================================================================================================
# Conformance checks -- the TOL / EXACT split made callable.
# =================================================================================================
def value_conformant(x, y, tol=TOL):
    """True iff two CONTINUOUS outputs agree within numeric tolerance (the TOL class)."""
    return float(np.max(np.abs(np.asarray(x, float) - np.asarray(y, float)))) <= tol


def exact_conformant(x, y):
    """True iff two outputs are bit-for-bit identical (the EXACT class: a reindex, an involution)."""
    return np.array_equal(np.asarray(x), np.asarray(y))


def decision_conformant(sims_a, sims_b):
    """True iff two similarity vectors yield the SAME cleanup decision under the contract's tie-break. This is
    the check the bind_batch class fails: two value-conformant similarity vectors can still pick different
    atoms, and THAT is the observable error -- so the decision is checked separately and exactly."""
    return argmax_tiebreak(sims_a) == argmax_tiebreak(sims_b)


# =================================================================================================
# Run the suite: every production base op vs its definitional reference.
# =================================================================================================
def run_conformance(dim=64, seed=0, n_cases=8):
    """Check each production base instruction against its reference on random cases. Returns
    {op: {'passed': bool, 'class': 'TOL'|'EXACT', 'max_diff': float}}. The kernel is conformant iff every op
    passes (TOL ops within tolerance, EXACT ops bit-for-bit)."""
    from holographic_ai import bind, unbind, bundle, cosine, involution, permute, random_vector, bind_batch

    rng = np.random.default_rng(seed)
    report = {}

    def tol_case(name, prod, ref, cases):
        worst = 0.0
        ok = True
        for args in cases:
            p = prod(*args)
            r = ref(*args)
            worst = max(worst, float(np.max(np.abs(np.asarray(p, float) - np.asarray(r, float)))))
            ok = ok and value_conformant(p, r)
        report[name] = {"passed": ok, "class": "TOL", "max_diff": worst}

    def exact_case(name, prod, ref, cases):
        ok = all(exact_conformant(prod(*args), ref(*args)) for args in cases)
        report[name] = {"passed": ok, "class": "EXACT", "max_diff": 0.0}

    pairs = [(random_vector(dim, rng), random_vector(dim, rng)) for _ in range(n_cases)]
    singles = [(random_vector(dim, rng),) for _ in range(n_cases)]

    tol_case("bind", bind, ref_bind, pairs)
    tol_case("bind_batch", lambda a, b: bind_batch(a[None, :], b[None, :])[0], lambda a, b: ref_bind(a, b), pairs)
    tol_case("unbind", lambda a, b: unbind(bind(a, b), a), lambda a, b: ref_unbind(bind(a, b), a), pairs)
    tol_case("bundle", lambda a, b: bundle([a, b]), lambda a, b: ref_bundle([a, b]), pairs)
    tol_case("cosine", lambda a, b: cosine(a, b), lambda a, b: ref_cosine(a, b), pairs)
    exact_case("involution", involution, ref_involution, singles)
    exact_case("permute", lambda a: permute(a, 5), lambda a: ref_permute(a, 5), singles)

    return report


def _selftest():
    """Every base op conforms to its reference; bind_batch is value-conformant (the lesson: value-conformant is
    not decision-safe, which is why decisions are pinned separately)."""
    report = run_conformance()
    for op, r in report.items():
        assert r["passed"], f"{op} not conformant: {r}"
    assert report["bind"]["class"] == "TOL" and report["bind"]["max_diff"] < 1e-9
    assert report["permute"]["class"] == "EXACT" and report["permute"]["max_diff"] == 0.0

    # the bind_batch class, in miniature: two similarity vectors that are VALUE-conformant (differ < tol) but
    # pick DIFFERENT atoms -- value_conformant passes, decision_conformant fails (the suite catches it).
    sims_ref = np.array([0.5, 0.5, 0.3])         # a tie at index 0/1; contract picks the lower index, 0
    sims_bad = sims_ref + np.array([0.0, 1e-12, 0.0])  # a sub-tolerance bump flips the decision to 1
    assert value_conformant(sims_ref, sims_bad)   # within numeric tolerance -> would PASS a value-only check
    assert not decision_conformant(sims_ref, sims_bad)  # but the DECISION flipped -> the suite FAILS it

    print("holographic_reference: ok")


if __name__ == "__main__":
    _selftest()
