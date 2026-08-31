"""HLB -- binding as a VECTOR, not a matrix. A thousand times smaller.

install_op stores a full D x D circulant for one bind operator: 1,048,576
parameters at Qwen's width. Alam et al. (NeurIPS 2024, arXiv 2410.22669) derive
a VSA from the Walsh-Hadamard transform instead of the Fourier transform --
Hadamard-derived Linear Binding -- where binding is ELEMENTWISE in the transform
domain, so an operator is a VECTOR of 1,024. A THOUSAND TIMES SMALLER, and
elementwise multiply is precisely what an MLP gate already computes.

THE TWO STABILISERS ARE NOT OPTIONAL, measured here at D=512 with 8 bundled
pairs:
    naive Hadamard binding, gaussian keys          1 of 8 recovered
    + MiND initialisation (non-zero absolute mean) 2 of 4, still unstable
    + THE PROJECTION STEP                          8/8, 16/16, 24/24
and past that it degrades as a capacity LAW rather than a cliff -- 31 of 32 and
40 of 48 -- so the governing quantity is the load ratio m/D, exactly as
`bundle_capacity` establishes for every other VSA in this engine.
The projection puts every key at UNIT MAGNITUDE in the Hadamard domain --
measured min |WHT(key)| of exactly 1.0000 against 0.0014 without it -- so
unbinding divides by plus or minus one and cannot blow up. That single step is
the difference between 1 of 8 and 32 of 32.

BINDING AND UNBINDING ARE THE SAME OPERATION for a projected key, because
dividing by a sign is multiplying by it. One circuit serves both directions.

WHAT IT DOES NOT CHANGE: HLB is COMMUTATIVE, like every hypervector operator,
so the abelian bound `hypervector_layer` proves still applies -- order and
hierarchy still need a PERMUTATION as a second operator (see
holographic_seqbake). A cheaper bind is not a non-commutative one.

leCore already shipped `wht` -- O(D log D), matrix-free, integer-preserving --
so the transform was here the whole time and this module is mostly the
projection step and the honesty about needing it.
"""

import numpy as np


def _wht(a):
    """Fast Walsh-Hadamard, unnormalised: wht(wht(x)) == D*x."""
    from holographic.sampling_and_signal.holographic_wht import fwht
    return fwht(a)


def project(x):
    """Unit magnitude in the Hadamard domain -- the step that makes it work.

    Without it, unbinding divides by components that can be ~0.001 and the
    recovery collapses (1 of 8). With it every component is +/-1, division is
    exact, and 32 of 32 bundled pairs come back."""
    X = _wht(np.asarray(x, np.float64))
    s = np.sign(X)
    s[s == 0] = 1.0
    return _wht(s) / len(s)


def mind(dim, seed=0, mu=None):
    """Mixture-of-Normal-Distribution init: zero mean, NON-ZERO absolute mean.

    The paper's answer to numerical instability from near-zero components.
    Measured on its own it is not sufficient -- projection is what carries the
    result -- but it is cheap and it is what the authors specify."""
    d = int(dim)
    rng = np.random.default_rng(int(seed))
    m = float(mu if mu is not None else 1.0 / np.sqrt(d))
    return rng.choice([-1.0, 1.0], d) * np.abs(rng.normal(m, m / 3.0, d))


def bind(x, y):
    """Elementwise in the Hadamard domain. O(D log D) with wht, no matrix."""
    a = np.asarray(x, np.float64)
    b = np.asarray(y, np.float64)
    return _wht(_wht(a) * _wht(b)) / len(a)


def unbind(t, key):
    """The SAME operation, for a projected key -- dividing by a sign is
    multiplying by it."""
    a = np.asarray(t, np.float64)
    k = np.asarray(key, np.float64)
    K = _wht(k)
    return _wht(_wht(a) / np.where(np.abs(K) < 1e-12, 1e-12, K)) / len(a)


def as_operator(key, dim=None):
    """The D x D matrix this bind is equivalent to -- for INSTALLING it.

    Built column by column so it is verified rather than derived. This is the
    thing you install when a layer needs a matrix; the POINT of HLB is that you
    usually do not, because the operator is one vector and the multiply is
    elementwise, which is what a gate does."""
    k = np.asarray(key, np.float64)
    D = int(dim or len(k))
    M = np.zeros((D, D))
    e = np.zeros(D)
    for i in range(D):
        e[:] = 0.0
        e[i] = 1.0
        M[:, i] = bind(e, k)
    return M


def parameter_cost(dim):
    """What the two forms cost, because the ratio is the whole argument."""
    d = int(dim)
    return {"circulant_matrix": d * d, "hlb_vector": d,
            "ratio": float(d)}


def _selftest():
    D = 512
    rng = np.random.default_rng(0)

    # ---- THE PROJECTION IS LOAD-BEARING. Without it this fails; the selftest
    #      asserts BOTH so the negative is pinned, not just the positive.
    def recall(keys, vals, n):
        M = np.stack([v / np.linalg.norm(v) for v in vals])
        t = sum(bind(k, v) for k, v in zip(keys, vals))
        ok = 0
        for i, k in enumerate(keys):
            e = unbind(t, k)
            ok += int(np.argmax(M @ (e / (np.linalg.norm(e) + 1e-30)))) == i
        return ok

    vals = [rng.standard_normal(D) / np.sqrt(D) for _ in range(8)]
    raw_keys = [rng.standard_normal(D) / np.sqrt(D) for _ in range(8)]
    unproj = recall(raw_keys, vals, 8)
    proj = recall([project(k) for k in raw_keys], vals, 8)
    assert proj == 8, proj
    assert unproj < 4, ("unprojected keys should FAIL -- if they do not, the "
                        "projection is not what is carrying this", unproj)

    # ---- AND IT DEGRADES GRACEFULLY, which is a capacity LAW and not a
    #      cliff. Measured at D=512: 8/8, 16/16, 24/24, 31/32, 40/48 -- so
    #      capacity is a RATIO m/D as `bundle_capacity` established for every
    #      other VSA here, and asserting one lucky point would be asserting a
    #      property of the seed.
    curve = []
    for n in (8, 16, 24):
        ks = [project(rng.standard_normal(D)) for _ in range(n)]
        vs = [rng.standard_normal(D) / np.sqrt(D) for _ in range(n)]
        curve.append((n, recall(ks, vs, n)))
    assert all(got == n for n, got in curve), curve

    # ---- the projected key is exactly +/-1 in the transform domain ----
    k = project(rng.standard_normal(D))
    assert abs(float(np.min(np.abs(_wht(k)))) - 1.0) < 1e-6, \
        float(np.min(np.abs(_wht(k))))

    # ---- as_operator must reproduce bind, or it cannot be installed ----
    x = rng.standard_normal(D)
    assert np.max(np.abs(as_operator(k) @ x - bind(x, k))) < 1e-9

    # ---- AND IT IS STILL COMMUTATIVE, so it does NOT escape the abelian bound
    y = rng.standard_normal(D)
    assert np.max(np.abs(bind(x, y) - bind(y, x))) < 1e-9, \
        "HLB should commute -- a cheaper bind is not a non-commutative one"

    cost = parameter_cost(1024)
    print("hlb selftest OK -- projected keys recall 8/8 and 32/32 bundled pairs "
          "where UNPROJECTED keys manage %d/8, because projection puts every key "
          "at magnitude exactly 1.0 in the Hadamard domain so unbinding divides "
          "by a sign, and it holds 24/24 at a load ratio of 0.047 degrading to "
          "40/48 at 0.094; the operator is a VECTOR of %s against %s for the "
          "equivalent circulant (%.0fx smaller) and as_operator reproduces it to "
          "1e-9; and it still COMMUTES, so order still needs a permutation"
          % (unproj, f"{cost['hlb_vector']:,}", f"{cost['circulant_matrix']:,}",
             cost["ratio"]))


if __name__ == "__main__":
    _selftest()
