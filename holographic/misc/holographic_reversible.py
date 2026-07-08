"""The reversible / error-correction model of the VSA ISA (ISA-8 -- the frontier, Tier 4).

The honest framing: VSA assembly is NOT x86. It is a noisy, bounded, partly-REVERSIBLE instruction set. Read it
through the lens reversible/error-correcting computing already uses:
  * bind / unbind / permute / involution are (essentially) REVERSIBLE -- information-preserving rotations with
    an exact inverse (bind<->unbind given a unitary key; permute by -shift; involution is self-inverse).
  * bundle / superpose / cleanup are INFORMATION-DESTROYING -- they sum away (or project away) detail. This is
    where the "coherence budget" is spent.
  * `cleanup` is exactly ERROR CORRECTION: it snaps a drifted vector back onto the codebook manifold, discarding
    the accumulated error (along with a little signal). `capacity` is the coherence budget; re-anchoring /
    coherence-gating is an error-correction round.

THE LOUD NEGATIVE (do not overclaim the physics): this is an ANALOGY, not a claim that VSA is a quantum computer.
There is NO exponential superposition, NO physical entanglement, NO quantum speedup. FHRR's bind happens to be a
diagonal-unitary (a per-frequency phase rotation), which is structurally gate-like, and that is a useful framing
for capacity bounds -- but it is framing. What we actually BORROW is the discipline: track an error budget,
correct before the cliff, and keep reversibility bookkeeping. The practical, measured payoff is the auto-cleanup
SCHEDULER below; the reversibility audit and the quantum framing are conceptual scaffolding around it.
"""

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, unbind, bundle, cosine, permute, involution, random_vector


# ---- (a) the reversibility audit -----------------------------------------------------------------------------
# Each base instruction classified by whether it has an exact inverse. REVERSIBLE ops preserve information;
# LOSSY ops destroy it (and are exactly where crosstalk/coherence is spent or restored).
_REVERSIBILITY = {
    "bind":       ("reversible", "inverse is unbind by the same (unitary) key"),
    "unbind":     ("reversible", "inverse is bind by the same key"),
    "permute":    ("reversible", "inverse is permute by the negated shift"),
    "involution": ("reversible", "self-inverse: involution(involution(x)) == x"),
    "bundle":     ("lossy",      "sum: the summands are not exactly recoverable -- coherence spent here"),
    "superpose":  ("lossy",      "raw sum: same, without renormalization"),
    "cleanup":    ("lossy",      "projection to the nearest codebook atom: discards the residual -- this IS the "
                                 "error-correction step"),
}


def reversibility_class(op):
    """Return ('reversible'|'lossy', reason) for a base instruction name. The audit (ISA-8 part a)."""
    if op not in _REVERSIBILITY:
        raise ValueError(f"unknown instruction {op!r}; known: {sorted(_REVERSIBILITY)}")
    return _REVERSIBILITY[op]


def reversibility_audit():
    """The full classification table -- which instructions are reversible vs information-destroying."""
    return dict(_REVERSIBILITY)


# ---- (b) the error budget + auto-cleanup scheduler (the practical core) ---------------------------------------
def health(v, codebook):
    """The oracle-free error estimate: cosine of v to its NEAREST codebook atom. 1.0 when v is a clean atom;
    it falls as v drifts into the no-man's-land between atoms -- i.e. it tracks how much the coherence budget has
    been spent. (Reuses the capacity diagnostic's idea: nearness to the manifold is the SNR proxy.)"""
    sims = [cosine(v, c) for c in codebook]
    return float(np.max(sims)), int(np.argmax(sims))


def snap(v, codebook):
    """The cleanup / error-correction step: snap v to its nearest codebook atom (a verbatim copy)."""
    _, idx = health(v, codebook)
    return codebook[idx].copy(), idx


def auto_cleanup_run(initial, steps, codebook, floor=0.9, schedule="adaptive", k=3):
    """Run a 'program' -- a list of vector->vector `steps` -- under an error-correction policy, returning
    (final_vector, n_cleanups). Each step is applied, then a `cleanup` is inserted by the chosen policy:
      * 'adaptive' : clean only when health(v) < floor -- correct just before the cliff (generalizes the
                     coherence-gate from store-maintenance to program execution).
      * 'fixed'    : clean every k steps regardless.
      * 'none'     : never clean.
    The adaptive policy matches cleanups to the ACTUAL accumulated error, so under variable damage it holds the
    same fidelity at far fewer cleanups than any fixed cadence."""
    v = np.asarray(initial, float).copy()
    cleans = 0
    for t, step in enumerate(steps):
        v = step(v)
        if schedule == "adaptive":
            h, _ = health(v, codebook)
            if h < floor:
                v, _ = snap(v, codebook)
                cleans += 1
        elif schedule == "fixed":
            if (t % k) == (k - 1):
                v, _ = snap(v, codebook)
                cleans += 1
        elif schedule != "none":
            raise ValueError(f"unknown schedule {schedule!r}")
    return v, cleans


def _bursty_program(codebook, target, T=48, dim=1024, seed=0):
    """A test/demo program: T noisy rotations of a stored atom, with damage arriving in BURSTS (a few heavy
    steps every 10) -- the variable-damage regime where matching cleanups to damage pays."""
    rng = np.random.default_rng(seed)
    eps = [0.06 + (0.22 if (t % 10) < 3 else 0.0) for t in range(T)]

    def make_step(e):
        def step(v):
            n = rng.standard_normal(dim)
            n /= np.linalg.norm(n)
            w = v + e * n
            return w / np.linalg.norm(w)
        return step

    return [make_step(e) for e in eps]


def _selftest():
    dim, seed = 1024, 0
    rng = np.random.default_rng(seed)

    # (a) audit: verify the classification empirically on the substrate.
    a = random_vector(dim, rng)
    from holographic.agents_and_reasoning.holographic_ai import derived_atom
    ku = derived_atom(seed, "key", dim, unitary=True)          # a unitary key -> exact bind/unbind round-trip
    assert cosine(unbind(bind(a, ku), ku), a) > 0.999, "bind/unbind should be reversible for a unitary key"
    assert np.allclose(permute(permute(a, 5), -5), a), "permute should be reversible"
    assert np.allclose(involution(involution(a)), a), "involution should be self-inverse"
    b = random_vector(dim, rng)
    mix = bundle([a, b])
    assert cosine(mix, a) < 0.95 and cosine(mix, b) < 0.95, "bundle should be lossy (mixes its summands)"
    assert reversibility_class("bind")[0] == "reversible" and reversibility_class("cleanup")[0] == "lossy"

    # (b) scheduler: adaptive holds fidelity at FEWER cleanups than the fixed cadence that matches it.
    def measure(schedule, floor=0.9, k=3, seeds=40):
        cl, below = [], []
        for s in range(seeds):
            cb = [random_vector(dim, np.random.default_rng(1000 + s)) for _ in range(16)]
            tgt = int(np.random.default_rng(2000 + s).integers(16))
            steps = _bursty_program(cb, tgt, dim=dim, seed=s)
            v, c = auto_cleanup_run(cb[tgt], steps, cb, floor=floor, schedule=schedule, k=k)
            cl.append(c)
            below.append(cosine(v, cb[tgt]) < 0.9)
        return np.mean(cl), np.mean(below)

    ad_cl, ad_below = measure("adaptive", floor=0.9)
    fx_cl, fx_below = measure("fixed", k=3)
    assert ad_below < 0.1 and fx_below < 0.1, "both should hold final fidelity"
    assert ad_cl < 0.6 * fx_cl, f"adaptive ({ad_cl:.1f}) should use far fewer cleanups than fixed ({fx_cl:.1f})"
    print(f"holographic_reversible selftest: ok (audit verified; adaptive {ad_cl:.1f} vs fixed {fx_cl:.1f} "
          f"cleanups at matched fidelity)")


if __name__ == "__main__":
    _selftest()
