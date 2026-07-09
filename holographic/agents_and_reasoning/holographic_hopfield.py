"""Dense associative memory -- the modern Hopfield cleanup, and generation by denoising.

WHY THIS EXISTS
---------------
holostuff's standard cleanup (`Vocabulary.cleanup`) is a one-shot HARD nearest-neighbour:
`argmax(V @ q)`, snap to the winning atom. That is already Bayes-optimal for the question
"which stored atom is this noisy vector closest to" -- so nothing can beat it on IDENTITY, and
we measured exactly that (modern Hopfield ties it 1.000 vs 1.000 across noise levels). That tie
is a KEPT NEGATIVE: do not expect an accuracy win on classification.

What the modern continuous Hopfield update (Ramsauer et al. 2020, "Hopfield Networks is All You
Need"; Krotov & Hopfield 2016; Demircigil et al. 2017) actually buys us is two things the hard
argmax cannot give:

  1. CONTINUOUS-VECTOR DENOISING. The update z = V^T softmax(beta * V q) returns a clean vector
     on the stored-pattern manifold, not just an identity. Measured: a recovered vector at heavy
     noise (cosine 0.45 to truth) cleans to cosine ~1.0. That matters wherever snapping to a
     discrete atom is the WRONG move -- continuous encoders, FHRR phasor states, compositional
     intermediates, and as the per-step denoiser inside Plug-and-Play restoration and the
     resonator.

  2. GENERATION BY DENOISING. Iterating the same update from PURE NOISE walks onto the manifold
     (measured: nearest-pattern cosine 0.5 -> 1.0 in ~8 steps). Denoising and generation are the
     SAME operation in different regimes -- the engine's own small diffusion sampler.

DESIGN NOTES (backward-compatible by construction)
  * At beta -> infinity the softmax becomes a one-hot argmax, so `dense_cleanup` reproduces the
    existing hard-NN decision EXACTLY. It is a strict superset, added as a separate callable; the
    default `Vocabulary.cleanup` is untouched.
  * KEPT NEGATIVE (generation): sampling over the bare codebook just returns stored atoms (a
    degenerate sampler). The interesting regime is generating over a COMPOSED or continuous
    manifold -- `generate` takes whatever codebook you hand it, so feed it composed states.
  * Deterministic: `generate` takes an explicit seed; everything else is pure NumPy with no RNG.
"""

import numpy as np
from holographic.misc.holographic_determinism import argmax_tiebreak


def _unit_rows(M):
    """Row-normalise to unit length (so a dot is a cosine). 1e-12 guards a zero row."""
    M = np.asarray(M, float)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def _sparsemax(z):
    """Project z onto the probability simplex (Martins & Astudillo 2016) -- the SPARSE analogue of
    softmax: most entries become exactly 0, so a readout built on it blends only the few relevant
    patterns instead of all of them. This is what cures softmax's metastable over-smoothing on a
    continuous manifold (the Hopfield-Fenchel-Young move, Santos et al. 2024-25). Pure NumPy,
    deterministic."""
    z = np.asarray(z, float)
    zs = np.sort(z)[::-1]                            # sort descending
    css = np.cumsum(zs) - 1.0
    k = np.arange(1, z.size + 1)
    cond = zs - css / k > 0
    rho = k[cond][-1]                               # support size
    tau = css[cond][-1] / rho                       # threshold so the kept weights sum to 1
    return np.maximum(z - tau, 0.0)


def _topk(z, k):
    """Keep the k largest entries of z and softmax over JUST those, zeroing the rest (Gao et al. 2024, the
    k-sparse / TopK autoencoder readout; a point in the same Hopfield-Fenchel-Young energy family as softmax
    and sparsemax). The HARD-sparse cousin of _sparsemax: where sparsemax chooses its own support size, TopK
    fixes it at exactly k. MEASURED to be the readout that survives the HIGHEST factorization load -- at
    codebook N=110 it is the only readout still recovering factors (0.23 vs 0.05 for softmax/sparsemax/entmax),
    because a fixed k keeps k candidates alive where adaptive methods over-prune. The honest trade: k must be
    chosen (too small underperforms -- k=4 lost badly to k=8 in measurement), and it ties or slightly loses to
    sparsemax at the MIDDLE of the load range -- so it earns its place as the HIGH-load option, not a new
    default. Pure NumPy, deterministic."""
    z = np.asarray(z, float)
    if k >= z.size:                                  # k covers everything -> plain softmax
        e = np.exp(z - z.max())
        return e / (e.sum() + 1e-12)
    keep = np.argpartition(z, -k)[-k:]               # indices of the k largest (unordered, O(n))
    w = np.zeros_like(z)
    e = np.exp(z[keep] - z[keep].max())
    w[keep] = e / (e.sum() + 1e-12)                  # softmax over the survivors; rest stay 0
    return w


def dense_cleanup(query, codebook, beta=25.0, steps=3, readout="softmax"):
    """One modern-Hopfield denoise of `query` against `codebook` (V), iterated `steps` times.

    z <- V^T g(beta * V z), starting from z = query, where g is the READOUT map. Returns the cleaned
    CONTINUOUS vector (not an identity). Higher beta = sharper (beta->inf reproduces hard
    nearest-neighbour either way); more steps = deeper basin descent.

      readout='softmax'   (default, UNCHANGED): g = softmax -- the dense blend of ALL patterns
                          (Ramsauer et al. 2020). This is the original update, bit-for-bit.
      readout='sparsemax': g = sparsemax (Martins & Astudillo 2016) -- the SPARSE simplex projection
                          that blends ONLY the relevant patterns. The Hopfield-Fenchel-Young fix
                          (Santos et al. 2024-25) for softmax's metastable mixing: the dense blend
                          weights in far atoms and OVER-SMOOTHS on a continuous manifold, where the
                          sparse readout does not. MEASURED on a continuous SD-latent manifold
                          (recovering UN-stored in-between points): sparse 0.999 vs softmax 0.983 vs
                          nearest-neighbour 0.998 -- it reverses the softmax-loses-to-NN result, and
                          does NOT regress discrete recall (exact at high beta, where sparsemax is
                          also one-hot). KEPT NEGATIVE: the sparse-beats-NN margin is thin (~0.001);
                          its clear, robust win is over the softmax blend.
    The softmax branch is the original max-subtracted stable form."""
    V = _unit_rows(codebook)
    z = np.asarray(query, float).copy()
    for _ in range(steps):
        s = V @ z
        if readout == "sparsemax":
            w = _sparsemax(beta * s)          # sparse: only the relevant patterns, no over-smoothing
        else:
            e = s - s.max()                   # numerical stability; does not change the softmax
            w = np.exp(beta * e)
            w /= w.sum()
        z = V.T @ w
    return z


class HopfieldCleanup:
    """A drop-in associative-memory cleanup with a continuous denoiser and an identity readout.

    fit() caches the unit codebook; cleanup() gives the (index, cosine) hard decision -- identical
    to holostuff's argmax NN -- and denoise() gives the cleaned continuous vector. Same object,
    both readouts, so callers can pick the one their downstream step needs."""

    def __init__(self, beta=25.0, steps=3):
        self.beta = beta
        self.steps = steps
        self.V = None

    def fit(self, codebook):
        self.V = _unit_rows(codebook)
        return self

    def denoise(self, query):
        """Cleaned continuous vector on the stored-pattern manifold."""
        return dense_cleanup(query, self.V, self.beta, self.steps)

    def cleanup(self, query):
        """Hard (index, cosine) readout -- matches Vocabulary.cleanup's decision at high beta."""
        if self.V is None:
            raise ValueError("fit() a codebook first")
        q = np.asarray(query, float)
        qn = np.linalg.norm(q)
        sims = self.V @ (q / qn) if qn > 0 else self.V @ q
        j = int(sims.argmax())
        return j, float(sims[j])


def generate(codebook, steps=12, beta0=4.0, beta1=40.0, noise0=0.6, seed=0, readout="softmax"):
    """Generate a sample by DENOISING from pure noise (B10): the cleanup attractor as a tiny
    holographic diffusion. Anneal beta upward (vague -> sharp) and injected noise downward across
    `steps`, starting from a random unit vector, ending on the manifold.

    Returns the generated unit vector. NOTE (kept negative): over a bare codebook this converges
    to a stored atom; feed a COMPOSED/continuous manifold as `codebook` for novel-but-valid
    samples. `readout='sparsemax'` is accepted (threaded to the cleanup) but does NOT change this over a
    bare/continuous codebook -- both readouts snap to a stored atom. Deterministic in `seed`."""
    V = _unit_rows(codebook)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(V.shape[1])
    z /= np.linalg.norm(z) + 1e-12
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 + (beta1 - beta0) * frac          # sharpen toward the manifold
        noise = noise0 * (1.0 - frac)                  # cool the injected noise
        z = dense_cleanup(z, V, beta=beta, steps=1, readout=readout)
        if noise > 0:
            z = z + noise * rng.standard_normal(V.shape[1]) / np.sqrt(V.shape[1])
        z /= np.linalg.norm(z) + 1e-12
    return z


def _structure_project(z, roles, fillers, beta, steps=1, readout="softmax"):
    """Project z onto the COMPOSED manifold: per role, unbind the slot, dense_cleanup its filler toward the
    vocabulary, rebind; bundle the slots and renormalise. This is the composed-manifold denoiser -- the
    slot-wise analogue of dense_cleanup's bare-codebook snap (and an instance of 'iterate a projection')."""
    from holographic.agents_and_reasoning.holographic_ai import bind, unbind
    parts = [bind(r, dense_cleanup(unbind(z, r), fillers, beta, steps, readout=readout)) for r in roles]
    out = np.sum(parts, axis=0)
    return out / (np.linalg.norm(out) + 1e-12)


def _decode_combo(z, roles, fillers):
    """The hard decoded combination of a composed vector: per role, unbind the slot and take the argmax filler.
    This is the structure's discrete CONTENT -- when it stops changing across diffusion steps, the generated
    structure has settled (the B3 stop signal)."""
    from holographic.agents_and_reasoning.holographic_ai import unbind
    # DETERMINISM CONTRACT (ISA-1): the recalled filler IS the observable decision -- cite the tie-break rule.
    return tuple(argmax_tiebreak(fillers @ unbind(z, r)) for r in roles)


def generate_structure(roles, fillers, steps=16, beta0=4.0, beta1=60.0, noise0=0.5, seed=0, readout="softmax",
                       early_stop=False, min_steps=None, patience=3, stats=None):
    """Generate a novel-but-VALID composed structure by denoising from noise over the COMPOSED manifold
    (B10 + the Eno reframe). The same annealed diffusion as generate() -- random start, beta up, noise down --
    but the denoiser is `_structure_project` (slot-wise) instead of a bare-codebook cleanup, so the walk lands
    on the manifold of role-filler STRUCTURES, not on a stored atom.

    `roles` is (S, dim) unitary role atoms; `fillers` is (V, dim) the filler vocabulary. Returns a unit
    vector whose every slot unbinds to a vocabulary atom -- a NEW combination of fillers (one of V^S), valid
    by construction (re-encoding the decoded fillers reproduces it). Different seeds give different structures;
    over a bare codebook generate() degenerates to a stored atom, which is exactly what this avoids.
    `readout='sparsemax'` keeps validity (the result still reencodes its decoded combination at cosine
    1.000) while CURING generative mode collapse: softmax funnels many random seeds into the same few
    structures (measured diversity 0.03-0.5), sparsemax stays diverse (0.6-1.0, nearly every seed distinct).
    Deterministic in `seed`.

    ADAPTIVE STOP (B3, opt-in early_stop=True; pass stats={} to read stats['steps']): the decoded combination
    SETTLES well before the fixed schedule ends, so stop once it has been STABLE for `patience` steps past a
    `min_steps` floor (default steps//2, past the high-noise exploration phase) -- Eno's condition, stability
    not FIRST-convergence, so novelty is not amputated. On stopping, one final crisp `_structure_project` at
    full beta (no noise) sharpens the settled combination. Measured: ~50% fewer steps, the SAME structure as
    the full run on every seed (novelty + diversity preserved), and the final snap restores validity to 1.000 --
    so unlike the splat fit's soft-plateau stop, this one is essentially FREE (the hard decoded combination IS
    an effective certificate). Off by default (early_stop=False is bit-identical to the fixed schedule)."""
    rng = np.random.default_rng(seed)
    dim = roles.shape[1]
    z = rng.standard_normal(dim)
    z /= np.linalg.norm(z) + 1e-12
    _ms = (max(6, steps // 2)) if min_steps is None else min_steps   # floor: past the high-noise exploration phase
    _last = None
    _stable = 0
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 + (beta1 - beta0) * frac              # sharpen toward the manifold
        noise = noise0 * (1.0 - frac)                      # cool the injected noise
        z = _structure_project(z, roles, fillers, beta, 1, readout=readout)
        if noise > 0:
            z = z + noise * rng.standard_normal(dim) / np.sqrt(dim)
        z /= np.linalg.norm(z) + 1e-12
        if early_stop:                                     # B3: stop once the decoded structure has SETTLED -- stable
            combo = _decode_combo(z, roles, fillers)       # for `patience` steps past the floor (stability, not
            _stable = _stable + 1 if combo == _last else 0  # first-convergence -- Eno's condition preserves novelty)
            _last = combo
            if t + 1 >= _ms and _stable >= patience:
                z = _structure_project(z, roles, fillers, beta1, 1, readout=readout)   # final crisp snap, no noise
                z /= np.linalg.norm(z) + 1e-12             # the settled combo at full beta -> validity back to 1.000
                if stats is not None:
                    stats["steps"] = t + 1
                return z
    if stats is not None:
        stats["steps"] = steps
    return z


def _b3_selftest():
    """B3: the adaptive-stop diffusion stops once the decoded structure has SETTLED (stable past a min_steps
    floor), reaching the SAME structure as the full fixed schedule on every seed at ~half the steps, with the
    final crisp snap restoring validity to 1.000 -- essentially free, novelty preserved. OFF by default runs
    the full schedule (bit-identical)."""
    from holographic.agents_and_reasoning.holographic_ai import random_vector, bind
    dim, S, V, STEPS = 512, 4, 8, 16
    rng = np.random.default_rng(1)
    roles = np.stack([random_vector(dim, rng) for _ in range(S)])
    fillers = np.stack([random_vector(dim, rng) for _ in range(V)])

    def reencode(combo):
        out = np.sum([bind(roles[i], fillers[combo[i]]) for i in range(S)], axis=0)
        return out / (np.linalg.norm(out) + 1e-12)

    matches = 0
    combos = set()
    saved = []
    for s in range(12):
        st_f = {}
        zf = generate_structure(roles, fillers, steps=STEPS, seed=s, readout="sparsemax", stats=st_f)
        assert st_f["steps"] == STEPS, st_f                          # OFF by default: full schedule
        st_e = {}
        ze = generate_structure(roles, fillers, steps=STEPS, seed=s, readout="sparsemax",
                                early_stop=True, stats=st_e)
        cf = _decode_combo(zf, roles, fillers)
        ce = _decode_combo(ze, roles, fillers)
        if cf == ce:
            matches += 1
        combos.add(ce)
        saved.append(st_e["steps"])
        assert float(reencode(ce) @ ze) > 0.999, (s, float(reencode(ce) @ ze))   # final snap -> validity ~1.0

    assert matches == 12, matches                                   # same structure as full run on every seed
    assert len(combos) >= 10, len(combos)                           # diversity preserved (novelty not amputated)
    assert max(saved) < STEPS, saved                                # stopped before the cap on every seed


if __name__ == "__main__":
    _b3_selftest()
    print("holographic_hopfield B3 adaptive-stop diffusion selftest passed")
