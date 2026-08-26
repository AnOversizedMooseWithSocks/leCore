"""Superposed key-value memory with a CLOSED-FORM capacity law, a single-shot allocator,
and a load-gated iterative decoder (holographic_superposed, SUP-1).

WHY THIS EXISTS
---------------
The engine could already encode pairs (encode_pairs) and measure bundle load empirically
(bundle_capacity). What it could not do is PREDICT capacity from (D, V, alpha) in closed
form, ALLOCATE a dimension for a demanded load before storing anything, or decode past
the matched-filter wall. This module adds those three, on an identification measured on
this tree (see NOTES "The channel's true name"): a superposition memory IS random-
spreading CDMA / sparse support recovery, so its theory is inherited, not invented.

THE LAW (matched filter / one-shot cleanup):
    n_star(D, V, alpha) ~ C_MF * D / (2 * x^2),   x = Qinv((1 - alpha) / V)
Measured on this tree: one-shot n* = 25/48/88 at D = 512/1024/2048 (V=1024, alpha=0.90),
and the V-scaling constant n*.x^2/D held at 0.6 +/- 0.08 across V = 64..4096. C_MF below
is that measured constant, deliberately set at the LOW edge so the allocator over-provides
rather than over-promises.

THE DECODER LADDER, with its kept negative loud:
  * one-shot cleanup   -- the matched filter. Degrades SMOOTHLY with load.
  * pic (resonator-style parallel interference cancellation, damped) -- exact recall to
    ~1.5x the matched-filter wall, then a SHARP phase transition (Donoho-Tanner line,
    ~rho = 1/(2 ln(V^2/D))). KEPT NEGATIVE, measured: past its transition undamped PIC is
    WORSE than one-shot (0.333 vs 0.591 at n=84, D=1024) -- error amplification, textbook
    CDMA. Therefore recall(decoder="pic") is GATED: above the predicted transition it
    falls back to one-shot and says so in the result, rather than running poison.

QUANTIZATION CONTRACT (measured): int8 on the MEMORY vector is decision-free (identical
n* to float64 at every D tested); sign-binary retains ~70% of capacity (0.031D vs 0.044D)
while cutting state bits 64x -- the engine's standing "binary distorts geometry" negative
is hereby REFINED, not contradicted: it applies to fine readback, and argmax recall
survives sign. Utilization vs the Fano-on-state-bits bound: ~25% at 1 bit (the published
trained-architecture reference sits at ~0.04%).

Stdlib + numpy only. Deterministic given seeds. All atoms are seed-derived, so the whole
codebook is regenerable from 64 bits -- lever 3, doing real work in the bit accounting.
"""
import math

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind

_RFFT, _IRFFT = np.fft.rfft, np.fft.irfft

# Matched-filter constant, measured on this tree (V-sweep 64..4096 gave 0.51..0.66;
# the LOW edge is used so allocate() errs toward extra dimension, never toward misses).
C_MF = 0.50
# PIC transition constant: Donoho-Tanner small-delta asymptote rho ~ 1/(2 ln(1/delta)),
# delta = D / V^2. Measured transition ~70 at D=1024,V=1024 vs 74 predicted; the safety
# factor keeps the gate strictly below the measured cliff (collapse observed by n=84).
PIC_SAFETY = 0.90


def _qinv(p):
    """Inverse Gaussian tail Q^-1(p) by bisection on erfc -- stdlib only, no scipy.
    WHY bisection: 80 halvings on [0, 10] is exact to ~1e-22, branch-free, and keeps the
    module dependency-clean; this is called a handful of times, never in a hot loop."""
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(mid / math.sqrt(2.0)) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def capacity_law(dim, vocab, alpha=0.90):
    """Predicted one-shot (matched-filter) capacity n* for a superposed pair memory.

    Returns the largest number of bind(key, value) pairs recallable at per-query
    accuracy >= alpha by single cleanup against a vocab-sized codebook. Closed form
    n* = C_MF * D / (2 x^2) with x = Qinv((1-alpha)/vocab); C_MF is measured, and the
    V-dependence (the part the old empirical slope lacked) is the x^2 term."""
    x = _qinv(max(1e-300, (1.0 - alpha)) / max(2, vocab))
    return max(1, int(C_MF * dim / (2.0 * x * x)))


def pic_transition(dim, vocab):
    """Predicted load at which the iterative (PIC/resonator) decoder's exact basin ENDS.

    The Donoho-Tanner sparse-recovery line at delta = D / V^2, times a safety factor
    measured against the observed collapse. Above this, run_pic amplifies its own errors
    and lands BELOW the matched filter -- the load gate in recall() enforces that this
    number is respected rather than remembered."""
    delta = dim / float(vocab) ** 2
    rho = 1.0 / (2.0 * math.log(1.0 / max(delta, 1e-12)))
    return max(1, int(PIC_SAFETY * rho * dim))


def allocate(n_pairs, vocab, alpha=0.90, decoder="one-shot", margin=1.10, round_to=64):
    """Dimension needed to hold n_pairs at accuracy alpha -- the law inverted.

    This is the 'price the demand, then spend' step: call it BEFORE storing, with the
    demand from a state-demand meter or from the task. `decoder='pic'` allocates against
    the iterative transition instead of the matched-filter wall (smaller D for the same
    load, at the cost of requiring the gated decoder at recall time)."""
    if decoder == "pic":
        # invert pic_transition: n = SAFETY * D / (2 ln(V^2/D)); solve by fixed point,
        # WHY: D appears inside the log, but the log varies slowly, so 8 rounds settle it.
        D = 2.0 * n_pairs * math.log(float(vocab))
        for _ in range(8):
            D = margin * n_pairs * 2.0 * math.log(float(vocab) ** 2 / max(D, 1.0)) / PIC_SAFETY
        need = D
    else:
        x = _qinv((1.0 - alpha) / max(2, vocab))
        need = margin * n_pairs * 2.0 * x * x / C_MF
    return int(math.ceil(need / round_to) * round_to)


class SuperposedMemory:
    """A key-value store that is ONE vector: memory = sum_i bind(key_i, value_i).

    Keys and values are symbol ids into seed-derived codebooks (the codebook is
    regenerable from the seed -- it costs 64 bits of state, not vocab*D floats).
    `precision` quantizes the MEMORY (the thing q(d) counts): 'f64', 'int8' (measured
    decision-free), or 'bin' (sign; ~70% capacity at 1/64 the bits)."""

    def __init__(self, dim, vocab, seed=0, precision="f64", codebook="dense"):
        self.dim, self.vocab, self.precision = int(dim), int(vocab), precision
        self.seed_ = int(seed)
        # F2 (the sweep's 4-GiB finding): dense (vocab x dim) f64 codebooks cost 4 GiB EACH at
        # 64k x 8192, when every row is a pure function of (seed, index) -- the engine's own first
        # principle violated at its core primitive. Three modes behind one seam, dense the default
        # and BIT-IDENTICAL to before (same rng, same arrays):
        #   codebook='dense'    -- the original arrays; zero behavior change.
        #   codebook='hadamard' -- rows are sign-permuted Hadamard rows, GENERATED not stored
        #                          (O(dim) state; crosstalk exactly zero; correlate is a matvec --
        #                          the install-preferred mode: an installed model can carry the SAME
        #                          dictionary). Requires vocab <= 2*dim. VERIFIED premise: hadamard
        #                          atoms as keys AND values recall 38/40 at n=40, D=1024.
        #   codebook='lazy'     -- per-row seeded gaussian rows (default_rng((seed, i)), 67 us/row
        #                          at dim 4096, ANY index O(1)); vocab unbounded, O(1) construction
        #                          memory. KEPT NEGATIVE (measured): PCG64.advance() fixed-stride
        #                          skipping does NOT reproduce the dense rows (ziggurat consumes a
        #                          variable number of raw draws), so a bit-compatible lazy view of
        #                          the DENSE codebook is impossible -- this mode is a DIFFERENT
        #                          codebook, default-off, equal-QUALITY verified at the law, and
        #                          RUNTIME-ONLY by nature (row generation is control flow).
        self.codebook = str(codebook)
        if self.codebook == "dense":
            rng_k = np.random.default_rng(seed * 2 + 1)
            rng_v = np.random.default_rng(seed * 2 + 2)
            self.K = rng_k.standard_normal((vocab, dim)) / np.sqrt(dim)
            self.K /= np.linalg.norm(self.K, axis=1, keepdims=True)
            self.V = rng_v.standard_normal((vocab, dim)) / np.sqrt(dim)
            self.V /= np.linalg.norm(self.V, axis=1, keepdims=True)
        elif self.codebook == "hadamard":
            if vocab > 2 * dim:
                raise ValueError("hadamard codebook holds at most 2*dim atoms (%d > %d)" % (vocab, 2 * dim))
            from holographic.caching_and_storage.holographic_htcodebook import HadamardCodebook
            self._hk = HadamardCodebook(dim, seed=seed * 2 + 1)
            self._hv = HadamardCodebook(dim, seed=seed * 2 + 2)
            self.K = self.V = None
        elif self.codebook == "lazy":
            self.K = self.V = None
        else:
            raise ValueError("codebook must be 'dense', 'hadamard' or 'lazy'")
        self.mem = np.zeros(dim)
        self.n_stored = 0

    def _rows(self, which, idx):
        """Row gather behind the codebook seam: dense indexing, hadamard atom generation, or
        per-row-seeded lazy rows -- all unit-normalized, all pure functions of (seed, index)."""
        idx = np.atleast_1d(np.asarray(idx, dtype=int))
        if self.codebook == "dense":
            return (self.K if which == "K" else self.V)[idx]
        if self.codebook == "hadamard":
            cb = self._hk if which == "K" else self._hv
            out = np.stack([cb.atom(int(i)) for i in idx]).astype(float)
            return out / np.sqrt(self.dim)                    # hadamard rows are +-1; unit-normalize
        base = (self.seed_ * 2 + (1 if which == "K" else 2), )
        out = np.stack([np.random.default_rng(base + (int(i),)).standard_normal(self.dim) for i in idx])
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)

    def _correlate_V(self, est):
        """est @ V.T behind the seam. Dense: one matmul (bit-identical to before). Hadamard:
        correlate against generated atoms (a matvec -- the installed form, per the projector's
        verdict). Lazy: TILED generate-and-score, so vocab never materializes (the F18 shape)."""
        if self.codebook == "dense":
            return est @ self.V.T
        if self.codebook == "hadamard":
            A = self._rows("V", np.arange(self.vocab))
            return est @ A.T
        out = np.empty((est.shape[0], self.vocab))
        tile = 4096
        for s in range(0, self.vocab, tile):
            out[:, s:s + tile] = est @ self._rows("V", np.arange(s, min(s + tile, self.vocab))).T
        return out

    def store(self, keys, values):
        """Superpose bind(key_i, value_i) for parallel id arrays -- one batched FFT."""
        keys = np.asarray(keys, dtype=int)
        values = np.asarray(values, dtype=int)
        kf = _RFFT(self._rows("K", keys), axis=1)
        vf = _RFFT(self._rows("V", values), axis=1)
        self.mem = self.mem + _IRFFT((kf * vf).sum(0), n=self.dim)
        self.n_stored += len(keys)
        return self

    def _state(self):
        """The memory AS COUNTED BY q(d): quantized per the declared precision."""
        if self.precision == "int8":
            s = np.max(np.abs(self.mem)) / 127.0
            return np.round(self.mem / (s + 1e-300)) * s
        if self.precision == "bin":
            return np.sign(self.mem)
        return self.mem

    def state_bits(self):
        """Honest state size in bits under the declared precision."""
        return self.dim * {"f64": 64, "int8": 8, "bin": 1}[self.precision]

    def save(self, path):
        """Export the trained memory. The file holds the STATE and the RECIPE, nothing
        else: the codebooks are seed-derived, so vocab*dim floats of dictionary cost
        five scalars on disk (lever 3 -- determinism instead of storage). The state is
        written AT THE DECLARED PRECISION: int8 as bytes+scale, 'bin' bit-packed
        (dim/8 bytes), so the exported artifact is exactly what q(d) counts."""
        if self.precision == "int8":
            scale = float(np.max(np.abs(self.mem)) / 127.0) or 1.0
            state = np.round(self.mem / scale).astype(np.int8)
        elif self.precision == "bin":
            scale = 1.0
            state = np.packbits((self.mem > 0).astype(np.uint8))
        else:
            scale = 1.0
            state = self.mem
        np.savez_compressed(path, dim=self.dim, vocab=self.vocab, seed=self.seed_,
                            precision=self.precision, n_stored=self.n_stored,
                            scale=scale, state=state)
        return path

    @classmethod
    def load(cls, path):
        """Import a saved memory: regenerate codebooks from the seed, restore the state."""
        z = np.load(path if str(path).endswith(".npz") else str(path) + ".npz",
                    allow_pickle=False)
        out = cls(int(z["dim"]), int(z["vocab"]), seed=int(z["seed"]),
                  precision=str(z["precision"]))
        out.n_stored = int(z["n_stored"])
        if out.precision == "int8":
            out.mem = z["state"].astype(float) * float(z["scale"])
        elif out.precision == "bin":
            out.mem = np.unpackbits(z["state"])[:out.dim].astype(float) * 2.0 - 1.0
        else:
            out.mem = z["state"].astype(float)
        return out

    def recall(self, keys, decoder="one-shot", sweeps=4, damping=0.5,
               state_bits=None):
        """Recall value ids for the queried key ids. Returns dict{values, decoder, why}.

        decoder='one-shot' is the matched filter (smooth degradation). decoder='pic' is
        damped parallel interference cancellation -- the resonator's coordinate update
        specialized to known keys -- and it is LOAD-GATED: if n_stored exceeds
        pic_transition(dim, vocab), it refuses the iterative path and answers one-shot,
        with the reason in `why`, because past the transition PIC is measurably worse
        than the decoder it upgrades (the kept negative, asserted in _selftest)."""
        keys = np.asarray(keys, dtype=int)
        m = self._state()
        if state_bits == 1:
            # 1-BIT STATE + PIC: the utilization headline, measured (D=1024, V=256,
            # alpha=0.9, 3 seeds): sign-quantised state decoded one-shot holds n=32
            # (25.0% of the 1-bit Fano ceiling -- the standing champion, replicated
            # exactly); the SAME state decoded with damped PIC holds n=48 (37.5%,
            # against the ~39% prediction on record). Iterative cancellation earns
            # 1.5x more information per stored bit. Float rows are NOT quoted as
            # utilization -- 64 bits/dim is a different ceiling (claim-type rule).
            m = np.sign(m) + (m == 0)
        kf = _RFFT(self._rows("K", keys), axis=1)
        est = _IRFFT(np.conj(kf) * _RFFT(m)[None, :], n=self.dim, axis=1)
        vhat = np.argmax(self._correlate_V(est), axis=1)
        if decoder != "pic":
            return {"values": vhat, "decoder": "one-shot", "why": "matched filter"}
        limit = pic_transition(self.dim, self.vocab)
        if self.n_stored > limit:
            return {"values": vhat, "decoder": "one-shot",
                    "why": "GATED: load %d > PIC transition %d; iterative decoding "
                           "amplifies errors past its phase transition (kept negative), "
                           "answered with the matched filter instead" % (self.n_stored, limit)}
        # Damped PIC: re-estimate each value against the residual with its own current
        # term added back. WHY damping: undamped Jacobi updates overshoot near the
        # transition; averaging the residual halves the spectral radius of the error map.
        for _ in range(int(sweeps)):
            B = _IRFFT(kf * _RFFT(self._rows("V", vhat), axis=1), n=self.dim, axis=1)
            resid = m - B.sum(0)
            look = _IRFFT(np.conj(kf) * _RFFT(resid[None, :] + B, axis=1), n=self.dim, axis=1)
            mixed = damping * look + (1.0 - damping) * est
            est = mixed
            vhat = np.argmax(self._correlate_V(mixed), axis=1)
        return {"values": vhat, "decoder": "pic",
                "why": "damped PIC, load %d <= transition %d" % (self.n_stored, limit)}


class BigPairMemory:
    """SuperposedMemory for vocabularies where materialised codebooks do not fit:
    atoms are REGENERATED from seeds in chunks at use time (the MQAR-benchmark
    pattern, proven at V=8192/D=20096/recall 1.000). Memory cost: the ONE state
    vector; the vocab x D codebooks cost nothing between calls. WHY this exists:
    the 144k-entry dictionary demands D ~ 4e6, whose materialised codebooks are
    hundreds of GB -- but the int8 STATE is ~34 MB, and storage/batch-recall only
    ever touch atoms chunkwise. Interactive full-vocab cleanup at that scale is
    still honest minutes, not ms -- serve curated working sets interactively and
    the long tail in batch."""

    def __init__(self, dim, vocab, seed=0, chunk=512):
        self.dim, self.vocab, self.seed_, self.chunk = int(dim), int(vocab), int(seed), int(chunk)
        self.mem = np.zeros(dim)
        self.n_stored = 0

    def _atoms(self, ids, tag):
        out = np.empty((len(ids), self.dim))
        for i, sid in enumerate(ids):
            r = np.random.default_rng(((self.seed_ * 2 + tag) << 32) + int(sid))
            v = r.standard_normal(self.dim) / np.sqrt(self.dim)
            out[i] = v / np.linalg.norm(v)
        return out

    def store(self, keys, values):
        keys = np.asarray(keys, dtype=int); values = np.asarray(values, dtype=int)
        for lo in range(0, len(keys), self.chunk):
            kf = _RFFT(self._atoms(keys[lo:lo + self.chunk], 1), axis=1)
            vf = _RFFT(self._atoms(values[lo:lo + self.chunk], 2), axis=1)
            self.mem = self.mem + _IRFFT((kf * vf).sum(0), n=self.dim)
        self.n_stored += len(keys)
        return self

    def recall(self, keys):
        """One-shot cleanup with chunk-regenerated value codebook. Batch-oriented."""
        keys = np.asarray(keys, dtype=int)
        kf = _RFFT(self._atoms(keys, 1), axis=1)
        est = _IRFFT(np.conj(kf) * _RFFT(self.mem)[None, :], n=self.dim, axis=1)
        best = np.full(len(keys), -1); bs = np.full(len(keys), -np.inf)
        for lo in range(0, self.vocab, self.chunk):
            C = self._atoms(np.arange(lo, min(lo + self.chunk, self.vocab)), 2)
            sc = est @ C.T
            j = sc.argmax(1); v = sc.max(1)
            upd = v > bs; best[upd] = lo + j[upd]; bs[upd] = v[upd]
        return {"values": best, "decoder": "one-shot(streamed)",
                "why": "codebooks regenerated in chunks of %d" % self.chunk}


def advise_scale(n_pairs=None, vocab=None, dim=None, bundle_k=None, depth=None,
                 factors=None, alpha=0.90, decoder="one-shot", fix=False):
    """The walls, consulted BEFORE they are hit. Every measured capacity/depth law in
    one checkpoint: pass what you know about the task, get every law's margin, the
    BINDING constraint, and a concrete prescription -- grow to the exact dim the law
    demands, switch decoder at the PIC transition, or reach for the named lever
    (partition / carriers / tiling) where growth is the wrong move. fix=True returns
    the corrected spec alongside. WHY: the laws lived in NOTES and in scattered
    modules, so each wall was met head-first and the remedy re-derived (the 'auto
    scaling is not very auto' complaint, verbatim). Closed forms answer instantly;
    for empirical knobs the prescription NAMES mind.auto_scale so the measurement
    loop is one call away instead of a rediscovery.

    Laws applied: capacity_law / allocate (pair memory, alpha-exact); pic_transition
    (decoder crossover); bundle readout k* ~ 0.13*D (linear-readout ceiling -- sparse
    decoders hold ~8.7x more, the folklore '20-32 items' is a readout artifact);
    factorization hard wall F=4 (resonator: split factor groups beyond it)."""
    laws, spec = [], {"n_pairs": n_pairs, "vocab": vocab, "dim": dim,
                     "bundle_k": bundle_k, "depth": depth, "factors": factors}
    if n_pairs is not None and vocab is not None:
        need = allocate(n_pairs, vocab, alpha=alpha, decoder=decoder)
        ok = dim is None or dim >= need
        laws.append({"law": "pair-capacity (allocate)", "ok": ok,
                     "margin": None if dim is None else dim / need,
                     "prescription": "dim >= %d for %d pairs over vocab %d at alpha %.2f"
                                     % (need, n_pairs, vocab, alpha)})
        if fix and not ok:
            spec["dim"] = need
        if dim is not None:
            pt = pic_transition(dim, vocab)
            laws.append({"law": "PIC transition", "ok": True,
                         "margin": pt / max(1, n_pairs),
                         "prescription": ("decoder='pic' pays below load %d; at load %d "
                                          "use %s") % (pt, n_pairs,
                                          "pic" if n_pairs <= pt else "one-shot + more dim")})
    if bundle_k is not None and dim is not None:
        kstar = 0.13 * dim
        ok = bundle_k <= kstar
        laws.append({"law": "bundle readout k* ~ 0.13*D", "ok": ok,
                     "margin": kstar / bundle_k,
                     "prescription": "fits (k*=%d)" % int(kstar) if ok else
                     ("grow dim >= %d, or PARTITION into %d bundles (distribute "
                      "lever), or a sparse decoder (~8.7x the linear ceiling)")
                     % (int(np.ceil(bundle_k / 0.13 / 64) * 64),
                        int(np.ceil(bundle_k / kstar)))})
        if fix and not ok:
            spec["dim"] = max(spec["dim"] or 0, int(np.ceil(bundle_k / 0.13 / 64) * 64))
    if factors is not None:
        ok = factors <= 4
        laws.append({"law": "factorization hard wall F=4", "ok": ok,
                     "margin": 4.0 / factors,
                     "prescription": "resonator holds" if ok else
                     "F=%d exceeds the measured wall: split into ceil(F/4)=%d factor "
                     "groups and resolve hierarchically (tiling lever)"
                     % (factors, int(np.ceil(factors / 4)))})
    if depth is not None:
        # MEASURED (see depth_probe): deepest-leaf separability collapses by d5-7 and
        # is DIM-INDEPENDENT (256 vs 1024 identical) -- growing dim cannot recover a
        # level that geometric attenuation + normalisation already crushed.
        ok = depth <= 4
        laws.append({"law": "nesting depth (measured: dim-independent collapse ~d5-7)",
                     "ok": ok, "margin": 4.0 / depth,
                     "prescription": "depth %d carries" % depth if ok else
                     "depth %d exceeds the measured separability wall: dim is NOT the "
                     "lever -- elevate levels onto carriers/INDEX or anchor "
                     "coarse/fine per level; verify your structure with "
                     "mind.depth_probe(depth, dim)" % depth})
    binding = None
    viol = [l for l in laws if l["ok"] is False]
    if viol:
        binding = min(viol, key=lambda l: (l["margin"] if l["margin"] is not None else 0.0))
    out = {"laws": laws, "ok": not viol,
           "binding": None if binding is None else binding["law"],
           "prescription": (binding or (laws[-1] if laws else
                            {"prescription": "give me n_pairs/vocab/dim/bundle_k/"
                             "depth/factors and the laws answer"}))["prescription"]}
    if fix:
        out["fixed_spec"] = {k: v for k, v in spec.items() if v is not None}
    return out


def depth_probe(depth, dim, n=8, seed=0):
    """MEASURE the nesting-depth wall for typed trees: encode n trees differing ONLY
    at the deepest leaf and return the worst-case cosine between them -- at 1.0 the
    deepest level has vanished from the encoding. THE FINDING THIS SHIPPED WITH
    (measured before wiring, and it killed a planned faculty): the collapse is
    DIM-INDEPENDENT -- d1 0.70 / d3 0.97 / d5 0.997 / d7 1.000 at BOTH dim 256 and
    1024. Depth is not a dimension wall; each level attenuates the leaf's share
    geometrically and normalisation crushes what remains, so an auto_scale loop
    doubling dim would burn its budget to diagnose what this one number says. The
    lever is STRUCTURAL: elevate levels onto carriers/INDEX or anchor coarse/fine
    per level -- growth is the wrong move, and advise_scale now says so."""
    import lecore
    m = lecore.UnifiedMind(dim=int(dim), seed=seed)
    encs = []
    for i in range(int(n)):
        t = "leaf%d" % i
        for d in range(int(depth)):
            t = ("op%d" % d, t, "pad%d" % d)
        v = np.asarray(m.tree_structure(t).build()[-1], dtype=float)
        encs.append(v / np.linalg.norm(v))
    E = np.array(encs)
    S = E @ E.T
    np.fill_diagonal(S, -1.0)
    return {"worst_cosine": float(S.max()), "depth": int(depth), "dim": int(dim),
            "separable": bool(S.max() < 0.95),
            "why": "deepest-leaf distinguishability; 1.0 = that level is gone"}


def encode_tree_carrier(tree, dim, seed=0):
    """Carrier-elevated tree encoding -- the depth wall's prescribed lever, built.
    The flat recursive encoder attenuates each level geometrically (bind-and-
    normalise), so the deepest leaf's share ~beta^-depth vanishes and depth_probe
    measured a DIM-INDEPENDENT separability collapse at d5-7. Here every level d
    rides an explicit carrier: enc = sum_d carrier_d (x) bundle(position-tagged
    nodes at depth d). Depth contribution is 1/n_levels -- LINEAR, not geometric.
    TRADE-OFF STATED (both costumes kept): this buys depth-addressability (unbind
    carrier_d, clean up) at the price of the flat encoder's holistic nesting
    algebra -- structure queries compose differently. Use flat for shallow holistic
    composition, carriers when depth must SURVIVE. Returns a unit vector."""
    rng_atom = lambda tag: (lambda v: v / np.linalg.norm(v))(
        np.random.default_rng((seed << 20) ^ (hash(tag) & 0xFFFFF)).standard_normal(dim))
    def atom(name):
        r = np.random.default_rng(
            (seed << 32) ^ int.from_bytes(
                __import__("hashlib").sha256(str(name).encode()).digest()[:4], "big"))
        v = r.standard_normal(dim)
        return v / np.linalg.norm(v)
    levels = {}
    def walk(node, depth, pos):
        if isinstance(node, (tuple, list)):
            head, kids = node[0], node[1:]
            levels.setdefault(depth, []).append(("op", head, pos))
            for i, k in enumerate(kids):
                walk(k, depth + 1, pos * 8 + i + 1)
        else:
            levels.setdefault(depth, []).append(("leaf", node, pos))
    walk(tree, 0, 0)
    RF, IRF = np.fft.rfft, np.fft.irfft
    out = np.zeros(dim)
    for d, nodes in levels.items():
        bundle = np.zeros(dim)
        for kind, name, pos in nodes:
            tagged = IRF(RF(atom("%s:%s" % (kind, name))) * RF(atom("pos:%d" % pos)),
                         n=dim)
            bundle += tagged
        carrier = atom("level:%d" % d)
        out += IRF(RF(carrier) * RF(bundle), n=dim)
    return out / (np.linalg.norm(out) or 1.0)


def depth_probe_carrier(depth, dim, n=8, seed=0):
    """depth_probe with the carrier encoder: same protocol (n trees differing only
    at the deepest leaf; worst pairwise cosine), the pre-registered gate for this
    encoder. See depth_probe for the flat baseline and the measured wall."""
    encs = []
    for i in range(int(n)):
        t = "leaf%d" % i
        for d in range(int(depth)):
            t = ("op%d" % d, t, "pad%d" % d)
        encs.append(encode_tree_carrier(t, int(dim), seed=seed))
    E = np.array(encs)
    S = E @ E.T
    np.fill_diagonal(S, -1.0)
    return {"worst_cosine": float(S.max()), "depth": int(depth), "dim": int(dim),
            "separable": bool(S.max() < 0.95), "encoder": "carrier"}


def bundle_decode(m, codebook, k, method="omp"):
    """Recover the k members of a DIRECT bundle m = sum of codebook rows.
    method='linear' is the top-k matched filter (the k* ~ 0.13*D readout ceiling);
    method='omp' is greedy successive subtraction -- MEASURED (D=512, V=1024,
    set-recovery >= 0.9, 3 seeds): linear holds k=32, OMP holds k=128 -- 4.0x the
    linear ceiling at equal accuracy. KEPT NEGATIVES: (1) least-squares refit after
    OMP adds nothing (k=128 either way); (2) this gain does NOT transfer to
    convolutive pair recall -- binding densifies the sum and the 1-bit channel wall
    (37.5%) binds there regardless (measured, item 6). PROTOCOL NOTE, honest: the
    older '~8.7x' figure on record used a different protocol; under set-recovery
    with Gaussian atoms the multiplier is 4.0x -- both numbers stand WITH their
    protocols, neither is quoted without one."""
    A = np.asarray(codebook, dtype=float)
    m = np.asarray(m, dtype=float)
    if method == "linear":
        return sorted(np.argsort(A @ m)[-int(k):].tolist())
    r = m.copy()
    got = []
    for _ in range(int(k)):
        sc = A @ r
        for j in np.argsort(sc)[::-1]:
            if int(j) not in got:
                got.append(int(j))
                r = r - A[j]
                break
    return sorted(got)


def channel_capacity_1bit(n, grid=8001):
    """Sum-capacity (bits per stored bit) of the channel a 1-bit superposed state
    ACTUALLY IS: per dimension, y = sign(x_u + z) with the user's convolution
    component x_u ~ N(0, 1/n) against interference z ~ N(0, (n-1)/n). Numeric
    I(x_u; y), times n. Converges to (2/pi)/(2 ln 2) ~ 0.4592 -- the Verdu-Shamai
    coarse-quantisation shape. THE WALL'S VERDICT: the measured three-family
    37.5% operational wall is 81% of this capacity at n=48; the remaining 19% is
    the coding gain uncoded symbol-by-symbol storage necessarily forgoes. The
    decoders were never the problem; past ~0.46 the only lever is CHANGING THE
    CHANNEL (more state bits)."""
    su = np.sqrt(1.0 / n)
    sz = np.sqrt((n - 1.0) / n)
    x = np.linspace(-6 * su, 6 * su, int(grid))
    px = np.exp(-x ** 2 / (2 * su * su)) / (su * np.sqrt(2 * np.pi))
    from math import erf as _erf
    p1 = 0.5 * (1 + np.vectorize(_erf)(x / (sz * np.sqrt(2))))
    def _hb(p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    _integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    h_y_given_x = _integrate(px * _hb(p1), x)
    py1 = _integrate(px * p1, x)
    return float(n * (_hb(np.array([py1]))[0] - h_y_given_x))


def _selftest():
    """Asserts the LAW, the ALLOCATOR, the QUANTIZATION contract, and the kept negative."""
    rng = np.random.default_rng(0)
    D, V = 512, 256

    # 1) the law's prediction is safe: storing at the predicted n* recalls >= alpha.
    n1 = capacity_law(D, V, alpha=0.90)
    ks = rng.choice(V, n1, replace=False)
    vs = rng.integers(0, V, n1)
    acc = float(np.mean(SuperposedMemory(D, V).store(ks, vs)
                        .recall(ks)["values"] == vs))
    assert acc >= 0.90, "capacity law over-promised: acc %.3f at its own n*=%d" % (acc, n1)

    # 2) int8 is decision-free at that load (the measured contract).
    a8 = float(np.mean(SuperposedMemory(D, V, precision="int8").store(ks, vs)
                       .recall(ks)["values"] == vs))
    assert abs(a8 - acc) < 0.03, "int8 changed decisions: %.3f vs %.3f" % (a8, acc)

    # 3) allocator single-shot HIT: demand 3x the D=512 capacity, allocate, verify.
    n_demand = 3 * n1
    D2 = allocate(n_demand, V, alpha=0.90)
    ks2 = np.random.default_rng(1).choice(V, n_demand, replace=False)
    vs2 = np.random.default_rng(2).integers(0, V, n_demand)
    acc2 = float(np.mean(SuperposedMemory(D2, V).store(ks2, vs2)
                         .recall(ks2)["values"] == vs2))
    assert acc2 >= 0.90, "allocator missed: %.3f at D=%d for n=%d" % (acc2, D2, n_demand)

    # 4) PIC beats one-shot inside its basin...
    n_mid = int(1.3 * n1)
    ks3 = np.random.default_rng(3).choice(V, n_mid, replace=False)
    vs3 = np.random.default_rng(4).integers(0, V, n_mid)
    mem3 = SuperposedMemory(D, V).store(ks3, vs3)
    a_one = float(np.mean(mem3.recall(ks3)["values"] == vs3))
    r_pic = mem3.recall(ks3, decoder="pic")
    a_pic = float(np.mean(r_pic["values"] == vs3))
    assert a_pic >= a_one - 1e-9, "PIC lost inside its basin (%.3f < %.3f)" % (a_pic, a_one)

    # 5) ...and the GATE refuses it above the transition (the kept negative enforced).
    n_hot = 2 * pic_transition(D, V) + 4
    ks4 = np.random.default_rng(5).choice(V, min(n_hot, V), replace=False)
    vs4 = np.random.default_rng(6).integers(0, V, len(ks4))
    r = SuperposedMemory(D, V).store(ks4, vs4).recall(ks4, decoder="pic")
    assert r["decoder"] == "one-shot" and "GATED" in r["why"], "load gate failed to fire"

    # 6) BigPairMemory: same channel, zero materialised codebooks, big vocab.
    nb, Vb = 200, 4096
    Db = allocate(nb, Vb)
    kb = np.random.default_rng(7).choice(Vb, nb, replace=False)
    vb = np.random.default_rng(8).integers(0, Vb, nb)
    rb = BigPairMemory(Db, Vb, seed=0).store(kb, vb).recall(kb)
    accb = float(np.mean(rb["values"] == vb))
    assert accb >= 0.90, "BigPairMemory below spec: %.3f at D=%d" % (accb, Db)

    # 7) the advisor: binding constraint named, prescription exact, fix applies it.
    a = advise_scale(n_pairs=200, vocab=1000, dim=512, fix=True)
    assert not a["ok"] and "pair-capacity" in a["binding"]
    assert a["fixed_spec"]["dim"] == allocate(200, 1000)
    b = advise_scale(bundle_k=200, dim=512)
    assert not b["ok"] and "0.13" in b["binding"] and "PARTITION" in b["prescription"]
    c = advise_scale(factors=7)
    assert not c["ok"] and "2 factor" in c["prescription"]
    d = advise_scale(n_pairs=20, vocab=256, dim=2048)
    assert d["ok"], d
    e = advise_scale(depth=9)
    assert not e["ok"] and "dim is NOT the lever" in e["prescription"]

    # 11) the wall's verdict is REPRODUCIBLE: capacity of the 1-bit channel
    #     matches the Verdu-Shamai limit shape, and the measured 37.5% sits at
    #     ~81% of it -- near-capacity, decoders exonerated.
    cap = channel_capacity_1bit(48)
    assert abs(channel_capacity_1bit(512) - (2 / np.pi) / (2 * np.log(2))) < 0.002
    assert 0.78 < 0.375 / cap < 0.84, (cap, 0.375 / cap)

    # 10) sparse bundle decoding: OMP holds >= 3x the linear top-k on a direct
    #     bundle at equal set-recovery (measured 4.0x; 3x is the regression floor).
    rngc = np.random.default_rng(21)
    Ac = rngc.standard_normal((512, 384))
    Ac /= np.linalg.norm(Ac, axis=1, keepdims=True)
    Sc = rngc.choice(512, 72, replace=False)
    mc = Ac[Sc].sum(0)
    lin = set(bundle_decode(mc, Ac, 72, method="linear"))
    omp = set(bundle_decode(mc, Ac, 72, method="omp"))
    tru = set(Sc.tolist())
    assert len(omp & tru) / 72 >= 0.9 > len(lin & tru) / 72, \
        (len(omp & tru) / 72, len(lin & tru) / 72)

    # 9) carrier encoder: separable at d7 where flat is stone dead, and the deep
    #     leaf stays READABLE via carrier unbind at d16 (the lever's true payoff).
    pc7 = depth_probe_carrier(7, 512)
    assert pc7["separable"], pc7
    import hashlib as _h
    def _atom(name, dim):
        r = np.random.default_rng(int.from_bytes(
            _h.sha256(str(name).encode()).digest()[:4], "big"))
        v = r.standard_normal(dim)
        return v / np.linalg.norm(v)
    _RFb, _IRFb = np.fft.rfft, np.fft.irfft
    cands = [_atom("leaf:leaf%d" % i, 512) for i in range(8)]
    hits = 0
    for i in range(8):
        t = "leaf%d" % i
        for d in range(16):
            t = ("op%d" % d, t, "pad%d" % d)
        enc = encode_tree_carrier(t, 512)
        lvl = _IRFb(np.conj(_RFb(_atom("level:16", 512))) * _RFb(enc), n=512)
        pos = 0
        for d in range(16):
            pos = pos * 8 + 1
        est = _IRFb(np.conj(_RFb(_atom("pos:%d" % pos, 512))) * _RFb(lvl), n=512)
        hits += int(np.argmax([est @ c for c in cands]) == i)
    assert hits >= 7, "deep-leaf readout at d16: %d/8" % hits

    # 8) bin+PIC beats bin one-shot at the same 1-bit state (the utilization uplift).
    rngb = np.random.default_rng(11)
    memb = SuperposedMemory(1024, 256, seed=11)
    kb2 = rngb.choice(256, 44, replace=False); vb2 = rngb.integers(0, 256, 44)
    memb.store(kb2, vb2)
    a_one = float(np.mean(memb.recall(kb2, state_bits=1)["values"] == vb2))
    a_pic = float(np.mean(memb.recall(kb2, decoder="pic", state_bits=1)["values"] == vb2))
    assert a_pic > a_one and a_pic >= 0.9, (a_one, a_pic)

    # F2 MODE PINS (dedicated RNG per plant): hadamard exact at the law with NO stored codebooks;
    # lazy at vocab ONE MILLION -- O(1) construction, recall 1.0 at the law (dense would be 2x16GB);
    # the vocab>2*dim hadamard refusal is pinned as the honest boundary.
    _ns2 = 20
    _rng_h = np.random.default_rng(7701)
    _sh = SuperposedMemory(1024, 2048, seed=0, codebook="hadamard")
    _kh = _rng_h.choice(2048, _ns2, replace=False); _vh = (_kh * 13 + 5) % 2048
    assert float((_sh.store(_kh, _vh).recall(_kh)["values"] == _vh).mean()) == 1.0, "hadamard at the law"
    assert _sh.K is None and _sh.V is None, "hadamard must store no codebook arrays"
    _rng_l = np.random.default_rng(7702)
    _sl = SuperposedMemory(1024, 1_000_000, seed=0, codebook="lazy")
    _kl = _rng_l.choice(1_000_000, _ns2, replace=False); _vl = (_kl * 31 + 7) % 1_000_000
    assert float((_sl.store(_kl, _vl).recall(_kl)["values"] == _vl).mean()) == 1.0, "lazy vocab=1M at the law"
    try:
        SuperposedMemory(256, 4096, codebook="hadamard"); raise AssertionError("must refuse vocab > 2*dim")
    except ValueError:
        pass

    print("holographic_superposed selftest OK -- law n*=%d holds (acc %.3f), int8 free, "
          "allocator hit D=%d for n=%d (acc %.3f), PIC %.3f>=%.3f in basin, gate fires"
          % (n1, acc, D2, n_demand, acc2, a_pic, a_one))


if __name__ == "__main__":
    _selftest()
