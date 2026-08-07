"""holographic_voidexplore.py -- VOID-1: the disciplined explorer of what a corpus implies but does not contain.

THE CLAIM: "undiscovered" is a measurable set, not a mood. Given a corpus, three instruments that
already exist in this engine, pointed at ABSENCE instead of presence, yield candidates that are real-
or-possible rather than imagined:

  WHERE IS NOTHING       the drift model's zeroth moment: z(x) = <enc(x), mu> is a KDE readout in one
                         dot product. A void is z ~ 0 INSIDE the support box. (holographic_hdrift)
  WHAT COULD BE THERE    the corpus's own structure: role-filler combinations the observed set
                         licenses but never instantiated -- the Mendeleev move. Gallium and germanium
                         were read off exactly this intersection: valid under the table's grammar,
                         absent from the observations. (holographic_ladder / learn-chunks discipline)
  IS THE VOID REAL       the shuffled-null gate: a finite sample of ANY distribution has low-density
                         pockets, and an overgenerating grammar (epicycles, aether, phlogiston) will
                         happily vouch for nonsense. A void counts only if it is deeper than the voids
                         that resampling noise alone produces; a grammar may vouch only if its
                         structure beats a shuffle. (the permutation_null / gain_over_null discipline)

TRANSFER -- the cross-disciplinary gate, strictly stronger than validity: a candidate ABSENT in corpus
A but PRESENT in corpus B (both read through one encoder space) is not merely grammatical, it is
instantiated somewhere real. Fourier's heat mathematics, Shannon's Boolean circuits: a shared level
between two corpora that neither corpus announces. Here it is literally z_A low AND z_B high.

WHAT THIS IS NOT (the honest boundary, stated in the module that most needs it): the explorer finds
what the corpus's structure implies and has not shown -- interpolations, legal recombinations,
transported patterns. It CANNOT find what needs an axiom the tower never climbed: Mendeleev could
predict gallium; the table could not predict quantum mechanics. Every report therefore carries its
warrant ('grammar', 'transfer') and its gate verdict; a candidate with neither is never returned.

REUSED, NOT REBUILT (Rule-0 on record: every 'find what is missing' phrasing returned fallbacks):
drift moments/fields from holographic_hdrift; the null discipline from permutation_null's pattern
(procedure-matched resamples scored identically); chunk promotion mirrors holographic_chunks.
"""

import numpy as np

from holographic.sampling_and_signal.holographic_hdrift import (
    DriftModel, build_drift_model, drift_moments, drift_field)


# ---------------------------------------------------------------------------------------------------
# WHERE IS NOTHING -- the continuous void map, null-gated.
# ---------------------------------------------------------------------------------------------------

def void_probe(model, x):
    """The raw instrument: density z(x) = <enc(x), mu> at one point. One dot product, N-independent.
    Interpretation is the caller's problem -- use void_map for the gated version."""
    return float(model.mu @ model.enc.encode(np.asarray(x, float)))


def void_map(model, train, n_probes=512, seed=0, n_null=24, alpha=0.05):
    """Map the REAL voids of a trained drift model: probe the support box, keep low-density points,
    and gate each against the resampling null -- 'a finite sample of anything has pockets', so a void
    counts only where z sits below what bootstrap-resampled corpora produce AT THAT POINT.

    The null is procedure-matched (the permutation_null discipline): rebuild the SAME moments from
    n_null bootstrap resamples of the training set and score the SAME probes; a probe whose real z is
    below the alpha-quantile of its own null distribution is a gated void. Everything else is
    reported as 'sparsity' -- visible thinness the data's own noise explains.

    Returns {'voids': (m, d) points, 'sparsity': points, 'z': per-probe density,
             'null_lo': per-probe alpha-quantile, 'probes': all probes}. Deterministic in seed."""
    Y = np.asarray(train, float)
    lo = np.array([b[0] for b in model.bounds]); hi = np.array([b[1] for b in model.bounds])
    # THE INSTRUMENT IS NOT THE SAMPLER. The model's bandwidth was probed for FIELD fidelity
    # (closest-to-unit spread -- right for generation), and that smoothness SMEARS absence: measured,
    # the inter-mode gap read 56% of data density at the sampler's bw 4.0 and -6% at bw 10 (FPE
    # convention: larger bandwidth = sharper kernel). So the void map probes its own bandwidth and
    # takes the SHARPEST candidate inside the honest window -- maximum resolution that is not yet
    # amplification -- and builds dedicated moments at that setting.
    from holographic.sampling_and_signal.holographic_hdrift import probe_bandwidth
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
    rep = probe_bandwidth(Y, dim=model.enc.dim, seed=seed)
    win_lo, win_hi = rep.get("window", (0.40, 2.5))
    honest = [b for b, s in rep["scores"].items() if win_lo < s < win_hi]
    bw_sharp = max(honest) if honest else model.enc.bandwidth[0]
    enc = VectorFunctionEncoder(len(lo), dim=model.enc.dim, bounds=model.bounds,
                                bandwidth=bw_sharp, seed=seed)
    mu, _ = drift_moments(Y, enc)
    rng = np.random.default_rng(seed)
    probes = rng.uniform(lo, hi, (int(n_probes), len(lo)))
    E = enc.encode_many(probes)                                # (P, dim) -- encode probes ONCE
    z = E @ mu
    # the null: same probes, same encoder, moments from resampled corpora. Bootstrap (with
    # replacement, same n) rather than shuffle -- coordinates are meaningful here and a shuffle would
    # destroy the support itself, testing the wrong hypothesis.
    znull = np.empty((int(n_null), len(probes)))
    for j in range(int(n_null)):
        rs = np.random.default_rng(1000 + seed * 131 + j)
        mu_b, _ = drift_moments(Y[rs.integers(0, len(Y), len(Y))], enc)
        znull[j] = E @ mu_b
    null_lo = np.quantile(znull, alpha, axis=0)
    # a void must be BOTH absolutely empty (z near zero against the data's own density scale) and
    # not above its null band -- absolute emptiness alone can be resampling luck at the margin, and
    # the band alone flags dense-but-variable regions.
    z_data = float(np.mean(enc.encode_many(Y) @ mu))
    is_void = (z < 0.05 * z_data) & (z <= null_lo + 0.02 * z_data)
    is_sparse = (z < 0.25 * z_data) & ~is_void
    return {"voids": probes[is_void], "sparsity": probes[is_sparse], "z": z,
            "null_lo": null_lo, "probes": probes, "z_data_mean": z_data,
            "instrument_bandwidth": bw_sharp}


# ---------------------------------------------------------------------------------------------------
# WHAT COULD BE THERE -- the Mendeleev move on a discrete corpus: valid under the observed structure,
# absent from the observations, gated by the structure's own right to vouch.
# ---------------------------------------------------------------------------------------------------

def structured_voids(observations, min_count=2, max_candidates=64, seed=0):
    """Given observations as tuples over discrete slots (role-filler structures: rows of a table,
    (subject, relation, object) triples, parameter records), return the combinations the observed
    STRUCTURE licenses but the observed SET lacks.

    The grammar is deliberately the weakest one that carried Mendeleev: per-slot alphabets from the
    observations, candidate = any cross-slot combination whose every PAIRWISE (slot_i=a, slot_j=b)
    co-occurrence was observed >= min_count times. Pairwise support means the parts are known to be
    mutually compatible somewhere; only the full assembly is new -- 'two of three slots shared', the
    generate_structure finding in discrete costume.

    THE VOUCHING GATE (the anti-epicycle clause): the grammar may propose only if its pairwise
    structure beats a shuffle -- observed co-occurrence concentration vs the same statistic on
    slot-wise shuffled corpora (structure destroyed, marginals kept). Below the null, the honest
    answer is that this corpus's slots are independent, EVERY unseen combination is equally 'valid',
    and the void list would be noise wearing a grammar; we refuse and say so.

    Returns {'candidates': [...], 'warrant': 'grammar', 'gate': {...}} or a refusal dict."""
    obs = [tuple(o) for o in observations]
    if not obs:
        return {"candidates": [], "warrant": None, "gate": {"why": "empty corpus"}}
    width = len(obs[0])
    seen = set(obs)
    # pairwise co-occurrence counts, observed
    def pair_counts(rows):
        c = {}
        for r in rows:
            for i in range(width):
                for j in range(i + 1, width):
                    c[(i, r[i], j, r[j])] = c.get((i, r[i], j, r[j]), 0) + 1
        return c
    real = pair_counts(obs)
    # concentration statistic: how far pair mass deviates from independence. Shuffle each slot's
    # column independently -> marginals identical, structure gone; the real corpus must stand out.
    def concentration(counts, n):
        p = np.array([v / n for v in counts.values()])
        return float((p * np.log(p * len(p) + 1e-12)).sum())   # KL-ish vs uniform over occupied pairs
    stat_real = concentration(real, len(obs))
    rng = np.random.default_rng(seed)
    null_stats = []
    cols = [ [r[i] for r in obs] for i in range(width) ]
    for _ in range(48):
        shuf = list(zip(*[list(rng.permutation(c)) for c in cols]))
        null_stats.append(concentration(pair_counts(shuf), len(obs)))
    p_val = (1 + sum(s >= stat_real for s in null_stats)) / (1 + len(null_stats))
    gate = {"stat": stat_real, "null_mean": float(np.mean(null_stats)), "p": p_val}
    if p_val > 0.05:
        return {"candidates": [], "warrant": None, "gate": gate,
                "why": "the corpus's slot structure does not beat a shuffle -- its grammar has no "
                       "right to vouch for unseen combinations (the epicycle refusal)"}
    # enumerate candidates: full combinations, unseen, with EVERY pair supported.
    alphabets = [sorted(set(c)) for c in cols]
    out = []
    # deterministic bounded enumeration -- product order over sorted alphabets, early-capped.
    def rec(prefix):
        if len(out) >= max_candidates:
            return
        i = len(prefix)
        if i == width:
            t = tuple(prefix)
            if t not in seen:
                out.append(t)
            return
        for a in alphabets[i]:
            ok = all(real.get((j, prefix[j], i, a), 0) >= min_count for j in range(i))
            if ok:
                rec(prefix + [a])
    rec([])
    return {"candidates": out, "warrant": "grammar", "gate": gate}


# ---------------------------------------------------------------------------------------------------
# TRANSFER -- present in B, absent in A: the cross-disciplinary warrant.
# ---------------------------------------------------------------------------------------------------

def transfer_voids(model_a, model_b, n=32, seed=0, thresh=0.15):
    """Candidates for corpus A's void that are INSTANTIATED in corpus B: sample B's drift model,
    keep points where A's density is below `thresh` of A's own on-support scale while B's is above
    it. Both models must share one encoder space (enforced by the hdrift algebra's rule).

    This is the strongest warrant short of execution: not 'the grammar allows it' but 'reality
    already contains it, elsewhere'. Fourier / Shannon / the unifier registry, as a query.
    Returns {'candidates', 'z_a', 'z_b', 'warrant': 'transfer'}."""
    from holographic.sampling_and_signal.holographic_hdrift import drift_sample, _same_space
    _same_space(model_a, model_b)
    X = drift_sample(model_b, n=int(n), seed=seed)
    Ea = model_a.enc.encode_many(X)
    za = Ea @ model_a.mu
    zb = model_b.enc.encode_many(X) @ model_b.mu
    # scale each against its own typical on-support density, probed from the model's own samples --
    # a raw z threshold would silently encode one dataset's size into the other's verdict.
    Xa = drift_sample(model_a, n=min(int(n), 32), seed=seed + 1)
    za_scale = float(np.mean(model_a.enc.encode_many(Xa) @ model_a.mu)) or 1e-9
    zb_scale = float(np.mean(zb)) or 1e-9
    keep = (za / za_scale < thresh) & (zb / zb_scale > thresh)
    return {"candidates": X[keep], "z_a": za / za_scale, "z_b": zb / zb_scale,
            "warrant": "transfer", "kept": int(keep.sum()), "of": int(n)}


# ---------------------------------------------------------------------------------------------------
# Selftest: planted ground truth for all three instruments, refusals included.
# ---------------------------------------------------------------------------------------------------

def _selftest():
    rng = np.random.default_rng(0)

    # --- void_map: two modes with a known gap; the gap must gate as void, the modes must not --------
    centers = np.array([[0.25, 0.25], [0.75, 0.75]])
    data = np.vstack([c + 0.05 * rng.standard_normal((80, 2)) for c in centers])
    m = build_drift_model(data, dim=2048, seed=0)
    vm = void_map(m, data, n_probes=400, seed=0)
    assert len(vm["voids"]) > 0, "the planted inter-mode gap must surface as gated voids"
    d_void_to_gap = np.linalg.norm(vm["voids"] - np.array([0.5, 0.5]), axis=1)
    d_void_to_modes = np.min(
        np.stack([np.linalg.norm(vm["voids"] - c, axis=1) for c in centers]), axis=0)
    assert (d_void_to_modes > 0.15).mean() > 0.9, \
        "gated voids must not sit on the modes (%.2f violated)" % (d_void_to_modes <= 0.15).mean()

    # --- void_map refusal side: a UNIFORM corpus has no voids beyond null ---------------------------
    uni = rng.uniform(0.1, 0.9, (160, 2))
    mu_ = build_drift_model(uni, dim=2048, seed=0)
    vmu = void_map(mu_, uni, n_probes=400, seed=0)
    frac_void = len(vmu["voids"]) / 400.0
    assert frac_void < 0.08, \
        "a uniform corpus must show (almost) no gated voids -- got %.2f (the sparsity!=void clause)" % frac_void

    # --- structured_voids: the Mendeleev test -- hold out combinations, recover them ----------------
    # corpus over 3 slots where slots are CORRELATED (structure real); hold out 2 full combinations
    # whose every pair is still observed elsewhere.
    rows = []
    for a in "AB":
        for b in "xy":
            for c in "12":
                rows += [(a, b, c)] * 3
    held = [("A", "x", "1"), ("B", "y", "2")]
    corpus = [r for r in rows if r not in held]
    # inject correlation so the shuffle gate passes: extra mass on matched combos
    corpus += [("A", "x", "2")] * 6 + [("B", "y", "1")] * 6
    sv = structured_voids(corpus, min_count=2)
    assert sv["warrant"] == "grammar", "structured corpus must pass the vouching gate: %s" % sv.get("gate")
    assert all(h in sv["candidates"] for h in held), \
        "held-out valid combinations must be recovered (got %s)" % sv["candidates"][:6]

    # --- structured_voids refusal: independent slots -> the epicycle refusal ------------------------
    ri = np.random.default_rng(3)
    indep = [(ri.choice(list("ABCD")), ri.choice(list("wxyz")), ri.choice(list("1234")))
             for _ in range(120)]
    svi = structured_voids([tuple(map(str, t)) for t in indep])
    assert svi["warrant"] is None and svi["candidates"] == [], \
        "independent slots must be refused, not enumerated (p=%.3f)" % svi["gate"]["p"]

    # --- transfer_voids: B holds a third mode A lacks; candidates must land there -------------------
    shared = [(0.0, 1.0), (0.0, 1.0)]
    A = build_drift_model(data, dim=2048, seed=0, bandwidth=6.0, bounds=shared)
    dataB = np.vstack([data, np.array([0.2, 0.8]) + 0.05 * rng.standard_normal((80, 2))])
    B = build_drift_model(dataB, dim=2048, seed=0, bandwidth=6.0, bounds=shared)
    tv = transfer_voids(A, B, n=48, seed=2)
    assert tv["kept"] > 0, "B's extra mode must yield transfer candidates for A"
    d3 = np.linalg.norm(tv["candidates"] - np.array([0.2, 0.8]), axis=1)
    assert (d3 < 0.2).mean() > 0.7, \
        "transfer candidates must concentrate on the mode A lacks (%.2f did)" % (d3 < 0.2).mean()
    # and symmetry of honesty: A has nothing B lacks, so the reverse direction stays (near-)empty
    tv_rev = transfer_voids(B, A, n=48, seed=2)
    assert tv_rev["kept"] <= max(2, tv["kept"] // 3), \
        "the reverse transfer must be (near-)empty -- A contains nothing B lacks (kept %d)" % tv_rev["kept"]

    print("holographic_voidexplore selftest OK -- planted void found, uniform refused, Mendeleev "
          "recovered, epicycles refused, transfer directional")


if __name__ == "__main__":
    _selftest()
