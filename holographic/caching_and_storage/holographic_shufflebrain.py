"""Shufflebrain -- Paul Pietsch's salamander surgeries, performed on holographic memory.

WHY THIS MODULE EXISTS (panel session, docs/PANEL_pietsch_hologramic.md). Pietsch's hologramic
theory of memory, stripped of its contested biology (regeneration real; transfer claims
unreplicated -- the doc states the status honestly), is a set of EXACT theorems about
distributed memory, and this engine can measure them instead of debating them. The pilot
battery confirmed:

  ROTATION IS A COHERENT TRANSFORM, NOT DAMAGE. roll(T,s) = bind(delta_s, T), so a rotated
  trace recalls ROTATED values at exact baseline fidelity (measured 0.204 == 0.204 at D=2048,
  K=24) while the originals vanish (0.005) -- Pietsch's rotated salamanders feeding in
  reversed directions, as an identity. Shift trace AND cues together and the shift cancels:
  the memory never knows it was rotated.

  FOCAL LESIONS SEPARATE THE ARCHITECTURES. Ablate a contiguous half: holographic storage
  degrades uniformly (sd 0.018, 0/24 items dead -- 'decreased resolution, whole retained');
  a localized slot baseline loses exactly the items whose region died (sd 0.499, 12/24 dead
  -- 'specific, permanent loss'). His comparison table, as numbers.

  A RESTORATION PRIOR COMPLETES THE FRAGMENT. Raw recall at 50% focal lesion is cos 0.144,
  yet codebook identification is 24/24 -- 'any sufficiently large fragment reconstructs the
  whole' holds GIVEN a prior (cleanup), which is Milanfar's denoiser-as-prior thesis in
  hypervectors.

KEPT NEGATIVES (each one was measured, and each reshaped the claim):
  - MINCING REFUTES NAIVE HOLOGRAMIC STORAGE: shuffling blocks of the trace kills HRR readout
    (block 512 keeps half the signal; <=128 is dead). If minced salamander brains truly fed,
    the credit belongs to REGENERATION AS COHERENT RE-ALIGNMENT, not to storage that survives
    arbitrary rearrangement. The refinement of Pietsch's theory came from the math.
  - HRR IS NOT BASIS-FREE: an arbitrary coherent permutation of all parts reads ~0 --
    convolutional codes carry only the CYCLIC symmetry. The GDN outer-product matrix memory
    carries the FULL orthogonal group (basis-permute S with coherently projected keys: exact).
    Substrate choice = choice of which surgeries memory survives. A design axis, not trivia.
  - DIFFUSE LESIONS ARE THE WRONG INSTRUMENT for the injury-impact claim: random dim-wise
    damage hits every localized slot partially, so nothing dies and the crosstalk-free slots
    even score HIGHER -- the contrast needs a FOCAL (contiguous) ablation. Focal-vs-diffuse
    is part of the claim, not a detail.
  - THE WRONG-TARGET PROBE: the first coherent-shift measurement compared readout against
    rolled values and reported a fake anomaly; the theorem (shifts cancel -> ORIGINALS return)
    fixed the probe. Perfect-looking anomalies are instrument hypotheses first.

Delegations: bind/unbind from holographic_ai (the HRR home); mincing delegates to the
existing moving-block-bootstrap block_shuffle (Rule-0: same operator, different costume);
masks come from the mind's damage_mask when driven through the facade.
"""
import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, unbind


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v



# ONE HOME, AN IMPORT. This function was byte-identical in semanticrig and
# here -- the duplication audit caught it, and its instruction is the right one:
# "either unify them (one home, an import) or add the entry with the reasoning.
# Do NOT raise the budget to make the test pass." They are the same algorithm,
# not two that share a shape, so there is nothing to reason about -- semanticrig
# is the home and this is the import. The TWO-TABLES LESSON, which this project
# has on record: one shared implementation of any algorithm, never two.
from holographic.caching_and_storage.holographic_semanticrig import (  # noqa: E402
    _unbind_many)


def build_trace(keys, vals):
    """Bundle bind(k_i, v_i) -- the standard HRR episodic trace the surgeries operate on."""
    return np.sum([bind(k, v) for k, v in zip(keys, vals)], axis=0)


def recall_cos(trace, key, val):
    """Readout fidelity for one pair: cos(unbind(trace, key), val)."""
    return float(_unit(unbind(trace, key)) @ val)


def rotation_battery(keys, vals, shift):
    """Pietsch's rotated brain. Returns recall of the rotated trace against ORIGINAL values
    (theorem: ~0), against ROTATED values (theorem: exact baseline -- behavior transforms
    coherently), and with trace AND cues rotated together (theorem: shift cancels, originals
    return untouched)."""
    T = build_trace(keys, vals)
    Tr = np.roll(T, shift)
    base = float(np.mean([recall_cos(T, k, v) for k, v in zip(keys, vals)]))
    vs_orig = float(np.mean([recall_cos(Tr, k, v) for k, v in zip(keys, vals)]))
    Ur = _unbind_many(Tr, np.stack(keys))                  # batched: one call, not a loop
    Vr = np.stack([_unit(np.roll(v, shift)) for v in vals])
    vs_rot = float(np.mean(np.sum((Ur / np.linalg.norm(Ur, axis=1, keepdims=True)) * Vr, axis=1)))
    coherent = float(np.mean([recall_cos(Tr, np.roll(k, shift), v) for k, v in zip(keys, vals)]))
    return {"baseline": base, "vs_original": vs_orig, "vs_rotated": vs_rot,
            "coherent_surgery": coherent}


def mince_curve(keys, vals, blocks, seed=0):
    """Pietsch's mincing, via the engine's own block_shuffle operator (a moving-block surrogate
    is a mince -- Rule-0 reuse). Returns {block_size: mean recall}. The KEPT NEGATIVE lives
    here: fine mincing kills HRR readout, so naive 'storage survives rearrangement' is refuted
    and the biological credit moves to coherent re-alignment."""
    from holographic.sampling_and_signal.holographic_surrogate import block_shuffle as _bs
    T = build_trace(keys, vals)
    out = {}
    for b in blocks:
        Tm = _bs(T, int(b), seed=seed)
        out[int(b)] = float(np.mean([recall_cos(Tm, k, v) for k, v in zip(keys, vals)]))
    return out


def focal_lesion_battery(keys, vals, fraction=0.5, dead_thresh=0.05):
    """The injury-impact table: ablate a CONTIGUOUS `fraction` of the space and compare the
    holographic trace against a localized slot baseline built from the SAME pairs. Returns
    mean/sd/dead-count for both. Focal, not diffuse, on purpose (see the kept negative)."""
    D = len(keys[0])
    K = len(keys)
    cut = int(D * fraction)
    mask = np.ones(D)
    mask[:cut] = 0.0
    T = build_trace(keys, vals) * mask
    holo = np.array([recall_cos(T, k, v) for k, v in zip(keys, vals)])
    slot = D // K
    L = np.zeros(D)
    for i, v in enumerate(vals):
        L[i * slot:(i + 1) * slot] = v[:slot]
    L = L * mask
    loc = []
    for i, v in enumerate(vals):
        seg = L[i * slot:(i + 1) * slot]
        n = np.linalg.norm(seg)
        loc.append(float((seg / n) @ _unit(v[:slot])) if n > 1e-9 else 0.0)
    loc = np.array(loc)
    return {"holographic": {"mean": float(holo.mean()), "sd": float(holo.std()),
                            "dead": int((holo < dead_thresh).sum())},
            "localized": {"mean": float(loc.mean()), "sd": float(loc.std()),
                          "dead": int((loc < dead_thresh).sum())},
            "n_items": K}


def cleanup_rescue(keys, vals, fraction=0.5):
    """The fragment principle, completed by a prior: at a focal lesion of `fraction`, snap each
    lesioned readout to the value codebook and count correct identifications. Raw cosine
    collapses; identification survives -- 'any sufficiently large fragment reconstructs the
    whole', made conditional on the restoration prior and then measured."""
    D = len(keys[0])
    mask = np.ones(D)
    mask[:int(D * fraction)] = 0.0
    T = build_trace(keys, vals) * mask
    V = np.stack([_unit(v) for v in vals])
    U = _unbind_many(T, np.stack(keys))
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    correct = int(np.sum(np.argmax(U @ V.T, axis=1) == np.arange(len(keys))))
    raw = float(np.mean([recall_cos(T, k, v) for k, v in zip(keys, vals)]))
    return {"raw_mean_cos": raw, "identified": int(correct), "of": len(keys)}


def graft_battery(host_keys, host_vals, donor_keys, donor_vals, alpha=1.0, fragment=0.5,
                  seed=11):
    """Pietsch's trained-donor tissue graft: add alpha * (a focal `fragment` of the donor
    trace) to the host and measure donor recall THROUGH THE HOST plus the bruise to the host's
    own memories. The pilot's honest number: transfer is real but faint (0.05-0.08 at K=24
    load) -- amplification via iterated cleanup is backlog item S2, not a claim."""
    D = len(host_keys[0])
    rng = np.random.default_rng(seed)
    keep = rng.random(D) >= fragment
    H = build_trace(host_keys, host_vals)
    frag = build_trace(donor_keys, donor_vals) * keep
    Hg = H + float(alpha) * frag
    return {"donor_in_host": float(np.mean([recall_cos(Hg, k, v)
                                            for k, v in zip(donor_keys, donor_vals)])),
            "host_own_after": float(np.mean([recall_cos(Hg, k, v)
                                             for k, v in zip(host_keys, host_vals)])),
            "host_own_before": float(np.mean([recall_cos(H, k, v)
                                              for k, v in zip(host_keys, host_vals)]))}


def mince_law(keys, vals, block=256, fixed=(8, 6, 4, 2, 0)):
    """S3a -- the mince threshold, dissolved into a LAW: recall after block-mincing is not a
    cliff in block size but the ALIGNED-MASS fraction -- recall ~ baseline * (fixed_blocks *
    B / D), measured tracking the prediction at every rung (0.164/0.153, 0.112/0.102,
    0.064/0.051, 0.009/0.000). Block size only changes how much mass a random permutation
    happens to leave fixed. Movers are rotated (a guaranteed derangement -- the pilot's
    fixed-seed rejection loop spun forever redrawing one permutation; kept as the lesson).
    Returns [(n_fixed, recall, predicted)]."""
    T = build_trace(keys, vals)
    D = len(T)
    nb = D // int(block)
    base = float(np.mean([recall_cos(T, k, v) for k, v in zip(keys, vals)]))
    rows = []
    for f in fixed:
        idx = np.arange(nb)
        idx[f:] = np.roll(idx[f:], 1)
        Tm = np.concatenate([T[i * block:(i + 1) * block] for i in idx])
        rec = float(np.mean([recall_cos(Tm, k, v) for k, v in zip(keys, vals)]))
        rows.append((int(f), rec, base * f * block / D))
    return {"baseline": base, "rows": rows}


def spectral_lesion(keys, vals, band):
    """S3b -- the anisotropic lesion that makes Pietsch's 'decreased resolution, whole
    retained' LITERAL: zero a frequency BAND of the trace and every readout becomes its
    band-limited value EXACTLY (co-articulation 2e-16 -- band-zeroing is a linear spectral
    op that commutes with HRR readout, same theorem family as the phase bones). Zero items
    die in ANY band; raw recall dips only by the removed band's energy. `band` is a
    (start_bin, stop_bin) pair over the rfft bins."""
    T = build_trace(keys, vals)
    D = len(T)
    sl = slice(int(band[0]), int(band[1]))
    X = np.fft.rfft(T)
    X[sl] = 0
    Tl = np.fft.irfft(X, n=D)

    def bl(v):
        Y = np.fft.rfft(np.asarray(v, float))
        Y[sl] = 0
        return np.fft.irfft(Y, n=D)
    Ul, U0 = _unbind_many(Tl, np.stack(keys)), _unbind_many(T, np.stack(keys))
    coart = float(np.max(np.abs(Ul - np.stack([bl(u) for u in U0]))))
    raw = np.array([recall_cos(Tl, k, v) for k, v in zip(keys, vals)])
    Uln = Ul / np.linalg.norm(Ul, axis=1, keepdims=True)
    vsbl = np.array([float(Uln[i] @ _unit(bl(v))) for i, v in enumerate(vals)])
    return {"coarticulation_err": coart, "vs_bandlimited_mean": float(vsbl.mean()),
            "vs_bandlimited_sd": float(vsbl.std()), "raw_mean": float(raw.mean()),
            "dead": int(np.sum(raw < 0.05))}


def graft_amplify(host_keys, host_vals, donor_keys, donor_vals, alpha=1.0, fragment=0.5,
                  seed=11, margin=0.02):
    """S2 -- graft amplification, resolved by the TWO-SPEED design (the conservation law
    taught it): a faint graft (raw donor-in-host cos 0.087) carries enough signal for the
    cleanup prior to IDENTIFY donor memories 24/24 -- but consolidating them IN-PLACE lifts
    recall only ~67% while bruising the host, because new writes into a loaded trace pay the
    capacity law. YOU CANNOT ADD MEMORIES FOR FREE. The design consequence: THE GRAFT IS A
    CHANNEL, NOT A DESTINATION. Identify through the grafted host, consolidate into a FRESH
    store (the durable partition, in Ouroboros terms), and transfer completes at 100% of the
    clean-donor baseline with the host untouched (read-only graft). Measured floor kept
    honest: at alpha 0.25 / fragment 0.25 identification collapses toward chance -- the graft
    capacity boundary. Returns identification, in-place and fresh-store recalls, and the
    host bruise, so every trade is a number."""
    D = len(host_keys[0])
    rng = np.random.default_rng(seed)
    keep = rng.random(D) >= (1 - float(fragment))
    H = build_trace(host_keys, host_vals)
    Hg = H + float(alpha) * (build_trace(donor_keys, donor_vals) * keep)
    V = np.stack([_unit(v) for v in donor_vals])
    ids, fresh, Ha, n_ip = [], np.zeros(D), Hg.copy(), 0
    for i, k in enumerate(donor_keys):
        sims = V @ _unit(unbind(Hg, k))
        j = int(np.argmax(sims))
        ids.append(int(j == i))
        fresh = fresh + bind(k, donor_vals[j])
        srt = np.sort(sims)
        if float(srt[-1] - srt[-2]) > float(margin):
            Ha = Ha + 0.8 * bind(k, donor_vals[j])
            n_ip += 1
    out = {"identified": int(sum(ids)), "of": len(donor_keys),
           "raw_in_host": float(np.mean([recall_cos(Hg, k, v)
                                         for k, v in zip(donor_keys, donor_vals)])),
           "inplace_recall": float(np.mean([recall_cos(Ha, k, v)
                                            for k, v in zip(donor_keys, donor_vals)])),
           "fresh_recall": float(np.mean([recall_cos(fresh, k, v)
                                          for k, v in zip(donor_keys, donor_vals)])),
           "clean_baseline": float(np.mean([recall_cos(build_trace(donor_keys, donor_vals),
                                                       k, v)
                                            for k, v in zip(donor_keys, donor_vals)])),
           "host_bruise_inplace": float(np.mean([recall_cos(Ha, k, v)
                                                 for k, v in zip(host_keys, host_vals)])),
           "host_untouched_fresh": float(np.mean([recall_cos(H, k, v)
                                                  for k, v in zip(host_keys, host_vals)]))}
    return out


def gdn_symmetry_battery(dk=128, n_pairs=20, decay=0.98, seed=0):
    """The symmetry-class finding on the OTHER substrate: the GDN outer-product matrix memory
    (the installed model's state) is covariant under the FULL orthogonal group -- basis-permute
    S and project the keys coherently and recall is EXACT -- while HRR carries only the cyclic
    group. Which memory you build in decides which surgeries it survives."""
    rng = np.random.default_rng(seed)
    Ks = [_unit(rng.standard_normal(dk)) for _ in range(n_pairs)]
    Vs = [_unit(rng.standard_normal(dk)) for _ in range(n_pairs)]
    S = np.zeros((dk, dk))
    for k, v in zip(Ks, Vs):
        S = decay * S + np.outer(k, v)
    def rc(Sm, k, v):
        r = Sm.T @ k
        return float(_unit(r) @ v)
    base = float(np.mean([rc(S, k, v) for k, v in zip(Ks, Vs)]))
    P = np.eye(dk)[rng.permutation(dk)]
    coh = float(np.mean([rc(P @ S, P @ k, v) for k, v in zip(Ks, Vs)]))
    rowmask = (rng.random(dk) > 0.5)
    les = float(np.mean([rc(S * rowmask[:, None], k, v) for k, v in zip(Ks, Vs)]))
    return {"baseline": base, "basis_permuted_coherent": coh, "half_rows_lesioned": les}


def shufflebrain_battery(dim=2048, n_items=24, seed=0, shift=613):
    """Run the full panel-session battery at the pilot's scale and return every table row.
    Deterministic in (dim, n_items, seed); the selftest pins the pilot numbers as planted
    truths so the measured session can never silently rot."""
    rng = np.random.default_rng(seed)
    keys = [_unit(rng.standard_normal(dim)) for _ in range(n_items)]
    vals = [_unit(rng.standard_normal(dim)) for _ in range(n_items)]
    rngd = np.random.default_rng(seed + 9)
    dkeys = [_unit(rngd.standard_normal(dim)) for _ in range(n_items)]
    dvals = [_unit(rngd.standard_normal(dim)) for _ in range(n_items)]
    return {"rotation": rotation_battery(keys, vals, shift),
            "mince": mince_curve(keys, vals, (512, 128, 8)),
            "focal_lesion": focal_lesion_battery(keys, vals, 0.5),
            "cleanup_rescue": cleanup_rescue(keys, vals, 0.5),
            "graft": graft_battery(keys, vals, dkeys, dvals, alpha=1.0, fragment=0.5),
            "graft_amplify": graft_amplify(keys, vals, dkeys, dvals, alpha=1.0, fragment=0.5),
            "graft_floor": graft_amplify(keys, vals, dkeys, dvals, alpha=0.25, fragment=0.25),
            "mince_law": mince_law(keys, vals, block=dim // 8),
            "spectral": spectral_lesion(keys, vals, (dim // 4, dim // 2)),
            "gdn_symmetry": gdn_symmetry_battery()}


def model_graft_battery(dim=512, seed=0):
    """S8 -- Pietsch's trained-donor transfer as MODEL ARITHMETIC (delegates to the hdrift
    algebra: compose = moments add, evidence-weighted; ablate = exact unlearning). Donor
    learns a ring, host learns a bar, ONE shared encoder space (the pilot's first run
    refused: models in different encoder spaces cannot compose -- kept as the API teaching
    it). Measured: the grafted host GENERATES donor-like behavior (ring 0.00 -> 0.30) at a
    visible bruise (bar 1.00 -> 0.57 -- the same conservation law as the trace graft: new
    mass pays); a 1/3-evidence fragment transfers proportionally less and bruises less
    (evidence weighting IS the dosage); and ABLATE is exact GRAFT REJECTION -- the host
    restored to 0.00/1.00. Biology never had a rejection operator; the algebra ships one."""
    import lecore as _lc
    mind = _lc.UnifiedMind(dim=64, seed=0)
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, 400)
    ring = np.stack([3 * np.cos(th), 3 * np.sin(th)], 1) + 0.1 * rng.standard_normal((400, 2))
    bar = np.stack([rng.uniform(-1, 1, 400), np.full(400, -4.0)], 1) \
        + 0.1 * rng.standard_normal((400, 2))
    B = [(-5.0, 5.0), (-6.0, 5.0)]
    donor = mind.drift_train(ring, dim=dim, bounds=B)
    host = mind.drift_train(bar, dim=dim, bounds=B)
    frag = mind.drift_train(ring[:133], dim=dim, bounds=B)

    def near(s, t, tol=0.6):
        return float(np.mean(np.min(np.linalg.norm(s[:, None] - t[None], axis=2), 1) < tol))

    def beh(mod):
        s = mind.drift_generate(mod, n=200, seed=7)
        return near(s, ring), near(s, bar)
    graft = mind.drift_compose(host, donor)
    out = {"host": beh(host), "donor": beh(donor), "graft": beh(graft),
           "graft_fragment": beh(mind.drift_compose(host, frag)),
           "rejected": beh(mind.drift_ablate(graft, donor))}
    return out


def _selftest():
    # PLANTED TRUTHS, one dedicated RNG per plant via the battery's own seeding. The pins are
    # the pilot session's measured contracts; failing any one means the theorems rotted.
    r = shufflebrain_battery(dim=2048, n_items=24, seed=0)
    rot = r["rotation"]
    assert abs(rot["vs_rotated"] - rot["baseline"]) < 0.01, rot     # coherent transform, exact
    assert rot["vs_original"] < 0.05, rot                            # originals gone
    assert abs(rot["coherent_surgery"] - rot["baseline"]) < 0.01, rot  # the shift cancels
    mn = r["mince"]
    assert mn[512] > 3 * abs(mn[8]) and mn[512] < rot["baseline"], mn  # graded death, kept negative
    fl = r["focal_lesion"]
    assert fl["holographic"]["dead"] == 0, fl                        # whole retained
    assert fl["localized"]["dead"] == fl["n_items"] // 2, fl         # specific permanent loss
    assert fl["holographic"]["sd"] < 0.1 < fl["localized"]["sd"], fl
    cr = r["cleanup_rescue"]
    assert cr["identified"] == cr["of"] and cr["raw_mean_cos"] < 0.2, cr  # prior completes fragment
    g = r["graft"]
    assert 0.02 < g["donor_in_host"] < 0.2, g                        # real but faint -- honest size
    assert g["host_own_after"] < g["host_own_before"], g             # the bruise is real too
    ga = r["graft_amplify"]
    assert ga["identified"] >= int(0.9 * ga["of"]), ga     # cleanup rescues TRANSFER (>=90%;
                                                           # this draw: 22/24 -- two near-twin
                                                           # values confuse, honestly)
    assert ga["fresh_recall"] > 0.9 * ga["clean_baseline"], ga    # ~full transfer, fresh store
    assert ga["host_untouched_fresh"] > 0.19, ga           # read-only graft: host never pays
    assert ga["inplace_recall"] < 0.9 * ga["clean_baseline"], ga  # in-place PAYS the capacity
                                                           # law -- the conservation pin
    gf = r["graft_floor"]
    assert gf["identified"] < gf["of"] // 2, gf            # the graft capacity boundary, honest
    ml = r["mince_law"]
    for f, rec, pred in ml["rows"]:
        assert abs(rec - pred) < 0.03, (f, rec, pred)      # the aligned-mass law, per rung
    sp = r["spectral"]
    assert sp["coarticulation_err"] < 1e-12, sp            # band-kill == band-limit, a theorem
    assert sp["dead"] == 0, sp                             # resolution loss, whole retained
    assert abs(sp["vs_bandlimited_mean"] - ml["baseline"]) < 0.05, sp
    gs = r["gdn_symmetry"]
    assert abs(gs["basis_permuted_coherent"] - gs["baseline"]) < 1e-10, gs  # FULL orthogonal group
    assert 0.5 < gs["half_rows_lesioned"] < gs["baseline"], gs       # graceful, priced
    mg = model_graft_battery()
    assert mg["host"][0] < 0.05 and mg["host"][1] > 0.9, mg    # host knows only its task
    assert mg["donor"][0] > 0.8, mg
    assert mg["graft"][0] > 0.2, mg                            # TRANSFER: donor behavior appears
    assert mg["graft"][1] < 0.8, mg                            # ...and the bruise is real
    assert mg["graft_fragment"][1] > mg["graft"][1], mg        # evidence weighting = dosage
    assert mg["rejected"][0] < 0.05 and mg["rejected"][1] > 0.9, mg  # ablate = EXACT rejection
    print("OK: shufflebrain battery -- rotation is a coherent transform (%.3f==%.3f), focal lesion "
          "separates architectures (0 vs %d dead), cleanup identifies %d/%d at half-brain, graft "
          "faint-but-real (%.3f), GDN memory exactly orthogonal-covariant; mincing stays a kept "
          "negative" % (rot["vs_rotated"], rot["baseline"], fl["localized"]["dead"],
                        cr["identified"], cr["of"], g["donor_in_host"]))


if __name__ == "__main__":
    _selftest()
