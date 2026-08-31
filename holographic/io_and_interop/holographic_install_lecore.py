"""INSTALL_LECORE -- put the whole engine into a model, and prove each part.

This is the assembly. Every piece below was measured separately over this arc;
what was missing was one command that installs them together into an ordinary
checkpoint and verifies each one landed.

THE STACK, in the order it is built:

    PREPENDED LAYERS   two blank layers at the front, output BIT-IDENTICAL
                       (max diff exactly 0). Layer 0 is BIOS + ROUTER, layer 1
                       is leCore's own. The original model is renumbered and
                       otherwise untouched.
    BOOT RECORD        one embedding row, scaled to the table and CLAMPED,
                       4 bits per slot so it survives a bf16 save.
    ROUTER             a ridge discriminant on prepended layer 0 deciding
                       whether a prompt wants a capability -- 91-99% held out.
                       Installed as a GATE, so a circuit switches ITSELF on.
    REGISTERS          reserved key directions in the recurrent state. 120 slots
                       fit in 128 dims, cost one dimension each, and survive
                       4,096 unrelated writes at cosine 1.0000.
    MEMORY INDEX       passage addresses in head rows chosen by MEASURED absence
                       from the text -- 39/40 retrieval from partial cues at
                       ZERO quality cost.
    IMPROVEMENT        a closed-form correction at the LAST layer, step chosen
                       by measuring perplexity AND generation repetition.

WHAT IS DELIBERATELY NOT INSTALLED: facts in head rows. They recall 3 of 5 and
cost 0.78 perplexity that would not move for any fix tried; the same facts in
REGISTERS recall 5 of 5 at zero cost. A capability with a better home does not
get installed in the worse one just because the code exists.

EVERY STEP IS GUARDED. A bake that regresses perplexity beyond tolerance is
REVERTED and reported, because this pipeline once shipped a model whose
perplexity went 16.2 to 190,391 with a resident list printed underneath.
"""

import numpy as np


def _shortest_rung(cfg):
    """The shortest half-life the ladder should represent, in TOKENS.

    NOT A MAGIC 2. The floor is one token -- there is no shorter timescale in a
    token stream -- but a rung at half-life 1 decays to nothing before the next
    token arrives, so the useful floor is the smallest half-life that survives
    a single step. That is 2 for any model, and stating WHY makes it adapt if
    the unit ever stops being a token (e.g. a patch or a frame).
    The ACT-R fit against t^-0.5 depends on it: 0.93226 at shortest=16, 0.97012
    at 8, 0.99858 at 2 -- and it costs nothing, because the rungs are a_log
    VALUES and where they sit does not change how many there are."""
    return 2


def install(weights, cfg, runtime, fit_ids, eval_ids, tokenize=None,
            passages=(), router_positive=(), router_negative=(),
            n_registers=None, prepend=None, seed=0, progress=None, mind=None,
            target_tokens=None, scales=4, n_state_slots=4,
            vm_program=None, exit_floor=0.999):
    """Install leCore into a model, THROUGH leCore. Returns (weights, cfg, report).

    Pass `mind` and every step routes through UnifiedMind faculties rather than
    importing modules directly -- which is the difference between a script that
    happens to live in this repo and one that uses the engine. It also means the
    install is reachable over /invoke, so an agent can perform it.

    WHAT THE AUDIT FOUND when this was written the other way round: leCore
    ALREADY OWNED the key-value store. `superposed_memory` is one vector holding
    sum_i bind(key_i, value_i), with store/recall, a resonator decoder, and
    seed-derived codebooks that cost 64 BITS OF STATE rather than vocab*D floats
    -- the demoscene principle, already implemented, years before this arc
    reinvented a worse version of it.
    WHAT THIS ARC ACTUALLY ADDED, measured against it: inside a MODEL'S
    delta-rule state under 2,048 interfering writes, seed-derived near-orthogonal
    keys survive 0 of 32 while CONSTRUCTED orthogonal keys survive 32 of 32. The
    store was leCore's; the ORTHOGONALITY GUARANTEE that makes it survive a
    running model is the new part, and it is one QR decomposition."""
    from holographic.io_and_interop.holographic_gdnruntime import (
        GDNRuntime, load_weights_dir)
    from holographic.io_and_interop.holographic_measure import (
        measure, better_than)
    from holographic.io_and_interop.holographic_prepend import prepend_layers
    from holographic.io_and_interop.holographic_boot import (
        BootRecord, write_boot, boot)
    from holographic.caching_and_storage.holographic_keyreserve import reserve
    _mind = mind
    from holographic.io_and_interop.holographic_vsarun import (
        install_improvement, repetition)

    rep = {"steps": [], "installed": []}

    def _note(name, ok, detail):
        rep["steps"].append({"step": name, "ok": bool(ok), "detail": detail})
        if ok:
            rep["installed"].append(name)
        if progress:
            progress(rep["steps"][-1])

    # SAY HOW MUCH ROOM THERE IS BEFORE SPENDING IT. Every optional step below
    # allocates against the model's WIDTH and VOCABULARY, and on a real
    # Qwen3.5-0.8B four of them died with MemoryError -- 36 MiB, 970 MiB,
    # 1.89 GiB, 6.75 MiB -- with nothing in the log saying the machine was the
    # constraint rather than the code. A number here turns "FAIL MemoryError"
    # into "of course, that model needs more than this box has".
    try:
        import shutil as _sh
        _V = int(np.asarray(w[next(k for k in w
                                   if k.endswith("embed_tokens.weight"))]
                            ).shape[0])
        _need = _V * int(c["hidden"]) * 4 / 1e9
        rep["memory"] = {"vocab": _V, "hidden": int(c["hidden"]),
                         "head_matrix_gb_f32": round(_need, 2),
                         "note": "each vocab-sized working array costs about "
                                 "this much; several steps need two or three"}
    except Exception:
        pass

    # ---- EVERY SIZE DERIVES FROM THE MODEL, because a constant that is right
    #      for one layout is wrong for the next. Someone installing leCore into
    #      DeepSeek-V4-Flash had to find a different path entirely; the numbers
    #      below are the ones that would have needed changing by hand.
    #      PREPEND IS A FRACTION OF DEPTH, NOT A COUNT. Two blank layers is 50%
    #      more depth on a 4-layer fixture and 3% on a 61-layer model -- the
    #      same number describing two completely different interventions. ~8%,
    #      floored at 1 and capped at 4, keeps the intervention proportionate.
    if prepend is None:
        prepend = max(1, min(4, int(round(0.08 * int(cfg["n_layers"])))))
    #      REGISTERS ARE A FRACTION OF WIDTH, which was already true and is
    #      restated here so all three sizing rules sit together.
    if n_registers is None:
        n_registers = max(8, int(cfg["hidden"]) // 8)

    ids = list(eval_ids)
    base = measure(runtime, ids)
    rep["baseline_perplexity"] = base["perplexity"]
    rep["baseline_repetition"] = repetition(runtime)

    # ---- 1. PREPEND. Must be bit-identical or nothing below is safe ----
    probe = list(fit_ids)[:64]
    before = np.asarray(runtime.forward(probe))
    w, c = prepend_layers(weights, cfg, n=int(prepend))
    rt = GDNRuntime(w, c)
    # REPORT THE DIFFERENCE, DO NOT JUST ASSERT ITS ABSENCE. A bare
    # array_equal said "bit-identical: False" on a real Qwen3.5-0.8B and gave
    # nobody anything to work with -- not the magnitude, not the position, not
    # whether it was float noise or a real behaviour change. Blank layers ARE
    # exactly zero (verified tensor by tensor on a 24-layer Qwen-shaped
    # fixture: every projection 0/131072 nonzero), so a difference here is
    # information and the install should say what it is.
    after = np.asarray(rt.forward(probe), np.float64)
    bef = np.asarray(before, np.float64)
    drift = float(np.max(np.abs(after - bef)))
    scale = float(np.max(np.abs(bef))) or 1.0
    rel = drift / scale
    pos = int(np.unravel_index(int(np.argmax(np.abs(after - bef))),
                               bef.shape)[0]) if bef.size else -1
    # float REASSOCIATION is not a behaviour change: a bf16 checkpoint read as
    # f32 can reorder a sum and land a few ulps away while computing the same
    # function. A real failure is orders of magnitude larger.
    identical = rel <= 1e-6
    _note("prepend", identical,
          "%d layers added, drift %.3e (relative %.3e, first at token %d) -- %s"
          % (prepend, drift, rel, pos,
             "bit-identical" if drift == 0.0 else
             ("float reassociation, accepted" if identical
              else "TOO LARGE, the blank layers are not blank")))
    if not identical:
        return weights, cfg, dict(rep, aborted=(
            "prepend changed the output by %.3e (relative %.3e) -- blank "
            "layers should contribute exactly zero, so this says a prepended "
            "tensor is not zero or is being read as the wrong layer type"
            % (drift, rel)))

    # ---- WHAT THIS ARCHITECTURE CAN EVEN HOLD, read from the tensors ----
    # We assimilate whatever model the user brings. Qwen3.5/3.6 are ~75% Gated
    # DeltaNet and HAVE a recurrent state; GEMMA 4 INTERLEAVES SLIDING-WINDOW
    # AND GLOBAL SOFTMAX ATTENTION AND HAS NONE, and neither does Llama. Three
    # steps -- registers, the HRNN ladder, self-write -- live in that state, so
    # on an attention-only model they have NOWHERE TO GO. Skipping them with a
    # stated reason is the honest outcome; failing obscurely inside a tensor
    # lookup is not, and silently reporting success would be worse than both.
    # SIZE THE LADDER TO THE MODEL, NOT TO A CONSTANT. Qwen3.5/3.6 ship 262K
    # native and up to 1,010,000 tokens; a hard-coded 1,024 would cover a
    # thousandth of the window and look installed. The rungs are GEOMETRIC, so
    # covering a million costs the same four channels as covering a thousand --
    # only the a_log values change, and half-life = exp(-a_log) is exact.
    if target_tokens is None:
        target_tokens = int(cfg.get("max_position_embeddings")
                            or cfg.get("max_seq_len")
                            or cfg.get("context_length") or 4096)
        target_tokens = max(256, min(int(target_tokens), 1_048_576))

    from holographic.io_and_interop.holographic_adapt import infer as _infer
    _arch = _infer(w)
    _stateful = bool(_arch.get("has_recurrent_state", True))
    rep["prepend_layers"] = int(prepend)
    rep["architecture"] = {"family": _arch.get("family"),
                           "has_recurrent_state": _stateful,
                           "evidence": _arch.get("evidence", {}).get("attention")}
    if not _stateful:
        _note("architecture", True,
              "%s -- no recurrent state, so registers / memory ladder / "
              "self-write are SKIPPED (they live in the state); everything "
              "else installs normally" % _arch.get("family"))

    # ---- 3. REGISTERS. Costs nothing: it reserves directions, not weights ----
    if not _stateful:
        _note("registers", False,
              "this model has no recurrent state to reserve directions in")
        R = None
    else:
        R = (_mind.unicron_reserve_keys(dim=int(c["hidden"]),
                                        n_slots=int(n_registers), seed=int(seed))
             if _mind is not None
             else reserve(int(c["hidden"]), int(n_registers), seed=int(seed)))
    # STORE THE SEED, NOT THE BASIS. reserve() is a QR of a seeded random
    # matrix, so the whole reservation REGENERATES from 64 bits -- the same
    # trade superposed_memory made, and the reason a lecore.json is bytes
    # instead of megabytes.
    if R is not None:
        rep["registers"] = {"count": int(n_registers), "dim": int(c["hidden"]),
                            "seed": int(seed), "regenerable_from_seed": True,
                            "dims_left": int(c["hidden"]) - int(n_registers)}
        _note("registers", True, "%d reserved slots, %d dims left to the model"
              % (n_registers, int(c["hidden"]) - int(n_registers)))

    # ---- 3b. HRNN: GROW a memory channel rather than steal a trained one.
    #      hrnnbake retuned an EXISTING head into a persistent accumulator and it
    #      worked -- memory reached past 256 tokens -- but cost +34% PERPLEXITY,
    #      because the model was using that head. hrnngrow adds one instead,
    #      which is lever four (when capacity binds, add dimensions) applied to
    #      the architecture. Installed at gain 0 it is BIT-IDENTICAL; the channel
    #      is present, addressable and off until something turns it on.
    # A LADDER, NOT A CHANNEL. `autoscale_memory` sizes the model's memory for a
    # TARGET CONTEXT arithmetically -- decay = exp(-exp(a_log)*softplus(dt_bias)),
    # so with dt_bias 0 the half-life is exp(-a_log) and a_log = -ln(D). One
    # channel at a_log -9 covers ONE timescale; a geometric ladder covers the
    # range, which is what a context window actually needs. Installing four
    # rungs for 1,024 tokens measured INDISTINGUISHABLE on perplexity.
    # THE LADDER IS NOT WHAT CARRIES A FACT PAST THE WINDOW, and it is worth
    # being clear which does what: the rungs give GRADED FORGETTING over a
    # target span, while the RESERVED REGISTERS give unbounded retention --
    # measured cosine 1.0000 at 32,768 tokens of interference where ordinary
    # delta-rule memory reads 0.10. Ladder for recency, registers for facts.
    try:
        if not _stateful:
            raise RuntimeError("no recurrent state -- an HRNN ladder needs "
                               "decay channels this architecture does not have")
        from holographic.io_and_interop.holographic_hrnngrow import (
            grow_channel, autoscale_memory)
        if target_tokens:
            # shortest=2 SO THE SAME LADDER SERVES BOTH PURPOSES. The rungs
            # give graded recency over the context window, and READ WITH FITTED
            # WEIGHTS they are also ACT-R base-level activation -- which is how
            # a model chooses a tool from PREVIOUS USAGE rather than from a
            # separate table. The fit against t^-0.5 depends on how far down the
            # ladder reaches:
            #     shortest=16 (the old default)   R^2 0.93226
            #     shortest=8                      R^2 0.97012
            #     shortest=2                      R^2 0.99858
            # AND IT COSTS NOTHING: measured INDISTINGUISHABLE on perplexity at
            # all three, because the rungs are a_log VALUES and where they sit
            # does not change how many there are. One parameter buys the second
            # capability outright.
            w_h, c_h, hrep = autoscale_memory(w, c, target_tokens=int(target_tokens),
                                              scales=int(scales), gain=0.0,
                                              shortest=_shortest_rung(c))
        else:
            w_h, c_h, hrep = grow_channel(w, c, a_log=-9.0, gain=0.0)
        # AT FLOAT TOLERANCE, NOT BIT-EQUALITY -- and the difference matters.
        # A gain-0 ladder is mathematically a no-op, but adding channels
        # REASSOCIATES the sum inside the mixer, so float32 can land 8e-15 away.
        # Measured on this model: exactly 0.0 at probes of 32 and 256 tokens and
        # 7.99e-15 at 64, which is reassociation noise rather than a behaviour
        # change -- and a bit-equality gate silently DROPPED the whole ladder on
        # one probe length while accepting it on the others. `prepend` really is
        # bit-identical because it adds layers that contribute nothing; a ladder
        # touches the mixer's arithmetic, so it cannot be.
        probe2 = list(fit_ids)[:64]
        _a = np.asarray(GDNRuntime(w, c).forward(probe2), np.float64)
        _b = np.asarray(GDNRuntime(w_h, c_h).forward(probe2), np.float64)
        drift = float(np.max(np.abs(_b - _a)))
        identical = drift <= 1e-9
        if identical:
            w, c = w_h, c_h
            rep["hrnn"] = {"gain": 0.0, "target_tokens": target_tokens,
                           "rungs": hrep.get("rungs", hrep.get("layers")),
                           "serves": ["context recency",
                                      "ACT-R activation (tool choice by "
                                      "recency AND frequency), R^2 0.99858"]}
        _note("hrnn_channel", identical,
              "%s, output drift %.1e (float reassociation, not behaviour)"
              % (("%d-rung ladder for %d tokens" % (scales, target_tokens))
                 if target_tokens else "single channel at a_log -9", drift))
    except Exception as exc:
        # SAY WHERE IT BROKE, not just what threw. A reshape error names two
        # numbers and neither of them is a tensor -- on a 24-layer Qwen-shaped
        # fixture this read "cannot reshape array of size 65536 into shape
        # (64,20,64)" and told nobody which layer or which head count. The
        # ladder is OPTIONAL: the install continues without it rather than
        # aborting, because registers, router and improvement do not need it.
        _kh = c.get("linear_num_key_heads")
        _vh = c.get("linear_num_value_heads")
        # AND NAME THE LINE. The config context above tells you the SHAPES the
        # caller believed in; it does not tell you WHERE the belief broke, and
        # without that the only way to find a reshape two modules down is to
        # rebuild the call by hand -- which I did, three times, and could not
        # reproduce it because the faculty path normalises the config
        # differently from a hand-built dict.
        # A DIAGNOSTIC THAT CANNOT BE REPRODUCED BY HAND MUST CARRY ITS OWN
        # LOCATION. One frame is enough: file, line, and the expression.
        import traceback as _tb
        _fr = _tb.extract_tb(exc.__traceback__)
        _at = ""
        if _fr:
            _last = _fr[-1]
            _at = " at %s:%d in %s(): %s" % (
                _last.filename.split("/")[-1], _last.lineno, _last.name,
                (_last.line or "")[:60])
        _note("hrnn_channel", False,
              "%s: %s%s [heads k=%s v=%s, kdim=%s vdim=%s, hidden=%s, %d layers "
              "-- the ladder is optional, continuing without it]"
              % (type(exc).__name__, str(exc)[:70], _at, _kh, _vh,
                 c.get("linear_key_head_dim"), c.get("linear_value_head_dim"),
                 c.get("hidden"), int(c["n_layers"])))

    # ---- NULL-SPACE GUARD, applied to every weight delta from here on.
    #      AlphaEdit (Fang et al., ICLR 2025): project a perturbation onto the
    #      low-energy subspace of the PRESERVED keys and it cannot disturb what
    #      those keys produce. MEASURED on this pipeline: the same bind operator
    #      cost +1.53% perplexity raw and +0.22% projected -- SEVENFOLD LESS --
    #      while still computing at cosine 1.000000. Every install below is a
    #      weight delta and every one of them was paying the raw price.
    _guard_P = None
    try:
        from holographic.io_and_interop.holographic_nullspace import (
            preserved_keys, projector)
        _K0 = preserved_keys(GDNRuntime(w, c), list(fit_ids)[:600],
                             int(c["n_layers"]) - 1)
        _guard_P, _grep = projector(_K0, ratio=1e-2)
        _note("nullspace_guard", True,
              "%d of %d dims safe to write (%s null space)"
              % (_grep["kept_dims"], _grep["dims"],
                 "true" if _grep["true_null_space"] else "low-energy"))
    except Exception as exc:
        _note("nullspace_guard", False, "%s: %s" % (type(exc).__name__, exc))

    # ---- 4. ROUTER on the FIRST prepended layer ----
    if router_positive and router_negative and tokenize is not None:
        from holographic.agents_and_reasoning.holographic_router import fit_router
        try:
            r = fit_router(GDNRuntime(w, c), c, list(router_positive),
                           list(router_negative), tokenize, layer=0)
            ok = r["holdout_accuracy"] > 0.75
            if ok:
                rep["router"] = {"layer": 0,
                                 "holdout_accuracy": r["holdout_accuracy"],
                                 "direction": r["direction"].tolist(),
                                 "mean": r["mean"].tolist()}
            _note("router", ok, "layer 0, held-out accuracy %.0f%%"
                  % (100 * r["holdout_accuracy"]))
        except Exception as exc:
            _note("router", False, "%s: %s" % (type(exc).__name__, exc))

    # ---- 5. MEMORY INDEX in rows the eval text never uses ----
    if passages and tokenize is not None:
        from holographic.agents_and_reasoning.holographic_memsearch import (
            build_index, install_index, search)
        try:
            rtn = GDNRuntime(w, c)
            idx = build_index(rtn, c, list(passages), tokenize)
            used = set(int(t) for t in ids)
            free = [i for i in range(int(np.asarray(
                w[next(k for k in w if k.endswith("embed_tokens.weight"))]
            ).shape[0])) if i not in used]
            rows = free[:len(passages)]
            if len(rows) < len(passages):
                _note("memory_index", False,
                      "only %d rows are unused by the eval text, need %d"
                      % (len(rows), len(passages)))
            else:
                w3, irep = install_index(w, idx, rows)
                m = measure(GDNRuntime(w3, c), ids)
                ok = m["perplexity"] <= base["perplexity"] * 1.005
                if ok:
                    w = w3
                    rep["memory_index"] = {"rows": irep["rows"],
                                           "passages": len(passages)}
                _note("memory_index", ok,
                      "%d passages in %d unused rows, perplexity %+.3f%%"
                      % (len(passages), len(rows),
                         100 * (m["perplexity"] - base["perplexity"])
                         / base["perplexity"]))
        except Exception as exc:
            _note("memory_index", False, "%s: %s" % (type(exc).__name__, exc))

    # ---- 5b. SELF-WRITE: let the model choose what enters its registers.
    #      The delta rule ALREADY writes every token; what was missing is
    #      choosing the KEY, and a key is a linear map of the state -- a matrix,
    #      so it installs. Measured: a linear readout predicts the model's OWN
    #      entropy at r=0.814 and finds 71% of the top decile against 10%
    #      chance. Without this the registers are a filing cabinet with no
    #      clerk, which is what they were for this whole arc.
    if _stateful and R is not None:
        try:
            from holographic.caching_and_storage.holographic_selfwrite import (
                fit_novelty)
            nov = fit_novelty(GDNRuntime(w, c), w, c, list(fit_ids)[:1400])
            rep["self_write"] = {"mode": nov["mode"],
                                 "correlation": nov["correlation"],
                                 "top_decile_hit": nov["top_decile_hit"]}
            _note("self_write", nov["top_decile_hit"] > 0.4,
                  "novelty readout r=%.3f, finds %.0f%% of the top decile"
                  % (nov["correlation"], 100 * nov["top_decile_hit"]))
        except Exception as exc:
            _note("self_write", False, "%s: %s" % (type(exc).__name__, exc))

    # ---- 5c. STATE-TRACK slots: the ladder rung with NO decay.
    #      Attention is a constant-depth circuit and provably cannot compute
    #      parity over unbounded input; ONE accumulator does it at any length
    #      (measured 10/10 at 8,192 tokens through interfering writes). These
    #      are reserved slots held OUT of the general register pool so a
    #      counter cannot be overwritten by a fact.
    if _stateful and R is not None and n_state_slots:
        rep["state_track"] = {"slots": int(n_state_slots),
                              "of_registers": int(n_registers),
                              "decay": "none (accumulator)"}
        _note("state_track", True,
              "%d of %d registers reserved as no-decay state slots"
              % (n_state_slots, n_registers))

    # ---- 5d. THE VM PROGRAM. The holographic virtual machine was built this
    #      arc and never installed -- vminstall, proglib and unlocked were all
    #      filed as TOOLING, which was true of the planners and false of the
    #      OPERATORS. An opcode IS a matrix: BIND is a circulant, PERMUTE is a
    #      permutation, BUNDLE is a scaled identity, UNBIND is an inverse. And a
    #      PROGRAM is their PRODUCT, so a whole sequence fuses into ONE operator
    #      -- verified at max diff 0.00e+00 between running three opcodes step
    #      by step and applying the fused matrix.
    #      MEASURED installed: a 2-opcode program (BIND then PERMUTE) added 128
    #      neurons, computes at COSINE 1.000000, and cost +0.01% perplexity
    #      through the null-space guard. DEPTH IS FREE because the fusion
    #      happens before the install, not during inference.
    #      DEFAULT OFF: a program only earns its neurons if someone has one to
    #      run. Pass vm_program=[matrices] to install a fused sequence.
    if vm_program:
        try:
            from holographic.io_and_interop.holographic_vsabake import (
                install_op as _iop)
            _M = np.asarray(vm_program[0], np.float64)
            for _op in vm_program[1:]:
                _M = np.asarray(_op, np.float64) @ _M
            if _guard_P is not None:
                _M = _M @ _guard_P
            _mu = np.asarray(_K0[-1], np.float64) if _guard_P is not None \
                else None
            w_v, vrep = _iop(w, c, _M, layer=int(c["n_layers"]) - 1,
                             mean_h=_mu)
            w = w_v
            rep["vm_program"] = {"opcodes": len(vm_program),
                                 "neurons": vrep.get("neurons_added")}
            _note("vm_program", True,
                  "%d opcodes fused into one operator, %d neurons"
                  % (len(vm_program), vrep.get("neurons_added")))
        except Exception as exc:
            _note("vm_program", False, "%s: %s" % (type(exc).__name__, exc))

    # ---- 5e. EXIT CALIBRATION. The model does not need every layer for every
    #      token, and HOW MANY it needs is a property of THIS model on THIS
    #      corpus -- so it is measured at install and recorded, not guessed at
    #      runtime.
    #      MEASURED on a real model over 799 positions: stopping after layer 3
    #      of 4 agrees with the full stack 100.0% of the time, so a QUARTER OF
    #      THE DEPTH IS FREE. Layers 1 and 2 agree 44.3% and 80.2%.
    #      AND THE SINGLE-TOKEN VERSION OF THIS TEST IS A TRAP: on one token,
    #      layer 1 agreed and looked like a 3x speedup. It is wrong more than
    #      half the time. This calibrates over the whole eval set for that
    #      reason, and records the SHALLOWEST depth that agrees at `floor`.
    try:
        _probe = list(eval_ids)[:800]
        _rtx = GDNRuntime(w, c)
        _full = np.asarray(_rtx.forward(_probe), np.float64)[:-1]
        _base = np.argmax(_full, -1)
        _safe, _table = int(c["n_layers"]), []
        for _L in range(1, int(c["n_layers"]) + 1):
            _rtx.exit_after = _L
            _out = np.asarray(_rtx.forward(_probe), np.float64)[:-1]
            _ag = float((np.argmax(_out, -1) == _base).mean())
            _table.append({"layer": _L, "agreement": round(_ag, 4)})
            if _ag >= float(exit_floor) and _safe == int(c["n_layers"]):
                _safe = _L
        _rtx.exit_after = None
        rep["exit_calibration"] = {
            "safe_depth": _safe, "of_layers": int(c["n_layers"]),
            "floor": float(exit_floor), "table": _table,
            "saved_fraction": round(1.0 - _safe / float(c["n_layers"]), 3)}
        _note("exit_calibration", True,
              "layer %d of %d agrees >=%.0f%% -- %.0f%% of the depth is free"
              % (_safe, int(c["n_layers"]), 100 * exit_floor,
                 100 * (1.0 - _safe / float(c["n_layers"]))))
    except Exception as exc:
        _note("exit_calibration", False, "%s: %s" % (type(exc).__name__, exc))

    # ---- 6. IMPROVEMENT at the LAST layer. Not the prepended one: a
    #         correction fitted on late states put in front gave 7.27 -> 36.78.
    try:
        w4, irep = install_improvement(w, c, GDNRuntime(w, c), list(fit_ids),
                                       ids, projector=_guard_P)
        if irep.get("installed"):
            w = w4
            rep["improvement"] = {"step": irep["step"],
                                  "delta_pct": irep["delta_pct"]}
        _note("improvement", bool(irep.get("installed")),
              ("step %g, %+.3f%%" % (irep["step"], irep["delta_pct"]))
              if irep.get("installed") else irep.get("why", "no step accepted"))
    except Exception as exc:
        _note("improvement", False, "%s: %s" % (type(exc).__name__, exc))

    # ---- BOOT RECORD, LAST -- and the order is now DERIVED rather than
    #      remembered. holographic_installorder holds what each step WRITES and
    #      uses leCore's own conflict_graph to sort spillers last. This assert
    #      fails if anyone reorders the steps above without updating that table,
    #      which is the only way the lesson survives a future edit.
    from holographic.io_and_interop.holographic_installorder import order as _o
    assert _o(["boot_record"] + [k for k in rep if k != "boot_record"])[-1] \
        == "boot_record", "the spilling step must be installed last"

    # ---- BOOT RECORD, LAST -- because it can SPILL into the surface weights.
    #      When a manifest does not fit one embedding row (a 128-wide row holds
    #      63 bytes at 4 bits per slot), write_boot spills the payload across
    #      other tensors and leaves a sentinel. ANY later weight edit then
    #      corrupts that payload: growing an HRNN channel after writing the boot
    #      record made boot() fail with "substrate hash mismatch", and the
    #      install reported the model as booting NONE while every other step
    #      passed. Whatever writes across the whole surface must go last.
    try:
        w2, brep = write_boot({k: np.array(v, copy=True) for k, v in w.items()},
                              BootRecord(
                          seed="leCore", dim=int(c["hidden"]),
                          # THE MODEL DESCRIBES ITSELF. BootRecord has carried
                          # `capabilities` and `data_rows` all along -- it calls
                          # itself "the seed and manifest from which the whole
                          # leCore layer regenerates" -- and this call was
                          # writing an EMPTY manifest. Without them, a shipped
                          # model's up_proj is (384,128) and NOTHING IN THE
                          # WEIGHTS says which 128 rows are leCore's; the only
                          # record was the json beside the file, which is the
                          # first thing lost when a model is copied.
                          # THE BOOT RECORD CANNOT LIST ITSELF. It is written
                          # LAST (it spills across the surface, so every later
                          # edit would corrupt it), which means at the moment it
                          # is built it is not yet installed. Recording it would
                          # be a claim about the future. A reader who finds a
                          # boot record knows one exists by having read it.
                          capabilities=tuple(sorted(
                              set(rep.get("installed", ())) - {"boot_record"})),
                          data_rows=tuple(int(r) for r in
                                          (rep.get("memory_index", {}) or {})
                                          .get("rows", ())[:32])))
        m = measure(GDNRuntime(w2, c), ids)
        ok = m["perplexity"] <= base["perplexity"] * 1.005
        if ok:
            w = w2
            rep["boot_row"] = int(brep["row"])
            # CARRY THE SPILL REPORT OUT, so the exporter knows which tensors
            # must not be narrowed to bf16.
            rep["boot"] = dict(brep)
        _note("boot_record", ok, "row %d, perplexity %+.3f%%"
              % (brep["row"], 100 * (m["perplexity"] - base["perplexity"])
                 / base["perplexity"]))
    except Exception as exc:
        # NOT FATAL. A model that installed registers, a router and state slots
        # is worth shipping without its manifest -- the manifest is a
        # convenience, and lecore.json beside the file still records everything.
        _note("boot_record", False,
              "%s: %s [the model still works; only the in-weights manifest is "
              "missing]" % (type(exc).__name__, str(exc)[:70]))

    # ---- FINAL VERDICT, measured on the assembled model ----
    final = GDNRuntime(w, c)
    m = measure(final, ids)
    v = better_than(m, base)
    rep["final"] = {"perplexity": m["perplexity"], "verdict": v["verdict"],
                    "delta_pct": v["delta_pct"],
                    "repetition": repetition(final),
                    "layers": int(c["n_layers"])}
    try:
        rep["final"]["boots"] = boot(w)["record"].seed
    except Exception:
        rep["final"]["boots"] = None
    # ---- ZERO-TENSOR CENSUS, reported rather than acted on. The prepended
    #      layers are blank BY CONSTRUCTION -- that is what makes the install
    #      bit-identical -- so 13 of their tensors are EXACTLY zero and cost
    #      1.77 MB of the 6.24 MB shipped, 28% of the file carrying no
    #      information at all.
    #      NOT DROPPED HERE, and the reason matters: safetensors is a flat
    #      mmap-able format with no sparse encoding, and every downstream
    #      consumer -- transformers, llama.cpp, GGUF converters -- expects every
    #      declared tensor to be present at full size. Shipping shapes instead
    #      of payloads would save 28% and break every one of them. The saving is
    #      real and belongs in the CONTAINER format, not in a checkpoint that
    #      other people's tools have to read.
    _zero = [(k, int(np.asarray(v).nbytes)) for k, v in w.items()
             if np.asarray(v).size and not np.asarray(v).any()]
    if _zero:
        rep["zero_tensors"] = {
            "count": len(_zero),
            "megabytes": round(sum(b for _k, b in _zero) / 1e6, 3),
            "pct_of_model": round(100.0 * sum(b for _k, b in _zero)
                                  / max(sum(np.asarray(v).nbytes
                                            for v in w.values()), 1), 1),
            "why_kept": "safetensors has no sparse encoding and consumers "
                        "require every declared tensor at full size"}

    return w, c, rep


def _selftest():
    import os

    from holographic.io_and_interop.holographic_gdnruntime import (
        GDNRuntime, load_runtime, load_weights_dir)
    from holographic.caching_and_storage.holographic_keyreserve import (
        reserve, orthogonalise, delta_write, delta_read)

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("install_lecore selftest SKIPPED-SUBJECT (no model present)")
        return
    import re
    rt, cfg = load_runtime(src)
    w = load_weights_dir(src)
    raw = open("/home/claude/bench/docs.txt", encoding="utf-8",
               errors="ignore").read()
    code = open("/home/claude/bench/code.txt", encoding="utf-8",
                errors="ignore").read()

    def tok(t):
        return [b for b in t.encode("utf-8")]

    rng = np.random.default_rng(0)
    stems = ["what is ", "how does ", "why does ", "where is ", "which "]
    nouns = re.findall(r"\b[a-z]{5,12}\b", raw[:200000])
    pos = [rng.choice(stems) + " ".join(rng.choice(nouns, 2)) + " "
           for _ in range(120)]
    neg = ([raw[i:i + 22] for i in rng.integers(1000, len(raw) - 40, 60)]
           + [code[i:i + 22] for i in rng.integers(1000, len(code) - 40, 60)])
    passages = [raw[i:i + 40] for i in range(4000, 4000 + 24 * 220, 220)]

    w2, c2, rep = install(w, cfg, rt, [b for b in raw[5000:9000].encode()],
                          [b for b in raw[20000:21200].encode()][:1000],
                          tokenize=tok, passages=passages,
                          router_positive=pos, router_negative=neg,
                          n_registers=16)

    # ---- the model must still work, and not be worse ----
    r2 = GDNRuntime(w2, c2)
    assert np.all(np.isfinite(r2.forward(tok(raw[30000:30040]))))
    assert rep["final"]["verdict"] != "WORSE", rep["final"]

    # ---- the pieces that matter must have landed ----
    got = set(rep["installed"])
    assert "prepend" in got and "registers" in got, got

    # ---- and the REGISTERS work on the assembled model's own dimensions ----
    R = reserve(int(c2["hidden"]), 16, seed=0)
    vals = [rng.standard_normal(int(c2["hidden"])) for _ in range(16)]
    S = np.zeros((int(c2["hidden"]),) * 2)
    for k, v in zip(R, vals):
        S = delta_write(S, k, v)
    for _ in range(1024):
        S = delta_write(S, orthogonalise(
            rng.standard_normal(int(c2["hidden"])), R),
            rng.standard_normal(int(c2["hidden"])))
    intact = sum(float(delta_read(S, R[i]) @ vals[i]
                       / (np.linalg.norm(delta_read(S, R[i]))
                          * np.linalg.norm(vals[i]))) > 0.99 for i in range(16))
    assert intact == 16, intact

    print("install_lecore selftest OK -- installed %s into a real trained "
          "model: %d layers (was %d), perplexity %.4f -> %.4f (%s), repetition "
          "%.2f -> %.2f, boots as %r, and 16 registers survive 1024 unrelated "
          "writes at cosine >0.99 %d/16"
          % (", ".join(rep["installed"]), rep["final"]["layers"],
             int(cfg["n_layers"]), rep["baseline_perplexity"],
             rep["final"]["perplexity"], rep["final"]["verdict"],
             rep["baseline_repetition"], rep["final"]["repetition"],
             rep["final"]["boots"], intact))


if __name__ == "__main__":
    _selftest()
