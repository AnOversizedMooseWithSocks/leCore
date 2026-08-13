"""Part 16 of UnifiedMind's faculty surface -- UNICRON: consume and read trained models.

NOT A STANDALONE MODULE. One slice of the single `UnifiedMind` class, assembled by
holographic/misc/holographic_unified.py, which remains the only import path anyone uses.

WHY THIS PART EXISTS
--------------------
Rule-0 audit on record: 'read model weights', 'inspect an LLM checkpoint', 'safetensors',
'compare two trained models' all returned fallbacks -- the license to build. The engine can
now DEVOUR foreign trained models (safetensors/npz, stdlib+NumPy parse, torch pickle refused
by contract) and read the weights the informative way: random-matrix theory per layer
(Marchenko-Pastur outliers = learned signal, heavy-tail alpha a la Martin & Mahoney), then a
holographic FINGERPRINT -- one hypervector per model, bind(layer role, metric encoding)
bundled over layers -- so whole models become points in FHRR space with cosine similarity
and +/- model algebra. The distillation audit (teacher vs student spectral drift) is the
capability Moose's friends' normal tooling does not have.

Every method DELEGATES to holographic_unicron; none reimplements.
"""

import numpy as np
from holographic.unified import check_part


class _UnifiedPart16:

    # ------------------------------------------------------------------ UNICRON: devour + read

    def unicron_load(self, path):
        """LOAD a trained model's weights ({name: array}) from .safetensors, .gguf, or .npz with
        stdlib+NumPy only -- bf16 decoded losslessly, torch pickle files REFUSED by contract
        (unpickling is an arbitrary-code-execution surface). .gguf (llama.cpp) supported:
        F32/F16/BF16 direct, Q8_0 dequantized, other quants refused by name.
        See holographic_unicron.load_model."""
        from holographic.io_and_interop import holographic_unicron as _u
        return _u.load_model(path)

    def unicron_analyze(self, model, min_dim=8, spacing=False):
        """READ a model's weights informatively: per-layer random-matrix report (Marchenko-Pastur
        edge, outlier count = learned low-rank signal, Hill tail alpha, stable rank; spacing=True
        adds the quantumstats spacing-ratio regime) plus model-level medians. `model` is a path
        or a {name: array} dict. See holographic_unicron.analyze_model."""
        from holographic.io_and_interop import holographic_unicron as _u
        if isinstance(model, str):
            model = _u.load_model(model)
        return _u.analyze_model(model, min_dim=min_dim, spacing=spacing)

    def unicron_fingerprint(self, model, dim=1024):
        """ONE HYPERVECTOR for a whole model: bundle over layers of bind(role(layer name),
        encoding(spectral metrics)); layer roles are hashlib-seeded so fingerprints are stable
        across processes. Accepts a path, a weights dict, or an unicron_analyze result. Compare
        with unicron_compare or plain cosine; +/- model algebra applies.
        See holographic_unicron.fingerprint."""
        from holographic.io_and_interop import holographic_unicron as _u
        if isinstance(model, str):
            model = _u.load_model(model)
        if isinstance(model, dict) and "layers" not in model:
            model = _u.analyze_model(model)
        return _u.fingerprint(model, dim=dim)

    def unicron_assimilate(self, model, out_path=None, mode="shrink", guard=True,
                           policy=True, big=4_000_000, rsvd_rank=256):
        """UNICRON'S FULL PASS, one call: load (safetensors/gguf) -> name-policy skip
        (embed/lm_head/conv/norm, decided by string match before any SVD) -> per-matrix
        MP filtering with the untrained-layer guard and a defrag safety valve (a layer the
        spike+bulk model does not fit is kept, not amputated) -> randomized SVD for huge
        matrices -> DENSE re-export under ORIGINAL tensor names, so the output loads
        wherever the input loaded. Returns (tensors, report); report["verify"] states the
        retention debt -- the output is UNVERIFIED until eval runs before-vs-after on the
        caller's runtime. See holographic_unicron.assimilate_model."""
        from holographic.io_and_interop import holographic_unicron as _u
        return _u.assimilate_model(model, out_path=out_path, mode=mode, guard=guard,
                                   policy=policy, big=big, rsvd_rank=rsvd_rank)

    def unicron_transform(self, model, mode="shrink", keep=None, guard=True, out_path=None):
        """TRANSFORM a whole model, Unicron's upgrade pass: rmt-filter every learned weight
        matrix (keep spectral outliers, drop the still-random MP bulk) and store factored
        (U,V thin pair) wherever that is genuinely smaller -- fewer parameters on disk AND
        fewer flops at inference. guard=True (default) passes through layers that look
        untrained: random-FEATURE layers are load-bearing while spectrally noise-like, and
        unguarded filtering measurably destroys them (-31 accuracy points on record).
        HONESTY CONTRACT: spectral surgery proves nothing about capability -- measure with
        unicron_retention. Input is a path or weights dict; out_path writes a .safetensors.
        Returns (new_tensors, report). See holographic_unicron.transform_model."""
        from holographic.io_and_interop import holographic_unicron as _u
        import numpy as _np
        if isinstance(model, str):
            model = _u.load_model(model)
        new, rep = _u.transform_model(model, mode=mode, keep=keep, guard=guard)
        if out_path:
            _u.save_safetensors(out_path, {k: _np.ascontiguousarray(v, _np.float32)
                                           for k, v in new.items()})
            rep["out_path"] = out_path
        return new, rep

    def unicron_reconstruct(self, model):
        """Exact inverse of unicron_transform's factored storage: every name.U/name.V thin
        pair multiplies back into a dense matrix. Path or dict in, dict out.
        See holographic_unicron.reconstruct_model."""
        from holographic.io_and_interop import holographic_unicron as _u
        if isinstance(model, str):
            model = _u.load_model(model)
        return _u.reconstruct_model(model)

    def unicron_retention(self, model_before, model_after, X, y, predict=None):
        """THE measurement every transform claim owes: accuracy before vs after on held-out
        data. Default predict handles the built-in pca_net/elm instrument models; any
        NumPy-callable predict(tensors, X) plugs in for other architectures. Returns
        {acc_before, acc_after, delta} -- numbers, no verdict words.
        See holographic_unicron.functional_retention."""
        from holographic.io_and_interop import holographic_unicron as _u
        import numpy as _np
        ms = []
        for m in (model_before, model_after):
            ms.append(_u.load_model(m) if isinstance(m, str) else m)
        return _u.functional_retention(ms[0], ms[1], _np.asarray(X), _np.asarray(y),
                                       predict=predict or _u.elm_predict)

    def unicron_runtime(self, model, cfg):
        """OWN the forward pass: a NumPy runtime for GDN-hybrid (Qwen3-Next / Qwen3.5
        class) models, VERIFIED against the reference implementation to 1.4e-7 relative
        logit error on a random model. Returns a GDNRuntime with .forward(ids, hooks=),
        .perplexity(ids) (the in-engine retention meter -- the standing eval debt now
        closes inside leCore), and .generate(). hooks={layer: fn(hidden)->delta|None}
        is the RESIDENCY injection point: leCore capabilities read and shape the live
        residual stream. Correctness-first (slow); text-only; dense MLP.
        See holographic_gdnruntime.GDNRuntime."""
        from holographic.io_and_interop import holographic_unicron as _u
        from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime
        if isinstance(model, str):
            model = _u.load_model(model)
        return GDNRuntime(model, cfg)

    def unicron_resident_memory(self, runtime, layer, keys, values, gain=1.0, threshold=0.3):
        """LECORE INSIDE THE MODEL: install a holographic associative memory as a
        resident expert at `layer`. Per token, the hook reads the hidden state, does a
        cosine cleanup against the stored keys (the same error-correcting recall the
        engine's cleanup memories use -- and the SAME delta rule GDN itself runs), and
        adds gain * value for confident matches. Returns the hooks dict to pass to
        forward/generate. Perfect recall at any capacity: the memory lives on OUR side
        of the boundary, so it is as infinite as leCore's memory is. Mechanics are
        verified in the runtime selftest; SEMANTIC effects on a trained model carry the
        eval debt, stated as always."""
        import numpy as _np
        K = _np.asarray(keys, _np.float64)
        V = _np.asarray(values, _np.float64)
        Kn = K / _np.maximum(_np.linalg.norm(K, axis=1, keepdims=True), 1e-12)

        def hook(h):
            hn = h / _np.maximum(_np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
            sim = hn @ Kn.T                                  # (S, n_mem)
            best = _np.argmax(sim, axis=1)
            conf = sim[_np.arange(len(best)), best]
            delta = _np.zeros_like(h)
            hit = conf > threshold
            delta[hit] = gain * V[best[hit]]
            return delta if hit.any() else None

        return {int(layer): hook}

    def unicron_bundle(self, path, weights, cfg, residents=(), notes="",
                       include_engine=True):
        """THE MODEL IS THE ENGINE: write a SELF-CONTAINED bundle -- weights, declarative
        resident manifest, the whole leCore source tree, its advertised capability
        schemas, and a `run.py` bootstrap. Boots on a machine where leCore was never
        installed (no build step, no compiled extension -- the NumPy/stdlib-only rule is
        what makes carrying the engine a directory copy). Verified by running it in an
        ISOLATED subprocess with leCore off the path. `python run.py serve` gives an
        OpenAI-compatible API; `--no-residents` gives the plain model.
        See holographic_galvabundle.bundle."""
        from holographic.io_and_interop import holographic_galvabundle as _b
        return _b.bundle(path, weights, cfg, residents=residents, notes=notes,
                         include_engine=include_engine)

    def unicron_capability_tools(self, limit=None):
        """The bundle's advertised feature set: every catalog capability as an
        OpenAI-style tool schema, generated from THIS running mind and the real method
        signatures (so parameter names are the actual ones, and nothing can be claimed
        that the engine does not have). Served live at /v1/capabilities, callable at
        /v1/invoke. See holographic_galvabundle.capability_tools."""
        from holographic.io_and_interop import holographic_galvabundle as _b
        return _b.capability_tools(self, limit=limit)

    def unicron_grounded_generate(self, runtime, token_ids, evidence, n_new=32,
                                  k=8, span=5, hooks=None):
        """DELIBERATION THAT MEASURABLY WORKS: fork the model's own top-k first tokens,
        continue each from the prefilled state, and keep the branch with the most spans
        SUPPORTED BY THE SOURCES (ties broken by likelihood).
        MEASURED against greedy over 10 runs on a trained subject: grounded fraction
        0.729 -> 0.921 (+19.3 points, up in EVERY run) and NLL 27.11 -> 23.58 (-13.0%).
        WHY THIS AND NOT IN-STREAM SWARM DIGESTS, also measured: injecting a
        deliberation digest was SILENT (identical branches make the contrast exactly
        zero) or, forced to fire with random steers, made NLL WORSE by 3.4 over 40
        tokens. The difference is the SCORER, not the branching -- self-likelihood
        cannot reward a branch for being RIGHT, only for being fluent, and the jury
        literature measures a model scoring its own candidates as the weakest selector
        available. Evidence support is external, so it can.
        See holographic_swarm.grounded_generate."""
        from holographic.agents_and_reasoning.holographic_swarm import (
            grounded_generate)
        return grounded_generate(runtime, token_ids, evidence, n_new=n_new,
                                 k=k, span=span, hooks=hooks)

    def unicron_retarget(self, weights, cfg, target_tokens=4096, kv_rank=64,
                         grow_gain=0.0, apply=False):
        """REBUILD A MODEL WHERE THE MEASUREMENT SAYS IT NEEDS REBUILDING, not uniformly.
        (Named `retarget` because `unicron_transform` was ALREADY TAKEN by the whole-model
        compression faculty -- defining it twice silently replaced the original, and the
        duplicate catalog key silently discarded the new aliases. Two silent overwrites
        from one name collision.)
        Recovers the block structure FROM THE WEIGHTS (which layers have linear-attention
        gates), measures each layer's memory, and targets each lever where it helps.
        MEASURED on Qwen3.5-0.8B: it is six blocks of (3 GDN + 1 full attention), and
        memory tracks POSITION IN BLOCK rather than depth -- the layer right after
        attention has a median half-life of 82 tokens against 9.7 and 9.9 for the other
        two, repeating in all six blocks.
        SO THE PLAN IS TARGETED:
          position 0  -> PRESERVE. The model's long memory already lives here; an edit
                         damages the thing that works. (Verified on real layer 12: plan
                         said preserve, weights came back untouched.)
          positions 1,2 -> GROW a long-memory channel. These are ~10-token local layers,
                         so the channel adds reach the model lacks and takes nothing
                         away; off by default, and verified BIT-IDENTICAL (6.2e-15) on
                         real split-layout tensors when the gain is zero.
          attention layers -> KV COMPRESSION, where the context ceiling actually is
                         (rank 64 measured 8x context at 1.3% attention error).
        Returns the PLAN as data by default so it can be inspected and diffed; apply=True
        carries it out. See holographic_transform."""
        from holographic.io_and_interop.holographic_transform import (
            plan as _plan, apply_plan)
        p = _plan(weights, cfg, target_tokens=target_tokens, kv_rank=kv_rank,
                  grow_gain=grow_gain)
        if not apply:
            return p
        return apply_plan(weights, cfg, p)

    def unicron_autoscale_memory(self, weights, cfg, target_tokens=4096, scales=4,
                                 gain=0.05, shortest=16):
        """SIZE THE MODEL'S MEMORY FOR A TARGET CONTEXT, arithmetically. Installs a
        geometric LADDER of holographic channels covering `shortest` to `target_tokens`.
        THE RULE IS DERIVED, NOT TUNED: decay = exp(-exp(a_log)*softplus(dt_bias)), so
        with dt_bias 0 the half-life is exp(-a_log) and a_log = -ln(D). Verified exact
        from 16 to 16,384 tokens.
        WHY A LADDER AND NOT ONE LONG CHANNEL, measured: three copies of the SAME channel
        add NOTHING (influence at 1024 identical to one) because reach is set by decay,
        not count -- extra accumulators buy capacity, not range. Measured at 1024 tokens:
        0.00026 for one channel against 0.00092 for a four-rung ladder, 3.5x the reach
        for +0.14% perplexity.
        WHAT THIS DOES NOT DO, since the phrase "context window" invites it: it does not
        lengthen the attention layer's window. On this GDN-hybrid that turned out not to
        be the binding constraint -- measured, perplexity barely moves from 128 to 1024
        tokens and RoPE scaling changes almost nothing, because most layers are linear
        attention carrying position through recurrence. The real limit was that the
        recurrent state FORGOT within a token, and that is what this fixes.
        See holographic_hrnngrow.autoscale_memory."""
        from holographic.io_and_interop.holographic_hrnngrow import autoscale_memory
        return autoscale_memory(weights, cfg, target_tokens=target_tokens,
                                scales=scales, gain=gain, shortest=shortest)

    def unicron_hrnn_grow(self, weights, cfg, a_log=-4.0, gain=0.0, layers=None):
        """ADD a holographic memory channel instead of stealing a trained head -- leCore's
        fourth lever (when capacity binds, add dimensions) applied to the architecture
        itself.
        unicron_hrnn_bake retuned an existing head and it worked, at +34.2% perplexity,
        because the model was TRAINED with that head forgetting fast. Growing a NEW
        key-head group costs almost nothing instead: the channel arrives with a slow
        decay so it accumulates, and a ZERO out_proj column so it contributes nothing
        until asked.
        MEASURED: with gain=0 the logits are BIT-IDENTICAL (max diff 0.0e+00) while the
        state carries the extra value-heads; at gain=0.05 the memory reaches further than
        the original ever did (influence at 256 tokens 0.00000 -> 0.00124) for
        +0.1% perplexity, against +34.2% for the retrofit.
        Every tensor grows as a plain weight edit (qkv/z, beta, conv, A_log, dt_bias,
        out_proj) and the config's head counts are bumped to match, so the result is an
        ordinary checkpoint any runtime can load.
        See holographic_hrnngrow.grow_channel."""
        from holographic.io_and_interop.holographic_hrnngrow import grow_channel
        return grow_channel(weights, cfg, a_log=a_log, gain=gain, layers=layers)

    def unicron_hrnn_bake(self, weights, cfg, heads=(0,), a_log=-4.0, layers=None):
        """THE MODEL'S OWN HEADS ARE HOLOGRAPHIC RNNs -- retune them instead of adding a
        resident. A gated-DeltaNet head computes S_t = a*S_{t-1} + b*k_t v_t^T, which IS
        leCore's HRNN: outer-product binding accumulated into a state with a decay gate.
        Nothing needs adding; the knob just needs setting, and a knob is a WEIGHT, so it
        survives export where a resident does not.
        WHAT THE AUDIT FOUND: on a trained checkpoint every head's half-life is ~0.1-0.2
        TOKENS. The heads forget within a single step -- which is why the causal memory
        horizon measured 32 tokens despite a 2048-number state. The architecture pays for
        a holographic memory and discards it every token.
        MEASURED after retuning head 0 to a_log=-4: influence at 256 tokens 0.00000 ->
        0.00059 with no vanishing horizon, at a cost of +34.2% perplexity; distilling the
        head back toward the original's logits recovers part of it (+24.1%, agreement
        0.734 -> 0.792) and cannot recover all, because a head fit changes how the state
        is READ, not what it IS.
        THIS IS A RETROFIT, NOT A FREE WIN: the model was TRAINED with fast-forgetting
        heads and its later layers depend on that. Default OFF. See holographic_hrnnbake."""
        from holographic.io_and_interop.holographic_hrnnbake import bake_channel
        return bake_channel(weights, cfg, heads=heads, a_log=a_log, layers=layers)

    def unicron_load_factors(self, runtime, factors):
        """MAKE THE SMALLER MODEL ACTUALLY FASTER. Attaches the low-rank factors from
        unicron_refactor so the forward pass USES them: (x@B.T)@A.T costs r*(m+n)
        multiplies against m*n, so a factored projection is cheaper to RUN, not merely
        smaller on disk. Without this the runtime reconstructs the dense matrix and
        throws the saving away -- which is how a "35% smaller model" ends up exactly as
        slow as before.
        MEASURED: per-matmul 1.24x / 1.28x / 1.64x at this model's shapes; whole forward
        1.20x with logits IDENTICAL to the reconstructed dense; generation 1.08x, and
        1.50x stacked with unicron_leap (762 -> 1144 tokens/sec, output token-identical).
        The gains are modest on a small model where NumPy call overhead dominates the
        arithmetic; the FLOP ratio is what scales with width.
        Anything not listed in `factors` stays dense, so this is additive."""
        return runtime.load_factors(factors)

    def unicron_gather_attention(self, Q, K, V, clusters=64, keep=4, tile=256,
                                 causal=False):
        """BANK THE ROUTING SAVING instead of reporting it. Screen routing could name the
        right ~38% of keys since the first arc, and the code still computed the DENSE
        score matrix and masked it -- measured, that path is SLOWER THAN DENSE (11.53s
        against 8.96s on 2048 tokens), because it does all the work plus an argpartition
        and a scatter.
        Two of the project's own levers fix it: BAKE ONCE, SAMPLE O(1) (centroids
        computed per sequence, not per query -- 64 centroids instead of 2048 keys) and
        PARTITION INTO A COMMUTATIVE MONOID (softmax over a selected union of clusters
        has the same shape as softmax over all of them, which is what makes the gather
        legal).
        MEASURED, wall clock, 2048x8x128:  dense 8.9615s | masked-after 11.5331s |
        GATHER FIRST 0.8601s -- 10.4x dense and 13.4x the old path.
        THE COST IS APPROXIMATION and it is a dial: at 2 of 64 clusters relative error
        0.616, at 8 of 64 it is 0.190, and keeping all clusters is exact to 1.8e-15.
        Causal mode reproduces dense causal attention exactly, so the router cannot leak
        the future -- the failure that announced itself in the first screen arc as a
        perplexity BELOW dense, which is impossible for a restriction.
        See holographic_gatherattn."""
        from holographic.io_and_interop.holographic_gatherattn import (
            gather_attention)
        return gather_attention(Q, K, V, clusters=clusters, keep=keep,
                                tile=tile, causal=causal)

    def unicron_kv_compress(self, rank=64, refit_every=0):
        """LONGER CONTEXT AT FIXED MEMORY -- shrink the KV cache, which is what actually
        bounds context, instead of the model.
        MEASURED on a real Qwen3.5-0.8B layer with its own activations, scored on the
        ATTENTION OUTPUT rather than the cache contents:
            rank  KV memory  attn error  context at the same RAM
              8      1.6%      0.0534         64x
             32      6.2%      0.0272         16x
             64     12.5%      0.0131          8x
            128     25.0%      0.0041          4x
        K and V compress because the residual stream does -- 95% of its energy sits in
        ~130 of 1024 directions and K/V are linear images of it, so they inherit the
        concentration (K needed rank 67 of 512 for 90% of its energy).
        The basis is FITTED from the sequence's own K/V during prefill and new tokens are
        PROJECTED onto it, one matmul per step, so the saving survives generation instead
        of existing only in a benchmark.
        HONEST LIMITS, both measured: it is LOSSY and the error grows as rank falls (the
        table is the whole trade); and the basis is stored too, so compression only pays
        past roughly 2*rank tokens -- break_even_tokens() reports where.
        See holographic_kvcompress.CompressedKV."""
        from holographic.caching_and_storage.holographic_kvcompress import (
            CompressedKV)
        return CompressedKV(rank=rank, refit_every=refit_every)

    def unicron_residual_correction(self, clean_fn, quant_fn, states, rank=32,
                                    ridge=1e-3, store_bits=8):
        """PREDICT QUANTIZATION DAMAGE FROM THE INPUT AND SUBTRACT IT -- the approach that
        worked after three that did not.
        Pruning, activation-aware scaling and readout cleanup all failed, and the last one
        died on a measurement: the error matrix needs rank 83 of 235 for 90% of its
        energy, so no projector separates error from signal. The measurement was right;
        the conclusion was wrong.
        Quantization error is NOT NOISE -- it is a DETERMINISTIC FUNCTION OF THE INPUT.
        And the model never explores its full input space: activations occupy ~130 of 1024
        dimensions. So the error's ACTION ON THE MANIFOLD THE MODEL USES is low rank even
        though the error MATRIX is not. Fit input -> residual, keep the top ranks, add it
        back with two small matmuls.
        MEASURED on a real layer, fitted on 160 positions, scored on 75 HELD OUT:
            4-bit plain          0.10616
            + rank 16 (+65 KB)   0.08937   -16%
            + rank 32 (+131 KB)  0.08449   -20%
            + rank 64 (+262 KB)  0.07790   -27%
        HONEST SIZE ACCOUNTING, shipped with the win rather than after it: 5-bit plain
        reaches 0.04963 and beats all of these OUTRIGHT -- but costs +25% size for -53%
        error, while rank 64 costs +4.8% for -27%. PER BYTE THE CORRECTION IS ~2.6x MORE
        EFFICIENT, so it wins at a fixed small budget and loses if you can simply afford
        another bit.
        ACCELERATIONS, both measured: the CORRECTION ITSELF COMPRESSES FOR FREE -- rank 32
        at 32/8/4/3 bits gives 0.08449 / 0.08450 / 0.08648 / 0.09341, so 8-bit storage is
        4x smaller at no cost and quadruples the byte-efficiency of the whole technique
        (default store_bits=8). And ITERATING IS A KEPT NEGATIVE: four greedy rank-8
        passes reach EXACTLY the same 0.08449 as one rank-32 truncation, which is what
        the SVD says must happen -- there is no free refinement.
        See holographic_refactor.fit_residual_correction."""
        from holographic.io_and_interop.holographic_refactor import (
            fit_residual_correction)
        return fit_residual_correction(clean_fn, quant_fn, states, rank=rank,
                                       ridge=ridge, store_bits=store_bits)

    def unicron_fold_correction(self, weights, cfg, correction, layer=None,
                                mean_h=None, gate_target=16.0):
        """MAKE THE CORRECTION PART OF THE MODEL -- a rank-r map IS r MLP neurons.
        An MLP neuron computes exactly one rank-1 term, so putting A[:, j] in the up row
        and B[j] in the down column, with the gate held near constant, turns the whole
        correction into ordinary weights. It then quantizes, exports and runs like any
        other neuron: no runtime hook, no separate matmul, nothing for a GGUF converter
        to drop.
        MEASURED on a real layer: 4-bit plain 0.10616 | correction as a separate matmul
        0.08449 | correction FOLDED as 32 neurons 0.08475. The fold costs 0.3% of the
        gain to the gate's per-token variation and widens the MLP by 0.9%.
        See holographic_refactor.fold_correction."""
        from holographic.io_and_interop.holographic_refactor import fold_correction
        return fold_correction(weights, cfg, correction, layer=layer,
                               mean_h=mean_h, gate_target=gate_target)

    def unicron_requantize(self, weights, cfg, eval_tokens, budget=0.01,
                           ladder=(8, 6, 5, 4, 3), group=64, progress=None):
        """CHOOSE A BIT WIDTH PER TENSOR BY MEASUREMENT -- the right lever for a
        heavy-tailed model, which is what real checkpoints are.
        MEASURED on a real Qwen3.5-0.8B layer with its OWN activations, comparing OUTPUT
        error at matched size:
            low-rank at 25% of fp16   error 0.54
            4-bit at 25% of fp16      error 0.107   <- 5x better
            8-bit at 50%              error 0.0062
        Every projection in that model is heavy-tailed (signal rank 9-23% of full by
        Marchenko-Pastur, yet truncation wrecks the output) -- the exact regime the router
        says to pass through for rank cuts. Heavy tails resist RANK and tolerate
        PRECISION; picking the wrong one is how a compressor lands 5x worse at the same
        size, which is what unicron_refactor alone was doing on real weights.
        KEPT NEGATIVE: correcting the quantization RESIDUAL with low rank (the qlr idea)
        barely helped -- 0.107 -> 0.096 for 8% more size -- because the residual is
        heavy-tailed too. These levers do not compose here.
        Group-wise symmetric quantization, the shape llama.cpp uses, so the result
        converts to GGUF without a second story. See holographic_refactor.requantize."""
        from holographic.io_and_interop.holographic_refactor import requantize
        return requantize(weights, cfg, eval_tokens, budget=budget,
                          ladder=ladder, group=group, progress=progress)

    def unicron_refactor(self, weights, cfg, eval_tokens, budget=0.01,
                         skip=("embed", "lm_head"), progress=None):
        """TAKE THE MODEL APART AND REBUILD IT SMALLER -- the decomposition half of
        Unicron's brief, which filtering was standing in for. A model is not a black box,
        it is vector data: every projection has a spectrum and most carry their behaviour
        in far fewer directions than they store. Each matrix is decomposed, the SMALLEST
        rank whose cost stays inside a measured budget is kept, and the model is rebuilt.
        MEASURED on a trained subject: budget +1% -> 35.0% fewer parameters at an actual
        +0.99%; budget +5% -> 42.8% fewer at +4.98%. The budget holds because every
        candidate rank is applied ALONE and scored, never predicted.
        TWO REFUSALS, both arithmetic rather than taste: it will not factor a matrix when
        r*(m+n) >= m*n (99%-energy factoring INFLATES 25 of 27 tensors on a small model --
        a compressor that grows its input is a bug with a press release), and it leaves
        embeddings and the head alone, since damage there shows up as garbled text rather
        than as a number.
        COMPATIBILITY IS THE POINT: reconstruct() returns ordinary dense tensors of the
        original shape, so the same rebuild converts to GGUF and loads in Ollama --
        smaller, with no runtime needing to know what happened.
        See holographic_refactor.decompose."""
        from holographic.io_and_interop.holographic_refactor import decompose
        return decompose(weights, cfg, eval_tokens, budget=budget, skip=skip,
                         progress=progress)

    def unicron_progbake(self, symbols=None, traces=None, vocabulary=None,
                         dim=1024, n_symbols=None, tag="prog"):
        """STORE PROGRAMS IN THE MODEL'S UNUSED VOCABULARY and project them back out.
        A checkpoint has vector-shaped rooms nobody uses: Qwen3.5-0.8B declares vocab
        248,320 while its tokenizer defines 248,044, leaving 276 dead rows in the
        embedding and head. They are exactly the shape of a hypervector, so a program --
        a WGSL shader, a procedural recipe, anything leCore generates on the fly -- is
        encoded as a role-filler trace, written into those rows, and projected out by
        unbinding a position and cleaning up against the codebook. Both operations
        already exist inside the weights (unbind is a circulant matrix, cleanup is
        argmax over a codebook, which is what lm_head is).
        DEMONSTRATED: a real 282-character WGSL vertex+fragment shader stored in ONE row
        and recovered SYMBOL-EXACT; a 140-symbol program chunked across 5 rows, exact.
        CAPACITY IS MEASURED AND SMALLER THAN THE OBVIOUS GUESS: 32 symbols per row
        (20/20 programs perfect at 32, 13/20 at 40). bundle_capacity reports 174 at
        d=1024 for ITS readout; quoting that here would have been a five-fold overclaim.
        276 rows x 32 ~ 8,800 symbols, about 50 KB of program text, carried inside the
        checkpoint and addressable by token id. See holographic_progbake."""
        from holographic.io_and_interop.holographic_progbake import (
            encode_program, decode_program)
        if traces is not None:
            return decode_program(traces, vocabulary, dim,
                                  n_symbols or 0, tag=tag)
        return encode_program(symbols or (), dim, tag=tag)

    def unicron_harden(self, weights, cfg, seed="leCore", facts=(), program=None,
                       machine=None, probe_ids=None):
        """PROVE THE INSTALLED LAYER WORKS, AND KEEPS WORKING WHEN ABUSED.
        Every piece of this stack has its own selftest and none of them answered the
        question that matters: can an INSTALLED model BOOT and USE the layer from the
        weights alone, and does it survive what happens to checkpoints in the wild?
        Eight checks, each one a failure this project has actually shipped at least once:
        BIOS POST and enumeration, boot from weights, DETERMINISTIC expansion (hashlib,
        not hash()), an ADDRESSED channel (a wrong seed must read noise), recall by key,
        a stored program that EXECUTES, and a cache that measurably saves work.
        VERIFIED: 8/8 on an installed model, and 4/8 on one never installed AND on one
        requantized afterwards -- a harness that cannot fail is decoration.
        IT IMMEDIATELY FOUND TWO REAL DEFECTS: the boot spill and the stored program both
        wrote the WHOLE surface and silently clobbered each other (each component's own
        selftest writes exactly one payload, so nothing else could have seen it), now
        fixed with a named-parts container; and this harness itself had an UNWRAPPED
        probe, so a damaged model raised out of it instead of being reported -- a
        verifier that crashes on the input it exists to judge tells you nothing.
        See holographic_harden."""
        from holographic.io_and_interop.holographic_harden import harden
        return harden(weights, cfg, seed=seed, facts=facts, program=program,
                      machine=machine, probe_ids=probe_ids)

    def unicron_evolve(self, params, fitness_fn=None, sigma=0.02, lr=0.3,
                       population=32, seed=0, rank=4, generations=20,
                       patience=None, progress=None):
        """EGGROLL-STYLE EVOLUTION STRATEGIES -- the training method this engine can
        actually run, because ES needs ONLY FORWARD PASSES and leCore is a forward-pass
        machine. The no-autodiff constraint that shaped every design decision here is
        IRRELEVANT to evolution strategies; that is the finding, not the code.
        THE AUDIT FIRST, so this does not rebuild what exists: `agent_benchmark` is
        already a REWARD FUNCTION (pre-registered false-action rate on a no-tool set,
        plus resolution rate and refusals, in ~2s) and `wgsl_device`/`wgsl_bind_batch`
        are already a vendor-neutral GPU path. Only the population harness was missing.
        THREE THINGS FROM THE PAPER: LOW-RANK PERTURBATIONS (a 0.8B's leCore additions
        are 10.31M parameters, 0.52M at rank 4 -- measured 177x smaller on a real pair of
        shapes); SEED-DERIVED MEMBERS (regenerated from a seed, so memory is O(population)
        integers and a run repeats in another process -- hashlib, never hash()); and
        ANTITHETIC PAIRS WITH RANK SHAPING so one outlier cannot own an update.
        VERIFIED: a quantised (non-differentiable) loss falls where no gradient exists,
        and on a REAL model ES lowered end-to-end perplexity through a full forward pass
        with no autodiff anywhere.
        HONEST LIMITS, all measured: ES LOSES to least squares on convex problems
        (0.08937 -> 0.08927, a rediscovery), LOSES badly on a 256k-dim discrete rounding
        search, and on a real model with a 192-forward budget it improved the objective
        but did NOT beat the base held-out. It belongs on end-to-end non-differentiable
        objectives with a real budget -- ~71 GPU-hours for a serious run by the
        arithmetic in NOTES. See holographic_evolve."""
        from holographic.agents_and_reasoning.holographic_evolve import Evolve
        ev = Evolve(params, sigma=sigma, lr=lr, population=population, seed=seed,
                    rank=rank)
        if fitness_fn is None:
            return ev
        return ev.run(fitness_fn, generations=generations, patience=patience,
                      progress=progress)

    def unicron_assess(self, model_dir, out_path, text=None, n_gen=32,
                       compare_paths=None):
        """MEASURE A MODEL SO SOMEONE ELSE CAN JUDGE IT. After a run there are several
        artifacts -- original, assimilated, repaired, requantized, the imbued bundle --
        and the only honest comparison is on the SAME probe with the SAME instrument.
        Writes ONE bundle per model directory: BIOS profile and POST, perplexity,
        generation tokens/sec, A_log/dt_bias gates, FULL singular values per 2-D tensor,
        hidden states at every layer, top-64 logits with the exact log-sum-exp so
        probabilities are recoverable, the bundle's resident roster, and the 8-check
        hardening audit.
        IT IS A PROFILE, NOT THE MODEL: no weight tensors, no training data, no text
        beyond the probe, and a manifest inside the file naming everything it contains.
        compare() lines several bundles up, which is the point -- one run's perplexity
        means nothing without the run beside it. See holographic_assess."""
        from holographic.io_and_interop.holographic_assess import assess, compare
        if compare_paths:
            return compare(compare_paths)
        return assess(model_dir, out_path, text=text, n_gen=n_gen)

    def unicron_deployable(self, bundle_dir, original_dir=None, probe_ids=None,
                           tolerance=0.01):
        """IS THIS ARTIFACT ACTUALLY DELIVERABLE? Convertible AND no worse.
        Moose's requirement, and the one this project had drifted from: a Galvatron has
        to run wherever the original ran and work at least as well. Reduced disk space is
        worthless on its own, and a size number has misled this work more than once.
        CHECK 1 -- CONVERTIBILITY. llama.cpp's convert_hf_to_gguf.py reads config.json in
        HUGGING FACE SHAPE (hidden_size, num_hidden_layers) beside model.safetensors. The
        bundle was shipping galvatron.json INSTEAD, so the artifact ran in leCore and
        NOWHERE ELSE -- found by checking a produced bundle against what the converter
        actually reads, not by assuming. imbue now carries config.json,
        generation_config.json and the full tokenizer set.
        CHECK 2 -- QUALITY. Perplexity against the original on the same tokens, with a
        tolerance the CALLER states rather than one this function invents.
        VERIFIED to catch both failures, not just to pass a good case: a healthy bundle
        reads deployable=True at -0.03%, one with config.json removed fails
        convertibility, and one with noised weights fails quality at +82.9%.
        See holographic_galvapack.check_deployable."""
        from holographic.io_and_interop.holographic_galvapack import (
            check_deployable)
        return check_deployable(bundle_dir, original_dir=original_dir,
                                probe_ids=probe_ids, tolerance=tolerance)

    def unicron_model_store(self, weights=None, cfg=None, path=None,
                            lazy=True, materialize_to=None, dtype=None):
        """KEEP THE MODEL IN leCORE'S FORMAT, HAND OUT A BORING CHECKPOINT.
        The compatibility curtain Moose asked for, and the audit found almost all of it
        already built: holographic_container is a TYPED-SECTION container whose defining
        property is that a section the reader does not understand ROUND-TRIPS UNTOUCHED
        (written for leStudio workspaces, exactly right here, changed not at all);
        LazyWeights already materialises per tensor on demand; middle_out_encode is the
        codec; export_portable already writes ordinary safetensors. Only the JOIN was
        missing -- the compressed store existed only AFTER loading a plain file, so it
        bought RAM and not disk, not load time, and not the memory bandwidth that bounds
        generation (3.49 GB per token at float32 on a 0.8B -- the reason that model ran
        at 0.6 tokens/sec).
        MEASURED end to end: 50 tensors, 27 encoded, 2.81 MB raw -> 0.89 MB on disk
        (3.16x), loading back into a RUNNING model both eagerly and lazily with a max
        logit deviation of 0.003, and materialize() writing an ordinary checkpoint that
        load_runtime opens.
        PER-TENSOR CHOICE: small tensors stay raw because a codec header outweighs them,
        and an encoding is KEPT ONLY IF SMALLER -- a compressor that grows its input is a
        bug with a press release, and this project shipped that one already.
        HONEST ABOUT DIRECTION: nothing here lets Ollama read the leCore format. It lets
        the leCore format be the ARCHIVE and produce a boring checkpoint on demand.
        See holographic_modelstore."""
        from holographic.io_and_interop.holographic_modelstore import (
            save_model, load_model, materialize)
        if materialize_to is not None:
            return materialize(path, materialize_to, dtype=dtype)
        if weights is not None:
            return save_model(weights, cfg, path)
        return load_model(path, lazy=lazy)

    def unicron_tensor_map(self, spectra, dim=512, query=None, k=5,
                           outlier_threshold=0.9):
        """EVERY WEIGHT TENSOR AS A HYPERVECTOR, AND THE MAP THAT FALLS OUT.
        A .safetensors file is a few hundred matrices with names, and every real question
        about one is RELATIONAL: which tensors resemble each other, does this checkpoint
        change partway down, did an edit make one tensor stop looking like its siblings.
        The audit found only pieces -- unicron_subspace compares TWO matrices by principal
        angles, delta_lineage ranks candidate BASES -- and nothing that laid out a whole
        file.
        A tensor's hypervector BINDS its ROLE (a hashed embedding of the name path, so
        mlp.up_proj across every layer shares one) to the SHAPE OF ITS SPECTRUM
        (log-binned normalised singular values, r50/r90/r99, and the heavy-tail signature
        that decided this project's whole compression strategy). Binding rather than
        concatenating means a match must satisfy BOTH halves -- concatenation lets a
        strong role match carry a weak spectral one.
        Everything is scale-free, so a 3584x1024 MLP and a 16x1024 gate compare directly.
        MEASURED ON A REAL Qwen3.5-0.8B, 246 tensors, from spectra alone (no weights):
            same-role tensors cohere at mean cosine 0.974 (0.997 for gate_proj)
            DIFFERENT roles sit at -0.014 -- they genuinely separate
            embed_tokens' nearest neighbour is 0.146, alone as it should be, because
                its rows are a vocabulary rather than a transform
            layer 0's up_proj neighbours are layers 5, 4, 3 at 0.998
            zero outliers on a healthy checkpoint, and a TAMPERED spectrum is flagged
        This is a DIAGNOSTIC, not a compressor: it says what a checkpoint is shaped like,
        and it catches an edit that made one tensor diverge from its siblings -- the
        failure a per-tensor selftest cannot see. See holographic_tensormap."""
        from holographic.io_and_interop.holographic_tensormap import (
            encode_file, role_coherence, neighbours, outliers)
        names, V = encode_file(spectra, dim=dim)
        if query is not None:
            return neighbours(names, V, query, k=k)
        return {"names": names, "vectors": V,
                "roles": role_coherence(names, V),
                "outliers": outliers(names, V, threshold=outlier_threshold)}

    def unicron_measure(self, runtime, token_ids, compare_to=None, alpha=0.05,
                        effect_pct=None):
        """PERPLEXITY WITH ERROR BARS, AND A VERDICT THAT CAN SAY "UNDECIDABLE".
        Moose asked what assimilation is actually doing. From his own run: 265 tensors
        examined in 149 seconds, 18 CHANGED, repair reverted 12 as harmful, SIX kept;
        original 76.83 -> assimilated 81.71 (6.4% WORSE) -> repaired 75.06, reported as
        "beats the original: True".
        Then I measured the measurement, on his real model, from the assessment bundle's
        own per-token likelihoods: bootstrap 95% CI over 161 positions is 16.90..36.61,
        i.e. +/-38.5%; in 40-token chunks the spread is +/-47.4%. THE 2.3% "WIN" WAS
        NEVER MEASURED -- it sits deep inside the noise of the instrument that reported it.
        A 40-token probe can only resolve effects above 70%; detecting 2% would need
        28,252 tokens.
        So: measure() returns perplexity WITH a bootstrap interval, and better_than()
        returns BETTER, WORSE or INDISTINGUISHABLE using a PAIRED test over the same
        positions -- pairing removes the probe-choice variance that swamps everything, so
        it can detect small CONSISTENT shifts an unpaired comparison cannot.
        VERIFIED on the case that matters most: a model compared to ITSELF reads
        INDISTINGUISHABLE rather than finding a winner (the first version called it WORSE
        on a zero-width interval), a noised model reads WORSE, and a short probe REPORTS
        what it is incapable of resolving. See holographic_measure."""
        from holographic.io_and_interop.holographic_measure import (
            measure, better_than, tokens_needed)
        m = measure(runtime, token_ids, alpha=alpha)
        if compare_to is not None:
            return better_than(m, compare_to, alpha=alpha)
        if effect_pct is not None:
            m = dict(m, power=tokens_needed(m, effect_pct))
        return m

    def unicron_sidecar(self, base_dir, path=None, gain=1.0, merge_to=None,
                        seed="leCore", notes=""):
        """LEAVE THE MODEL ALONE. PUT leCORE IN FRONT OF IT.
        Moose, after watching three runs damage a model and then repair it: replace the
        file with a WRAPPER that pulls from elsewhere, and put the leCore weights, bios
        and the rest in a small thing in FRONT of the real model -- not in the Qwen
        weights themselves. He is right, and it makes every failure of this arc
        STRUCTURALLY IMPOSSIBLE, because all of them came from editing the base:
        assimilation filtered 18 tensors and made the model 6.4% WORSE; repair reverted 12
        of them and claimed a win inside the noise; a boot record written into a TIED
        embedding row destroyed the output head; bakes that landed, bakes that silently
        did not, and guards built to catch the damage. None of it can happen to a file
        nobody writes to.
        THE BASE STAYS BYTE-IDENTICAL, always deployable, always convertible. The sidecar
        carries the boot record, per-tensor LOW-RANK deltas, installed circuits and the
        call-token head rows -- about 10 MB against a 1.75 GB base.
        THREE WAYS TO CONSUME IT: load() materialises base+sidecar in memory; merge()
        writes ONE ordinary checkpoint for llama.cpp and Ollama, which expose no loader
        hook; and doing nothing still leaves a model that runs unchanged.
        VERIFIED: a 0.070 MB sidecar beside an 86 MB base -- gain=0 leaves the base
        BYTE-IDENTICAL, gain=1 changes exactly the tensors it declared and nothing else,
        the base file is never written to, and merge() produces a directory load_runtime
        opens.
        WHY IT BEATS BAKING BEYOND SAFETY: every component becomes separately MEASURABLE
        and separately REVERTIBLE -- a delta that does not earn its place is deleted from
        a manifest instead of reverted out of a 1.75 GB file, and the comparison is
        base vs base+delta on the SAME probe, which is the paired measurement that finally
        has the power to decide anything. See holographic_sidecar."""
        from holographic.io_and_interop.holographic_sidecar import (
            new_sidecar, load, merge)
        if merge_to is not None:
            return merge(base_dir, path, merge_to, gain=gain)
        if path is not None:
            return load(base_dir, path, gain=gain)
        return new_sidecar(base_dir, seed=seed, notes=notes)

    def unicron_install_facts(self, weights, cfg, runtime, facts, margin=1.0,
                              max_cosine=0.25, probe_prompts=None):
        """TEACH A MODEL TO SAY WHAT IT COULD NOT SAY -- and know when it cannot.
        The demonstration that leCore is really IN the weights: pick a prompt the model
        has no opinion about, name an answer token it ranks near last, and make it the
        answer, WEIGHTS-ONLY, with nothing running.
        The mechanism is one line of linear algebra -- the head turns a hidden state into
        logits, so raising ONE logit for ONE state is a rank-1 term on ONE row:
        row[answer] += need * h / (h @ h).
        MEASURED: 6 facts the model ranked at position 621 on average now come out FIRST,
        40 of 40 guard prompts byte-for-byte unchanged, exactly 6 of 2048 head rows
        touched.
        SEPARATION IS EVERYTHING, which is why this REFUSES rather than tries. If two
        prompts produce nearly the same hidden state, a fact attached to one IS attached
        to the other and no update can prevent it. Same code, same margins, two models:
            SmolLM2 sliced to 4 of 30 layers   cosine 0.581, 45 eff dims of 576
                                               -> 2/8 facts, 31/80 guards survived
            a full-depth model                 cosine 0.002, 138 eff dims of 512
                                               -> 8/8 facts, ALL 80 guards unchanged
        Depth is where representations separate; a model missing 87% of its depth has
        states that all point the same way. On Moose's own slice this reads cosine 0.796
        with SIX effective dimensions and declines, leaving the weights untouched.
        RECOVERY OF THE HEAD INPUT IS BY LEAST SQUARES, not by a hook: this runtime's
        hooks expose the residual stream at layer ENTRY, so the last layer and the final
        norm are both missing -- measured as a 160x scale error and a fit that taught
        nothing. The head is overdetermined, so lstsq is exact to 1e-13.
        WHAT IT IS NOT: the fact is attached to a PROMPT, not to a meaning, so a
        paraphrase lands elsewhere. See holographic_factbake."""
        from holographic.io_and_interop.holographic_factbake import install_facts
        return install_facts(weights, cfg, runtime, facts, margin=margin,
                             max_cosine=max_cosine, probe_prompts=probe_prompts)

    def unicron_vsa_run(self, weights=None, cfg=None, key=None, codebook=None,
                        rows=None, layer=None, gain=1.0, mean_h=None,
                        improve=None):
        """leCORE'S READ PATH EXECUTING IN THE FORWARD PASS, not stored beside it.
        A boot record is DATA. A fact in a head row is DATA. Neither computes. What
        computes in a forward pass is a matmul and a nonlinearity -- so a leCore operation
        belongs inside a model exactly when it can be written as one, and the VSA read
        path can be:
            UNBIND   circular correlation with a key is LINEAR in the trace, so it is one
                     fixed H x H matrix -- installable as MLP neurons
            CLEANUP  nearest neighbour in a codebook is an argmax over dot products, which
                     is what an output head already does
PROVEN, on our own trained model: unbind and bind agree with the FFT to 1e-10; a 6-pair
        memory returns 6/6 by matmul and argmax alone; and INSTALLED as 128 MLP neurons
        the circuit computes the unbind on the LIVE residual stream at cosine 1.000000.
        The model performs leCore's algebra on every token, from the weights, with nothing
        loaded.
        AND leCORE CAN MAKE THE MODEL BETTER FROM INSIDE, on every prompt. Pass
        improve=<runtime>, key=<fit token ids>, codebook=<held-out token ids> and this
        fits a CLOSED-FORM correction -- no gradients, because the direction that raises
        the true token IS A[true] - E_p[A] for a linear head -- then CHOOSES the step by
        measuring on held-out text with a paired bootstrap rather than by eye. MEASURED on
        our own trained model: -0.068% at step 32, -0.258% at 128, -0.480% at 256,
        -1.061% at 1024, monotone and BETTER at every point.
        NOT YET WORKING, said plainly: routing the READ path's output to the head so the
        model's own argmax reads a stored value back -- measured 1 of 6. The unbind result is ADDED to a
        residual that still holds the trace, and the trace dominates. Gain from 1 to 1000
        changes nothing, which rules out attenuation; the gate attenuates a foreign vector
        8x but does not close it. The circuit needs to write where the trace is not, which
        is an extra-dimensions problem rather than a gain problem.
        See holographic_vsarun."""
        from holographic.io_and_interop.holographic_vsarun import (
            install_read_path, unbind_matrix, bind_matrix, make_memory,
            install_improvement)
        if improve is not None:
            return install_improvement(weights, cfg, improve, key, codebook,
                                       layer=layer)
        if weights is None:
            return {"unbind_matrix": unbind_matrix, "bind_matrix": bind_matrix,
                    "make_memory": make_memory}
        return install_read_path(weights, cfg, key, codebook, rows,
                                 layer=layer, gain=gain, mean_h=mean_h)

    def unicron_memory_search(self, runtime=None, cfg=None, passages=None,
                              tokenize=None, cue=None, index=None, k=3,
                              weights=None, rows=None, decay=0.99):
        """SEARCHABLE MEMORY THAT LIVES IN THE WEIGHTS AND RUNS IN THE FORWARD PASS.
        Moose's requirement: the model loads in Ollama like any other model, and when it
        is used leCore runs AS PART OF IT -- no Python called out to.
        THE PANEL SETTLED THE DESIGN. Kanerva: an associative memory is a codebook plus a
        nearest match, and a transformer's HEAD IS ALREADY BOTH -- the search does not need
        building, it needs POPULATING. Quilez: do not inject what the machine can address
        itself; every earlier attempt pushed a trace in from outside and the trace drowned
        the answer. Milanfar: cleanup IS denoising, which is why one mechanism serves
        recall, search and correction.
        MEASURED on our own trained model, 64 passages:
            addressing by the LAST hidden state      2/64 -- it reflects recent tokens
            addressing by a BUNDLE over positions   57/64 top-1, 60/64 top-3,
                                                    from a cue with 24 of 40 characters
        The 2-to-57 jump is the whole design, and it is Kanerva's distributed address.
        AND THE BUNDLE IS COMPUTABLE IN THE PASS: a normalised exponential accumulator
        reproduces the mean over positions at COSINE 0.9998, and a linear-attention channel
        with A_log near zero IS that recurrence -- leCore already grows those. Normalising
        matters: without it the address scales with LENGTH and retrieval drops to 18/64,
        because a short cue and a long passage land at different magnitudes.
        SO THE WHOLE PATH IS WEIGHTS: a grown channel accumulates the address, stored
        addresses occupy head rows, and the model's own argmax ranks them.
        WHAT IT DOES NOT DO: the model does not DECIDE to search -- it computes the address
        on every token because that is what a channel does. Conditional retrieval is
        control flow, and a forward pass has none. See holographic_memsearch."""
        from holographic.agents_and_reasoning.holographic_memsearch import (
            build_index, search, install_index)
        if weights is not None and index is not None:
            return install_index(weights, index, rows)
        if cue is not None and index is not None:
            return search(runtime, index, cue, tokenize, k=k)
        return build_index(runtime, cfg, passages, tokenize, decay=decay)

    def unicron_router(self, runtime=None, cfg=None, positive=(), negative=(),
                       tokenize=None, layer=None, text=None, router=None,
                       weights=None, operator=None, gain=1.0):
        """THE MODEL DECIDING, INSIDE ONE FORWARD PASS -- the piece Moose named.
        I had been reporting, correctly and repeatedly, that "a forward pass emits logits,
        not control flow", and drawing the wrong conclusion from it. A forward pass has no
        TOKEN-LEVEL control flow. It has GATING: a direction computed by an EARLY layer
        switches a circuit on or off in a LATER one, and that is a decision made inside
        the pass by the weights with nothing running. Two stages, one model -- the first
        layers route, the later layers act.
        MEASURED on our own trained model, separating "this prompt wants a lookup" from
        ordinary continuation:
            layer 0  92% train  98% HELD-OUT      layer 2  97%  99%
            layer 1  96%        98%               layer 3  98%  99%
        The model already knew what kind of thing it was reading; nothing had asked it.
        INSTALLED AS A GATE the circuit reads +30.98 on a question and -1.52 on plain
        text, so it switches ITSELF on. That is the difference between a model carrying a
        memory and a model that consults one when the prompt calls for it -- every circuit
        installed before this fired on every token, because install_op deliberately holds
        its gate near-constant.
        HONEST SHAPE: the decision is a linear readout of an early hidden state, so it
        decides what it was fitted to decide. It is a ROUTER, not a reasoner -- and a
        router was the only missing piece, because everything downstream was already
        built and measured. A router fitted on 18 examples scored 100% train and 61%
        held-out; the accuracy is reported for that reason. See holographic_router."""
        from holographic.agents_and_reasoning.holographic_router import (
            fit_router, route, install_routed)
        if weights is not None and operator is not None:
            return install_routed(weights, cfg, operator, router, layer=layer,
                                  gain=gain)
        if text is not None and router is not None:
            return route(runtime, router, text, tokenize)
        return fit_router(runtime, cfg, positive, negative, tokenize, layer=layer)

    def unicron_prepend_layers(self, weights, cfg, n=2, intermediate=128):
        """GIVE ANY MODEL A leCORE LAYER, without knowing anything about it.
        Moose's architecture: a custom FIRST layer (BIOS -- whatever is needed so leCore
        can run), a SECOND layer where leCore lives, and the third layer is where the
        original model begins. Rather than making leCore work with every architecture in
        the world, bring the layer with you.
        IT IS VIABLE AND IT IS STANDARD PRACTICE UNDER OTHER NAMES. Adapters (Houlsby and
        everything since) require "a near-identity initialization" so the base is
        unaffected -- this project's own rule that a capability arrives OFF. Invertible
        adapters are placed "after the input embedding layer, i.e. BEFORE the first
        Transformer layer" -- Moose's layer 1, in the literature. And mergekit ships
        "frankenmerging, layer stacking, model surgery" with a passthrough method built
        for exactly this.
        MEASURED on our own trained model: 1, 2 and 3 prepended layers each leave the
        output BIT-IDENTICAL (max diff exactly 0, not merely small), and filling one
        demonstrably changes the output -- the slots are real and empty. A router fitted
        on PREPENDED layer 0 reads 91% train / 91% held-out and calls
        "what is the memory " -> use, plain prose -> don't.
        THE PLACEMENT LESSON, which cost a measurement: installing the IMPROVEMENT
        operator into prepended layer 1 gave ppl 7.27 -> 36.78, a catastrophe. That
        correction is fitted against LATE-layer states and belongs near the head; the
        ROUTER is fitted against EARLY states and belongs at the front. A leCore layer is
        not a place to put everything -- it is a place to put what operates on the
        representations available THERE.
        WHAT GOES WHERE: prepended layer 0 = BIOS + router (decisions); prepended layer 1
        = circuits acting on early representations; original layers untouched byte for
        byte; last layer = operators needing the finished representation.
        See holographic_prepend."""
        from holographic.io_and_interop.holographic_prepend import prepend_layers
        return prepend_layers(weights, cfg, n=n, intermediate=intermediate)

    def unicron_prefix_cache(self, runtime, max_nodes=512):
        """NEVER COMPUTE THE SAME CONVERSATION PREFIX TWICE -- and know when that pays.
        Moose runs a 0.8B on a CPU laptop and the largest waste in a conversation is not
        the arithmetic, it is that every turn RE-PREFILLS the whole history. MEASURED on a
        six-turn exchange: 489 tokens processed, 137 of them new -- SEVENTY-TWO PERCENT
        REPEATED, and the fraction grows every turn.
        A radix tree over TOKENS answers "what is the longest prefix I have already
        computed?" -- a dictionary on the whole prompt misses that turn 4 shares three
        turns with turn 3. vLLM and SGLang call this RadixAttention.
        AND THE MEASUREMENT THAT SAVED IT FROM BEING A REGRESSION: resuming replays the
        tail ONE TOKEN AT A TIME while a fresh call prefills in one batched pass, and
        stepping is 5.8-6.6x slower PER TOKEN on this runtime. Saving 72% of the tokens
        was a NET LOSS in wall clock -- 0.124s against 0.088s. The cache now MEASURES its
        own step cost at construction and resumes only when (tail x step_cost) beats a
        fresh prefill, so it declines when declining is right and is never slower.
        Accuracy when it does resume: matches a full recompute to 8.9e-15, which is float
        association order, not error -- asserting BIT-identity failed a correct cache.
        WHAT WOULD MAKE IT A REAL SPEEDUP: prefilling the resumed tail in a BATCH rather
        than stepping it, which needs forward() to accept an initial state. That is the
        concrete next piece of work and it is exactly what vLLM's chunked prefill does.
        See holographic_session.PrefixCache."""
        from holographic.caching_and_storage.holographic_session import PrefixCache
        return PrefixCache(runtime, max_nodes=max_nodes)

    def unicron_state_io(self, state=None, data=None, memory_only=True):
        """WHAT A HARNESS MUST STORE SO leCORE'S MEMORY SURVIVES -- and it is 63 KB.
        Moose: file IO does not belong in a model, so how does the adapter PERSIST what
        it accumulates, and what must be exposed for an external harness to store it?
        THE ANSWER WAS ALREADY IN THE ARCHITECTURE. leCore accumulates in the
        linear-attention RECURRENT STATE -- the S matrix a gated-delta layer carries token
        to token. MEASURED on our own model:
                tokens      GDN state      KV cache
                    16        63.0 KB        16.4 KB
                  1024        63.0 KB      1048.6 KB
        THE HOLOGRAPHIC MEMORY IS CONSTANT. A bundle is a sum and a sum has one shape, so
        it does not grow with the conversation while the KV cache grows linearly. That is
        the whole reason to put memory there rather than in context.
        SO THE CONTRACT IS SMALL: a harness that can save and restore recurrent state
        already persists leCore's memory. Harnesses running Mamba, RWKV or Qwen3.5-style
        hybrids ALREADY DO THIS -- a recurrent model is unusable without it, and llama.cpp
        calls them session files. We are not asking for a new capability, only to be told
        where it is.
        EXPOSED: export_memory/import_memory (the fixed-size accumulator alone -- a
        conversation's KV is disposable because it rebuilds from the text, the fold over
        everything seen is not), export_state/import_state (everything, exact), and a
        format tag so a blob written today is REFUSED rather than misread tomorrow.
        VERIFIED: a restored state continues the sequence with error EXACTLY 0.0, the
        memory blob is 62.1 KB against 104.1 KB for the full state, and a blob whose
        shapes do not match this model is refused rather than broadcast into place --
        because a foreign state broadcast into position produces fluent nonsense, which
        is the most expensive failure mode this project knows.
        See holographic_stateio."""
        from holographic.caching_and_storage.holographic_stateio import (
            export_memory, import_memory, export_state, import_state, sizes)
        if data is not None:
            return (import_memory(state, data) if memory_only
                    else import_state(state, data))
        if state is not None:
            return (export_memory(state) if memory_only else export_state(state))
        return sizes

    def unicron_reserve_keys(self, dim=None, n_slots=4, seed=0, keys=None,
                             reserved=None, enforce=False):
        """PERMANENT MEMORY IN A RECURRENT STATE, by reserving a key direction.
        A marker written into a gated-delta state vanished within 1,024 tokens and I had
        THREE explanations, all measured, all WRONG: decay did not do it (A_log=-9 gives a
        5,617-token half-life while the signal fell 300x by 1,024); the erase gate did not
        (zeroing beta moved 0.00364 to 0.00293); dilution did not (the ABSOLUTE signal
        fell 5.38 -> 0.00006 while the state norm plateaued).
        THE ANSWER WAS IN THE UPDATE RULE THE WHOLE TIME:
            S <- a * S (I - beta k k^T) + beta v k^T
        THE ERASE IS DIRECTIONAL. It removes only the component along the CURRENT key. A
        memory is not lost to time or volume -- it is OVERWRITTEN by later writes whose
        keys overlap its own. Random keys in D dimensions overlap by ~1/sqrt(D): small per
        step, fatal over a thousand.
        SO RESERVE A DIRECTION. MEASURED at D=64, recall cosine of a marker from step 0:
            tokens after      random keys      keys ORTHOGONAL to the marker
                      32           0.0042                             1.0000
                     512           0.2084                             1.0000
                    2048          -0.0811                             1.0000
        And in the full delta-rule state, 4 memories survive 2,048 unrelated writes at
        cosine 1.0000 with enforcement, and are destroyed (-0.12..0.12) without it.
        THIS IS THE DEMOSCENE MOVE -- reserve a channel and route everything else around
        it -- and it is also Kanerva's: a distributed memory works because addresses are
        near-orthogonal, and the failure mode is ADDRESS COLLISION, not capacity.
        THE PRICE: a reserved direction is one fewer dimension for the model, and the
        reservation must be ENFORCED -- orthogonalise() projects other keys off it and
        collision() measures the overlap (1.6e-16 after, 0.407 before) rather than
        assuming it. See holographic_keyreserve."""
        from holographic.caching_and_storage.holographic_keyreserve import (
            reserve, orthogonalise, collision)
        if keys is not None and reserved is not None:
            return (orthogonalise(keys, reserved) if enforce
                    else collision(keys, reserved))
        return reserve(dim, n_slots, seed=seed)

    def unicron_install_lecore(self, weights, cfg, runtime, fit_ids, eval_ids,
                               tokenize=None, passages=(), router_positive=(),
                               router_negative=(), n_registers=16, prepend=2,
                               seed=0, progress=None):
        """INSTALL leCORE INTO A MODEL. The assembly of everything this arc measured.
        VERIFIED END TO END on our own trained model -- six components, each guarded:
            prepend       2 layers added, output BIT-IDENTICAL (max diff exactly 0)
            boot_record   row 255, perplexity +0.000%, 4 bits/slot so it survives bf16
            registers     16 reserved key directions, 112 of 128 dims left to the model
            router        prepended layer 0, 91% HELD-OUT accuracy, installed as a GATE
            memory_index  24 passages in rows the eval text never uses, +0.000%
            improvement   step 128 chosen by measuring, -0.258%
        RESULT: 6 layers (was 4), perplexity 7.2659 -> 7.2471 BETTER, repetition
        0.43 -> 0.35, boots as 'leCore', and 16 registers survive 1024 unrelated writes
        at cosine >0.99, 16/16. Written to disk as an ORDINARY checkpoint it reloads at
        6 layers, still boots, and retrieves 23/24 passages from partial cues.
        DELIBERATELY NOT INSTALLED: facts in head rows. They recall 3 of 5 and cost 0.78
        perplexity that would not move for clamping, row choice or ordering -- the same
        facts in REGISTERS recall 5 of 5 at ZERO cost. A capability with a better home
        does not get installed in the worse one just because the code exists.
        EVERY STEP IS GUARDED and a regression is REVERTED, because this pipeline once
        shipped a model whose perplexity went 16.2 to 190,391 with a resident list printed
        underneath. See holographic_install_lecore."""
        from holographic.io_and_interop.holographic_install_lecore import install
        return install(weights, cfg, runtime, fit_ids, eval_ids,
                       tokenize=tokenize, passages=passages,
                       router_positive=router_positive,
                       router_negative=router_negative,
                       n_registers=n_registers, prepend=prepend, seed=seed,
                       progress=progress)

    def unicron_write_policy(self, runtime, text, tokenize, n_slots=16,
                             min_nats=None):
        """WHAT DESERVES ONE OF THE PERMANENT REGISTERS -- the last gap, closed.
        leCore could hold 128 memories forever at fixed cost and had NO POLICY for
        filling them, which is an empty filing cabinet.
        WHAT THE FIELD DOES, checked first: Google's Titans learns to memorise at test
        time using a SURPRISE metric -- the gradient of the memory's associative loss with
        respect to the input -- plus momentum and an adaptive forget gate. Their stated
        weakness is that the gradient "can become extremely small after several surprising
        steps". MIRAS generalises it.
        OUR PROBLEM WAS SHARPER: raw surprise fired on NOISE. The most surprising
        characters in real prose were 'a4.*i,rgol5*pk6&kW' -- punctuation, digits and an
        encoding artifact. That policy fills 128 permanent registers with mojibake.
        MEASURED, top-30 selections scored for content:
            surprise, per-character MEAN     16/30
            x local recurrence               11/30  WORSE -- frequency measures
                COMMONNESS, so it promotes "the" and "a". Kept as a negative.
            x TF-IDF                         19/30  better, filler still leaks
            SURPRISE SUMMED OVER THE SPAN    30/30
        AVERAGING WAS THE BUG, and the fix is not a trick but the correct quantity.
        Surprise is measured in NATS; information has an AMOUNT. A five-character word
        carrying 4 nats each carries TWENTY, while a stray byte carries eight. A mean is a
        RATE, and normalising by length threw away exactly the thing being measured.
        THE DEMOSCENE FRAMING that pointed at it: keep what costs the most to REGENERATE.
        Total surprise IS that cost -- the nats you would have to supply to reconstruct
        the span. Selected from real prose: ISA_REVERSIBLE, holographic_reversible,
        reversibility, superposition, summands -- no filler in the top thirty.
        And it costs ONE SUBTRACTION from logits the head already produced: no gradient
        and no second model, because our memory is a fold rather than a trained module.
        See holographic_writepolicy."""
        from holographic.agents_and_reasoning.holographic_writepolicy import select
        return select(runtime, text, tokenize, n_slots=n_slots,
                      min_nats=min_nats)

    def unicron_early_exit(self, runtime, weights, cfg, ids, layer=None,
                           fit_ids=None, threshold=0.95, calibration=None):
        """STOP CLIMBING WHEN THE ANSWER IS ALREADY DECIDED -- shortcuts through the layers.
        Moose, looking at the usual LLM diagram: "all these lines connecting at different
        spots along some vertical lines, which I guess are layers... I feel like we can
        speed that up and offer shortcuts on that level." Exactly right, and measurable.
        THE MODEL RUNS EVERY LAYER FOR EVERY TOKEN whether or not the answer changed.
        Reading the residual stream through the output head at each depth:
            after layer 0   29.0% of tokens already match the FINAL prediction
            after layer 1   44.1%       after layer 2   78.4%      after layer 3   88.2%
        Four out of five tokens are done by the halfway point; the rest of the stack
        confirms what is already true, at full cost.
        THE HARD PART IS KNOWING WHICH, and a raw confidence read CANNOT: a mid-layer
        stream through the final head gives probabilities of 0.007 to 0.026 on every
        token, because the head was trained on the scale of the LAST layer. ONE
        TEMPERATURE PER LAYER, fitted once offline so that mean confidence equals measured
        accuracy, fixes it -- fitted 21.0 here.
        HELD-OUT, exiting at layer 2 of 4:
            confidence>0.50   85% exit, 86.5% correct, 21% compute saved
            confidence>0.80   60% exit, 93.5% correct, 15% saved
            confidence>0.99   30% exit, 98.0% correct,  7% saved
        A DIAL, NOT A PROMISE -- accuracy and saving trade, and the caller picks.
        AND IT PAYS MORE ON A REAL MODEL: saving is (layers skipped / total), so 4 layers
        exiting at 2 caps at 25%, while 24 layers exiting at 12 saves 50% on every token
        that exits -- which on CPU is exactly where it is felt.
        It changes nothing, needs no training, and is EXACT for tokens that do not exit.
        See holographic_earlyexit."""
        from holographic.io_and_interop.holographic_earlyexit import (
            calibrate, exit_plan)
        L = int(int(cfg["n_layers"]) // 2 if layer is None else layer)
        cal = calibration or calibrate(runtime, weights, cfg,
                                       fit_ids if fit_ids is not None else ids,
                                       L)
        return exit_plan(runtime, weights, cfg, ids, cal, threshold=threshold)

    def unicron_adapt(self, weights, tokenizer_dir=None):
        """READ A MODEL WE HAVE NEVER SEEN, FROM ITS TENSORS ALONE.
        Moose: Unicron should install leCore into ANY model, and since we already demux and
        decompose UNLABELED DATASETS this should be easier. The framing is the useful part
        -- A CHECKPOINT IS AN UNLABELED DATASET. A few hundred arrays with names someone
        else chose, and every question about it (which axis is carrier, which is payload,
        where does structure repeat) is one leCore already answers for unlabeled data.
        WHAT IT RECOVERS WITHOUT A CONFIG:
            depth       the numeric field that REPEATS in tensor names
            width       the MODAL dimension -- a hidden size touches nearly every tensor
                        while head dims and intermediate sizes touch a subset
            head        2-D, one axis hidden, the other much larger
            tied        is there a separate lm_head tensor at all
            free rows   the tokenizer's added_tokens, when a tokenizer is present
        VERIFIED ON THREE FAMILIES IT HAD NEVER SEEN, config withheld:
            llama       8/8 layers, 512/512 hidden, 32000/32000 vocab, untied
            gpt2        12/12,      768/768,        50257/50257,       tied
            qwen3.5-vl  24/24,      1024/1024,      248320/248320,     tied
        and on the real bench model it matched a config.json it never read, confidence
        1.00. THE VISION TOWER DID NOT CONFUSE THE WIDTH: its 96 appears in 3 tensors
        against 1024 in 121, which is exactly why the modal dimension is the right signal.
        IT REPORTS CONFIDENCE, NOT A VERDICT. Shape inference is a strong prior, not a
        proof -- a model whose width equals its head count, or which numbers layers in a
        different field, will be read wrongly. Every field comes back with the EVIDENCE
        that produced it, and confidence drops to 0.30 on a checkpoint with no structure
        rather than guessing. A wrong guess that announces itself is recoverable; one that
        does not is the most expensive failure this project knows.
        See holographic_adapt."""
        from holographic.io_and_interop.holographic_adapt import infer
        return infer(weights, tokenizer_dir=tokenizer_dir)

    def unicron_self_write(self, runtime, weights, cfg, ids, layer=None,
                           mode="entropy"):
        """THE MODEL DECIDING WHAT TO STORE, IN ITS OWN FORWARD PASS.
        The largest item on the list of things an installed model still could not do:
        write to its own registers. Every register in every test was written from
        OUTSIDE, which makes a memory a filing cabinet with no clerk.
        THE REFRAME THAT DISSOLVED IT: look at the update rule again --
            S <- a S (I - beta k k^T) + beta v k^T
        THE MODEL ALREADY WRITES ON EVERY TOKEN. Writing was never missing. What was
        missing is CHOOSING THE KEY, and a key is a linear map of the state, which is a
        matrix, which installs like everything else.
        MEASURED, held out: a linear readout of the state predicts its OWN ENTROPY at
        r=0.814 and finds 71% of the top decile against 10% chance; it predicts the
        surprise of the token JUST CONSUMED at r=0.605 (53%); and it predicts the surprise
        of the NEXT token at only r=0.487, which it must, because a state cannot know what
        will surprise it.
        THREE FAILURES ON THE WAY, all kept:
          A BLENDED KEY DESTROYS THE RESERVATION -- (1-g)*ordinary + g*slot is not
            orthogonal to the other slots for any g between 0 and 1, and a stored value
            fell to cosine 0.525. A hard switch with the ordinary branch PROJECTED OFF the
            reservation is required.
          ONE SLOT IS A LATCH, NOT A MEMORY -- 79 of 700 positions routed to slot 0 and
            every one overwrote the last. The slot must be chosen by CONTENT.
          AND SLOT CHOICE COLLAPSES WITHOUT CENTRING -- argmax over R @ h is dominated by
            the component every state shares: 64 slots used SIX, busiest taking 54 of 79.
            Centred, 15 distinct with busiest 19. That is the THIRD place in this arc
            where centring was the fix.
        RESULT: 11% of positions route to a reserved slot, spread across 15 registers, and
        a value landing in one survives 512 writes to the OTHERS at COSINE 0.995.
        WHAT IT IS NOT: a linear readout stores what it was fitted to call surprising, so
        the model remembers UNUSUAL things rather than IMPORTANT ones. In text those
        overlap, which is why it works; they are not the same thing.
        See holographic_selfwrite."""
        from holographic.caching_and_storage.holographic_selfwrite import (
            fit_novelty, key_for, slot_for)
        return fit_novelty(runtime, weights, cfg, ids, layer=layer, mode=mode)

    def unicron_sequence(self, dim=None, seed=0, symbols=None, seq=None,
                         P=None, trace=None, position=None, codebook=None):
        """ORDER AND HIERARCHY IN THE WEIGHTS -- what circulants forbid.
        leCore states the bound as a theorem (hypervector_layer): a hypervector used as an
        operator is ALWAYS THE ABELIAN IDEAL, because bind is a circular convolution and a
        convolution algebra can only represent an abelian group. Verified: circulant(a)
        and circulant(b) commute to 1.4e-14, and even a ROLL commutes because a roll IS
        the circulant of a basis vector -- my first attempt to break commutativity picked
        one and proved nothing.
        SO ORDER CANNOT COME FROM ANOTHER VECTOR. It needs a different OPERATOR, and a
        random permutation is one: 6.17 non-commutativity against a circulant, still just
        a matrix, so it installs identically.
        THE ENCODING is Plate's: trace = P^0 a + P^1 b + P^2 c, each item permuted by its
        POSITION, and reading position j is P^-j then cleanup -- an un-permute and an
        argmax, both of which a layer already does.
        MEASURED at D=256: 40 of 40 three-item sequences read back IN ORDER, and
        store([a,b,c]) against store([c,b,a]) is cosine 0.42 where a PLAIN BUNDLE GIVES
        EXACTLY 1.0 because addition commutes. AND IT RUNS IN THE MODEL: the inverse
        permutation installed as MLP neurons, the codebook in head_key rows, all three
        positions read back correctly from the model's own logits.
        THE COST: one operator PER POSITION, so a depth-k reader is k circuits. That is
        the price of leaving the abelian ideal, and the alternative is not a cheaper
        non-commutative bind -- it is not having order at all. See holographic_seqbake."""
        from holographic.io_and_interop.holographic_seqbake import (
            permutation, store_sequence, read_position, unpermute_operator)
        if trace is not None and position is not None:
            return read_position(trace, position, P, codebook)
        if symbols is not None and seq is not None:
            return store_sequence(symbols, seq, P if P is not None
                                  else permutation(len(symbols[0]), seed))
        return permutation(dim, seed=seed)

    def unicron_hlb(self, key=None, dim=None, seed=0, x=None, y=None,
                    trace=None, operator=False):
        """BINDING AS A VECTOR, NOT A MATRIX -- a thousand times smaller.
        install_op stores a full D x D circulant for ONE bind operator: 1,048,576
        parameters at Qwen's width. Alam et al. (NeurIPS 2024, arXiv 2410.22669) derive a
        VSA from the WALSH-HADAMARD transform instead of the Fourier transform, where
        binding is ELEMENTWISE in the transform domain -- so the operator is a VECTOR of
        1,024. And elementwise multiply is precisely what an MLP GATE already computes.
        THE TWO STABILISERS ARE NOT OPTIONAL, measured at D=512:
            naive Hadamard binding, gaussian keys        1 of 8 recovered
            + MiND initialisation                        2 of 4, still unstable
            + THE PROJECTION STEP                        8/8, 16/16, 24/24
        Projection puts every key at magnitude EXACTLY 1.0 in the Hadamard domain against
        0.0014 without it, so unbinding divides by a SIGN and cannot blow up. That one
        step is the difference between 1 of 8 and 24 of 24, and the selftest pins the
        NEGATIVE as well as the positive so nobody drops it.
        PAST THAT IT DEGRADES AS A LAW, not a cliff -- 31 of 32, 40 of 48 -- so capacity
        is the load ratio m/D, exactly as bundle_capacity establishes for every other VSA
        here.
        VERIFIED INSTALLED: an HLB operator expanded to a matrix computes bind on the live
        residual stream at COSINE 1.000000, identical to a circulant, while being defined
        by D numbers instead of D squared.
        WHAT IT DOES NOT CHANGE: HLB COMMUTES, like every hypervector operator, so the
        abelian bound still holds and order still needs a PERMUTATION as a second operator
        (unicron_sequence). A cheaper bind is not a non-commutative one.
        leCore already shipped `wht`, so the transform was here the whole time.
        See holographic_hlb."""
        from holographic.sampling_and_signal.holographic_hlb import (
            project, bind, unbind, as_operator, mind, parameter_cost)
        if operator and key is not None:
            return as_operator(key, dim)
        if trace is not None and key is not None:
            return unbind(trace, key)
        if x is not None and y is not None:
            return bind(x, y)
        if dim is not None and key is None:
            return project(mind(dim, seed=seed))
        return parameter_cost(dim or 1024)

    def unicron_model_vault(self, objects=None, data=None, entry=None,
                            kind=None):
        """A TRAINED MODEL GOES IN, A RUNNABLE MODEL COMES BACK.
        Moose asked that trained models store in leCore's holographic storage like anything
        else, and recall and RUN on demand. The audit found the pieces built and never
        joined: holographic_container is a typed-section format that keeps arrays plus
        arbitrary JSON verbatim, and every leCore trained object -- an HDRIFT drift model,
        a register reservation, a codebook -- is a few arrays plus the numbers needed to
        rebuild its encoder.
        WHAT REGENERATES IS NOT STORED, which is the demoscene rule and the whole saving.
        An HDRIFT model trained on 400 points is (mu, nu) -- 6,144 learned values -- plus
        an encoder that regenerates EXACTLY from FOUR NUMBERS (dim, bounds, bandwidth,
        seed). MEASURED: stored in 48.3 KB against 49.2 KB of learned moments, recalled,
        and producing a drift field IDENTICAL to the original at max diff 0.0. A 16-slot
        register reservation round-trips from a SEED ALONE, with no arrays in the file.
        NOT a checkpoint format for foreign models -- those go through unicron_model_store,
        which hands out an ordinary safetensors directory. This is for leCore's OWN
        trained objects, which are hypervectors and therefore already in the format the
        container was built for. See holographic_modelvault."""
        from holographic.caching_and_storage.holographic_modelvault import (
            store, recall, store_drift, rebuild_drift, store_registers,
            rebuild_registers)
        if entry is not None:
            return (rebuild_registers(entry) if kind == "registers"
                    else rebuild_drift(entry))
        if data is not None:
            return recall(data)
        return store(objects)

    def unicron_program_library(self, machine=None, dim=None, context=None,
                                library=None, k=3, program=None,
                                faculties=None, procedures=None,
                                vocabulary=False):
        """VSA PROGRAMS THAT FIND THEMSELVES WHEN THE CONTEXT CALLS FOR THEM.
        Moose asked whether there are VSA programs we can run on the fly, self-contained,
        composable and discoverable from context. Rule 0 answered most of it: leCORE
        ALREADY HAS THE PROGRAMS. HoloMachine calls itself "a formatted holographic drive
        that can store and execute stored programs" with FOURTEEN OPCODES -- LOAD, STORE,
        BIND, BUNDLE, PERMUTE, RECALL, PUSH, POP, APPLY, CALL, IFMATCH, ITERATE, REPEAT,
        HALT -- which is the VSA algebra plus control flow. `assemble` turns
        (opcode, operand) pairs into ONE HYPERVECTOR; `define` names a procedure other
        programs CALL; `APPLY` reaches any named faculty. VERIFIED: a program run inline
        and the same program reached through CALL give IDENTICAL accumulators to 1e-6, so
        composition is EXACT rather than approximate.
        SO SELF-CONTAINED AND COMPOSABLE WERE ALREADY TRUE. What was missing is DISCOVERY
        -- a library nobody can find by describing their situation is the same failure
        Rule 0 exists to prevent for capabilities.
        THIS ADDS IT with the mechanism already in the engine: a program is indexed by a
        BUNDLE-over-words address of its description, matched by cosine, exactly as
        memsearch indexes passages -- so a partial description still lands. MEASURED: 3 of
        3 plain-language situations find the right program, an unrelated context correctly
        ABSTAINS rather than running its best guess, and the whole library vaults with
        every ADDRESS REGENERATED from its description rather than stored.
        AND THE OPERANDS ARE NOT FREE STRINGS. The VM cleans every operand up to the NEAREST
        atom of that opcode's type, so a made-up name becomes whatever was closest and NOTHING
        RAISES -- my first program assembled with invented operands and decoded as
        ('LOAD','f'), ('BIND','d'). The real vocabulary is data a-f, registers R0-R7, counts
        1-8, faculties cleanup/denoise/matmul (plus any the host supplies), and procedure names
        for CALL/ITERATE. Pass `program=` to CHECK one before assembly, or `vocabulary=True` to
        read the whole codebook.
        SEMANTICS VERIFIED against the algebra, not merely "it ran": LOAD, BIND, BUNDLE,
        PERMUTE, STORE/RECALL and PUSH/POP all match to cosine 1.000000; IFMATCH genuinely
        BRANCHES (1.0000 on a match, 0.0183 on a miss); ITERATE runs a named procedure to a
        FIXED POINT; and REPEAT is exact at counts 1-4 -- but ONLY in its correct form, REPEAT n
        followed by CALL. Written as REPEAT n; PERMUTE it silently gives cosine 0.018 to the
        intended result, which is exactly the trap the checker exists for.
        THE HONEST LIMIT: discovery is by DESCRIPTION SIMILARITY, not by understanding what
        a program does. A badly described program is unfindable, exactly as a catalog entry
        with poor aliases is unreachable -- which is why skill_lint exists.
        See holographic_proglib."""
        from holographic.agents_and_reasoning.holographic_proglib import (
            ProgramLibrary, check, VOCABULARY)
        if program is not None:
            return check(program, faculties=faculties or (),
                         procedures=procedures or ())
        if vocabulary:
            return dict(VOCABULARY)
        if library is not None and context is not None:
            return library.find(context, k=k)
        return ProgramLibrary(machine, dim=dim)

    def unicron_device(self, runtime=None, want="auto", ids=None):
        """RUN THE MODEL ON WHATEVER HARDWARE IS THERE, AND PROVE IT AGREES.
        An LLM is usually run on a GPU. This runtime was pure host NumPy, so on a machine
        with a card it left the ENTIRE FORWARD PASS on the CPU -- the FLOPs are in the
        model, and leCore's WGSL path only covered leCore's OWN kernels.
        leCORE ALREADY HAD THE SWITCH and the runtime never asked for it:
        `array_module()` returns cupy when a device is present AND the policy allows and
        numpy otherwise, `gpu_available` / `backend_status` say what is there, and
        `resource_policy(gpu=...)` decides. So this is not a GPU port -- it is the missing
        WIRE between a switch that existed and a forward pass that ignored it.
        RESIDENCY IS THE POINT, and the backend's own docstring says why: every
        host-to-device transfer costs, and a small per-call op loses to the transfer that
        feeds it. WEIGHTS MOVE ONCE AND STAY; ids and logits are small and cross per call.
        A runtime that moved weights per layer would be SLOWER on a GPU than on a CPU and
        would look like the GPU was at fault.
        ASKING FOR A GPU THAT IS NOT THERE IS NOT AN ERROR -- it reports cpu and runs,
        because a pipeline that dies on a laptop is worse than one that is merely slower.
        TESTED WITHOUT A GPU, because an untested path rots: the selftest substitutes a
        fake device module and drives the whole dispatch, making 50 weight tensors
        resident and returning output BIT-IDENTICAL to the host path.
        WHAT IS NOT CLAIMED: no speedup, because none was measured on real hardware.
        `gpu_crossover` exists to find where a device starts winning and needs a real
        adapter to answer. The claim here is PARITY -- the same numbers either way -- which
        is what makes the speed question safe to ask later. See holographic_devicerun."""
        from holographic.io_and_interop.holographic_devicerun import (
            status, place, parity)
        if runtime is None:
            return status()
        if ids is not None:
            return parity(runtime, ids)
        return place(runtime, want=want)

    def unicron_vm_install(self, unit=None, table=None, rule=None, A=None,
                           k=1, chain=None, U=None, V=None, step=None):
        """WHICH OF leCORE'S VIRTUAL MACHINE FITS INSIDE A MODEL, AND WHICH CANNOT.
        Moose asked for the virtual GPU and the L1/L2/L3/L4/RAM hierarchy installed INSIDE
        the model. Rule 0 found the whole thing built AND already measured:
        holographic_machinemodel is "THE leCORE VIRTUAL MACHINE, named and measured" with
        SEVENTEEN units -- simt_width, simd_lanes, gather_unit, texture_unit, rt_core, rng,
        scheduler, occupancy_gate, kernel_fusion, operator_power, and tiers t0 to t6.
        AND IT ALREADY REFUTED THE OBVIOUS FRAME. The textbook ladder (registers, L1, L2,
        L3, RAM, each ~10x slower) is WRONG here, measured per scalar access: RAM indexing
        132 ns is AS FAST AS the compiled tier, a MarginCache hit is 26x SLOWER than RAM,
        and a texture fetch 2,850x slower. A latency-ordered hierarchy would say never use
        any of them, which is nonsense -- NONE OF THEM ARE SCALAR UNITS. Every one is a
        BATCH unit whose per-access cost collapses with N, and gather's marginal cost is
        CONSTANT IN N: 8 lookups or 2,048, still about 4 microseconds, a measured 182,010x.
        WHAT THAT MEANS FOR INSTALLING: a layer computes matmul, elementwise, add. So
        6 OF 17 UNITS INSTALL and 11 DO NOT.
            INSTALLS   gather_unit (T @ r is ONE matvec -- verified computing on the live
                       residual stream at COSINE 1.000000, and it is the unit whose cost
                       is already constant in N, so a layer IS a constant-cost gather);
                       operator_power (A^k is a MATRIX whatever k is -- A^4 costs the SAME
                       128 neurons as A^1, the loop folded at bake time); texture_unit;
                       simd_lanes and simt_width (already what a layer does); rng.
            CANNOT     rt_core (an unbounded loop with a data-dependent exit); scheduler,
                       occupancy_gate, kernel_fusion (control over WHICH work runs -- a
                       gate attenuates output but cannot skip compute, which is why
                       exit_after lives in the RUNTIME); tiers t0-t6 (eviction,
                       compression, durability are STATE OVER TIME, and the model-side
                       equivalent already exists as the register file).
        AND FOUR OF MY REFUSALS WERE WRONG. Moose pushed back on leaving units out for want
        of an immediate use, and the demoscene answer is decisive: a demo has NO OS and NO
        allocator, and demosceners wrote those anyway, in 4KB, because you cannot call what
        is not there. Re-walked against the engine's OWN five levers, and 6 of 17 became
        10 of 17:
          rt_core            LEVER 5, tile under an orchestrator. A LAYER has no loop but
                             the TOKEN LOOP does -- one sphere-trace step installs at
                             cosine 1.000000 and iterating it converges, residual
                             5.392 -> 0.00295 over 12 steps. The route the resonator took.
          kernel_fusion      LEVER 1, bake once. Fusing A then B IS the product B@A to
                             5.6e-16, and it SAVES A LAYER -- two installs become one
                             operator with the same neuron count. This unit PAYS to
                             install rather than merely fitting.
          t4_compressed_ram  a LowRankField IS U@V, a matrix. 2,048 parameters against
                             16,384 dense -- the compression is the POINT.
          t2_baked_grid      the BAKE is a table and sampling it is a matvec. I had
                             conflated the DATA with the CACHE POLICY around it.
        WHAT REMAINS OUT is now stated rather than shrugged at: scheduler and
        occupancy_gate install their DECISION (the router already does) but not the ACT of
        skipping, which is why exit_after lives in the runtime; and the t0/t1/t3/t5/t6
        tiers are eviction, lifetime and durability -- STATE THAT CHANGES OVER TIME, which
        a forward pass does not have.
        THE BOUNDARY IS STRUCTURAL, not unfinished: a forward pass is arithmetic, so its
        arithmetic installs and its control and storage do not. Naming which side each unit
        falls on is the deliverable, so nobody re-tries the impossible half.
        See holographic_vminstall."""
        from holographic.io_and_interop.holographic_vminstall import (
            classify, installable_units, gather_matrix, power_matrix, fuse,
            low_rank, token_step)
        if chain:
            return fuse(*chain)
        if U is not None and V is not None:
            return low_rank(U, V)
        if step is not None:
            return token_step(step)
        if table is not None:
            return gather_matrix(table, rule)
        if A is not None:
            return power_matrix(A, k)
        if unit is not None:
            return classify(unit)
        return {"installable": installable_units(), "all": classify()}

    def unicron_install_plan(self, ops=None, iteration=None,
                             max_condition=1e6):
        """HOW SHOULD THIS BE INSTALLED: fused, at its limit, per token, or in stages?
        Moose asked what the new machinery unlocks. It is bigger than four reclassified
        units, because two of them change the ECONOMICS of installing rather than adding
        one more installable thing.
        A CHAIN NOW COSTS WHAT ONE OPERATOR COSTS. `fuse` folds a chain into a single
        matrix, so DEPTH IS FREE. Measured on the live residual stream: 1, 4, 16 and 32
        operators all install in 128 NEURONS at COSINE 1.000000. Thirty-two operations for
        the price of one, exact. Anything leCore expresses as a SEQUENCE of linear
        transforms -- transform_bank's apply_chain, a shader pipeline's stages, a VSA
        program that is all BIND and PERMUTE -- installs WHOLE. The layer budget stopped
        being the constraint.
        AND A CONVERGING ITERATION INSTALLS AT ITS ANSWER. leCore already had
        `accelerate_convergence` -- jump to a solver's limit when convergence is lawful --
        and for a LINEAR iteration the limit IS a matrix: x <- Ax + b converges to
        (I-A)^-1 b. MEASURED: 200 iterations agree with the closed form at COSINE
        1.000000, and that limit installs and computes live at COSINE 1.000000 in 128
        neurons. So every faculty that is "iterate a projection" -- and this project's own
        note says IK, PBD, PnP and the resonator are that same thing in different costumes
        -- installs AT ITS CONVERGED ANSWER WITH NO LOOP. The loop was never the
        requirement; it was one way to reach the fixed point.
        WHEN THE ITERATION IS NOT LINEAR OR NOT CONTRACTING, token_step carries one step
        per token -- the resonator's route, now the FALLBACK rather than the only option.
        AND IT REFUSES RATHER THAN LYING: fusion multiplies CONDITION NUMBERS along with
        matrices, so a chain of harmless operators can fuse into an ill-conditioned one
        that is right in exact arithmetic and wrong in float32. `fusible` checks and
        returns "stages" instead. A divergent iteration returns token_step, never a
        plausible-looking limit matrix. See holographic_unlocked."""
        from holographic.io_and_interop.holographic_unlocked import plan
        return plan(ops=ops, iteration=iteration, max_condition=max_condition)

    def unicron_install_order(self, steps=None, step=None, before=None,
                              after=None):
        """WHICH INSTALL STEPS COLLIDE, AND WHAT ORDER IS SAFE.
        install_lecore ran its steps in the order they were written, and one collision was
        found BY ACCIDENT: growing an HRNN channel AFTER writing the boot record made the
        model report booting as NONE, because a manifest too big for one embedding row
        SPILLS across the surface weights and the channel edit corrupted the payload.
        boot() failed with "substrate hash mismatch" while every other step reported
        success. The fix -- write the boot record last -- was right and reached expensively.
        leCORE ALREADY HAD THE GENERAL TOOL: `conflict_graph(item_keys)` builds the graph
        where two tasks are adjacent iff they share a resource, key-first so the cost is
        the sum of squared key degrees rather than O(n^2). So the ordering is DERIVABLE
        from what each step WRITES rather than remembered.
        AND DECLARING THAT HONESTLY IS THE HARD PART, which my first attempt proved: I
        guessed `improvement` writes head rows, the conflict graph dutifully flagged a
        collision with `memory_index`, and MEASUREMENT SAID 0 OF 256 HEAD ROWS CHANGE --
        it writes MLP weights. THE CONFLICT WAS IN MY DECLARATION, NOT THE CODE. A resource
        table written from memory produces confident false alarms, so `verify_declaration`
        re-checks a step against a real model instead of trusting the table.
        THE SPILL RULE is the one that actually bit: a step whose payload can spread across
        arbitrary weights conflicts with EVERY weight writer and must go last. That is a
        consequence of the substrate encoding, not a preference -- and it only appears when
        the manifest does not fit one row, which is WIDTH-DEPENDENT: invisible on a wide
        model, fatal on a narrow one. See holographic_installorder."""
        from holographic.io_and_interop.holographic_installorder import (
            conflicts, order, verify_declaration)
        if step is not None and before is not None and after is not None:
            return verify_declaration(step, before, after)
        if steps is not None:
            return {"order": order(steps), "conflicts": conflicts(steps)}
        return {"order": order(), "conflicts": conflicts()}

    def unicron_long_context(self, target_tokens=1e9, dim=1024, n_slots=128,
                             precision="float32", state=None, keys=None,
                             values=None):
        """CONTEXT PAST A BILLION TOKENS -- what reaches it, and what does not.
        Three mechanisms were candidates and ONE survives the arithmetic.
        THE KV CACHE IS OUT, and not narrowly: at Qwen3.5-0.8B's shapes a million tokens
        is 49 GB and a BILLION IS 49 TERABYTES. Sparse attention, eviction and compression
        change the constant, not the exponent.
        THE LADDER UNDERFLOWS FIRST, around 1e8. decay = exp(-exp(a_log)*softplus(dt_bias))
        so a half-life of D needs a_log = -ln(D), and in float32 1-decay reaches EXACTLY
        ZERO at a 1e8 half-life. Past that a rung is a pure accumulator -- infinite
        retention with no forgetting, which sounds like a win and is not, because an
        undecayed sum of a billion terms has SNR going as 1/sqrt(n).
        THE REGISTERS REACH IT, because their bound is not TIME. The delta rule's erase
        term is DIRECTIONAL -- S <- aS(I - b k k^T) + b v k^T -- so a write whose key is
        orthogonal to a reserved direction leaves it EXACTLY untouched; the projector has
        a zero there.
        SO THE LIMIT IS PRECISION, AND IT IS A CLIFF NOT A SLOPE. Measured in float32:
        1.000000 at 30,000 writes, 0.999580 at 80,000, 0.951 at 100,000, and 0.057 by
        140,000. float64 holds 1.000000 throughout. IT IS NOT DILUTION -- ||S|| stays at
        245 across the whole run, which was my first explanation and was wrong. A CLIFF IS
        MORE DANGEROUS THAN A SLOPE: a system tested at 50,000 writes reads perfect and
        fails at 140,000, one long session later.
        AND THE FIX IS DRAM REFRESH, which is the correct name for it. A DRAM cell loses
        charge and is rewritten on a schedule; a register loses its orthogonality and is
        rewritten the same way -- one delta_write per slot. MEASURED: refresh every 10,000
        writes restores COSINE 1.000000 at 6.4% overhead for 128 slots.
        WHAT "A BILLION TOKENS" HONESTLY MEANS HERE, because the phrase invites a bigger
        claim than the mechanism supports: THE MODEL DOES NOT ATTEND TO A BILLION TOKENS.
        It RETAINS a bounded number of facts -- d slots, chosen by the write policy --
        across an UNBOUNDED stream. What became unbounded is the WINDOW over which those
        slots survive, not the slot count. See holographic_billionctx."""
        from holographic.caching_and_storage.holographic_billionctx import (
            plan, refresh, refresh_interval)
        if state is not None and keys is not None and values is not None:
            return refresh(state, keys, values)
        return plan(target_tokens, dim=dim, n_slots=n_slots,
                    precision=precision)

    def unicron_self_heal(self, state=None, keys=None, codebook=None,
                          baseline_margin=None, drop=0.5, check_only=False):
        """REGISTERS THAT REPAIR THEMSELVES, WITH NO EXTERNAL COPY.
        The DRAM-style refresh in billionctx works and has a weakness: it rewrites KNOWN
        VALUES, so the harness must hold a copy of everything the register file contains.
        A memory that needs an external copy of itself is a CACHE, not a memory.
        leCore had the levers and I had not used them -- `cleanup_batch` (clean many noisy
        cues against a codebook), `decide_confidence` (top, score and MARGIN),
        `superposed_memory` (key->value and value->key), and denoise, which is the same
        operation in another costume.
        THE INSIGHT THAT REMOVES THE COPY: values are drawn from a KNOWN ALPHABET, a
        codebook is a CONSTRAINT, and a constraint IS error correction. So repair is READ,
        CLEAN UP AGAINST THE CODEBOOK, WRITE THE CLEANED VALUE BACK -- and nothing outside
        the model needs to know what was stored.
        MEASURED at float32, 8 registers, a 64-entry codebook, repairing periodically:
            healthy margin                                   0.8544
            after 140,000 interfering writes, UNREPAIRED      0.0237  (collapsed)
            after 200,000 writes WITH repair                  0.8544, 8/8 slots exact
        AND CONFIDENCE SAYS WHEN, so repair is not a blind schedule. The MARGIN collapses
        BEFORE the top score: 0.8544 healthy, 0.3692 while the top score had already
        halved to 0.5242, then 0.0342. THE TRIGGER MUST BE RELATIVE -- an absolute 0.35
        threshold called that middle stage FINE, missing the point where repair was still
        cheap. Comparing against this file's own healthy baseline catches it, which is the
        same lesson proglib learned about abstaining on score instead of margin.
        THE HONEST RESIDUAL: this repairs values that live in a CODEBOOK. A register
        holding an arbitrary vector has no constraint to correct against, and for those the
        external copy is unavoidable -- a reason to prefer codebook values wherever the
        application allows. See holographic_selfheal."""
        from holographic.caching_and_storage.holographic_selfheal import (
            health, repair, maintain)
        if check_only:
            return health(state, keys, codebook)
        if baseline_margin is not None or drop != 0.5:
            return maintain(state, keys, codebook,
                            baseline_margin=baseline_margin, drop=drop)
        return repair(state, keys, codebook)

    def unicron_actr(self, items=None, now=0.0, half_lives=None,
                     threshold=None, forget_below=None):
        """NOOA'S MEMORY RANKING, COMPUTED BY THE LADDER WE ALREADY INSTALL.
        docs/COMPETITIVE_NOOA.md checks arXiv:2607.20709 and lists six NOOA capabilities.
        FIVE ARE HARNESS FEATURES -- pass-by-reference previews, code-as-action in a
        persistent REPL, typed return validation, sandboxed execution, event history --
        and none of those live in weights. They are things a RUNNER does.
        THE SIXTH IS THE ONE WITH A NUMBER: a long-term memory subsystem with ACT-R
        ACTIVATION RANKING and DECAY-BASED FORGETTING, measured at +11.8 RHAE POINTS over
        the same agent with markdown notes. leCore was marked PARTIAL -- recall exists,
        the curation and decay did not.
        AND WE HAD ALREADY INSTALLED THE HARD PART WITHOUT NAMING IT. ACT-R's base-level
        activation is A = ln(sum_j t_j^-d) with d~0.5, A POWER LAW over how long ago each
        use was. The HRNN ladder is a sum of EXPONENTIALS at GEOMETRIC half-lives, and a
        geometric sum of exponentials APPROXIMATES a power law. Measured against t^-0.5
        over five decades:
            2 rungs   R^2 0.85055
            4 rungs   R^2 0.99282   <-- what install_lecore puts in by default
            6 rungs   R^2 0.99891
        SO THE LADDER IS ACT-R BASE-LEVEL ACTIVATION IN THE WEIGHTS, rather than in a
        SQLite file beside the agent, and the state IS the log of use times.
        THE RUNG WEIGHTS ARE NOT OPTIONAL, and this is the trap: reading the ladder with
        UNIT weights over-counts the long rungs, because every rung contributes about 1
        for an item younger than its half-life. Measured, that ranked ONE RECENT USE BELOW
        TWO OLD ONES -- inverting the entire point of a recency-weighted memory. The fit is
        closed-form least squares over log-spaced ages and the selftest pins the failure.
        WHAT IS NOT CLAIMED: NOOA's +11.8 was measured on RHAE with a full agent loop.
        Nothing here reproduces that, and leCore still has no result on any external
        agentic benchmark -- which the competitive note already says plainly. The claim is
        that the MECHANISM is present and correct. See holographic_actr."""
        from holographic.agents_and_reasoning.holographic_actr import (
            rank, forget, fit_rung_weights, base_level)
        if forget_below is not None:
            return forget(items, now, forget_below)
        if items is None and half_lives is not None:
            return fit_rung_weights(half_lives)
        return rank(items, now, half_lives=half_lives, threshold=threshold)

    def unicron_nullspace(self, runtime=None, ids=None, layer=None, delta=None,
                          ratio=1e-2, keys=None):
        """INSTALL INTO THE DIRECTIONS THE MODEL WAS NOT USING.
        From the research survey's FIRST recommendation: AlphaEdit (Fang et al., ICLR 2025
        Outstanding Paper, arXiv 2410.02355) projects a weight perturbation onto the NULL
        SPACE of the preserved-knowledge key matrix before applying it, so preserved keys
        produce unchanged output. The paper reports it boosts locate-then-edit methods "by
        an average of 36.7% with a single line of additional code for projection solely".
        WHY IT MATTERS HERE: every install in this pipeline was checked by MEASUREMENT --
        bit-identical when empty, or perplexity did not regress. That is weaker than a
        CONSTRUCTION that cannot disturb what it must not touch.
        MEASURED, the same bind operator installed three ways:
            projection            kept energy   perplexity   bind cosine
            none (raw)                   1.00       7.3772     1.000000
            drop eig > 1e-2*max          0.78       7.2820     1.000000
            drop eig > 1e-3*max          0.51       7.2790     1.000000
            (baseline, no install)          -       7.2659
        THE COST OF INSTALLING FELL SEVENFOLD, +1.53% to +0.22%, AND THE OPERATOR STILL
        COMPUTES EXACTLY at cosine 1.000000. The circuit does the same arithmetic in
        directions the model was not using.
        AND THE HONEST CAVEAT, which a small model exposes and the paper's setting hides:
        ALPHAEDIT'S GUARANTEE NEEDS AN ACTUAL NULL SPACE, and a full-rank key covariance
        has none. Measured, 600 preserved keys at width 128 gave eigenvalues spanning 2.03
        to 1.29e4 -- THE SMALLEST IS 2.03, NOT ZERO. So this computes a LOW-ENERGY
        SUBSPACE and the disturbance FALLS 3.2x rather than vanishing. The guarantee
        degrades gracefully into a reduction, and calling it a proof here would be the
        overclaim. It is a width-and-sample question: 600 keys at width 1024 leaves a real
        null space, at width 128 it does not. See holographic_nullspace."""
        from holographic.io_and_interop.holographic_nullspace import (
            guard, projector, preserved_keys, project)
        if keys is not None and delta is not None:
            P, rep = projector(keys, ratio=ratio)
            return project(delta, P), rep
        if delta is None:
            return projector(preserved_keys(runtime, ids, layer), ratio=ratio)
        return guard(runtime, ids, layer, delta, ratio=ratio)

    def unicron_state_track(self, symbols=None, transition=None, dim=None,
                            n_slots=2, seed=0, codebook=None, start=0):
        """THE ONE THING ATTENTION PROVABLY CANNOT DO, AND THE INSTALLED STATE CAN.
        Moose read that recurrent models may be more capable than transformers. The
        literature's ACTUAL claim is narrower than "RNNs beat LLMs and do not
        hallucinate", and the narrow version is the useful one because it is PROVEN:
          Merrill and Sabharwal show saturated transformers are CONSTANT-DEPTH THRESHOLD
            CIRCUITS, and constant-depth circuits cannot compute PARITY over unbounded
            input. A complexity result, not a benchmark.
          "Transformers ... specifically lack state-tracking capabilities" (arXiv
            2410.01201); "the only inference-time memory accessible to Transformers is
            their limited input window, whereas RNNs can update their internal state
            INFINITE TIMES" (arXiv 2511.10457).
        WHAT IS NOT ESTABLISHED, and this faculty does not repeat it: that recurrence
        eliminates hallucination. No paper here claims that.
        SO THE WIN IS STATE TRACKING, a structural advantage rather than a benchmark
        delta. MEASURED, parity carried in the MODEL'S OWN delta-rule state through
        interfering writes on every non-transition token:
            length     16    128   1024   8192      -> 10/10 at every length
        and a 4-state mod-4 automaton 8/8 at length 512, so it is not parity-specific. A
        tracked value survives 5,000 interfering writes; a 20,000-symbol run reads back
        correctly. LENGTH DOES NOT MATTER because the update is O(1) and the erase term is
        DIRECTIONAL.
        WHY THE HRNN IS THE RIGHT HOME: the ladder already puts decay channels in the
        weights, and a state tracker is simply the rung with decay set to NONE -- an
        accumulator, addressed through a reserved key so nothing overwrites it. Not new
        machinery; the a_log -> -inf end of a structure already installed.
        THE HONEST BOUNDARY, and it is why this is a COMPONENT and not an architecture:
        THE TRACKER MUST BE TOLD WHAT TO TRACK. Parity works because a program says
        "toggle on 1". Nothing here DISCOVERS that a task needs a counter, and the model
        does not learn to use one. State tracking becomes a capability the model CAN BE
        GIVEN, not one it acquires -- the same boundary as the write policy: mechanism
        installed, policy supplied. See holographic_statetrack."""
        from holographic.agents_and_reasoning.holographic_statetrack import (
            tracker, run_automaton)
        if symbols is None or transition is None:
            return tracker(dim, n_slots=n_slots, seed=seed)
        K = tracker(dim, n_slots=n_slots, seed=seed)
        return run_automaton(symbols, transition, K, codebook, start=start,
                             seed=seed)

    def unicron_hybrid(self, logits=None, quantile=0.90, targets=None,
                       recalled=None):
        """THE LLM AND THE HRNN EACH DOING WHAT THE OTHER STRUCTURALLY CANNOT.
        Moose asked for a hybrid with the full power of both and I had answered a narrower
        question -- what can the HRNN do that attention cannot. That is a FEATURE LIST,
        not an architecture.
        THE DEMOSCENE FRAMING IS THE RIGHT ONE: a demo does not CHOOSE between the CPU and
        the blitter. It runs each on what it is good at, and THE WIN IS IN THE HANDOFF --
        the copper list changing registers mid-frame while the blitter moves memory the
        CPU could never move in time. Neither chip does the effect. THE SCHEDULE DOES.
        SO THE QUESTION IS THE DIVISION OF LABOUR AND THE SWITCH, and both measure.
        MEASURED ON ONE 3,000-TOKEN STREAM:
            most confident quartile   mean surprise 0.746 nats
            top entropy decile        mean surprise 3.520 nats, TOP-1 7.8%
            THE SAME TOKENS, recalled from the recurrent store after every
            intervening write                                  100.0% EXACT
        A 92-POINT GAP ON IDENTICAL POSITIONS, using 64 slots for 2,999 tokens.
        AND IT IS NOT A COINCIDENCE, which is what makes it an architecture rather than a
        trick: HIGH ENTROPY MEANS LOW REDUNDANCY, and low redundancy is EXACTLY what a
        lossy predictor cannot reconstruct and EXACTLY what a store holds cheaply because
        there is little of it. The two failure modes are complementary BY INFORMATION
        THEORY.
            redundant tokens   the LLM predicts them free; storing them wastes slots
            surprising tokens  the LLM cannot predict them; the store holds them exactly
        Store everything and you need a slot per token; store nothing and you lose every
        fact. THE ENTROPY QUANTILE IS THE CORRECT PLACE TO CUT, and it is a fraction
        because slot count is the budget.
        AND THE SWITCH IS FREE: the model computes its own entropy every token as a
        by-product of producing logits, correlating 0.573 with its actual error. It does
        not need to be told where it is weak -- IT ALREADY PUBLISHES IT.
        WHAT THIS IS NOT: the model does not LEARN to consult the store, and no weight
        moves toward doing so. The handoff is a policy the harness runs on numbers the
        model supplies -- mechanism installed, SCHEDULE supplied, which is precisely how a
        copper list works and why the framing holds all the way down.
        See holographic_hybrid."""
        from holographic.agents_and_reasoning.holographic_hybrid import (
            split, compare, entropy_of)
        if recalled is not None and targets is not None:
            return compare(logits, targets, recalled)
        return split(logits, quantile=quantile)

    def unicron_runtime(self, runtime=None, cfg=None, keys=None, codebook=None,
                        store_quantile=0.90, exit_after=None, device="auto"):
        """THE LOOP THAT ACTUALLY USES WHAT WAS INSTALLED.
        A wiring audit found most of this arc's capabilities were library code NOTHING
        CALLED. Three belonged in the weights and are now installed. THE OTHER SIX WERE
        CORRECTLY OUTSIDE THE WEIGHTS AND EQUALLY UNUSED -- because being correctly outside
        is not the same as being wired, and galvatron.py's chat loop called plain forward()
        and used none of them.
        THE SCHEDULE, each step delegating to where it was measured:
            1  place the model on whatever hardware is present (devicerun)
            2  resume from a cached prefix when the tail beats a recompute (2.7x)
            3  forward, with an early-exit budget if one is calibrated
            4  read the model's OWN entropy off the logits it just produced
            5  above the quantile, consult the register store instead of generating
            6  below it, let the model generate -- cheaper AND right
            7  store what the write policy selects, by TOTAL surprise
            8  repair registers when their MARGIN falls against baseline
        STEP 4 IS WHY THIS COSTS ALMOST NOTHING: the switch is a BY-PRODUCT of producing
        logits, so the schedule is free -- the same reason a copper list is free, riding a
        signal the hardware was generating anyway.
        MEASURED end to end on 900 tokens: 90 routed to the store by the model's own
        entropy, recalled at 100% against the model's 9% TOP-1 ON IDENTICAL POSITIONS, and
        'designed' / 'These' selected as the spans worth keeping.
        WHAT IT DOES NOT DO: change a weight, learn anything, or make the model CHOOSE to
        consult the store. It is a SCHEDULE over installed mechanisms, which is the same
        boundary every capability in this arc has landed on. See holographic_lecorerun."""
        from holographic.io_and_interop.holographic_lecorerun import (
            LeCoreRuntime)
        # RETURN A HANDLE, NOT THE OBJECT. A LeCoreRuntime is not JSON
        # serialisable, so returning it directly made this capability invisible
        # to the one caller it was built for -- the exact failure
        # holographic_objectref exists to fix, and one the usage audit caught in
        # the same session that shipped the bug.
        r = LeCoreRuntime(runtime, cfg, keys=keys, codebook=codebook,
                          store_quantile=store_quantile,
                          exit_after=exit_after, device=device)
        return {"ref": self.unicron_ref(r), "device": r.device,
                "store_quantile": float(store_quantile)}

    def unicron_ref(self, obj=None, handle=None, args=None, stats=False):
        """A HANDLE FOR OBJECTS JSON CANNOT CARRY -- so a capability is reachable over HTTP.
        holographic_objectref was written for exactly this and NOTHING CALLED IT, which the
        new usage_audit caught. Its own docstring names the failure: a capability returning
        a live object is "reachable in-process, DEAD AT THE BOUNDARY", because /invoke
        hands back {"type": "Scene", "repr": "<...object at 0x7fe17ba58fe0>"} and A MEMORY
        ADDRESS IS NOT A HANDLE.
        AND THE CAPABILITY I SHIPPED ONE MESSAGE EARLIER HAD THE SAME BUG. `unicron_runtime`
        returns a LeCoreRuntime; json.dumps fails with "Object of type LeCoreRuntime is not
        JSON serializable". By this repo's governing rule -- a capability an agent cannot
        call over /invoke with strict json.dumps does not exist -- I had shipped a
        capability that did not exist, in the same session as an audit built to catch
        precisely that.
        SO: put(obj) -> "ref:LeCoreRuntime:1", get(handle) -> the live object, and
        resolve(args) swaps every ref-string in a call's arguments back for its object,
        recursively. A bounded per-process registry, not a persistence format -- an object
        that must outlive the process goes through unicron_model_vault instead.
        THE FACULTIES THIS MAKES REACHABLE: unicron_runtime, unicron_program_library,
        unicron_model_vault and unicron_self_heal all return live objects and all were
        agent-invisible until now. See holographic_objectref."""
        from holographic.io_and_interop.holographic_objectref import (
            ObjectRefs, is_ref)
        if not hasattr(self, "_objrefs"):
            self._objrefs = ObjectRefs()
        if stats:
            return self._objrefs.stats()
        if args is not None:
            return self._objrefs.resolve(args)
        if handle is not None:
            return self._objrefs.get(handle)
        return self._objrefs.put(obj)

    def unicron_turn_memory(self, n_turns=32, per_turn=32, vocab=512, dim=None,
                            seed=0):
        """A BASE PER TURN, so a conversation stops EVICTING and starts ACCUMULATING.
        The chat schedule stored uncertain tokens in a FLAT register file and filled it on
        turn one -- 30 tokens into 32 slots -- after which every turn evicted. ACT-R
        eviction made that survivable by overwriting the least active slot, but eviction
        is a loss, and a flat file is the reason it was needed.
        `nested_memory` was already built and unused: "A LIBRARY of knowledge bases in ONE
        vector, any fact from any base in a SINGLE unbind -- bind's associativity makes
        two-level lookup cost ONE operation". Its own docstring explains why: the keys are
        composited with the base name IN FOURIER, where bind is elementwise, so the
        two-level query "is literally a multiplication reordering".
        MAP IT ONTO THE CONVERSATION: A BASE IS A TURN. MEASURED at dim 1024:
             4 turns x 32 facts =   128 total   100% recalled
            32 turns x 32       =  1024         100%
            64 turns x 32       =  2048         100%   (load m/D = 2.0)
           128 turns x 32       =  4096         100%   (load m/D = 4.0)
        FOUR TIMES THE FLAT CAPACITY LAW AT FULL ACCURACY, because crosstalk is between
        BASES rather than among all facts -- a query decodes 32 keys against ONE base's
        subspace, not 4,096 against everything.
        SO THE EVICTION WAS AN ARTEFACT OF THE FLAT LAYOUT, not a capacity limit. A
        register file that had to forget after one turn now holds a hundred and twenty
        eight turns without forgetting anything.
        THE LIMIT I DID FIND was fixture memory: the library allocates
        n_bases x facts_per_base x dim, and 128 x 64 x 1024 was killed on this box. That
        is an allocation ceiling of the machine, not of the method, and it should be stated
        that way. See holographic_nested (nested_memory)."""
        return self.nested_memory(n_bases=int(n_turns),
                                  facts_per_base=int(per_turn),
                                  vocab=int(vocab), seed=int(seed))

    def unicron_recipe(self, base_weights=None, installed_weights=None,
                       report=None, rules=None, arrays=None, prepend=2):
        """SHIP WHAT leCORE ADDED, NOT THE MODEL IT WAS ADDED TO.
        Moose asked why the installed model inflates, and whether we are doing this
        holographically. MEASURED, and it is worse than "inflated":
            original                     2.81 MB
            installed                    6.24 MB     +122%
            EXACTLY-ZERO BYTES           2.26 MB     36% OF THE FILE
        and tensor by tensor: 1.45 MB IDENTICAL to the layer it came from (just
        renumbered), 2.72 MB GROWN by the ladder, and 0.00 MB GENUINELY DIFFERENT
        VALUES. THE INSTALL ADDS 3.43 MB OF FILE FOR ZERO MB OF NEW INFORMATION.
        AND leCORE ALREADY NAMES THIS AS AN ERROR. `bank_or_formula` is the demoscene
        economy as a measured gate -- keep the FORMULA, not the samples -- and says
        outright that a bank of things a cheap formula gives you for free is NEGATIVE
        VALUE. We were banking zeros.
        SO THE RECIPE STORES RULES: a blank layer is a SHAPE, a renumbered layer is the
        SAME ARRAY under a different key, a ladder-widened tensor is a base tensor plus a
        small remainder, a register reservation is 64 BITS of seed. Only the router
        direction and the improvement correction are genuinely new, and both are small.
        MEASURED: 6.24 MB expands from 2.31 MB of real arrays -- 28 renames, 13 all-zero
        shapes, 18 base-plus-padding, 29 actually new -- and expand() rebuilds EVERY
        TENSOR BYTE-EXACT, which is the only thing that makes a recipe a format rather
        than a hope.
        ONE TRAP WORTH THE COMMENT: the padding is NOT always zero. The ladder writes real
        a_log values into the new heads, and assuming otherwise failed the exact rebuild on
        in_proj_ba where rows 8 and 9 carry the new rungs. The recipe stores the REMAINDER,
        which is nothing for a blank pad and a handful of rows for a rung.
        NOT A REPLACEMENT for the safetensors output -- other people's loaders need every
        declared tensor at full size. This is the leCore-native form for storing,
        versioning and sending an install. See holographic_recipe."""
        from holographic.io_and_interop.holographic_recipe import (
            build, expand, cost)
        if rules is not None and arrays is not None:
            return expand(rules, arrays, base_weights)
        r, a = build(base_weights, installed_weights, report or {},
                     prepend=prepend)
        return {"rules": r, "arrays": a,
                "cost": cost(r, a, installed_weights)}

    def unicron_vm_install(self, program=None, dim=None, seed=0):
        """PUT THE HOLOGRAPHIC VIRTUAL MACHINE IN THE WEIGHTS.
        Moose asked whether the installed leCore uses the VM architecture we built. IT DID
        NOT. vminstall, proglib and unlocked were all filed as TOOLING by the usage audit
        -- which is true of the PLANNERS and false of the OPERATORS.
        AN OPCODE IS A MATRIX. BIND is a circulant, PERMUTE is a permutation matrix,
        BUNDLE is a scaled identity, UNBIND is an inverse. Each applies as one matvec,
        which is exactly what install_op bakes into MLP neurons.
        AND A PROGRAM IS THEIR PRODUCT, so a whole opcode SEQUENCE fuses into ONE operator
        before it is ever installed -- verified at MAX DIFF 0.00e+00 between running three
        opcodes step by step and applying the fused matrix. DEPTH IS FREE, because the
        fusion happens at install time rather than at inference time. That is the same
        result holographic_unlocked measured at 32 operators into 128 neurons at cosine
        1.000000, finally pointed at the install instead of at a report.
        MEASURED IN A REAL MODEL: a 2-opcode program (BIND then PERMUTE) added 128
        neurons, computes at COSINE 1.000000, and cost +0.01% perplexity through the
        null-space guard -- and a full install carrying it still came out BETTER overall.
        DEFAULT OFF, because a program only earns its neurons if someone has one to run.
        install_lecore takes vm_program=[matrices]. See holographic_vminstall,
        holographic_unlocked."""
        import numpy as _np
        from holographic.io_and_interop.holographic_vsabake import circulant
        if program is None:
            d = int(dim or 128)
            g = _np.random.default_rng(int(seed))
            return {"BIND": circulant(g.standard_normal(d) / _np.sqrt(d)),
                    "PERMUTE": _np.roll(_np.eye(d), 1, axis=0),
                    "BUNDLE": 2.0 * _np.eye(d)}
        M = _np.asarray(program[0], _np.float64)
        for op in program[1:]:
            M = _np.asarray(op, _np.float64) @ M
        return M

    def unicron_bios(self, weights, cfg, model_dir=None, probe_ids=None,
                     payload_bytes=None, bits=1):
        """ENUMERATE THE MACHINE BEFORE BOOTING AN OS ON IT -- the layer that was missing.
        Every scale bug in this arc was the SAME bug in different clothes: a hardcoded
        "model.layers." while the checkpoint used "model.language_model.layers." (a
        testkit shipped ZERO layer arrays while its manifest claimed otherwise); packed
        in_proj_qkvz assumed where split was found; vocab_size assumed to equal the
        tokenizer; float16 carriers assumed on a float32 model; one uniform capacity, so
        a 128-wide model overran a boot row whose check had passed it. Each component
        reached into the weights with its own assumptions because nothing enumerated the
        hardware first.
        A BIOS does three things and they are exactly the three that were missing: POST
        (does this machine run at all -- checked BEFORE anything is written, since
        installing onto a NaN model yields a NaN model and a clean report), ENUMERATION
        (root, layer count, block period, which layers are attention, projection layout,
        vocabulary slack, carrier dtypes and capacity at 1/2/4 bits, and whether leCore is
        ALREADY installed), and ABSTRACTION (the OS consumes a profile and never touches
        the chipset).
        VERIFIED against the real Qwen3.5-0.8B: root model.language_model., SPLIT layout,
        18 linear-attention + 6 attention layers in blocks of 4, hidden 1024, vocab
        248320, mixed float16/float32 carriers, not installed.
        AND IT MAKES REFUSAL POSSIBLE BEFORE WRITING: `fits()` answered that the engine
        tarball is 2.6x too large for a single layer's surface and fits across 24 --
        which is the answer you want before an install, not halfway through one.
        See holographic_bios."""
        from holographic.io_and_interop.holographic_bios import report, fits
        prof = report(weights, cfg, model_dir=model_dir, probe_ids=probe_ids)
        if payload_bytes is not None:
            prof["fits"] = fits(prof, int(payload_bytes), bits=bits)
        return prof

    def unicron_install(self, weights, cfg, record=None, payload=None,
                        seed="leCore", states=None, audit_only=False,
                        probe_ids=None):
        """INSTALL leCORE INTO A MODEL, THEN AUDIT THAT IT IS REACHABLE.
        This project's governing rule is that a capability which cannot be surfaced does
        not exist, and three audits have caught more real defects here than any test
        suite -- a faculty silently overwritten by a duplicate method, aliases silently
        discarded by a duplicate dict key, a ward "verified" before the edit that broke
        it. Weights deserve the same rule and get it less: a boot record can be written
        to a row nobody reads, a projector installed at a layer nothing consults, a
        program stored in bits the next quantizer erases -- and NOTHING RAISES.
        So this is half installer, half auditor, and the auditor is the half that
        matters. Each check corresponds to a defect that has actually occurred:
            boot_record_reads              a record written where nothing reads it
            channel_is_addressed           hidden is not addressed; a wrong seed must
                                           read noise
            payload_round_trips            checkpoints are float32, not float64
            model_still_runs               an installed operator can emit NaNs quietly
            declared_capabilities_reachable   the governing rule itself
        VERIFIED: 4/4 on a fresh install, and the audit FAILS on a model that was never
        installed (1/4) and on one requantized afterwards (2/4) -- so it verifies rather
        than decorates. An install that writes cleanly and audits 3/5 is a model carrying
        dead weight it will never use. See holographic_install."""
        from holographic.io_and_interop.holographic_install import install, audit
        if audit_only:
            return audit(weights, seed=seed, payload=payload, cfg=cfg,
                         probe_ids=probe_ids)
        return install(weights, cfg, record=record, payload=payload, seed=seed,
                       states=states)

    def unicron_query_path(self, dim=1024, ridge=1e-2):
        """THE MODEL ASKS ITS OWN LAYER -- the last blocker, removed. Storage, seed
        expansion, capacity and the read path were settled; nothing could produce the
        KEY. A ridge-fitted projection from the residual stream does.
        MEASURED on a real Qwen3.5-0.8B stream (layer 12): fitted on the FIRST occurrence
        of 32 repeated tokens, tested on a LATER occurrence in different surrounding
        text -- train 32/32, HELD-OUT 27/32 against chance 0.031.
        A CLAIM I HAD TO RETRACT: I first concluded "keys must be derived from content"
        after arbitrary keys scored 0/16 held out. Tested properly through the same
        store, arbitrary keys score 29/32 -- the original failure was an experiment that
        gave every position a UNIQUE fact and tested on DIFFERENT positions, so there was
        nothing to generalise to. What the projection needs is RECURRING CONTENT.
        Content-derived keys remain the default for PORTABILITY (hashlib means no lookup
        table travels), not accuracy.
        This completes query -> unbind -> cleanup inside the model's own arithmetic: the
        projection is a matrix, unbinding is a shift, cleanup is lm_head.
        LIMIT: a key derived from a term is a LEXICAL address -- it retrieves what a word
        names, not what a sentence means. See holographic_querypath."""
        from holographic.agents_and_reasoning.holographic_querypath import QueryPath
        return QueryPath(dim=dim, ridge=ridge)

    def unicron_quantsafe(self, tensor, payload_bits=None, reference=None,
                          bits=4, group=64, threshold=0.45):
        """STORAGE THAT SURVIVES GGUF CONVERSION -- hide IN the quantizer, not under it.
        The low-bit substrate dies in Q4 because Q4 rewrites exactly those bits. But a
        weight whose scaled value lands near a bucket boundary can round EITHER WAY and
        both are legitimate quantizations, so the choice carries a bit -- and that bit IS
        the quantized value, so it survives.
        MEASURED on a real Qwen tensor at 4 bits:
            threshold 0.45   9.9% of weights carry   quant error 0.1131 -> 0.1165
            threshold 0.40  19.7%                             0.1131 -> 0.1259
            threshold 0.30  39.3%                             0.1131 -> 0.1583
        At 0.45 that is ~10.8 MB across a 0.8B for a 0.3% relative change in
        quantization error -- enough to carry the entire 6.96 MB engine tarball through a
        GGUF conversion.
        Needs the ORIGINAL tensor to identify carriers on read, since rounding destroys
        that information; in practice the carrier positions travel as a hash.
        See holographic_substrate.write_quantsafe."""
        from holographic.caching_and_storage.holographic_substrate import (
            write_quantsafe, read_quantsafe, quant_carriers)
        if payload_bits is not None:
            return write_quantsafe(tensor, payload_bits, bits=bits, group=group,
                                   threshold=threshold)
        if reference is not None:
            return read_quantsafe(tensor, reference, bits=bits, group=group,
                                  threshold=threshold)
        mask, _x, _s = quant_carriers(tensor, bits, group, threshold)
        return {"carriers": int(mask.sum()), "of": int(mask.size),
                "bytes": int(mask.sum()) // 8}

    def unicron_seeded_channel(self, tensor, payload_bits=None, seed="leCore",
                               rate=0.05, bits=4, group=64):
        """QUANTIZATION-SAFE STORAGE READABLE FROM A SEED ALONE -- no original tensor.
        unicron_quantsafe picks carriers by proximity to a bucket boundary: nearly free
        (0.3% error for ~10.8 MB) but the reader needs the ORIGINAL, because rounding
        destroys the proximity. This picks carriers from a SEED and encodes in the PARITY
        OF THE LEVEL, which is a property of the shipped weights.
        MEASURED against a plain 4-bit error of 0.1131 on a real Qwen tensor:
            rate 0.01    1.1 MB across a 0.8B    +1.5%
            rate 0.05    5.4 MB                  +7.4%
            rate 0.10   10.9 MB                 +14.3%
        The two schemes are a CHOICE, not a ranking: boundary-selected is cheap and needs
        the original; seed-selected is self-describing and costs error. A boot record
        belongs here at rate 0.01; a 7 MB engine belongs in the boundary channel.
        A WRONG SEED READS NOISE at chance, so the channel is addressed rather than
        merely hidden. See holographic_substrate.write_seeded."""
        from holographic.caching_and_storage.holographic_substrate import (
            write_seeded, read_seeded)
        if payload_bits is not None:
            return write_seeded(tensor, payload_bits, seed=seed, rate=rate,
                                bits=bits, group=group)
        return read_seeded(tensor, seed=seed, rate=rate, bits=bits, group=group)

    def unicron_store_program(self, weights, machine, program, bits=1, read=False):
        """PUT leCORE CODE IN THE MODEL, using the VM this project ALREADY HAS.
        Rule 0 first, and it mattered: `compile_program` and `vm_decode_plan` already
        exist, and HoloMachine is described in its own docstring as "a formatted
        holographic drive that can store and execute stored programs" -- 14 opcodes
        (LOAD/BIND/BUNDLE/PERMUTE/CALL/APPLY/IFMATCH/ITERATE/REPEAT/HALT/STORE/RECALL/
        PUSH/POP), 8 registers, an assembler that folds a program into ONE vector, and a
        decode cache measured at 6.7-14x. None of that is re-implemented here; this is
        the drive CONTROLLER, not a new machine.
        WHAT IS NEW is where the drive lives: the program vector is written into the LOW
        BITS OF ORDINARY WEIGHTS, so leCore code travels inside the checkpoint.
        VERIFIED end to end: a 7-instruction program (LOAD/APPLY/STORE/LOAD/BIND/APPLY/
        HALT) assembled into one 1024-dim vector, stored in a 3584x1024 weight tensor,
        read back EXACT, and EXECUTED with an identical trace and accumulator -- with the
        carrier weights perturbed by 0.000048 relative, which is invisible.
        See holographic_substrate.store_program."""
        from holographic.caching_and_storage.holographic_substrate import (
            store_program, load_program)
        if read:
            return load_program(weights, bits=bits)
        return store_program(weights, machine, program, bits=bits)

    def unicron_store_route(self, points, dim=512, seed=0, extend=None,
                            model=None):
        """ASK WHAT THE DATA IS BEFORE CHOOSING HOW TO STORE IT -- HRNN and HDRIFT, which
        every storage path here had been ignoring.
        Every Galvatron channel treats a payload as opaque bytes. Correct, and wasteful:
        some payloads are the OUTPUT OF A GENERATOR, and a generator is smaller than its
        output. holographic_rnn already measures this and I never asked it -- its ladder
        "measures before it models" and returns a REGIME.
        MEASURED on the real classifier:
            a ramp                 -> generator, identify(denoised), NRMSE 0.000
            repeated facts         -> generator, NRMSE 0.000
            four Gaussian clusters -> structured, demand 2.0 bits, floor 0.015
            white noise            -> INCOMPRESSIBLE, entropy rate 1.99, with an
                                      allocator quote -- it REFUSES to pretend
        So: store the RULE when a rule exists, an HDRIFT model when the data is
        structured, and the BYTES when nothing smaller is honest. A compressor that
        always compresses is lying about the incompressible case.
        AND THE MEMORY STAYS EXTENSIBLE AFTER SHIPPING: drift_compose adds moment vectors
        evidence-weighted, so a model learned later merges with the one baked in.
        GOTCHA FOUND BY TRYING IT: compose needs ONE encoder space, and drift_train probes
        bandwidth FROM THE DATA -- an extension must pin the shipped model's bandwidth and
        bounds or composing raises. Composing models that measured different scales would
        be adding numbers with different units. See holographic_storeroute."""
        from holographic.caching_and_storage.holographic_storeroute import (
            route, extend_drift)
        if extend is not None and model is not None:
            return extend_drift(self, model, extend, dim=dim)
        return route(self, points, dim=dim, seed=seed)

    def unicron_resilient_store(self, weights, data=None, seed="leCore",
                                overhead=2.5, bits=1, drop_fraction=0.0):
        """A PAYLOAD THAT SURVIVES LOSING PART OF ITS CARRIER -- leOS's answer, and the
        code was already in the tree, IMPORT-ONLY.
        Every storage channel here had a failure mode I had been documenting as
        unavoidable: the low-bit surface dies in Q4, the quant-parity channel costs
        accuracy, the vocabulary rows are tiny. holographic_fountain implements Luby
        Transform codes -- k blocks become an unlimited stream of droplets, each an XOR
        of a random subset, and ANY k(1+eps) droplets recover everything by peeling. It
        had no faculty and no catalog entry, so find_capability could not surface it: the
        solution was sitting unwired while I wrote around the problem.
        MEASURED: a 4 KB payload in 16 blocks and 40 droplets recovers EXACTLY from 28,
        so 30% of the carrier can be destroyed; and in the substrate, a payload recovered
        exactly after a QUARTER of it was destroyed, while 70% loss correctly fails.
        See holographic_substrate.write_resilient and holographic_fountain."""
        from holographic.caching_and_storage.holographic_substrate import (
            write_resilient, read_resilient)
        if data is not None:
            return write_resilient(weights, data, seed=seed, overhead=overhead,
                                   bits=bits)
        return read_resilient(weights, bits=bits, drop_fraction=drop_fraction)

    def unicron_fountain(self, data, block_size=256, overhead=2.5, seed=0):
        """LUBY TRANSFORM (RATELESS ERASURE) CODES -- k blocks become an unlimited stream
        of droplets, each the XOR of a random subset drawn from the Robust Soliton
        distribution; a receiver who collects ANY k(1+eps) of them, in any order, recovers
        all k EXACTLY by peeling (a degree-1 droplet reveals its block, which is XORed out
        of the rest, creating new degree-1 droplets).
        This module was IMPORT-ONLY -- built, documented as "the last clean idea from
        leOS", and unreachable through find_capability, which by this project's own rule
        means it did not exist. It is the robustness axis every storage channel in the
        Galvatron needed. MEASURED: 4 KB in 16 blocks, 40 droplets, exact recovery from
        28. See holographic_fountain.Fountain."""
        from holographic.agents_and_reasoning.holographic_fountain import Fountain
        f = Fountain.from_bytes(bytes(data), block_size=int(block_size))
        k = len(f.blocks)
        return f, f.droplets(max(int(k * float(overhead)), k + 4), seed=int(seed))

    def unicron_substrate(self, weights, data=None, bits=1, read=False):
        """THE MODEL'S WEIGHT SURFACE AS A STORAGE MEDIUM -- the platter, not the spare
        sectors. A floppy, a CD and a tape were all irregularities on a surface that
        someone chose a pattern for; the capacity was in the SURFACE. The unused
        vocabulary rows were the spare sectors (276 rows, ~0.56 MB). Every weight in the
        model is the surface, and a float16's low bits carry almost nothing -- the same
        measurement that showed 4-bit quantization costs only 0.11 output error.
        MEASURED on a real Qwen3.5-0.8B layer, overwriting low bits and scoring the
        layer's OUTPUT:
            1 bit/weight   0.00107 error   INVISIBLE   -> 109 MB across the model
            2 bits         0.00317         usable      -> 218 MB
            4 bits         0.00822         usable      -> 435 MB
            8 bits         0.06972         damaging
        Two hundred times the spare rows, in space the model already carries.
        A HEADER GOES FIRST (magic, length, content hash) because every bit pattern is a
        valid float: without it a reader always "succeeds" and always returns garbage. An
        unwritten model is REJECTED, not misread.
        THE LIMIT THAT MATTERS, and it is a very common workflow: QUANTIZATION DESTROYS
        THE PAYLOAD -- converting to GGUF Q4 rewrites exactly these bits. The substrate
        is for a model shipped as float weights, and the reader catches the corruption by
        hash rather than returning it. Embeddings are never used as carriers, since
        damage there shows up as garbled text. See holographic_substrate."""
        from holographic.caching_and_storage.holographic_substrate import (
            write_payload, read_payload, capacity_bytes)
        if read:
            return read_payload(weights, bits=bits)
        if data is None:
            return {"capacity_bytes": capacity_bytes(weights, bits), "bits": bits}
        return write_payload(weights, data, bits=bits)

    def unicron_boot(self, weights=None, record=None, row=None, key=None,
                     write=False):
        """leCORE AS A BOOTABLE LAYER inside the model's own weights -- the OS, not glue.
        A demoscene 4k intro does not STORE its content; it stores a SEED and a bootstrap
        and EXPANDS deterministically. Same shape here, because a model has room for a
        seed and no room for a library.
        WHAT THE LAYER COSTS, once the parts are named honestly:
            role vocabulary   cyclic shifts              ZERO (roles are integers)
            symbol codebook   seeded hypervectors        ZERO (hashlib from a seed)
            capability table  name -> hypervector        ZERO (same rule)
            instruction set   bind/unbind/bundle/cleanup ZERO (shifts, adds, lm_head)
            THE DATA          bound key/value traces     32 facts per row
            THE BOOT RECORD   seed + manifest            ONE row
        Everything except the DATA regenerates from one seed, so the model carries a BOOT
        SECTOR -- one vocabulary row with a magic number, seed, version and contents --
        and the other rows are DELTAS on top of what the seed already builds.
        VERIFIED: a model carrying ONE row booted a full layer from the weights alone,
        the codebook and capability table regenerated identically (hashlib, so it agrees
        across processes where hash() would not), 6 facts recalled by key, the record
        survived a float32 round trip, random weights were REJECTED rather than misread,
        and an oversized manifest refused rather than truncated.
        STILL OPEN: the model does not QUERY the layer by itself -- something must supply
        the key hypervector. Storage, expansion, capacity and the read path are settled;
        the query path is not. See holographic_boot."""
        from holographic.io_and_interop import holographic_boot as B
        if write and weights is not None and record is not None:
            return B.write_boot(weights, record, key=key, row=row)
        if weights is not None:
            return B.boot(weights, row=row, key=key)
        return B

    def unicron_call_tokens(self, weights=None, cfg=None, runtime=None,
                            names=(), tokenizer_size=None, positives=None,
                            negatives=(), table=None, generate=None, n_new=32):
        """THE MODEL EMITS A CAPABILITY CALL, AND SOMETHING RUNS IT -- the piece every
        other bake was one step short of.
        A forward pass emits LOGITS, not function calls, so no weight surgery lets a model
        invoke fluid_step. But a model can emit a TOKEN, and a token can NAME a
        capability. That is how every tool-calling system works and it is the only
        mechanism that turns installed data and circuits into invoked behaviour.
        THE MYCELIUM IS THE UNUSED VOCABULARY: Qwen3.5-0.8B declares 248,320 rows against
        a tokenizer that defines 248,044, so 276 rows the model never emits and never
        reads become CALL TOKENS -- addressable by id, carried in the weights, invisible
        to anything not looking.
        THE CHAIN, all verified WEIGHTS-ONLY: allocate capabilities to free rows; a
        ridge-fitted head emits the right call in 4/4 contexts and stays SILENT in 3/3
        negatives (negatives are not optional -- a model that calls a tool on every prompt
        is worse than one that never does); a generation loop catches the token,
        dispatches, and continues. Measured end to end: 26 tokens generated during which
        the model called bundle_capacity ON ITS OWN, with the call token CONSUMED rather
        than emitted as text.
        SAFETY IS THE WHITELIST, inherited from the toolbelt: only allocated names are
        callable, a capability needing arguments the stream cannot supply is REFUSED
        rather than guessed, and every dispatch is logged.
        WHAT IT STILL IS NOT: the capability runs OUTSIDE the forward pass, in the
        harness. That is what tool calling is -- llama-server and every agent framework
        work this way. The model's contribution is DECIDING, which is the part that could
        not be faked. See holographic_calltoken."""
        from holographic.agents_and_reasoning.holographic_calltoken import (
            allocate, free_rows, teach_calls, dispatch, generate_with_calls)
        if generate is not None:
            return generate_with_calls(runtime, generate, table, self,
                                       n_new=n_new)
        if positives is not None:
            return teach_calls(weights, cfg, runtime, positives, negatives, table)
        if names:
            rows = free_rows(weights, tokenizer_size)
            return allocate(names, rows)
        return dispatch

    def unicron_swarm_bake(self, weights, cfg, experts, states, layer=None,
                           gain=0.0, temperature=6.0):
        """A SWARM THAT RUNS INSIDE ONE FORWARD PASS, in ordinary weights.
        The runtime SwarmResident cannot do this: it BRANCHES -- runs the model several
        times and compares -- and a single forward pass cannot branch. It also needs
        leCore present, so it vanishes on export. What DOES fit is a ROUTED MIXTURE: N
        specialist circuits plus a gate that picks per token, deliberating in parallel
        rather than by re-running.
        THE GATE MUST ROUTE BY CONTENT, which is what separates a swarm from decoration.
        install_op's gate is deliberately NEAR-CONSTANT so an operator applies uniformly;
        a swarm needs the opposite, so the gates are keyed to the stream's own leading
        directions -- derived from the model's activations, not chosen.
        MEASURED on a real Qwen3.5-0.8B stream of 235 tokens spanning prose, facts, code,
        SQL, markdown and questions:
            4 experts usage [0.39 0.20 0.18 0.22], entropy 1.34 of 1.39
            prose -> expert 0 (78%) | code -> expert 2 (47%) | SQL+md -> expert 1 (59%)
        Different registers select different specialists -- the property a swarm needs,
        and the one the runtime version could never show (its branches were IDENTICAL, so
        its contrast digest was exactly zero, measured earlier in this arc).
        BIT-IDENTICAL AT gain=0 and measurably active at 0.05.
        WHAT IT IS NOT: the experts are CIRCUITS (linear maps installed as neurons), not
        leCore faculties. This routes a denoiser, a binding or a correction by content. It
        does not let the model call fluid_step, and nothing in a forward pass can, because
        a forward pass emits logits rather than function calls.
        See holographic_swarmbake."""
        from holographic.io_and_interop.holographic_swarmbake import install_swarm
        return install_swarm(weights, cfg, experts, states, layer=layer,
                             gain=gain, temperature=temperature)

    def unicron_vsa_roles(self, dim=1024):
        """A WORKING ROLE-FILLER MACHINE inside the model, at zero storage cost.
        The first version of this used one circulant matrix PER ROLE. It worked and was
        unaffordable: eight roles wanted 8,192 MLP neurons against a 3,584-wide MLP --
        228% of the layer for eight slots. The fix is the oldest trick in VSA: make roles
        POWERS OF ONE OPERATOR. A cyclic shift is a permutation, shifting k times IS role
        k, so bind and unbind are index permutations with no multiplies and no stored
        operators at all. Bundling is addition (the residual stream already does it) and
        cleanup is argmax over a codebook (lm_head, already present).
        MEASURED capacity, cleanup against the value codebook:
            2/4/8/16/24/32 pairs -> ALL recovered;  48 -> 45/48;  96 -> 81/96
        Thirty-two role-filler pairs in one 1024-dim vector, exactly recovered, storing
        nothing.
        WHAT IT ADDS: somewhere to put STRUCTURE. A residual stream is a bag of features
        with no way to say "the subject is X and the object is Y" without spending
        separate dimensions per slot; role-filler binding says it in one vector.
        HONEST LIMIT: roles are fixed shift amounts and cleanup needs a known codebook --
        an addressable structured register, not a general symbolic reasoner.
        See holographic_vsaroles."""
        from holographic.io_and_interop import holographic_vsaroles as R
        return R

    def unicron_vsabake(self, weights, cfg, role, layer=None, mean_h=None,
                        unbind=False, gate_target=16.0, scale=1.0):
        """INSTALL leCORE'S ALGEBRA INSIDE THE WEIGHTS -- a holographic computing space
        the model runs itself, with no residents and in any runtime.
        The reason it works is small: BIND WITH A FIXED ROLE is circular convolution with
        a known vector, which is a CIRCULANT MATRIX, which is a weight tensor (verified
        to 9e-17 against the FFT). UNBIND is the same with the role's involution. BUNDLE
        is addition -- what a residual stream already does for free. CLEANUP is argmax
        over a codebook, which is lm_head. Three of the four primitives are things the
        architecture computes anyway; the fourth is a matrix.
        A transformer MLP is down @ (silu(gate.h) * (up.h)): set the gate for a
        near-constant positive activation, put the circulant rows in up, and the block
        computes the binding. MEASURED on a real stream: direction cosine 1.000000, gain
        spread 0.47 which is harmless because every VSA readout is direction-based.
        LIMIT, asserted in the selftest rather than left to the reader: ROLES ARE FIXED
        AT BAKE TIME. Binding two RUNTIME values is bilinear and no fixed weight matrix
        computes it -- this is a machine with a baked instruction set, not a general VSA
        interpreter. See holographic_vsabake."""
        from holographic.io_and_interop.holographic_vsabake import (
            circulant, involution, install_op)
        r = involution(role) if unbind else role
        return install_op(weights, cfg, circulant(r), layer=layer,
                          mean_h=mean_h, gate_target=gate_target, scale=scale)

    def unicron_distill(self, weights, cfg, teacher_logits_fn, prompts, lr=0.05,
                        head_key=None):
        """TEACH THE WEIGHTS TO DO WHAT THE RESIDENTS DO -- the move that gets NONLINEAR
        residents into a plain checkpoint. A resident-equipped Galvatron is a function
        from tokens to logits; the student does not have to reproduce the MECHANISM, only
        the OUTPUT, so behaviour that consults a corpus, repairs a stream or runs a
        recurrence can still land in weights.
        Head-only by least squares: logits are lm_head @ h and h is what the student
        already computes, so this is a LINEAR problem -- no autodiff through 24 layers,
        blast radius exactly one tensor, ridge-regularised toward the original head
        because a head that fits 6 prompts perfectly has learned the prompts.
        MEASURED, teacher agreement before -> after (train / held-out):
            weak teacher   0.941/0.951 -> 0.997/0.993   perplexity 6.35 -> 6.34
            medium         0.826/0.854 -> 0.972/0.958                6.35 -> 6.50
            strong         0.545/0.590 -> 0.962/0.903                6.35 -> 8.59
        The gain generalises to held-out prompts (it is not a lookup table) and the cost
        is visible: a strong teacher moves the head far enough to hurt perplexity, which
        is the trade to watch rather than hide.
        KEPT LIMIT: a head edit cannot change WHAT h IS, so it absorbs what is linearly
        readable from the final state and no deeper. And a corpus you will edit tomorrow
        should NOT be frozen into weights today. See holographic_galvadistill."""
        from holographic.io_and_interop.holographic_galvadistill import distill_head
        return distill_head(weights, cfg, teacher_logits_fn, prompts, lr=lr,
                            head_key=head_key)

    def unicron_bake(self, weights, cfg, banned=(), memories=(), steer=None,
                     layer=None, probe_logits=None, mean_h=None,
                     calibration=None):
        """SMUGGLE RESIDENTS INTO THE WEIGHTS so they survive any runtime, quantizer or
        format. "A GGUF file has nowhere to put a function that runs between layers" was
        the wrong conclusion: several residents are mathematically identical to a WEIGHT
        EDIT, and weights travel everywhere.
        WARD -- VERIFIED: a ban is a logit bias and logits are lm_head @ h, so a banned
        row pointed against the high-scoring directions is driven far below every
        competitor. Survived a weights-only runtime on 4 prompts with banned logits >5
        below the winner. (Zeroing the row instead -- the obvious move -- fails: 85% of
        real logits are NEGATIVE, so a zero would outrank most of the vocabulary.)
        MEMORY -- PARTIAL, and stated as such: an MLP is already a key-value store, so a
        memory is a NEW NEURON (a row in up/gate, a column in down). It flips the target
        token from pure weights, but at the magnitude needed it also perturbs unrelated
        prompts -- value strength and selectivity pull against each other in one neuron.
        Two real bugs were found here: identical gate/up rows make NON-matches multiply
        to a POSITIVE activation, and a cosine-unit threshold is meaningless against a
        norm-scaled dot product.
        NOT BAKEABLE, honestly: anything needing state the architecture does not compute
        -- the Wiener dreamer's per-batch variance, the HRNN's recurrence, retrieval over
        a corpus. See holographic_galvabake."""
        from holographic.io_and_interop.holographic_galvabake import (
            bake_ward, bake_memory, bake_steer)
        out, reps = dict(weights), []
        if banned:
            out, r = bake_ward(out, cfg, banned, probe_logits=probe_logits)
            reps.append(("ward", r))
        if memories:
            out, r = bake_memory(out, cfg, memories, layer=layer, mean_h=mean_h,
                                 calibration=calibration)
            reps.append(("memory", r))
        if steer is not None:
            out, r = bake_steer(out, cfg, steer, layer=layer)
            reps.append(("steer", r))
        return out, dict(reps)

    def unicron_port(self, pack_dir, out_dir, model_name="galvatron", port=5931):
        """CARRY AS MUCH OF A GALVATRON AS llama.cpp / OLLAMA CAN HOLD, and say plainly
        what it cannot. Measured first: loading a pack's model.safetensors in another
        framework gives the BARE model -- same prompt, the leCore run held its ward and
        the weights-only run BREACHED it.
        WHAT TRAVELS, each into a native mechanism: the WARD becomes a GBNF grammar
        (llama.cpp constrains sampling to a formal grammar, so the ban is enforced by
        their sampler); the MANIFEST becomes GGUF metadata (GGUF carries arbitrary
        key/value pairs, so the roster rides inside the file); MEMORY, TOOLBELT and
        VERIFIER become an MCP/OpenAI sidecar (llama-server has function calling and MCP
        hooks); LEAP maps to llama.cpp's own speculative decoding.
        WHAT DOES NOT: dreamer, carrier, hrnn, screen -- they act on the residual stream
        between layers and a GGUF file has nowhere to put a function that runs there.
        The emitted README names the losses, not only the wins.
        This does NOT convert weights: that is llama.cpp's convert_hf_to_gguf.py, which
        is well-tested, and a reimplementation would be a worse copy.
        See holographic_galvaport.export."""
        from holographic.io_and_interop.holographic_galvaport import export
        return export(pack_dir, out_dir, model_name=model_name, port=port)

    def unicron_cache(self, runtime=None, verify=False):
        """STOP THE MODEL REDOING WORK IT ALREADY DID. Content-keyed memo over the paths
        measured to repeat: attention screen routing (k-means was re-run ONCE PER HEAD
        PER FORWARD on unchanged keys), capability routing (0.2914s cold -> 0.000022s
        warm, 13,036x) and corpus retrieval. Branch-and-select generation multiplies all
        three by k, which is exactly where it pays.
        MEASURED end to end on grounded generation with k=6 branches: 75% hit rate and
        output BIT-IDENTICAL to the uncached run.
        Keys are hashlib digests of the actual bytes, shapes and dtypes -- never hash(),
        which is salted per process and would make the cache miss across restarts and
        break determinism. verify=True re-runs each hit and asserts equality, so "fast"
        can never quietly mean "wrong". See holographic_galvacache."""
        from holographic.caching_and_storage.holographic_galvacache import install
        return install(runtime=runtime, mind=self, verify=bool(verify))

    def unicron_toolbelt(self, hidden_dim, layer=0, families=(), deny=(),
                         gain=1.0, max_calls=32):
        """GIVE THE MODEL THE WHOLE CATALOG, not a hand-picked dozen. Carries the ROUTER
        (find_capability) rather than one named capability, so demux, resonator
        factoring, denoisers, drift algebra, fluid steps, path tracing, linear solves and
        the VSA primitives are all reachable BY DESCRIPTION from inside the forward pass
        -- 1,863 invocable capabilities instead of whichever twelve a packager thought of.
        Safety is a whitelist, not a hope: `families`/`deny` bound what may run, an arity
        guard SKIPS anything whose arguments cannot be supplied from the stream rather
        than guessing them, and every call is logged with the query that selected it.
        HONEST LIMIT: this is ACCESS, not competence -- a small model will not learn to
        drive a path tracer. What it buys is that the RESULT of a real computation enters
        the stream instead of a guess about it, and that a harness can audit which
        computation ran. See holographic_toolbelt.ToolbeltResident."""
        from holographic.agents_and_reasoning.holographic_toolbelt import (
            ToolbeltResident)
        return ToolbeltResident(self, hidden_dim, layer=layer, families=families,
                                deny=deny, gain=gain, max_calls=max_calls)

    def unicron_memory(self, dim=1024, namespace="mem"):
        """THE GALVATRON'S OWN MEMORY, in leCore's holographic database -- not in files.
        Notes and their provenance are ROWS (id, title, author, kind, tags, session), so
        "what did the swarm conclude" is a WHERE clause rather than a convention; links
        are an EDGE TABLE whose adjacency() gives forward and reverse traversal, so
        backlinks are data instead of a re-parse of prose. Free text is ranked by BM25
        and can be scoped by a SQL filter -- binding a paragraph as a categorical filler
        would encode a document as one symbol and rank it by accident. Durability is the
        database's own crash-safe snapshot/restore.
        memory.passages() feeds the corpus resident and the fact checker, so what the
        model can retrieve is exactly what it may assert.
        CORRECTION ON RECORD: an earlier version of this wrote markdown files and derived
        backlinks by re-parsing text -- a filesystem built beside an engine that already
        had a database. See holographic_memory.Memory."""
        from holographic.caching_and_storage.holographic_memory import Memory
        return Memory(self, dim=int(dim), namespace=str(namespace))

    def unicron_vault(self, root):
        """IMPORT A FOLDER OF MARKDOWN NOTES (an existing Obsidian vault) so its content
        can be moved into unicron_memory, which is where a Galvatron's memory belongs.
        KEPT ONLY AS A CONVERTER: writing .md files and re-parsing them for backlinks was
        the wrong build -- leCore already has a holographic database with tables, SQL,
        edge-table adjacency, views, a journal and crash-safe snapshots. Use
        unicron_memory for new work.
        A local linked markdown vault -- Obsidian's actual
        core, which is small: plain .md files, [[wikilinks]], backlinks DERIVED from the
        text (never stored, so they cannot disagree with it), tags, aliases, a graph with
        clusters and orphans, and unresolved links reported rather than swallowed. The
        files are the product; an existing Obsidian vault opens here unchanged and these
        notes open in Obsidian.
        WHAT MAKES IT MORE THAN A NOTE APP: vault.passages() is a grounding corpus for
        the corpus resident (retrieval into the residual stream, no context window
        spent), the fact checker builds evidence from the SAME notes so retrievable ==
        assertable, and residents WRITE notes back with author/kind frontmatter -- a
        swarm conclusion becomes a linked note that later retrieval finds, and is never
        mistaken for something a person wrote. A human and a swarm keep one notebook.
        See holographic_vault.Vault."""
        from holographic.caching_and_storage.holographic_vault import Vault
        return Vault(root)

    def unicron_knowledge(self, root, session=None):
        """EVERYTHING THE MODEL IS EVER TOLD, kept and findable: conversation turns,
        documents handed over for RAG, its own outputs, and NOTES THE RESIDENTS WRITE --
        one cataloged, deduplicated, persistent store with provenance on every entry
        (kind, source, author, session, timestamp). Retrieval without provenance is how
        a model's own guess returns three turns later wearing a citation.
        ONE STORE, TWO READERS: the corpus resident retrieves from it and the fact
        checker builds its evidence from it (store.evidence()), so anything retrievable
        is assertable and nothing else is -- two indexes would eventually disagree, and
        the disagreement would look exactly like hallucination.
        Knowledge spans sessions BY DEFAULT (a fact does not belong to the thread that
        happened to mention it) but that is a POLICY, not a law: store.set_scope("all" |
        "session" | "none") decides what a given conversation may reference, it persists
        across restarts (a privacy setting that forgets itself is worse than none,
        because the user believes it held), and it binds BOTH readers -- a session that
        cannot retrieve a fact cannot have the fact checker certify it either.
        store.prune(session=/kind=/source=/older_than=) deletes with a dry_run preview
        and REFUSES to run with no filter; clear(confirm=True) is the deliberate
        everything. See holographic_knowledgestore.KnowledgeStore."""
        from holographic.caching_and_storage.holographic_knowledgestore import (
            KnowledgeStore)
        return KnowledgeStore(root, session=session)

    def unicron_scribe(self, store, author="swarm", layer=0, partition=None,
                       summarize=None):
        """Let a resident WRITE to the shared record: partitioned notes that rank in the
        same retrieval as the user's turns and documents, tagged with their author and
        partition so an inner conclusion is never mistaken for an input. An OBSERVER by
        construction -- hook() records and returns None, because a component that both
        writes the record and changes the behaviour it records is not auditable.
        See holographic_knowres.ScribeResident."""
        from holographic.agents_and_reasoning.holographic_knowres import ScribeResident
        return ScribeResident(store, author=author, layer=layer,
                              partition=partition, summarize=summarize)

    def unicron_sessions(self, root, runtime=None):
        """PERSISTENT NAMED CONTEXTS -- a Galvatron's context as a FILE, not a process.

        A context here is inference STATE (GDN matrices, conv windows, KV, position),
        not a transcript, so resuming costs NO re-prefill: a long context comes back in
        the time it takes to read an npz and the model continues mid-thought. Sessions
        are independent by construction, so a harness can keep one per user, document or
        task, swap them by name, fork one into two, and expire them on its own schedule
        -- days or weeks, not one process lifetime.
        CONTRACT, asserted in the selftest: generation resumed from a reloaded session is
        TOKEN-IDENTICAL to generation that never stopped. Sessions also record the model
        fingerprint and REFUSE to load into a different checkpoint, because restoring
        into the wrong model produces confident nonsense.
        Serve them over the OpenAI-compatible API by passing session_root to
        unicron_serve_openai. See holographic_session.SessionStore."""
        from holographic.io_and_interop.holographic_session import (
            SessionStore, runtime_fingerprint)
        fp = runtime_fingerprint(runtime) if runtime is not None else None
        return SessionStore(root, fingerprint=fp)

    def unicron_imbue(self, model_dir, out_dir, corpus=(), probe_text=None,
                      banned=(), bundle_engine=True, notes=""):
        """ONE CALL: ordinary checkpoint in, IMBUED GALVATRON out -- weights plus the
        resident roster, the CALIBRATION those residents need (healthy stream statistics
        harvested by actually running the model, salience quantiles, carrier basis), the
        grounding corpus, and leCore itself with a run.py.
        HONEST ABOUT THE WORD: nothing is written into the weights -- residents are
        structure in the forward pass and cannot live in a tensor. What ships is
        everything needed to RECONSTRUCT them at load. Open model.safetensors in another
        framework and you get the bare model back exactly, with the ward, oracle, corpus
        grounding, fact checker and time travel all gone; the manifest says so itself.
        See holographic_galvapack.imbue."""
        from holographic.io_and_interop.holographic_galvapack import imbue
        return imbue(model_dir, out_dir, self, corpus=corpus,
                     probe_text=probe_text, banned=banned,
                     bundle_engine=bundle_engine, notes=notes)

    def unicron_maximal_specs(self, runtime, healthy_hiddens, corpus=(), banned=(),
                              memories=(), carrier_pairs=None, capability=None,
                              capability_args=None):
        """THE MAXIMAL GALVATRON: every resident kind leCore can express -- ward,
        dreamer, oracle, salience-gated corpus RAG, carrier, capability call, HRNN
        observer -- as a DECLARATIVE spec list you can inspect, edit, save and diff
        before anything is built. Layer placement is derived from the model's depth:
        repair early (fix a corrupted stream before later layers compound it), knowledge
        and memory late (near the decision, where an injection reaches the logits),
        observation last (where the trajectory is complete). Feed the result to
        unicron_save_pack or unicron_bundle. See holographic_galvapack.maximal_specs."""
        from holographic.io_and_interop import holographic_galvapack as _p
        return _p.maximal_specs(runtime, healthy_hiddens, corpus=corpus,
                                banned=banned, memories=memories,
                                carrier_pairs=carrier_pairs,
                                capability=capability,
                                capability_args=capability_args)

    def unicron_best_portable(self, weights, cfg, out_path, eval_tokens=None,
                              filter_model=True, n_refine=None):
        """THE BEST PLAIN CHECKPOINT WE CAN HONESTLY PRODUCE -- because the compatible
        model has to push its limits too, even though residents cannot travel in
        weights. Applies only levers that survive in ordinary weights: regime-routed
        spectral filtering (heavy-tail layers PASS THROUGH -- forcing a cut there is what
        produced the measured collapse), then a plain safetensors export at the chosen
        fidelity. With eval_tokens, perplexity is measured IN-ENGINE before and after, so
        the export ships with a NUMBER instead of the usual UNVERIFIED disclaimer.
        See holographic_galvapack.best_portable."""
        from holographic.io_and_interop import holographic_galvapack as _p
        return _p.best_portable(weights, cfg, out_path, eval_tokens=eval_tokens,
                                filter_model=filter_model, n_refine=n_refine)

    def unicron_save_pack(self, path, weights, cfg, residents=(), notes=""):
        """Ship a Galvatron as a PACKAGE: plain safetensors (converts and runs anywhere,
        residents absent) plus galvatron.json -- a DECLARATIVE resident manifest that is
        data, never code (no pickle, no exec crossing a file boundary). The manifest
        states plainly what running the bare checkpoint gives up.
        See holographic_galvapack.save_pack."""
        from holographic.io_and_interop import holographic_galvapack as _p
        return _p.save_pack(path, weights, cfg, residents=residents, notes=notes)

    def unicron_load_pack(self, path, lazy=False, with_mind=True):
        """Load a Galvatron package into a running model with its residents rebuilt from
        the manifest. DEGRADES GRACEFULLY: without a mind (or for resident kinds this
        leCore does not know) it serves the plain model and SAYS SO in the report --
        never a silent downgrade. Returns (galvatron, report).
        See holographic_galvapack.load_pack."""
        from holographic.io_and_interop import holographic_galvapack as _p
        return _p.load_pack(path, mind=(self if with_mind else None), lazy=lazy)

    def unicron_serve_openai(self, galvatron, port=5930, model_name="galvatron",
                             tokenizer=None, run=True, session_root=None,
                             mind_tools=False):
        """Put an OpenAI-compatible front door on a Galvatron: /v1/models,
        /v1/completions, /v1/chat/completions -- what LM Studio clients, the OpenAI SDK
        and most agent frameworks already speak, so the scaffolding is invisible to
        them while residents run underneath. Without a tokenizer the API exchanges
        TOKEN IDS rather than inventing a vocabulary it does not have. run=False
        returns the Flask app instead of serving. With session_root, /v1/chat/completions
        accepts a "session" (or "user") field for PERSISTENT multi-turn contexts with no
        re-prefill, and /v1/sessions lists, forks and deletes them -- so a harness manages
        many contexts exactly as it would with any other model.
        See holographic_galvapack.make_app."""
        from holographic.io_and_interop import holographic_galvapack as _p
        app = _p.make_app(galvatron, model_name=model_name, tokenizer=tokenizer,
                          mind=(self if mind_tools else None),
                          session_root=session_root)
        if not run:
            return app
        app.run(port=int(port), use_reloader=False)

    def unicron_hf_wrapper(self, galvatron):
        """Wrap a Galvatron in the shape transformers callers expect --
        .generate(input_ids, max_new_tokens=...) returning (1, T+n), plus a callable
        returning logits -- so existing harness code runs unmodified with residents
        live underneath. See holographic_galvapack.HFCompatWrapper."""
        from holographic.io_and_interop import holographic_galvapack as _p
        return _p.HFCompatWrapper(galvatron)

    def unicron_lazy_weights(self, model, max_cached=8, n_refine=6, base_bits=3,
                             max_bits=9):
        """COMPRESSION INSIDE THE MODEL: hold weights as middle-out codes in RAM and
        decode each tensor on demand as the forward pass reaches it (LRU working set).
        Drop-in for a weights dict -- pass the result straight to unicron_runtime.
        Measured: 2.67x smaller resident store, argmax sequence identical to dense.
        A FOOTPRINT lever, not a speed lever (a cache miss costs a decode). Truncating
        refinement layers trades fidelity for size with a knob, never silently.
        See holographic_unicron.LazyWeights."""
        from holographic.io_and_interop import holographic_unicron as _u
        if isinstance(model, str):
            model = _u.load_model(model)
        return _u.LazyWeights(model, max_cached=max_cached, n_refine=n_refine,
                              base_bits=base_bits, max_bits=max_bits)

    def unicron_export_portable(self, weights, out_path, n_refine=None, dtype="F32"):
        """Decode a compressed/lazy store to a PLAIN safetensors file -- the bridge to
        every standard harness. Ollama / LM Studio / llama.cpp consume GGUF produced
        from an ordinary Hugging Face directory (convert_hf_to_gguf.py); none expose a
        custom-loader hook, so the portable artifact is deliberately boring and
        converts like any checkpoint. VERIFIED: an exported model loads into
        transformers with 0 missing / 0 unexpected keys and generates. What does NOT
        travel: residents are runtime behaviour, not weights -- a portable export is
        the model alone. See holographic_unicron.export_portable."""
        from holographic.io_and_interop import holographic_unicron as _u
        return _u.export_portable(weights, out_path, n_refine=n_refine, dtype=dtype)

    def unicron_middleout(self, matrix, n_refine=6, base_bits=3, max_bits=9):
        """PROGRESSIVE weight code -- one artifact, many fidelity points. Coarse base
        plus successive-approximation refinement layers; decode any PREFIX, so the same
        stored file serves a 3-bit edge deployment and a 9-bit server one with no
        re-encode and no rank/cut decision (the decision that made the heavy-tail regime
        so treacherous). HONEST: per-byte quality is at PARITY with flat uniform
        quantization, never better -- three refutations pinned in the module selftest.
        This ships for progressivity, not compression. Returns the code dict; pair with
        unicron_middleout_decode. See holographic_unicron.middle_out_encode."""
        from holographic.io_and_interop import holographic_unicron as _u
        import numpy as _np
        return _u.middle_out_encode(_np.asarray(matrix), n_refine=n_refine,
                                    base_bits=base_bits, max_bits=max_bits)

    def unicron_middleout_decode(self, code, n_refine=None):
        """Decode a middle-out stream at a chosen truncation point (None = full depth):
        fewer refinement layers = smaller, coarser weights from the SAME artifact.
        Returns (weights, bytes_used) so the budget is a number, not a hope.
        See holographic_unicron.middle_out_decode."""
        from holographic.io_and_interop import holographic_unicron as _u
        return (_u.middle_out_decode(code, n_refine=n_refine),
                _u.middle_out_bytes(code, n_refine=n_refine))

    def unicron_capability_resident(self, capability, hidden_dim, layer, trigger,
                                    gain=1.0, reduce=None):
        """TIER C -- let the model CALL leCore's catalog from inside its own forward
        pass. A resident watches the residual stream; when `trigger(hidden)` returns an
        args dict, it invokes `capability` through this mind's own front door (the same
        contract /invoke uses -- so fluid_step, smoke_step, market analytics, mesh and
        image ops are all reachable), then encodes the RESULT back into the stream so
        the next layers think WITH it. Scalars go through leCore's ScalarEncoder, so the
        number is recoverable, not just a nudge. Every call is logged for audit.
        Deterministic. NOTE the hard negative it exists to answer: exact programs are
        NOT weight deltas and cannot be imbued into weights (unicron_imbue moves
        fine-tune learning only) -- the model reaches them instead of absorbing them.
        See holographic_capresident.CapabilityResident."""
        from holographic.agents_and_reasoning.holographic_capresident import (
            CapabilityResident)
        return CapabilityResident(self, capability, hidden_dim, layer, trigger,
                                  gain=gain, reduce=reduce)

    def unicron_salience_trigger(self, runtime, healthy_hiddens, quantile=0.8,
                                 use="entropy"):
        """LET THE MODEL ASK. Every other resident fires on a trigger the CALLER writes,
        which makes a Galvatron capable but not self-directed. This reads the model's own
        hidden state through the final norm and LM head (the logit lens) and fires where
        the model is UNCERTAIN -- so retrieval, memory and tool calls happen where it
        actually needs them, with no training and no new tokens: the model never has to
        learn a <search> token because its hesitation is readable directly. MEASURED:
        the lens tracks true final-token entropy at corr 0.98. The threshold is a
        QUANTILE of the model's own distribution (relative, so it transfers), and
        .gate(payload_fn) wraps any resident's trigger. See
        holographic_knowres.SalienceTrigger."""
        from holographic.agents_and_reasoning.holographic_knowres import SalienceTrigger
        st = SalienceTrigger(runtime, use=use)
        st.calibrate(healthy_hiddens, quantile=quantile)
        return st

    def unicron_corpus_resident(self, corpus, hidden_dim, layer, query_fn,
                                gain=1.0, top=1):
        """RAG whose result lands in the RESIDUAL STREAM, not the prompt. BM25 over your
        corpus (delegates to mind.bm25_rank -- exact lexical matching, pure NumPy), the
        winning passage encoded to a vector, and the model consumes it before choosing
        its next token. The corpus is unbounded and lives on leCore's side, so it costs
        NO context window. Every retrieval is logged with its query and passage --
        retrieval nobody can audit is worse than none. query_fn(hidden)->str or None.
        See holographic_knowres.CorpusResident."""
        from holographic.agents_and_reasoning.holographic_knowres import CorpusResident
        return CorpusResident(self, corpus, hidden_dim, layer, query_fn,
                              gain=gain, top=top)

    def unicron_hrnn_resident(self, hidden_dim, layer, dim=1024, seed=0, gain=0.0):
        """Run leCore's Holographic RNN on the model's OWN hidden trajectory: the LLM
        emits a sequence of hidden states, and HRNN is the engine built to characterize
        sequences (regime, mechanism, provenance). Defaults to gain=0 -- a pure OBSERVER
        that leaves logits bit-identical (asserted), because an observer that silently
        steers is a bug; influence is opt-in. Its verdict summary is built from VALUES
        only (the raw verdict embeds live closures whose repr carries memory addresses).
        See holographic_knowres.HRNNResident."""
        from holographic.agents_and_reasoning.holographic_knowres import HRNNResident
        return HRNNResident(self, hidden_dim, layer, dim=dim, seed=seed, gain=gain)

    def unicron_manifold_voids(self, points, n_probes=800, mix=3, q=0.999,
                               seed=1, surrogate_trials=5):
        """Find the regions a model's activations NEVER visit -- holes inside its own
        territory, not extrapolation outside it. Probes are convex combinations of real
        states (inside the support by construction), scored against the data's OWN
        nearest-neighbour spacing, with a matched-covariance surrogate control because
        raw void counts are dimension-confounded.
        VALIDATED before use: 0 false positives on uniform data, 100% of voids inside a
        planted hole at three radii, and split-half stable (held-out data stays 3x
        further from a void than a typical point).
        KEPT NEGATIVE: leCore's mind.void_map is the WRONG instrument here -- on a
        planted hole its z read LOWER inside than outside, because the drift model's
        smooth kernel fills holes in. See holographic_voidmanifold.manifold_voids."""
        from holographic.agents_and_reasoning.holographic_voidmanifold import (
            manifold_voids)
        return manifold_voids(points, n_probes=n_probes, mix=mix, q=q, seed=seed,
                              surrogate_trials=surrogate_trials)

    def unicron_void_probe(self, runtime, layer, basis, mean, void_points,
                           token_ids, hooks=None):
        """DECODE a void: substitute a never-visited state into the residual stream and
        read what the model would say from there. This is the mechanism behind exploring
        where a model has never been -- and it is honest about being only a mechanism:
        it returns distributions, and scores no novelty or soundness. On a trained model
        these are worth reading; on a random one they are noise.
        See holographic_voidmanifold.void_probe."""
        from holographic.agents_and_reasoning.holographic_voidmanifold import void_probe
        return void_probe(runtime, layer, basis, mean, void_points, token_ids,
                          hooks=hooks)

    def unicron_carrier(self, healthy_hiddens, reserve=16, amplitude=0.5):
        """THE RESIDUAL STREAM IS A BUS: every block computes h = h + f(h), so a vector
        injected at one layer is still there at the next. MEASURED: a payload written at
        layer 1 was read back EXACTLY at layer 3 (cosine 1.0000). This carrier reserves
        the stream's lowest-energy directions and runs leCore's role-filler binding
        there -- the model keeps computing in its subspace, leCore keeps exact
        structured state in the complement, both on the same bus, with readout by
        UNBINDING (no training, no sparse autoencoder, no approximation).
        HONEST: capacity and interference are a measured TRADE, and .report() states it
        -- on the tiny reference model, 32 reserved dims borrow 15.6% of stream energy
        for 0.22 relative logit change. Capacity grows with dimension and interference
        with borrowed energy, so a real 1024-dim concentrated stream should trade far
        better -- a PREDICTION to measure, not a result.
        See holographic_carrier.StreamCarrier."""
        from holographic.agents_and_reasoning.holographic_carrier import StreamCarrier
        return StreamCarrier(healthy_hiddens, reserve=reserve, amplitude=amplitude)

    def unicron_forward_embeds(self, runtime, embeds, hooks=None, step_hooks=None):
        """Run a model from HIDDEN STATES rather than token ids -- superpositions,
        interpolations, steered or synthesized states, anything that is not a single
        token. Asserted to be EXACTLY forward() when handed the embeddings it would have
        looked up, because without that guarantee every experiment above the token layer
        measures the plumbing instead of the idea (measured: an early superposed-decoding
        run silently re-tokenized its own input and looked like a failure of the method).
        See holographic_gdnruntime.GDNRuntime.forward_embeds."""
        return runtime.forward_embeds(embeds, hooks=hooks, step_hooks=step_hooks)

    def unicron_layer_schedule(self, runtime, schedule=None):
        """RUN THE SAME WEIGHTS AS A DIFFERENT ARCHITECTURE -- instantly, no re-export.
        `schedule` is the list of layer indices to execute, in order, with repeats
        allowed: [0,1,2,1,2,3] is SOLAR/Goliath-style depth up-scaling, [0,1,2,2,3] is
        layer recursion, [0,2,3] is pruning. Owning the forward pass turns architecture
        surgery into a list instead of a checkpoint rebuild.

        MEASURED on the trained subject (dense baseline ppl 4.9969), and the numbers are
        the point rather than the pitch: depth up-scaling COSTS perplexity without
        training -- repeat-middle +8.7%, repeat-all +9.9%, single-layer recursion +8.5%,
        dropping a layer +91%. That reproduces exactly what the frankenmerge literature
        reports: the initial merge is worse and continued pretraining is what recovers
        it.
        INFERENCE-TIME HEALING, measured and only partly successful: re-aligning the
        stream to the distribution a repeated layer normally sees recovers 5.4195 ->
        5.3473 on single-layer recursion (~13% of the loss) and does NOT help full
        duplication. Use `step_hooks` on runtime.forward to target repeated occurrences.
        KEPT NEGATIVE: hooks keyed by LAYER heal the legitimate first pass too, which
        made every schedule worse until the runtime grew step-keyed hooks."""
        if schedule is None:
            runtime.cfg.pop("layer_schedule", None)
            return list(range(runtime.cfg["n_layers"]))
        runtime.cfg["layer_schedule"] = [int(i) for i in schedule]
        return list(runtime.cfg["layer_schedule"])

    def unicron_screen_routing(self, runtime, block=32, blocks=2, window=32,
                               accumulators=1, mode=None, clusters=50, topk=8,
                               rank=8, enable=True):
        """READ THE BOUNDARY, NOT THE VOLUME. mode="ball" is the strong version and
        should be preferred: it BEATS the centroid screen it replaced.

        Why the centroid was beatable at all: it ranks a block by its MEAN inner
        product while routing needs the MAX, so it is a heuristic that silently misses.
        mode="ball" groups keys by SIMILARITY (deterministic k-means) and uses the
        admissible bound max q.k <= q.c + r||q|| to skip clusters that PROVABLY cannot
        hold a top-k key -- a certificate, not a guess.
        MEASURED head to head on the trained subject: ball at 80 clusters gives EXACT
        top-8 for 100% of queries while scoring 38.5% of keys; the centroid screen gives
        0.87 recall at 80%. End to end at 50 clusters: top-1 agreement 1.0000 and
        perplexity 4.9957 against a dense 4.9969, versus 0.9975 / 5.0113 for the
        centroid.
        rank>0 adds the BOUNDARY READ: all keys share one low-rank shell, so a query is
        projected into it ONCE and every score becomes an r-dim dot against stored
        coordinates -- the key is never read. Each key's TAIL NORM certifies the read
        (|approx - true| <= tail*||q||), so only keys whose upper bound can crack the
        running top-k are rescored exactly. MEASURED: exact top-8 at 33.5% of dense
        flops (rank 8) versus 38.5% for the bound alone. Per-cluster bases were refused
        -- an ~8-key cluster cannot amortize its own r*d projection; the shared shell
        can, because it is projected once for the whole volume. Partition the KV volume into blocks, give
        each a fixed-size screen (its key centroid), score the SCREENS (T/block work),
        then pay full attention price only inside the few blocks the screens point at,
        plus a recent window. This is the boundary/volume accounting turned into a
        shortcut: the information is concentrated (90% of softmax mass in ~6% of keys),
        so a summary can find it without scanning everything.
        MEASURED on the trained subject at 400 tokens: 38% of keys scored -> 0.998 top-1
        agreement (+0.26% perplexity); 26% -> 0.983. Allowing every block reproduces
        dense attention to 6e-15 (the null test).
        RULE-0 FAILURE ON RECORD: the first screen bundled all 400 keys into one 512-dim
        vector and scored recall 0.19. mind.bundle_capacity(dim=512) answers 87 items at
        F1 1.0 -- the engine would have said "4.6x over capacity" BEFORE the build, and
        hierarchical_recall's own docstring reports 18.3% for flat recall against 100%
        with cleanup between levels. That 18.3% is the 0.19 I measured. The lesson is not
        that the idea was wrong; it is that the capacity law is a FACULTY, not folklore,
        and it was one call away.
        LEVERS THEN WALKED, measured: hierarchy (each level sized under capacity) and
        `accumulators` (lever 4 -- r summaries per block, filled round-robin, scored by
        best-match) lifted recall@8 0.667 -> 0.698 tight / 0.858 -> 0.871 loose.
        AUDITED NEGATIVE, per the engine's own law about recording where the fancy tech
        does not apply: an HRR bundle never beat a plain key CENTROID at any setting
        (0.789 vs 0.797), because this task is SUMMARIZATION FOR RANKING, not storage and
        exact retrieval. VSA earns its place where binding and clean readout are needed;
        here neither was.
        Set enable=False to restore dense attention. See holographic_gdnruntime."""
        if not enable:
            runtime.cfg.pop("attn_screen", None)
            return None
        if str(mode) == "ball":
            runtime.cfg["attn_screen"] = {"mode": "ball", "clusters": int(clusters),
                                          "topk": int(topk), "window": int(window),
                                          "rank": int(rank)}
        else:
            runtime.cfg["attn_screen"] = {"block": int(block), "blocks": int(blocks),
                                          "window": int(window),
                                          "accumulators": int(accumulators)}
        return dict(runtime.cfg["attn_screen"])

    def unicron_capacity_report(self, runtime, token_ids,
                                marks=(8, 16, 32, 64, 128, 256)):
        """BOUNDARY vs VOLUME accounting for a model -- which account is actually doing
        the long-range work.

        A recurrent model has a literal boundary: the state S, through which every token
        of history must reach the future, at FIXED size. The KV cache is the volume term,
        growing linearly and read quadratically. This measures the boundary's size, how
        much of its own dimension it USES (participation ratio), and its CAUSAL MEMORY
        HORIZON -- perturb one token, see how far ahead the state still differs. That
        last number is the honest answer to "how much context does this model actually
        use through its state", as distinct from the window it advertises.
        MEASURED on the trained subject at 512 tokens: boundary 2048 numbers/layer using
        7% of its own dimension, a one-token change stops reaching the state after ~32
        tokens, and the KV volume is 11x the boundary -- so essentially all long-range
        capability is being bought the expensive way.
        The physics analogy is STRUCTURAL only; nothing here computes an entropy bound.
        The measurement is the point. See holographic_holocap.capacity_report."""
        from holographic.io_and_interop.holographic_holocap import capacity_report
        return capacity_report(runtime, token_ids, marks=tuple(marks))

    def unicron_memory_horizon(self, runtime, token_ids,
                               marks=(8, 16, 32, 64, 128, 256), position=0):
        """How far back does a model's RECURRENT STATE actually remember? Change one
        token, then measure how far into the future the state still differs. Where it
        reaches zero, the boundary is carrying nothing -- a hard statement, since past
        that point the state is bit-identical whether or not the token existed.
        See holographic_holocap.memory_horizon."""
        from holographic.io_and_interop.holographic_holocap import memory_horizon
        return memory_horizon(runtime, token_ids, marks=tuple(marks),
                              position=position)

    def unicron_attention_waste(self, runtime, token_ids, layer=None,
                                fractions=(0.9, 0.95, 0.99)):
        """HOW MUCH OF ATTENTION IS WASTE? Measures how few keys actually carry the
        softmax mass, and what happens if the rest are dropped.

        The question is not new -- Attention has been shown to approximate Kanerva's
        Sparse Distributed Memory (1988), which is the Marr (1969) / Albus (1971)
        cerebellum model, and SDM reads only the locations inside a RADIUS. Attention
        softmaxes over every key instead. MEASURED on a trained subject over 400
        positions: 90% of the mass sits in a median of 23 keys, the top key alone
        carries 41%, and keeping 32 of 400 (8%) preserves 0.993 top-1 agreement at
        +0.17% perplexity.
        Set runtime.cfg["attn_top_k"] to apply the radius (default off, bit-identical).
        HONEST: this measures redundancy, it does not yet bank the saving -- scores are
        computed then masked. Cashing it needs an index that finds the top keys without
        scoring the rest, which is precisely what SDM's addressing does."""
        import numpy as _np
        base = runtime.forward(token_ids)
        btop = _np.argmax(base, -1)
        report = {"n_positions": len(token_ids), "sweep": []}
        for k in (128, 64, 32, 16, 8, 4):
            if k >= len(token_ids):
                continue
            runtime.cfg["attn_top_k"] = int(k)
            try:
                o = runtime.forward(token_ids)
            finally:
                runtime.cfg.pop("attn_top_k", None)
            report["sweep"].append({
                "keys": int(k),
                "fraction_of_context": k / float(len(token_ids)),
                "top1_agreement": float(_np.mean(_np.argmax(o, -1) == btop)),
                "rel_logit_error": float(_np.max(_np.abs(o - base))
                                         / _np.max(_np.abs(base)))})
        return report

    def unicron_leap(self, runtime, token_ids, n_new=32, memory=None, k=4,
                     hooks=None, learn=True):
        """GENERATE FASTER THAN THE MODEL ALONE, with output PROVABLY identical to
        greedy decoding. leCore learns the routes the model walks (an online n-gram
        route memory), drafts the next k tokens for free, and verifies them in ONE
        batched pass (runtime.extend -- a GEMM over the chunk where normal generation
        does k GEMVs). Only the longest provably-correct prefix is accepted, so a bad
        drafter can waste time but can NEVER change the output -- asserted against a
        hostile always-wrong drafter in the selftest. MEASURED (token-identity enforced,
        mean of 3): 1.6x-3.0x at prompt 32 and 1.3x-1.9x at prompt 128 as k goes 2->16,
        100% acceptance on a walked route; on NOVEL text acceptance falls to ~0 and the
        wasted verification makes it SLOWER -- the win is a property of the text
        repeating, not of the drafter's cleverness. Returns (ids, memory, report).
        See holographic_leap.leap_generate."""
        from holographic.agents_and_reasoning.holographic_leap import leap_generate
        return leap_generate(runtime, token_ids, n_new=n_new, memory=memory,
                             k=k, hooks=hooks, learn=learn)

    def unicron_verified_generate(self, runtime, token_ids, evidence, n_new=12,
                                  k=4, max_retries=4, hooks=None):
        """FACT-CHECK BEFORE EMITTING: propose a continuation, verify every span against
        evidence, veto the exact token that broke grounding, and re-propose FROM THE SAME
        SNAPSHOT. An agent harness runs this loop by emitting tokens, parsing them and
        calling the model again -- which re-prefills the whole context every round, the
        dominant cost of agent loops in practice. Here a rejected proposal costs one
        verification pass and the retry resumes from state that was never spent: no
        re-prefill, no tokens crossing the boundary, no second model to judge the first.
        MEASURED against a re-prefilling loop: 1.9x at prompt 32 (8 rounds), 3.9x at 128,
        6.5x at 512 -- the gap grows with context, exactly where harnesses hurt.
        `evidence` is an EvidenceStore of allowed token spans. Returns (ids, report);
        an honest exhaustion beats a confident fabrication.
        See holographic_swarm.verified_generate."""
        from holographic.agents_and_reasoning.holographic_swarm import verified_generate
        return verified_generate(runtime, token_ids, evidence, n_new=n_new,
                                 k=k, max_retries=max_retries, hooks=hooks)

    def unicron_evidence(self, sequences=(), span=3):
        """Build the evidence store the fact-check gate verifies against: allowed token
        spans from retrieved passages or source documents. Exact and model-free -- a
        fact-checker that needs a language model to judge a language model is a regress.
        See holographic_swarm.EvidenceStore."""
        from holographic.agents_and_reasoning.holographic_swarm import EvidenceStore
        return EvidenceStore(sequences, span=span)

    def unicron_swarm(self, runtime, members, layer=3, horizon=4, gain=1.0,
                      digest="contrast", max_depth=2):
        """A SUBCONSCIOUS: many inner agents deliberate BETWEEN tokens by forking the
        model's own inference state, and only their DIGEST reaches its thinking -- the
        monologue is never emitted. Unlike ordinary multi-agent (separate chats pasted
        back into a prompt), branches are forks of the same mind at the same moment and
        return a residual-stream delta, not text. Members may themselves carry swarms
        (nested, bounded by max_depth -- cost multiplies per level, measured).
        digest='contrast' is provably SILENT when members agree. Pair with
        unicron_swarm_mind. See holographic_swarm.SwarmResident."""
        from holographic.agents_and_reasoning.holographic_swarm import SwarmResident
        return SwarmResident(runtime, members, layer=layer, horizon=horizon,
                             gain=gain, digest=digest, max_depth=max_depth)

    def unicron_swarm_mind(self, runtime, swarm, guards=(), vote_strength=1.0):
        """The outer loop over a subconscious: emits tokens while the swarm deliberates
        between them. vote_strength is in units of the model's OWN decision margin
        (0 = silent, 1 = can close a decided gap, >1 = can overrule) -- because an
        influence with an arbitrary magnitude is either silent or dictatorial depending
        on a model's embedding scale, and both look like success from outside.
        Reports .influenced (how often the subconscious actually changed the token).
        See holographic_swarm.SwarmMind."""
        from holographic.agents_and_reasoning.holographic_swarm import SwarmMind
        return SwarmMind(runtime, swarm, guards=guards, vote_strength=vote_strength)

    def unicron_galvatron(self, runtime, residents=(), guards=()):
        """REBUILD a model into a Galvatron: the runtime plus a stack of leCore
        residents living in its forward pass. Residents see the live residual stream
        each token (OracleResident: the mind's native learn/recall as editable perfect
        memory; DreamerResident: subspace thought-repair that provably never touches a
        clean stream); guards reshape the logits (WardResident: hard bans/whitelists --
        contracts, not prompts). Returns a Galvatron with .generate(). All contracts
        measured in holographic_galvatron's selftest, including under composition."""
        from holographic.agents_and_reasoning.holographic_galvatron import Galvatron
        return Galvatron(runtime, residents=residents, guards=guards)

    def unicron_council(self, runtime, token_ids, branches, n_new=12, horizon=8):
        """Deliberation over branched futures: snapshot the InferenceState, run each
        (residents, guards) branch from its own copy, score each by the model's OWN
        mean next-token NLL under that branch's rules, return ranked best-first.
        Self-consistency without a second model, built on snapshot/branch temporal
        awareness. See holographic_galvatron.council."""
        from holographic.agents_and_reasoning.holographic_galvatron import council
        return council(runtime, token_ids, branches, n_new=n_new, horizon=horizon)

    def unicron_generator_audit(self, tensor):
        """Is a tensor's generator DISCOVERABLE? Delegates to HRNN's compressibility
        gate. Measured answer for both seed-born and trained weights: NO -- which is
        precisely why unicron_archive's RECIPE rung takes caller-supplied provenance
        and hash-verifies it, instead of searching for seeds no measurement could
        confirm. See holographic_unicron.generator_audit."""
        from holographic.io_and_interop import holographic_unicron as _u
        return _u.generator_audit(tensor)

    def unicron_archive(self, models, reference=None, recipes=None):
        """Archive a FLEET of models with leCore's storage ladder, per tensor: SAME
        (pointer to reference), RECIPE (seed/generator instead of data, hash-verified),
        DELTA (exact XOR-delta vs reference, zlib'd -- the task-vector insight applied
        to storage), RAW (the honesty rung). Reconstruction is BIT-exact. Kept
        negatives on record: trained weights are never seed-searched, and arithmetic
        float deltas are not bit-exact (XOR is). Returns (archive, report with per-rung
        counts and the measured ratio). See holographic_unicron.archive_models."""
        from holographic.io_and_interop import holographic_unicron as _u
        return _u.archive_models(models, reference=reference, recipes=recipes)

    def unicron_restore(self, archive, name):
        """Bit-exact reconstruction of one model from a unicron_archive.
        See holographic_unicron.restore_model."""
        from holographic.io_and_interop import holographic_unicron as _u
        return _u.restore_model(archive, name)

    def unicron_shelve(self, model, label):
        """SEMANTIC model memory: fingerprint a model (the FHRR bundle over layer
        roles) and learn it in the mind under `label`. Models become first-class
        holographic objects the mind can recognize -- data and identity in the same
        composable space."""
        from holographic.io_and_interop import holographic_unicron as _u
        if isinstance(model, str):
            model = _u.load_model(model)
        fp = _u.fingerprint(_u.analyze_model(model), dim=self.dim)
        self.learn(np.real(fp), label)          # mind memory is real-valued
        return {"label": label, "dim": int(self.dim)}

    def unicron_identify(self, model):
        """WHICH model is this? Fingerprint the mystery checkpoint and recall against
        every shelved model -- lineage recognition by content, robust to small edits
        (fingerprints are bundles; perturbation moves them little). Returns the
        recalled (label, confidence)."""
        from holographic.io_and_interop import holographic_unicron as _u
        if isinstance(model, str):
            model = _u.load_model(model)
        fp = _u.fingerprint(_u.analyze_model(model), dim=self.dim)
        r = self.recall(np.real(fp))
        label = r[0][0] if isinstance(r, tuple) and isinstance(r[0], tuple) else r
        conf = float(r[1]) if isinstance(r, tuple) and len(r) > 1 else 1.0
        return {"label": label, "confidence": conf}

    def unicron_report(self, model, sample_layers=8, candidate_bases=None,
                       roles=("mlp.gate_proj.weight", "self_attn.q_proj.weight")):
        """ONE CALL, THE WHOLE PICTURE -- the front door over the entire Unicron arc.
        Hand it a checkpoint and get: a spectral regime census (which layers even have a
        filterable gap), blind head structure, per-role depth redundancy, optional
        lineage detection, a RANKED list of size levers each carrying its measured
        evidence, and the REFUTATIONS -- loudly. The refuted levers ship with every
        report on purpose: a report that lists only what might work is how someone
        retries MP filtering on a heavy-tailed model, which is exactly what produced the
        measured 256-newline collapse. See holographic_unicron.full_report."""
        from holographic.io_and_interop import holographic_unicron as _u
        cands = None
        if candidate_bases:
            cands = {n: (_u.load_model(c) if isinstance(c, str) else c)
                     for n, c in candidate_bases.items()}
        return _u.full_report(model, sample_layers=sample_layers,
                              roles=tuple(roles), candidate_bases=cands)

    def unicron_lineage(self, model, candidates, k=64):
        """WHICH BASE was this fine-tune derived from? Ranked from WEIGHT EVIDENCE alone
        -- no model cards, no metadata. Scores candidates by principal-angle overlap of
        leading singular subspaces, which survives the small rotations a fine-tune
        induces. Returns the ranking, the winner, and the MARGIN over the runner-up (a
        lineage call with no margin is a guess, and you should be able to see that).
        Answers the "missing lineage metadata" limitation named in TStore
        (arXiv 2604.17104) -- correct pairing is what makes delta storage possible.
        See holographic_unicron.delta_lineage."""
        from holographic.io_and_interop import holographic_unicron as _u
        if isinstance(model, str):
            model = _u.load_model(model)
        cands = {n: (_u.load_model(c) if isinstance(c, str) else c)
                 for n, c in candidates.items()}
        return _u.delta_lineage(model, cands, k=k)

    def unicron_delta_store(self, base, finetuned, energy=0.9999, bits=8,
                            mode="lowrank"):
        """Store a fine-tune as a DELTA rather than a second model. Unchanged tensors
        cost ZERO; touched ones go low-rank at a rank discovered from the delta's own
        spectrum; a fat delta stays dense rather than paying factor overhead (earn your
        bytes). mode="qlr" uses the D-QRELO recipe (arXiv 2604.16940): one-bit dominant
        structure plus low-rank on the smaller residual, which the literature reports is
        more robust for LARGE-SFT deltas; both modes ship, priced per subject. NOTE THE REVERSAL: low-rank lost to plain quantization four times on
        trained WEIGHTS -- but a delta is not a trained matrix, it is the residue of one
        task's learning, and it is structurally thin. Measured: exactly rank-8 of 60 on
        a learning instrument, lossless, 5.4x vs dense. Pair with unicron_delta_apply.
        See holographic_unicron.delta_encode."""
        from holographic.io_and_interop import holographic_unicron as _u
        ms = [(_u.load_model(x) if isinstance(x, str) else x)
              for x in (base, finetuned)]
        return _u.delta_encode(ms[0], ms[1], energy=energy, bits=bits, mode=mode)

    def unicron_delta_apply(self, base, delta, scale=1.0):
        """Rebuild a fine-tune from base + stored delta. scale<1 interpolates between
        the two models (the same knob task arithmetic uses); scale=0 returns the base
        exactly. See holographic_unicron.delta_apply."""
        from holographic.io_and_interop import holographic_unicron as _u
        if isinstance(base, str):
            base = _u.load_model(base)
        return _u.delta_apply(base, delta, scale=scale)

    def unicron_taskvector(self, base, finetuned):
        """EXTRACT a capability from a fine-tune as an object: tau = finetuned - base, per
        tensor. The weight-space form of drift-model algebra, on models themselves. Paths
        or weight dicts. See holographic_unicron.task_vector."""
        from holographic.io_and_interop import holographic_unicron as _u
        ms = [(_u.load_model(m) if isinstance(m, str) else m) for m in (base, finetuned)]
        return _u.task_vector(ms[0], ms[1])

    def unicron_imbue(self, target, tau, scale=1.0, policy=True, out_path=None):
        """WRITE a capability INTO a model (the Galvatron operation): target + scale*tau.
        Grounded in task arithmetic (Ilharco et al. ICLR 2023). THE LINEAGE LAW, measured:
        deltas are basis-bound -- donor and target must share the SAME base checkpoint, or
        the transplant scrambles instead of transferring (pinned negative in the module
        selftest). policy=True never writes embeddings/norms/visual/mtp. Output is UNVERIFIED
        until the caller's eval runs -- doubly so here. See holographic_unicron.imbue."""
        from holographic.io_and_interop import holographic_unicron as _u
        import numpy as _np
        if isinstance(target, str):
            target = _u.load_model(target)
        if isinstance(tau, tuple):
            tau = tau[0]
        out = _u.imbue(target, tau, scale=scale, policy=policy)
        if out_path:
            _u.save_safetensors(out_path, {k: _np.ascontiguousarray(v)
                                           for k, v in out.items()})
        return out

    def unicron_heads(self, matrix, candidates=(2, 4, 8, 16, 32)):
        """BLIND head-count discovery for a projection matrix: reshape candidates scored by
        two agreeing instruments -- analyze_axes must call the head axis an index/carrier,
        and the per-slice stable-rank ELBOW marks where merging heads doubles rank but
        splitting one leaves it flat. Kept negative on record: demux_series is the wrong
        tool (heads are blocks, not strides). See holographic_unicron.head_structure."""
        from holographic.io_and_interop import holographic_unicron as _u
        import numpy as _np
        return _u.head_structure(_np.asarray(matrix), candidates=tuple(candidates))

    def unicron_depthshare(self, model, role_suffix="mlp.gate_proj.weight", min_dim=8):
        """HOW MUCH of a model is depth-REPEATED structure? Stacks every layer's matrices for
        one role (name suffix) and reads the layer-mode spectrum via holographic_tucker's
        unfolding: shared_frac near 1 = one matrix wearing L costumes (a real structural-
        compression lever: shared basis + per-layer cores); near the 1/L chance floor =
        depth is NOT redundant. A measurement of the wasteful-structure hypothesis, per
        role. Accepts a path, weights dict, or a plain list of matrices.
        See holographic_unicron.depth_sharing."""
        from holographic.io_and_interop import holographic_unicron as _u
        import numpy as _np
        if isinstance(model, (list, tuple)):
            return _u.depth_sharing(model)
        if isinstance(model, str):
            model = _u.load_model(model)
        import re as _re
        picked = []
        for name in sorted(model, key=lambda k: [int(x) if x.isdigit() else x
                                                 for x in _re.split(r"(\d+)", k)]):
            t = _np.asarray(model[name])
            # same policy gate as assimilation: visual/mtp matrices would
            # contaminate a language-stack depth measurement (caught live: the
            # rehearsal's mtp stub matched the suffix and made n_layers 9 of 8)
            if name.endswith(role_suffix) and t.ndim == 2 \
                    and min(t.shape) >= min_dim \
                    and not any(pat in name.lower() for pat in ("visual", "mtp")):
                picked.append(t)
        if len(picked) < 2:
            raise ValueError("fewer than 2 layers matched role suffix %r" % role_suffix)
        out = _u.depth_sharing(picked)
        out["role_suffix"] = role_suffix
        return out

    def unicron_localize(self, matrix, k=10):
        """WHERE does the learned information live in a weight matrix? Porter-Thomas test
        (Thamm/Staats/Rosenow PRE 2022): noise singular vectors have Gaussian entries (IPR at
        3/n); learned vectors LOCALIZE on the coordinates that matter. Reports per-vector IPR
        and kurtosis against the Gaussian baseline. See holographic_unicron.vector_localization."""
        from holographic.io_and_interop import holographic_unicron as _u
        import numpy as _np
        return _u.vector_localization(_np.asarray(matrix), k=k)

    def unicron_filter(self, matrix, keep=None, mode="truncate"):
        """DENOISE a weight matrix the RMT way: keep spectral outliers, discard the
        Marchenko-Pastur bulk (Staats/Thamm/Rosenow PRE 2023 -- the principled noise/information
        cut; most of a trained spectrum is still initialization noise). mode="shrink" debiases
        kept spikes by the noise floor. Distinct from mind.denoise (manifold projection of
        hypervectors) and Tucker/TT (no noise model). Returns (filtered, info).
        See holographic_unicron.rmt_filter."""
        from holographic.io_and_interop import holographic_unicron as _u
        import numpy as _np
        return _u.rmt_filter(_np.asarray(matrix), keep=keep, mode=mode)

    def unicron_trajectory(self, checkpoints, dim=1024, min_dim=8):
        """READ A TRAINING RUN: per-checkpoint fingerprints, step cosines, cosine-from-start,
        per-layer metric time-series. Checkpoints are paths / weight dicts / analyze results,
        in time order. Theory anchor: singular values under SGD follow Dyson Brownian motion
        toward bulk+tail (Olsen et al. 2507.12709) -- a mid-run step-cosine drop marks a regime
        change. See holographic_unicron.checkpoint_trajectory."""
        from holographic.io_and_interop import holographic_unicron as _u
        outs = []
        for m in checkpoints:
            if isinstance(m, str):
                m = _u.load_model(m)
            if isinstance(m, dict) and "layers" not in m:
                m = _u.analyze_model(m, min_dim=min_dim)
            outs.append(m)
        return _u.checkpoint_trajectory(outs, dim=dim)

    def unicron_subspace(self, matrix_a, matrix_b, k=8, side="left"):
        """DO two weight matrices encode in the SAME DIRECTIONS? Principal-angle cosines
        between their top-k singular subspaces (Bjorck-Golub, exact) with the k/n chance
        floor reported -- two layers can share every scalar spectral statistic and still
        be orthogonal; this is the metric that sees it. See holographic_unicron.subspace_overlap."""
        from holographic.io_and_interop import holographic_unicron as _u
        return _u.subspace_overlap(matrix_a, matrix_b, k=k, side=side)

    def unicron_compare(self, model_a, model_b, min_dim=8, subspace_k=None):
        """COMPARE two trained models: matched-layer spectral deltas (b - a) + fingerprint cosine.
        subspace_k (default OFF) adds per-layer principal-angle subspace overlap vs its chance
        floor -- the direction-level distillation check scalar metrics cannot see.
        The distillation audit: a student inheriting the teacher's function drifts toward the
        teacher's spectral structure; noise deltas mean it is memorising, not inheriting.
        Inputs are paths, weight dicts, or unicron_analyze results.
        See holographic_unicron.compare_models."""
        from holographic.io_and_interop import holographic_unicron as _u
        out, raw = [], []
        for m in (model_a, model_b):
            if isinstance(m, str):
                m = _u.load_model(m)
            raw.append(m if isinstance(m, dict) and "layers" not in m else None)
            if isinstance(m, dict) and "layers" not in m:
                m = _u.analyze_model(m, min_dim=min_dim)
            out.append(m)
        result = _u.compare_models(out[0], out[1])
        if subspace_k and raw[0] is not None and raw[1] is not None:
            # direction-level check needs the WEIGHTS, not just the reports
            sub = {}
            for name in result["layer_deltas"]:
                if name in raw[0] and name in raw[1]:
                    import numpy as _np
                    Wa = _np.asarray(raw[0][name]); Wb = _np.asarray(raw[1][name])
                    sub[name] = _u.subspace_overlap(Wa.reshape(Wa.shape[0], -1),
                                                    Wb.reshape(Wb.shape[0], -1),
                                                    k=int(subspace_k))
            result["subspace"] = sub
        return result
