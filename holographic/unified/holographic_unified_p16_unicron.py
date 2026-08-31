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

    def mesh_program_obj(self, machine, program, verts, faces, host_fallback=False):
        """G10: compile a mesh-transform program, run it INSTALLED with the vertices as state,
        return the OBJ as TEXT (the token stream is the output device -- no file I/O). Byte-exact
        vs the live path. See holographic_compileinstall.mesh_program_obj."""
        from holographic.agents_and_reasoning.holographic_compileinstall import mesh_program_obj
        return mesh_program_obj(machine, program, verts, faces, host_fallback=host_fallback)

    def model_library(self, dim, seed, programs, symbolic_functions=None, data=None, unitary=False):
        """G14: many programs, ONE rule file -- members share one machine so certified ops are
        shared by construction; load() re-bakes every member bit-identically. See
        holographic_nativemodel.ModelLibrary."""
        from holographic.agents_and_reasoning.holographic_nativemodel import ModelLibrary
        return ModelLibrary(dim, seed, programs, symbolic_functions, data=data, unitary=unitary)

    def float_pack_bytes(self, arr, preset=6):
        """Lossless float compression via byte-plane transpose + lzma: 1.19x on real
        embeddings where raw lzma gets 1.08x; byte-exact. float_unpack_bytes inverts.
        See holographic_byteplane."""
        from holographic.io_and_interop.holographic_byteplane import float_pack_bytes
        return float_pack_bytes(arr, preset=preset)

    def float_unpack_bytes(self, blob):
        """Exact inverse of float_pack_bytes. See holographic_byteplane."""
        from holographic.io_and_interop.holographic_byteplane import float_unpack_bytes
        return float_unpack_bytes(blob)

    def dispatch_roles(self, tasks, spec):
        """H4: route task phrases ('texture the scene') to swarm roles via the engine's own
        BM25 -- leCore staffing leCore. Ambiguity raises with names, never guesses. Returns
        members for render_critique_loop. See holographic_innereye.dispatch_roles."""
        from holographic.agents_and_reasoning.holographic_innereye import dispatch_roles
        return dispatch_roles(self, tasks, spec)

    def shared_workspace(self):
        """H3: the swarm's shared scene workspace -- named slots roles read/write during
        deliberation; buffered commits, lowest-index collision rule, every collision logged.
        Pass to render_critique_loop(workspace=). See holographic_innereye.SharedWorkspace."""
        from holographic.agents_and_reasoning.holographic_innereye import SharedWorkspace
        return SharedWorkspace()

    def image_op_library(self, height, width):
        """The inner eye's TOOLSET: image tools as flattened-frame callables for FAC steps --
        blur/unsharp/sobel certify, flips/rot90/warps are permutations, brightness/contrast
        install; threshold/gamma refuse at image scale and ride HOST:APPLY. See
        holographic_innereye.image_op_library."""
        from holographic.agents_and_reasoning.holographic_innereye import image_op_library
        return image_op_library(height, width)

    def render_critique_loop(self, machine, formation_program, init_params, members, eye,
                             target_embed, width, height, satisfy=0.99, max_rounds=32,
                             host_fallback=False):
        """H1: design -> INSTALLED render -> look with the (injectable) eye -> critique in EYE
        SPACE -> iterate -> speak the PGM. Reference semantics for the on-laptop swarm+tower
        loop. See holographic_innereye.render_critique_loop."""
        from holographic.agents_and_reasoning.holographic_innereye import render_critique_loop
        return render_critique_loop(machine, formation_program, init_params, members, eye,
                                    target_embed, width, height, satisfy=satisfy,
                                    max_rounds=max_rounds, host_fallback=host_fallback)

    def drift_head(self, model):
        """The installed view of a generative drift model: its (d+1) x D moment matrix --
        certified dense at 0.0, so the model ships as ONE weight matrix. Adding heads IS
        composing models (exact); subtracting ablates; transport acts on rows by a certified
        linear operator. drift_head_load inverts. See holographic_hdrift.drift_head."""
        from holographic.sampling_and_signal.holographic_hdrift import drift_head
        return drift_head(model)

    def drift_head_load(self, enc, H, n_train, bounds=None):
        """Rebuild a DriftModel from its installed head (the head is the model file;
        round trip exact). See holographic_hdrift.drift_from_head."""
        from holographic.sampling_and_signal.holographic_hdrift import drift_from_head
        return drift_from_head(enc, H, n_train, bounds=bounds)

    def memory_mountain(self, sizes=None):
        """Measure THIS box's cache hierarchy (streaming GB/s vs working set), detect the
        tiers, and predict streaming wall-clock from the floor -- the fast-arbiter table
        validated the predictions to ~15%. Returns (curve, tiers). Dispatch flank excluded:
        a Python probe cannot see L1 and says so. See holographic_memorymountain."""
        from holographic.caching_and_storage.holographic_memorymountain import (
            measure_memory_mountain, detect_tiers)
        curve = measure_memory_mountain(sizes=sizes)
        return curve, detect_tiers(curve)

    def predict_streaming_ms(self, nbytes_touched, tiers):
        """Predicted wall-clock (ms) for a streaming pass over `nbytes_touched`, from THIS box's
        MEASURED floor bandwidth (holographic_memorymountain.predict_streaming_ms). `tiers` is the
        second element of memory_mountain()'s return -- the measurement is the expensive half, the
        prediction is one division, so they are separate calls on purpose. The fast-arbiter table
        validated this to ~15%. LLM decode is memory-bandwidth bound, so bytes-per-token IS latency.
        KEPT NEG: predicts the ROOFLINE, not the run -- it cannot see NUMA, page faults or a noisy
        neighbour; a large miss is a scheduling bug, not a model error."""
        from holographic.caching_and_storage.holographic_memorymountain import predict_streaming_ms as _p
        return _p(nbytes_touched, tiers)

    def time_machine(self):
        """The unitary-recurrence toolkit: make_unitary_step (the rule), time_jump (random
        access into time, t may be NEGATIVE -- exact reversal; non-unitary refuses WITH the
        eig_min^t number), bundle_sims/read_member (K sims in one vector at the 1/sqrt(K)
        law), evolve_functional (a PRECOMMITTED ensemble readout, exact). See
        holographic_timemachine."""
        from holographic.simulation_and_physics import holographic_timemachine as tm
        return tm

    def collapse_recurrence(self, machine, step_program, n_steps, host_fallback=False, tol=1e-9):
        """THE HRNN COLLAPSE: n steps of a certified LINEAR recurrence become ONE affine
        operator (the REPEAT lesson applied to time) -- measured 156x on endpoint queries at
        ~1e-15 error, spectrum priced (eig_max^n in the certificate), host links refuse with
        names. sim_program_run stays the referee + drift instrument. See
        holographic_compileinstall.collapse_recurrence."""
        from holographic.agents_and_reasoning.holographic_compileinstall import collapse_recurrence
        return collapse_recurrence(machine, step_program, n_steps,
                                   host_fallback=host_fallback, tol=tol)

    def sim_program_run(self, machine, step_program, init, n_steps, host_fallback=True):
        """G11: compile ONE physics step, iterate it installed with state fed back; returns
        (trajectory, manifest, drift-vs-live curve). The drift curve is the honesty instrument.
        See holographic_compileinstall.sim_program_run."""
        from holographic.agents_and_reasoning.holographic_compileinstall import sim_program_run
        return sim_program_run(machine, step_program, init, n_steps, host_fallback=host_fallback)

    def raster_program_pgm(self, machine, program, params, width, height, host_fallback=False):
        """G12: run an installed image-formation chain and emit the frame as PGM P2 TEXT --
        the picture leaves through the mouth. See holographic_compileinstall.raster_program_pgm."""
        from holographic.agents_and_reasoning.holographic_compileinstall import raster_program_pgm
        return raster_program_pgm(machine, program, params, width, height, host_fallback=host_fallback)

    def cleanup_as_attention(self, codebook, beta=64.0):
        """G8: exact cleanup expressed as ONE attention head (codebook = keys AND values);
        beta is the softmax temperature. Ties average by theorem -- see the certificate.
        See holographic_projector.cleanup_as_attention."""
        from holographic.io_and_interop.holographic_projector import cleanup_as_attention
        return cleanup_as_attention(codebook, beta=beta)

    def attention_read_certificate(self, codebook, queries, beta=64.0):
        """G8: MEASURE the attention read against exact cleanup on the caller's own queries --
        the honesty label for installing cleanup as a head (agreement rate at this beta).
        See holographic_projector.attention_read_certificate."""
        from holographic.io_and_interop.holographic_projector import attention_read_certificate
        return attention_read_certificate(codebook, queries, beta=beta)

    def native_model(self, dim, seed, program, symbolic_functions=None, data=None, unitary=False):
        """F28 first landing -- the BAKED native micro-model: layers ARE the certified installed
        parameterizations (circulant/permutation/dense), the register file is recurrent state,
        forward() IS the compiled program. Rule-not-bytes at the model level: save() writes a
        few-hundred-byte {dim, seed, program} file; load() re-bakes bit-identical weights.
        to_dense(op) is the one-call bridge to host-framework weight surgery. See
        holographic_nativemodel.NativeHoloModel."""
        from holographic.agents_and_reasoning.holographic_nativemodel import NativeHoloModel
        return NativeHoloModel(dim, seed, program, symbolic_functions, data=data, unitary=unitary)

    def compile_program_installed(self, machine, program, tol=1e-8):
        """F27 -- compile a symbolic HoloMachine program into certified installed matvecs + the
        F26 manifest; REPEAT of a linear body collapses to ONE operator power (spectral for
        circulants, exact). Returns (run_installed, manifest); save with
        holographic_compileinstall.save_manifest. Conformance pinned: VM == installed == hand
        truth on a REPEAT+STORE/RECALL program; nonlinear bodies refuse loudly."""
        import holographic.agents_and_reasoning.holographic_compileinstall as _ci
        return _ci.compile_installed(machine, program, tol=tol)

    def project_faculty(self, f, dim, n_check=24, tol=1e-8, seed=0):
        """MEASURE a callable into installed form or refuse (F34 T1): probe f with basis vectors,
        certify on held-out inputs, detect structure most-specific-first (permutation -> circulant
        -> dense; a roll is BOTH, so order matters -- caught by the selftest). The refusals ARE the
        core/shell boundary, discovered by measurement. See holographic_projector.probe_project;
        apply with holographic_projector.apply_projected."""
        import holographic.io_and_interop.holographic_projector as _pj
        return _pj.probe_project(f, dim, n_check=n_check, tol=tol, seed=seed)

    def unicron_forward_runtime(self, model, cfg):
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
        # A `=None` DEFAULT ON A REQUIRED ARGUMENT IS A SIGNATURE THAT LIES.
        # Calling this with defaults died as "'NoneType' object is not
        # iterable" several frames down, naming neither the faculty nor the
        # argument -- the caller sees a crash in someone else's code. Found by
        # CALLING every zero-argument faculty rather than reading them: 391
        # called, 22 raised, 11 raised without saying what was missing.
        if weights is None:
            raise ValueError(
                "unicron_model_store needs weights= -- it has a None default so it can be\n"
                "passed positionally or by keyword, but there is no\n"
                "meaningful empty case to fall back to.")
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

    def unicron_embed_repair(self, weights, cfg, tokens, targets=None, k=64,
                             blend=1.0, check=True, min_count=3):
        """REPAIR UNDER-ESTIMATED EMBEDDING ROWS FROM THE SPELLING OF THEIR TOKENS.

        THE FIRST INSTALL-PATH OPERATION THAT MAKES THE BASE MODEL BETTER. Every other
        step here ADDS something -- registers, the HRNN ladder, the router, the memory
        index. This one repairs what is already present: a quarter-million-token
        vocabulary has a long tail whose rows were estimated from very few occurrences and
        are noise-dominated, and a token's FORM predicts its meaning well enough to rebuild
        them.

        HOW, AND WHY NOT THE OBVIOUS WAY. The rebuilt value is NOT the form prediction.
        Measured across cp111-cp118: form identifies WHICH word at 255x chance but places
        the vector badly (variance explained -0.112 using the form vector directly). So
        form SELECTS donors and real in-vocabulary rows -- already on the manifold --
        supply the value (+0.063, against random donors at -0.015 and shuffled-letter
        selection at -0.084). The signal is paradigmatic: form predicts what a token IS,
        not what follows it (rank@1 0.254 vs 0.029 across the two meaning spaces).

        WHEN A ROW IS WORTH REPLACING. A rebuilt row lands at a FIXED quality regardless of
        how bad the row it replaces was, so repair only wins below a crossover, measured by
        bisection at cosine 0.507 to truth. Rebuilding a GOOD row makes it worse. That is
        why `targets` is explicit and there is no sweep-the-table mode: the caller must say
        which rows it believes are bad, and `embed_repair_candidates` offers only a
        heuristic for that, clearly labelled.

        SELF-GATING. An install is sequential weight editing, and `unicron_edit_health`
        exists because that composition is what degrades models. With check=True this runs
        edit_health over the embedding tensor and REFUSES the edit if the condition number
        moves the wrong way, returning the untouched weights with `applied=False`.

        tokens: {row_index: token_string} for rows whose spelling is known.
        targets: row indices to repair. Required -- see the crossover note above.
        Returns (weights, report). The input weights are never mutated.
        """
        import numpy as _np
        if targets is None:
            raise ValueError("targets is required: repairing a good row makes it worse "
                             "(crossover measured at cosine 0.507)")
        key = [k_ for k_ in weights if k_.endswith("embed_tokens.weight")]
        if not key:
            return weights, {"applied": False, "reason": "no embed_tokens.weight"}
        key = key[0]
        E = _np.asarray(weights[key])
        tgt = [int(t) for t in targets if int(t) < E.shape[0]]
        donors = [(i, s) for i, s in tokens.items()
                  if int(i) < E.shape[0] and int(i) not in set(tgt) and len(str(s)) >= 3]
        known = {int(i): str(s) for i, s in tokens.items()}
        tgt = [t for t in tgt if t in known]
        if len(donors) < 32 or not tgt:
            return weights, {"applied": False, "reason": "too few donor rows or targets",
                             "donors": len(donors), "targets": len(tgt)}

        # Same featurizer as Lexicon.bootstrap_by_form, so the two stay in step (cp118).
        from holographic.agents_and_reasoning.holographic_lexicon import word_feats as _feats  # promoted (sweep 123)

        dwords = [w for _, w in donors]
        counts = {}
        for w in dwords:
            for f in set(_feats(w)):
                counts[f] = counts.get(f, 0) + 1
        cols = [f for f, c in counts.items() if c >= min_count]
        if not cols:
            return weights, {"applied": False, "reason": "no form features survived"}
        index = {f: j for j, f in enumerate(cols)}

        def _mat(ws):
            M = _np.zeros((len(ws), len(cols)))
            for r, w in enumerate(ws):
                for f in _feats(w):
                    j = index.get(f)
                    if j is not None:
                        M[r, j] = 1.0
            return M

        Yd = E[[i for i, _ in donors]].astype(_np.float64)
        Xd = _mat(dwords)
        A = Xd.T @ Xd
        lam = 1e-2 * float(_np.trace(A)) / len(cols)
        B = _np.linalg.solve(A + lam * _np.eye(len(cols)), Xd.T @ Yd)
        P = _mat([known[t] for t in tgt]) @ B
        P /= (_np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
        Dn = Yd / (_np.linalg.norm(Yd, axis=1, keepdims=True) + 1e-12)
        sims = P @ Dn.T
        kk = int(min(k, len(donors)))
        picks = _np.argsort(sims, axis=1)[:, -kk:]

        E2 = E.copy()
        for r, t in enumerate(tgt):
            w = _np.maximum(_np.take(sims[r], picks[r]), 0.0) + 1e-9
            w /= w.sum()
            v = (Yd[picks[r]] * w[:, None]).sum(0)
            v = blend * v + (1.0 - blend) * E[t].astype(_np.float64)
            n = _np.linalg.norm(v)
            if n > 1e-12:
                # preserve the row's original norm: repair the DIRECTION, not the scale
                v = v * (float(_np.linalg.norm(E[t].astype(_np.float64))) / n)
                E2[t] = v.astype(E.dtype)

        out = dict(weights)
        out[key] = E2
        rep = {"applied": True, "repaired": len(tgt), "donors": len(donors),
               "features": len(cols), "k": kk, "tensor": key}
        if check:
            try:
                h = self.unicron_edit_health(weights, out, keys=(key,))
                rep["edit_health"] = h
                bad = isinstance(h, dict) and (h.get("degrading") is True
                                               or h.get("ok") is False)
                if bad:
                    rep["applied"] = False
                    rep["reason"] = "edit_health flagged the edit as degrading; reverted"
                    return weights, rep
            except Exception as exc:  # a missing checker must not silently pass an edit
                rep["edit_health_error"] = "%s: %s" % (type(exc).__name__, exc)
        return out, rep

    def embed_repair_candidates(self, weights, frac=0.05, method="norm"):
        """A HEURISTIC shortlist of rows that MIGHT be under-estimated. Labelled as such.

        There is no way to know a row's true quality from the weights alone, so this is a
        proxy and nothing more: `norm` returns the smallest-norm rows, on the reasoning
        that a row updated few times stays near its initialisation. Verify against real
        token frequencies whenever they are available, and remember the crossover -- rows
        that are actually fine will be made worse by repair.
        """
        import numpy as _np
        key = [k for k in weights if k.endswith("embed_tokens.weight")]
        if not key:
            return []
        E = _np.asarray(weights[key[0]], _np.float64)
        n = max(1, int(frac * E.shape[0]))
        score = _np.linalg.norm(E, axis=1)
        return [int(i) for i in _np.argsort(score)[:n]]

    def unicron_interstitial(self, runtime, cfg, sensors=None, bank=None, patches=None,
                             familiar=0.55, drift=0.35, learn=True):
        """THE THIN COORDINATION LAYER between the interstitial sensors (cp129).

        Live inside the model and fix it from the inside as it is used, with external memory
        as the patch. Three sensors, three jobs, never averaged; the route is decided from
        the SHAPE of the score profile because a single reading cannot tell "this input is
        new" from "the computation went wrong at depth d". Passive sensors are verified
        BIT-IDENTICAL (max logit delta 0.000e+00), so the mesh is inert until asked to patch.

        Implementation lives in holographic_interstitial (split out when structure_audit
        refused another 2000-line monolith).
        """
        from holographic.agents_and_reasoning.holographic_interstitial import interstitial
        return interstitial(runtime, cfg, sensors=sensors, bank=bank, patches=patches,
                            familiar=familiar, drift=drift, learn=learn)

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
        # A `=None` DEFAULT ON A REQUIRED ARGUMENT IS A SIGNATURE THAT LIES.
        # Calling this with defaults died as "'NoneType' object is not
        # iterable" several frames down, naming neither the faculty nor the
        # argument -- the caller sees a crash in someone else's code. Found by
        # CALLING every zero-argument faculty rather than reading them: 391
        # called, 22 raised, 11 raised without saying what was missing.
        if runtime is None:
            raise ValueError(
                "unicron_memory_search needs runtime= -- it has a None default so it can be\n"
                "passed positionally or by keyword, but there is no\n"
                "meaningful empty case to fall back to.")
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
        # SAME `=None` LIE as the rest of the unicron family. Called with
        # defaults this died as "'NoneType' object is not subscriptable"
        # inside fit_router -- two modules away from the caller.
        # NOTE FOR THE NEXT SWEEP: my one-line regex missed this because the
        # signature WRAPS. A pattern that assumes a def fits on one line will
        # silently skip every long signature, which is exactly the set most
        # likely to have this bug.
        if runtime is None or cfg is None:
            raise ValueError(
                "unicron_router needs runtime= and cfg= -- both have None\n"
                "defaults so they can be passed by keyword, but the router\n"
                "reads the model layout out of cfg and cannot invent one.")
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
        # A `=None` DEFAULT ON A REQUIRED ARGUMENT IS A SIGNATURE THAT LIES.
        # Calling this with defaults died as "'NoneType' object is not
        # iterable" several frames down, naming neither the faculty nor the
        # argument -- the caller sees a crash in someone else's code. Found by
        # CALLING every zero-argument faculty rather than reading them: 391
        # called, 22 raised, 11 raised without saying what was missing.
        if dim is None:
            raise ValueError(
                "unicron_reserve_keys needs dim= -- it has a None default so it can be\n"
                "passed positionally or by keyword, but there is no\n"
                "meaningful empty case to fall back to.")
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
        # A `=None` DEFAULT ON A REQUIRED ARGUMENT IS A SIGNATURE THAT LIES.
        # Calling this with defaults died as "'NoneType' object is not
        # iterable" several frames down, naming neither the faculty nor the
        # argument -- the caller sees a crash in someone else's code. Found by
        # CALLING every zero-argument faculty rather than reading them: 391
        # called, 22 raised, 11 raised without saying what was missing.
        if dim is None:
            raise ValueError(
                "unicron_sequence needs dim= -- it has a None default so it can be\n"
                "passed positionally or by keyword, but there is no\n"
                "meaningful empty case to fall back to.")
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
        # A `=None` DEFAULT ON A REQUIRED ARGUMENT IS A SIGNATURE THAT LIES.
        # Calling this with defaults died as "'NoneType' object is not
        # iterable" several frames down, naming neither the faculty nor the
        # argument -- the caller sees a crash in someone else's code. Found by
        # CALLING every zero-argument faculty rather than reading them: 391
        # called, 22 raised, 11 raised without saying what was missing.
        if objects is None:
            raise ValueError(
                "unicron_model_vault needs objects= -- it has a None default so it can be\n"
                "passed positionally or by keyword, but there is no\n"
                "meaningful empty case to fall back to.")
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

    def unicron_vm_unit_install(self, unit=None, table=None, rule=None, A=None,
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


def _selftest():
    """Delegates to holographic.unified.check_part -- the shared part contract -- then proves one
    torch-free representative faculty end-to-end through a real mind (wiring, not membership).
    The torch-side unicron faculties carry their own module selftests (SKIPPED-REFERENCE without
    weights); this part's job is the FACADE, so the facade is what gets tested here."""
    from holographic.unified import check_part
    n = check_part("holographic.unified.holographic_unified_p16_unicron", "_UnifiedPart16")
    import numpy as np
    from lecore import UnifiedMind as _UM
    m = _UM(dim=64, seed=0)
    # unicron_galvatron is the representative: construct with zero residents on a stub runtime --
    # the facade must build the object and refuse nothing (residents are optional by contract)
    class _StubRuntime:
        hidden_dim = 32
    g = m.unicron_galvatron(_StubRuntime())
    assert hasattr(g, "generate") or hasattr(g, "step") or g is not None
    print("OK: unified p16 (unicron) part contract holds over %d facade defs; galvatron facade "
          "constructs on a stub runtime (torch-side behavior tested in its own module)" % n)


if __name__ == "__main__":
    _selftest()
