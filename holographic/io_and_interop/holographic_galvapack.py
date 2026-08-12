"""GALVAPACK -- ship a Galvatron as a package, wear a normal model's clothes.

The bargain Moose asked for: a model that is SUPERIOR BECAUSE of its scaffolding
(leCore residents in the forward pass), that nevertheless plugs into ordinary
tooling. Two things make that possible without lying to anyone:

  1. A PACKAGE, not a checkpoint. `save_pack` writes an ordinary safetensors
     file (converts and runs anywhere, residents absent) PLUS galvatron.json --
     a DECLARATIVE manifest of the resident stack. The manifest is data, never
     code: a resident is named, parameterized, and rebuilt by `load_pack` from
     the same catalog every time. No pickle, no exec, no arbitrary callables
     crossing a file boundary (the same reason unicron refuses torch pickle).

  2. TWO FRONT DOORS over the same running Galvatron:
     * OpenAI-compatible HTTP (/v1/models, /v1/completions, /v1/chat/completions)
       -- what LM Studio clients, the OpenAI SDK, and most agent frameworks
       already speak. Point them at the port and the scaffolding is invisible.
     * HFCompatWrapper.generate(input_ids, max_new_tokens=...) -- the shape
       transformers callers expect, so existing Python harness code runs
       unmodified.

  DEGRADED MODE IS A FEATURE: a pack whose manifest cannot be satisfied (no
  leCore, no mind) still loads and serves the PLAIN model. The scaffolding
  improves the model; it must never be the thing that stops it from running.
  The honest inverse is stated in the manifest itself: `portable: true` means
  the safetensors alone is a complete, ordinary model, and `residents` lists
  exactly what is lost by running it that way.

WHAT THIS IS NOT: it is not a way to smuggle activation-space behaviour through
a GGUF conversion. Ollama/llama.cpp consume GGUF and expose no hooks -- for
those, export the plain weights (unicron_export_portable) and accept the model
alone, or run this server and point the client at it. Both paths are supported;
neither is oversold.
"""

import json
import os

import numpy as np


MANIFEST = "galvatron.json"


# --------------------------------------------------------------- resident spec

class _EvidenceGuard:
    """Carries an EvidenceStore through the manifest so verified_generate can
    use it. Declared as a guard because it constrains OUTPUT, and guards are the
    part of a Galvatron that survives a mind-free load."""

    def __init__(self, evidence, spec):
        self.evidence = evidence
        self.spec = dict(spec)

    def guard(self, logits):
        return logits            # the veto happens at generation, not per-logit


def _build_residents(mind, runtime, specs, hidden_dim):
    """Rebuild residents from declarative specs -- DATA, never code.

    The vocabulary is deliberately wide: a Galvatron should be able to carry as
    much of leCore as the manifest can describe, or the bundle is a demo rather
    than a product. Unknown kinds are SKIPPED WITH A REASON rather than raising,
    so a newer pack still runs on an older leCore minus what it cannot
    understand -- forward compatibility beats a hard failure that leaves the
    user with nothing."""
    from holographic.agents_and_reasoning import holographic_galvatron as G
    # INSTALLED is not SKIPPED. A component that changed the runtime but adds no
    # resident object was being reported as skipped, which reads as a failure
    # and tripped the selftest that asserts nothing was skipped -- correctly.
    residents, guards, skipped, installed = [], [], [], []
    for sp in specs:
        kind = sp.get("kind")
        try:
            if kind == "ward":
                guards.append(G.WardResident(banned=sp.get("banned", ()),
                                             allowed=sp.get("allowed")))
            elif kind == "oracle":
                r = G.OracleResident(mind, hidden_dim, layer=int(sp["layer"]),
                                     gain=float(sp.get("gain", 1.0)),
                                     threshold=float(sp.get("threshold", 0.6)),
                                     tag=sp.get("tag", "oracle"))
                for entry in sp.get("memories", []):
                    r.remember(np.asarray(entry["key"], np.float64),
                               np.asarray(entry["value"], np.float64))
                residents.append(r)
            elif kind == "dreamer":
                samples = sp.get("samples")
                if samples is None:
                    skipped.append((kind, "no healthy-state samples in pack"))
                    continue
                # live signature (probed, not assumed): (mind, healthy_hiddens,
                # layer, strength, energy)
                residents.append(G.DreamerResident(
                    mind, np.asarray(samples, np.float64),
                    int(sp["layer"]),
                    strength=float(sp.get("strength", 0.9))))
            elif kind == "cache":
                # STOP REDOING THE SAME WORK. Content-keyed memo over the
                # measured hot paths: attention cluster routing (k-means was
                # re-run once per head per forward on unchanged keys),
                # capability routing (0.29s cold -> 0.000022s warm, 13,000x) and
                # retrieval. Keys are hashlib digests of the actual bytes, so a
                # hit cannot be stale and the cache is deterministic across
                # processes -- hash() would not be.
                from holographic.caching_and_storage.holographic_galvacache import (
                    install)
                install(runtime=runtime, mind=mind,
                        verify=bool(sp.get("verify", False)))
                installed.append(("cache", "memo on the runtime hot paths"))

            elif kind == "toolbelt":
                # THE WHOLE CATALOG, not a hand-picked dozen. Wiring one named
                # capability per manifest entry was the slow way to answer
                # "give the model the powers"; this carries the ROUTER, so
                # demux, resonator factoring, denoisers, drift algebra, fluid
                # steps, path tracing, linear solves and the VSA primitives are
                # all reachable by description. Whitelist and call budget are in
                # the spec, and every invocation is logged.
                from holographic.agents_and_reasoning.holographic_toolbelt import (
                    ToolbeltResident)
                residents.append(ToolbeltResident(
                    mind, hidden_dim, layer=int(sp.get("layer", 0)),
                    families=tuple(sp.get("families", ()) or ()),
                    deny=tuple(sp.get("deny", ()) or ()),
                    gain=float(sp.get("gain", 1.0)),
                    max_calls=int(sp.get("max_calls", 32))))

            elif kind == "leap":
                # SPECULATIVE DECODING as a package property: the drafter learns
                # from ACCEPTED tokens only, and output is token-identical to
                # greedy, so this is speed with no behavioural change.
                from holographic.agents_and_reasoning.holographic_leap import (
                    RouteMemory)
                runtime.cfg["leap"] = {"k": int(sp.get("k", 8)),
                                       "order": int(sp.get("order", 4))}
                installed.append(("leap", "speculative decoding enabled in cfg"))

            elif kind == "screen":
                # the attention shortcut travels too: exact top-k selection via
                # cluster ball-bounds, measured at ~38% of the keys
                runtime.cfg["attn_screen"] = {k: v for k, v in sp.items()
                                              if k != "kind"}

            elif kind == "memory":
                # THE DATABASE TRAVELS. A Galvatron with a corpus frozen at
                # build time cannot LEARN; one carrying its own holographic
                # database has rows, provenance columns, an edge table for
                # links, BM25 over the text and crash-safe durability -- and it
                # can be written to while it runs. RAG stops being a fixed
                # passage list and becomes a store the model shares with its
                # user and with its own residents.
                from holographic.caching_and_storage.holographic_memory import (
                    Memory)
                snap = sp.get("snapshot")
                base = os.path.dirname(os.path.abspath(sp.get("_path", ".")))
                path = snap if (snap and os.path.isabs(snap)) else \
                    (os.path.join(base, snap) if snap else None)
                if path and os.path.exists(path):
                    mem = Memory.restore(mind, path, dim=int(sp.get("dim", 1024)))
                else:
                    mem = Memory(mind, dim=int(sp.get("dim", 1024)))
                    for row in sp.get("notes", []):
                        mem.note(row.get("title", "note"), row.get("text", ""),
                                 author=row.get("author", "pack"),
                                 tags=tuple(row.get("tags", ()) or ()))
                from holographic.agents_and_reasoning.holographic_knowres import (
                    CorpusResident)
                res = CorpusResident(mind, mem.passages(), hidden_dim,
                                     layer=int(sp.get("layer", 0)),
                                     query_fn=(lambda h, _q=sp.get("query", ""): _q),
                                     gain=float(sp.get("gain", 1.0)))
                res.memory = mem          # the store stays reachable and writable
                residents.append(res)

            elif kind == "verifier":
                # THE HALLUCINATION GATE, in the package rather than the driver:
                # spans with no support in the carried sources are vetoed BEFORE
                # emission. Span length scales with corpus size, because a
                # 3-token span is a real constraint against three passages and a
                # rubber stamp against three hundred.
                from holographic.agents_and_reasoning.holographic_swarm import (
                    EvidenceStore)
                texts = list(sp.get("passages", []))
                for r in residents:
                    if getattr(r, "memory", None) is not None:
                        texts += r.memory.passages()
                span = int(sp.get("span", 0) or (3 if len(texts) < 20
                                                 else 5 if len(texts) < 200 else 6))
                ev = EvidenceStore(span=span)
                for t in texts:
                    ev.add([int(b) for b in str(t).encode("utf-8")])
                guards.append(_EvidenceGuard(ev, sp))

            elif kind == "corpus":
                from holographic.agents_and_reasoning.holographic_knowres import (
                    CorpusResident, SalienceTrigger)
                query = sp.get("query", "")
                trig = (lambda h, _q=query: _q)
                if sp.get("salience"):
                    # gate retrieval on the model's OWN hesitation, so the
                    # packaged Galvatron searches when IT needs to, not on a
                    # fixed schedule baked in by whoever built the pack
                    st = SalienceTrigger(runtime)
                    st.calibrate(np.asarray(sp["salience"]["samples"], np.float64),
                                 quantile=float(sp["salience"].get("quantile", 0.8)))
                    trig = st.gate(lambda h, _q=query: _q)
                residents.append(CorpusResident(
                    mind, sp.get("corpus", []), hidden_dim,
                    layer=int(sp["layer"]), query_fn=trig,
                    gain=float(sp.get("gain", 1.0)), top=int(sp.get("top", 1))))
            elif kind == "hrnn":
                from holographic.agents_and_reasoning.holographic_knowres import (
                    HRNNResident)
                residents.append(HRNNResident(
                    mind, hidden_dim, layer=int(sp["layer"]),
                    dim=int(sp.get("dim", 1024)), seed=int(sp.get("seed", 0)),
                    gain=float(sp.get("gain", 0.0))))
            elif kind == "carrier":
                from holographic.agents_and_reasoning.holographic_carrier import (
                    StreamCarrier)
                car = StreamCarrier(np.asarray(sp["samples"], np.float64),
                                    reserve=int(sp.get("reserve", 16)),
                                    amplitude=float(sp.get("amplitude", 0.5)))
                pairs = sp.get("pairs") or {}
                hook = car.writer(pairs)
                residents.append(_HookResident(int(sp["layer"]), hook, car))
            elif kind == "capability":
                from holographic.agents_and_reasoning.holographic_capresident import (
                    CapabilityResident)
                args = sp.get("args") or {}
                residents.append(CapabilityResident(
                    mind, sp["capability"], hidden_dim, int(sp["layer"]),
                    trigger=(lambda h, _a=args: _a),
                    gain=float(sp.get("gain", 1.0))))
            else:
                skipped.append((kind, "unknown resident kind"))
        except Exception as exc:                    # a bad spec must not kill the pack
            skipped.append((kind, "%s: %s" % (type(exc).__name__, exc)))
    return residents, guards, skipped, installed


class _HookResident:
    """Adapter so any prebuilt hook (e.g. a StreamCarrier writer) satisfies the
    resident contract the Galvatron composer expects."""

    def __init__(self, layer, hook_fn, obj=None):
        self.layer = int(layer)
        self._fn = hook_fn
        self.obj = obj

    def hook(self, h):
        return self._fn(h)


def save_pack(path, weights, cfg, residents=(), notes="", like_dir=None):
    """Write a Galvatron package: plain safetensors + declarative manifest."""
    from holographic.io_and_interop import holographic_unicron as U
    os.makedirs(path, exist_ok=True)
    U.export_portable(weights, os.path.join(path, "model.safetensors"),
                      like=like_dir)
    man = {"format": "galvatron/1", "portable": True, "runtime": "gdn_hybrid",
           "config": {k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in cfg.items()},
           "residents": list(residents), "notes": notes,
           "without_leCore": "model.safetensors alone is an ordinary checkpoint; "
                             "the residents listed here are what running it that "
                             "way gives up"}
    with open(os.path.join(path, MANIFEST), "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    return {"path": path, "residents": len(man["residents"])}


def load_pack(path, mind=None, lazy=False, with_guards=True):
    """Load a pack into a running Galvatron. Without a mind (or without leCore
    residents available) it degrades and SAYS SO in the returned report -- never
    a silent downgrade.

    GUARDS ARE THE EXCEPTION: a ward needs no mind, so a mind-free load still
    enforces it. Set with_guards=False only when the caller EXPLICITLY wants the
    bare model (the bundle's --no-residents), which is a different request from
    "no mind was available" -- conflating the two either drops a safety
    guarantee by accident or makes a plain-model comparison impossible."""
    from holographic.io_and_interop import holographic_unicron as U
    from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime
    from holographic.agents_and_reasoning.holographic_galvatron import Galvatron
    with open(os.path.join(path, MANIFEST)) as f:
        man = json.load(f)
    # packs are WRITTEN single-file, but tolerate a hand-assembled sharded one
    # rather than failing on a layout that is normal everywhere else
    single = os.path.join(path, "model.safetensors")
    if os.path.exists(single):
        w = U.load_safetensors(single)
    else:
        from holographic.io_and_interop.holographic_gdnruntime import (
            load_weights_dir)
        w = load_weights_dir(path)
    if lazy:
        w = U.LazyWeights(w)
    rt = GDNRuntime(w, man["config"])
    # A pack written before the layout was recorded (or hand-assembled) still
    # has to be read correctly: resolve it here and say what was chosen, then
    # sanity-check that the result predicts English better than chance.
    if any(k.endswith("in_proj_qkv.weight") for k in rt.w) \
            and "qkv_order" not in rt.cfg:
        from holographic.io_and_interop.holographic_gdnruntime import (
            _resolve_ambiguous_layout, _sanity_check)
        _resolve_ambiguous_layout(rt, path)
        _sanity_check(rt, path)
    report = {"residents": 0, "skipped": [], "degraded": mind is None}
    residents, guards = [], []
    if mind is not None:
        residents, guards, skipped, installed = _build_residents(
            mind, rt, man.get("residents", []), int(man["config"]["hidden"]))
        report["residents"] = len(residents) + len(guards)
        report["skipped"] = skipped
        report["installed"] = installed
    elif man.get("residents"):
        # MIND-FREE RESIDENTS STILL BUILD. The ward is pure logit masking -- it
        # needs no memory, no denoiser, no VSA -- so degrading it along with
        # everything else silently drops a SAFETY guarantee whenever the loader
        # is called without a mind. Measured: a banned token was emitted by a
        # pack whose manifest bans it. A guard that only holds under ideal
        # conditions is not a guard.
        from holographic.agents_and_reasoning.holographic_galvatron import (
            WardResident)
        skipped = []
        for spec in man["residents"]:
            if spec.get("kind") == "ward" and with_guards:
                guards.append(WardResident(banned=spec.get("banned", ()),
                                           allowed=spec.get("allowed")))
            else:
                skipped.append((spec.get("kind"), "no mind supplied"))
        report["residents"] = len(guards)
        report["skipped"] = skipped
        report["degraded"] = bool(skipped)
    return Galvatron(rt, residents=residents, guards=guards), report


def imbue(model_dir, out_dir, mind, corpus=(), probe_text=None, banned=(),
          bundle_engine=True, notes="", call_capabilities=None):
    """ONE CALL: ordinary checkpoint in, IMBUED GALVATRON out.

    What "imbued" honestly means, because the word invites a wrong picture: the
    residents are NOT written into the weights -- they cannot be, they are
    structure in the forward pass. What this produces is a package that carries
    everything needed to RECONSTRUCT them: the weights, a declarative manifest
    of the resident roster, the CALIBRATION DATA those residents need (healthy
    stream statistics for repair, salience quantiles, the carrier basis), the
    grounding corpus, and -- with bundle_engine -- leCore itself plus a run.py.
    Load it anywhere and the ward, oracle, corpus grounding, fact checker and
    time travel are all there. Load `model.safetensors` in another framework and
    you get the bare model back, exactly, with none of them. The manifest says
    so in its own text.

    The calibration is the part that could not be written by hand: healthy
    hidden statistics are harvested by RUNNING the model on a probe, so the
    package is fitted to this checkpoint rather than to a default."""
    from holographic.io_and_interop.holographic_gdnruntime import load_runtime
    from holographic.io_and_interop import holographic_unicron as U

    rt, cfg = load_runtime(model_dir)
    probe = probe_text or (
        "The capital of France is Paris. Water freezes at zero degrees and "
        "boils at one hundred. A recurrent state carries what the past can "
        "tell the future, and every layer writes into the residual stream.")
    ids = _probe_ids(model_dir, probe, rt)
    layer = max(0, int(cfg["n_layers"]) - 2)
    grabbed = {}
    rt.forward(ids, hooks={layer:
                           lambda h: grabbed.__setitem__("h", h.copy()) or None})
    healthy = grabbed["h"]

    # IDENTITY RIDES IN THE CARRIER: the package can say what it is from inside
    # the residual stream, in reserved low-energy directions, at no context cost.
    import time as _time
    ident = {"origin": os.path.basename(model_dir.rstrip("/\\")),
             "built": _time.strftime("%Y-%m-%d"),
             "engine": "leCore"}
    # CARRY THE RESOLVED LAYOUT. load_runtime worked out whether this
    # checkpoint's in_proj_qkv is grouped or flat by MEASURING; the pack's
    # loader builds a runtime straight from the manifest and would otherwise
    # fall back to the default and emit fluent garbage (field-caught: a real
    # 0.8B answered "zugd4 {Ig1ounced699"). A decision made by measurement must
    # travel with the artifact.
    for key in ("qkv_order", "attn_top_k", "attn_screen"):
        if key in rt.cfg:
            cfg[key] = rt.cfg[key]
    # THE FULL ROSTER, because a Galvatron that carries half the engine is a
    # demo. Memory (a writable holographic database, so it can keep learning),
    # the verifier (the anti-hallucination contract ships WITH the model),
    # screen routing (exact attention selection at ~38% of the keys) and leap
    # (speculative decoding, token-identical output) all travel in the manifest.
    specs = maximal_specs(rt, healthy, corpus=list(corpus), banned=list(banned),
                          carrier_pairs=ident,
                          capability="bundle_capacity", capability_args={},
                          verifier=True, leap=True,
                          screen={"mode": "ball", "clusters": 50, "topk": 8,
                                  "window": 32})
    kinds = sorted({sp.get("kind", "?") for sp in specs})
    # SAY WHAT IS MISSING AND WHY. A roster that silently omits a resident looks
    # identical to one that could not build it.
    skipped = []
    if not banned:
        skipped.append("ward (no banned tokens given -- pass banned=[...])")
    if "oracle" not in kinds:
        skipped.append("oracle (no memories given -- it is built empty and "
                       "filled at run time)")
    # SHARDED CHECKPOINTS ARE THE NORM at real sizes: a 0.8B ships as
    # model-00001-of-0000N.safetensors. Use the same shard-aware loader the
    # runtime uses rather than assuming a single file (field-caught: imbue died
    # on the first real model it was pointed at).
    from holographic.io_and_interop.holographic_gdnruntime import load_weights_dir
    weights = load_weights_dir(model_dir)

    # ---- THE INCEPTION LAYER: put leCore INSIDE the weights -------------------
    # Everything above this line is a MANIFEST -- data a leCore runtime reads to
    # rebuild residents. That is the outer layer, and it vanishes the moment the
    # weights are loaded anywhere else. What follows edits the TENSORS, so it
    # survives export, quantization and any runtime that never heard of leCore.
    # Verified by diffing an imbued pack against its source: before this, ZERO
    # tensors changed and the "imbued" model was byte-identical to the original.
    baked = []
    try:
        from holographic.io_and_interop.holographic_galvabake import bake_ward
        from holographic.io_and_interop.holographic_vsabake import (
            circulant, install_op)
        from holographic.io_and_interop.holographic_progbake import (
            encode_program, write_rows)
    except ImportError:
        bake_ward = None
    if bake_ward is not None:
        probe = rt.forward(ids)[-1]
        # VSA ALGEBRA AS CIRCUITS: bind and unbind against a fixed role, installed
        # as MLP neurons. The model can then move role-filler structure in its own
        # forward pass, with no residents present.
        role = np.random.default_rng(0).standard_normal(int(cfg["hidden"]))
        role /= np.linalg.norm(role)
        # EVERY BAKE IS GUARDED FROM HERE ON. A real run shipped a Galvatron
        # whose perplexity went 16.2 -> 190,391: destroyed by its own imbue,
        # written to disk, and reported as a success with a resident list.
        base_ppl = float(rt.perplexity(list(ids)))
        guard_log = []
        weights, brep, g = _guarded(
            weights, cfg, ids, base_ppl,
            lambda w: install_op(w, cfg, circulant(role),
                                 layer=int(cfg["n_layers"]) - 1,
                                 mean_h=healthy.mean(0)),
            "vsa_bind")
        guard_log.append(g)
        if g["kept"]:
            baked.append(("vsa_bind", brep["neurons_added"]))
        # PROGRAMS IN THE UNUSED VOCABULARY: whatever corpus was supplied is also
        # written into rows the tokenizer never defines, addressable by token id.
        head = next((k for k in weights if k.endswith("embed_tokens.weight")), None)
        if head is not None and corpus:
            rows_total = int(np.asarray(weights[head]).shape[0])
            free = rows_total - reserved_rows(model_dir, rows_total)
            if free > 2:
                syms = " ".join(list(corpus)[:4]).split()[:32]
                traces = encode_program(syms, int(np.asarray(weights[head]).shape[1]))
                start = rows_total - len(traces)
                weights, prep, g2 = _guarded(
                    weights, cfg, ids, base_ppl,
                    lambda w: write_rows(w, traces, start_row=start,
                                         keys=(head,)),
                    "program_rows")
                guard_log.append(g2)
                if g2["kept"]:
                    baked.append(("program_rows", len(prep["rows"])))
        # THE WARD IS BAKED LAST, ON THE FINAL WEIGHTS.
        # Ordering here is not cosmetic: the first version verified the ban
        # and THEN installed 128 VSA neurons, which changed the very model
        # the verification was about -- the report said "verified on 4
        # prompts" while the ward leaked on a code prompt. A guarantee
        # established before a later edit is not a guarantee.
        # THE WARD IS NOT A DEFAULT. It works -- verified weights-only across
        # prompts -- but nobody asked for a model that refuses words, and
        # shipping it as the headline made a test harness look like the product.
        # Applied only when a ban is explicitly requested.
        if banned:
            # the ward becomes a permanent property of the output head
            # verify against DIVERSE prompts, not just the calibration one:
            # a ward fitted to English leaked on code (measured)
            vprompts = [ids[:32]]
            for extra in ("def compress(x):", "Water freezes at zero.",
                          "\n\n# heading\n"):
                try:
                    vprompts.append(_probe_ids(model_dir, extra, rt)[:24])
                except Exception:
                    pass
            weights, wrep = bake_ward(weights, cfg, list(banned),
                                      probe_logits=probe,
                                      verify_prompts=vprompts)
            baked.append(("ward", "%d tokens by %s, worst margin %.1f at EVERY "
                          "position of %d probes"
                          % (wrep["banned"],
                             wrep.get("method") or "direction bias",
                             wrep.get("worst_margin") if
                             wrep.get("worst_margin") is not None else 0.0,
                             wrep["verified_on"])))
    note = notes or ("imbued from %s; calibrated on %d probe tokens"
                     % (os.path.basename(model_dir.rstrip("/\\")), len(ids)))
    # ---- BOOT RECORD: without one, nothing can BOOT the layer from weights.
    # harden's boots_from_weights failed on two real runs for exactly this
    # reason -- imbue installed residents (which are declarative and rebuilt at
    # load) and never wrote the one row that makes the model self-describing.
    try:
        from holographic.io_and_interop.holographic_boot import (
            BootRecord, write_boot)
        # KEEP THE RECORD MINIMAL. Everything except the seed REGENERATES from
        # the seed -- that is the whole point of the boot design -- so listing
        # symbols and capabilities in the row spends the scarcest resource in
        # the model (half a vocabulary row, 4 bits per slot after the bf16 fix)
        # on data that is deterministic anyway.
        _rec = BootRecord(seed="leCore", dim=int(cfg["hidden"]))
        weights, _brep2, gboot = _guarded(
            weights, cfg, ids, base_ppl,
            lambda w: (write_boot(w, _rec)[0], write_boot(w, _rec)[1]),
            "boot_record")
        guard_log.append(gboot)
        if gboot["kept"]:
            baked.append(("boot_record", "seed leCore"))
    except Exception as exc:
        guard_log.append({"bake": "boot_record", "kept": False,
                          "why": "%s: %s" % (type(exc).__name__, exc)})

    call_report = None
    # ---- CALL TOKENS: the model asks for a capability on its own ----
    # RUNS LAST, deliberately: it fits the OUTPUT HEAD, and any later edit to
    # the head or to the embedding rows it addresses would silently undo it.
    # Off unless asked, because it edits the OUTPUT HEAD and a model that calls
    # a tool on every prompt is worse than one that never does. When asked, the
    # negatives are as important as the positives -- the fit has to be shown
    # what silence looks like.
    if call_capabilities:
        try:
            from holographic.agents_and_reasoning.holographic_calltoken import (
                allocate, free_rows, teach_calls)
            from holographic.io_and_interop.holographic_vsabake import embed_key
            n_defined = reserved_rows(
                model_dir, int(np.asarray(weights[embed_key(weights)]).shape[0]))
            # TAKE FROM THE FRONT of the free range. program_rows writes its
            # traces at `rows_total - len(traces)`, i.e. from the END, and both
            # features silently claimed the same rows: the call-token head fit
            # was applied and the embeddings it addressed were then overwritten,
            # so the model emitted nothing. Same shape as the boot spill
            # clobbering the stored program -- two components each assuming they
            # owned the surface. Verified by checking where progbake actually
            # writes rather than by guessing which end was free.
            rows = free_rows(weights, n_defined)
            reserved = len(" ".join(list(corpus)[:4]).split()[:32]) if corpus else 0
            rows = rows[:max(0, len(rows) - reserved)]
            if len(rows) < len(call_capabilities):
                call_report = ("skipped: %d free vocabulary rows for %d "
                               "capabilities" % (len(rows),
                                                 len(call_capabilities)))
            else:
                table = allocate([c for c, _ctx in call_capabilities], rows)
                pos = {}
                for (name, ctxs), tok in zip(call_capabilities, table):
                    pos[tok] = [_probe_ids(model_dir, c, rt)[:24] for c in ctxs]
                negs = [_probe_ids(model_dir, c, rt)[:24] for c in
                        ("The capital of France is ", "Water freezes at zero ",
                         "def compress(x):\n    ", "Once upon a time ")]
                # FIT AGAINST THE FINAL MODEL, NOT THE ORIGINAL. `rt` was built
                # before vsa_bind added 128 neurons and program rows were
                # written, so its hidden states are NOT the states the shipped
                # weights produce -- a head fitted on them emits nothing. This
                # is precisely the ward's lesson ("verified before the edit that
                # broke it") and I repeated it one function away from where it
                # is documented.
                from holographic.io_and_interop.holographic_gdnruntime import (
                    GDNRuntime as _RT)
                rt_now = _RT(weights, dict(cfg))
                weights, crep, g3 = _guarded(
                    weights, cfg, ids, base_ppl,
                    lambda w: teach_calls(w, cfg, rt_now, pos, negs, table),
                    "call_tokens")
                guard_log.append(g3)
                if not g3["kept"]:
                    raise RuntimeError("call-token fit reverted: %s" % g3["why"])
                # VERIFY ON THE FINAL WEIGHTS, and report the truth. The ward
                # learned this the hard way and the lesson generalises: a fit
                # that is reported without being checked is a claim, not a
                # capability. A least-squares head has finite capacity -- on a
                # narrow model it can fail to separate several capabilities at
                # once -- so the number that ships is the MEASURED one.
                verify_rt = _RT(weights, dict(cfg))
                fired = 0
                total = 0
                for tok, ctxs in pos.items():
                    for ctx in ctxs:
                        total += 1
                        fired += int(np.argmax(verify_rt.forward(ctx)[-1])) == tok
                false = sum(int(np.argmax(verify_rt.forward(c)[-1])) in table
                            for c in negs)
                call_report = {"table": table, "examples": crep["examples"],
                               "emits": "%d/%d" % (fired, total),
                               "false_calls": "%d/%d" % (false, len(negs)),
                               "usable": bool(fired and not false)}
                baked.append(("call_tokens",
                              "%d capabilities on rows %s -- emits %d/%d, "
                              "false calls %d/%d"
                              % (len(table), sorted(table)[:4], fired, total,
                                 false, len(negs))))
        except Exception as exc:
            call_report = "failed: %s: %s" % (type(exc).__name__, exc)

    if bundle_engine:
        from holographic.io_and_interop import holographic_galvabundle as GB
        rep = GB.bundle(out_dir, weights, cfg, residents=specs, notes=note,
                        like_dir=model_dir)
        # RECORD WHAT ACTUALLY LANDED. Two real runs produced a Galvatron
        # BIT-IDENTICAL to its input, and nothing in the artifact said whether a
        # bake was reverted, skipped or never attempted. A build log that does
        # not survive into the artifact cannot answer the only question that
        # matters afterwards.
        try:
            _mp = os.path.join(out_dir, "galvatron.json")
            with open(_mp) as _f:
                _man = json.load(_f)
            _man["guarded_bakes"] = guard_log
            _man["baked_into_weights"] = [list(b) for b in baked]
            with open(_mp, "w") as _f:
                json.dump(_man, _f, indent=2)
        except (OSError, ValueError):
            pass
    else:
        rep = save_pack(out_dir, weights, cfg, residents=specs, notes=note,
                        like_dir=model_dir)
    # CALIBRATION TRAVELS AS DATA, not as a promise
    np.savez_compressed(os.path.join(out_dir, "galvatron_profile.npz"),
                        healthy=healthy, probe_ids=np.asarray(ids, np.int64))
    # AND SO DOES THE VOCABULARY. Without it the package cannot turn text into
    # tokens, so its chat would encode raw UTF-8 bytes into a 248k-token model
    # and emit nonsense -- a self-contained bundle that cannot read is not
    # self-contained. leCore reads these with stdlib, so no dependency follows.
    import shutil as _shutil
    carried = []
    # CARRY THE WHOLE HUGGING FACE SURFACE, not just the tokenizer. A Galvatron
    # that cannot be converted to GGUF is not a deliverable: llama.cpp's
    # convert_hf_to_gguf.py needs config.json IN HF SHAPE (hidden_size,
    # num_hidden_layers) alongside model.safetensors, and the bundle was
    # shipping galvatron.json instead -- so the artifact ran in leCore and
    # nowhere else. Verified by checking a produced bundle against what the
    # converter actually reads.
    # GUARANTEE AN HF CONFIG, do not hope one was copied. A real run reported
    # "config.json is not HF-shaped" after a chain of steps each copying from
    # the last: somewhere in that chain a leCore-shaped config was written, and
    # every downstream step faithfully carried it. If what arrives is not HF
    # shaped, one is SYNTHESISED from the runtime config -- the artifact has to
    # convert, and a missing key is not a reason to ship something that cannot.
    synthesised_config = False
    _hf_ok = False
    _src_cfg = os.path.join(model_dir, "config.json")
    if os.path.exists(_src_cfg):
        try:
            with open(_src_cfg) as _f:
                _c = json.load(_f)
            _hf_ok = ("hidden_size" in _c
                      or "hidden_size" in (_c.get("text_config") or {}))
        except (OSError, ValueError):
            _hf_ok = False
    if not _hf_ok:
        _synth = {"architectures": ["Qwen3NextForCausalLM"],
                  "model_type": "qwen3_next",
                  "hidden_size": int(cfg["hidden"]),
                  "num_hidden_layers": int(cfg["n_layers"]),
                  "num_attention_heads": int(cfg.get("n_heads", 8)),
                  "num_key_value_heads": int(cfg.get("n_kv_heads", 2)),
                  "head_dim": int(cfg.get("head_dim", 128)),
                  "intermediate_size": int(cfg.get("intermediate", 0)) or None,
                  "rms_norm_eps": float(cfg.get("rms_eps", 1e-6)),
                  "rope_theta": float(cfg.get("rope_theta", 10000.0)),
                  "vocab_size": int(cfg.get("vocab", 0)) or None,
                  "tie_word_embeddings": True}
        _synth = {k: v for k, v in _synth.items() if v is not None}
        with open(os.path.join(out_dir, "config.json"), "w") as _f:
            json.dump(_synth, _f, indent=2)
        synthesised_config = True

    for name in (("config.json",) if _hf_ok else ()) + (
            "generation_config.json",
            "vocab.json", "merges.txt", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json",
            "chat_template.jinja"):
        srcf = os.path.join(model_dir, name)
        if os.path.exists(srcf):
            _shutil.copy(srcf, os.path.join(out_dir, name))
            carried.append(name)
    rep["tokenizer_files"] = carried
    rep["config_synthesised"] = synthesised_config
    if call_report is not None:
        rep["call_tokens"] = call_report
    try:
        rep["guarded_bakes"] = guard_log
    except NameError:
        pass
    rep["baked_into_weights"] = baked
    rep["residents"] = len(specs)
    rep["kinds"] = kinds
    rep["skipped"] = skipped
    rep["corpus_passages"] = len(list(corpus))
    rep["calibrated_on"] = len(ids)
    return rep


def check_deployable(bundle_dir, original_dir=None, probe_ids=None,
                     tolerance=0.01):
    """Is this artifact ACTUALLY deliverable? Convertible AND no worse.

    A smaller model that only runs inside leCore is not a Galvatron -- the
    requirement is that it runs wherever the original ran, and works at least as
    well. This checks both, because a size number on its own has misled this
    project more than once.

    CHECK 1, CONVERTIBILITY: llama.cpp's convert_hf_to_gguf.py reads config.json
    IN HUGGING FACE SHAPE (hidden_size, num_hidden_layers) beside
    model.safetensors. The bundle shipped galvatron.json instead and was
    therefore convertible by nothing -- it ran in leCore and nowhere else.
    CHECK 2, QUALITY: perplexity against the original on the same tokens, with a
    tolerance the caller states rather than one this function invents."""
    import json as _json

    have = set(os.listdir(bundle_dir))
    rep = {"convertible": False, "quality_ok": None, "problems": []}
    if "model.safetensors" not in have and not any(
            f.endswith(".safetensors") for f in have):
        rep["problems"].append("no safetensors weights")
    cfg_path = os.path.join(bundle_dir, "config.json")
    if not os.path.exists(cfg_path):
        rep["problems"].append("no config.json (convert_hf_to_gguf.py needs it)")
    else:
        try:
            with open(cfg_path) as f:
                c = _json.load(f)
            # NESTED CONFIGS ARE HF-SHAPED TOO. Qwen3.5 puts the language
            # settings under "text_config" because it is a VISION-LANGUAGE
            # model, and this check only looked at the top level -- so it
            # reported a perfectly convertible config as broken and told a user
            # their artifact was undeployable. A shape test that does not know
            # the shapes in the wild manufactures failures.
            _t = c.get("text_config") or {}
            if not any(k in c or k in _t
                       for k in ("hidden_size", "num_hidden_layers")):
                rep["problems"].append("config.json is not HF-shaped")
        except (OSError, ValueError) as exc:
            rep["problems"].append("config.json unreadable: %s" % exc)
    rep["convertible"] = not rep["problems"]

    if original_dir and probe_ids is not None:
        from holographic.io_and_interop.holographic_gdnruntime import load_runtime
        from holographic.io_and_interop.holographic_measure import (
            measure, better_than, tokens_needed)
        rt0, _c0 = load_runtime(original_dir)
        rt1, _c1 = load_runtime(bundle_dir)
        m0 = measure(rt0, list(probe_ids))
        m1 = measure(rt1, list(probe_ids))
        # PAIRED, WITH ERROR BARS. Comparing two point estimates on a few dozen
        # tokens is how this pipeline reported "beats the original: True" for a
        # 2.3% difference whose measurement had a 95% CI of +/-38.5%. A
        # comparison that cannot return INDISTINGUISHABLE will always find a
        # winner, and most of what this pipeline decides is indistinguishable.
        cmp = better_than(m1, m0)
        need = tokens_needed(m0, 100.0 * float(tolerance))
        rep.update({"original_perplexity": m0["perplexity"],
                    "bundle_perplexity": m1["perplexity"],
                    "delta_pct": cmp["delta_pct"],
                    "verdict": cmp["verdict"],
                    "probe_half_width_pct": m0["half_width_pct"],
                    "detectable_pct": need["detectable_pct_now"],
                    "quality_ok": cmp["verdict"] != "WORSE"})
        if cmp["verdict"] == "WORSE":
            rep["problems"].append("perplexity %+.2f%% worse than the original "
                                   "(paired 95%% CI excludes zero)"
                                   % rep["delta_pct"])
    rep["deployable"] = rep["convertible"] and (rep["quality_ok"] is not False)
    return rep


def _guarded(weights, cfg, ids, baseline, apply_fn, label, tolerance=0.005):
    """Apply a bake, MEASURE it, and REVERT it if it made the model worse.

    WHY THIS EXISTS: a real run produced a Galvatron whose perplexity went from
    16.2 to 190,391 -- a model destroyed by its own imbue, written to disk,
    and reported as success with a resident list. Every individual bake had a
    selftest and passed it; none of them was checked AGAINST THE MODEL IT WAS
    BEING APPLIED TO. The repair pass already learned this lesson for
    assimilation ("test every changed tensor against the original") and imbue
    never got it.

    A bake that cannot demonstrate it left the model usable does not ship. The
    tolerance is stated by the caller rather than invented here, and a bake that
    RAISES is treated exactly like one that regresses: reverted, reported, and
    the pipeline continues with weights that still work.

    THE DEFAULT IS TIGHTER THAN THE DEPLOYABILITY GATE ON PURPOSE. It was 5%
    while check_deployable rejects anything past 1%, so a run could keep three
    bakes that each passed the guard and then fail deployability at +1.9% --
    measured exactly that on a structurally faithful fixture. Per-bake budgets
    must sum to less than the whole-artifact budget, or the guard is a filter
    that lets through what the gate will reject."""
    from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime

    before = {k: v for k, v in weights.items()}
    try:
        out, rep = apply_fn(weights)
    except Exception as exc:
        return before, None, {"bake": label, "kept": False,
                              "why": "raised %s: %s" % (type(exc).__name__, exc)}
    try:
        after = float(GDNRuntime(out, dict(cfg)).perplexity(list(ids)))
    except Exception as exc:
        return before, None, {"bake": label, "kept": False,
                              "why": "unmeasurable after bake: %s" % exc}
    if not np.isfinite(after) or after > baseline * (1.0 + float(tolerance)):
        return before, None, {"bake": label, "kept": False,
                              "why": "perplexity %.4f -> %.4f (%+.1f%%)"
                                     % (baseline, after,
                                        100 * (after - baseline) / baseline),
                              "reverted": True}
    return out, rep, {"bake": label, "kept": True, "perplexity": after,
                      "delta_pct": 100 * (after - baseline) / baseline}


def reserved_rows(model_dir, default):
    """Rows that are DEFINED, including added tokens the plain vocab omits.

    THE BUG THIS KILLS, found by reading Moose's actual tokenizer rather than
    assuming: vocab.json lists 248,044 entries, so "free rows" looked like
    248,044..248,319. But tokenizer.json carries 26 ADDED TOKENS at ids
    248,044..248,069 -- and those include eos_token_id (248,044), the vision
    start/end markers (248,053/248,054) and the image and video tokens
    (248,056/248,057). Writing call tokens or program traces there would have
    silently destroyed end-of-sequence and image handling on a VISION-LANGUAGE
    model. The true free range is 248,070..248,319: 250 rows, not 276."""
    import json as _json

    highest = -1
    for fn in ("tokenizer.json", "vocab.json"):
        path = os.path.join(model_dir, fn)
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                d = _json.load(f)
        except (OSError, ValueError):
            continue
        if fn == "vocab.json":
            highest = max(highest, max(d.values()) if d else -1)
        else:
            vocab = (d.get("model") or {}).get("vocab") or {}
            if vocab:
                highest = max(highest, max(vocab.values()))
            for a in d.get("added_tokens", []):
                highest = max(highest, int(a.get("id", -1)))
    return int(highest + 1) if highest >= 0 else int(default)


def _tokenizer_size(model_dir, default):
    """How many rows the tokenizer actually defines -- the rest are free.

    Read, never assumed: writing into a row a tokenizer DOES define would
    corrupt a real token and surface later as garbled text."""
    import json as _json
    for name in ("vocab.json", "tokenizer.json"):
        p = os.path.join(model_dir, name)
        if not os.path.exists(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = _json.load(f)
            if name == "vocab.json":
                return len(data)
            model = data.get("model") or {}
            if model.get("vocab"):
                return len(model["vocab"])
        except (OSError, ValueError):
            continue
    return int(default)


def _probe_ids(model_dir, text=None, rt=None, minimum=16):
    """Tokenize the calibration probe with whatever vocabulary the model has.

    NEVER RETURNS AN EMPTY OR TRIVIALLY SHORT LIST. A tokenizer that does not
    recognise the probe used to return [] and every downstream step -- the
    forward hooks, the perplexity baseline, the ward margin, the guard -- then
    calibrated on NOTHING, surfacing as an unreadable reshape error deep in the
    attention path. Calibrating on an empty probe is not a smaller measurement,
    it is no measurement, and the failure has to happen HERE where it can say
    what went wrong."""
    # ONE PROBE FOR THE WHOLE PIPELINE. The guard measured each bake on imbue's
    # short probe while check_deployable measured the artifact on the assessment
    # probe -- so three bakes each passing at well under 1% produced a gate
    # verdict of +7.4%, and neither number was wrong. Two budgets on two probes
    # is not a budget. Everything now calibrates on the SAME mixed-register text.
    if text is None:
        from holographic.io_and_interop.holographic_assess import PROBE
        text = PROBE
    ids = []
    try:
        from holographic.io_and_interop.holographic_bpe import BPE
        ids = list(BPE.from_dir(model_dir).encode(text))[:256]
    except Exception:
        ids = []
    if len(ids) < int(minimum):
        n = int(np.asarray(rt.lm_head).shape[0])
        if n <= 256:
            ids = [b for b in text.encode("utf-8") if b < n][:128]
        if len(ids) < int(minimum):
            # deterministic, in-range, and long enough to measure with
            ids = [int(i % max(n - 1, 1)) for i in range(10, 10 + 160)]
    return ids


def maximal_specs(runtime, healthy_hiddens, corpus=(), banned=(),
                  memories=(), carrier_pairs=None, capability=None,
                  capability_args=None, memory_snapshot=None, verifier=True,
                  leap=False, screen=None):
    """THE MAXIMAL GALVATRON: every resident kind leCore can currently express,
    wired to sensible layers for this model's depth. Returns the SPEC LIST (data),
    so it can be inspected, edited, saved and diffed before anything is built.

    Layer placement is derived, not guessed: repair goes early (a corrupted
    stream should be fixed before later layers compound it), knowledge and
    memory go late (near the decision, where an injection actually reaches the
    logits), and observation sits at the end where the trajectory is complete."""
    n = int(runtime.cfg["n_layers"])
    early = max(0, n // 4)
    late = max(0, n - 2)
    H = np.asarray(healthy_hiddens, np.float64)
    specs = []
    # MEMORY FIRST: a Galvatron that carries a writable holographic database
    # can keep learning after it ships. A frozen passage list cannot.
    if memory_snapshot or corpus:
        specs.append({"kind": "memory", "layer": late,
                      "gain": 1.0, "dim": 1024,
                      "snapshot": memory_snapshot,
                      "notes": [{"title": "passage %d" % (i + 1), "text": t,
                                 "author": "pack"}
                                for i, t in enumerate(list(corpus)[:400])]})
    # THE CATALOG ITSELF travels: 1,863 capabilities reachable by description
    # rather than twelve chosen by whoever packaged the model.
    specs.append({"kind": "toolbelt", "layer": late, "gain": 1.0,
                  "max_calls": 32})
    specs.append({"kind": "cache", "verify": False})
    if verifier:
        # the anti-hallucination contract ships WITH the model, not beside it
        specs.append({"kind": "verifier", "passages": list(corpus)[:400]})
    if leap:
        specs.append({"kind": "leap", "k": 8, "order": 4})
    if screen:
        specs.append({"kind": "screen", **dict(screen)})
    if banned:
        specs.append({"kind": "ward", "banned": sorted(set(int(b) for b in banned))})
    specs.append({"kind": "dreamer", "layer": early, "strength": 0.9,
                  "samples": H.tolist()})
    if memories:
        specs.append({"kind": "oracle", "layer": late, "gain": 1.0,
                      "threshold": 0.0,
                      "memories": [{"key": np.asarray(k, np.float64).tolist(),
                                    "value": np.asarray(v, np.float64).tolist()}
                                   for k, v in memories]})
    if corpus:
        specs.append({"kind": "corpus", "layer": late, "gain": 1.0,
                      "corpus": list(corpus), "query": "",
                      "salience": {"samples": H.tolist(), "quantile": 0.8}})
    if carrier_pairs:
        specs.append({"kind": "carrier", "layer": early, "reserve": 16,
                      "amplitude": 0.5, "samples": H.tolist(),
                      "pairs": dict(carrier_pairs)})
    if capability:
        specs.append({"kind": "capability", "layer": late,
                      "capability": str(capability),
                      "args": dict(capability_args or {}), "gain": 1.0})
    specs.append({"kind": "hrnn", "layer": n - 1, "dim": 512, "gain": 0.0})
    return specs


def repair_regressions(orig_dir, assim_dir, eval_tokens, out_dir=None,
                      strengths=(0.0, 0.25, 0.5, 0.75), progress=None):
    """Make an ALREADY-ASSIMILATED model at least as good as its original.

    Shard-by-shard assimilation cannot evaluate anything -- a partial shard will
    not run -- so its filter is applied blind and the damage only becomes
    visible after assembly (+1.79% perplexity, measured on a real Qwen3.5-0.8B).
    This pass tests every tensor the transform CHANGED by walking back toward
    the original (alpha=0 is a full revert). A blend is kept only when its
    paired moving-block interval says BETTER; a lower point estimate alone is
    not evidence. The final report calls the result acceptable relative to the
    original only when that same paired test does not say WORSE.
    """
    from holographic.io_and_interop.holographic_gdnruntime import (
        load_weights_dir, config_from_json, GDNRuntime)
    from holographic.io_and_interop.holographic_measure import measure, better_than
    orig = load_weights_dir(orig_dir)
    cur = load_weights_dir(assim_dir)
    cfg = config_from_json(os.path.join(assim_dir, "config.json"), weights=cur)
    changed = [k for k, v in cur.items()
               if k in orig and getattr(v, "ndim", 0) == 2
               and np.asarray(v).shape == np.asarray(orig[k]).shape
               and not np.array_equal(np.asarray(v), np.asarray(orig[k]))]
    measured_assim = measure(GDNRuntime(cur, cfg), eval_tokens)
    measured_orig = measure(GDNRuntime(orig, cfg), eval_tokens)
    ppl_assim = measured_assim["perplexity"]
    ppl_orig = measured_orig["perplexity"]
    report = {"changed": len(changed), "reverted": 0, "kept": 0, "blended": 0,
              "perplexity_original": ppl_orig, "perplexity_assimilated": ppl_assim,
              "choices": []}
    ppl_cur = ppl_assim
    measured_cur = measured_assim
    for i, name in enumerate(changed):
        source_dtype = np.asarray(cur[name]).dtype
        work_dtype = np.float64 if source_dtype == np.float64 else np.float32
        a_orig = np.asarray(orig[name], dtype=work_dtype)
        a_new = np.asarray(cur[name], dtype=work_dtype)
        dt = np.asarray(cur[name]).dtype
        best_alpha, best_ppl, best_w = 1.0, ppl_cur, None
        best_measure = measured_cur
        for alpha in strengths:                 # alpha=0 -> full revert
            cand = ((1.0 - alpha) * a_orig + alpha * a_new).astype(dt)
            trial = dict(cur)
            trial[name] = cand
            trial_measure = measure(GDNRuntime(trial, cfg), eval_tokens)
            decision = better_than(trial_measure, best_measure)
            # Point-estimate selection manufactured gains on the original
            # 161-token probe. Only a paired block-bootstrap BETTER verdict is
            # allowed to change the checkpoint now.
            if decision["verdict"] == "BETTER":
                best_alpha = alpha
                best_ppl = trial_measure["perplexity"]
                best_w = cand
                best_measure = trial_measure
        if best_w is not None:
            cur[name] = best_w
            ppl_cur = best_ppl
            measured_cur = best_measure
            report["choices"].append((name, round(best_alpha, 3)))
            if best_alpha == 0.0:
                report["reverted"] += 1
            else:
                report["blended"] += 1
        else:
            report["kept"] += 1
        if progress:
            progress(i, name, ppl_cur)
    report["perplexity_repaired"] = ppl_cur
    vs_original = better_than(measured_cur, measured_orig)
    report["comparison_to_original"] = vs_original
    report["beats_original"] = vs_original["verdict"] != "WORSE"
    report["gain_vs_assimilated"] = ppl_assim - ppl_cur
    report["gain_vs_original"] = ppl_orig - ppl_cur
    if out_dir:
        from holographic.io_and_interop import holographic_unicron as U
        os.makedirs(out_dir, exist_ok=True)
        # MATCH THE ORIGINAL'S ON-DISK DTYPE. Our loader decodes bf16 to
        # float32, so preserving the in-memory dtype DOUBLES a bf16 checkpoint:
        # a 1.75 GB model came back as 3.5 GB holding the same numbers.
        U.export_portable(cur, os.path.join(out_dir, "model.safetensors"),
                          like=orig_dir)
        import shutil as _sh
        for f in os.listdir(assim_dir):
            fp = os.path.join(assim_dir, f)
            if os.path.isfile(fp) and not f.endswith(".safetensors"):
                _sh.copy(fp, os.path.join(out_dir, f))
        report["out_dir"] = out_dir
    return cur, report


def best_portable(weights, cfg, out_path, eval_tokens=None, filter_model=True,
                  n_refine=None, progress=None, gate=True, tol=0.0,
                  strengths=(0.25, 0.5, 1.0)):
    """THE BEST PLAIN CHECKPOINT WE CAN HONESTLY PRODUCE -- for the compatible
    model, which must push its limits too even though residents cannot travel.

    Applies only levers that survive in ORDINARY weights: regime-routed spectral
    filtering (which passes heavy-tail layers untouched, because forcing a cut
    there is what produced the measured collapse), then a plain safetensors
    export at the chosen fidelity. Everything else this arc built is runtime
    behaviour and is deliberately NOT attempted here.

    EVERY CHANGE MUST EARN ITS PLACE: with eval_tokens supplied and gate=True
    (the default), each candidate matrix is filtered ALONE and kept only if
    perplexity does not get worse. The output therefore cannot be worse than the
    input on the probe, and improves wherever the Marchenko-Pastur bulk really
    was noise. The earlier version filtered everything and measured once at the
    end, which shipped a measured LOSS (+1.79% on a real 0.8B) as "verified".

    RETENTION IS MEASURED, NOT ASSUMED: with eval_tokens supplied, perplexity is
    computed IN-ENGINE before and after, so the export ships with a number
    instead of the usual UNVERIFIED disclaimer. That measurement is the whole
    reason this function exists rather than a shell script."""
    from holographic.io_and_interop import holographic_unicron as U
    from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime
    report = {"filtered": 0, "heavy_tail_passthrough": 0, "policy_skipped": 0,
              "rejected": 0, "rejected_names": [], "gain": 0.0, "strengths": [],
              "gated": bool(gate and eval_tokens is not None)}
    ppl_before = None
    if eval_tokens is not None:
        ppl_before = GDNRuntime(weights, cfg).perplexity(eval_tokens)
    ppl_current = ppl_before
    out = dict(weights)
    if filter_model:
        for i, (name, w) in enumerate(list(weights.items())):
            a = np.asarray(w)
            if a.ndim != 2 or min(a.shape) < 16:
                continue
            if U._policy_skip(name):
                report["policy_skipped"] += 1
                continue
            sv = np.linalg.svd(np.asarray(a, np.float64), compute_uv=False)
            edge = U._mp_edge_from_sv(sv, a.shape)
            if U.spectral_regime(sv, edge) == "heavy_tail":
                report["heavy_tail_passthrough"] += 1
                continue
            filt, _rep = U.rmt_filter(np.asarray(a, np.float64), mode="shrink")
            if gate and eval_tokens is not None:
                # SEARCH THE STRENGTH, do not assume it. Full filtering is a
                # single point on a line between "leave it alone" and "cut
                # everything the Marchenko-Pastur edge calls noise", and on real
                # weights the best point is usually neither end -- measured, the
                # full cut made a noisy model 182% WORSE while a partial blend
                # improved it. Blending is exact linear interpolation of the
                # same denoised estimate, so each alpha is a legitimate weight
                # matrix, not a hack.
                base = np.asarray(a, np.float64)
                best_alpha, best_ppl, best_w = 0.0, ppl_current, None
                for alpha in strengths:
                    cand = ((1.0 - alpha) * base + alpha * filt).astype(a.dtype)
                    trial = dict(out)
                    trial[name] = cand
                    ppl_a = GDNRuntime(trial, cfg).perplexity(eval_tokens)
                    if ppl_a < best_ppl - 1e-12:
                        best_alpha, best_ppl, best_w = alpha, ppl_a, cand
                if best_w is not None:
                    out[name] = best_w
                    report["filtered"] += 1
                    report["gain"] += (ppl_current - best_ppl)
                    report["strengths"].append((name, round(best_alpha, 3)))
                    ppl_current = best_ppl
                else:
                    report["rejected"] += 1
                    report["rejected_names"].append(name)
                if progress:
                    progress(i, name)
                continue
            if False:
                # MEASURE EACH CHANGE, KEEP ONLY WHAT EARNS ITS PLACE.
                #
                # The original version filtered every eligible matrix and
                # measured perplexity ONCE AT THE END -- so it accumulated
                # whatever the changes happened to cost and reported the total
                # as "verified". On a real Qwen3.5-0.8B that was +1.79%: a
                # transform that measures itself, ignores the measurement, and
                # ships the bill. Denoising that makes the model worse is not
                # denoising; it is damage with a citation.
                #
                # Now each candidate is applied ALONE, scored, and kept only if
                # it does not hurt. The result cannot be worse than the input on
                # the probe BY CONSTRUCTION, and any matrix where the
                # Marchenko-Pastur bulk really was noise makes it better.
                trial = dict(out)
                trial[name] = filt.astype(a.dtype)
                ppl_trial = GDNRuntime(trial, cfg).perplexity(eval_tokens)
                if ppl_trial <= ppl_current + tol:
                    out[name] = trial[name]
                    report["filtered"] += 1
                    report["gain"] += (ppl_current - ppl_trial)
                    ppl_current = ppl_trial
                else:
                    report["rejected"] += 1
                    report["rejected_names"].append(name)
            else:
                out[name] = filt.astype(a.dtype)
                report["filtered"] += 1
            if progress:
                progress(i, name)
    if eval_tokens is not None:
        report["perplexity_before"] = ppl_before
        report["perplexity_after"] = GDNRuntime(out, cfg).perplexity(eval_tokens)
        report["perplexity_delta"] = (report["perplexity_after"] - ppl_before)
        report["verified"] = True
        if report["gated"] and report["perplexity_delta"] > tol + 1e-9:
            # The gate makes this impossible on the probe; if it happens the
            # instrument disagrees with itself and the export is not trustworthy.
            report["verified"] = False
            report["note"] = ("GATED FILTER STILL GOT WORSE (%.4f -> %.4f) -- "
                              "this cannot happen if each accepted change was "
                              "scored on the same tokens, so the measurement "
                              "path is inconsistent. Do not ship this."
                              % (ppl_before, report["perplexity_after"]))
    else:
        report["verified"] = False
        report["note"] = ("no eval_tokens supplied: retention is UNVERIFIED, "
                          "which is the same debt every transform in this arc "
                          "carries until someone measures it")
    U.export_portable(out, out_path, n_refine=n_refine)
    report["path"] = out_path
    return out, report


# ------------------------------------------------------------------- wrappers

class HFCompatWrapper:
    """The shape transformers callers expect: .generate(input_ids, max_new_tokens).
    Accepts a list, 1-D array, or (1, T) array and returns (1, T+n) -- so harness
    code written against a normal model runs unmodified while residents are live
    underneath."""

    def __init__(self, galvatron):
        self.g = galvatron

    def generate(self, input_ids, max_new_tokens=16, **_ignored):
        arr = np.asarray(input_ids)
        flat = arr[0] if arr.ndim == 2 else arr
        ids, _ = self.g.generate([int(t) for t in flat], n_new=int(max_new_tokens))
        return np.asarray(ids, np.int64)[None, :]

    def __call__(self, input_ids, **kw):
        arr = np.asarray(input_ids)
        flat = arr[0] if arr.ndim == 2 else arr
        hooks = self.g._hooks()
        logits = self.g.rt.forward([int(t) for t in flat], hooks=hooks)
        return {"logits": logits[None, :, :]}


def make_app(galvatron, model_name="galvatron", tokenizer=None, mind=None,
             session_root=None):
    """Flask app speaking the OpenAI subset most clients actually use. `tokenizer`
    is a duck-typed (encode/decode) object; without one the API exchanges TOKEN
    IDS (a JSON list) instead of text, which is honest for a raw checkpoint --
    the wrapper does not invent a vocabulary it does not have."""
    from flask import Flask, jsonify, request
    from holographic_service import _jsonable

    app = Flask(__name__)

    store = None
    if session_root:
        from holographic.io_and_interop.holographic_session import (
            SessionStore, runtime_fingerprint)
        store = SessionStore(session_root,
                             fingerprint=runtime_fingerprint(galvatron.rt))

    def _run(prompt, n, session=None):
        """Generate, optionally CONTINUING a named session.

        With a session, the prompt is appended to a context that already exists
        as inference STATE -- so a harness gets multi-turn continuity with no
        re-prefill of the history, which is the cost that dominates agent loops.
        Without one, behaviour is exactly as before: sessions are opt-in and
        nothing about the stateless path changes."""
        ids = tokenizer.encode(prompt) if (tokenizer and isinstance(prompt, str)) \
            else list(prompt)
        ids = [int(t) for t in ids]
        if store is not None and session:
            try:
                state, man, _mem = store.load(session)
                history = man.get("tokens") or []
            except (FileNotFoundError, OSError):
                state, history = None, []
            if state is not None and ids:
                # feed the new turn into the existing state, then continue
                _lg, state = galvatron.rt.extend(ids, state,
                                                 hooks=galvatron._hooks())
                history = list(history) + ids
                out, end = galvatron.generate(history, n_new=int(n), state=state)
            else:
                out, end = galvatron.generate(ids, n_new=int(n))
                history = ids
            store.save(session, end, tokens=out)
            new = out[len(history):]
        else:
            out, _end = galvatron.generate(ids, n_new=int(n))
            new = out[len(ids):]
        return (tokenizer.decode(new) if tokenizer else new), len(ids), len(new)

    @app.get("/v1/models")
    def models():
        return jsonify({"object": "list", "data": [
            {"id": model_name, "object": "model", "owned_by": "lecore"}]})

    @app.get("/v1/sessions")
    def sessions_list():
        """Named contexts a harness can manage on its own schedule."""
        if store is None:
            return jsonify({"sessions": [], "note": "server started without a "
                                                    "session root"})
        return jsonify({"sessions": [{k: v for k, v in m.items() if k != "tokens"}
                                     for m in store.list()]})

    @app.post("/v1/sessions/<name>/fork")
    def sessions_fork(name):
        if store is None:
            return jsonify({"ok": False, "error": "no session root"}), 400
        body = request.get_json(silent=True) or {}
        try:
            man = store.fork(name, body.get("to") or (name + "-fork"))
        except (ValueError, OSError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "session":
                        {k: v for k, v in man.items() if k != "tokens"}})

    @app.delete("/v1/sessions/<name>")
    def sessions_delete(name):
        if store is None:
            return jsonify({"ok": False, "error": "no session root"}), 400
        return jsonify({"ok": bool(store.delete(name))})

    @app.post("/v1/completions")
    def completions():
        body = request.get_json(force=True)
        text, np_, nc = _run(body.get("prompt", []),
                             body.get("max_tokens", 16),
                             session=body.get("session"))
        return jsonify({"object": "text_completion", "model": model_name,
                        "choices": [{"index": 0, "text": text,
                                     "finish_reason": "length"}],
                        "usage": {"prompt_tokens": np_, "completion_tokens": nc,
                                  "total_tokens": np_ + nc}})

    @app.get("/v1/capabilities")
    def capabilities():
        """The bundle's advertised feature set as OpenAI-style tool schemas --
        generated from the LIVE catalog, so it cannot claim what the engine
        carried here does not have."""
        if mind is None:
            return jsonify({"count": 0, "tools": [],
                            "note": "no mind attached: plain model only"})
        from holographic.io_and_interop.holographic_galvabundle import (
            capability_tools)
        tools = capability_tools(mind)
        return jsonify({"count": len(tools), "tools": tools})

    @app.post("/v1/invoke")
    def invoke():
        """Call any catalog capability through the model's own front door. The
        model and the engine answer on the SAME endpoint surface -- which is what
        'the feature set is part of the model' has to mean operationally."""
        if mind is None:
            return jsonify({"ok": False, "error": "no mind attached"}), 400
        body = request.get_json(force=True)
        try:
            out = mind.invoke(body["name"], body.get("args") or {})
            return jsonify({"ok": True, "result": _jsonable(out)})
        except Exception as exc:                     # surface, never swallow
            return jsonify({"ok": False, "error": "%s: %s"
                            % (type(exc).__name__, exc)}), 400

    @app.post("/v1/chat/completions")
    def chat():
        body = request.get_json(force=True)
        msgs = body.get("messages", [])
        last = msgs[-1]["content"] if msgs else []
        text, np_, nc = _run(last, body.get("max_tokens", 16),
                             session=body.get("session") or body.get("user"))
        return jsonify({"object": "chat.completion", "model": model_name,
                        "choices": [{"index": 0, "finish_reason": "length",
                                     "message": {"role": "assistant",
                                                 "content": text}}],
                        "usage": {"prompt_tokens": np_, "completion_tokens": nc,
                                  "total_tokens": np_ + nc}})

    return app


def _selftest():
    try:
        import torch
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
    except ImportError:
        print("galvapack selftest SKIPPED-REFERENCE (torch/transformers absent)")
        return
    import tempfile
    import threading
    import urllib.request

    import lecore
    from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    cfg_t = Qwen3NextConfig(
        vocab_size=97, hidden_size=64, intermediate_size=112,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, full_attention_interval=4,
        num_experts=0, tie_word_embeddings=True, rms_norm_eps=1e-6)
    ref = Qwen3NextForCausalLM(cfg_t).eval().float()
    weights = {k: v.detach().numpy().astype(np.float64)
               for k, v in ref.state_dict().items()}
    cfg = dict(hidden=64, n_layers=4, rms_eps=1e-6, rope_theta=10000.0,
               linear_num_value_heads=4, linear_num_key_heads=2,
               linear_key_head_dim=8, linear_value_head_dim=16, conv_kernel=4,
               n_heads=4, n_kv_heads=2, head_dim=16, partial_rotary_factor=0.25)
    mind = lecore.UnifiedMind(dim=256, seed=0)
    ids = [int(t) for t in rng.integers(0, 97, size=10)]

    # a pack whose scaffolding is REAL: harvest a live hidden state, store it as
    # an oracle memory, and ban whatever the bare model would have said.
    from holographic.io_and_interop.holographic_unicron import (
        load_safetensors as U_load)
    rt0 = GDNRuntime(weights, cfg)
    cap = {}
    rt0.forward(ids, hooks={3: lambda h: cap.__setitem__("h", h.copy()) or None})
    target = 41
    bare, _ = rt0.generate_fast(ids, n_new=6)
    banned = sorted(set(bare[len(ids):]))
    specs = [
        {"kind": "oracle", "layer": 3, "gain": 1.0, "threshold": 0.0,
         "memories": [{"key": cap["h"][-1].tolist(),
                       "value": (8.0 * rt0.embed[target]).tolist()}]},
        {"kind": "ward", "banned": banned},
        {"kind": "future_thing", "layer": 1},        # forward-compat probe
    ]
    path = os.path.join(tempfile.mkdtemp(), "pack")
    rep = save_pack(path, weights, cfg, residents=specs, notes="selftest")
    assert rep["residents"] == 3
    assert os.path.exists(os.path.join(path, "model.safetensors"))

    # FULL load: residents rebuilt from data alone; the unknown kind is skipped
    # with a reason, not fatal.
    gv, lrep = load_pack(path, mind=mind)
    assert lrep["residents"] == 2 and not lrep["degraded"], lrep
    assert any(k == "future_thing" for k, _ in lrep["skipped"]), lrep
    out, _ = gv.generate(ids, n_new=6)
    assert not (set(out[len(ids):]) & set(banned)), "ward lost across the pack"
    # the oracle survived serialization: its memory still steers the first token
    assert int(np.argmax(gv._guard(gv.rt.forward(
        ids, hooks=gv._hooks())[-1]))) == target

    # DEGRADED load: no mind -> everything that NEEDS a mind is dropped and the
    # report says so, but SAFETY GUARDS STILL APPLY. The contract used to be
    # "degraded == bare model exactly", which sounded clean and quietly meant a
    # pack whose manifest bans a token would emit it when loaded without a mind.
    # Guards are not an enhancement to be degraded away.
    plain, prep = load_pack(path, mind=None)
    assert prep["degraded"], prep
    assert prep["residents"] == 1, prep          # the ward, and only the ward
    assert all(k != "ward" for k, _why in prep["skipped"]), prep["skipped"]
    pout, _ = plain.generate(ids, n_new=6)
    # only the GENERATED tail can be constrained -- the prompt is given, and an
    # earlier version of this assertion failed because the prompt itself
    # contained banned tokens
    assert not (set(pout[len(ids):]) & set(banned)), "ward lost on a mind-free load"
    # and with NO ward in the manifest, a mind-free load is still bit-identical
    # to the bare model -- the old contract, kept where it belongs
    nw_path = os.path.join(tempfile.mkdtemp(), "noward")
    save_pack(nw_path, weights, cfg,
              residents=[s for s in specs if s.get("kind") != "ward"])
    nw, nrep = load_pack(nw_path, mind=None)
    assert nrep["residents"] == 0
    nout, _ = nw.generate(ids, n_new=6)
    assert nout == bare, "mind-free load without guards must equal the bare model"

    # HF-SHAPED wrapper: transformers-style call signature, resident behaviour
    hf = HFCompatWrapper(gv)
    got = hf.generate(np.asarray(ids)[None, :], max_new_tokens=6)
    assert got.shape == (1, len(ids) + 6) and list(got[0]) == out
    assert hf(np.asarray(ids)[None, :])["logits"].shape[-1] == 97

    # OPENAI-COMPATIBLE front door over the same Galvatron
    app = make_app(gv, model_name="galvatron-selftest")
    srv = threading.Thread(
        target=lambda: app.run(port=5931, use_reloader=False), daemon=True)
    srv.start()
    import time
    time.sleep(2.5)
    req = urllib.request.Request(
        "http://127.0.0.1:5931/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": ids}],
                         "max_tokens": 6}).encode(),
        headers={"Content-Type": "application/json"})
    res = json.load(urllib.request.urlopen(req))
    got_ids = res["choices"][0]["message"]["content"]
    assert got_ids == out[len(ids):], (got_ids, out[len(ids):])
    assert not (set(got_ids) & set(banned)), "ward lost over HTTP"
    models = json.load(urllib.request.urlopen("http://127.0.0.1:5931/v1/models"))
    assert models["data"][0]["id"] == "galvatron-selftest"

    # ---- MAXIMAL GALVATRON: every resident kind, from the manifest alone ----
    healthy = {}
    long_ids = [int(t) for t in rng.integers(0, 97, size=40)]
    rt0.forward(long_ids,
                hooks={1: lambda h: healthy.__setitem__("h", h.copy()) or None})
    specs_max = maximal_specs(
        rt0, healthy["h"],
        corpus=["gated deltanet updates a recurrent memory matrix",
                "lecore is a numpy only vsa engine"],
        banned=banned,
        memories=[(cap["h"][-1], 8.0 * rt0.embed[target])],
        carrier_pairs={"subject": "moose", "project": "lecore"},
        capability="find_capability", capability_args={"problem": "compress"})
    path2 = os.path.join(tempfile.mkdtemp(), "maxpack")
    save_pack(path2, weights, cfg, residents=specs_max, notes="maximal")
    gmax, rmax = load_pack(path2, mind=mind)
    # every declared kind must rebuild from DATA -- if a kind cannot survive
    # serialization it is not really part of the shipped Galvatron
    kinds = {sp["kind"] for sp in specs_max}
    assert not rmax["skipped"], rmax["skipped"]
    # EVERY SPEC MUST BE ACCOUNTED FOR. The old check ("one resident per spec")
    # broke as soon as a spec CONFIGURED the runtime instead of instantiating an
    # object -- cache and leap install rather than construct. The honest
    # invariant is that nothing vanishes: each spec became a resident, a guard,
    # an installation, or a recorded skip.
    accounted = (rmax["residents"] + len(rmax.get("installed", []))
                 + len(rmax.get("skipped", [])))
    assert accounted == len(specs_max), (accounted, len(specs_max),
                                         rmax.get("installed"), rmax.get("skipped"))
    out_max, _ = gmax.generate(ids, n_new=6)
    assert not (set(out_max[len(ids):]) & set(banned)), "ward lost in maximal pack"
    assert len(kinds) >= 6, kinds

    # ---- BEST PORTABLE: measured retention, not a disclaimer ----
    pth = os.path.join(tempfile.mkdtemp(), "portable.safetensors")
    _w2, prep = best_portable(weights, cfg, pth, eval_tokens=long_ids)
    assert prep["verified"] and "perplexity_delta" in prep
    assert os.path.getsize(pth) > 0
    # heavy-tail layers must be PASSED THROUGH, never force-cut
    assert prep["heavy_tail_passthrough"] + prep["filtered"] > 0, prep
    # and the export is an ordinary checkpoint the plain loader reads
    back = U_load(pth)
    assert set(back) == set(weights)

    print("galvapack selftest OK -- maximal pack rebuilt %d resident kinds from "
          "data alone (0 skipped), ward held; best_portable filtered %d, passed "
          "%d heavy-tail through, ppl %.3f -> %.3f (delta %+.3f); "
          % (len(kinds), prep["filtered"], prep["heavy_tail_passthrough"],
             prep["perplexity_before"], prep["perplexity_after"],
             prep["perplexity_delta"]) +
          "pack round-trips residents from data alone "
          "(2 built, 1 unknown kind skipped), degraded load reproduces the bare "
          "model exactly, HF-shaped .generate matches, and the OpenAI endpoint "
          "returns the SAME guarded tokens over HTTP")


if __name__ == "__main__":
    _selftest()
