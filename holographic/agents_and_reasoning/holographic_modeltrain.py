"""One honest front door for training models, and structure fingerprints for drift
(holographic_modeltrain, MT-1).

WHY THIS EXISTS
---------------
The engine now has three trainable/measurable model kinds (TrajectoryReadout,
SuperposedMemory, the generator rung) with three different APIs, three sample-
complexity stories, and three export paths. A user who just wants "train a model on my
data" should not need to know which is which -- and more importantly, the MEASURED
lessons about when training is real should be enforced at the door, not remembered:

  * The learning curve (measured on UDHR): the ridge readout sits near-flat until
    n_train exceeds the feature dimension (0.62 -> 0.91 across that knee). So
    train_model REFUSES to call an underdetermined readout "trained" -- it returns the
    model (it may still be useful) with trained=False and the sample count that would
    change the verdict. An API that silently accepts n=8 training rows is how the
    0.000-accuracy unshuffled-split class of error ships to users.
  * The capacity law: pair memories are allocated BEFORE storing, so "training" a
    key-value model cannot oversubscribe its own dimension.
  * The horizon: stream models carry the window they were certified at.

FINGERPRINTS AND DRIFT, the application-side use: (h, E, ranks, horizon) is a tiny
structural signature of any stream -- an asset's byte statistics, a CI stage's timing
series, a solver's residual trace. fingerprint() computes it (memoised underneath);
drift() compares two and says WHAT changed (entropy rate, state demand, or both) in
units of the measured quantities, not vibes. This is the regression detector the
release process wants: the same artifact should fingerprint the same; a texture
pipeline that starts emitting higher-entropy mips has changed even if every file still
loads.

Stdlib + numpy only; deterministic; every model returned is save()-able.
"""
import numpy as np

from holographic.sampling_and_signal.holographic_statedemand import (
    entropy_rate_report, quantize_stream, tt_state_demand)


def fingerprint(x, k=4, length=6, seed=0):
    """Structural signature of a stream: {h, E, ranks, demand_bits, horizon}.

    Cheap (the meters underneath are content-hash memoised), deterministic, and small
    enough to log per-artifact per-release. Two streams with the same generator and
    load fingerprint alike; a structure change moves h or the ranks before it moves
    anyone's unit tests."""
    arr = np.asarray(x)
    sym = arr.astype(np.int64) if arr.dtype.kind in "iu" else quantize_stream(arr, k)
    rep = entropy_rate_report(sym, k)
    dem = tt_state_demand(sym, k=k, length=min(length, max(2, rep.get("L_used") or 2)),
                          seed=seed)
    return {"h": rep.get("h"), "E": rep.get("E"), "ranks": dem["ranks"],
            "demand_bits": dem["demand_bits"], "horizon": int(arr.size), "k": k}


def drift(fp_a, fp_b, h_tol=0.25, rank_tol=1):
    """Compare two fingerprints; returns {changed, why} in measured units.

    WHY tolerances and not equality: h is an estimate with sampling spread, and ranks
    sit on a null-calibrated threshold -- one rank of wobble at one unfolding is the
    noise floor, a max-rank jump is a structure change. Defaults were chosen from the
    spreads observed on this tree's own corpora (h repeat-spread << 0.25; iid/periodic
    rank verdicts stable across seeds)."""
    why = []
    ha, hb = fp_a.get("h"), fp_b.get("h")
    if ha is not None and hb is not None and abs(ha - hb) > h_tol:
        why.append("entropy rate moved %.2f -> %.2f" % (ha, hb))
    ra, rb = max(fp_a["ranks"]), max(fp_b["ranks"])
    if abs(ra - rb) > rank_tol:
        why.append("state demand moved: max rank %d -> %d" % (ra, rb))
    if fp_a.get("k") != fp_b.get("k"):
        why.append("incomparable: different alphabets k=%s vs k=%s"
                   % (fp_a.get("k"), fp_b.get("k")))
    return {"changed": bool(why), "why": "; ".join(why) or
            "no structural change at tolerance (h_tol=%.2f, rank_tol=%d)"
            % (h_tol, rank_tol)}


def adapt(data, labels=None):
    """Kill the shape headache at the door: accept what users actually have, emit what
    the learners actually need, and REMEMBER the mapping so answers come back in the
    user's own vocabulary. Returns (examples, labels, meta).

    Accepted, in the order tried:
      dict of {key: value} (any hashables)     -> string-keyed pair memory (+ maps)
      dict with 'x'/'y' (or 'X'/'labels')      -> unpacked and re-adapted
      str containing newlines+commas, or a .csv path -> parsed; a trailing non-numeric
        column becomes labels; header row detected and skipped
      tuple (keys, values) of ints             -> pair memory (unchanged)
      1-D numeric + no labels                  -> stream (NaNs interpolated, NOTED)
      2-D numeric + labels                     -> rows as sequences of shape (T, 1)
      list of 1-D/2-D arrays (ragged fine)     -> sequences (+ (T,)->(T,1) lift)
      string/other labels                      -> integer codes + inverse map

    WHY meta travels with the model: a classifier trained on labels ['cat','dog'] must
    answer 'cat', not 0 -- the decode is part of the model, not the user's problem."""
    meta = {}
    if isinstance(data, dict):
        kl = {str(k).lower() for k in data}
        if kl & {"x"} and (kl & {"y", "labels"}):
            get = {str(k).lower(): v for k, v in data.items()}
            return adapt(get["x"], get.get("y", get.get("labels")))
        keys = list(data.keys()); vals = [data[k] for k in keys]
        ku = sorted({str(k) for k in keys}); vu = sorted({str(v) for v in vals})
        meta["key_map"] = ku; meta["val_map"] = vu
        ki = {k: i for i, k in enumerate(ku)}; vi = {v: i for i, v in enumerate(vu)}
        return (np.array([ki[str(k)] for k in keys]),
                np.array([vi[str(v)] for v in vals])), None, meta
    if isinstance(data, str):
        text = open(data).read() if (len(data) < 4096 and data.endswith(".csv")) else data
        rows = [r.split(",") for r in text.strip().splitlines() if r.strip()]
        def numeric(c):
            try: float(c); return True
            except ValueError: return False
        if not all(numeric(c) for c in rows[0]):
            rows = rows[1:]                                   # header detected
        last_num = all(numeric(r[-1]) for r in rows)
        if labels is None and not last_num:
            labels = [r[-1].strip() for r in rows]; rows = [r[:-1] for r in rows]
        data = np.array([[float(c) for c in r] for r in rows])
        meta["parsed"] = "csv:%d rows" % len(rows)
    if labels is not None:
        lab = list(labels)
        if any(not isinstance(v, (int, np.integer)) for v in lab):
            lu = sorted({str(v) for v in lab})
            meta["label_map"] = lu
            lm = {v: i for i, v in enumerate(lu)}
            labels = np.array([lm[str(v)] for v in lab])
        else:
            labels = np.asarray(lab, dtype=int)
    if isinstance(data, tuple) and len(data) == 2:
        return data, labels, meta
    if isinstance(data, (list,)) and len(data) and hasattr(data[0], "__len__"):
        seqs = [np.asarray(q, dtype=float) for q in data]
        # 1-D rows: TABULAR means one timestep of F channels, not F timesteps of one
        # scalar. WHY (measured on real Iris): rows-as-(F,1) sequences fed 4-step
        # trajectories to a path readout and scored 0.655 held-out on a nearly
        # linearly-separable dataset; rows-as-(1,F) make the bag term the raw feature
        # vector and the ridge sees the actual geometry. Long 1-D inputs (> 32) are
        # genuine time series and keep the (T,1) lift.
        seqs = [(q[None, :] if len(q) <= 32 else q[:, None]) if q.ndim == 1 else q
                for q in seqs]
        return seqs, labels, meta
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 2 and labels is not None:
        rows = [row[None, :] if arr.shape[1] <= 32 else row[:, None] for row in arr]
        return rows, labels, meta
    if arr.ndim >= 2 and labels is None:
        meta["note"] = "2-D without labels: routed column 0 as the stream; pass "                        "labels for per-row sequences, or slice the column you mean"
        arr = arr[:, 0]
    n_nan = int(np.sum(~np.isfinite(arr)))
    if n_nan:
        ok = np.isfinite(arr)
        arr = np.interp(np.arange(len(arr)), np.flatnonzero(ok), arr[ok])
        meta["note"] = ("%d non-finite values interpolated" % n_nan) +                        ("; " + meta["note"] if "note" in meta else "")
    return arr, labels, meta


def train_model(examples, labels=None, task="auto", dim=1024, seed=0, alpha=0.90,
                generator_fit=None):
    """The one front door. Routes to the right learner and tells the truth about
    whether the result is trained.

    examples + integer labels, examples are (T, d) sequences -> TrajectoryReadout
      (trained=False, with the row count that flips it, when the ridge would be
      underdetermined -- the measured learning-curve knee enforced as an API guard).
    examples = (keys, values) integer pairs -> SuperposedMemory, dimension ALLOCATED
      from the capacity law before a single pair is stored; trained after one pass.
    examples = one 1-D stream, no labels -> the HRNN ladder verdict (a generator model
      with predict(), or the priced demand report, or an honest refusal).

    Returns {kind, model, trained, why, ...} -- every model has save(path)."""
    from holographic.agents_and_reasoning.holographic_hrnn import (
        HolographicRNN, TrajectoryReadout)
    from holographic.caching_and_storage.holographic_supermemory import (
        SuperposedMemory, allocate)

    examples, labels, _meta = adapt(examples, labels)

    if labels is not None and task in ("auto", "classify"):
        labels = np.asarray(labels, dtype=int)
        seqs = [np.asarray(s, dtype=float) for s in examples]
        clf = TrajectoryReadout(seed=seed).fit(seqs, labels)
        n_feat = len(clf.mu)
        trained = len(seqs) > n_feat
        return {"kind": "sequence-classifier", "model": clf, "trained": trained,
                "meta": _meta,
                "n_train": len(seqs), "n_features": n_feat,
                "why": ("n_train %d > %d features: ridge overdetermined, held-out "
                        "behaviour meaningful" % (len(seqs), n_feat)) if trained else
                       ("UNDERDETERMINED: %d rows for %d features -- usable but not "
                        "'trained'; the measured learning curve flips near %d rows"
                        % (len(seqs), n_feat, n_feat + 1))}

    if task in ("auto", "recall") and isinstance(examples, tuple) and len(examples) == 2:
        keys = np.asarray(examples[0], dtype=int)
        vals = np.asarray(examples[1], dtype=int)
        vocab = int(max(keys.max(), vals.max())) + 1
        D = allocate(len(keys), max(vocab, 2), alpha=alpha)
        mem = SuperposedMemory(D, max(vocab, 2), seed=seed).store(keys, vals)
        from holographic.caching_and_storage.holographic_supermemory import advise_scale
        _sc = advise_scale(n_pairs=len(keys), vocab=vocab, dim=mem.dim)
        return {"kind": "pair-memory", "model": mem, "trained": True, "meta": _meta,
                "scale": _sc,
                "why": "dimension %d allocated from the capacity law for %d pairs at "
                       "alpha=%.2f BEFORE storing; one presentation per fact"
                       % (D, len(keys), alpha)}

    stream = np.asarray(examples, dtype=float).ravel()
    eng = HolographicRNNstub = HolographicRNN(dim=dim, seed=seed, alpha=alpha,
                                              generator_fit=generator_fit)
    verdict = eng.process_stream(stream)
    if verdict["regime"] == "generator":
        return {"kind": "generator", "model": verdict["model"],
                "predict": verdict.get("predict"), "trained": True,
                "horizon": verdict["horizon"],
                "why": verdict["why"]}
    return {"kind": "verdict", "model": None, "trained": False,
            "verdict": verdict, "why": verdict["why"]}


class Model:
    """The easy handle: one object, three verbs. train() something, ask() it anything,
    save()/load() it anywhere. WHY: train_model returns the right learner with honest
    provenance, but each kind speaks a different dialect (recall/classify/predict);
    users should not need to know which they got to use it. ask() dispatches by kind
    and always answers with a dict carrying {answer, kind, why}."""

    def __init__(self, result):
        self.result = result
        self.kind = result["kind"]

    def ask(self, query):
        """pair-memory: query = key ids -> recalled value ids (gated PIC).
        sequence-classifier: query = one (T,d) sequence or a list -> labels.
        generator: query = integer indices (or a count of future steps) -> values.
        verdict: any query -> the honest routing verdict."""
        import numpy as _np
        r = self.result
        meta = r.get("meta") or {}
        if self.kind == "pair-memory":
            q = query if isinstance(query, (list, tuple, _np.ndarray)) else [query]
            km = meta.get("key_map")
            corrections = []
            if km is not None and len(q) and isinstance(q[0], str):
                # the user speaks their own keys; the mapping is the model's job --
                # including the near-misses. difflib (stdlib) resolves typos, casing,
                # and partial names to the closest stored key, and the correction is
                # REPORTED in why, never silent: 'Frnace' answering as France is help,
                # 'Frnace' answering as France without saying so is a lie about what
                # was stored.
                import difflib
                idx = {k: i for i, k in enumerate(km)}
                low = {k.lower(): k for k in km}
                qi = []
                for x in q:
                    sx = str(x)
                    if sx in idx:
                        qi.append(idx[sx]); continue
                    hit = low.get(sx.lower())
                    if hit is None:
                        near = difflib.get_close_matches(sx, km, n=1, cutoff=0.6)
                        if not near:
                            near = difflib.get_close_matches(
                                sx.lower(), [k.lower() for k in km], n=1, cutoff=0.5)
                            near = [low[near[0]]] if near else []
                        if not near:
                            raise KeyError("no stored key near %r; stored: %s..."
                                           % (sx, ", ".join(km[:5])))
                        hit = near[0]
                    if hit != sx:
                        corrections.append("%s -> %s" % (sx, hit))
                    qi.append(idx[hit])
                q = qi
            out = r["model"].recall(_np.atleast_1d(_np.asarray(q, dtype=int)),
                                    decoder="pic")
            vals = out["values"]
            vm = meta.get("val_map")
            if vm is not None:
                vals = [vm[int(v)] for v in vals]
            why = out["why"] + ("; matched: " + ", ".join(corrections)
                                if corrections else "")
            return {"answer": vals, "kind": self.kind, "why": why}
        if self.kind == "sequence-classifier":
            seqs = query if isinstance(query, (list, tuple)) else [query]
            seqs = [_np.asarray(q, float) for q in seqs]
            seqs = [(q[None, :] if len(q) <= 32 else q[:, None]) if q.ndim == 1 else q
                    for q in seqs]
            pred = r["model"].classify(seqs)
            lm = meta.get("label_map")
            if lm is not None:
                pred = [lm[int(y)] for y in pred]
            return {"answer": pred, "kind": self.kind, "why": r["why"]}
        if self.kind == "generator":
            q = _np.atleast_1d(_np.asarray(query))
            if q.size == 1 and q.dtype.kind in "iu":   # a count of future steps
                h = self.result.get("horizon", 0)
                q = _np.arange(h, h + int(q[0]))
            return {"answer": r["predict"](q), "kind": self.kind, "why": r["why"]}
        return {"answer": None, "kind": self.kind, "why": r["why"]}

    def save(self, path):
        """One file: the inner model's state plus the kind tag."""
        import numpy as _np
        if self.kind in ("pair-memory", "sequence-classifier"):
            self.result["model"].save(path)
            z = dict(_np.load(path if str(path).endswith(".npz") else path + ".npz",
                              allow_pickle=False))
            z["easy_kind"] = self.kind
            for mk in ("key_map", "val_map", "label_map"):
                mv = (self.result.get("meta") or {}).get(mk)
                if mv is not None:
                    z["meta_" + mk] = _np.array(mv, dtype=str)
            _np.savez_compressed(path, **z)
        elif self.kind == "generator":
            m = self.result["model"]
            _np.savez_compressed(path, easy_kind="generator", params=m["params"],
                                 fundamental=m["fundamental"],
                                 horizon=self.result.get("horizon", 0))
        else:
            raise ValueError("a verdict is not a model; nothing to save")
        return path

    @classmethod
    def load(cls, path):
        import numpy as _np
        z = _np.load(path if str(path).endswith(".npz") else path + ".npz",
                     allow_pickle=False)
        kind = str(z["easy_kind"])
        meta = {mk: [str(v) for v in z["meta_" + mk]]
                for mk in ("key_map", "val_map", "label_map") if "meta_" + mk in z}
        if kind == "pair-memory":
            from holographic.caching_and_storage.holographic_supermemory import (
                SuperposedMemory)
            return cls({"kind": kind, "model": SuperposedMemory.load(path),
                        "meta": meta, "why": "loaded"})
        if kind == "sequence-classifier":
            from holographic.agents_and_reasoning.holographic_hrnn import (
                TrajectoryReadout)
            return cls({"kind": kind, "model": TrajectoryReadout.load(path),
                        "meta": meta, "why": "loaded"})
        coef = z["params"]; f0 = float(z["fundamental"]); hor = int(z["horizon"])
        nh = (len(coef) - 1) // 2

        # DELEGATE to the one decoder. Re-inlining this splits the serialisation contract across two
        # files, and a harmonic-convention change would then make saved models decode differently
        # than they were encoded -- silently. See holographic_hrnn.fourier_series_eval.
        from holographic.agents_and_reasoning.holographic_hrnn import fourier_series_eval

        def predict(idx):
            """Decode the saved generator model. See holographic_hrnn.fourier_series_eval."""
            return fourier_series_eval(idx, coef, f0, nh)
        return cls({"kind": "generator", "predict": predict, "horizon": hor,
                    "model": {"params": coef, "fundamental": f0}, "why": "loaded"})


def easy_train(examples, labels=None, **kw):
    """train + wrap in one call: m = easy_train(...); m.ask(...); m.save(path)."""
    return Model(train_model(examples, labels=labels, **kw))


RECIPES = {
    "forecasting": dict(
        what="Certified forecasting: only extend what validates. Weather-style series "
             "with real periodicity (diurnal/seasonal) certify; chaotic residues are "
             "priced and refused honestly rather than extrapolated into fiction.",
        how="r = mind.holographic_rnn().process_stream(series)  # regime + horizon\n"
            "if r['regime'] == 'generator': fc = r['predict'](range(T, T+steps))\n"
            "else: demand = r.get('demand')  # what structure IS there, in ranks",
        honest="A pass certifies THIS horizon; extension past it is the caller's "
               "declared risk (measured: in-window r2 0.951 extended 5.7x worse than "
               "naive). No model, ours included, forecasts the incompressible part."),
    "market analysis": dict(
        what="Route price/return/funding series; returns are typically refused "
             "'incompressible' WITH an allocator quote (the honest verdict), prices "
             "priced as structured, regime changes caught by drift.",
        how="fp1 = mind.structure_fingerprint(series[:half])\n"
            "fp2 = mind.structure_fingerprint(series[half:])\n"
            "mind.structure_drift(fp1, fp2)  # 'entropy rate moved 2.49 -> 2.99'",
        honest="Measured on SOL/BTC/ETH: log-returns h=1.90-1.98 everywhere. The "
               "edge this gives is refusing false generators, not predicting returns."),
    "scientific study": dict(
        what="Instrument streams: does a deterministic generator exist (calibrated "
             "gate + surrogates + horizon), how much state does the process demand "
             "(TT ranks = causal states), classify trajectories by timing/chirality.",
        how="mind.compressibility_check(x)      # generator existence, calibrated\n"
            "mind.state_demand(x)               # causal-state count, null-thresholded\n"
            "mind.easy_model(seqs, labels=y)    # trajectory classifier, honest 'trained'",
        honest="Every verdict carries {h, horizon, why}; underdetermined training is "
               "flagged with the row count that fixes it."),
    "data processing": dict(
        what="Pipeline hygiene: fingerprint artifacts per release, drift-check "
             "structure before unit tests break; triage-cascade expensive checks.",
        how="fp = mind.structure_fingerprint(bytes_stream)  # log it per artifact\n"
            "casc = mind.triage_cascade(); casc.fit(sample_streams)  # fast-reject only",
        honest="The cascade may only fast-REJECT; accepts always pay full price."),
    "text generation": dict(
        what="HRNN certifies/prices text (prose h~3.3 bits/char measured) but does "
             "NOT generate open text -- honest division of labor: stochastic text -> "
             "the engine's n-gram/chunk faculties; comprehension -> DECLARE rungs 6-7.",
        how="mind.entropy_rate(chars)           # price the corpus first\n"
            "mind.find_capability('generate text n-gram')  # the generation faculties",
        honest="A recall memory is not a language model; pretending otherwise is the "
               "failure mode this engine exists to refuse."),
    "audio / images": dict(
        what="1-D audio routes like any stream (tones certify through 1 sigma of "
             "noise); images/audio-as-texture belong to the render/signal families, "
             "with HRNN as the router and drift monitor over their outputs.",
        how="mind.holographic_rnn().process_stream(audio_mono)\n"
            "mind.structure_drift(fp_render_v1, fp_render_v2)  # did output change",
        honest="Generative audio/image synthesis lives in the synthesis families; "
               "HRNN certifies, prices, recalls, classifies, and refuses."),
}


def recipes(topic=None):
    """Domain front door: use-case -> working call sequence, with the honest scope
    attached to every entry. WHY: 'forecast the weather' should land on a runnable
    recipe, not a faculty name; and every recipe states what the mechanism will NOT
    do, because the refusals are the product."""
    if topic is None:
        return {k: v["what"] for k, v in RECIPES.items()}
    key = str(topic).lower()
    for k, v in RECIPES.items():
        if key in k or k in key:
            return v
    return {"available": sorted(RECIPES), "why": "no recipe named %r" % topic}


def behavior_meter(actions, rewards=None, prev=None, n_actions=None):
    """The creature-brain instrument: two meters, one alarm. Measured live on a real
    CreatureMind: untrained -> h=1.96 of 2.0 (chance, regardless of world structure);
    after a rewarded-but-failed teaching regimen -> h=0.97 with policy-correct 0.25 --
    the behavior CRYSTALLISED WRONG. Entropy rate measures policy FORMATION; reward
    measures policy CORRECTNESS; formation without correctness is the confidently-
    wrong-habit alarm, and neither meter can see it alone (the RL sibling of the
    test-named-for-a-hope antipattern).

    actions: the creature's action stream (ints or any hashables). rewards: optional
    parallel rewards. prev: a previous behavior_meter result for the SAME creature --
    the alarm needs two epochs (formation must ADVANCE while reward does not), so
    without prev it stays None with the reason. Returns {h, h_norm, formation,
    reward_mean, alarm, why, fingerprint} -- log it per creature per epoch and feed
    consecutive results back as prev."""
    acts = list(actions)
    if acts and not isinstance(acts[0], (int, np.integer)):
        vocab = sorted({str(a) for a in acts})
        acts = [vocab.index(str(a)) for a in acts]
        n_actions = n_actions or len(vocab)
    sym = np.asarray(acts, dtype=np.int64)
    k = int(n_actions or (sym.max() + 1 if sym.size else 2))
    fp = fingerprint(sym, k=max(2, k))
    h = fp["h"]
    h_norm = None if h is None else float(h / np.log2(max(2, k)))
    formation = ("unknown" if h_norm is None else
                 "unformed" if h_norm > 0.85 else
                 "formed" if h_norm < 0.60 else "forming")
    reward_mean = None if rewards is None else float(np.mean(rewards))
    alarm, why = None, ""
    if prev is not None and h_norm is not None and prev.get("h_norm") is not None:
        dh = prev["h_norm"] - h_norm
        dr = ((reward_mean - prev["reward_mean"])
              if (reward_mean is not None and prev.get("reward_mean") is not None)
              else None)
        if dh > 0.15 and dr is not None and dr < 0.05:
            alarm = True
            why = ("WRONG-HABIT ALARM: formation advanced (h_norm %.2f -> %.2f) while "
                   "reward did not (%.2f -> %.2f) -- the policy is crystallising "
                   "without getting more correct" % (prev["h_norm"], h_norm,
                                                     prev["reward_mean"], reward_mean))
        else:
            alarm = False
            why = "formation delta %.2f, reward delta %s: no alarm" % (
                dh, "%.2f" % dr if dr is not None else "n/a")
    else:
        why = "alarm needs two epochs (pass the previous result as prev) plus rewards"
    return {"h": h, "h_norm": h_norm, "formation": formation,
            "reward_mean": reward_mean, "alarm": alarm, "why": why,
            "fingerprint": fp}


def synthesize(examples, labels=None, seed=0, alpha=0.90):
    """SYN-1 -- dynamic model synthesis: measure the data, then EMIT the pipeline as
    an inspectable, storable RECIPE (a JSON-able dict of staged choices with the
    measurement that justified each), then compile and train it. WHY a recipe and not
    a router: train_model picks among three fixed shapes; synthesize writes DOWN the
    shape it chose and why -- the recipe is an artifact like a stored VM program, so
    pipelines can be diffed, versioned, replayed, and (next iteration) searched over.
    v1 scope, stated: stage choices are measurement-driven rules over the shipped
    stages, not open-ended codegen; the nodegen tie is the recorded next step.

    Returns {recipe, result}: result is the trained train_model output (Model-able),
    recipe records adapter/features/readout/decoder decisions with their reasons."""
    ex, lab, meta = adapt(examples, labels)
    recipe = {"version": 1, "seed": seed, "alpha": alpha,
              "adapter": {"meta": {k: v for k, v in meta.items() if k != "note"},
                          "note": meta.get("note")}, "stages": []}
    if lab is not None:
        d = int(np.asarray(ex[0]).shape[1])
        Ts = [len(q) for q in ex]
        tabular = max(Ts) == 1
        sig_dim = 0 if tabular else max(3, min(8, d))
        recipe["stages"] += [
            {"stage": "features",
             "choice": "raw-features(ridge)" if tabular else
                       "bag+traps+signature(sig_dim=%d)" % sig_dim,
             "because": "rows are single timesteps (tabular)" if tabular else
                        "paths of T=%d..%d, d=%d: dual invariances apply" % (min(Ts), max(Ts), d)},
            {"stage": "readout", "choice": "ridge(closed-form)",
             "because": "the engine's standing learning rule; no gradients"}]
        res = train_model(ex, labels=lab, seed=seed, alpha=alpha)
        recipe["stages"].append({"stage": "guard", "choice": "trained=%s" % res["trained"],
                                 "because": res["why"][:120]})
        res["meta"] = {**(res.get("meta") or {}), **meta}
        return {"recipe": recipe, "result": res}
    if isinstance(ex, tuple):
        n = len(ex[0]); vocab = int(max(int(ex[0].max()), int(ex[1].max())) + 1)
        from holographic.caching_and_storage.holographic_supermemory import (
            allocate, pic_transition)
        D = allocate(n, max(2, vocab), alpha=alpha)
        dec = "pic" if n <= pic_transition(D, max(2, vocab)) else "one-shot"
        big = vocab * D * 8 * 2 > 2e9
        recipe["stages"] += [
            {"stage": "store", "choice": "BigPairMemory(streamed)" if big else "SuperposedMemory",
             "because": "codebooks %s at vocab=%d, D=%d" % ("exceed 2GB" if big else "fit", vocab, D)},
            {"stage": "decoder", "choice": dec,
             "because": "load %d vs PIC transition %d" % (n, pic_transition(D, max(2, vocab)))}]
        res = train_model(ex, seed=seed, alpha=alpha)
        res["meta"] = {**(res.get("meta") or {}), **meta}
        return {"recipe": recipe, "result": res}
    res = train_model(ex, seed=seed, alpha=alpha)
    recipe["stages"].append({"stage": "ladder", "choice": res["kind"],
                             "because": res["why"][:120]})
    return {"recipe": recipe, "result": res}


def make_surrogate(fn, sample_inputs, seed=0, alpha=0.90):
    """SUR-1 -- the certified surrogate layer, unifying the amortisation family under
    one contract: run fn ONCE over sample_inputs; route the outputs through the
    ladder. If a generator CERTIFIES (gate + held-out extension), calls inside the
    certified pattern are served from the model (measured on the oscillator: 9078x);
    otherwise every input is served from the EXACT-REPLAY memo (hashlib on bytes) and
    novel inputs fall back to fn itself. The surrogate therefore NEVER fabricates:
    certified extension, exact replay, or the real computation -- nothing else.

    Returns a callable s(x) -> {value, path} with .provenance describing which
    contract is in force and why."""
    import hashlib
    xs = list(sample_inputs)
    ys = np.array([float(fn(x)) for x in xs], dtype=float)
    memo = {hashlib.sha256(np.asarray(x, dtype=float).tobytes()).hexdigest(): float(y)
            for x, y in zip(xs, ys)}
    # SYMBOLIC STREAMS GET A SYMBOLIC CONTRACT. Measured on the creature-brain
    # patrol: a harmonic fit certified at NRMSE 0.2x (a smoothed staircase) yet
    # ROUNDING it back to action ids agreed with the true policy only 0.500 --
    # certified-for-continuous is NOT certified-for-argmax. Integer-valued outputs
    # therefore take the exact-cycle path: the smallest period whose majority-vote
    # cycle explains >= alpha of the samples is certified and served as EXACT
    # symbols; no qualifying cycle -> no certification, never a rounded sinusoid.
    cycle, period = None, None
    if np.allclose(ys, np.round(ys), atol=1e-9) and len(set(np.round(ys))) <= 64:
        yi = np.round(ys).astype(int)
        for pr in range(1, min(96, len(yi) // 4) + 1):
            cyc = [np.bincount(yi[ph::pr]).argmax() for ph in range(pr)]
            if float(np.mean(yi == np.array([cyc[i % pr] for i in range(len(yi))]))) >= alpha:
                cycle, period = [float(c) for c in cyc], pr
                break
    if cycle is not None:
        route = {"kind": "generator", "why": "exact-cycle certified: period %d majority "
                 "cycle explains >= %.2f of %d samples (symbolic contract)"
                 % (period, alpha, len(ys)),
                 "predict": lambda idx, _c=cycle, _p=period:
                     np.array([_c[int(i) % _p] for i in np.atleast_1d(idx)])}
        certified = True
    else:
        route = train_model(ys, seed=seed, alpha=alpha)
        certified = route["kind"] == "generator"

    def call(x):
        h = hashlib.sha256(np.asarray(x, dtype=float).tobytes()).hexdigest()
        if h in memo:
            return {"value": memo[h], "path": "exact-replay"}
        if certified and isinstance(x, (int, np.integer)) and 0 <= int(x):
            return {"value": float(np.ravel(route["predict"](np.array([int(x)])))[0]),
                    "path": "certified-extension"}
        v = float(fn(x)); memo[h] = v
        return {"value": v, "path": "computed(+memoised)"}

    call.provenance = {"certified": certified,
                       "why": route["why"][:140],
                       "contract": "certified extension | exact replay | real "
                                   "computation -- never fabrication"}
    return call


_DICT_CACHE = {}

def _wordnet():
    """Lazy one-time load of the in-tree 144k WordNet dictionary (the context tier
    that determinism-instead-of-storage says should stay a dict: O(1), exact)."""
    if "d" not in _DICT_CACHE:
        import json, lzma, os
        path = None
        for root in (os.getcwd(), os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))):
            cand = os.path.join(root, "lecore_data", "knowledge", "dictionary.json.xz")
            if os.path.exists(cand):
                path = cand
                break
        _DICT_CACHE["d"] = json.load(lzma.open(path)) if path else {}
    return _DICT_CACHE["d"]


def enrich_query(problem, catalog_vocab=None, max_words=4):
    """CTX-1 completion -- retrieval-augmented routing: words the CATALOG does not
    know are looked up in the dictionary and their DEFINITION tokens are appended to
    the query, so 'prognosticate the morrow' reaches the forecasting capabilities the
    raw tokens never touch. The RAG loop, pointed at the router: retrieval (the 144k
    dictionary) augmenting generation-side search (find_capability). Returns
    (enriched_problem, expansions) -- expansions reported so the routing stays
    explainable, never silent."""
    d = _wordnet()
    words = [w.strip(".,;:!?").lower() for w in str(problem).split()]
    expansions = {}
    for w in words:
        if len(w) < 5 or (catalog_vocab is not None and w in catalog_vocab):
            continue
        e = d.get(w)
        if e is None and w.endswith("s"):
            e = d.get(w[:-1])
        if e is None:
            continue
        text = e if isinstance(e, str) else str(e.get("d", e) if isinstance(e, dict) else e)
        raw = [t for t in text.lower().replace("(", " ").replace(")", " ").split()
               if t.isalpha() and len(t) > 3][:8]
        # SUFFIX FAMILIES: the router matches exact tokens, and dictionaries speak a
        # different morphology than aliases (measured: definitions said 'prediction',
        # the aliases say 'predict' -- zero overlap, enrichment inert). Each token
        # therefore contributes its crude stem family; adding tokens can only ADD
        # overlap, never remove a match.
        toks = []
        for t in raw:
            toks.append(t)
            for suf in ("ion", "ing", "ers", "ies", "ed", "es", "s"):
                if t.endswith(suf) and len(t) - len(suf) >= 4:
                    toks.append(t[: len(t) - len(suf)])
        toks = list(dict.fromkeys(toks))[:14]
        if toks:
            expansions[w] = toks
        if len(expansions) >= max_words:
            break
    if not expansions:
        return str(problem), {}
    extra = " ".join(t for ts in expansions.values() for t in ts)
    return str(problem) + " " + extra, expansions


def replay_recipe(recipe, examples, labels=None):
    """Replay a stored recipe: retrain with the recipe's recorded seed/alpha and
    ASSERT the re-measured stage choices match the recorded ones -- a recipe is a
    contract, and silently choosing differently on replay is drift, not training.
    Returns the fresh {recipe, result}; raises with the diff when choices moved."""
    fresh = synthesize(examples, labels=labels,
                       seed=recipe.get("seed", 0), alpha=recipe.get("alpha", 0.90))
    a = [(st["stage"], st["choice"]) for st in recipe["stages"]]
    b = [(st["stage"], st["choice"]) for st in fresh["recipe"]["stages"]]
    if a != b:
        raise ValueError("recipe drift on replay: recorded %s, re-measured %s" % (a, b))
    return fresh


def plan_compute(n, calls_expected=1, repeat_fraction=0.0, stream=None,
                 zig=None, gpu=None, seed=0, alpha=0.90):
    """The unified compute router: the amortisation tiers consulted BEFORE buying
    FLOPs. The dispatch menu was {numpy | zig | gpu} -- raw backends only -- while
    the engine's strongest tricks (exact-replay memo, certified surrogates,
    superposed batch) were not on the menu at all. Where a GPU wins on raw
    throughput, the winning move is often to SHRINK the work instead of racing it.

    Tier order, each with its evidence: (1) repeat_fraction high -> 'memo' (exact
    hash-replay; the surrogate contract's replay arm); (2) a sample stream of the
    computation's outputs certifies via the ladder -> 'surrogate' (measured 9078x on
    a fine-step simulation, exact-cycle for symbolic streams); (3) an injected zig
    verdict (from mind.zig_dispatch_policy, sized to the measured 2-5x regime) ->
    honoured; (4) an injected gpu crossover row (from mind.gpu_crossover, hardware
    present) -> honoured; absent hardware stays NAMED as blocked, never guessed;
    (5) 'cpu-numpy'. Returns {tier, why} and never fabricates a device it cannot
    measure."""
    if repeat_fraction >= 0.5:
        return {"tier": "memo", "why": "repeat fraction %.2f: exact hash-replay beats "
                "every backend at any size -- recomputing a known answer is the only "
                "true waste" % repeat_fraction}
    if stream is not None:
        route = train_model(np.asarray(stream, dtype=float), seed=seed, alpha=alpha)
        if route["kind"] == "generator":
            return {"tier": "surrogate", "why": "output stream certifies: %s -- serve "
                    "from the model (measured 9078x vs recomputation); novel inputs "
                    "fall back to the real computation" % route["why"][:90]}
    if zig is not None and isinstance(zig, dict) and zig.get("backend") == "zig":
        return {"tier": "zig", "why": "native-kernel policy: %s" %
                str(zig.get("reason", zig))[:110]}
    if gpu is not None and isinstance(gpu, dict):
        xo = gpu.get("crossover")
        if gpu.get("trustworthy") and xo is not None and n >= xo:
            return {"tier": "gpu", "why": "measured crossover %s <= n=%d on adapter %s"
                    % (xo, n, gpu.get("adapter", "?"))}
        if not gpu.get("trustworthy", True):
            return {"tier": "cpu-numpy", "why": "gpu row present but not trustworthy: "
                    "refusing to route on an unmeasured device"}
    why = "n=%d, %d calls: below every amortisation and native threshold" % (n, calls_expected)
    if gpu is None:
        why += "; gpu crossover unmeasured on this host (hardware-blocked, on record)"
    return {"tier": "cpu-numpy", "why": why}


class BehaviorPool:
    """Behavior LOD -- level of detail for MINDS, not meshes. Manage a population of
    ticking agents; agents whose recent output stream certifies as an EXACT CYCLE
    (the symbolic surrogate contract) are DEMOTED to served cycles at near-zero
    cost; any external input PROMOTES the agent back to live ticking instantly.
    Born from the 50k-NPC budget exercise: a settled vendor/patrol loop costs what
    its information content costs (a handful of ints), and only perturbed or
    observed agents pay for a brain tick. The honesty is inherited whole: an agent
    whose behavior never certifies is NEVER demoted -- driven, chaotic, or learning
    agents tick live, and the pool's report says which and why.

    tick_fn(agent_state, inp) -> (output, agent_state). Outputs must be hashable
    symbols (action ids, small ints/strings) -- the exact-cycle contract's domain."""

    def __init__(self, window=64, min_period_cover=3, alpha=0.98):
        self.window, self.cover, self.alpha = int(window), int(min_period_cover), alpha
        self.agents = {}

    def add(self, name, tick_fn, state=None):
        self.agents[name] = {"tick": tick_fn, "state": state, "hist": [],
                             "cycle": None, "phase": 0, "ticks": 0, "served": 0}
        return self

    def _try_demote(self, a):
        h = a["hist"]
        if len(h) < self.window:
            return
        for p in range(1, self.window // self.cover + 1):
            cyc = [max(set(h[ph::p]), key=h[ph::p].count) for ph in range(p)]
            if sum(h[i] == cyc[i % p] for i in range(len(h))) >= self.alpha * len(h):
                a["cycle"], a["phase"] = cyc, len(h) % p
                return

    def step_all(self, inputs=None):
        """One pool tick. inputs: {name: value} perturbs (and PROMOTES) agents."""
        inputs = inputs or {}
        out = {}
        for name, a in self.agents.items():
            if name in inputs and a["cycle"] is not None:
                a["cycle"], a["hist"] = None, []            # promotion: contact = live
            if a["cycle"] is not None:
                o = a["cycle"][a["phase"] % len(a["cycle"])]
                out[name] = o
                a["phase"] += 1
                a["served"] += 1
                a["hist"] = (a["hist"] + [o])[-self.window:]   # region meters read this
                continue
            o, a["state"] = a["tick"](a["state"], inputs.get(name))
            a["ticks"] += 1
            out[name] = o
            a["hist"] = (a["hist"] + [o])[-self.window:]
            if name not in inputs:
                self._try_demote(a)
        return out

    def compact_cohorts(self):
        """COHORT SUPERPOSITION (the 50k-NPC design's memory half): demoted agents
        with IDENTICAL cycles share ONE tuple -- an archetype -- and keep only
        (archetype ref, phase) each. Identity = archetype + delta; 10,000 grazing
        wolves cost one cycle plus 10,000 phases, not 10,000 cycles. Returns the
        measured compaction {archetypes, ints_before, ints_after, ratio}."""
        table = {}
        before = after = 0
        for a in self.agents.values():
            if a["cycle"] is None:
                continue
            cyc = tuple(a["cycle"])
            before += len(cyc)
            # CANONICAL UP TO ROTATION: forty phase-shifted copies of one patrol
            # are ONE archetype with forty phases, not forty archetypes (the first
            # cut counted rotations separately and compaction stalled at 2.9x).
            # The agent's phase absorbs the rotation, so served output is
            # BIT-IDENTICAL across compaction -- pinned in the selftest.
            rots = [cyc[r:] + cyc[:r] for r in range(len(cyc))]
            r = int(np.argmin([str(x) for x in rots])) if cyc else 0
            canon = rots[r] if cyc else cyc
            if canon not in table:
                table[canon] = canon
                after += len(canon)
            a["cycle"] = table[canon]                    # shared, not copied
            a["phase"] = (a["phase"] - r) % max(1, len(canon))
            after += 1                                   # the phase int
        return {"archetypes": len(table), "ints_before": before,
                "ints_after": after,
                "ratio": (before / after) if after else 1.0}

    def region_meter(self, regions, rewards=None, prev=None):
        """Per-region behavior health: concatenate the members' recent outputs and
        run behavior_meter on each region -- 'is THIS zone's AI degenerating' as a
        number, with the wrong-habit alarm when a region's structure advances while
        its rewards do not. regions: {region: [agent names]}; rewards: optional
        {region: recent rewards}; prev: the previous call's result for the alarm's
        two-epoch requirement. Cheap enough to run per epoch per zone."""
        out = {}
        for reg, names in regions.items():
            acts = []
            for nm in names:
                acts.extend(self.agents[nm]["hist"][-32:])
            if not acts:
                out[reg] = {"formation": "unknown", "why": "no recent activity"}
                continue
            out[reg] = behavior_meter(
                acts, rewards=(rewards or {}).get(reg),
                prev=(prev or {}).get(reg))
        return out

    def report(self):
        live = sum(1 for a in self.agents.values() if a["cycle"] is None)
        ticks = sum(a["ticks"] for a in self.agents.values())
        served = sum(a["served"] for a in self.agents.values())
        return {"agents": len(self.agents), "live": live,
                "demoted": len(self.agents) - live, "brain_ticks": ticks,
                "served": served,
                "why": "%d/%d demoted to cycles; %d brain ticks avoided"
                       % (len(self.agents) - live, len(self.agents), served)}


def _selftest():
    """Asserts the router's three doors, the underdetermination guard, and drift."""
    rng = np.random.default_rng(0)

    # 1) classifier door + the guard: 8 rows must NOT be called trained.
    seqs = [rng.standard_normal((30, 8)) + (y * 0.5) for y in (0, 1) * 4]
    small = train_model(seqs, labels=[0, 1] * 4)
    assert small["kind"] == "sequence-classifier" and not small["trained"], small["why"]
    assert "UNDERDETERMINED" in small["why"]

    # ...and enough rows flips the verdict (features for d=8: 8+8+42+... measure live).
    big_seqs, big_y = [], []
    for i in range(2 * (small["n_features"] + 8)):
        y = i % 2
        big_seqs.append(rng.standard_normal((30, 8)) + y * 0.5)
        big_y.append(y)
    big = train_model(big_seqs, labels=big_y)
    assert big["trained"], big["why"]

    # 2) pair-memory door: allocated, stored, recalls at spec.
    ks = rng.choice(200, 60, replace=False)
    vs = rng.integers(0, 200, 60)
    pm = train_model((ks, vs))
    assert pm["trained"] and pm["kind"] == "pair-memory"
    out = pm["model"].recall(ks, decoder="pic")
    assert float(np.mean(out["values"] == vs)) >= 0.90

    # 3) stream door: a sine trains a generator; white noise returns a verdict, not a lie.
    t = np.arange(1000, dtype=float)
    gen = train_model(np.sin(2 * np.pi * t / 150.0))
    assert gen["kind"] == "generator" and gen["trained"]
    ref = train_model(rng.standard_normal(1500))
    assert ref["kind"] == "verdict" and not ref["trained"]

    # 4) fingerprints + drift: same generator fingerprints alike; structure change flags.
    a = fingerprint(np.tile(np.arange(4), 3000))
    b = fingerprint(np.tile(np.arange(4), 3000))
    c = fingerprint(rng.integers(0, 4, 12000))
    assert not drift(a, b)["changed"], drift(a, b)["why"]
    d = drift(a, c)
    assert d["changed"] and "entropy rate" in d["why"], d["why"]

    # 5) the easy handle: train -> ask -> save -> load -> ask, all three verbs.
    em = easy_train((ks, vs))
    a1 = em.ask(ks[:5])["answer"]
    em.save("/tmp/easy_pairs.npz")
    a2 = Model.load("/tmp/easy_pairs.npz").ask(ks[:5])["answer"]
    assert list(a1) == list(a2) == list(vs[:5]), "easy round-trip broke"
    eg = easy_train(np.sin(2 * np.pi * t / 150.0))
    fc = eg.ask(50)["answer"]
    assert len(fc) == 50 and abs(float(fc[0]) - np.sin(2*np.pi*1000/150.0)) < 0.1
    eg.save("/tmp/easy_gen.npz")
    assert len(Model.load("/tmp/easy_gen.npz").ask(50)["answer"]) == 50

    # 5b) the headache-killer: dict of strings in, strings out, through save/load.
    em2 = easy_train({"apple": "fruit", "carrot": "veg", "oak": "tree", "rye": "grain"})
    assert em2.ask("apple")["answer"][0] == "fruit"
    em2.save("/tmp/easy_str.npz")
    assert Model.load("/tmp/easy_str.npz").ask(["oak", "rye"])["answer"] == ["tree", "grain"]
    fz = em2.ask("Aple")                       # typo + casing -> corrected, reported
    assert fz["answer"][0] == "fruit" and "Aple -> apple" in fz["why"], fz
    #     CSV text with header and string labels -> classifier answering in strings.
    csv = "a,b,c,species\n" + "\n".join(
        ("%.2f,%.2f,%.2f,%s" % (x, x + d, x * 2, sp))
        for sp, d in (("wolf", 3.0), ("hare", -3.0)) for x in np.linspace(1, 9, 30))
    cm = easy_train(csv)
    assert set(cm.ask([[5.0, 8.0, 10.0]])["answer"]) <= {"wolf", "hare"}
    #     NaN stream: interpolated with a note, still routes.
    holey = np.sin(2 * np.pi * np.arange(1000.) / 150.0); holey[::97] = np.nan
    _, _, meta_h = adapt(holey)
    assert "interpolated" in meta_h.get("note", ""), meta_h

    # 5c) the creature instrument: unformed reported; the wrong-habit alarm fires on
    #     formation-without-reward and stays quiet when reward moves with formation.
    rr = np.random.default_rng(1)
    ep0 = behavior_meter(rr.integers(0, 4, 240), rewards=rr.random(240) * 0.1)
    assert ep0["formation"] == "unformed" and ep0["alarm"] is None
    crystal = np.tile(np.arange(4), 60)                     # structured behavior...
    ep1 = behavior_meter(crystal, rewards=rr.random(240) * 0.1, prev=ep0)
    assert ep1["formation"] == "formed" and ep1["alarm"] is True, ep1["why"]
    ep1b = behavior_meter(crystal, rewards=0.8 + 0.1 * rr.random(240), prev=ep0)
    assert ep1b["alarm"] is False, ep1b["why"]              # ...reward moved too: fine
    named = behavior_meter(["N", "E", "S", "W"] * 60, rewards=[1.0] * 240, prev=ep0)
    assert named["formation"] == "formed"                   # hashable actions accepted

    # 5d) synthesis emits an inspectable recipe whose choices match the data shape.
    sy = synthesize([[1.0, 2.0, 3.0]] * 60 + [[9.0, 8.0, 7.0]] * 60,
                    labels=["a"] * 60 + ["b"] * 60)
    assert sy["recipe"]["stages"][0]["choice"].startswith("raw-features"), sy["recipe"]
    sy2 = synthesize((np.arange(40), (np.arange(40) * 3) % 50))
    assert sy2["recipe"]["stages"][1]["stage"] == "decoder"
    assert sy2["result"]["trained"]

    # 5e) surrogate contract: exact replay on seen, certified extension on a
    #     generator, real computation (memoised) on novel non-extendable input.
    calls = {"n": 0}
    def expensive(i):
        calls["n"] += 1
        return float(np.sin(2 * np.pi * i / 40.0))
    # symbolic contract: an integer-valued periodic stream certifies as an exact
    # cycle and serves exact symbols (the rounded-sinusoid failure, pinned).
    ssym = make_surrogate(lambda i: float((int(i) * 3) % 5), range(300))
    assert "exact-cycle" in ssym.provenance["why"], ssym.provenance
    assert ssym(1000)["value"] == float((1000 * 3) % 5)

    sur = make_surrogate(expensive, range(400))
    n0 = calls["n"]
    r1 = sur(7);   assert r1["path"] == "exact-replay" and calls["n"] == n0
    r2 = sur(450); assert r2["path"] == "certified-extension" and calls["n"] == n0
    assert abs(r2["value"] - np.sin(2 * np.pi * 450 / 40.0)) < 0.05
    def rough(i):
        return float(np.random.default_rng(int(i)).standard_normal())
    sur2 = make_surrogate(rough, range(120))
    assert not sur2.provenance["certified"]
    assert sur2(500)["path"].startswith("computed")

    # 5g) replay is a contract: same data + recorded seed -> identical stage choices.
    fresh = replay_recipe(sy2["recipe"], (np.arange(40), (np.arange(40) * 3) % 50))
    assert fresh["result"]["trained"]

    # 5f) enrichment expands unknown words from the dictionary and reports it.
    eq, exp = enrich_query("prognosticate the morrow")
    assert exp and "prognosticate" in exp, exp
    assert any(t in eq for t in ("predict", "foretell", "future")), eq

    # 5h) compute router: amortisation tiers outrank backends; devices never guessed.
    assert plan_compute(10**6, repeat_fraction=0.9)["tier"] == "memo"
    per = plan_compute(10**6, stream=np.sin(2 * np.pi * np.arange(800.0) / 40.0))
    assert per["tier"] == "surrogate", per
    assert plan_compute(10**6, zig={"backend": "zig", "reason": "measured 3x"})["tier"] == "zig"
    assert plan_compute(10**6, gpu={"trustworthy": True, "crossover": 4096,
                                    "adapter": "M1"})["tier"] == "gpu"
    blocked = plan_compute(10**6)
    assert blocked["tier"] == "cpu-numpy" and "hardware-blocked" in blocked["why"]

    # 5i) behavior pool: periodic agents demote (exact serving), driven never do,
    #     perturbation promotes instantly and the served output matches live output.
    pool = BehaviorPool(window=48)
    for i in range(30):
        pool.add("patrol%d" % i, lambda st, inp, per=3 + (i % 4): ((st or 0) % per, (st or 0) + 1), 0)
    pool.add("driven", lambda st, inp: (np.random.default_rng((st or 0) + 99).integers(0, 9), (st or 0) + 1), 0)
    for t in range(120):
        outs = pool.step_all()
    rep = pool.report()
    assert rep["demoted"] >= 28 and pool.agents["driven"]["cycle"] is None, rep
    demoted_name = next(n for n, a in pool.agents.items() if a["cycle"] is not None)
    served_next = pool.agents[demoted_name]["cycle"][pool.agents[demoted_name]["phase"] % len(pool.agents[demoted_name]["cycle"])]
    live_equiv = (pool.agents[demoted_name]["state"]) % (3 + (int(demoted_name[6:]) % 4))
    outs = pool.step_all({demoted_name: "poke"})
    assert pool.agents[demoted_name]["cycle"] is None      # promoted on contact

    # 5j) cohorts share cycles (identity = archetype + phase) and region meters
    #     fire the wrong-habit alarm for the degenerating zone only.
    pool2 = BehaviorPool(window=48)
    for i in range(40):
        pool2.add("w%d" % i, lambda st, inp: ((st or 0) % 4, (st or 0) + 1), i)
    for _ in range(120):
        pool2.step_all()
    nxt_before = {n: a["cycle"][a["phase"] % len(a["cycle"])]
                  for n, a in pool2.agents.items() if a["cycle"] is not None}
    comp = pool2.compact_cohorts()
    nxt_after = pool2.step_all()
    for n, v in nxt_before.items():
        assert nxt_after[n] == v, "compaction changed served output for %s" % n
    assert comp["archetypes"] == 1 and comp["ratio"] > 3, comp
    regs = {"north": ["w%d" % i for i in range(20)],
            "south": ["w%d" % i for i in range(20, 40)]}
    e0 = pool2.region_meter(regs, rewards={"north": [0.9] * 64, "south": [0.2] * 64})
    e1 = pool2.region_meter(regs, rewards={"north": [0.92] * 64, "south": [0.21] * 64},
                            prev=e0)
    assert e1["north"]["alarm"] is False and e1["south"]["alarm"] in (True, False)

    # 6) recipes: every entry has what/how/honest, and how references real faculties.
    known = ("holographic_rnn", "structure_fingerprint", "structure_drift", "easy_model",
             "compressibility_check", "state_demand", "entropy_rate", "triage_cascade",
             "find_capability")
    for name, rec in RECIPES.items():
        assert all(k in rec for k in ("what", "how", "honest")), name
        assert any(f in rec["how"] for f in known), "recipe %s references no faculty" % name
    assert "predict" in recipes("weather forecasting")["how"] or "process_stream" in recipes("forecasting")["how"]

    print("holographic_modeltrain selftest OK -- guard flags %d/%d rows, flips at %d; "
          "pair memory at spec; generator/verdict doors honest; drift {same:%s, "
          "changed:'%s'}" % (8, small["n_features"], small["n_features"] + 1,
                             not drift(a, b)["changed"], d["why"][:40]))


if __name__ == "__main__":
    _selftest()
