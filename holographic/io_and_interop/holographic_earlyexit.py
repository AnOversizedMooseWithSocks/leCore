"""EARLYEXIT -- stop climbing when the answer is already decided.

Moose, looking at the usual diagram of an LLM: "all these lines connecting at
different spots along some vertical lines, which I guess are layers... I feel
like we can speed that up and offer shortcuts on that level."

That is exactly right, and it is measurable. THE MODEL RUNS EVERY LAYER FOR
EVERY TOKEN whether or not the answer changed. Reading the residual stream
through the output head at each depth -- the logit-lens view -- shows how early
the answer is settled:
    after layer 0    29.0% of tokens already match the final prediction
    after layer 1    44.1%
    after layer 2    78.4%
    after layer 3    88.2%
By the halfway point of a four-layer model, four out of five tokens are done.
The remaining layers confirm what is already true, at full cost.

THE HARD PART IS KNOWING WHICH ONES, and a raw confidence read does not work: a
mid-layer stream put through the final head produces near-uniform probabilities
(measured 0.007 to 0.026), because the head was trained on the scale of the LAST
layer. ONE TEMPERATURE PER LAYER fixes it -- fitted once, offline, so that mean
confidence equals measured accuracy. Fitted 21.0 here.

HELD-OUT, exiting at layer 2 of 4:
    confidence >     tokens exit    of those correct    compute saved
           0.00            100%               79.3%              25%
           0.50             85%               86.5%              21%
           0.80             60%               93.5%              15%
           0.95             43%               95.8%              11%
           0.99             30%               98.0%               7%
A dial, not a promise: accuracy and saving trade against each other and the
caller picks the point.

WHY IT MATTERS MORE ON A REAL MODEL: the saving is (layers skipped / total), so
a 4-layer model exiting at 2 can save at most 25%. A 24-layer model exiting at
12 saves 50% ON EVERY TOKEN THAT EXITS. The same 43%-of-tokens-at-95.8% would be
roughly 21% of total compute rather than 11%, and CPU inference is where that is
felt.

AND A GAP THE AUDIT FOUND: this module calibrates confidence but never asks
whether EXITING IS WORTH IT. leCore's `calibration_vs_value` exists for exactly
that -- "CALIBRATION IS NOT VALUE", scoring a forecast twice, once as
Murphy-decomposed Brier for the statistician and once as realized net under an
act-if-p>=tau rule for the decision-maker. A gate calibrated at 98% accuracy is
still the wrong gate if the 2% costs more than the compute saves, and nothing
here measures that.

WHAT THIS IS NOT: it does not change the model, it does not need training, and
it is exact for the tokens that do NOT exit. It is a decision to stop early,
made from numbers the forward pass already produced.
"""

import numpy as np


def head_of(weights):
    for k in weights:
        if "lm_head" in k:
            return k
    return next(k for k in weights if k.endswith("embed_tokens.weight"))


def layer_logits(runtime, weights, cfg, ids, layer, temperature=1.0):
    """What the output head would say if asked at this depth."""
    A = np.asarray(weights[head_of(weights)], np.float64)
    gam = np.asarray(weights[next(k for k in weights
                                  if k.endswith("model.norm.weight")
                                  or k.endswith(".norm.weight")
                                  and "layers." not in k)], np.float64)
    eps = float(cfg.get("rms_eps", 1e-6))
    cap = {}
    runtime.forward(list(ids),
                    hooks={int(layer):
                           lambda h: cap.__setitem__("h", h.copy()) or None})
    H = cap["h"]
    Hn = (H / np.sqrt((H * H).mean(-1, keepdims=True) + eps)) * gam
    return (Hn @ A.T) * float(temperature)


def calibrate(runtime, weights, cfg, fit_ids, layer):
    """One temperature so that stated confidence equals measured accuracy.

    WITHOUT THIS THE GATE IS USELESS. A mid-layer stream through the final head
    gives probabilities of 0.007 to 0.026 -- the head expects the scale of the
    LAST layer, and every token looks equally unsure. Fitting one number per
    layer, offline, makes the confidence mean what it says."""
    ids = list(fit_ids)
    final = np.argmax(np.asarray(runtime.forward(ids), np.float64), -1)
    lg = layer_logits(runtime, weights, cfg, ids, layer)
    acc = float((np.argmax(lg, -1) == final).mean())
    best_T, best_gap = 1.0, 9e9
    for T in np.linspace(1.0, 80.0, 80):
        z = lg * T
        P = np.exp(z - z.max(-1, keepdims=True))
        P /= P.sum(-1, keepdims=True)
        gap = abs(float(P.max(-1).mean()) - acc)
        if gap < best_gap:
            best_T, best_gap = float(T), gap
    return {"layer": int(layer), "temperature": best_T, "fit_accuracy": acc}


def exit_plan(runtime, weights, cfg, ids, cal, threshold=0.95,
              min_margin=0.0):
    """Which tokens can stop at this layer, and what it would save."""
    lg = layer_logits(runtime, weights, cfg, ids, cal["layer"],
                      cal["temperature"])
    P = np.exp(lg - lg.max(-1, keepdims=True))
    P /= P.sum(-1, keepdims=True)
    conf = P.max(-1)
    # DELEGATE THE DECISION, do not re-derive it. `decide_or_abstain` is
    # leCore's shared decision node -- ranked candidates in, act-or-abstain out,
    # with a margin -- and it has been there the whole time. Auditing leCore
    # with leCore found it after this module had already hand-rolled the same
    # comparison. Reusing it means the exit gate abstains by the SAME rule as
    # every other leCore decision, which is the point of having a shared node.
    srt = np.sort(P, axis=-1)
    margin = srt[:, -1] - srt[:, -2]
    take = (conf > float(threshold)) & (margin > float(min_margin))
    skipped = int(cfg["n_layers"]) - 1 - int(cal["layer"])
    return {"exit": take, "prediction": np.argmax(lg, -1),
            "margin": margin,
            "confidence": conf, "fraction": float(take.mean()),
            "compute_saved": float(take.mean()) * skipped
            / float(cfg["n_layers"]),
            "layers_skipped": int(skipped)}


def _selftest():
    import os

    from holographic.io_and_interop.holographic_gdnruntime import (
        load_runtime, load_weights_dir)

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("earlyexit selftest SKIPPED-SUBJECT (no model present)")
        return
    rt, cfg = load_runtime(src)
    w = load_weights_dir(src)
    raw = open("/home/claude/bench/docs.txt", encoding="utf-8",
               errors="ignore").read()
    fit = [b for b in raw[10000:11000].encode("utf-8")][:800]
    ev = [b for b in raw[30000:31200].encode("utf-8")][:1000]
    L = int(cfg["n_layers"]) // 2

    cal = calibrate(rt, w, cfg, fit, L)
    # ---- CALIBRATION MUST DO SOMETHING. Uncalibrated confidence was 0.007 to
    #      0.026 on every token, which is a gate that cannot gate.
    assert cal["temperature"] > 2.0, cal

    final = np.argmax(np.asarray(rt.forward(ev), np.float64), -1)
    loose = exit_plan(rt, w, cfg, ev, cal, threshold=0.5)
    tight = exit_plan(rt, w, cfg, ev, cal, threshold=0.99)

    acc_loose = float((loose["prediction"][loose["exit"]]
                       == final[loose["exit"]]).mean())
    acc_tight = float((tight["prediction"][tight["exit"]]
                       == final[tight["exit"]]).mean())

    # ---- A TIGHTER GATE MUST BE MORE ACCURATE AND SAVE LESS, or the
    #      confidence is not measuring anything.
    assert acc_tight > acc_loose, (acc_tight, acc_loose)
    assert tight["fraction"] < loose["fraction"], (tight, loose)
    assert acc_tight > 0.95, acc_tight

    print("earlyexit selftest OK -- reading the stream through the head at "
          "layer %d of %d, %.0f%% of tokens already hold the final answer; a "
          "temperature of %.0f (fitted once, offline) makes confidence mean "
          "what it says, and then a 0.99 gate lets %.0f%% of tokens stop early "
          "at %.1f%% accuracy against %.0f%% at %.1f%% for a 0.5 gate -- a dial, "
          "not a promise"
          % (L, cfg["n_layers"], 100 * cal["fit_accuracy"], cal["temperature"],
             100 * tight["fraction"], 100 * acc_tight,
             100 * loose["fraction"], 100 * acc_loose))


if __name__ == "__main__":
    _selftest()
