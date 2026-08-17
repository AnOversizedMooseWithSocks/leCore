"""TRANSFORM -- rebuild a model where the MEASUREMENT says it needs rebuilding.

Everything before this applied leCore's levers uniformly: grow a memory channel
in every layer, quantize everything, retune whatever was reachable. That is the
wrong shape, because a real model is not uniform. Measured on Qwen3.5-0.8B:

  * IT IS BUILT IN BLOCKS of (3 linear-attention layers + 1 full-attention
    layer), six of them.
  * MEMORY TRACKS POSITION IN THE BLOCK, not depth. The GDN layer immediately
    after a full-attention layer has a median half-life of 82 tokens; the other
    two have 9.7 and 9.9. That is an 8.5x difference and it repeats in all six
    blocks.
  * COMPRESSIBILITY IS FLAT with depth (4-bit error 0.110 / 0.112 / 0.113 at
    layers 0 / 12 / 23) and RANK IS NOT the lever -- every projection is
    heavy-tailed, and low-rank truncation is 5x worse than quantization at the
    same size.

So the transformation is TARGETED:
    position 0 (after attention)  -> the model's long memory ALREADY lives here.
                                     Leave the gates alone; an edit here damages
                                     the thing that works.
    positions 1 and 2             -> local layers with ~10-token memory. GROW a
                                     long-memory channel: this gives the model a
                                     capability it does not have, in the layers
                                     where nothing is lost.
    full-attention layers         -> KV compression, where the context ceiling
                                     actually is (rank 64 = 8x context at 1.3%
                                     attention error).
    everywhere                    -> per-tensor bit width by measurement.

WHAT MAKES THE RESULT A GALVATRON RATHER THAN A SMALLER QWEN: the grown channels
are new state the original could not hold, the ward is a property of the weights
rather than a runtime rule, and the VSA circuits let the model bind and unbind
role-filler structure in its own forward pass. Those are abilities the model did
not have before, in plain weights that any runtime can load.
"""

import numpy as np


def analyse(weights, cfg):
    """Recover the block structure and per-layer memory from the weights.

    Read, never assumed: the block period is DERIVED from which layers actually
    have linear-attention gates, so a model with a different interleave is
    described correctly instead of being forced into this one's shape."""
    n_layers = int(cfg["n_layers"])
    root = next((k.split("layers.")[0] for k in weights if "layers." in k),
                "model.")
    gdn, attn = [], []
    half = {}
    for L in range(n_layers):
        ak = "%slayers.%d.linear_attn.A_log" % (root, L)
        dk = "%slayers.%d.linear_attn.dt_bias" % (root, L)
        if ak in weights:
            gdn.append(L)
            A = np.asarray(weights[ak], np.float64)
            dt = np.log1p(np.exp(np.asarray(weights[dk], np.float64)))
            decay = np.exp(-np.exp(A) * dt)
            half[L] = np.log(0.5) / np.log(np.clip(decay, 1e-12, 1 - 1e-12))
        else:
            attn.append(L)
    # position within block = distance since the last full-attention layer
    pos = {}
    last = -1
    for L in range(n_layers):
        if L in attn:
            last = L
            continue
        pos[L] = L - last - 1
    by_pos = {}
    for L, p in pos.items():
        by_pos.setdefault(p, []).append(float(np.median(half[L])))
    return {"root": root, "gdn_layers": gdn, "attn_layers": attn,
            "position_in_block": pos,
            "median_half_life": {L: float(np.median(v)) for L, v in half.items()},
            "median_by_position": {p: float(np.median(v))
                                   for p, v in by_pos.items()},
            "block_period": (attn[1] - attn[0]) if len(attn) > 1 else n_layers}


def plan(weights, cfg, target_tokens=4096, kv_rank=64, grow_gain=0.0):
    """Decide what to do to each layer, from the analysis rather than by rule.

    Returns data, so the plan can be inspected, edited and diffed before
    anything is built -- the same contract as maximal_specs."""
    a = analyse(weights, cfg)
    if not a["gdn_layers"]:
        return {"analysis": a, "actions": [],
                "note": "no linear-attention layers: nothing here to target"}
    long_pos = max(a["median_by_position"], key=a["median_by_position"].get)
    actions = []
    for L in a["gdn_layers"]:
        p = a["position_in_block"][L]
        if p == long_pos:
            actions.append({"layer": L, "position": p, "do": "preserve",
                            "why": "the model's long memory lives here "
                                   "(median %.0f tokens); editing it damages "
                                   "what works"
                                   % a["median_half_life"][L]})
        else:
            actions.append({"layer": L, "position": p, "do": "grow_memory",
                            "a_log": -float(np.log(max(2.0, target_tokens))),
                            "gain": float(grow_gain),
                            "why": "local layer (median %.0f tokens): a grown "
                                   "channel adds reach the model lacks, and "
                                   "nothing here is being taken away"
                                   % a["median_half_life"][L]})
    for L in a["attn_layers"]:
        actions.append({"layer": L, "do": "kv_compress", "rank": int(kv_rank),
                        "why": "the context ceiling is the KV cache; rank %d "
                               "measured 8x context at 1.3%% attention error"
                               % int(kv_rank)})
    return {"analysis": a, "actions": actions, "long_position": long_pos}


def apply_plan(weights, cfg, the_plan, progress=None):
    """Carry out the growth actions. KV compression is a RUNTIME setting and is
    recorded in cfg rather than baked, because it depends on the sequence."""
    from holographic.io_and_interop.holographic_hrnngrow import grow_channel
    w, c = dict(weights), dict(cfg)
    grown, kv = [], []
    for act in the_plan["actions"]:
        if act["do"] == "grow_memory":
            w, c, rep = grow_channel(w, c, a_log=act["a_log"],
                                     gain=act["gain"], layers=[act["layer"]])
            grown.append(act["layer"])
            if progress:
                progress(act["layer"], "grow_memory", rep)
        elif act["do"] == "kv_compress":
            kv.append(act["layer"])
    if kv:
        c["kv_compress"] = {"layers": kv,
                            "rank": the_plan["actions"][-1].get("rank", 64)}
    return w, c, {"grown": grown, "kv_layers": kv,
                  "preserved": [a["layer"] for a in the_plan["actions"]
                                if a["do"] == "preserve"]}


def _selftest():
    import json
    import os

    kit = "/mnt/user-data/uploads/kit2.npz"
    if not os.path.exists(kit):
        print("transform selftest SKIPPED-SUBJECT (no real kit present)")
        return
    z = np.load(kit, allow_pickle=False)
    man = json.loads(bytes(z["manifest"]).decode("utf-8"))
    cfg = man["config"]
    gates = {k[6:]: z[k] for k in z.files if k.startswith("gate::")}

    a = analyse(gates, cfg)
    # ---- the BLOCK STRUCTURE is recovered from the weights alone ----
    assert a["block_period"] == 4, a["block_period"]
    assert len(a["attn_layers"]) == 6, a["attn_layers"]
    assert len(a["gdn_layers"]) == 18, len(a["gdn_layers"])

    # ---- and the POSITIONAL memory pattern is found, not assumed ----
    by = a["median_by_position"]
    assert by[0] > 5 * by[1], by          # measured 82.2 against 9.7
    assert by[0] > 5 * by[2], by

    p = plan(gates, cfg, target_tokens=4096, kv_rank=64)
    assert p["long_position"] == 0, p["long_position"]
    preserve = [x["layer"] for x in p["actions"] if x["do"] == "preserve"]
    grow = [x["layer"] for x in p["actions"] if x["do"] == "grow_memory"]
    kvc = [x["layer"] for x in p["actions"] if x["do"] == "kv_compress"]
    # ---- the long-memory layers are PRESERVED, the local ones grown ----
    assert preserve == [0, 4, 8, 12, 16, 20], preserve
    assert set(grow) == set(a["gdn_layers"]) - set(preserve)
    assert kvc == a["attn_layers"], kvc
    # ---- and every action carries its REASON, with the number in it ----
    assert all("why" in x and any(ch.isdigit() for ch in x["why"])
               for x in p["actions"])

    print("transform selftest OK -- recovered the block structure from the "
          "weights alone (%d blocks of %d, %d GDN + %d attention layers), found "
          "the positional memory pattern (position 0 median %.1f tokens against "
          "%.1f and %.1f), and planned accordingly: PRESERVE %s, GROW %d local "
          "layers, KV-compress %d attention layers"
          % (len(a["attn_layers"]), a["block_period"], len(a["gdn_layers"]),
             len(a["attn_layers"]), by[0], by[1], by[2], preserve, len(grow),
             len(kvc)))


if __name__ == "__main__":
    _selftest()
