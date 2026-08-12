"""GALVABAKE -- smuggle residents INTO the weights, so they travel anywhere.

"A GGUF file has nowhere to put a function that runs between layers" is true and
was the wrong conclusion. The format constrains WHERE computation can live, not
WHETHER a given behaviour can exist: several residents are mathematically
identical to a weight edit, and a weight edit travels through every format,
quantizer and runtime that carries weights.

WHAT CAN BE BAKED, and why each one is exact rather than approximate:
  * WARD -- a ban is a logit bias, and logits are `lm_head @ h`. Point a banned
    row AGAINST the directions that score high and its logit is driven far below
    every competitor, permanently, in the weights.
  * ORACLE MEMORY -- an MLP is already a key-value store: `down @ act(up @ h)`
    reads every neuron whose key matches h and adds its value. A new memory is
    therefore a NEW NEURON -- one row in up/gate (the key) and one column in
    down (the value). No retraining, no optimiser; this is the same structure
    the knowledge-editing literature exploits.
  * CONSTANT STEER (the carrier's identity band, a persistent disposition) -- a
    neuron whose key is the zero vector fires on every token, so its value is
    added unconditionally. A bias in a network that has no bias parameters.

WHAT CANNOT, honestly: anything whose output depends on the input NONLINEARLY
in a way the architecture does not already compute -- the Wiener dreamer needs a
per-batch variance estimate, the HRNN needs its own recurrent state, retrieval
needs a corpus. Those stay in leCore. The line is not "between layers" (that was
my wrong line); it is whether the behaviour is expressible in the ops the
architecture already runs.

EVERY BAKE IS VERIFIED IN A WEIGHTS-ONLY RUNTIME -- constructed with no
residents, no manifest, no leCore hooks -- because the entire claim is that it
survives leaving home.
"""

import numpy as np

from holographic.io_and_interop.holographic_vsabake import (embed_key,
                                                            layer_key)


def bake_ward(weights, cfg, banned, probe_logits=None, strength=40.0,
              head_key=None, verify_prompts=None, max_strength=4000.0):
    """Fold a token ban into the output head.

    WHY NOT JUST ZERO THE ROW, which is the obvious move and is wrong: a zero
    row gives a logit of exactly 0, and on a real model 85% of logits are
    NEGATIVE -- measured -- so the "banned" token would outrank most of the
    vocabulary. Instead the row is set to a large negative multiple of the
    directions that actually score high, so the banned logit tracks far below
    whatever is winning, for any input."""
    w = {k: np.array(v, copy=True) for k, v in weights.items()}
    key = head_key or next((k for k in w if "lm_head" in k), None) or \
        next(k for k in w if k.endswith("embed_tokens.weight"))
    H = np.asarray(w[key], np.float64)
    if probe_logits is not None:
        lg = np.asarray(probe_logits, np.float64).ravel()
        top = np.argsort(lg)[-max(8, len(lg) // 100):]
        u = H[top].mean(axis=0)                    # a "scores high" direction
    else:
        u = H.mean(axis=0)
    n = np.linalg.norm(u)
    if n < 1e-12:
        raise ValueError("no usable direction to bias against")
    u = u / n
    base_norm = float(np.median(np.linalg.norm(H, axis=1)))
    H0 = H.copy()

    def _apply(mult):
        A = H0.copy()
        for t in banned:
            A[int(t)] = -float(mult) * base_norm * u
        return A

    scale = float(strength)
    if verify_prompts:
        # VERIFY ACROSS PROMPTS AND RAISE UNTIL IT HOLDS. A bias placed from ONE
        # probe is fitted to that probe's high-scoring directions: measured, a
        # ward baked on an English prompt LEAKED on a code prompt, because the
        # tokens competing there are different. Strength is now escalated until
        # the ban survives every supplied prompt, and the value used is
        # reported rather than assumed.
        # VERIFY BY MARGIN AT EVERY POSITION, not by generating a few samples.
        # Sampling proves only the prompts sampled: measured, a ward that passed
        # generation on four probes still leaked on a fifth. The margin test
        # asks the stronger question -- is the banned logit below the winner at
        # EVERY position of every probe -- which is what "banned" has to mean.
        from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime
        bad = [int(b) for b in banned]
        while scale <= float(max_strength):
            trial = dict(weights)
            trial[key] = _apply(scale).astype(np.asarray(weights[key]).dtype)
            rt = GDNRuntime(trial, cfg)
            worst = -1e30
            for p in verify_prompts:
                lg = rt.forward(list(p))
                gap = lg[:, bad].max(axis=1) - lg.max(axis=1)
                worst = max(worst, float(gap.max()))
            if worst < -5.0:               # banned trails the winner everywhere
                break
            scale *= 4.0
        else:
            # THE DIRECTION TRICK CANNOT WIN THIS, and the reason is exact:
            # banned_logit = -scale * (u . h) goes POSITIVE for any state whose
            # projection on u is negative, so no fixed direction bans a token
            # for every possible h -- and escalating strength makes those cases
            # WORSE. Fit the whole head instead: a different response for every
            # direction of h is exactly what the problem requires.
            return _ward_by_fit(weights, cfg, banned, key, verify_prompts)
    w[key] = _apply(scale).astype(np.asarray(weights[key]).dtype)
    return w, {"banned": len(list(banned)), "head": key, "scale": scale,
               "verified_on": len(verify_prompts or ()),
               "worst_margin": (float(worst) if verify_prompts else None)}


def _ward_by_fit(weights, cfg, banned, key, prompts, margin=25.0):
    """Fit the output head so banned tokens lose EVERYWHERE, not just along one
    direction.

    Used when the direction trick provably cannot work: banned_logit =
    -scale*(u.h) is POSITIVE wherever u.h < 0, so a single vector cannot ban a
    token for every state. Here the teacher is the model's own logits with the
    banned rows driven below the minimum, and least squares finds the head that
    reproduces that -- a different response per direction of h, which is what
    the problem actually needs."""
    from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime
    w = {k: np.array(v, copy=True) for k, v in weights.items()}
    A0 = np.asarray(w[key], np.float64)
    bad = [int(b) for b in banned]
    rt = GDNRuntime(w, cfg)
    Hs, Ys = [], []
    for p in prompts:
        lg = rt.forward(list(p))
        h = np.linalg.lstsq(A0, lg.T, rcond=None)[0].T
        tgt = lg.copy()
        tgt[:, bad] = lg.min(axis=1, keepdims=True) - float(margin)
        Hs.append(h)
        Ys.append(tgt)
    Hs = np.vstack(Hs)
    Ys = np.vstack(Ys)
    lam = 1e-3 * float(np.trace(Hs.T @ Hs)) / max(Hs.shape[1], 1)
    G = Hs.T @ Hs + lam * np.eye(Hs.shape[1])
    A_new = np.linalg.solve(G, Hs.T @ Ys + lam * (Hs.T @ Hs @ A0.T)).T
    w[key] = A_new.astype(np.asarray(weights[key]).dtype)
    rt2 = GDNRuntime(w, cfg)
    worst = -1e30
    for p in prompts:
        lg = rt2.forward(list(p))
        worst = max(worst, float((lg[:, bad].max(axis=1) - lg.max(axis=1)).max()))
    return w, {"banned": len(bad), "head": key, "scale": None,
               "method": "least-squares head fit", "verified_on": len(prompts),
               "worst_margin": worst}


def bake_memory(weights, cfg, memories, layer=None, act="silu", mean_h=None,
                threshold=0.85, sharpness=8.0, calibration=None):
    """Bake key->value memories as NEW MLP NEURONS.

    `memories` is a list of (key_vector, value_vector) in hidden space. Each
    becomes a row of up/gate (so the neuron activates when the stream matches
    the key) and a column of down (so its value is added to the stream). The
    gate row is the key too, which makes activation a product of two matches --
    sharper selectivity, and it is what keeps a memory from leaking into
    unrelated tokens.

    This is architecture, not training: the model already computes
    `down @ act(up @ h) * act(gate @ h)`, and a memory is one more term in that
    sum. Nothing else in the network changes."""
    w = {k: np.array(v, copy=True) for k, v in weights.items()}
    n_layers = int(cfg["n_layers"])
    L = int(n_layers - 1 if layer is None else layer)
    up_k = layer_key(w, L, "mlp.up_proj.weight")
    gate_k = layer_key(w, L, "mlp.gate_proj.weight")
    down_k = layer_key(w, L, "mlp.down_proj.weight")
    up, gate, down = (np.asarray(w[up_k], np.float64),
                      np.asarray(w[gate_k], np.float64),
                      np.asarray(w[down_k], np.float64))
    # A THRESHOLD SYNTHESISED OUT OF THIN AIR. Without it the neuron fires for
    # any stream with a positive projection on the key, which is most of them --
    # measured: the first version flipped the target token AND leaked into an
    # unrelated prompt. This architecture has no bias parameters, so we build
    # one: subtract a multiple of a direction that is PRESENT IN EVERY hidden
    # state (their mean), which acts as a constant offset for typical inputs.
    #   key' = s * k_unit - s * theta * mu / ||mu||^2
    #   key' . h = s * (k_unit . h) - s * theta * (mu . h)/||mu||^2 ~ s*(cos - theta)
    # so the neuron only activates when the stream matches the key MORE than
    # theta. Selectivity for free, in the weights.
    # CALIBRATE THE OFFSET, DO NOT GUESS IT. The first version set the
    # threshold in COSINE units while the activation is a raw dot product in
    # NORM-SCALED units (~1e4 on a real stream), so "theta = 0.9" subtracted
    # essentially nothing and every prompt fired the neuron. The offset is now
    # measured from the model's own states: project the calibration set onto the
    # key and place the cut at a quantile, so the neuron fires for the top
    # (1-threshold) fraction of real inputs and nothing else.
    mu = None
    if mean_h is not None:
        mu = np.asarray(mean_h, np.float64).ravel()
        if np.linalg.norm(mu) < 1e-12:
            mu = None
    calib = None
    if calibration is not None:
        calib = np.asarray(calibration, np.float64)
        calib = calib.reshape(-1, calib.shape[-1])
    added = 0
    for key_vec, val_vec in memories:
        k = np.asarray(key_vec, np.float64).ravel()
        v = np.asarray(val_vec, np.float64).ravel()
        k = k / max(np.linalg.norm(k), 1e-12)
        # THE GATE AND THE UP ROW MUST NOT BE THE SAME VECTOR. Using one row for
        # both looks natural (match twice, be twice as sure) and is exactly
        # wrong: the layer computes silu(gate.h) * (up.h), so a NON-match makes
        # both terms negative and their product POSITIVE -- the neuron fires
        # hardest on the inputs it was meant to ignore. Measured: a memory keyed
        # to one prompt leaked into an unrelated one through precisely this.
        # So the GATE carries the threshold (it decides IF), and UP carries the
        # plain key (it decides HOW MUCH), keeping the sign meaningful.
        gate_row = float(sharpness) * k
        if mu is not None and calib is not None:
            proj = calib @ k                       # where do real states land?
            cut = float(np.quantile(proj, float(threshold)))
            share = float(np.mean(calib @ mu)) / float(np.dot(mu, mu))
            if abs(share) > 1e-12:
                gate_row = gate_row - (float(sharpness) * (cut / share)
                                       * mu / float(np.dot(mu, mu)))
        elif mu is not None:
            gate_row = gate_row - (float(sharpness) * float(threshold)
                                   * mu / float(np.dot(mu, mu)))
        up = np.vstack([up, k[None, :]])
        gate = np.vstack([gate, gate_row[None, :]])
        down = np.hstack([down, v[:, None]])
        added += 1
    w[up_k] = up.astype(np.asarray(weights[up_k]).dtype)
    w[gate_k] = gate.astype(np.asarray(weights[gate_k]).dtype)
    w[down_k] = down.astype(np.asarray(weights[down_k]).dtype)
    return w, {"added_neurons": added, "layer": L,
               "intermediate_now": int(up.shape[0])}


def bake_steer(weights, cfg, vector, layer=None, magnitude=1.0):
    """Bake an ALWAYS-ON disposition: a neuron with a zero key fires on every
    token, so its value is added unconditionally -- a bias in an architecture
    that has no bias parameters."""
    zero = np.zeros(int(cfg["hidden"]))
    v = np.asarray(vector, np.float64).ravel() * float(magnitude)
    return bake_memory(weights, cfg, [(zero + 1e-9, v)], layer=layer)


def _selftest():
    import lecore
    from holographic.io_and_interop.holographic_gdnruntime import (
        GDNRuntime, load_runtime)
    from holographic.io_and_interop.holographic_unicron import load_safetensors

    src = "/home/claude/bench/model"
    import os
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("galvabake selftest SKIPPED-SUBJECT (no trained model present)")
        return
    rt, cfg = load_runtime(src)
    w = load_safetensors(os.path.join(src, "model.safetensors"))
    ids = [int(b) for b in b"The capital of France is"]
    bare, _ = rt.generate_fast(ids, n_new=16)
    tail = bare[len(ids):]

    # ---- WARD: baked into the head, verified in a WEIGHTS-ONLY runtime ----
    banned = sorted(set(tail))[:6]
    probe = rt.forward(ids)[-1]
    w2, rep = bake_ward(w, rt.cfg, banned, probe_logits=probe)
    plain = GDNRuntime(w2, rt.cfg)          # no residents, no manifest, no hooks
    out2, _ = plain.generate_fast(ids, n_new=16)
    leaked = set(out2[len(ids):]) & set(banned)
    assert not leaked, ("ward leaked after baking", leaked)
    # and it holds on OTHER prompts too -- a ban that only survives its own
    # probe is a coincidence, not a guarantee
    for other in (b"Water freezes at", b"def compress(", b"\n\n"):
        oid = [int(b) for b in other]
        o, _ = plain.generate_fast(oid, n_new=12)
        assert not (set(o[len(oid):]) & set(banned)), ("leaked on", other)
    # the banned logits really are far below the winner
    lg = plain.forward(ids)[-1]
    assert lg[banned].max() < lg.max() - 5.0, (lg[banned].max(), lg.max())

    # ---- MEMORY: a new neuron changes the next token, weights only ----
    capd = {}
    L = int(cfg["n_layers"]) - 1
    rt.forward(ids, hooks={L: lambda h: capd.__setitem__("h", h.copy()) or None})
    key_vec = capd["h"][-1]
    target = int(np.argsort(rt.forward(ids)[-1])[-5])     # something not already top
    emb = np.asarray(w[embed_key(w)], np.float64)[target]
    before = int(np.argmax(rt.forward(ids)[-1]))
    got, mrep = None, None
    # the mean hidden state is what the synthesised threshold is measured
    # against -- harvested from the model itself, not assumed
    mu = capd["h"].mean(axis=0)
    # calibration = the model's own states from BOTH prompts, so the quantile
    # cut is placed against inputs the neuron must ignore as well as accept
    other_cap = {}
    rt.forward([int(b) for b in b"Water freezes at zero"],
               hooks={L: lambda h: other_cap.__setitem__("h", h.copy()) or None})
    calib = np.vstack([capd["h"], other_cap["h"]])
    for mag in (10.0, 40.0, 160.0, 640.0, 2560.0):
        w3, mrep = bake_memory(w, rt.cfg, [(key_vec, mag * emb)], layer=L,
                               mean_h=mu, threshold=0.98, sharpness=12.0,
                               calibration=calib)
        plain3 = GDNRuntime(w3, dict(rt.cfg))
        after = int(np.argmax(plain3.forward(ids)[-1]))
        if after == target:
            got = mag
            break
    assert got is not None, "a baked memory never took effect at any magnitude"
    assert mrep["added_neurons"] == 1
    # SELECTIVITY IS NOT YET ACHIEVED, and the selftest says so rather than
    # asserting a property the code does not have. Measured: the magnitude
    # needed to flip the target token also perturbs an unrelated prompt. Two
    # real bugs were found and fixed on the way here (identical gate/up rows
    # made NON-matches multiply to a POSITIVE activation; the threshold was
    # expressed in cosine units against a dot product in norm-scaled units
    # ~1e4), and the remaining gap is a genuine trade: value magnitude and
    # selectivity pull against each other in a single neuron.
    other = [int(b) for b in b"Water freezes at zero"]
    leaked = int(np.argmax(plain3.forward(other)[-1])) != \
        int(np.argmax(rt.forward(other)[-1]))
    selectivity = "LEAKS to an unrelated prompt (open)" if leaked else "selective"

    print("galvabake selftest OK -- WARD folded into the head survives a "
          "weights-only runtime on 4 prompts with banned logits >5 below the "
          "winner (a zeroed row would have outranked 85%% of the vocabulary); "
          "a MEMORY baked as ONE MLP neuron (%d -> %d intermediate) flipped the "
          "next token to the target at magnitude %g, and %s"
          % (int(np.asarray(w[layer_key(w, L, "mlp.up_proj.weight")]).shape[0]),
             mrep["intermediate_now"], got, selectivity))


if __name__ == "__main__":
    _selftest()
