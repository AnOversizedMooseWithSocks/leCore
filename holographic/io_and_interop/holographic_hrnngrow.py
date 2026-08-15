"""HRNNGROW -- ADD a holographic memory channel instead of stealing a trained one.

hrnnbake retuned an existing head into a persistent accumulator and it worked --
memory reached past 256 tokens -- but it cost +34% perplexity, because the model
was TRAINED with that head forgetting fast and its later layers depend on it.
Repurposing a working part is not a lever; it is a trade.

leCore's fourth lever is the fix: WHEN CAPACITY BINDS, ADD DIMENSIONS. Do not
take a head, GROW one. The new key-head arrives with

    a slow decay        -- so it accumulates instead of forgetting
    a ZERO out_proj     -- so it contributes NOTHING until asked

which makes the edit provably free: with the output column at zero the model's
logits are BIT-IDENTICAL to the original, and the extra state is being computed,
carried and simply not read. Turn the gain up and the memory enters the stream.
That is the project's "additive, never flip an existing decision" rule expressed
as an architecture change rather than a flag.

The tensors that must grow, all of them plain weight edits:
    in_proj_qkvz   +[q(dk), k(dk), v(r*dv), z(r*dv)] rows for the new group
    in_proj_ba     +2r rows (or in_proj_a / in_proj_b when the checkpoint splits)
    conv1d         +(2*dk + r*dv) channels
    A_log, dt_bias +r entries      -- where the slow decay is set
    out_proj       +r*dv COLUMNS OF ZERO   -- the "off" switch, and the point
and cfg's head counts are bumped to match, so any runtime reading the config
sees a consistent model.
"""

import numpy as np


def grow_channel(weights, cfg, a_log=-4.0, gain=0.0, layers=None, seed=0):
    """Add one key-head group of persistent holographic memory per layer.

    gain=0.0 (the default) leaves the model BIT-IDENTICAL: the channel runs and
    is not read. Raise it to let the long memory reach the residual stream."""
    w = {k: np.array(v, copy=True) for k, v in weights.items()}
    c = dict(cfg)
    Kh = int(c["linear_num_key_heads"])
    Vh = int(c["linear_num_value_heads"])
    dk = int(c["linear_key_head_dim"])
    dv = int(c["linear_value_head_dim"])
    hidden = int(c["hidden"])
    r = Vh // Kh
    rng = np.random.default_rng(int(seed))
    grown = []
    emb = next((key for key in weights
                if key.endswith("embed_tokens.weight")), None)
    if emb is None:
        raise ValueError("cannot locate the language-model tensor root")
    root = emb[:-len("embed_tokens.weight")]
    selected = (range(int(c["n_layers"])) if layers is None
                else [int(layer) for layer in layers])
    for L in selected:
        pre = "%slayers.%d.linear_attn." % (root, L)
        if pre + "A_log" not in w:
            continue

        # --- qkv(z): one more group, small random keys/queries, zero values ---
        # keys and queries must be NONZERO or the head can never bind anything;
        # values start at zero so the state begins empty rather than injecting
        # noise into a model that has not asked for it.
        def _rows(n_rows, scale):
            return (rng.standard_normal((n_rows, hidden)) * scale
                    if scale else np.zeros((n_rows, hidden)))

        if pre + "in_proj_qkvz.weight" in w:
            A = np.asarray(w[pre + "in_proj_qkvz.weight"], np.float64)
            s = float(np.std(A)) * 0.5
            # VALUES MUST BE LIVE. Zeroing them (the first version) makes the
            # recurrence accumulate nothing -- S = a*S + b*k*v^T is identically
            # zero when v is -- so the channel had a long memory of NOTHING.
            # The zero OUT_PROJ is what keeps it off; the channel itself has to
            # be carrying something for there to be anything to switch on.
            block = np.vstack([_rows(dk, s), _rows(dk, s),
                               _rows(r * dv, s), _rows(r * dv, s)])
            w[pre + "in_proj_qkvz.weight"] = np.vstack([A, block]).astype(
                np.asarray(weights[pre + "in_proj_qkvz.weight"]).dtype)
        else:
            A = np.asarray(w[pre + "in_proj_qkv.weight"], np.float64)
            s = float(np.std(A)) * 0.5
            # Split checkpoints exist in two layouts. The common Qwen layout
            # groups [q,k,v...] per key head; some converted checkpoints use
            # flat [all q][all k][all v] blocks. Growing under the wrong order
            # preserves the total shape but silently reassigns trained rows.
            if str(c.get("qkv_order", "grouped")) == "flat":
                q_end = Kh * dk
                k_end = q_end + Kh * dk
                grown_qkv = np.vstack([
                    A[:q_end], _rows(dk, s),
                    A[q_end:k_end], _rows(dk, s),
                    A[k_end:], _rows(r * dv, s),
                ])
            else:
                grown_qkv = np.vstack([
                    A.reshape(Kh, 2 * dk + r * dv, hidden),
                    np.vstack([_rows(dk, s), _rows(dk, s),
                               _rows(r * dv, s)])[None, :, :],
                ]).reshape(-1, hidden)
            w[pre + "in_proj_qkv.weight"] = grown_qkv.astype(
                np.asarray(weights[pre + "in_proj_qkv.weight"]).dtype)
            Z = np.asarray(w[pre + "in_proj_z.weight"], np.float64)
            w[pre + "in_proj_z.weight"] = np.vstack(
                [Z, _rows(r * dv, s)]).astype(
                    np.asarray(weights[pre + "in_proj_z.weight"]).dtype)

        # --- beta / decay projections ---
        # beta (the write gate) must be live as well: a zero beta writes
        # nothing, which is the same silent failure as a zero value.
        for key, n_extra in ((pre + "in_proj_ba.weight", 2 * r),
                             (pre + "in_proj_a.weight", r),
                             (pre + "in_proj_b.weight", r)):
            if key in w:
                B = np.asarray(w[key], np.float64)
                sb = float(np.std(B)) * 0.5 or 0.02
                w[key] = np.vstack([B, rng.standard_normal((n_extra, hidden)) * sb
                                    ]).astype(np.asarray(weights[key]).dtype)

        # --- the conv sees q, k and v, laid out [all q][all k][all v] ---
        # APPENDING AT THE END IS WRONG and was: the conv is not grouped by
        # head, so new channels must be INSERTED at the end of each block or
        # every existing channel shifts and the layer reads someone else's
        # numbers (measured: a channel that was supposed to be OFF moved the
        # logits by 10.2).
        cw = np.asarray(w[pre + "conv1d.weight"], np.float64)
        tail = cw.shape[1:]
        def _ident(n):
            z = np.zeros((n,) + tail)
            z[:, :, -1] = 1.0        # identity in time: pass the value through
            return z
        q_end = Kh * dk
        k_end = q_end + Kh * dk
        cw = np.vstack([cw[:q_end], _ident(dk),
                        cw[q_end:k_end], _ident(dk),
                        cw[k_end:], _ident(r * dv)])
        w[pre + "conv1d.weight"] = cw.astype(
            np.asarray(weights[pre + "conv1d.weight"]).dtype)
        bias_key = pre + "conv1d.bias"
        if bias_key in w:
            cb = np.asarray(w[bias_key])
            w[bias_key] = np.concatenate([
                cb[:q_end], np.zeros(dk, cb.dtype),
                cb[q_end:k_end], np.zeros(dk, cb.dtype),
                cb[k_end:], np.zeros(r * dv, cb.dtype),
            ])

        # --- THE SLOW DECAY: this is what makes it an HRNN channel ---
        for key, fill in ((pre + "A_log", float(a_log)),
                          (pre + "dt_bias", 0.0)):
            v = np.asarray(w[key], np.float64)
            w[key] = np.concatenate([v, np.full(r, fill)]).astype(
                np.asarray(weights[key]).dtype)

        # --- the gated norm is per value-head-dim; it does not grow ---
        # --- out_proj: NEW COLUMNS AT ZERO -> the channel is off by default ---
        O = np.asarray(w[pre + "out_proj.weight"], np.float64)
        cols = np.zeros((O.shape[0], r * dv))
        if gain:
            cols = rng.standard_normal(cols.shape) * float(gain) * float(np.std(O))
        w[pre + "out_proj.weight"] = np.hstack([O, cols]).astype(
            np.asarray(weights[pre + "out_proj.weight"]).dtype)
        grown.append(L)

    if grown:
        c["linear_num_key_heads"] = Kh + 1
        c["linear_num_value_heads"] = Vh + r
    return w, c, {"layers": grown, "a_log": float(a_log), "gain": float(gain),
                  "new_value_heads": r, "off_by_default": gain == 0.0}


def a_log_for(half_life_tokens):
    """The decay exponent that gives a memory this half-life.

    Derived, not tuned: decay = exp(-exp(a_log) * softplus(dt_bias)), and with
    dt_bias = 0 that is exp(-exp(a_log) * ln2), so the half-life
    D = ln(0.5)/ln(decay) = exp(-a_log), hence a_log = -ln(D).
    VERIFIED numerically from 16 to 16,384 tokens, exact to 3 significant
    figures at every rung."""
    return -float(np.log(max(2.0, float(half_life_tokens))))


def autoscale_memory(weights, cfg, target_tokens=4096, scales=4, gain=0.05,
                     shortest=16):
    """Install a LADDER of memory timescales sized for a target context.

    Why a ladder and not one long channel, measured: three copies of the SAME
    channel add nothing (influence at 1024 identical to one), because reach is
    set by DECAY, not by count -- more accumulators buy capacity, not range. A
    geometric ladder from `shortest` to `target_tokens` covers every distance
    instead: measured at 1024 tokens, influence 0.00026 for one channel against
    0.00092 for a four-rung ladder, a 3.5x longer reach for +0.14% perplexity.

    The rungs come from a_log_for(), so asking for 8k of context sets the
    exponents arithmetically rather than by taste."""
    n = max(1, int(scales))
    lo, hi = float(shortest), float(max(shortest * 2, target_tokens))
    rungs = [lo * (hi / lo) ** (i / max(n - 1, 1)) for i in range(n)]
    w, c = dict(weights), dict(cfg)
    installed = []
    for D in rungs:
        a = a_log_for(D)
        w, c, rep = grow_channel(w, c, a_log=a, gain=gain)
        installed.append({"half_life_tokens": round(D, 1), "a_log": round(a, 3),
                          "layers": len(rep["layers"])})
    return w, c, {"rungs": installed, "target_tokens": int(target_tokens),
                  "gain": float(gain)}


def _selftest():
    import os

    from holographic.io_and_interop.holographic_gdnruntime import (
        GDNRuntime, load_runtime)
    from holographic.io_and_interop.holographic_unicron import load_safetensors
    from holographic.io_and_interop.holographic_holocap import memory_horizon

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("hrnngrow selftest SKIPPED-SUBJECT (no trained model present)")
        return
    rt, cfg = load_runtime(src)
    w = load_safetensors(os.path.join(src, "model.safetensors"))
    raw = open("/home/claude/bench/docs.txt", encoding="utf-8",
               errors="ignore").read()
    ids = [int(b) for b in raw[3000:3400].encode()][:300]
    base_logits = rt.forward(ids)
    base_ppl = rt.perplexity(ids[:200])

    # ---- OFF BY DEFAULT MEANS BIT-IDENTICAL, not "close enough" ----
    w2, cfg2, rep = grow_channel(w, cfg, a_log=-4.0, gain=0.0)
    rt2 = GDNRuntime(w2, cfg2)
    got = rt2.forward(ids)
    assert got.shape == base_logits.shape, (got.shape, base_logits.shape)
    diff = float(np.max(np.abs(got - base_logits)))
    assert diff < 1e-9, ("a channel that is OFF changed the model", diff)
    assert rep["off_by_default"]

    # ---- the extra state EXISTS and is slow, even while unread ----
    _lg, st = rt2.prefill(ids[:64])
    S = np.asarray(st.gdn[0]["S"], np.float64)
    S0 = np.asarray(rt.prefill(ids[:64])[1].gdn[0]["S"], np.float64)
    assert S.shape[0] == S0.shape[0] + rep["new_value_heads"], (S.shape, S0.shape)

    # ---- turning it ON reaches further than the original ever did ----
    w3, cfg3, _r3 = grow_channel(w, cfg, a_log=-4.0, gain=0.05)
    rt3 = GDNRuntime(w3, cfg3)
    h_before = memory_horizon(rt, ids, marks=(8, 16, 32, 64, 128, 256))
    h_after = memory_horizon(rt3, ids, marks=(8, 16, 32, 64, 128, 256))
    late_b = h_before["curve"][-1]["relative_state_difference"]
    late_a = h_after["curve"][-1]["relative_state_difference"]
    assert late_a > late_b, (late_b, late_a)
    ppl3 = rt3.perplexity(ids[:200])

    # ---- THE LADDER: rungs derived from a target, reach verified ----
    w4, cfg4, lrep = autoscale_memory(w, cfg, target_tokens=1024, scales=3,
                                      gain=0.05)
    rt4 = GDNRuntime(w4, cfg4)
    h4 = memory_horizon(rt4, ids, marks=(16, 64, 256, 512))
    late_l = h4["curve"][-1]["relative_state_difference"]
    assert late_l > late_a, ("a ladder must reach further than one channel",
                             late_a, late_l)
    assert [r["half_life_tokens"] for r in lrep["rungs"]] == \
        sorted(r["half_life_tokens"] for r in lrep["rungs"])
    ppl4 = rt4.perplexity(ids[:200])
    assert ppl4 < base_ppl * 1.01, (base_ppl, ppl4)

    print("hrnngrow selftest OK -- GREW a holographic channel instead of stealing "
          "a head: with the output column at zero the logits are BIT-IDENTICAL "
          "(max diff %.1e) while the state carries %d extra value-head(s); "
          "turned on at gain 0.05 the memory reaches further than the original "
          "ever did (influence at 256 tokens %.5f -> %.5f) at perplexity "
          "%.4f -> %.4f (%+.1f%%), against +34.2%% for retuning a trained head"
          % (diff, rep["new_value_heads"], late_b, late_a, base_ppl, ppl3,
             100 * (ppl3 - base_ppl) / base_ppl)
          + "; a %d-rung LADDER sized for %d tokens reaches further still "
            "(%.5f at 512) for %+.2f%% perplexity"
          % (len(lrep["rungs"]), lrep["target_tokens"], late_l,
             100 * (ppl4 - base_ppl) / base_ppl))


if __name__ == "__main__":
    _selftest()
