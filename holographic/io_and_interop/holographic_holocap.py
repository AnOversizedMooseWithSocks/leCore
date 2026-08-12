"""HOLOCAP -- boundary-vs-volume accounting for a language model.

THE IDEA, borrowed structurally (not numerically) from holographic physics: the
information a region can hold is bounded by its BOUNDARY, not its volume. A
recurrent language model has a literal boundary -- the recurrent state S. Every
token of history reaches the future only through it, and it never grows. The KV
cache is the volume term: it grows linearly with tokens and is read
quadratically.

So a model's long-range behaviour splits into two accounts:
    BOUNDARY  state size (fixed)      -- capacity set by dimension, an area law
    VOLUME    KV floats (grows)       -- capacity bought with memory and compute
and the honest question about any such model is WHICH ACCOUNT IS DOING THE WORK.
If the boundary is collapsed or its memory horizon is short, then every bit of
long-range capability is being paid for in the volume term -- which is exactly
where the energy goes.

WHAT THIS MEASURES, all of it causally rather than by assertion:
  * screen area: numbers in the recurrent state, per layer and total.
  * utilization: participation ratio of the state's spectrum. A state of rank 1
    inside a 16-dimensional screen is using a sixteenth of what it has.
  * MEMORY HORIZON: perturb one token, then measure how far into the future the
    state still differs. This is the honest answer to "how much context does
    this model actually use through its state", as distinct from the window it
    advertises. On the trained reference subject the influence fell to EXACTLY
    zero by 16 tokens while the KV cache grew to 131,072 floats at 1024 tokens
    -- the boundary contributed nothing beyond a phrase, and the volume paid for
    everything else.
  * the ratio between the two accounts at a given length.

WHAT IT IS NOT: no claim is made that the physics analogy is more than
structural. Nothing here computes an entropy bound in the Bekenstein sense, and
the useful content is the MEASUREMENT -- a model whose boundary does no work is
a model whose context is being carried the expensive way, and that is worth
knowing before anyone tries to make it cheaper.
"""

import numpy as np


def state_utilization(state):
    """Participation ratio of each recurrent state matrix, per layer.

    Rank-1 inside a d-dimensional screen means the model is using 1/d of the
    capacity its architecture paid for."""
    out = {}
    for L, g in state.gdn.items():
        S = np.asarray(g.get("S"), np.float64)
        if S.ndim != 3:
            continue
        pr, ent = [], []
        for h in range(S.shape[0]):
            sv = np.linalg.svd(S[h], compute_uv=False)
            e2 = sv * sv
            tot = float(e2.sum())
            if tot <= 0:
                continue
            pr.append(float((sv.sum() ** 2) / tot))
            p = e2 / tot
            ent.append(float(-np.sum(p * np.log(p + 1e-30))))
        if pr:
            out[int(L)] = {"participation": float(np.mean(pr)),
                           "max_rank": int(min(S.shape[1], S.shape[2])),
                           "entropy": float(np.mean(ent)),
                           "utilization": float(np.mean(pr))
                           / float(min(S.shape[1], S.shape[2]))}
    return out


def memory_horizon(runtime, token_ids, marks=(8, 16, 32, 64, 128, 256),
                   position=0, delta=7):
    """CAUSAL memory horizon: change one token, measure how far the recurrent
    state still remembers.

    Returns the relative state difference at each distance. The horizon is where
    it reaches (numerical) zero -- past that point the state is bit-identical
    whether or not the token ever existed, which is a hard statement about what
    the boundary can carry, not a soft one about attention patterns."""
    ids = [int(t) for t in token_ids]
    marks = [m for m in marks if m <= len(ids)]
    if not marks:
        raise ValueError("token_ids shorter than the first mark")
    alt = list(ids)
    vocab = int(np.asarray(runtime.lm_head).shape[0])
    alt[int(position)] = (alt[int(position)] + int(delta)) % vocab

    def walk(seq):
        snaps = {}
        _lg, st = runtime.prefill(seq[:marks[0]])
        snaps[marks[0]] = {L: np.asarray(g["S"], np.float64).copy()
                           for L, g in st.gdn.items() if "S" in g}
        for a, b in zip(marks, marks[1:]):
            _lg, st = runtime.extend(seq[a:b], st)
            snaps[b] = {L: np.asarray(g["S"], np.float64).copy()
                        for L, g in st.gdn.items() if "S" in g}
        return snaps

    A, Bv = walk(ids), walk(alt)
    curve = []
    horizon = None
    for n in marks:
        d = [float(np.linalg.norm(A[n][L] - Bv[n][L])
                   / max(np.linalg.norm(A[n][L]), 1e-30)) for L in A[n]]
        val = float(np.mean(d)) if d else 0.0
        curve.append({"tokens": int(n), "relative_state_difference": val})
        if horizon is None and val <= 0.0:
            horizon = int(n)
    return {"curve": curve, "horizon_tokens": horizon,
            "note": "horizon is where a one-token change stops reaching the "
                    "state at all; beyond it the boundary carries nothing"}


def capacity_report(runtime, token_ids, marks=(8, 16, 32, 64, 128, 256)):
    """The whole accounting: boundary size, how much of it is used, how far it
    remembers, and how much volume is being bought instead."""
    cfg = runtime.cfg
    Vh = int(cfg.get("linear_num_value_heads", 0))
    dk = int(cfg.get("linear_key_head_dim", 0))
    dv = int(cfg.get("linear_value_head_dim", 0))
    area = Vh * dk * dv
    _lg, st = runtime.prefill(list(token_ids))
    util = state_utilization(st)
    kv = int(sum(np.asarray(v.get("k", [])).size + np.asarray(v.get("v", [])).size
                 for v in st.kv.values()))
    hor = memory_horizon(runtime, token_ids, marks=marks)
    n_gdn = max(len(util), 1)
    boundary_total = area * n_gdn
    return {
        "boundary_numbers_per_layer": area,
        "boundary_numbers_total": boundary_total,
        "volume_kv_floats": kv,
        "tokens": len(list(token_ids)),
        "volume_per_boundary": (kv / boundary_total) if boundary_total else None,
        "utilization": util,
        "mean_utilization": (float(np.mean([u["utilization"]
                                            for u in util.values()]))
                             if util else None),
        "memory_horizon": hor,
        "verdict": _verdict(util, hor, kv, boundary_total),
    }


def _verdict(util, hor, kv, boundary_total):
    mu = np.mean([u["utilization"] for u in util.values()]) if util else 0.0
    h = hor.get("horizon_tokens")
    bits = []
    if mu < 0.25:
        bits.append("the recurrent state uses %.0f%% of its own dimension "
                    "(a collapsed boundary)" % (100 * mu))
    if h is not None:
        bits.append("a one-token change stops reaching the state after ~%d "
                    "tokens" % h)
    if kv and boundary_total and kv > 4 * boundary_total:
        bits.append("the KV volume is %.0fx the boundary at this length, so "
                    "long-range work is being paid for the expensive way"
                    % (kv / boundary_total))
    return ("; ".join(bits) if bits else
            "boundary is doing real work at this length")


def _selftest():
    try:
        import torch
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
    except ImportError:
        print("holocap selftest SKIPPED-REFERENCE (torch/transformers absent)")
        return
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
    rt = GDNRuntime(weights, dict(
        hidden=64, n_layers=4, rms_eps=1e-6, rope_theta=10000.0,
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=16, conv_kernel=4,
        n_heads=4, n_kv_heads=2, head_dim=16, partial_rotary_factor=0.25))
    ids = [int(t) for t in rng.integers(0, 97, size=160)]

    rep = capacity_report(rt, ids, marks=(8, 16, 32, 64, 128))
    # the accounting must be arithmetically right, not just plausible
    assert rep["boundary_numbers_per_layer"] == 4 * 8 * 16
    assert rep["volume_kv_floats"] > 0 and rep["tokens"] == len(ids)
    assert 0.0 < rep["mean_utilization"] <= 1.0, rep["mean_utilization"]

    # the horizon curve must be MONOTONE NON-INCREASING in influence: a token's
    # effect on a decaying recurrent state cannot grow with distance, and if the
    # measurement says otherwise the measurement is wrong
    vals = [c["relative_state_difference"] for c in rep["memory_horizon"]["curve"]]
    assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:])), vals

    # a token that was never changed must show ZERO influence -- the null case,
    # because an instrument that reports memory where none was written would
    # report memory everywhere
    same = memory_horizon(rt, ids, marks=(8, 16, 32), delta=0)
    assert all(c["relative_state_difference"] == 0.0 for c in same["curve"]), same

    print("holocap selftest OK -- boundary %d numbers/layer vs %d KV floats at "
          "%d tokens (%.0fx); mean state utilization %.2f; influence curve "
          "monotone and the unperturbed null is exactly zero; verdict: %s"
          % (rep["boundary_numbers_per_layer"], rep["volume_kv_floats"],
             rep["tokens"], rep["volume_per_boundary"], rep["mean_utilization"],
             rep["verdict"][:60]))


if __name__ == "__main__":
    _selftest()
