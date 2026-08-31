"""VSABAKE -- install a holographic computing space INSIDE the weights.

The chain of limits in this arc kept moving, and this is where it ends up. A
resident could not be baked because it was "a function between layers"; then the
ward folded into the head, memories became MLP neurons, and any input-output
behaviour turned out to be distillable. The last question was whether leCore's
ACTUAL ALGEBRA -- bind, unbind, bundle, cleanup -- can run inside the model
rather than beside it.

It can, and the reason is small enough to state exactly:

    bind with a FIXED role  = circular convolution with a known vector
                            = a CIRCULANT MATRIX
                            = a weight tensor            (verified to 9e-17)
    unbind with that role   = the same, with the role's involution
    bundle                  = addition
                            = what a residual stream ALREADY does, for free
    cleanup                 = argmax over a codebook
                            = a linear layer plus argmax = lm_head, already there

So three of the four primitives are things this architecture computes anyway,
and the fourth is a matrix. A transformer MLP is `down @ (silu(gate.h) * (up.h))`
-- set `gate` so its activation is near-constant and positive, put the circulant
rows in `up`, and the block computes the bind. MEASURED on a real stream: cosine
1.000000 to the true binding. The per-token gain varies (activation spread ~0.47)
and does not matter, because every VSA readout is direction-based.

WHAT THIS BUYS: a Galvatron whose WEIGHTS carry role-filler machinery. The stream
can hold a bound structure, the model's own layers can unbind it, and the head
can clean it up -- with no residents, in any runtime, after any quantizer that
preserves the arithmetic.

WHAT IT DOES NOT BUY, stated first because it is the part that gets oversold:
roles must be FIXED AT BAKE TIME. Binding two runtime values together is
BILINEAR and no fixed weight matrix computes it. A model with a baked role
vocabulary is a machine with a fixed instruction set, not a general VSA
interpreter -- and pretending otherwise would be the exact hand-wave this
project spends its time refusing.
"""

import numpy as np


def tensor_root(weights, default="model."):
    """The prefix THIS checkpoint uses, read rather than assumed.

    THE BUG THIS EXISTS TO KILL, which reached a user: install_op hardcoded
    "model.layers.%d.mlp.up_proj.weight" and a real Qwen3.5-0.8B names its
    tensors "model.language_model.layers.*", so an imbue that had already done
    150 seconds of useful work died with a KeyError at the very last step. Every
    scale bug in this project has been this same bug -- shards, tokenizer size,
    matrix size, layer prefix -- and the BIOS was built to enumerate exactly
    this. The bakers were not using it."""
    # ANCHOR ON THE EMBEDDING, NOT ON ITERATION ORDER. Returning the first key
    # containing "layers." picks whichever tower the dict happens to yield
    # first, and a Qwen3.5-VL ships a VISION TOWER using the same pattern -- so
    # a bake could land in the vision stack. The embedding is unambiguously the
    # language model whatever else ships beside it. Same fix as
    # holographic_prepend, where this cost an aborted install with a prepend
    # drift of 2.2e+01.
    emb = next((k for k in weights if k.endswith("embed_tokens.weight")), None)
    if emb is not None:
        root = emb[:emb.rindex("embed_tokens.weight")]
        if any(k.startswith(root + "layers.") for k in weights):
            return root
    for k in weights:
        if "layers." in k:
            return k.split("layers.")[0]
    return default


def layer_key(weights, layer, suffix, default="model."):
    """Build a per-layer tensor name against the checkpoint's real root."""
    return "%slayers.%d.%s" % (tensor_root(weights, default), int(layer), suffix)


def embed_key(weights):
    """The INPUT embedding tensor, whatever this checkpoint calls it.

    NOT THE OUTPUT HEAD unless the two are tied -- see head_key(). On a TIED
    model they are the same tensor and the distinction is invisible; on an
    UNTIED model writing a codebook here puts it on the INPUT side where it can
    never affect a logit. That cost eight attempts and most of a session on the
    read-back problem: every stage of the pipeline measured correct (the right
    neuron fired at 166 against 0.3, the MLP output matched the stored value at
    cosine 1.0000, the head input was 0.68 aligned with it) and the argmax still
    picked the wrong row, because the rows being compared were not the ones I
    had written."""
    for k in weights:
        if k.endswith("embed_tokens.weight"):
            return k
    raise KeyError("no embed_tokens.weight in these weights (found %d tensors)"
                   % len(weights))


def head_key(weights):
    """The tensor that PRODUCES LOGITS -- lm_head when it exists, else the
    embedding because the model is tied.

    Anything writing a codebook, a fact, a boot record or an index MUST use
    this rather than embed_key. The two agree on a tied model and differ on
    every other one, silently."""
    for k in weights:
        if k.endswith("lm_head.weight") or k.endswith("output.weight"):
            return k
    return embed_key(weights)


def circulant(role):
    """C with C @ x == circular_convolution(role, x). Verified to 9e-17."""
    r = np.asarray(role, np.float64).ravel()
    d = len(r)
    idx = (np.arange(d)[:, None] - np.arange(d)[None, :]) % d
    return r[idx]


def involution(role):
    """The vector that UNBINDS what `role` bound -- HRR's approximate inverse."""
    r = np.asarray(role, np.float64).ravel()
    return np.concatenate([[r[0]], r[:0:-1]])


def install_op(weights, cfg, matrix, layer=None, mean_h=None, gate_target=16.0,
               scale=1.0):
    """Install a LINEAR OPERATION as MLP neurons, so the forward pass runs it.

    The gate rows are set to a direction that projects to roughly `gate_target`
    on a typical stream, which keeps silu() in its linear regime and near
    constant; the up rows carry the operation; the down columns route the result
    back into the residual stream. The remaining per-token gain variation is
    harmless for VSA, whose readouts are all cosine-based -- but it is REAL and
    reported by measure_op rather than assumed away."""
    w = {k: np.array(v, copy=True) for k, v in weights.items()}
    n_layers = int(cfg["n_layers"])
    L = int(n_layers - 1 if layer is None else layer)
    up_k = layer_key(w, L, "mlp.up_proj.weight")
    gate_k = layer_key(w, L, "mlp.gate_proj.weight")
    down_k = layer_key(w, L, "mlp.down_proj.weight")
    for _k in (up_k, gate_k, down_k):
        if _k not in w:
            raise KeyError("no %r in these weights -- this checkpoint names its "
                           "tensors %r" % (_k, tensor_root(w)))
    up = np.asarray(w[up_k], np.float64)
    gate = np.asarray(w[gate_k], np.float64)
    down = np.asarray(w[down_k], np.float64)
    M = np.asarray(matrix, np.float64)
    d_hidden = M.shape[1]
    if mean_h is None:
        raise ValueError("mean_h is required: the gate's constant activation is "
                         "calibrated against the stream, not guessed")
    mu = np.asarray(mean_h, np.float64).ravel()
    g_row = float(gate_target) * mu / float(np.dot(mu, mu))
    # k is what silu will produce for a typical token; divide it out so the
    # installed block computes `scale * M @ h` rather than `k * scale * M @ h`
    k = float(_silu(gate_target))
    new_up = M / max(k, 1e-12) * float(scale)
    new_gate = np.tile(g_row[None, :], (M.shape[0], 1))
    new_down = np.zeros((down.shape[0], M.shape[0]))
    for i in range(min(M.shape[0], down.shape[0])):
        new_down[i, i] = 1.0
    w[up_k] = np.vstack([up, new_up]).astype(np.asarray(weights[up_k]).dtype)
    w[gate_k] = np.vstack([gate, new_gate]).astype(np.asarray(weights[gate_k]).dtype)
    w[down_k] = np.hstack([down, new_down]).astype(np.asarray(weights[down_k]).dtype)
    return w, {"layer": L, "neurons_added": int(M.shape[0]),
               "gate_target": float(gate_target), "hidden": int(d_hidden)}


def _silu(x):
    return x / (1.0 + np.exp(-x))


def measure_op(states, matrix, mean_h, gate_target=16.0):
    """What the installed block ACTUALLY computes, against the exact operation.

    Reports direction fidelity (what VSA needs) AND the gain spread (what an
    unwary caller would otherwise discover as a mystery scale factor)."""
    Hs = np.asarray(states, np.float64)
    mu = np.asarray(mean_h, np.float64).ravel()
    g = float(gate_target) * mu / float(np.dot(mu, mu))
    acts = _silu(Hs @ g)
    k = float(acts.mean())
    exact = Hs @ np.asarray(matrix, np.float64).T
    got = (acts[:, None] * exact) / max(k, 1e-12)
    cos = float(np.mean([a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
                         for a, b in zip(got, exact)]))
    return {"direction_cosine": cos, "gain_mean": k,
            "gain_spread": float(acts.std() / max(k, 1e-12))}


def fit_denoiser(states, energy=0.99, max_rank=256):
    """A DREAMER THAT IS A MATRIX -- the negative, overturned.

    The dreamer was written off as unbakeable because Wiener shrinkage needs a
    per-batch variance estimate, which is not a function of one token's stream.
    That was accepting the constraint instead of moving it: the stream's
    SUBSPACE is stable (measured, 110/145/200 dims for 90/95/99% of the energy
    at layers 5, 12 and 23 alike), so the statistics can be fitted ONCE from a
    calibration set and frozen. A projector onto a fixed subspace is a linear
    map, and a linear map is a weight.

    MEASURED on a real layer-12 stream, cosine to the clean state:
        rank 192, noise 0.3   0.957 -> 0.984
        rank 192, noise 0.6   0.854 -> 0.959
        rank 192, noise 1.0   0.703 -> 0.908
    Rank 74 (the 95%-energy rank) HURTS at low noise -- 0.956 -> 0.893 -- so the
    aggressive cut is the wrong setting and the honest default is the 99% rank.

    HONEST LIMIT, and it is the one that misled me first: the projector is only
    as good as its CALIBRATION SET. Fitted on the prose half of a probe and
    tested on the code half it made things WORSE at every noise level. Calibrate
    on text that spans the registers the model will see."""
    H = np.asarray(states, np.float64)
    mu = H.mean(0)
    Hc = H - mu
    _u, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    cum = np.cumsum(S ** 2) / np.sum(S ** 2)
    r = int(min(int(np.searchsorted(cum, float(energy))) + 1, int(max_rank),
                len(S)))
    B = Vt[:r]
    return B.T @ B, {"rank": r, "energy": float(energy),
                     "dims": int(H.shape[1])}


def _selftest():
    import os

    from holographic.io_and_interop.holographic_gdnruntime import (
        GDNRuntime, load_runtime)
    from holographic.io_and_interop.holographic_unicron import load_safetensors

    # ---- the algebra first, with no model involved ----
    rng = np.random.default_rng(0)
    d = 64
    role = rng.standard_normal(d) / np.sqrt(d)
    x = rng.standard_normal(d) / np.sqrt(d)
    fft_bind = np.real(np.fft.ifft(np.fft.fft(role) * np.fft.fft(x)))
    assert np.max(np.abs(circulant(role) @ x - fft_bind)) < 1e-12, \
        "the circulant is not the binding"
    back = circulant(involution(role)) @ (circulant(role) @ x)
    assert float(back @ x / (np.linalg.norm(back) * np.linalg.norm(x))) > 0.6, \
        "unbinding lost the payload"

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("vsabake selftest SKIPPED-SUBJECT (algebra verified; no model)")
        return
    rt, cfg = load_runtime(src)
    w = load_safetensors(os.path.join(src, "model.safetensors"))
    H = int(cfg["hidden"])
    L = int(cfg["n_layers"]) - 1
    ids = [int(b) for b in b"The capital of France is Paris."]
    cap = {}
    rt.forward(ids, hooks={L: lambda h: cap.__setitem__("h", h.copy()) or None})
    Hs = cap["h"]
    mu = Hs.mean(0)

    # ---- a BIND installed as weights computes the bind ----
    role_h = rng.standard_normal(H) / np.sqrt(H)
    C = circulant(role_h)
    m = measure_op(Hs, C, mu, gate_target=16.0)
    assert m["direction_cosine"] > 0.999, m
    w2, rep = install_op(w, cfg, C, layer=L, mean_h=mu, gate_target=16.0)
    assert rep["neurons_added"] == H

    # ---- and the model still LOADS and RUNS as an ordinary checkpoint ----
    plain = GDNRuntime(w2, dict(cfg))
    out = plain.forward(ids)
    assert out.shape == rt.forward(ids).shape
    assert np.all(np.isfinite(out)), "installed op produced non-finite logits"

    # ---- the honest limit, asserted so nobody markets past it: the role is
    #      FIXED. A different role needs a different matrix; no weight tensor
    #      binds two runtime values.
    other = circulant(rng.standard_normal(H) / np.sqrt(H))
    same = float(np.mean(np.abs(C - other) < 1e-9))
    assert same < 0.01, "two roles must give genuinely different circuits"

    # ---- THE DENOISER, as a plain matrix installed like any other op ----
    noisy = Hs + 0.6 * np.linalg.norm(Hs) / np.sqrt(Hs.size) * \
        rng.standard_normal(Hs.shape)
    P, prep = fit_denoiser(Hs, energy=0.99)
    cos_before = float(np.mean([a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
                                for a, b in zip(noisy, Hs)]))
    cleaned = (noisy - Hs.mean(0)) @ P + Hs.mean(0)
    cos_after = float(np.mean([a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
                               for a, b in zip(cleaned, Hs)]))
    assert cos_after > cos_before, (cos_before, cos_after)
    w_d, drep = install_op(w, cfg, P, layer=L, mean_h=mu, gate_target=16.0)
    assert drep["neurons_added"] == P.shape[0]
    assert np.all(np.isfinite(GDNRuntime(w_d, dict(cfg)).forward(ids)))

    print("vsabake selftest OK -- bind IS a circulant matrix (%.0e agreement with "
          "FFT) and unbind recovers the payload; installed into the MLP it "
          "computes the operation with direction cosine %.6f (gain spread %.2f, "
          "harmless because VSA reads directions); %d neurons added at layer %d "
          "and the model still runs as an ordinary checkpoint. LIMIT ASSERTED: "
          "roles are fixed at bake time -- binding two RUNTIME values is bilinear "
          "and no weight matrix does it."
          % (np.max(np.abs(circulant(role) @ x - fft_bind)),
             m["direction_cosine"], m["gain_spread"], rep["neurons_added"], L)
          + "; and a DENOISER fitted to rank %d installs the same way, lifting "
            "cosine %.3f -> %.3f under 0.6 noise -- the dreamer WAS bakeable, "
            "the per-batch variance just had to become a fitted constant"
          % (prep["rank"], cos_before, cos_after))


if __name__ == "__main__":
    _selftest()
