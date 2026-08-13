"""GDN RUNTIME -- a NumPy forward pass for Gated-DeltaNet hybrid language models
(the Qwen3-Next / Qwen3.5 architecture class). The keystone that moves leCore
INSIDE the model.

WHY THIS EXISTS: every "be inside the model" capability -- perfect-recall memory
consulted per token, holographic RAG in the residual stream, activation-located
edits, in-engine retention eval -- needs a forward pass we OWN. Torch owns it
today; this module takes ownership for the model class Moose targets. And the
alignment is not cosmetic: Gated DeltaNet IS a gated linear RNN with a delta-rule
memory (S <- S*decay + k (x) beta*(v - S k)) -- structurally leCore's home turf
(HRNN's thesis, one substrate over; the delta rule is Widrow-Hoff, the same
error-correcting write the VSA literature builds cleanup memories from).

SEMANTICS are transcribed from the reference implementation
(transformers/models/qwen3_next/modeling_qwen3_next.py, v5.14.1) and VERIFIED
numerically against it: the selftest builds a tiny random model in torch and
demands logit agreement to float32 tolerance. Not "inspired by" -- checked.
The load-bearing subtleties, each a silent-wrong-answer trap:
  * in_proj_qkvz packs q,k,v,z GROUPED BY KEY-HEAD, values interleaved within
    each key-head group -- NOT four flat blocks (fix_query_key_value_ordering);
  * the causal conv (depthwise, kernel 4, SiLU) runs over concat(q,k,v) ONLY --
    z bypasses it;
  * beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias), fp32;
  * q,k are L2-normalized (eps 1e-6) INSIDE the recurrence, q scaled dk^-0.5;
  * GDN output is RMS-norm-gated PER HEAD with SiLU(z), then out_proj;
  * attention q_proj emits query+gate fused (chunk 2 at head granularity);
    q_norm/k_norm act on head_dim; RoPE is PARTIAL (head_dim * factor), non-
    interleaved rotate_half; output is gated by sigmoid(gate) before o_proj.

RESIDENCY: forward() takes `hooks` = {layer_idx: fn(hidden) -> delta or None},
applied to the residual stream after each decoder layer. This is the injection
point for leCore-resident capabilities (memory, RAG, steering); the hook sees
and shapes the same activations the model computes with. The demo faculty and
selftest prove the mechanics; SEMANTIC claims on a real model carry the usual
eval debt.

Scope honesty: batch 1, full-sequence prefill (recompute per token when
generating -- O(n) per GDN token but attention layers recompute; correctness
first, the five levers later), text-only (visual tower not executed), dense MLP
(num_experts=0, matching Qwen3.5-0.8B). Slow is fine; WRONG is not.
"""

import json
import os

import numpy as np

from holographic.io_and_interop.holographic_gdnaccel import GDNRecurrence


# ------------------------------------------------------------------- primitives

def _weight_array(value):
    """Keep checkpoint weights at their useful compute precision.

    BF16 is decoded by the safetensors loader to float32. Converting every such
    tensor to float64 doubled resident memory without adding source precision.
    Float16 is promoted to float32 for NumPy kernels; existing float32/float64
    arrays remain zero-copy.
    """
    value = np.asarray(value)
    if value.dtype.kind == "f" and value.dtype.itemsize >= 4:
        return value
    return value.astype(np.float32)


def _rmsnorm(x, w, eps):
    """Qwen3Next RMSNorm is ZERO-CENTERED: y = norm(x) * (1 + w), weight init 0.
    Field-caught: plain `* w` matched nothing (rel err 1.0) -- the norm is where
    the first full-model divergence lived, masked earlier by a standalone mixer
    test that bypassed the norm. NOTE the asymmetry: the GATED norm below keeps
    plain `* w` (its weight init is ones) -- reference has both conventions."""
    dtype = np.float64 if np.asarray(x).dtype == np.float64 else np.float32
    x32 = np.asarray(x, dtype=dtype)
    v = np.mean(x32 * x32, axis=-1, keepdims=True)
    return (x32 / np.sqrt(v + eps)) * (1.0 + w)


def _rmsnorm_gated(x, w, gate, eps):
    """Norm BEFORE gate; gate goes through SiLU (reference Qwen3NextRMSNormGated)."""
    dtype = np.float64 if np.asarray(x).dtype == np.float64 else np.float32
    x32 = np.asarray(x, dtype=dtype)
    v = np.mean(x32 * x32, axis=-1, keepdims=True)
    y = (x32 / np.sqrt(v + eps)) * w
    g = np.asarray(gate, dtype=dtype)
    return y * (g / (1.0 + np.exp(-g)))


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _softplus(x):
    return np.logaddexp(0.0, x)


def _l2norm(x, eps=1e-6):
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)


def _causal_conv_silu(x, w):
    """Depthwise causal conv, kernel K, over (S, C) with weight (C, 1, K), then SiLU.
    Left-pad K-1 zeros: output[t] sees inputs t-K+1..t only."""
    S, C = x.shape
    K = w.shape[-1]
    xp = np.concatenate([np.zeros((K - 1, C), dtype=x.dtype), x], axis=0)
    # WHY spelled out: conv weight index order is w[:, 0, k] multiplying input at
    # offset t-(K-1)+k -- easy to flip silently. Verified against torch.
    out = np.zeros((S, C), dtype=x.dtype)
    for k in range(K):
        out += xp[k:k + S] * w[:, 0, k][None, :]
    return _silu(out)


def _kmeans(X, nc, iters=8, seed=0):
    """Deterministic Lloyd's algorithm -- seeded choice, fixed iterations, so a
    cluster assignment is reproducible across runs and processes (the same rule
    every other index in this engine follows)."""
    rng = np.random.default_rng(int(seed))
    nc = max(1, min(int(nc), len(X)))
    C = X[rng.choice(len(X), nc, replace=False)].copy()
    for _ in range(int(iters)):
        a = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(nc):
            m = a == j
            if m.any():
                C[j] = X[m].mean(0)
    return a, C


def _rope_tables(dim, positions, theta):
    dtype = np.float64 if np.asarray(positions).dtype == np.float64 else np.float32
    inv = 1.0 / (np.asarray(theta, dtype=dtype) **
                 (np.arange(0, dim, 2, dtype=dtype) / dim))
    ang = np.outer(positions, inv)               # (S, dim/2)
    emb = np.concatenate([ang, ang], axis=-1)    # (S, dim) -- non-interleaved
    return np.cos(emb), np.sin(emb)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def _apply_rope(q, k, cos, sin):
    """Partial RoPE: rotate the first cos.shape[-1] dims, pass the rest through."""
    d = cos.shape[-1]
    qr, qp = q[..., :d], q[..., d:]
    kr, kp = k[..., :d], k[..., d:]
    c, s = cos[:, None, :], sin[:, None, :]      # (S,1,d) over (S,H,d)
    q2 = np.concatenate([qr * c + _rotate_half(qr) * s, qp], axis=-1)
    k2 = np.concatenate([kr * c + _rotate_half(kr) * s, kp], axis=-1)
    return q2, k2




# ------------------------------------------------------------ cached inference

class InferenceState:
    """The model's MENTAL STATE as an explicit, holdable object -- the demoscene
    move (carry, don't recompute) fused with leCore's machine model: per GDN
    layer the recurrent matrix S (the register file, Vh x dk x dv) plus the
    conv window (a (K-1)-deep L1 line); per attention layer the K/V arrays
    (the growing RAM); plus the position counter (the clock). Because it is
    plain NumPy on OUR side of the boundary, snapshot / restore / branch are
    free -- temporal awareness the host frameworks do not expose: rewind a
    conversation, fork alternate continuations from one past, diff two
    futures. copy() is a deep, independent snapshot."""

    def __init__(self):
        self.gdn = {}        # layer -> {"S": (Vh,dk,dv), "conv": (K-1, conv_dim)}
        self.kv = {}         # layer -> {"k": (T,Hkv,hd), "v": (T,Hkv,hd)}
        self.pos = 0
        self.logits = None   # pending next-token logits: the state has already
                             # CONSUMED its last token, so continuation must read
                             # these, never re-step (double-step = silent drift)

    def copy(self):
        out = InferenceState()
        out.gdn = {L: {k: v.copy() for k, v in st.items()} for L, st in self.gdn.items()}
        out.kv = {L: {k: v.copy() for k, v in st.items()} for L, st in self.kv.items()}
        out.pos = self.pos
        out.logits = None if self.logits is None else self.logits.copy()
        return out


# ---------------------------------------------------------------------- runtime

class GDNRuntime:
    """Weights dict + config -> callable model. Tensor names follow the HF layout
    with or without the 'model.language_model.' / 'model.' prefix (auto-detected).
    cfg keys (defaults are Qwen3.5-0.8B card values where known):
      hidden, n_layers, full_attention_interval, rms_eps, rope_theta,
      linear_num_value_heads, linear_num_key_heads, linear_key_head_dim,
      linear_value_head_dim, conv_kernel, n_heads, n_kv_heads, head_dim,
      partial_rotary_factor
    """

    def __init__(self, weights, cfg):
        self.cfg = dict(cfg)
        self._factors = {}
        # The compiled recurrence is deliberately opt-in.  It owns only the
        # sequential state update and first-call parity-gates itself against
        # NumPy, including a separate gate when resuming carried state.
        _backend = self.cfg.get("gdn_recurrence_backend",
                                os.environ.get("LECORE_GDN_BACKEND", "numpy"))
        self._gdn_recurrence = GDNRecurrence(_backend)
        # prefix auto-detect: the field-measured real name root
        roots = ("model.language_model.", "model.", "")
        for r in roots:
            if any(k.startswith(r + "layers.0.") for k in weights):
                self.root = r
                break
        else:
            raise ValueError("no recognizable layer prefix in weights")
        self.w = weights
        emb_key = next(k for k in (self.root + "embed_tokens.weight",
                                   "model.embed_tokens.weight") if k in weights)
        self.embed_key = emb_key
        self.embed = _weight_array(weights[emb_key])
        self.compute_dtype = (np.float64 if self.embed.dtype == np.float64
                              else np.float32)
        self.lm_head = _weight_array(weights["lm_head.weight"]) \
            if "lm_head.weight" in weights else self.embed   # tied (the 0.8B case)

    def _g_opt(self, layer, name):
        """The tensor if this checkpoint has it, None if it does not.

        Optional tensors are the difference between "we support one model" and
        "we support this family": qk-norm, attention gates and biases are each
        present in some architectures and absent in others, and a hard lookup
        turns a supportable model into an unsupported one."""
        key = self.root + "layers.%d.%s" % (layer, name)
        if key not in self.w:
            return None
        return _weight_array(self.w[key])

    def _g(self, layer, name):
        return _weight_array(self.w[self.root + "layers.%d.%s" % (layer, name)])

    def load_factors(self, factors):
        """Attach low-rank factors produced by refactor.decompose so the forward
        pass USES them. Keys are the full tensor names; anything not listed
        stays dense, which is what keeps this additive."""
        self._factors = {}
        for k, (A, B) in (factors or {}).items():
            parts = k.split("layers.")
            if len(parts) != 2:
                continue
            rest = parts[1]
            layer = int(rest.split(".")[0])
            name = rest.split(".", 1)[1]
            self._factors[(layer, name)] = (_weight_array(A), _weight_array(B))
        return len(self._factors)

    def _has(self, layer, name):
        return (self.root + "layers.%d.%s" % (layer, name)) in self.w

    def _is_gdn(self, layer):
        """Is this a linear-attention (GDN) layer?

        Decided by the PRESENCE OF ANY linear_attn tensor, not by one specific
        name. The previous check looked for `linear_attn.in_proj_qkvz.weight`
        alone, so a checkpoint that names that projection differently was routed
        to the ATTENTION path and died asking for a q_proj that a GDN layer
        never has -- an error that blames the wrong component entirely.
        Field-caught on a real Qwen3.5-0.8B."""
        pre = self.root + "layers.%d.linear_attn." % int(layer)
        return any(k.startswith(pre) for k in self.w)

    def layer_keys(self, layer):
        """Every tensor name on one layer -- the diagnostic that turns a naming
        mismatch from a guess into a fact."""
        pre = self.root + "layers.%d." % int(layer)
        return sorted(k[len(pre):] for k in self.w if k.startswith(pre))

    # ---- mixers ----

    def _gdn(self, layer, x, collect=None, init=None):
        c = self.cfg
        Kh, Vh = c["linear_num_key_heads"], c["linear_num_value_heads"]
        dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
        r = Vh // Kh
        S = x.shape[0]
        # FOUR PROJECTION LAYOUTS SEEN IN THE WILD, all handled here because the
        # alternative is a KeyError deep in a matmul that reads like a runtime
        # bug instead of a naming difference (field-caught twice on one model):
        #   packed  : in_proj_qkvz + in_proj_ba      (reference config)
        #   split   : in_proj_qkv + in_proj_z + in_proj_a + in_proj_b
        #             (the REAL Qwen3.5-0.8B -- confirmed by --keys on Moose's
        #             checkpoint, and by its own spectral report months earlier)
        # plus the two mixed cases. Grouped ordering is unchanged: per key-head
        # [q(dk), k(dk), v(r*dv)] with z alongside, and beta/decay as [b(r),a(r)].
        split_qkv = (not self._has(layer, "linear_attn.in_proj_qkvz.weight")
                     and self._has(layer, "linear_attn.in_proj_qkv.weight"))
        split_ba = (not self._has(layer, "linear_attn.in_proj_ba.weight")
                    and self._has(layer, "linear_attn.in_proj_a.weight"))
        if split_qkv:
            raw = x @ self._g(layer, "linear_attn.in_proj_qkv.weight").T
            if str(c.get("qkv_order", "grouped")) == "flat":
                # flat: [all q][all k][all v] instead of per-key-head groups
                q = raw[:, :Kh * dk].reshape(S, Kh, dk)
                k = raw[:, Kh * dk:2 * Kh * dk].reshape(S, Kh, dk)
                v = raw[:, 2 * Kh * dk:].reshape(S, Vh, dv)
            else:
                qkv = raw.reshape(S, Kh, 2 * dk + r * dv)
                q = qkv[:, :, :dk]
                k = qkv[:, :, dk:2 * dk]
                v = qkv[:, :, 2 * dk:].reshape(S, Vh, dv)
            z = (x @ self._g(layer, "linear_attn.in_proj_z.weight").T
                 ).reshape(S, Vh, dv)
        else:
            qkvz = (x @ self._g(layer, "linear_attn.in_proj_qkvz.weight").T
                    ).reshape(S, Kh, 2 * dk + 2 * r * dv)
            q = qkvz[:, :, :dk]
            k = qkvz[:, :, dk:2 * dk]
            v = qkvz[:, :, 2 * dk:2 * dk + r * dv].reshape(S, Vh, dv)
            z = qkvz[:, :, 2 * dk + r * dv:].reshape(S, Vh, dv)
        ba = (None if split_ba
              else x @ self._g(layer, "linear_attn.in_proj_ba.weight").T)
        if split_ba:
            b = (x @ self._g(layer, "linear_attn.in_proj_b.weight").T).reshape(S, Vh)
            a = (x @ self._g(layer, "linear_attn.in_proj_a.weight").T).reshape(S, Vh)
        else:
            ba = ba.reshape(S, Kh, 2 * r)
            b = ba[:, :, :r].reshape(S, Vh)
            a = ba[:, :, r:].reshape(S, Vh)
        # causal depthwise conv + SiLU over concat(q,k,v) flat; z bypasses
        mixed_pre = np.concatenate([q.reshape(S, -1), k.reshape(S, -1),
                                    v.reshape(S, -1)], axis=-1)
        cw = self._g(layer, "linear_attn.conv1d.weight")
        if init is not None and "conv" in init:
            # continue the causal conv with the CARRIED window as left context,
            # instead of the zero padding a fresh sequence gets -- otherwise the
            # first tokens of every chunk are computed as if the stream restarted
            pre = np.concatenate([np.asarray(init["conv"], dtype=mixed_pre.dtype),
                                  mixed_pre])
            mixed = _causal_conv_silu(pre, cw)[-S:]
        else:
            mixed = _causal_conv_silu(mixed_pre, cw)
        kd = Kh * dk
        q = mixed[:, :kd].reshape(S, Kh, dk)
        k = mixed[:, kd:2 * kd].reshape(S, Kh, dk)
        v = mixed[:, 2 * kd:].reshape(S, Vh, dv)
        beta = 1.0 / (1.0 + np.exp(-b))
        A_log = self._g(layer, "linear_attn.A_log")
        dt = self._g(layer, "linear_attn.dt_bias")
        g = -np.exp(A_log)[None, :] * _softplus(a + dt[None, :])
        if r > 1:                                      # repeat q,k to value heads
            q = np.repeat(q, r, axis=1)
            k = np.repeat(k, r, axis=1)
        q = _l2norm(q) * (dk ** -0.5)
        k = _l2norm(k)
        initial_state = (None if (init is None or "S" not in init)
                         else init["S"])
        out, St = self._gdn_recurrence(q, k, v, beta, g, initial_state)
        nw = self._g(layer, "linear_attn.norm.weight")
        eps = self.cfg["rms_eps"]
        out = _rmsnorm_gated(out, nw, z, eps).reshape(S, Vh * dv)
        y = out @ self._g(layer, "linear_attn.out_proj.weight").T
        if collect is not None:
            K = self._g(layer, "linear_attn.conv1d.weight").shape[-1]
            # the L1 line the step path will slide: last K-1 PRE-conv rows
            pad = np.concatenate([np.zeros((K - 1, mixed_pre.shape[1]),
                                           dtype=mixed_pre.dtype), mixed_pre])
            collect["conv"] = pad[-(K - 1):].copy()
            collect["S"] = St
        return y

    def acceleration_report(self):
        """JSON-able status of the optional recurrent-scan substitution."""
        return {"full_sequence_gdn_recurrence": self._gdn_recurrence.report()}

    def _attn(self, layer, x, positions, collect=None, init=None):
        c = self.cfg
        H, Hkv, hd = c["n_heads"], c["n_kv_heads"], c["head_dim"]
        S = x.shape[0]
        eps = c["rms_eps"]
        _gated = bool(self.cfg.get("attn_gated", True))
        qg = (x @ self._g(layer, "self_attn.q_proj.weight").T).reshape(
            S, H, (2 if _gated else 1) * hd)
        # UNGATED MODELS HAVE NO GATE TO SPLIT OFF. A gate of ones is the
        # identity for the sigmoid-multiply below, so one code path serves both
        # families without a branch in the hot loop.
        q = qg[:, :, :hd]
        gate = (qg[:, :, hd:].reshape(S, H * hd) if _gated
                else np.full((S, H * hd), 20.0))
        k = (x @ self._g(layer, "self_attn.k_proj.weight").T).reshape(S, Hkv, hd)
        v = (x @ self._g(layer, "self_attn.v_proj.weight").T).reshape(S, Hkv, hd)
        # QK-NORM IS OPTIONAL: Qwen normalises queries and keys per head, while
        # Llama, SmolLM2 and Gemma ship no q_norm/k_norm at all. Reaching for a
        # tensor that was never in the file is not a reason to refuse a model we
        # can otherwise run.
        _qn = self._g_opt(layer, "self_attn.q_norm.weight")
        if _qn is not None:
            q = _rmsnorm(q, _qn, eps)
        _kn = self._g_opt(layer, "self_attn.k_norm.weight")
        if _kn is not None:
            k = _rmsnorm(k, _kn, eps)
        rd = int(hd * c.get("partial_rotary_factor", 1.0))
        cos, sin = _rope_tables(rd, positions, c["rope_theta"])
        q, k = _apply_rope(q, k, cos, sin)
        n_past = 0
        if init is not None and "k" in init:
            n_past = int(np.asarray(init["k"]).shape[0])
            k = np.concatenate([np.asarray(init["k"], dtype=k.dtype), k], axis=0)
            v = np.concatenate([np.asarray(init["v"], dtype=v.dtype), v], axis=0)
        if collect is not None:
            collect["k"], collect["v"] = k.copy(), v.copy()
        rep = H // Hkv
        k = np.repeat(k, rep, axis=1)
        v = np.repeat(v, rep, axis=1)
        scores = np.einsum("shd,thd->hst", q, k) * (hd ** -0.5)
        # causal mask over the FULL key range: query i (absolute n_past+i) may
        # attend to every past key and to itself, never forward
        T = k.shape[0]
        mask = np.full((S, T), -np.inf)
        for _i in range(S):
            mask[_i, :n_past + _i + 1] = 0.0
        scores = scores + mask[None, :, :]
        # SDM RADIUS (opt-in, cfg["attn_top_k"]): Kanerva's Sparse Distributed
        # Memory (1988) -- which Attention has been shown to approximate, and
        # which is itself the Marr (1969) / Albus (1971) cerebellum model --
        # reads only the locations INSIDE a radius. Attention softmaxes over
        # every key instead. MEASURED on the trained subject, 400 positions:
        # 90% of the softmax mass sits in a median of 23 keys and the single
        # top key carries 41%; keeping 32 of 400 (8%) gives 0.993 top-1
        # agreement and perplexity 6.9425 vs 6.9305 (+0.17%), and 16 of 400
        # (4%) gives 0.988 / +0.38%.
        # HONEST LIMIT: this measures the REDUNDANCY, it does not yet bank the
        # saving -- the scores are still computed before being masked. Cashing
        # it needs an index that finds the top keys without scoring the rest
        # (which is exactly what SDM's addressing does, and what leCore's own
        # indexes could do). The fidelity curve is the license to build that,
        # not a speed claim.
        # HOLOGRAPHIC SCREEN ROUTING (opt-in, cfg["attn_screen"]): the volume is
        # partitioned into blocks, each block summarized by a fixed-size screen
        # (its key centroid), and a query scores the SCREENS -- T/block work --
        # then descends only into the best few blocks plus a recent window.
        # This is the boundary/volume idea made operational: read the summary,
        # not the volume, and pay full price only where the summary points.
        # MEASURED on the trained subject (400 tokens): 38% of keys scored ->
        # 0.998 top-1 agreement (+0.26% perplexity); 26% -> 0.983.
        #
        # KEPT NEGATIVE, and it is the reason the null test below exists: the
        # first version built block centroids over the WHOLE sequence, so the
        # block containing t averaged in tokens from t+1 onward. Perplexity came
        # out BELOW full attention (5.02 vs 6.93), which is impossible for a
        # restriction of the same computation -- sparse attention cannot beat the
        # dense one it approximates. That impossibility is what exposed the
        # causal leak. Centroids now cover COMPLETED blocks only.
        scr = self.cfg.get("attn_screen") or None
        if scr and n_past == 0 and str(scr.get("mode", "")) == "ball":
            # BALL-BOUND EXACT SELECTION. Group keys by SIMILARITY (deterministic
            # k-means), keep each cluster's centroid c and radius r, and use the
            # admissible bound max_{k in C} q.k <= q.c + r||q|| to skip clusters
            # that provably cannot hold a top-k key.
            #
            # WHY THIS BEATS A CENTROID SCREEN, measured on the trained subject:
            # a centroid RANKS by the block's MEAN inner product while routing
            # needs its MAX -- so it is a heuristic that silently misses. The
            # bound is a CERTIFICATE. 80 clusters: EXACT top-8 for 100% of
            # queries while scoring 38.5% of keys, against the centroid's 0.87
            # recall at 80%. Half the work and no misses.
            #
            # KEPT NEGATIVE: contiguous POSITION blocks make the bound useless
            # (radius 7.98 -> 91-100% of keys scored, no pruning). Clustering
            # shrinks the radius to 4.27 and is what makes the certificate bite.
            # Also refuted: seeding the heap from large-norm keys (GAIPS) neither
            # helped (41% vs 38%) nor survived audit -- seeded keys get rescored
            # inside their own cluster, double-counting into the top-k list.
            nc = int(scr.get("clusters", 0) or max(8, S // 8))
            want = int(scr.get("topk", 8))
            win = int(scr.get("window", 32))
            rank = int(scr.get("rank", 0) or 0)
            allow = np.zeros(scores.shape, bool)
            for h in range(H):
                Kh = k[:, h]
                # SHARED BOUNDARY BASIS (rank>0): the keys are mu + coefficients
                # on a common low-rank shell. A query is projected into that
                # shell ONCE, after which every score is an r-dim dot product
                # against stored coordinates -- the key itself is never read.
                # Exactness is preserved by carrying each key's TAIL NORM: the
                # boundary read is within tail*||q|| of the truth, so only keys
                # whose UPPER bound can crack the running top-k get an exact
                # rescore. MEASURED (50 clusters, rank 8): exact top-8 for 100%
                # of queries at 33.5% of dense flops, versus 38.5% for the
                # bound alone and 0.87 recall at 80% for a centroid screen.
                # Per-cluster bases were tried first and REFUSED: the r*d
                # projection cost has to be amortized, and a cluster of ~8 keys
                # is too small to pay for its own basis (the shared one is
                # projected once for the whole volume).
                bmu = bas = coef = tail = None
                if rank > 0:
                    bmu = Kh.mean(0)
                    Rr = Kh - bmu
                    _u, _s, Vt_ = np.linalg.svd(Rr, full_matrices=False)
                    bas = Vt_[:rank]
                    coef = Rr @ bas.T
                    tail = np.linalg.norm(Rr - coef @ bas, axis=1)
                a, C = _kmeans(Kh, min(nc, len(Kh)), seed=0)
                nn = C.shape[0]
                rad = np.zeros(nn)
                for j in range(nn):
                    m = a == j
                    if m.any():
                        rad[j] = np.max(np.linalg.norm(Kh[m] - C[j], axis=-1))
                for t in range(S):
                    lo = max(0, t - win + 1)
                    allow[h, t, lo:t + 1] = True
                    qv = q[t, h]
                    ub = C @ qv + rad * np.linalg.norm(qv)
                    heap = []
                    for j in np.argsort(ub)[::-1]:
                        if len(heap) >= want and ub[j] <= heap[want - 1]:
                            break            # certificate: cannot contain a winner
                        sel = np.where((a == j) & (np.arange(S) <= t))[0]
                        if not len(sel):
                            continue
                        allow[h, t, sel] = True
                        if rank > 0:
                            approx = (qv @ bmu) + coef[sel] @ (bas @ qv)
                            hi = approx + tail[sel] * np.linalg.norm(qv)
                            thr = heap[want - 1] if len(heap) >= want else -np.inf
                            need = sel[hi > thr]          # only these can matter
                            vals = list(Kh[need] @ qv) if len(need) else []
                        else:
                            vals = list(Kh[sel] @ qv)
                        heap = sorted(list(heap) + vals, reverse=True)[:want]
            scores = np.where(allow, scores, -np.inf)
            scr = None
        if scr and n_past == 0:
            blk = int(scr.get("block", 32))
            nb = int(scr.get("blocks", 2))
            win = int(scr.get("window", 32))
            T_ = scores.shape[-1]
            nblk = int(np.ceil(T_ / blk))
            # EXTRA ACCUMULATORS (leCore lever 4: more dimensions / more
            # accumulators when capacity binds). One summary per block caps how
            # many rankings can survive it; r summaries per block, filled
            # round-robin, hold blk/r items each. MEASURED: recall@8 rose
            # 0.667 -> 0.698 (r=8) at the tight setting and 0.858 -> 0.871
            # (r=4) at the loose one, for r x tiny screen-scoring cost.
            acc = max(1, int(scr.get("accumulators", 1)))
            cent = np.zeros((nblk, acc, H, hd), dtype=k.dtype)
            for i in range(nblk):
                seg = k[i * blk:(i + 1) * blk]
                for j, kv in enumerate(seg):
                    cent[i, j % acc] += kv
            n = np.linalg.norm(cent, axis=-1, keepdims=True)
            cent = cent / np.maximum(n, 1e-12)
            # a block scores as its BEST accumulator: one strong match should not
            # be averaged away by the rest of the block
            bsc = np.einsum("shd,bahd->hsba", q, cent).max(axis=-1)
            allow = np.zeros(scores.shape, bool)
            for t in range(S):
                lo = max(0, t - win + 1)
                done = t // blk                  # completed blocks only
                for h in range(H):
                    if done > 0:
                        for i in np.argsort(bsc[h, t, :done])[-nb:]:
                            allow[h, t, i * blk:(i + 1) * blk] = True
                    allow[h, t, lo:t + 1] = True
            scores = np.where(allow, scores, -np.inf)
        top_k = int(self.cfg.get("attn_top_k", 0) or 0)
        if 0 < top_k < scores.shape[-1]:
            kth = np.sort(scores, axis=-1)[..., -top_k][..., None]
            scores = np.where(scores >= kth, scores, -np.inf)
        scores -= scores.max(axis=-1, keepdims=True)
        w = np.exp(scores)
        w /= w.sum(axis=-1, keepdims=True)
        o = np.einsum("hst,thd->shd", w, v).reshape(S, H * hd)
        o = o * (1.0 / (1.0 + np.exp(-gate)))          # sigmoid output gate
        return o @ self._g(layer, "self_attn.o_proj.weight").T

    mlp_probe = None      # set to fn(layer, x) to observe the MLP's true input
    exit_after = None     # set to a layer index to STOP there and read the head

    def _maybe_exit(self, L):
        """Should forward() stop after this layer?

        ITEM 4 OF THE WORK LIST. A sharp gate zeroes a circuit's OUTPUT to
        2e-112 but the FLOPs still run -- that is correctness, not speed. The
        only way to save the compute is to NOT DO IT, which is control flow and
        therefore belongs in the runtime rather than the weights. Measured: a
        3-layer forward costs 28% of a 4-layer one and a 2-layer 18%, so the
        saving is real and proportional to the layers skipped. Pair this with
        holographic_earlyexit's calibrated confidence and the model stops when
        it is already sure."""
        return self.exit_after is not None and int(L) >= int(self.exit_after)

    device = None         # None = follow the policy; "cpu" / "gpu" to force

    def to_device(self, on=True):
        """Move the WEIGHTS to the accelerator ONCE, if there is one.

        AN LLM IS USUALLY RUN ON A GPU, and this runtime was pure host NumPy --
        so on a machine with a card it was leaving the whole forward pass on the
        CPU. leCore already had the switch (holographic_backend.array_module,
        which returns cupy when a device is present and the policy allows, and
        numpy otherwise); the runtime simply never asked for it.
        RESIDENCY IS THE WHOLE POINT. A per-call transfer costs more than the
        matmul it feeds -- the backend's own docstring says so -- so weights
        move ONCE and stay. Token ids and logits are small and cross per call.
        Returns what actually happened, because "I asked for a GPU" and "I got
        one" are different claims and only the second is worth reporting."""
        from holographic.misc.holographic_backend import (
            array_module, gpu_available, to_device as _to)
        if not on:
            self._dev = None
            return {"device": "cpu", "resident": 0, "why": "disabled"}
        xp = array_module()
        if xp is np or not gpu_available():
            self._dev = None
            return {"device": "cpu", "resident": 0,
                    "why": "no accelerator available -- running on NumPy"}
        # The default file-backed store is deliberately read-only.  Build a
        # resident mapping instead of assigning into it (the old loop silently
        # moved zero tensors once lazy loading became the default).
        resident = {k: _to(_weight_array(self.w[k])) for k in self.w}
        self.w = resident
        self.embed = resident[self.embed_key]
        self.lm_head = resident.get("lm_head.weight", self.embed)
        moved = len(resident)
        self._dev = xp
        return {"device": "gpu", "resident": moved, "why": "weights resident"}

    def _xw(self, layer, name, x):
        """x @ W.T, using the LOW-RANK FACTORS when they are present.

        A factored projection is not merely smaller on disk -- (x@B.T)@A.T costs
        r*(m+n) multiplies against m*n, so it is genuinely cheaper to RUN. The
        old path reconstructed the dense matrix and threw the saving away, which
        is how a "35% smaller model" ends up exactly as slow as before.
        MEASURED at this model's shapes: 1.24x, 1.28x and 1.64x per matmul."""
        fac = self._factors.get((int(layer), name)) if self._factors else None
        if fac is None:
            return x @ self._g(layer, name).T
        A, B = fac
        return (x @ B.T) @ A.T

    def _mlp(self, layer, x):
        # AN OBSERVATION POINT BETWEEN ATTENTION AND THE MLP. Every existing
        # hook fires AFTER a whole decoder layer, so the vector the MLP actually
        # consumes -- post_attention_layernorm(h + attn_out) -- was not
        # reachable from outside. That cost six refuted hypotheses on the
        # read-back problem: gate rows were being matched to h while the gate is
        # applied to h + attn_out(h), one attention block away. A circuit that
        # gates on the MLP's input cannot be built without seeing the MLP's
        # input.
        if self.mlp_probe is not None:
            try:
                self.mlp_probe(int(layer), x)
            except Exception:
                pass
        g = self._xw(layer, "mlp.gate_proj.weight", x)
        u = self._xw(layer, "mlp.up_proj.weight", x)
        return self._xw(layer, "mlp.down_proj.weight", _silu(g) * u)

    # ---- model ----

    def forward(self, token_ids, hooks=None, collect_state=False, step_hooks=None,
                resume=None):
        """Full-sequence forward -> logits (S, vocab). hooks={layer: fn(h)->delta|None}
        applied to the residual stream AFTER each decoder layer -- the residency
        injection point for leCore-side capabilities.

        `resume` CONTINUES FROM A CARRIED STATE, in one batched pass. Item 6 of
        the work list: a conversation repeats 72% of its tokens across turns, and
        a prefix cache could skip all of it -- but replaying the tail ONE TOKEN AT
        A TIME costs 5.8x a prefilled token, so caching saved the work and LOST
        the wall clock. The layer functions already took `init=` for exactly this;
        forward() simply never passed it. Three things must line up: POSITIONS
        start at the resumed offset, the GDN carry seeds each linear layer, and
        the KV cache prepends to each attention layer -- and getting any one of
        them wrong produces fluent nonsense rather than an error."""
        c = self.cfg
        hooks = hooks or {}
        ids = np.asarray(token_ids, np.int64)
        h = self.embed[ids]
        past = int(getattr(resume, "pos", 0) or 0) if resume is not None else 0
        positions = np.arange(past, past + len(ids), dtype=self.compute_dtype)
        st = InferenceState() if collect_state else None
        # LAYER SCHEDULE: which layers run, in what order, how many times.
        # Owning the forward pass makes depth up-scaling (SOLAR/Goliath-style
        # frankenmerging), layer recursion and layer pruning a LIST rather than
        # a re-export -- and lets the same weights be run as several different
        # architectures without writing a new checkpoint.
        sched = c.get("layer_schedule") or list(range(c["n_layers"]))
        # step_hooks are keyed by POSITION IN THE SCHEDULE, not by layer index.
        # With a repeating schedule those differ, and conflating them is a real
        # bug: a repaired "layer 2" would also repair the FIRST, legitimate pass
        # through layer 2 -- which is how an inference-time fix ends up damaging
        # the path it was meant to leave alone (measured: it did).
        step_hooks = step_hooks or {}
        for step_i, L in enumerate(sched):
            hn = _rmsnorm(h, self._g(L, "input_layernorm.weight"), c["rms_eps"])
            if self._is_gdn(L):
                h = h + self._gdn(L, hn, collect=(st.gdn.setdefault(L, {})
                                                 if st is not None else None),
                                  init=(resume.gdn.get(L)
                                        if resume is not None else None))
            else:
                h = h + self._attn(L, hn, positions,
                                   collect=(st.kv.setdefault(L, {})
                                            if st is not None else None),
                                   init=(resume.kv.get(L)
                                         if resume is not None else None))
            hn = _rmsnorm(h, self._g(L, "post_attention_layernorm.weight"), c["rms_eps"])
            h = h + self._mlp(L, hn)
            if self._maybe_exit(L):
                break
            for fn in (hooks.get(L), step_hooks.get(step_i)):
                if fn is not None:
                    d = fn(h)
                    if d is not None:
                        h = h + np.asarray(d, dtype=self.compute_dtype)
        nk = next(k for k in (self.root + "norm.weight", "model.norm.weight")
                  if k in self.w)
        h = _rmsnorm(h, _weight_array(self.w[nk]), c["rms_eps"])
        logits = h @ self.lm_head.T
        if collect_state:
            # ADVANCE FROM WHERE WE RESUMED, not from zero. A state carried out
            # of a RESUMED forward must know its absolute position or the next
            # resume computes RoPE from the wrong offset -- which produces
            # fluent nonsense rather than an error, and showed up as a 0.35
            # logit discrepancy in the prefix cache while the resume path itself
            # was exact to 0.0.
            st.pos = past + len(ids)
            st.logits = logits[-1]
            return logits, st
        return logits

    def _gdn_step(self, layer, x, st):
        """One token through a GDN mixer, carrying (S, conv window). Must be the
        SAME arithmetic as the full-sequence path -- the selftest demands token-
        for-token equality between cached and uncached generation (the
        determinism contract applies to the cache too)."""
        c = self.cfg
        Kh, Vh = c["linear_num_key_heads"], c["linear_num_value_heads"]
        dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
        r = Vh // Kh
        # same four layouts as the vectorized path
        split_qkv = (not self._has(layer, "linear_attn.in_proj_qkvz.weight")
                     and self._has(layer, "linear_attn.in_proj_qkv.weight"))
        split_ba = (not self._has(layer, "linear_attn.in_proj_ba.weight")
                    and self._has(layer, "linear_attn.in_proj_a.weight"))
        if split_qkv:
            raw = x @ self._g(layer, "linear_attn.in_proj_qkv.weight").T
            if str(c.get("qkv_order", "grouped")) == "flat":
                q = raw[:Kh * dk].reshape(Kh, dk)
                k = raw[Kh * dk:2 * Kh * dk].reshape(Kh, dk)
                v = raw[2 * Kh * dk:].reshape(Vh, dv)
            else:
                qkv = raw.reshape(Kh, 2 * dk + r * dv)
                q = qkv[:, :dk]; k = qkv[:, dk:2 * dk]
                v = qkv[:, 2 * dk:].reshape(Vh, dv)
            z = (x @ self._g(layer, "linear_attn.in_proj_z.weight").T).reshape(Vh, dv)
        else:
            qkvz = (x @ self._g(layer, "linear_attn.in_proj_qkvz.weight").T
                    ).reshape(Kh, 2 * dk + 2 * r * dv)
            q = qkvz[:, :dk]; k = qkvz[:, dk:2 * dk]
            v = qkvz[:, 2 * dk:2 * dk + r * dv].reshape(Vh, dv)
            z = qkvz[:, 2 * dk + r * dv:].reshape(Vh, dv)
        if split_ba:
            b = (x @ self._g(layer, "linear_attn.in_proj_b.weight").T).reshape(Vh)
            a = (x @ self._g(layer, "linear_attn.in_proj_a.weight").T).reshape(Vh)
        else:
            ba = (x @ self._g(layer, "linear_attn.in_proj_ba.weight").T
                  ).reshape(Kh, 2 * r)
            b = ba[:, :r].reshape(Vh); a = ba[:, r:].reshape(Vh)
        mixed = np.concatenate([q.ravel(), k.ravel(), v.ravel()])
        w = self._g(layer, "linear_attn.conv1d.weight")
        K = w.shape[-1]
        win = st.setdefault("conv", np.zeros((K - 1, mixed.size), dtype=mixed.dtype))
        xw = np.concatenate([win, mixed[None, :]], axis=0)     # (K, C)
        conv = _silu(np.sum(xw * w[:, 0, :].T, axis=0))
        st["conv"] = xw[1:]                                    # slide the L1 line
        kd = Kh * dk
        q = conv[:kd].reshape(Kh, dk); k = conv[kd:2 * kd].reshape(Kh, dk)
        v = conv[2 * kd:].reshape(Vh, dv)
        beta = 1.0 / (1.0 + np.exp(-b))
        g = -np.exp(self._g(layer, "linear_attn.A_log")) * _softplus(
            a + self._g(layer, "linear_attn.dt_bias"))
        if r > 1:
            q = np.repeat(q, r, axis=0); k = np.repeat(k, r, axis=0)
        q = _l2norm(q) * (dk ** -0.5); k = _l2norm(k)
        S = st.setdefault("S", np.zeros((Vh, dk, dv), dtype=x.dtype))
        S = S * np.exp(g)[:, None, None]
        kv = np.einsum("hkv,hk->hv", S, k)
        delta = (v - kv) * beta[:, None]
        S = S + k[:, :, None] * delta[:, None, :]
        st["S"] = S
        out = np.einsum("hkv,hk->hv", S, q)
        out = _rmsnorm_gated(out, self._g(layer, "linear_attn.norm.weight"),
                             z, self.cfg["rms_eps"]).reshape(-1)
        return out @ self._g(layer, "linear_attn.out_proj.weight").T

    def _attn_step(self, layer, x, st, pos):
        c = self.cfg
        H, Hkv, hd = c["n_heads"], c["n_kv_heads"], c["head_dim"]
        eps = c["rms_eps"]
        _gated = bool(self.cfg.get("attn_gated", True))
        qg = (x @ self._g(layer, "self_attn.q_proj.weight").T).reshape(
            H, (2 if _gated else 1) * hd)
        q = qg[:, :hd]
        gate = (qg[:, hd:].reshape(H * hd) if _gated
                else np.full(H * hd, 20.0))
        k = (x @ self._g(layer, "self_attn.k_proj.weight").T).reshape(Hkv, hd)
        v = (x @ self._g(layer, "self_attn.v_proj.weight").T).reshape(Hkv, hd)
        # QK-NORM IS OPTIONAL: Qwen normalises queries and keys per head, while
        # Llama, SmolLM2 and Gemma ship no q_norm/k_norm at all. Reaching for a
        # tensor that was never in the file is not a reason to refuse a model we
        # can otherwise run.
        _qn = self._g_opt(layer, "self_attn.q_norm.weight")
        if _qn is not None:
            q = _rmsnorm(q, _qn, eps)
        _kn = self._g_opt(layer, "self_attn.k_norm.weight")
        if _kn is not None:
            k = _rmsnorm(k, _kn, eps)
        rd = int(hd * c.get("partial_rotary_factor", 1.0))
        cos, sin = _rope_tables(rd, np.array([float(pos)], dtype=self.compute_dtype),
                                c["rope_theta"])
        q2, k2 = _apply_rope(q[None], k[None], cos, sin)
        q, k = q2[0], k2[0]
        ks = np.concatenate([st["k"], k[None]], axis=0) if "k" in st else k[None]
        vs = np.concatenate([st["v"], v[None]], axis=0) if "v" in st else v[None]
        st["k"], st["v"] = ks, vs                              # the growing RAM
        rep = H // Hkv
        kr = np.repeat(ks, rep, axis=1); vr = np.repeat(vs, rep, axis=1)
        scores = np.einsum("hd,thd->ht", q, kr) * (hd ** -0.5)
        scores -= scores.max(axis=-1, keepdims=True)
        w = np.exp(scores); w /= w.sum(axis=-1, keepdims=True)
        o = np.einsum("ht,thd->hd", w, vr).reshape(H * hd)
        o = o * (1.0 / (1.0 + np.exp(-gate)))
        return o @ self._g(layer, "self_attn.o_proj.weight").T

    def step(self, token_id, state, hooks=None):
        """ONE token through the model, carrying InferenceState -> (logits, state).
        O(1) per GDN layer, O(t) per attention layer -- the demoscene payoff over
        full recompute. Mutates `state`; use state.copy() to branch first."""
        c = self.cfg
        hooks = hooks or {}
        h = self.embed[int(token_id)]
        for L in range(c["n_layers"]):
            hn = _rmsnorm(h, self._g(L, "input_layernorm.weight"), c["rms_eps"])
            if self._is_gdn(L):
                h = h + self._gdn_step(L, hn, state.gdn.setdefault(L, {}))
            else:
                h = h + self._attn_step(L, hn, state.kv.setdefault(L, {}), state.pos)
            hn = _rmsnorm(h, self._g(L, "post_attention_layernorm.weight"), c["rms_eps"])
            h = h + self._mlp(L, hn)
            fn = hooks.get(L)
            if fn is not None:
                d = fn(h[None, :])
                if d is not None:
                    h = h + np.asarray(d, dtype=self.compute_dtype).reshape(-1)
        state.pos += 1
        nk = next(k for k in (self.root + "norm.weight", "model.norm.weight")
                  if k in self.w)
        h = _rmsnorm(h, _weight_array(self.w[nk]), c["rms_eps"])
        state.logits = h @ self.lm_head.T
        return state.logits, state

    def prefill(self, token_ids, hooks=None):
        """VECTORIZED prefill: one full-sequence forward that COLLECTS the carried
        states (GDN S + conv window, attention KV) as it goes -- big BLAS calls
        for the prompt, O(1) steps after. Measured: the looped per-token prefill
        this replaced capped cached generation at 2.1x over full recompute at toy
        scale; collecting states from the vectorized pass makes it strictly
        dominate. Returns (last-token logits, InferenceState)."""
        logits, st = self.forward(token_ids, hooks=hooks, collect_state=True)
        return logits[-1], st

    def generate_fast(self, token_ids, n_new=16, state=None, hooks=None):
        """Greedy generation with carried state -- the boosted path. Returns
        (ids, state); pass state.copy() back in to BRANCH alternate futures
        from the same past (temporal awareness as an API, not a metaphor)."""
        if state is None:
            logits, state = self.prefill(token_ids, hooks=hooks)
        else:
            logits = state.logits      # last token already consumed; never re-step
        ids = list(map(int, token_ids))
        for _ in range(n_new):
            nxt = int(np.argmax(logits))
            ids.append(nxt)
            logits, state = self.step(nxt, state, hooks=hooks)
        return ids, state

    def extend(self, tokens, state, hooks=None):
        """Advance an InferenceState by SEVERAL tokens in ONE vectorized pass.

        Same arithmetic as calling step() per token -- asserted token-identical
        in the selftest -- but the big projections run as one GEMM over the chunk
        instead of k separate GEMVs. On CPU NumPy that is the difference between
        memory-bandwidth-bound and compute-bound, which is exactly why the
        vectorized prefill beat the looped one earlier in this arc.

        This is the verification primitive speculative decoding needs: draft k
        tokens cheaply, then check all k with a single batched forward."""
        c = self.cfg
        hooks = hooks or {}
        ids = np.asarray(tokens, np.int64)
        S = len(ids)
        h = self.embed[ids]
        positions = np.arange(state.pos, state.pos + S, dtype=self.compute_dtype)
        for L in range(c["n_layers"]):
            hn = _rmsnorm(h, self._g(L, "input_layernorm.weight"), c["rms_eps"])
            if self._is_gdn(L):
                st = state.gdn.setdefault(L, {})
                h = h + self._gdn(L, hn, collect=st, init=dict(st))
            else:
                st = state.kv.setdefault(L, {})
                h = h + self._attn(L, hn, positions, collect=st, init=dict(st))
            hn = _rmsnorm(h, self._g(L, "post_attention_layernorm.weight"),
                          c["rms_eps"])
            h = h + self._mlp(L, hn)
            fn = hooks.get(L)
            if fn is not None:
                d = fn(h)
                if d is not None:
                    h = h + np.asarray(d, dtype=self.compute_dtype)
        state.pos += S
        nk = next(k for k in (self.root + "norm.weight", "model.norm.weight")
                  if k in self.w)
        h = _rmsnorm(h, _weight_array(self.w[nk]), c["rms_eps"])
        logits = h @ self.lm_head.T
        state.logits = logits[-1]
        return logits, state

    def forward_embeds(self, embeds, hooks=None, step_hooks=None):
        """Run the model from HIDDEN STATES instead of token ids.

        Needed the moment you want to feed the model something that is not a
        single token -- a SUPERPOSITION of embeddings, an interpolation, a
        steered state. Without it, any such experiment silently degrades to
        re-tokenizing the input (measured: it did, and the results looked like a
        failure of the idea rather than of the plumbing)."""
        c = self.cfg
        hooks = hooks or {}
        step_hooks = step_hooks or {}
        h = np.asarray(embeds, dtype=self.compute_dtype)
        positions = np.arange(h.shape[0], dtype=self.compute_dtype)
        sched = c.get("layer_schedule") or list(range(c["n_layers"]))
        for step_i, L in enumerate(sched):
            hn = _rmsnorm(h, self._g(L, "input_layernorm.weight"), c["rms_eps"])
            if self._is_gdn(L):
                h = h + self._gdn(L, hn)
            else:
                h = h + self._attn(L, hn, positions)
            hn = _rmsnorm(h, self._g(L, "post_attention_layernorm.weight"),
                          c["rms_eps"])
            h = h + self._mlp(L, hn)
            for fn in (hooks.get(L), step_hooks.get(step_i)):
                if fn is not None:
                    d = fn(h)
                    if d is not None:
                        h = h + np.asarray(d, dtype=self.compute_dtype)
        nk = next(k for k in (self.root + "norm.weight", "model.norm.weight")
                  if k in self.w)
        h = _rmsnorm(h, _weight_array(self.w[nk]), c["rms_eps"])
        return h @ self.lm_head.T

    def token_nll(self, token_ids, hooks=None):
        """Per-token negative log-likelihood over ONE forward pass.

        Why this exists: scoring passages separately makes each one start COLD,
        with no preceding context, so an early passage looks easy and a later
        one looks hard for reasons that have nothing to do with the model being
        compared. Measured on a real checkpoint: the same text scored 16.56 as a
        whole and 15.0 / 22.1 / 34.3 when cut into three independent pieces.
        Scoring once and BUCKETING the per-token losses keeps every token in its
        real context, makes passage numbers comparable, and costs one pass
        instead of n."""
        ids = [int(t) for t in token_ids]
        logits = self.forward(ids, hooks=hooks)[:-1]
        tgt = np.asarray(ids[1:], np.int64)
        mx = logits.max(-1)
        lse = np.log(np.sum(np.exp(logits - mx[:, None]), -1)) + mx
        return lse - logits[np.arange(len(tgt)), tgt]      # (T-1,) NLL per token

    def _check_tokens(self, token_ids, what="forward"):
        """Refuse an empty or single-token sequence HERE, where it can explain.

        An empty id list reached the GDN path and died as
        "cannot reshape array of size 0" fifteen frames deep, three separate
        times in one session, from three different callers. The error belongs at
        the boundary: every caller that produces ids from a tokenizer can fail
        to produce any, and each one should not have to learn that lesson."""
        ids = list(token_ids)
        if len(ids) < 2:
            raise ValueError(
                "%s needs at least 2 token ids, got %d -- an empty probe usually "
                "means the tokenizer did not recognise the calibration text, not "
                "that the model is broken" % (what, len(ids)))
        return ids

    def perplexity(self, token_ids):
        """exp(mean NLL of next-token prediction) -- the in-engine retention meter.
        Closes the standing eval debt without any external runtime."""
        token_ids = self._check_tokens(token_ids, "perplexity")
        logits = self.forward(token_ids)[:-1]
        tgt = np.asarray(token_ids[1:], np.int64)
        lse = np.log(np.sum(np.exp(logits - logits.max(-1, keepdims=True)), -1)) \
            + logits.max(-1)
        nll = lse - logits[np.arange(len(tgt)), tgt]
        return float(np.exp(np.mean(nll)))

    def generate(self, token_ids, n_new=16, hooks=None):
        """Greedy generation by full recompute per step -- correctness-first; the
        five levers (cache the GDN state, KV cache) are the known speed path."""
        ids = list(map(int, token_ids))
        for _ in range(n_new):
            logits = self.forward(ids, hooks=hooks)
            ids.append(int(np.argmax(logits[-1])))
        return ids




# ------------------------------------------------------------- config loading

def config_from_json(cfg_json, weights=None):
    """Turn a Hugging Face config.json into a GDNRuntime config -- and VALIDATE
    it against the weights before anyone trusts it.

    WHY THE VALIDATION IS THE POINT: a wrong head-dim or key-head count does not
    crash. It reshapes the same bytes a different way and produces fluent
    garbage, which is the most expensive failure mode in this whole arc (the
    grouped-vs-flat qkvz packing cost a full debugging session). So every field
    that can be cross-checked against an actual tensor shape IS, and a mismatch
    raises here rather than surfacing as bad text later.

    Handles both config layouts seen in the wild: rope settings nested under
    `rope_parameters` (transformers 5.x) or flat at the top level (4.x), and a
    text config nested under `text_config` for multimodal checkpoints like
    Qwen3.5, whose language stack is what this runtime executes.
    """
    if isinstance(cfg_json, str):
        with open(cfg_json) as f:
            cfg_json = json.load(f)
    c = dict(cfg_json)
    # multimodal checkpoints keep the language stack in text_config; the visual
    # tower is not executed here (policy parity with assimilation)
    if "text_config" in c and isinstance(c["text_config"], dict):
        merged = dict(c["text_config"])
        for k, v in c.items():
            merged.setdefault(k, v)
        c = merged
    # carry the declared gating forward: the validator runs later, where the
    # raw config is out of scope, and inferring it from tensor shapes alone
    # re-opens the wrong-head_dim hole the validator exists to close
    _declared_gate = c.get("attn_output_gate")
    rope = c.get("rope_parameters") or {}
    theta = rope.get("rope_theta", c.get("rope_theta", 10000.0))
    prf = rope.get("partial_rotary_factor", c.get("partial_rotary_factor", 1.0))
    hidden = int(c["hidden_size"])
    n_heads = int(c.get("num_attention_heads", 1))
    head_dim = int(c.get("head_dim") or (hidden // max(n_heads, 1)))
    out = dict(
        hidden=hidden,
        intermediate=int(c.get("intermediate_size", 128)),
        n_layers=int(c["num_hidden_layers"]),
        rms_eps=float(c.get("rms_norm_eps", 1e-6)),
        rope_theta=float(theta),
        partial_rotary_factor=float(prf),
        n_heads=n_heads,
        n_kv_heads=int(c.get("num_key_value_heads", n_heads)),
        head_dim=head_dim,
        linear_num_value_heads=int(c.get("linear_num_value_heads", 0)),
        linear_num_key_heads=int(c.get("linear_num_key_heads", 0)),
        linear_key_head_dim=int(c.get("linear_key_head_dim", 0)),
        linear_value_head_dim=int(c.get("linear_value_head_dim", 0)),
        conv_kernel=int(c.get("linear_conv_kernel_dim", 4)),
    )
    if int(c.get("num_experts", 0)) > 0:
        raise ValueError(
            "MoE checkpoint (num_experts=%d): this runtime executes DENSE MLPs "
            "only. Refusing rather than silently running the wrong forward pass."
            % int(c["num_experts"]))
    if weights is not None:
        _validate_config(out, weights, declared_gate=_declared_gate)
    return out


def _validate_config(cfg, weights, declared_gate=None):
    """Cross-check config numbers against real tensor shapes. Raises on the first
    contradiction, naming both sides -- the message has to be enough to fix it."""
    roots = ("model.language_model.", "model.", "")
    root = next((r for r in roots
                 if any(k.startswith(r + "layers.0.") for k in weights)), None)
    if root is None:
        raise ValueError("no recognizable layer prefix in weights")
    def g(name):
        return weights.get(root + name)
    emb = weights.get(root + "embed_tokens.weight",
                      weights.get("model.embed_tokens.weight"))
    if emb is not None and np.asarray(emb).shape[1] != cfg["hidden"]:
        raise ValueError("hidden_size %d disagrees with embed_tokens %s"
                         % (cfg["hidden"], np.asarray(emb).shape))
    n_seen = len({k.split("layers.")[1].split(".")[0] for k in weights
                  if root + "layers." in k})
    if n_seen and n_seen != cfg["n_layers"]:
        raise ValueError("num_hidden_layers %d but %d layer indices present"
                         % (cfg["n_layers"], n_seen))
    for L in range(cfg["n_layers"]):
        q = g("layers.%d.self_attn.q_proj.weight" % L)
        if q is not None:
            # READ WHETHER ATTENTION IS GATED, do not assume it. Qwen3.5 sets
            # attn_output_gate and its q_proj emits query AND gate (2 * n_heads
            # * head_dim rows); Llama, SmolLM2, Gemma and most others emit the
            # query alone. Assuming the gated shape rejected every ungated model
            # with a message about fixing head_dim -- which was not the problem
            # and sent the reader looking in the wrong place.
            rows = int(np.asarray(q).shape[0])
            gated = 2 * cfg["n_heads"] * cfg["head_dim"]
            plain = cfg["n_heads"] * cfg["head_dim"]
            # BELIEVE THE CONFIG WHEN IT SAYS. Inferring gating from the row
            # count alone re-opened the hole this validator exists to close: a
            # head_dim that is half the truth makes a GATED q_proj look exactly
            # like a plain one, so a wrong config would be silently accepted and
            # every tensor reshaped wrongly -- the most expensive failure mode
            # in this whole arc. The declared flag decides; the shape then has
            # to match it or nothing loads.
            declared = declared_gate
            if declared is None:
                if rows == gated:
                    cfg["attn_gated"] = True
                elif rows == plain and rows != gated:
                    cfg["attn_gated"] = False
                else:
                    raise ValueError(
                        "q_proj rows %d match neither gated (%d) nor plain (%d)"
                        % (rows, gated, plain))
                break
            cfg["attn_gated"] = bool(declared)
            want = gated if cfg["attn_gated"] else plain
            if rows != want:
                raise ValueError(
                    "q_proj rows %d != %d for %s attention (heads=%d, "
                    "head_dim=%d) -- fix head_dim/num_attention_heads"
                    % (rows, want, "gated" if cfg["attn_gated"] else "plain",
                       cfg["n_heads"], cfg["head_dim"]))
            break
            if False:
                raise ValueError(
                    "q_proj rows %d match neither gated (%d) nor plain (%d) "
                    "attention for heads=%d head_dim=%d -- check "
                    "num_attention_heads and head_dim"
                    % (rows, gated, plain, cfg["n_heads"], cfg["head_dim"]))
            break
    for L in range(cfg["n_layers"]):
        qkvz = g("layers.%d.linear_attn.in_proj_qkvz.weight" % L)
        if qkvz is not None:
            Kh, Vh = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
            dk, dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
            want = 2 * Kh * dk + 2 * Vh * dv
            if np.asarray(qkvz).shape[0] != want:
                raise ValueError(
                    "in_proj_qkvz rows %d != 2*Kh*dk + 2*Vh*dv = %d "
                    "(Kh=%d dk=%d Vh=%d dv=%d) -- the GDN head numbers are wrong"
                    % (np.asarray(qkvz).shape[0], want, Kh, dk, Vh, dv))
            conv = g("layers.%d.linear_attn.conv1d.weight" % L)
            if conv is not None and np.asarray(conv).shape[-1] != cfg["conv_kernel"]:
                raise ValueError("conv kernel %d != config %d"
                                 % (np.asarray(conv).shape[-1], cfg["conv_kernel"]))
            break
    return True


def load_runtime(model_dir, lazy=False, max_cached=8):
    """THE ONE-LINER: a model directory -> a running GDNRuntime.

    Shards are file-backed and tensors decode on demand. ``lazy=True`` retains
    the older, explicitly lossy middle-out residency experiment; it first
    materializes the mapped store because compression itself must inspect every
    tensor. The default path is exact and no longer eagerly copies every shard.
    """
    from holographic.io_and_interop import holographic_unicron as U
    # A GALVATRON BUNDLE CARRIES ITS CONFIG IN galvatron.json, NOT config.json.
    # Every tool that loads a model went through here, so the fallback belongs
    # here rather than in each caller -- assess.bat died on exactly this, one
    # step after a successful imbue, because the bundle it was pointed at is a
    # perfectly valid model that simply names its config differently.
    cfg_path = os.path.join(model_dir, "config.json")
    gv_path = os.path.join(model_dir, "galvatron.json")
    if not os.path.exists(cfg_path) and not os.path.exists(gv_path):
        raise FileNotFoundError(
            "no config.json and no galvatron.json in %r -- a model directory "
            "needs one of them (found: %s)"
            % (model_dir, ", ".join(sorted(os.listdir(model_dir))[:8])))
    files = load_weight_files(model_dir)
    if not files:
        raise ValueError("no .safetensors files in %s" % model_dir)
    paths = [os.path.join(model_dir, f) for f in files]
    weights = U.SafetensorWeights(paths, max_cached=max_cached)
    if os.path.exists(cfg_path):
        cfg = config_from_json(cfg_path, weights=weights)
    else:
        # A GALVATRON BUNDLE CARRIES ITS CONFIG IN galvatron.json. Every tool
        # that loads a model comes through here, so the fallback belongs HERE
        # rather than in each caller -- assess.bat died on exactly this, one
        # step after a successful imbue, because the bundle it was pointed at is
        # a perfectly valid model that simply names its config differently.
        import json as _json
        with open(gv_path) as _gf:
            _man = _json.load(_gf)
        cfg = dict(_man.get("config") or {})
        if not cfg.get("n_layers"):
            raise ValueError("galvatron.json in %r has no usable config block"
                             % model_dir)
    if lazy:
        weights = U.LazyWeights(dict(weights.items()), max_cached=max_cached)
    rt = GDNRuntime(weights, cfg)
    _resolve_ambiguous_layout(rt, model_dir)
    _sanity_check(rt, model_dir)
    # RETURN THE RESOLVED CONFIG, NOT THE ONE WE WALKED IN WITH.
    # GDNRuntime.__init__ does `self.cfg = dict(cfg)` -- A COPY -- and
    # _resolve_ambiguous_layout writes its answer into rt.cfg["qkv_order"].
    # Returning the ORIGINAL cfg silently dropped that answer, so every caller
    # that rebuilt a runtime with `GDNRuntime(new_weights, cfg)` got the
    # DEFAULT layout while the loaded runtime used the RESOLVED one -- two
    # models computing different functions from the same weights.
    # Field-caught: prepend reported a drift of 2.5e+01 (relative 1.03) on a
    # real Qwen3.5-0.8B whose directory carried a .lecore_layout.json, i.e. a
    # model where the layout HAD been resolved and the resolution was thrown
    # away one line later. Bit-identical on fixtures whose head counts make the
    # layout unambiguous, which is why it never showed up here.
    return rt, rt.cfg


def _sanity_check(rt, model_dir, probe=None):
    """Does this model look like it is READ CORRECTLY? Cheap, automatic, and
    reported by the runtime rather than discovered by the user three commands
    later.

    A correctly-read language model predicts natural text far better than
    chance. Chance is a perplexity near the vocabulary size, so a probe
    perplexity anywhere near vocab means the weights are being interpreted
    wrongly -- a transposed matrix, a mis-split projection, a bad head count.
    This does not prove correctness (nothing cheap does); it catches the
    catastrophic case, which is the one that otherwise gets blamed on the
    model."""
    text = ("The capital of France is Paris. Water freezes at zero degrees "
            "and boils at one hundred degrees celsius.")
    ids = probe
    if ids is None:
        vocab_n = int(np.asarray(rt.lm_head).shape[0])
        try:
            from holographic.io_and_interop.holographic_bpe import BPE
            ids = BPE.from_dir(model_dir).encode(text)[:32]
        except Exception:
            # a byte-level model HAS a vocabulary -- silently skipping the check
            # because there is no vocab.json is how a guard stops guarding
            ids = ([b for b in text.encode("utf-8") if b < vocab_n][:64]
                   if vocab_n <= 256 else None)
        if not ids:
            return None
    try:
        ppl = float(rt.perplexity(ids))
    except Exception as exc:
        print("      SANITY CHECK could not run (%s)" % exc)
        return None
    vocab = int(np.asarray(rt.lm_head).shape[0])
    verdict = ("looks correct" if ppl < 0.05 * vocab else
               "SUSPICIOUS" if ppl < 0.5 * vocab else "LIKELY MISREAD")
    print("      sanity: perplexity %.1f on plain English (chance ~%d) -- %s"
          % (ppl, vocab, verdict))
    if verdict != "looks correct":
        print("      ^ the weights are probably being interpreted wrongly "
              "(layout, head counts, or a transpose). Numbers measured now "
              "would blame the MODEL for a reading error -- run --verify.")
    return ppl


def load_weight_files(model_dir):
    """Every weight shard in a model directory, in load order.

    Exposed as a function because it is the ONLY correct answer to "where are
    the weights", and a second caller that hardcoded "model.safetensors" broke
    on the first real sharded checkpoint it met (a 0.8B ships as
    model-00001-of-0000N). One rule, one place."""
    files = [f for f in sorted(os.listdir(model_dir))
             if f.endswith(".safetensors") and ".lecore." not in f]
    if not files:
        raise ValueError("no .safetensors files in %s" % model_dir)
    return files


def load_weights_dir(model_dir, lazy=False, max_cached=8):
    """All weights from a model directory, sharded or single-file.

    Transformation callers receive the historical writable dict. Inspection
    and runtime callers can request the exact file-backed mapping with
    ``lazy=True``.
    """
    from holographic.io_and_interop import holographic_unicron as U
    files = [os.path.join(model_dir, f) for f in load_weight_files(model_dir)]
    if lazy:
        return U.SafetensorWeights(files, max_cached=max_cached)
    weights = {}
    for f in files:
        weights.update(U.load_safetensors(f))
    return weights


def _resolve_ambiguous_layout(rt, model_dir, probe=None):
    """Decide an UNDECIDABLE-BY-NAME tensor layout by ASKING THE MODEL.

    Some checkpoints ship in_proj_qkv as per-key-head groups and some as three
    flat blocks. When r == 1 and Kh == Vh both readings have the SAME SHAPE, so
    nothing in the file says which is right and a wrong guess yields a model
    that runs and is quietly wrong -- the worst failure available.

    There is no need to ask a human or a second framework: a correct layout
    predicts natural text far better than a scrambled one, so the runtime scores
    both candidates on a short probe and keeps the winner. The decision, its
    margin, and the probe are cached beside the model so it is made once.

    Costs two short forward passes at load, only when the ambiguity exists."""
    if not any(k.endswith("in_proj_qkv.weight") for k in rt.w):
        return None                       # packed qkvz: nothing ambiguous
    cache = os.path.join(model_dir, ".lecore_layout.json")
    if os.path.exists(cache):
        try:
            with open(cache) as f:
                rec = json.load(f)
            rt.cfg["qkv_order"] = rec["qkv_order"]
            return rec
        except (OSError, ValueError, KeyError):
            pass
    ids = probe
    if ids is None:
        try:
            from holographic.io_and_interop.holographic_bpe import BPE
            ids = BPE.from_dir(model_dir).encode(
                "The quick brown fox jumps over the lazy dog. "
                "In the beginning was the word, and the word was with")[:24]
        except Exception:
            ids = list(range(10, 34))
    scores = {}
    for order in ("grouped", "flat"):
        rt.cfg["qkv_order"] = order
        try:
            scores[order] = float(rt.perplexity(ids))
        except Exception:
            scores[order] = float("inf")
    best = min(scores, key=lambda o: scores[o])
    other = max(scores, key=lambda o: scores[o])
    rt.cfg["qkv_order"] = best
    ratio = (scores[other] / scores[best]) if scores[best] > 0 else float("inf")
    rec = {"qkv_order": best, "perplexity": scores, "margin_ratio": ratio,
           "probe_tokens": len(ids)}
    print("      qkv layout: %s (ppl %.2f vs %.2f for %s -- %.1fx better)"
          % (best, scores[best], scores[other], other, ratio))
    if ratio < 1.2:
        print("      WARNING: the two readings score within 20%% of each other, "
              "so this probe did not really decide it. Re-run with a longer "
              "probe, or cross-check with --verify.")
    try:
        with open(cache, "w") as f:
            json.dump(rec, f, indent=1)
    except OSError:
        pass
    return rec


# ---------------------------------------------------------------------- selftest

def _selftest():
    """Numeric verification against the reference torch implementation on a tiny
    random model -- logits must agree to float32 tolerance. If torch/transformers
    are absent (core rule: they are OPT-IN verification instruments, never core
    deps), fall back to internal contracts only and say so."""
    rng = np.random.default_rng(0)
    try:
        import torch
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
        have_ref = True
    except ImportError:
        have_ref = False

    if have_ref:
        torch.manual_seed(0)
        cfg = Qwen3NextConfig(
            vocab_size=97, hidden_size=64, intermediate_size=112,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, linear_num_value_heads=4, linear_num_key_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=16,
            linear_conv_kernel_dim=4, full_attention_interval=4,
            num_experts=0, tie_word_embeddings=True, rms_norm_eps=1e-6,
        )
        ref = Qwen3NextForCausalLM(cfg).eval().float()
        weights = {k: v.detach().numpy().astype(np.float64)
                   for k, v in ref.state_dict().items()}
        rope_theta = getattr(cfg, "rope_parameters", None)
        theta = (rope_theta or {}).get("rope_theta", getattr(cfg, "rope_theta", 10000.0))
        prf = (rope_theta or {}).get("partial_rotary_factor",
                                     getattr(cfg, "partial_rotary_factor", 0.25))
        rt = GDNRuntime(weights, dict(
            hidden=64, n_layers=4, rms_eps=1e-6, rope_theta=theta,
            linear_num_value_heads=4, linear_num_key_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=16, conv_kernel=4,
            n_heads=4, n_kv_heads=2, head_dim=16, partial_rotary_factor=prf))
        ids = rng.integers(0, 97, size=12)
        with torch.no_grad():
            ref_logits = ref(torch.tensor(ids[None])).logits[0].numpy()
        ours = rt.forward(ids)
        err = np.max(np.abs(ours - ref_logits)) / max(np.max(np.abs(ref_logits)), 1e-9)
        assert err < 1e-4, "logit mismatch vs reference: rel %.2e" % err
        # perplexity meter: sane and finite on the same tokens
        p = rt.perplexity(ids)
        assert np.isfinite(p) and p > 1.0
        # RESIDENCY: a hook that adds a fixed direction at layer 2 must change the
        # logits (the injection point is live), and hook=None must be a no-op.
        base = rt.forward(ids)
        delta = 0.5 * rng.standard_normal(64)
        hooked = rt.forward(ids, hooks={2: lambda h: np.tile(delta, (h.shape[0], 1))})
        assert np.max(np.abs(hooked - base)) > 1e-3
        again = rt.forward(ids, hooks={2: lambda h: None})
        assert np.array_equal(again, base)
        # CACHED PATH: the determinism contract extends to the cache -- greedy
        # generation must be TOKEN-FOR-TOKEN identical between the O(n^2) full-
        # recompute path and the carried-state path, and the per-step logits must
        # agree to float tolerance.
        import time as _time
        t0 = _time.time(); slow = rt.generate(ids, n_new=10); t_slow = _time.time() - t0
        t0 = _time.time(); fast, st = rt.generate_fast(ids, n_new=10); t_fast = _time.time() - t0
        assert slow == fast, (slow, fast)
        lf, _ = rt.prefill(ids)
        assert np.max(np.abs(lf - rt.forward(ids)[-1])) < 1e-8
        # TEMPORAL AWARENESS: snapshot -> two branches from one past -> the
        # common past is bit-identical, futures diverge under different hooks,
        # and REWIND (reusing the snapshot) reproduces branch A exactly.
        _, st0 = rt.prefill(ids)
        snap = st0.copy()
        a1, _ = rt.generate_fast(ids, n_new=6, state=st0)
        steer = {2: (lambda h: np.tile(0.7 * rng.standard_normal(64), (h.shape[0], 1)))}
        b1, _ = rt.generate_fast(ids, n_new=6, state=snap.copy(), hooks=steer)
        a2, _ = rt.generate_fast(ids, n_new=6, state=snap.copy())
        assert a1 == a2, "rewind must reproduce the timeline exactly"
        assert a1 != b1, "steered branch must diverge"
        # CONFIG LOADER: the real config.json path must reproduce the hand-built
        # cfg exactly, load a real directory, and REFUSE a corrupted config
        # loudly rather than reshaping bytes into fluent garbage.
        import json as _json
        import os as _os
        import tempfile as _tf
        from holographic.io_and_interop import holographic_unicron as _U
        d = _tf.mkdtemp()
        with open(_os.path.join(d, "config.json"), "w") as _f:
            _json.dump(cfg.to_dict(), _f, default=str)
        _U.save_safetensors(_os.path.join(d, "model.safetensors"),
                            {k: np.ascontiguousarray(v) for k, v in weights.items()})
        cfg2 = config_from_json(_os.path.join(d, "config.json"), weights=weights)
        for _k in ("hidden", "n_layers", "n_heads", "n_kv_heads", "head_dim",
                   "linear_num_value_heads", "linear_num_key_heads",
                   "linear_key_head_dim", "linear_value_head_dim", "conv_kernel"):
            assert cfg2[_k] == rt.cfg[_k], (_k, cfg2[_k], rt.cfg[_k])
        rt2, _c = load_runtime(d)
        assert np.allclose(rt2.forward(ids), ours, atol=1e-8)
        # a wrong head_dim must RAISE, not run: the failure mode that costs days
        bad = dict(cfg2); bad["head_dim"] = cfg2["head_dim"] * 2
        try:
            # PASS THE DECLARED FLAG, as the loader does. Without it the
            # validator must INFER gating from the row count, and a doubled
            # head_dim makes a gated q_proj look exactly like a plain one -- the
            # hole this assertion exists to guard. Inference is for models whose
            # config says nothing; a config that speaks is believed.
            _validate_config(bad, weights, declared_gate=True)
            raise AssertionError("validator accepted a wrong head_dim")
        except ValueError as _e:
            assert "head_dim" in str(_e)
        bad2 = dict(cfg2); bad2["linear_num_key_heads"] = cfg2["linear_num_key_heads"] + 1
        try:
            _validate_config(bad2, weights, declared_gate=True)
            raise AssertionError("validator accepted wrong GDN head counts")
        except ValueError as _e:
            assert "qkvz" in str(_e)

        # SDM-radius attention: default OFF (bit-identical), and a tight radius
        # must degrade GRACEFULLY rather than catastrophically -- the property
        # that makes the redundancy exploitable at all.
        assert np.array_equal(rt.forward(ids), ours), "attn_top_k default changed behaviour"
        # SCREEN ROUTING: allowing every block must reproduce dense attention
        # EXACTLY. This null test is what caught a causal leak that made sparse
        # attention look BETTER than dense -- an impossibility, and therefore a
        # bug rather than a result.
        rt.cfg["attn_screen"] = {"block": 4, "blocks": 999, "window": len(ids)}
        assert np.max(np.abs(rt.forward(ids) - ours)) < 1e-9, "screen null test failed"
        rt.cfg["attn_screen"] = {"block": 4, "blocks": 1, "window": 4}
        routed = rt.forward(ids)
        rt.cfg.pop("attn_screen")
        assert float(np.mean(np.argmax(routed, -1) == np.argmax(ours, -1))) > 0.4
        assert np.array_equal(rt.forward(ids), ours), "screen flag leaked"

        rt.cfg["attn_top_k"] = 4
        sparse = rt.forward(ids)
        rt.cfg.pop("attn_top_k")
        agree_sparse = float(np.mean(np.argmax(sparse, -1) == np.argmax(ours, -1)))
        assert agree_sparse > 0.5, agree_sparse
        assert np.array_equal(rt.forward(ids), ours), "flag leaked after removal"

        # forward_embeds must be EXACTLY forward() when handed the same
        # embeddings it would have looked up -- otherwise every superposition
        # experiment measures the plumbing instead of the idea.
        emb_in = rt.embed[np.asarray(ids, np.int64)]
        assert np.max(np.abs(rt.forward_embeds(emb_in) - ours)) < 1e-9

        # SPLIT a/b LAYOUT: the real Qwen3.5-0.8B ships separate in_proj_a and
        # in_proj_b instead of a packed in_proj_ba. Rebuild the same weights in
        # that layout and demand IDENTICAL logits -- "handled" must mean equal,
        # not merely "runs without raising".
        w_split = dict(weights)
        _Kh = rt.cfg["linear_num_key_heads"]
        _Vh = rt.cfg["linear_num_value_heads"]
        _r = _Vh // _Kh
        for _k in [x for x in weights if x.endswith("in_proj_ba.weight")]:
            _W = np.asarray(weights[_k], np.float64)
            _Wr = _W.reshape(_Kh, 2 * _r, _W.shape[1])
            _pre = _k[:-len("in_proj_ba.weight")]
            w_split[_pre + "in_proj_b.weight"] = _Wr[:, :_r, :].reshape(_Vh, -1).copy()
            w_split[_pre + "in_proj_a.weight"] = _Wr[:, _r:, :].reshape(_Vh, -1).copy()
            del w_split[_k]
        for _k in [x for x in list(w_split) if x.endswith("in_proj_qkvz.weight")]:
            _W = np.asarray(w_split[_k], np.float64)
            _dk = rt.cfg["linear_key_head_dim"]; _dv = rt.cfg["linear_value_head_dim"]
            _Wr = _W.reshape(_Kh, 2 * _dk + 2 * _r * _dv, _W.shape[1])
            _pre = _k[:-len("in_proj_qkvz.weight")]
            w_split[_pre + "in_proj_qkv.weight"] = \
                _Wr[:, :2 * _dk + _r * _dv, :].reshape(-1, _W.shape[1]).copy()
            w_split[_pre + "in_proj_z.weight"] = \
                _Wr[:, 2 * _dk + _r * _dv:, :].reshape(-1, _W.shape[1]).copy()
            del w_split[_k]
        rt_split = GDNRuntime(w_split, dict(rt.cfg))
        assert np.max(np.abs(rt_split.forward(ids) - ours)) < 1e-9, \
            "fully split qkv/z/a/b layout differs from packed"
        _l1, _s1 = rt.prefill(ids); _l2, _s2 = rt_split.prefill(ids)
        assert np.max(np.abs(rt.step(5, _s1)[0] - rt_split.step(5, _s2)[0])) < 1e-9
        # and GDN layers must be identified by PRESENCE of linear_attn tensors,
        # not by one hard-coded name (the field bug: a real checkpoint got
        # routed to the attention path and died asking for a q_proj)
        assert rt_split._is_gdn(0) and not rt_split._is_gdn(rt.cfg["n_layers"] - 1)

        print("gdnruntime selftest OK -- logits match reference to rel %.1e; "
              "perplexity %.2f; residency hook live; cached==uncached over 10 "
              "tokens (%.1fx faster); rewind exact, branch diverges; "
              "config.json loader round-trips and rejects wrong shapes"
              % (err, p, t_slow / max(t_fast, 1e-9)))
    else:
        print("gdnruntime selftest SKIPPED-REFERENCE (torch/transformers not "
              "installed); internal contracts only")
        # minimal internal contract: rope roundtrip identity at position 0
        cos, sin = _rope_tables(8, np.array([0.0]), 10000.0)
        q = np.ones((1, 1, 16))
        q2, _ = _apply_rope(q, q, cos, sin)
        assert np.allclose(q2, q)


if __name__ == "__main__":
    _selftest()
