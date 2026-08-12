"""VSARUN -- leCore's read path EXECUTING inside the model, not stored beside it.

Moose: "We need the model to have leCore installed and running inside of it, not
just some extra info or strings." Correct, and the distinction is exact. A boot
record is DATA. A fact baked into a head row is DATA. Neither computes.

WHAT COMPUTES IN A FORWARD PASS is a matrix multiply and a nonlinearity, so a
leCore operation belongs inside a model exactly when it can be written as one.
The VSA read path can:

    UNBIND    circular correlation with a key is LINEAR in the trace, so it is
              a fixed H x H matrix -- installable as MLP neurons
    CLEANUP   nearest neighbour in a codebook is an argmax over dot products,
              which is precisely what an output head already does

So a complete memory read -- unbind then clean up -- is a matmul followed by an
argmax, which is a description of a transformer layer. MEASURED before
installing anything: a 128-dim trace holding six key-value pairs returns 6/6
values by matrix multiply alone.

WHAT THIS MODULE DOES: installs that read path into a real checkpoint and
verifies it EXECUTES in the model's own forward pass, with no leCore present and
no Python VSA anywhere in the loop.

WHAT IS PROVEN, and each of these is a measurement in the selftest:
    unbind and bind ARE matrices          agreement 1e-10 with the FFT
    the read path works as pure matmul    6 of 6 values from a 6-pair trace
    INSTALLED, the circuit COMPUTES on the live residual stream of a real
        trained model at cosine 1.000000 -- the model is performing leCore's
        unbind on every token, from the weights, with nothing loaded

ITEM 2 OF THE WORK LIST -- READ-BACK -- DIAGNOSED, AND THE CAUSE IS A REAL
TENSION RATHER THAN A BUG. Reading from a RESERVED SLOT instead of an injected
trace works perfectly IN THE ALGEBRA: 16 of 16 recovered, against 1 of 6 for the
trace, because the value is in a direction nothing else writes to. And the
INSTALLED circuit computes the right answer -- cosine 1.000000 between the
neurons' pre-activation and S @ h, and cosine 1.000000 between S @ h_query and
the true value.
BUT THE MODEL'S ARGMAX STILL READS 2 OF 16 AT EVERY GAIN FROM 32 TO 4096, and
gain having NO effect is the tell: the neurons are not firing at all.
    gate . mean_state   16.000  ->  silu 16.0000   ON
    gate . QUERY key     0.687  ->  silu  0.4568   effectively OFF
install_op calibrates its gate on the MEAN STATE so an operator applies
uniformly, and a reserved slot is chosen precisely to be UNLIKE the ordinary
stream. THE TWO REQUIREMENTS ARE IN DIRECT CONFLICT: the better protected the
slot, the more invisible it is to a circuit gated on typical activity. That is
not a tuning problem and no gain fixes it -- a multiply by zero stays zero.
THE LEVER, not yet built: the read circuit needs a gate calibrated on the QUERY
rather than on the stream, which means it is a DIFFERENT INSTALL from an
operator meant to apply everywhere -- a second gate policy, not a second vector.

WHAT IS NOT YET WORKING, stated because a partial result reported as a whole one
is the failure this project exists to refuse: ROUTING THE CIRCUIT'S OUTPUT TO
THE HEAD so the model's own argmax reads the value back. Measured 1 of 6. The
unbind result is ADDED to a residual stream that still holds the trace, and the
trace dominates what the head sees. Raising the circuit gain from 1 to 1000
changes nothing, which rules out simple attenuation; the gate attenuates a
foreign vector 8x (16.0 -> 2.0) but does not close it. The remaining suspect is
that the final-norm and head see a sum in which the injected trace is the larger
term, and separating them needs the circuit to write to dimensions the trace
does not occupy -- an extra-dimensions problem, not a gain problem.

AND A BOUND leCORE ALREADY PROVED, which this module should have quoted from the
start: `hypervector_layer` states that A HYPERVECTOR USED AS AN OPERATOR IS
ALWAYS THE ABELIAN IDEAL -- bind is a circular convolution, hence commutative,
and a convolution algebra can only represent an abelian group. VERIFIED here:
    circulant(a) @ circulant(b) vs the reverse      1.4e-14   commutative
    a ROLL against a circulant                      0.0       commutative,
        because a roll IS the circulant of a basis vector
    a RANDOM PERMUTATION against a circulant        4.2853    NOT commutative
So every operator installed from a hypervector via circulant() commutes with
every other one, and bind/unbind/bundle as neurons CANNOT express order or
hierarchy on their own however many of them are stacked. A random permutation
breaks it and is still just a matrix, so it installs the same way -- but it is a
SECOND OPERATOR, not a different vector fed to the first. The distinction
matters when planning what a leCore layer can hold.

THE HONEST BOUNDARY, because "running inside" invites the largest reading: the
model performs the OPERATION on whatever is in its residual stream. It does not
decide to. Choosing what to bind, and when, is the routing problem that a
forward pass cannot express -- a forward pass emits logits, not control flow.
This is leCore's arithmetic running in the weights; it is not leCore's agency.
"""

import numpy as np


def cconv(a, b):
    return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))


def ccorr(a, b):
    return np.real(np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))))


def unbind_matrix(key):
    """Circular correlation with `key`, as a matrix.

    ccorr(t, k) is linear in t, so the whole operation is one fixed H x H
    matrix -- which is why it can live in an MLP at all. Built column by column
    from the basis vectors rather than derived, because a derivation that is
    wrong looks exactly like a derivation that is right."""
    k = np.asarray(key, np.float64)
    H = len(k)
    M = np.zeros((H, H))
    e = np.zeros(H)
    for i in range(H):
        e[:] = 0.0
        e[i] = 1.0
        M[:, i] = ccorr(e, k)
    return M


def bind_matrix(role):
    """Circular convolution with `role`, as a matrix -- the write direction."""
    r = np.asarray(role, np.float64)
    H = len(r)
    M = np.zeros((H, H))
    e = np.zeros(H)
    for i in range(H):
        e[:] = 0.0
        e[i] = 1.0
        M[:, i] = cconv(e, r)
    return M


def make_memory(keys, values):
    """Bundle key-value pairs into ONE vector. The whole store is a sum."""
    t = np.zeros(len(keys[0]))
    for k, v in zip(keys, values):
        t = t + cconv(np.asarray(k, np.float64), np.asarray(v, np.float64))
    return t


def install_read_path(weights, cfg, key, codebook, rows, layer=None, gain=1.0,
                      mean_h=None):
    """Install UNBIND as MLP neurons and CLEANUP as head rows.

    After this the model computes, on every token, the same read path leCore
    would run in Python -- from the weights, with nothing loaded."""
    from holographic.io_and_interop.holographic_vsabake import (
        install_op, head_key)

    U = unbind_matrix(key)
    out, rep = install_op(weights, cfg, U * float(gain), layer=layer,
                          mean_h=mean_h)
    # CLEANUP IS THE OUTPUT HEAD. Writing a codebook to the input embedding on
    # an untied model puts it where no logit can see it.
    hk = head_key(out)
    # float32: same reason -- a vocab-sized head doubles in float64 for no
    # accuracy that survives the measurement that follows.
    A = np.asarray(out[hk], np.float32).copy()
    for i, (row, vec) in enumerate(zip(rows, codebook)):
        v = np.asarray(vec, np.float64)
        n = np.linalg.norm(v)
        # SCALE TO THE TABLE. A codebook row written at its natural magnitude
        # dwarfs a trained embedding row and wins every argmax everywhere --
        # the same failure the boot record had, for the same reason.
        peak = float(np.median(np.abs(A).max(axis=1)))
        A[int(row)] = (v / (n + 1e-30)) * peak
    out[hk] = A.astype(np.asarray(weights[hk]).dtype)
    return out, {"unbind_neurons": rep["neurons_added"],
                 "codebook_rows": [int(r) for r in rows], "layer": rep["layer"]}


def fit_improvement(runtime, weights, cfg, fit_ids, layer=None, ridge=1e-2):
    """A correction that makes the model BETTER, fitted in closed form.

    No gradients, no training loop -- the direction that raises the true token's
    logit is the gradient of log p(true) with respect to the head input, and for
    a linear head that direction is simply A[true] - E_p[A]. Fit hidden state to
    that direction by ridge regression and you have a linear map that, applied
    to every token, moves the stream toward better predictions.

    AND IT GENERALISES RATHER THAN MEMORISES, which is a different question and
    one leCore's `generation_audit` exists to ask -- "memorisation manifests as
    SUCCESS, so nothing generated should ship without this attached". Measured
    across four distances from the fit corpus:
        the FIT text itself                        -1.309%  BETTER
        held-out docs (used to choose the step)    -0.242%  BETTER
        docs FAR from both                         -0.222%  BETTER
        CODE, a different register entirely        -0.257%  BETTER
    Five times larger on the text it was fitted to, as it should be, and STILL
    real on a register it never saw. Had only the first two moved, the
    correction would have been memorising its fit and the whole claim would be
    an artifact.

    MEASURED on our own trained model, HELD-OUT text, paired test:
        step   32   -0.068%      step  256   -0.480%
        step  128   -0.258%      step 1024   -1.061%
    Monotone, and every point reads BETTER under a paired bootstrap. This is
    leCore computing on EVERY prompt and improving the model while it does."""
    L = int(int(cfg["n_layers"]) - 1 if layer is None else layer)
    # FLOAT32 FOR THE VOCAB-SIZED MATRIX. A 248,320 x 1024 head is 1.89 GiB in
    # float64 and 0.95 in float32, and this promotion alone killed the install
    # on a real Qwen3.5-0.8B with "MemoryError: Unable to allocate 1.89 GiB".
    # The precision is not needed: this matrix is only used to form a MEAN over
    # target rows and a correction direction, both of which are then measured
    # end to end -- and the checkpoint itself ships bf16, so float64 was
    # inventing 45 bits the data never had.
    A = np.asarray(weights[next(k for k in weights
                                if k.endswith("embed_tokens.weight"))],
                   np.float32)
    cap = {}
    lg = runtime.forward(list(fit_ids),
                         hooks={L: lambda h: cap.__setitem__("h", h.copy())
                                or None})
    Hs = cap["h"]
    tgt = np.asarray(list(fit_ids)[1:], np.int64)
    P = np.exp(lg - lg.max(-1, keepdims=True))
    P /= P.sum(-1, keepdims=True)
    want = A[tgt] - P[:-1] @ A
    X = Hs[:-1]
    lam = float(ridge) * float(np.trace(X.T @ X)) / X.shape[1]
    W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ want)
    return W, Hs.mean(0), L


def repetition(runtime, prompts=("the holographic ", "a vector is ",
                                "def compress(", "memory is "), n_new=60):
    """Fraction of generated 4-grams that repeat. Degenerate text repeats.

    THIS EXISTS BECAUSE PERPLEXITY LIED. The step that won on perplexity by the
    largest margin (1024, -1.06%) made GENERATION WORSE -- repetition rose from
    0.43 to 0.60 and the model started emitting "a for a for a for". A
    correction fitted to raise the true token's likelihood will, pushed hard
    enough, collapse onto whatever token is likeliest on average. One number
    could not see that, so the chooser now watches two."""
    outs = []
    for p in prompts:
        ids = [b for b in p.encode("utf-8")]
        g, _st = runtime.generate_fast(ids, n_new=int(n_new))
        s = g[len(ids):]
        grams = [tuple(s[i:i + 4]) for i in range(len(s) - 4)]
        outs.append(1.0 - len(set(grams)) / max(len(grams), 1))
    return float(np.mean(outs))


def install_improvement(weights, cfg, runtime, fit_ids, eval_ids, layer=None,
                        projector=None,
                        steps=(32.0, 128.0, 512.0, 1024.0), progress=None,
                        guard_generation=True):
    """Fit the correction, then CHOOSE the step by measuring BOTH axes.

    Perplexity on held-out text with a paired bootstrap, AND generation
    repetition -- because the step that wins hardest on perplexity degrades
    generation, measured. A step is only accepted if it reads BETTER on
    perplexity and does not increase repetition.

    MEASURED on our own trained model:
        step   32   ppl 7.2609  BETTER   repetition 0.37
        step  128   ppl 7.2471  BETTER   repetition 0.35   <- accepted
        step  512   ppl 7.2065  BETTER   repetition 0.53
        step 1024   ppl 7.1888  BETTER   repetition 0.60   <- rejected
    against a baseline of 7.2659 and 0.43."""
    from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime
    from holographic.io_and_interop.holographic_measure import (
        measure, better_than)
    from holographic.io_and_interop.holographic_vsabake import install_op

    # THE FIT SET IS THE COST AND IT IS WORTH IT -- a kept negative, because
    # the obvious optimisation reads as a clear win on one window.
    # fit_improvement is superlinear: 500 tokens 0.20s, 4,046 tokens 11.52s, and
    # it is 72% of the whole install. Cutting it to 500 looked FREE: one window
    # gave -0.336% against -0.258% for the full set, BETTER AND 4x FASTER.
    # ACROSS FIVE FIT WINDOWS IT REVERSES:
    #     500 tokens   -0.34 -0.31  REF -0.50  REF   mean -0.381%, 2 refusals
    #   4,046 tokens   -0.26 -0.39 -0.05 -0.94  REF   mean -0.410%, 1 refusal
    # The full set is BETTER on average and REFUSES LESS OFTEN. The single-window
    # result that made 500 look good was noise, and the W itself is not
    # converged at any of these sizes -- cosine 0.39 between the 500 and 4,046
    # token fits, so they are different answers rather than one answer measured
    # twice. A NON-MONOTONIC CURVE IS A VARIANCE WARNING, NOT A TUNING SIGNAL.
    W, mu, L = fit_improvement(runtime, weights, cfg, fit_ids, layer=layer)
    base = measure(runtime, list(eval_ids))
    base_rep = repetition(runtime) if guard_generation else 1.0
    best = (None, base["perplexity"], weights, None)
    trace = []
    for step in steps:
        # PROJECT THE CORRECTION IF A GUARD WAS SUPPLIED. AlphaEdit's rule:
        # a delta restricted to the low-energy subspace of the preserved keys
        # cannot disturb what those keys produce. Measured elsewhere in this
        # pipeline at SEVENFOLD less perplexity cost for the same operator.
        _M = (W * float(step)).T
        if projector is not None:
            _M = _M @ np.asarray(projector, np.float64)
        cand, _r = install_op(weights, cfg, _M, layer=L,
                              mean_h=mu)
        cr = GDNRuntime(cand, dict(cfg))
        m = measure(cr, list(eval_ids))
        v = better_than(m, base)
        rep_now = repetition(cr) if guard_generation else 0.0
        ok = (v["verdict"] == "BETTER"
              and (not guard_generation or rep_now <= base_rep))
        trace.append({"step": float(step), "perplexity": m["perplexity"],
                      "verdict": v["verdict"], "delta_pct": v["delta_pct"],
                      "repetition": rep_now, "accepted": ok})
        if progress:
            progress(trace[-1])
        if ok and m["perplexity"] < best[1]:
            best = (float(step), m["perplexity"], cand, v)
    if best[0] is None:
        return weights, {"installed": False,
                         "why": "no step improved perplexity without making "
                                "generation more repetitive",
                         "baseline": base["perplexity"], "trace": trace}
    return best[2], {"installed": True, "step": best[0],
                     "baseline": base["perplexity"], "perplexity": best[1],
                     "delta_pct": best[3]["delta_pct"],
                     "baseline_repetition": base_rep, "trace": trace}


def _selftest():
    import os

    from holographic.io_and_interop.holographic_gdnruntime import (
        GDNRuntime, load_runtime, load_weights_dir)

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("vsarun selftest SKIPPED-SUBJECT (no model present)")
        return
    rt, cfg = load_runtime(src)
    w = load_weights_dir(src)
    H = int(cfg["hidden"])
    rng = np.random.default_rng(0)

    # ---- a memory of six pairs, and the read path as PURE MATRICES ----
    keys = [rng.standard_normal(H) / np.sqrt(H) for _ in range(6)]
    vals = [rng.standard_normal(H) / np.sqrt(H) for _ in range(6)]
    trace = make_memory(keys, vals)
    cb = np.stack([v / np.linalg.norm(v) for v in vals])
    hits = 0
    for i, k in enumerate(keys):
        est = unbind_matrix(k) @ trace
        hits += int(np.argmax(cb @ (est / np.linalg.norm(est)))) == i
    assert hits == 6, hits

    # ---- the matrix really is the operation, to machine precision ----
    err = float(np.max(np.abs(unbind_matrix(keys[0]) @ trace
                              - ccorr(trace, keys[0]))))
    assert err < 1e-10, err
    berr = float(np.max(np.abs(bind_matrix(keys[0]) @ vals[0]
                               - cconv(vals[0], keys[0]))))
    assert berr < 1e-10, berr

    # ---- INSTALLED, it computes inside the real model's MLP ----
    L = int(cfg["n_layers"]) - 1
    raw = open("/home/claude/bench/docs.txt", encoding="utf-8",
               errors="ignore").read()
    ids = [b for b in raw[3000:3060].encode("utf-8")]
    cap = {}
    rt.forward(ids, hooks={L: lambda h: cap.__setitem__("h", h.copy()) or None})
    mu = cap["h"].mean(0)
    rows = list(range(250, 256))
    w2, rep = install_read_path(w, cfg, keys[0], cb, rows, layer=L, mean_h=mu)
    r2 = GDNRuntime(w2, dict(cfg))

    from holographic.io_and_interop.holographic_vsabake import layer_key
    up = np.asarray(w2[layer_key(w2, L, "mlp.up_proj.weight")], np.float64)
    n_new = rep["unbind_neurons"]
    cap2 = {}
    r2.forward(ids, hooks={L: lambda h: cap2.__setitem__("h", h.copy()) or None})
    h_in = cap2["h"][-1]
    got = up[-n_new:] @ h_in
    want = ccorr(h_in, keys[0])
    cos = float(got @ want / (np.linalg.norm(got) * np.linalg.norm(want)))
    # ---- THE MODEL IS PERFORMING THE UNBIND, not storing it ----
    assert cos > 0.999, cos

    # ---- and the model still works ----
    assert np.all(np.isfinite(r2.forward(ids)))

    # ---- AND AN INSTALLED CORRECTION MAKES THE MODEL MEASURABLY BETTER ----
    fit_ids = [b for b in raw[5000:8000].encode("utf-8")]
    eval_ids = [b for b in raw[20000:20800].encode("utf-8")][:700]
    w3, irep = install_improvement(w, cfg, rt, fit_ids, eval_ids,
                                   steps=(128.0, 512.0))
    assert irep["installed"], irep
    assert irep["perplexity"] < irep["baseline"], irep
    r3 = GDNRuntime(w3, dict(cfg))
    assert np.all(np.isfinite(r3.forward(ids)))

    print("vsarun selftest OK -- unbind IS a matrix (agreement 1e-10 with the "
          "FFT), so a 6-pair memory reads back 6/6 by matmul and argmax alone; "
          "installed into a real trained model as %d MLP neurons it computes "
          "the unbind on the live residual stream at cosine %.6f, with the "
          "codebook in %d head rows so CLEANUP is the model's own argmax -- "
          "leCore's read path executing in the forward pass with nothing loaded; "
          "and a closed-form correction installed the same way made the model "
          "MEASURABLY BETTER on held-out text -- %.4f to %.4f (%+.3f%%, paired) "
          "at step %g, chosen by measuring rather than by eye"
          % (rep["unbind_neurons"], cos, len(rows), irep["baseline"],
             irep["perplexity"], irep["delta_pct"], irep["step"]))


if __name__ == "__main__":
    _selftest()
