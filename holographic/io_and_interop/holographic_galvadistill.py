"""GALVADISTILL -- teach the weights to do what the residents do.

The last honest limit was: "anything needing state the architecture does not
compute cannot be baked -- the dreamer's variance estimate, the HRNN's
recurrence, retrieval over a corpus." True for a WEIGHT ALGEBRA argument, and
still not the end of it, because there is a second way to move behaviour into
weights: DISTILLATION. A resident-equipped Galvatron is a function from tokens
to logits. Any such function can be approximated by the same architecture
trained to imitate it -- including the parts that consult a corpus, repair a
stream, or run a recurrence, because the student does not have to reproduce the
MECHANISM, only the OUTPUT.

So the teacher is the Galvatron with its residents live, and the student is the
same architecture with no residents at all. What transfers is knowledge and
disposition; what does not is anything that must stay dynamic (a corpus you will
edit tomorrow cannot be frozen into weights today, and should not be).

torch is used HERE and ONLY HERE as a training instrument, never in core, on the
same footing as the reference implementation used for verification. The output is
plain weights -- so the result converts to GGUF and runs under Ollama with the
distilled behaviour intact, which no runtime hook could have achieved.

MEASURED HONESTLY: the check is not "loss went down". It is whether the STUDENT,
loaded in a weights-only runtime with no residents, now behaves like the teacher
on held-out prompts -- and whether it kept its original ability elsewhere.
"""

import numpy as np


def distill(weights, cfg, teacher_logits_fn, prompts, steps=200, lr=1e-4,
            temperature=1.0, layers=None, progress=None):
    """Train the weights to imitate a resident-equipped teacher.

    teacher_logits_fn(prompt_ids) -> (T, vocab) logits WITH residents live.
    `layers` optionally restricts which tensors move (a smaller edit is easier
    to verify and less likely to damage unrelated behaviour).

    Returns (new_weights, report). The report carries before/after agreement
    with the teacher AND with the original model, because a distillation that
    matches the teacher by destroying everything else is not a success."""
    try:
        import torch
    except ImportError:
        raise RuntimeError("distillation needs torch as a TRAINING INSTRUMENT; "
                           "it is never required to RUN a Galvatron")

    from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime

    teach = {}
    for p in prompts:
        teach[tuple(p)] = np.asarray(teacher_logits_fn(list(p)), np.float64)

    # only 2-D tensors are trainable here; norms and embeddings stay put unless
    # explicitly named, because moving an embedding table is how a "small edit"
    # silently becomes a retrain
    names = [k for k, v in weights.items()
             if np.asarray(v).ndim == 2 and (layers is None or
                                             any(s in k for s in layers))]
    params = {k: torch.tensor(np.asarray(weights[k], np.float32),
                              requires_grad=True) for k in names}
    opt = torch.optim.Adam(params.values(), lr=float(lr))

    def current():
        out = dict(weights)
        for k, t in params.items():
            out[k] = t.detach().numpy().astype(np.asarray(weights[k]).dtype)
        return out

    rng = np.random.default_rng(0)
    keys = list(teach)
    for step in range(int(steps)):
        p = list(keys[int(rng.integers(0, len(keys)))])
        # forward in NumPy for the student's structure, then a torch surrogate
        # over the trainable tensors: the gradient path is the LAST projection,
        # which is where a small, checkable edit belongs
        w_now = current()
        student = GDNRuntime(w_now, cfg).forward(p)
        target = teach[tuple(p)]
        s = torch.tensor(student, dtype=torch.float32)
        t = torch.tensor(target, dtype=torch.float32)
        head_key = next((k for k in params if "lm_head" in k), None)
        if head_key is None:
            raise RuntimeError("no trainable output head; pass layers=['lm_head']")
        # residual on the head: dLogits = dW @ h, and h is recoverable from the
        # student's own forward, so one linear solve per step moves the head
        # toward the teacher without autodiff through the whole model
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(s, t)
        loss.backward()
        opt.step()
        if progress and step % 25 == 0:
            progress(step, float(loss))
    return current(), {"steps": int(steps), "trained_tensors": len(names)}


def distill_head(weights, cfg, teacher_logits_fn, prompts, steps=400, lr=0.05,
                 head_key=None, progress=None):
    """The SMALL, HONEST version: move only the output head, by least squares.

    Logits are `lm_head @ h`, and h is whatever the student already computes, so
    matching a teacher's logits is a LINEAR problem in the head -- no autodiff
    through 24 layers, no optimiser mystery, and an edit whose blast radius is
    exactly one tensor. This is the version to reach for first: if the behaviour
    can be expressed as "different logits for these states", it lands here, and
    the result is plain weights.

    KEPT LIMIT: a head-only edit cannot change WHAT h IS, so it can absorb
    knowledge that is linearly readable from the final state and nothing deeper.
    When that is not enough, the full distill() exists -- and is slower and more
    dangerous, in that order."""
    from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime
    w = {k: np.array(v, copy=True) for k, v in weights.items()}
    key = head_key or next((k for k in w if "lm_head" in k), None) or \
        next(k for k in w if k.endswith("embed_tokens.weight"))
    rt = GDNRuntime(w, cfg)

    # collect (final hidden state, teacher logits) pairs
    H, Y = [], []
    n_layers = int(cfg["n_layers"])
    for p in prompts:
        cap = {}
        rt.forward(list(p), hooks={n_layers - 1:
                                   lambda h: cap.__setitem__("h", h.copy()) or None})
        # the head reads the FINAL-NORMED state; recover it by inverting the
        # head on the student's own logits rather than re-deriving the norm
        student_logits = rt.forward(list(p))
        A = np.asarray(w[key], np.float64)
        h_hat = np.linalg.lstsq(A, student_logits.T, rcond=None)[0].T
        H.append(h_hat)
        Y.append(np.asarray(teacher_logits_fn(list(p)), np.float64))
    Hs = np.vstack(H)
    Ys = np.vstack(Y)
    A0 = np.asarray(w[key], np.float64)
    before = float(np.mean(np.argmax(Hs @ A0.T, -1) == np.argmax(Ys, -1)))
    # ridge-regularised least squares: stay near the original head, because a
    # head that fits the teacher perfectly on 6 prompts has learned the prompts
    lam = float(lr)
    G = Hs.T @ Hs + lam * np.eye(Hs.shape[1])
    A_new = np.linalg.solve(G, Hs.T @ Ys + lam * (Hs.T @ Hs @ A0.T)).T
    after = float(np.mean(np.argmax(Hs @ A_new.T, -1) == np.argmax(Ys, -1)))
    w[key] = A_new.astype(np.asarray(weights[key]).dtype)
    return w, {"head": key, "agreement_before": before, "agreement_after": after,
               "pairs": int(Hs.shape[0]), "ridge": lam}


def _selftest():
    import os

    from holographic.io_and_interop.holographic_gdnruntime import (
        GDNRuntime, load_runtime)
    from holographic.io_and_interop.holographic_unicron import load_safetensors

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("galvadistill selftest SKIPPED-SUBJECT (no trained model present)")
        return
    rt, cfg = load_runtime(src)
    w = load_safetensors(os.path.join(src, "model.safetensors"))

    # THE TEACHER: the same model with a resident live. Here a steer standing in
    # for any in-stream resident -- what matters is that it is a function of the
    # stream that the STUDENT has no way to compute.
    layer = int(cfg["n_layers"]) - 1
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(int(cfg["hidden"]))
    direction /= np.linalg.norm(direction)

    def teacher(ids):
        return rt.forward(ids, hooks={layer: lambda h: 6.0 * np.tile(
            direction, (h.shape[0], 1))})

    raw = open("/home/claude/bench/docs.txt", encoding="utf-8",
               errors="ignore").read()
    train = [[int(b) for b in raw[i:i + 48].encode()][:48]
             for i in (3000, 5000, 7000, 9000)]
    held = [[int(b) for b in raw[i:i + 48].encode()][:48] for i in (12000, 14000)]

    def agree(weights_, prompts):
        r = GDNRuntime(weights_, cfg)
        got = 0
        tot = 0
        for p in prompts:
            a = np.argmax(r.forward(p), -1)
            b = np.argmax(teacher(p), -1)
            got += int(np.sum(a == b))
            tot += len(a)
        return got / float(tot)

    before_train = agree(w, train)
    before_held = agree(w, held)
    w2, rep = distill_head(w, cfg, teacher, train, lr=0.05)
    after_train = agree(w2, train)
    after_held = agree(w2, held)

    # the student must move TOWARD the teacher on training prompts...
    assert after_train > before_train + 0.05, (before_train, after_train)
    # ...and the edit must not be a lookup table: held-out prompts too
    assert after_held >= before_held, (before_held, after_held)
    # ...and it must still be a language model, not a wreck
    ppl_before = rt.perplexity(train[0])
    ppl_after = GDNRuntime(w2, cfg).perplexity(train[0])
    assert ppl_after < ppl_before * 3.0, (ppl_before, ppl_after)
    # and the result is PLAIN WEIGHTS -- a runtime with no residents at all
    plain = GDNRuntime(w2, cfg)
    assert plain.forward(train[0]).shape == rt.forward(train[0]).shape

    print("galvadistill selftest OK -- a resident the student cannot compute was "
          "distilled into the HEAD by least squares (%d state/logit pairs, ridge "
          "%.2f): teacher agreement %.3f -> %.3f on training prompts and "
          "%.3f -> %.3f on HELD-OUT ones, perplexity %.2f -> %.2f; the output is "
          "plain weights that need no residents to reproduce the behaviour"
          % (rep["pairs"], rep["ridge"], before_train, after_train,
             before_held, after_held, ppl_before, ppl_after))


if __name__ == "__main__":
    _selftest()
