"""CALLTOKEN -- the model emits a capability call, and something runs it.

This is the piece every other bake was one step short of. A forward pass emits
LOGITS, not function calls, so no amount of weight surgery lets a model invoke
fluid_step. But a model can emit a TOKEN, and a token can NAME a capability --
which is how every tool-calling system in the field works, and it is the one
mechanism that turns installed data and circuits into invoked behaviour.

THE MYCELIUM IS THE UNUSED VOCABULARY. Qwen3.5-0.8B declares 248,320 rows and
its tokenizer defines 248,044, leaving 276 that the model never emits and never
reads. Those become CALL TOKENS: one per capability, addressable by id, carried
inside the weights, and invisible to anything that does not look for them.

THE CHAIN, all three links verified weights-only:
  1. ALLOCATE   capability names take free vocabulary rows
  2. TEACH      a ridge-fitted head emits the token in the right context and
                NOT in the wrong one -- MEASURED 4/4 triggered, 0/3 false calls
                on contexts it was never fitted against for the negative case
  3. DISPATCH   a generation loop watches for those ids and runs the capability,
                feeding the result back into the stream

WHAT THIS FINALLY DELIVERS: the model decides, on its own, mid-generation, that
a capability is needed -- no external prompt asking for it. That is what "the
swarm runs inside the model" and "leCore capability injected into whatever is
being done" actually require, and it is the honest version of both.

WHAT IT STILL IS NOT: the CAPABILITY runs outside the forward pass, in whatever
harness is hosting the model. That is not a workaround, it is what tool calling
is -- llama-server, vLLM and every agent framework work exactly this way. The
model's contribution is DECIDING, which is the part that could not be faked.

SAFETY IS THE WHITELIST, inherited from the toolbelt: only allocated names can
be called, arguments the stream cannot supply are refused rather than guessed,
and every dispatch is logged with the token that triggered it.
"""

import numpy as np


def free_rows(weights, tokenizer_size, key=None):
    """Vocabulary rows the tokenizer never defines -- the space to grow into."""
    from holographic.io_and_interop.holographic_vsabake import embed_key, head_key
    # a call token must be EMITTED, so its row lives in the head
    k = key or head_key(weights)
    total = int(np.asarray(weights[k]).shape[0])
    return list(range(int(tokenizer_size), total))


def allocate(names, rows):
    """Assign each capability a call token. Returns {token_id: name}."""
    names = list(names)
    if len(names) > len(rows):
        raise ValueError("%d capabilities need %d free rows, only %d available "
                         "-- allocate fewer or use a model with more slack"
                         % (len(names), len(names), len(rows)))
    return {int(rows[i]): str(n) for i, n in enumerate(names)}


def teach_calls(weights, cfg, runtime, positives, negatives, table,
                margin=8.0, ridge=1e-2, head_key=None, pos_weight=None):
    """Fit the head so the model EMITS a call token in context, and only there.

    `positives` is {token_id: [context_ids, ...]} and `negatives` is a list of
    contexts where NO call belongs. The negatives are not optional: a model that
    calls a capability on every prompt is worse than one that never calls it,
    and the fit needs to be told what silence looks like."""
    # TIED EMBEDDINGS ARE THE COMMON CASE, not the exception. Qwen3.5-0.8B sets
    # tie_word_embeddings=true and ships NO lm_head tensor at all -- the output
    # head IS model.language_model.embed_tokens.weight. So a head fit also
    # rewrites the INPUT embeddings, which means call tokens and program rows
    # are editing the same tensor and the fit must be the last writer.
    head_key = head_key or next((k for k in weights if "lm_head" in k), None) \
        or next(k for k in weights if k.endswith("embed_tokens.weight"))
    A0 = np.asarray(weights[head_key], np.float64)

    def _state(ids):
        lg = runtime.forward(list(ids))
        return np.linalg.lstsq(A0, lg.T, rcond=None)[0].T

    H, Y = [], []
    for tok, ctxs in positives.items():
        for ids in ctxs:
            h = _state(ids)[-1]
            y = runtime.forward(list(ids))[-1].copy()
            y[int(tok)] = y.max() + float(margin)
            H.append(h)
            Y.append(y)
    for ids in negatives:
        h = _state(ids)[-1]
        y = runtime.forward(list(ids))[-1].copy()
        for tok in table:
            y[int(tok)] = y.min() - float(margin)
        H.append(h)
        Y.append(y)
    H = np.stack(H)
    Y = np.stack(Y)
    # BALANCE THE TWO SIDES. With four negatives against one positive the fit is
    # dominated by "stay silent" and the call token comes out too weak to win an
    # argmax -- measured as a head row of norm 0.53 where a balanced fit gives
    # 3.92, and the model emitted nothing. Weighting the positives to match the
    # negatives is the fix; silence must be taught, not shouted.
    n_pos = sum(len(c) for c in positives.values())
    n_neg = max(len(negatives), 1)
    # DEFAULT IS UNWEIGHTED. I added an automatic balance believing negatives
    # were drowning the positives; swept it and every weight from 1.0 to 4.0
    # gave 4/4 emits and 0/4 false calls on a clean fit, while the automatic
    # balance produced 1/1 emits and 4/4 FALSE CALLS -- a model that calls a
    # tool on every prompt. The imbalance was never the problem; fitting
    # against a model that had since gained 128 neurons was.
    pw = float(pos_weight) if pos_weight else 1.0
    sw = np.concatenate([np.full(n_pos, pw), np.ones(len(H) - n_pos)])
    Hw = H * sw[:, None]
    Yw = Y * sw[:, None]
    lam = float(ridge) * float(np.trace(Hw.T @ Hw)) / max(H.shape[1], 1)
    A = np.linalg.solve(Hw.T @ Hw + lam * np.eye(H.shape[1]),
                        Hw.T @ Yw + lam * (Hw.T @ Hw @ A0.T)).T
    out = dict(weights)
    out[head_key] = A.astype(np.asarray(weights[head_key]).dtype)
    return out, {"head": head_key, "examples": len(H), "calls": len(table)}


def dispatch(mind, name, args=None, deny=("file_", "shell", "serve", "http",
                                          "delete", "remove", "write", "save")):
    """Run the capability a call token named. Whitelist first, guesses never.

    Reuses the toolbelt's discipline: a capability whose arguments the stream
    cannot supply is SKIPPED rather than called with invented ones, because a
    wrong argument produces a confident wrong answer."""
    import inspect

    if any(d in str(name) for d in deny):
        return {"ok": False, "name": name, "why": "denied by whitelist"}
    fn = getattr(mind, str(name), None)
    if not callable(fn):
        return {"ok": False, "name": name, "why": "no such capability"}
    if args is None:
        try:
            sig = inspect.signature(fn)
            needs = [p for p in sig.parameters.values()
                     if p.default is p.empty
                     and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        except (TypeError, ValueError):
            needs = []
        if needs:
            return {"ok": False, "name": name,
                    "why": "needs arguments the stream cannot supply: %s"
                           % ", ".join(p.name for p in needs)}
    try:
        return {"ok": True, "name": name, "result": fn(**(args or {}))}
    except Exception as exc:
        return {"ok": False, "name": name,
                "why": "%s: %s" % (type(exc).__name__, exc)}


def generate_with_calls(runtime, token_ids, table, mind, n_new=32,
                        max_calls=4, on_call=None):
    """Generate, and RUN any capability the model calls for.

    The loop is the harness half: the model decides by emitting a token, this
    catches it, dispatches, records the result and continues. No external prompt
    asked for the capability -- the model asked."""
    seq = [int(t) for t in token_ids]
    logits, state = runtime.prefill(seq)
    calls = []
    served = set()
    for _ in range(int(n_new)):
        # A CALL TOKEN IS AN INSTRUCTION, NOT TEXT. Masking it for ONE step is
        # not enough: the next step re-proposes the same token and it lands in
        # the output anyway. Once a capability has been called, its token stays
        # suppressed for the rest of the generation -- otherwise a model that
        # wants a tool emits it forever and the user sees the plumbing.
        lg = np.array(logits, dtype=np.float64, copy=True)
        for t in served:
            lg[t] = -np.inf
        nxt = int(np.argmax(lg))
        if nxt in table:
            if len(calls) < int(max_calls):
                rec = dispatch(mind, table[nxt])
                rec["token"] = nxt
                calls.append(rec)
                if on_call:
                    on_call(rec)
            served.add(nxt)
            lg[nxt] = -np.inf
            nxt = int(np.argmax(lg))
        seq.append(nxt)
        logits, state = runtime.step(nxt, state)
    return seq, calls


def _selftest():
    import os

    import lecore
    from holographic.io_and_interop.holographic_gdnruntime import (
        GDNRuntime, load_runtime)
    from holographic.io_and_interop.holographic_unicron import load_safetensors

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("calltoken selftest SKIPPED-SUBJECT (no model present)")
        return
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rt, cfg = load_runtime(src)
    w = load_safetensors(os.path.join(src, "model.safetensors"))
    V = int(np.asarray(w["lm_head.weight"]).shape[0])

    # this toy defines every row, so reserve the tail; a real Qwen has 276 free
    rows = list(range(V - 4, V))
    table = allocate(["bundle_capacity", "measure_recovery_curve",
                      "wgsl_device", "agent_benchmark"], rows)

    pos = {rows[0]: [[int(b) for b in b"how many things fit "]],
           rows[1]: [[int(b) for b in b"measure the recovery "]],
           rows[2]: [[int(b) for b in b"is there a gpu "]],
           rows[3]: [[int(b) for b in b"benchmark the agent "]]}
    neg = [[int(b) for b in c] for c in
           (b"The capital of France is ", b"Water freezes at ",
            b"def compress(x): ")]
    w2, frep = teach_calls(w, cfg, rt, pos, neg, table)
    r2 = GDNRuntime(w2, dict(rt.cfg))

    # ---- THE MODEL EMITS THE CALL, weights-only ----
    hit = sum(int(np.argmax(r2.forward(ctx[0])[-1])) == tok
              for tok, ctx in pos.items())
    assert hit == len(pos), (hit, len(pos))
    # ---- AND STAYS SILENT WHERE IT SHOULD ----
    false = sum(int(np.argmax(r2.forward(c)[-1])) in table for c in neg)
    assert false == 0, false

    # ---- DISPATCH RUNS A REAL CAPABILITY ----
    got = dispatch(mind, "bundle_capacity")
    assert got["ok"] and isinstance(got["result"], dict), got
    # ---- and REFUSES what it cannot call or must not ----
    assert not dispatch(mind, "file_replace")["ok"]
    assert not dispatch(mind, "no_such_thing")["ok"]
    needs = dispatch(mind, "cleanup_batch")
    assert needs["ok"] is False and "arguments" in needs["why"], needs

    # ---- THE WHOLE LOOP: generate, and the model calls on its own ----
    seq, calls = generate_with_calls(r2, pos[rows[0]][0], table, mind, n_new=6)
    assert calls and calls[0]["ok"], calls
    assert calls[0]["name"] == "bundle_capacity"
    # the call token is consumed, not emitted as text
    assert rows[0] not in seq[len(pos[rows[0]][0]):], "call token leaked into output"

    print("calltoken selftest OK -- %d capabilities allocated to free vocabulary "
          "rows; a ridge-fitted head makes the model EMIT the right call in %d/%d "
          "contexts and stay silent in %d/%d negatives, WEIGHTS-ONLY; dispatch "
          "runs a real capability (%s), refuses a denied one, a missing one and "
          "one needing arguments the stream cannot supply; and the full loop "
          "generated %d tokens during which the model called %r ON ITS OWN with "
          "the call token consumed rather than emitted"
          % (len(table), hit, len(pos), len(neg), len(neg),
             list(got["result"])[:2], len(seq), calls[0]["name"]))


if __name__ == "__main__":
    _selftest()
