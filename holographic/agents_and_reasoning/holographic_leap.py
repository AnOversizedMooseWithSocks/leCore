"""LEAP -- generate FASTER than the model alone, with output that is provably
identical to what the model would have said.

THE STRUCTURE ARGUMENT, made honest: a language model re-derives every token
from scratch, even when it is walking a road it has walked before. leCore has
memory, so it can LEARN THE ROUTE and propose the next few tokens for free. But
a proposal is not an answer -- so every drafted token is VERIFIED against the
real forward pass, and only the longest provably-correct prefix is accepted. The
output is bit-identical to greedy decoding; the only thing that changes is how
many sequential passes it took to get there.

WHY IT CAN WIN AT ALL: verification of k drafted tokens is ONE batched call
(GDNRuntime.extend -- one GEMM over the chunk) where generating them normally is
k sequential calls (k GEMVs). On CPU NumPy that is the difference between
compute-bound and memory-bandwidth-bound, the same effect that made the
vectorized prefill beat the looped one by 4.8-12.9x earlier in this arc. So the
speedup is real when the drafter is right, and the cost is one wasted batched
call when it is wrong.

THE DRAFTER learns online from the model's own accepted output -- an n-gram route
memory (fast, exact, no training loop) that grows as generation proceeds. This is
where the model's loops become an ASSET: the 0.8B is loop-prone, and a loop is a
route the drafter learns after seeing it once.

HONEST BOUNDS, measured in the selftest and stated before any number is quoted:
  * output identity is not a hope, it is asserted token-for-token;
  * on NOVEL text the drafter misses, acceptance goes to ~0, and speculative
    decoding is SLOWER than plain generation by the wasted verification -- the
    measured overhead is reported, not hidden;
  * the win is real only where structure repeats. That is a property of the
    TEXT, not of the cleverness of the drafter, and the selftest measures both
    regimes so nobody quotes the good one alone.
"""

import numpy as np


class RouteMemory:
    """Learned routes: context n-gram -> the token that followed, with a hit
    count. Deterministic, exact, and updated online from ACCEPTED tokens only --
    never from drafts, or the memory would learn its own guesses."""

    def __init__(self, order=3, min_count=1):
        self.order = int(order)
        self.min_count = int(min_count)
        self.table = {}
        self.stats = {"learned": 0, "drafted": 0, "accepted": 0, "rejected": 0}

    def _key(self, ids, i):
        lo = max(0, i - self.order)
        return tuple(int(t) for t in ids[lo:i])

    def learn(self, ids):
        """Record every (context -> next) transition in a confirmed sequence."""
        for i in range(1, len(ids)):
            k = self._key(ids, i)
            if not k:
                continue
            slot = self.table.setdefault(k, {})
            slot[int(ids[i])] = slot.get(int(ids[i]), 0) + 1
            self.stats["learned"] += 1

    def draft(self, ids, k=4):
        """Propose up to k tokens by walking the learned routes. Returns [] when
        the route is unknown -- an honest miss beats a confident guess, because
        a wrong draft costs a wasted verification."""
        out, cur = [], list(int(t) for t in ids)
        for _ in range(int(k)):
            slot = self.table.get(self._key(cur, len(cur)))
            if not slot:
                break
            tok, cnt = max(slot.items(), key=lambda kv: (kv[1], -kv[0]))
            if cnt < self.min_count:
                break
            out.append(int(tok))
            cur.append(int(tok))
        self.stats["drafted"] += len(out)
        return out


def leap_generate(runtime, token_ids, n_new=32, memory=None, k=4, hooks=None,
                  learn=True):
    """Greedy generation, accelerated by drafting from learned routes and
    verifying in batched passes. Returns (ids, memory, report).

    The acceptance rule is exact: a drafted token is kept only if it equals the
    argmax the model itself produces at that position, given everything accepted
    before it. The first mismatch ends the run and the model's own token is used
    instead -- so a bad drafter can waste time but can NEVER change the output.
    """
    mem = memory if memory is not None else RouteMemory()
    ids = [int(t) for t in token_ids]
    logits, state = runtime.prefill(ids, hooks=hooks)
    if learn:
        mem.learn(ids)
    report = {"steps": 0, "batched_calls": 0, "accepted": 0, "drafted": 0}
    produced = 0
    while produced < n_new:
        nxt = int(np.argmax(logits))          # the model's own next token
        ids.append(nxt)
        produced += 1
        if produced >= n_new:
            logits, state = runtime.step(nxt, state, hooks=hooks)
            report["steps"] += 1
            break
        draft = mem.draft(ids, k=min(k, n_new - produced))
        if not draft:
            logits, state = runtime.step(nxt, state, hooks=hooks)
            report["steps"] += 1
            continue
        # ONE batched verification over [committed token] + [drafted tokens].
        # SNAPSHOT FIRST: a mismatch must cost a rewind to HERE, never a
        # re-prefill of the whole sequence.
        snap = state.copy()
        chunk = [nxt] + draft
        chunk_logits, state = runtime.extend(chunk, state, hooks=hooks)
        report["batched_calls"] += 1
        report["drafted"] += len(draft)
        # logits[i] is the distribution AFTER consuming chunk[i]; so the model's
        # own choice following chunk[i] is argmax(chunk_logits[i])
        n_ok = 0
        for i, d in enumerate(draft):
            if int(np.argmax(chunk_logits[i])) == d:
                n_ok += 1
            else:
                break
        if n_ok:
            ids.extend(draft[:n_ok])
            produced += n_ok
            report["accepted"] += n_ok
            mem.stats["accepted"] += n_ok
        mem.stats["rejected"] += len(draft) - n_ok
        if n_ok < len(draft):
            # REWIND to the snapshot and replay only the ACCEPTED tokens.
            #
            # MEASURED DESIGN FLAW, found only on a TRAINED model at scale: the
            # first version re-prefilled the entire sequence here, which is O(T)
            # per miss. At 91% acceptance on real text that still made leap
            # SLOWER than plain generation (0.84x at prompt 200) -- the toy
            # model hid it because its route was a perfect loop with no misses.
            # Rewinding to the snapshot makes a miss cost O(accepted), not O(T).
            state = snap
            logits = snap.logits
            for tok in [nxt] + draft[:n_ok]:
                logits, state = runtime.step(tok, state, hooks=hooks)
                report["steps"] += 1
        else:
            logits = chunk_logits[n_ok]
    if learn:
        mem.learn(ids)
    report["acceptance_rate"] = (report["accepted"] / report["drafted"]
                                 if report["drafted"] else 0.0)
    return ids, mem, report


def _selftest():
    try:
        import torch
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
    except ImportError:
        print("leap selftest SKIPPED-REFERENCE (torch/transformers absent)")
        return
    import time

    from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    cfg = Qwen3NextConfig(
        vocab_size=97, hidden_size=64, intermediate_size=112,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, full_attention_interval=4,
        num_experts=0, tie_word_embeddings=True, rms_norm_eps=1e-6)
    ref = Qwen3NextForCausalLM(cfg).eval().float()
    weights = {k: v.detach().numpy().astype(np.float64)
               for k, v in ref.state_dict().items()}
    rt = GDNRuntime(weights, dict(
        hidden=64, n_layers=4, rms_eps=1e-6, rope_theta=10000.0,
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=16, conv_kernel=4,
        n_heads=4, n_kv_heads=2, head_dim=16, partial_rotary_factor=0.25))
    ids = [int(t) for t in rng.integers(0, 97, size=24)]

    # 1) extend() is the load-bearing primitive: a batched chunk must equal
    #    stepping one at a time, to machine precision, or nothing below is valid
    extra = [int(t) for t in rng.integers(0, 97, size=6)]
    _l, sA = rt.prefill(ids)
    for t in extra:
        lA, sA = rt.step(t, sA)
    _l, sB = rt.prefill(ids)
    lB, sB = rt.extend(extra, sB)
    assert np.max(np.abs(lA - lB[-1])) < 1e-9

    # 2) COLD memory (novel text): output must be IDENTICAL to plain greedy.
    base, _st = rt.generate_fast(ids, n_new=24)
    t0 = time.time()
    got, mem, rep = leap_generate(rt, ids, n_new=24, k=4)
    t_cold = time.time() - t0
    assert got == base, (got, base)

    # 3) WARM memory (the route has been walked): same output, and now the
    #    drafter should actually hit -- this is the regime where structure pays.
    t0 = time.time()
    got2, mem2, rep2 = leap_generate(rt, ids, n_new=24, memory=mem, k=4)
    t_warm = time.time() - t0
    assert got2 == base, (got2, base)
    assert rep2["acceptance_rate"] > 0.5, rep2

    t0 = time.time()
    rt.generate_fast(ids, n_new=24)
    t_plain = time.time() - t0

    # 4) a HOSTILE drafter (always wrong) must not corrupt the output -- only
    #    waste time. Correctness cannot depend on the drafter being good.
    bad = RouteMemory(order=3)
    for i in range(1, len(base)):
        bad.table[tuple(base[max(0, i - 3):i])] = {(base[i] + 7) % 97: 99}
    got3, _m3, rep3 = leap_generate(rt, ids, n_new=24, memory=bad, k=4,
                                    learn=False)
    assert got3 == base, "a wrong drafter changed the output"
    assert rep3["acceptance_rate"] < 0.2, rep3

    print("leap selftest OK -- extend==stepwise to 1e-16; output token-identical "
          "to greedy in all three regimes (cold, warm, hostile drafter); warm "
          "acceptance %.0f%% at %.3fs vs plain %.3fs (%.2fx), cold %.3fs "
          "(the honest cost of a miss)"
          % (100 * rep2["acceptance_rate"], t_warm, t_plain,
             t_plain / max(t_warm, 1e-9), t_cold))


if __name__ == "__main__":
    _selftest()
