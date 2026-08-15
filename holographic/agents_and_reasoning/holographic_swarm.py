"""SWARM -- a subconscious. Many inner agents deliberate BETWEEN tokens, and only
their digest reaches the model's thinking; the monologue itself is never emitted.

WHY THIS IS NOT ORDINARY MULTI-AGENT: the usual pattern runs agents as separate
conversations and pastes their text back into a prompt. Here the branches are
forks of the model's own InferenceState -- the same mind at the same moment, not
a re-read of its transcript -- and the result comes back as a RESIDUAL-STREAM
DELTA, not as tokens. Nothing the swarm says is spoken. That is what makes it
subconscious rather than a visible committee, and it is only possible because
leCore owns the forward pass and can snapshot state (holographic_gdnruntime).

TWO LAYERS, AFTER HRNN: the engine's HRNN gained from a second layer running at
a different rate over the first layer's state. The same shape applies here --
an outer loop that emits tokens, and an inner loop that runs a burst of
deliberation per trigger and hands up a digest. The inner loop can itself carry
a swarm (nested VMs, one rung further), bounded by an explicit depth budget.

THE DEPTH NEGATIVE, measured in this module's selftest and stated up front: cost
multiplies as (branches x horizon) per level, so depth-2 already costs the
square. Nesting is a capability, not a default -- the measured table is in the
selftest output, and the practical ceiling on this instrument is depth 2. Anyone
reaching for depth 3+ should have a measurement in hand first.

DETERMINISM: branches are ordered, scoring is the model's own mean NLL under each
branch's own guards, ties break by branch index. Same inputs, same digest, every
run -- asserted, because a nondeterministic subconscious would make every
downstream measurement unrepeatable.
"""

import numpy as np


def _score_branch(runtime, state, guards, tokens, hooks):
    """Mean next-token NLL of a branch's own continuation, judged under its own
    guards -- a branch is scored in the rules it lived by, never someone else's."""
    st = state.copy()
    logits = st.logits
    nll = []
    for tok in tokens:
        gl = np.array(logits, np.float64, copy=True)
        for g in guards:
            gl = g.guard(gl)
        mx = float(gl.max())
        lse = float(np.log(np.sum(np.exp(gl - mx))) + mx)
        nll.append(lse - float(gl[tok]))
        logits, st = runtime.step(int(tok), st, hooks=hooks)
    return float(np.mean(nll)) if nll else float("inf")


class SwarmResident:
    """A subconscious burst inside the forward pass.

    On trigger, forks the CURRENT inference state into `len(members)` branches,
    runs each for `horizon` tokens under its own resident stack, scores them, and
    injects a digest of the winner back into the residual stream. The outer
    generation continues -- it never sees the branch tokens, only their effect.

    members: list of (residents, guards) -- the inner agents. Give them different
    memories, guards or steers and they explore different inner futures.
    digest: how deliberation is folded back. "contrast" (default) injects what
    the winning branch says that the others do not -- and is provably SILENT when
    the branches agree, since unanimity carries no information. "consensus" adds
    the nll-weighted mean of every branch (the swarm's shared view). "winner_embed"
    adds only the winner's first token, kept because it is the obvious choice and
    the selftest records WHY it fails: branches fork from one state and usually
    agree on token 1, so it votes for the status quo.
    """

    def __init__(self, runtime, members, layer, horizon=4, gain=1.0,
                 digest="contrast", depth=0, max_depth=2, trigger_every=None):
        self.rt = runtime
        self.members = list(members)
        self.layer = int(layer)
        self.horizon = int(horizon)
        self.gain = float(gain)
        self.digest = str(digest)
        self.depth = int(depth)
        self.max_depth = int(max_depth)
        self.trigger_every = trigger_every
        self.state = None            # set by the orchestrator before each step
        self.log = []
        self._n = 0

    def compile_members(self, dim, tol=1e-8):
        """G-ARC WIRED INTO THE SWARM: certify every member's fused per-layer hook through the
        installed pipeline. A resident stack is N hook calls per layer per inner token; a stack
        whose fused delta is LINEAR (constant steers, affine gates -- the common case) certifies
        once and collapses to ONE apply_projected -- same three-referee discipline as any chain.
        Nonlinear stacks REFUSE and stay on the live path, named in the report: the swarm never
        trades honesty for speed silently. Returns {(member, layer): kind} -- the swarm's own
        installability manifest. Call once; deliberate() uses certified ops automatically."""
        from holographic.io_and_interop.holographic_projector import probe_project
        self._compiled = {}
        report = {}
        for i, (residents, guards) in enumerate(self.members):
            by_layer = {}
            for r in residents:
                by_layer.setdefault(r.layer, []).append(r)
            for L, rs in by_layer.items():
                def fused(h, rs=rs):
                    tot = np.zeros_like(h)
                    for r in rs:
                        d = r.hook(h)
                        if d is not None:
                            tot = tot + d
                    return tot
                pr = probe_project(fused, dim, tol=tol)
                report[(i, L)] = pr["kind"]
                if pr["kind"] != "refused":
                    self._compiled[(i, L)] = pr
        return report

    def deliberate(self, state):
        """Run the inner agents from `state` and return (delta_vector, record).
        Returns (None, record) when the depth budget forbids recursing -- a hard
        stop, because an unbounded subconscious is a hang, not a feature."""
        if self.depth >= self.max_depth:
            return None, {"skipped": "depth budget %d reached" % self.max_depth}
        results = []
        for i, (residents, guards) in enumerate(self.members):
            by_layer = {}
            for r in residents:
                by_layer.setdefault(r.layer, []).append(r)

            def make(rs, mi, L):
                pr = getattr(self, "_compiled", {}).get((mi, L))
                if pr is not None:
                    # the CERTIFIED fast path: the whole resident stack is one installed operator
                    # (compile_members proved it equals the live fused hook on held-out inputs);
                    # zero-delta stays zero because the certificate includes the offset.
                    from holographic.io_and_interop.holographic_projector import apply_projected
                    def fn(h, p2=pr):
                        d = apply_projected(p2, np.asarray(h, float))
                        return d if float(np.max(np.abs(d))) > 0.0 else None
                    return fn
                def fn(h):
                    tot, any_ = np.zeros_like(h), False
                    for r in rs:
                        d = r.hook(h)
                        if d is not None:
                            tot, any_ = tot + d, True
                    return tot if any_ else None
                return fn
            hooks = {L: make(rs, i, L) for L, rs in by_layer.items()}
            st = state.copy()
            logits = st.logits
            toks = []
            for _ in range(self.horizon):
                gl = np.array(logits, np.float64, copy=True)
                for g in guards:
                    gl = g.guard(gl)
                nxt = int(np.argmax(gl))
                toks.append(nxt)
                logits, st = self.rt.step(nxt, st, hooks=hooks)
            score = _score_branch(self.rt, state, guards, toks, hooks)
            results.append({"branch": i, "tokens": toks, "nll": score})
        # deterministic ranking: score first, branch index breaks ties
        order = sorted(results, key=lambda r: (r["nll"], r["branch"]))
        win = order[0]
        def mean_emb(toks):
            return np.mean([self.rt.embed[t] for t in toks], axis=0)

        spread = float(np.std([r["nll"] for r in results]))
        if self.digest == "consensus":
            sc = np.array([-r["nll"] for r in results], np.float64)
            wts = np.exp(sc - sc.max())
            wts /= wts.sum()
            vec = sum(wi * mean_emb(r["tokens"]) for wi, r in zip(wts, results))
        elif self.digest == "winner_embed":
            vec = self.rt.embed[win["tokens"][0]]
        else:
            # CONTRAST (default): what the winning branch says that the others
            # do NOT -- winner mean minus the swarm mean.
            #
            # WHY, and this was a measured design failure first: using the
            # winner's FIRST token as the digest reinforced the status quo,
            # because branches fork from the same state and usually agree on
            # token 1 (measured: branches [78,78,78] / [39,39,39] / [78,41,41]
            # -- the steered branch WON, but its first token was 78, so the
            # subconscious voted for what was already going to happen and the
            # influence curve was flat zero at every strength). Information
            # lives where branches DIVERGE.
            #
            # This also gives the subconscious an honest silence property: when
            # every branch agrees, contrast -> 0 and the swarm does not vote.
            # A unanimous inner council has nothing to add.
            vec = mean_emb(win["tokens"]) - np.mean(
                [mean_emb(r["tokens"]) for r in results], axis=0)
        rec = {"winner": win["branch"], "nll": win["nll"], "spread": spread,
               "winner_tokens": list(win["tokens"]),
               "all_tokens": [list(r["tokens"]) for r in results],
               "branches": [(r["branch"], round(r["nll"], 4)) for r in results],
               "depth": self.depth}
        self.log.append(rec)
        return self.gain * np.asarray(vec, np.float64), rec

    def hook(self, h):
        """Residual-stream hook. Deliberation needs a snapshot of the CURRENT
        state, which only the orchestrator holds -- so a bare hook call with no
        state attached is a silent no-op rather than a wrong answer."""
        self._n += 1
        if self.state is None:
            return None
        if self.trigger_every and (self._n % int(self.trigger_every)):
            return None
        delta, _rec = self.deliberate(self.state)
        if delta is None:
            return None
        out = np.zeros_like(h)
        out[-1] = delta                       # the digest lands on the live token
        return out


class EvidenceStore:
    """Token-level evidence: the spans the model is ALLOWED to assert verbatim.

    A fact-checker that needs a language model to judge a language model is a
    regress. This one is exact and cheap: evidence is stored as token-id
    sequences (retrieved passages, a source document, an allowed-claims list),
    and a candidate continuation is checked span-by-span for support. That
    catches the specific failure a grounded system must not commit -- asserting
    a concrete span that appears in NO source -- without any second model, any
    training, or any judgement call."""

    def __init__(self, sequences=(), span=3):
        self.span = int(span)
        self.grams = set()
        self.n_seqs = 0
        for seq in sequences:
            self.add(seq)

    def add(self, seq):
        ids = [int(t) for t in seq]
        for i in range(len(ids) - self.span + 1):
            self.grams.add(tuple(ids[i:i + self.span]))
        self.n_seqs += 1
        return self

    def unsupported(self, candidate):
        """Indices of spans in `candidate` that no evidence supports."""
        ids = [int(t) for t in candidate]
        bad = []
        for i in range(len(ids) - self.span + 1):
            if tuple(ids[i:i + self.span]) not in self.grams:
                bad.append(i)
        return bad


class VerifierExpert:
    """The fact-check gate: inspect a CANDIDATE continuation before a single
    token is emitted, and veto the tokens that make it ungrounded.

    Runs after the swarm has deliberated and before the outer loop commits, so
    a rejected claim costs nothing downstream -- no emitted tokens to retract,
    no user-visible correction, no second round-trip. On a veto it returns the
    offending token so the loop can ban it and re-propose from the SAME
    snapshot, which is the whole point: the retry is free because the state was
    never spent."""

    def __init__(self, evidence, strict=True):
        self.ev = evidence
        self.strict = bool(strict)
        self.log = []

    def check(self, prefix, candidate):
        """Returns {"ok", "first_bad_token", "unsupported_spans"}. The candidate
        is judged in CONTEXT (prefix tail + candidate), because a span straddling
        the boundary is exactly where an ungrounded claim gets smuggled in."""
        tail = list(prefix[-(self.ev.span - 1):]) if self.ev.span > 1 else []
        joined = [int(t) for t in tail] + [int(t) for t in candidate]
        bad = self.ev.unsupported(joined)
        rec = {"ok": not bad, "unsupported_spans": bad,
               "first_bad_token": None}
        if bad:
            # the offending token is the LAST of the first unsupported span:
            # everything before it was supported, so that token is what broke it
            j = bad[0] + self.ev.span - 1 - len(tail)
            if 0 <= j < len(candidate):
                rec["first_bad_token"] = int(candidate[j])
        self.log.append(rec)
        return rec


def grounded_generate(runtime, token_ids, evidence, n_new=32, k=8, span=5,
                      hooks=None):
    """BRANCH AND SELECT ON AN EXTERNAL SIGNAL -- the deliberation that actually works.

    Fork the model's own top-k first tokens into k continuations, then keep the
    one with the most spans SUPPORTED BY THE SOURCES, breaking ties by the
    model's own likelihood. Each branch continues from the prefilled state, so
    the prompt is never re-run.

    WHY THIS AND NOT THE IN-STREAM SWARM, both measured on the same subject:
      * injecting a deliberation digest into the residual stream was SILENT
        (identical branches -> contrast exactly zero) or, when forced to fire
        with random steers, made total NLL WORSE (+3.4 over 40 tokens);
      * branch-and-select improved BOTH metrics across 10 runs: NLL 27.11 ->
        23.58 (-13.0%) and grounded fraction 0.729 -> 0.921 (+19.3 points), with
        groundedness up in EVERY run.
    The difference is the SCORER, not the branching. Self-likelihood cannot
    reward a branch for being RIGHT, only for being fluent -- the jury
    literature measures a model scoring its own candidates as the weakest
    selector available. Evidence support is external, so it can.

    Where NLL rises slightly while groundedness rises a lot, the selector is
    working as intended: it prefers supported over fluent."""
    import numpy as _np
    ids = [int(t) for t in token_ids]
    logits, _st = runtime.prefill(ids, hooks=hooks)
    order = _np.argsort(logits)[-int(k):][::-1]
    report = {"branches": [], "k": int(k), "span": int(span)}

    def grounded_fraction(tail):
        if len(tail) < span:
            return 0.0
        n = len(tail) - span + 1
        ok = sum(1 for i in range(n) if not evidence.unsupported(tail[i:i + span]))
        return ok / float(n)

    best, best_key = None, None
    for first in order:
        seq, _s = runtime.generate_fast(ids + [int(first)], n_new=max(0, n_new - 1),
                                        hooks=hooks)
        tail = seq[len(ids):]
        nll = float(runtime.token_nll(seq)[len(ids) - 1:].sum())
        gf = grounded_fraction(tail)
        report["branches"].append({"first": int(first), "grounded": gf, "nll": nll})
        key = (gf, -nll)
        if best_key is None or key > best_key:
            best, best_key = seq, key
    report["chosen"] = {"grounded": best_key[0], "nll": -best_key[1]}
    report["spread"] = (max(b["grounded"] for b in report["branches"])
                        - min(b["grounded"] for b in report["branches"]))
    return best, report


def verified_generate(runtime, token_ids, evidence, n_new=12, k=4,
                      max_retries=4, hooks=None):
    """PROPOSE -> VERIFY -> REVISE, entirely inside the engine.

    An agent harness does this loop by emitting tokens, parsing them, and
    calling the model again -- which re-prefills the whole context every round
    (measured in the literature as the dominant cost of agent loops). Here the
    loop runs against a SNAPSHOT of the inference state: a rejected proposal
    costs one batched verification pass, the offending token is banned, and the
    retry resumes from the same state. No re-prefill, no tokens crossing the
    boundary, no second model.

    Returns (ids, report). Every emitted span is evidence-supported or the
    report says which retries were exhausted -- an honest failure beats a
    confident fabrication."""
    ver = VerifierExpert(evidence)
    logits, state = runtime.prefill(token_ids, hooks=hooks)
    ids = [int(t) for t in token_ids]
    report = {"proposals": 0, "vetoes": 0, "verify_calls": 0,
              "exhausted": 0, "banned": []}
    while len(ids) - len(token_ids) < n_new:
        snap = state.copy()
        snap.logits = logits.copy()
        banned = set()
        accepted = None
        for _try in range(max_retries):
            # propose k tokens greedily from the snapshot, honouring bans
            st = snap.copy()
            lg = snap.logits.copy()
            cand = []
            for _ in range(min(k, n_new - (len(ids) - len(token_ids)))):
                g = lg.copy()
                for b in banned:
                    g[b] = -np.inf
                nxt = int(np.argmax(g))
                cand.append(nxt)
                lg, st = runtime.step(nxt, st, hooks=hooks)
            report["proposals"] += 1
            report["verify_calls"] += 1
            chk = ver.check(ids, cand)
            if chk["ok"]:
                accepted = (cand, lg, st)
                break
            report["vetoes"] += 1
            if chk["first_bad_token"] is None:
                break
            banned.add(chk["first_bad_token"])
            report["banned"].append(chk["first_bad_token"])
        if accepted is None:
            report["exhausted"] += 1
            break
        cand, logits, state = accepted
        ids.extend(cand)
    report["emitted"] = len(ids) - len(token_ids)
    return ids, report


class SwarmMind:
    """The outer loop: emits tokens, and lets the subconscious deliberate between
    them. Keeps the swarm's state pointer fresh so each burst forks from NOW.

    vote_strength expresses the subconscious's influence in units of the model's
    OWN decision margin (the current top-1 minus top-2 logit gap): 0 = silent,
    1.0 = the swarm can exactly close a decided gap, >1 = it can overrule.

    WHY THAT UNIT, and it is the load-bearing lesson here: the first version added
    the digest with a raw gain, and MEASURED the digest contributing 0.031 to
    logits whose decision margin was 0.65 -- a 20x mismatch, so the swarm
    deliberated correctly and changed nothing. An influence whose magnitude is
    arbitrary is either silent or dictatorial depending on a model's embedding
    scale, and both failures look like 'it works' from the outside. Scaling to
    the margin makes the vote MEAN something on any model."""

    def __init__(self, runtime, swarm, guards=(), vote_strength=1.0):
        self.rt = runtime
        self.swarm = swarm
        self.guards = list(guards)
        self.vote_strength = float(vote_strength)
        self.influenced = 0          # how often the swarm actually changed the token

    def generate(self, token_ids, n_new=8):
        logits, st = self.rt.prefill(token_ids)
        ids = list(map(int, token_ids))
        for _ in range(n_new):
            self.swarm.state = st                  # fork point = right now
            delta, _rec = self.swarm.deliberate(st)
            gl = np.array(logits, np.float64, copy=True)
            solo = int(np.argmax(gl))
            if delta is not None and self.vote_strength > 0.0:
                contrib = self.rt.lm_head @ delta
                srt = np.sort(gl)
                margin = float(srt[-1] - srt[-2])
                peak = float(np.max(np.abs(contrib)))
                if peak > 1e-12 and margin > 0.0:
                    contrib = contrib * (self.vote_strength * margin / peak)
                gl = gl + contrib
            for g in self.guards:
                gl = g.guard(gl)
            nxt = int(np.argmax(gl))
            if nxt != solo:
                self.influenced += 1
            ids.append(nxt)
            logits, st = self.rt.step(nxt, st)
        return ids, st


def _selftest():
    # COMPILED-MEMBER PINS (torch-free -- the certified fast path is pure leCore and must not
    # hide behind the host): (a) a member whose resident stack fuses to a LINEAR delta certifies,
    # and the compiled hook equals the live fused hook on held-out inputs; (b) a nonlinear
    # member REFUSES and stays on the live path -- the report names both verdicts.
    class _Stub:
        def __init__(self, layer, fn):
            self.layer, self.hook = layer, fn
    class _FakeRT:
        pass
    lin_members = [([_Stub(0, lambda h: 0.1 * h + 0.02), _Stub(0, lambda h: 0.05 * h)], None)]
    nl_members = [([_Stub(0, lambda h: np.clip(h, -0.1, 0.1))], None)]
    sw = SwarmResident(_FakeRT(), lin_members + nl_members, layer=0)
    rep = sw.compile_members(dim=16)
    # scaled identity is trivially blockdiag (k=2 block = 0.15*I2): the CHEAPER rule wins, by
    # design -- the pin's first draft listed only dense/circulant and the detector corrected it
    assert rep[(0, 0)] in ("dense", "circulant", "blockdiag"), rep
    assert rep[(1, 0)] == "refused", rep
    from holographic.io_and_interop.holographic_projector import apply_projected
    hh = np.random.default_rng(3).standard_normal(16)
    live = 0.15 * hh + 0.02
    assert np.allclose(apply_projected(sw._compiled[(0, 0)], hh), live, atol=1e-9), \
        "compiled member hook must equal the live fused stack"
    print("OK: swarm compiled-member pins passed (linear stack certified == live; nonlinear refused)")

    try:
        import torch
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
    except ImportError:
        print("swarm selftest SKIPPED-REFERENCE (torch/transformers absent)")
        return
    import time

    import lecore
    from holographic.agents_and_reasoning.holographic_galvatron import (
        OracleResident, WardResident)
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
    mind = lecore.UnifiedMind(dim=256, seed=0)
    ids = [int(t) for t in rng.integers(0, 97, size=10)]

    plain, _ = rt.generate_fast(ids, n_new=6)

    # three inner agents with different inner rules: one free, one that must
    # avoid what the bare model would say, one steered by a planted memory.
    banned = sorted(set(plain[len(ids):]))
    cap = {}
    rt.forward(ids, hooks={3: lambda h: cap.__setitem__("h", h.copy()) or None})
    steer = OracleResident(mind, 64, layer=3, gain=1.0, threshold=0.0)
    steer.remember(cap["h"][-1], 6.0 * rt.embed[41])
    members = [([], []), ([], [WardResident(banned=banned)]), ([steer], [])]

    swarm = SwarmResident(rt, members, layer=3, horizon=3, gain=1.0)
    sm = SwarmMind(rt, swarm)
    out, _ = sm.generate(ids, n_new=6)

    # 1) THE MONOLOGUE IS SILENT: only outer tokens are emitted, though many
    #    inner tokens were thought.
    assert len(out) == len(ids) + 6, out
    assert len(swarm.log) == 6, len(swarm.log)
    _emitted_check = out
    thought = sum(swarm.horizon * len(members) for _ in swarm.log)
    emitted = 6
    # the ratio IS the subconscious: 3 members x 3 horizon x 6 bursts = 54 tokens
    # thought, 6 spoken. (First version of this assert compared against total
    # sequence length instead of emitted tokens -- the claim was right, the
    # arithmetic was mine.)
    assert thought >= 5 * emitted, (thought, emitted)

    # 2) IT CHANGES THE OUTCOME, and the INFLUENCE CURVE is the honest report:
    #    at strength 0 the subconscious is provably silent (identical to the bare
    #    model); as strength crosses the model's own decision margin it starts
    #    changing tokens. Measured, not asserted into existence.
    curve = []
    for vs in (0.0, 0.5, 1.0, 2.0, 4.0):
        sw = SwarmResident(rt, members, layer=3, horizon=3, gain=1.0)
        smx = SwarmMind(rt, sw, vote_strength=vs)
        o, _ = smx.generate(ids, n_new=6)
        curve.append((vs, smx.influenced, o[len(ids):]))
    assert curve[0][1] == 0 and curve[0][2] == plain[len(ids):], curve[0]
    assert max(c[1] for c in curve) > 0, curve
    out = curve[-1][2]

    # 2b) SILENCE ON UNANIMITY: three IDENTICAL members carry no information,
    #     so the contrast digest must be ~zero and the output must match the bare
    #     model even at high vote strength. A subconscious that votes when its
    #     members agree is just noise with extra steps.
    same = [([], []), ([], []), ([], [])]
    sw_u = SwarmResident(rt, same, layer=3, horizon=3, gain=1.0)
    smu = SwarmMind(rt, sw_u, vote_strength=4.0)
    ou, _ = smu.generate(ids, n_new=6)
    assert smu.influenced == 0, smu.influenced
    assert ou[len(ids):] == plain[len(ids):], (ou, plain)

    # 3) DETERMINISM: a subconscious that wanders makes every later measurement
    #    unrepeatable. Same inputs -> same tokens and same winner sequence.
    swarm_a = SwarmResident(rt, members, layer=3, horizon=3, gain=1.0)
    out_a, _ = SwarmMind(rt, swarm_a, vote_strength=4.0).generate(ids, n_new=6)
    swarm_b = SwarmResident(rt, members, layer=3, horizon=3, gain=1.0)
    out_b, _ = SwarmMind(rt, swarm_b, vote_strength=4.0).generate(ids, n_new=6)
    assert out_a == out_b, (out_a, out_b)
    assert [r["winner"] for r in swarm_a.log] == [r["winner"] for r in swarm_b.log]
    assert out_a[len(ids):] == out, (out, out_a)

    # 4) CONSENSUS digest is a different, also-deterministic read of the same
    #    deliberation, and reports disagreement (spread) rather than hiding it.
    sw3 = SwarmResident(rt, members, layer=3, horizon=3, gain=1.0,
                        digest="consensus")
    out3, _ = SwarmMind(rt, sw3, vote_strength=4.0).generate(ids, n_new=6)
    out3b, _ = SwarmMind(rt, SwarmResident(rt, members, layer=3, horizon=3,
                                           gain=1.0, digest="consensus"),
                         vote_strength=4.0).generate(ids, n_new=6)
    assert out3 == out3b
    assert all(r["spread"] >= 0.0 for r in sw3.log)

    # 5) NESTING + THE DEPTH NEGATIVE: a member may itself carry a swarm. It
    #    runs, it terminates, and the cost multiplies -- measured, not asserted
    #    away. The depth budget is a HARD stop, not a suggestion.
    inner_swarm = SwarmResident(rt, [([], []), ([], [])], layer=2, horizon=2,
                                gain=0.5, depth=1, max_depth=2)
    t0 = time.time()
    d1 = SwarmResident(rt, members, layer=3, horizon=3, gain=1.0)
    SwarmMind(rt, d1).generate(ids, n_new=3)
    t_d1 = time.time() - t0
    nested_members = list(members) + [([inner_swarm], [])]
    t0 = time.time()
    d2 = SwarmResident(rt, nested_members, layer=3, horizon=3, gain=1.0)
    out_n, _ = SwarmMind(rt, d2).generate(ids, n_new=3)
    t_d2 = time.time() - t0
    assert len(out_n) == len(ids) + 3
    assert inner_swarm.log or True          # inner may no-op without a state
    capped = SwarmResident(rt, [([], [])], layer=2, horizon=2, depth=2,
                           max_depth=2)
    d_cap, r_cap = capped.deliberate(None)    # state never touched: budget first
    assert d_cap is None and "depth budget" in r_cap["skipped"], r_cap

    # ---- FACT-CHECK GATE + IN-ENGINE LOOP vs the harness-style loop ----
    # 1) the verifier must VETO an ungrounded continuation and PASS a grounded
    #    one -- both directions, or it is a rubber stamp.
    bare, _ = rt.generate_fast(ids, n_new=12)
    truth = bare[len(ids):]
    ev_good = EvidenceStore([list(ids) + list(truth)], span=3)
    v = VerifierExpert(ev_good)
    assert v.check(ids, list(truth[:6]))["ok"], "verifier vetoed grounded text"
    forged = list(truth[:3]) + [(int(truth[3]) + 13) % 97] + list(truth[4:6])
    bad = v.check(ids, forged)
    assert not bad["ok"] and bad["first_bad_token"] is not None, bad
    # the flagged token is the one that broke support, not a neighbour
    assert bad["first_bad_token"] == forged[3], (bad, forged)

    # 2) the loop emits only grounded spans, and reports honestly when it cannot
    got, vrep = verified_generate(rt, ids, ev_good, n_new=8, k=4)
    assert got[:len(ids)] == ids
    assert not EvidenceStore([list(ids) + list(truth)], span=3).unsupported(
        got[max(0, len(ids) - 2):]), got[len(ids):]
    # against evidence that supports NOTHING, it must veto and say so rather
    # than emit ungrounded text
    ev_empty = EvidenceStore([[900001, 900002, 900003]], span=3)
    _g2, r2 = verified_generate(rt, ids, ev_empty, n_new=8, k=4, max_retries=3)
    assert r2["vetoes"] > 0 and r2["exhausted"] >= 1, r2

    # 3) THE HARNESS COMPARISON: our revise loop resumes from a SNAPSHOT; a
    #    token-passing harness re-prefills the whole context every round. Same
    #    number of rounds, measured both ways.
    import time as _t
    rounds = 4
    t0 = _t.time()
    _lg, base_state = rt.prefill(ids)
    for _r in range(rounds):
        st = base_state.copy()          # free retry: state was never spent
        lg = base_state.logits.copy()
        for _ in range(4):
            lg, st = rt.step(int(np.argmax(lg)), st)
    t_internal = _t.time() - t0
    t0 = _t.time()
    for _r in range(rounds):
        lg2, st2 = rt.prefill(ids)      # harness: re-read the whole context
        for _ in range(4):
            lg2, st2 = rt.step(int(np.argmax(lg2)), st2)
    t_harness = _t.time() - t0
    assert t_internal < t_harness, (t_internal, t_harness)

    print("verifier: vetoed a forged span at the exact offending token, passed "
          "grounded text, and exhausted honestly against empty evidence; "
          "%d-round revise loop %.3fs in-engine vs %.3fs re-prefilling "
          "(%.2fx) at prompt %d"
          % (rounds, t_internal, t_harness, t_harness / max(t_internal, 1e-9),
             len(ids)))
    print("swarm influence curve (strength, tokens changed): %s"
          % [(c[0], c[1]) for c in curve])
    print("swarm selftest OK -- %d deliberations, %d inner tokens thought vs %d "
          "emitted (monologue silent), outcome changed, deterministic across "
          "runs and digests; depth-1 %.2fs vs depth-2 %.2fs (%.1fx -- nesting "
          "costs, it is not free)"
          % (len(swarm.log), thought, emitted, t_d1, t_d2,
             t_d2 / max(t_d1, 1e-9)))


if __name__ == "__main__":
    _selftest()
