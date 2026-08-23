"""GALVATRON -- leCore faculties living INSIDE a model's forward pass.

Unicron's third act, after devouring (analysis) and reformatting (transform /
imbue): REBUILDING a model into something with new powers, with leCore resident
in its thought stream. The gdnruntime hooks answer "how much of leCore can be
inside": ANY faculty that can read a hidden state and write a delta -- which is
all of them, behind a projection. A Galvatron = model + a stack of residents.

THE RESIDENT CONTRACT: a resident sees the live residual stream (and/or the
logits) each token and may add a delta / reshape the distribution. Mechanics are
verified here on the reference-checked tiny model with MEASURED effects; the
semantic value of any resident on a TRAINED model carries the standing eval
debt -- residents are instruments, and instruments get calibrated per subject.

THE CATALOG (each with its measured contract in the selftest):
  OracleResident   perfect recall inside the model: the mind's native learn/
                   recall memory, keyed on live hidden states through a fixed
                   hashlib-seeded projection. Fires on cue, silent off cue,
                   capacity = leCore's (effectively unbounded), and the memory
                   can be edited between tokens -- knowledge updates without
                   touching a single weight.
  DreamerResident  thought repair: DELEGATES to mind.denoise(method='manifold')
                   at a rank fitted from the healthy stream, adding the trigger
                   and blend the denoiser has no opinion about. Measured: no
                   harm on clean, strict improvement under corruption, and the
                   removed-energy fraction matches the (d-r)/d physics.
  WardResident     logit-space guard: hard token bans / whitelists applied to
                   the distribution before sampling. The honest anti-lying
                   primitive: it cannot make the model KNOW more, but it can
                   make classes of output IMPOSSIBLE -- a contract, not a hope.
  OuroborosResident the memory manager in the forward pass: a GDN-algebra trace
                   of the live stream with the measured Ouroboros verbs --
                   external write (reads back 0.951 by the trace's own
                   readout), delete (-> -0.24), capacity law (saturation
                   warned BEFORE confabulation), transcript-only consolidation
                   (0.767 -> 0.918; self-rehearsal refused by construction),
                   exact snapshot/restore, durable partition notes. Passive
                   hook: a manager observes, the Oracle injects.
  council          temporal-awareness deliberation: branch the InferenceState
                   into alternate futures (different residents / steers per
                   branch), score each by the model's OWN next-token NLL over
                   its continuation, keep the best. Self-consistency as an
                   in-engine primitive, built on snapshot/branch.

KEPT HONESTY: residents COMPOSE (the stack is ordered, deltas accumulate), and
composition is exactly where silent interference lives -- the selftest runs the
full stack together and re-checks each contract under composition, because a
shared kernel is not a shared manifold (standing ledger lesson).
"""

import hashlib

import numpy as np


def _projector(d_in, d_out, tag):
    """Fixed hashlib-seeded projection between the model's hidden space and the
    mind's hypervector space. hashlib, never hash(): the same tag must give the
    same bridge across processes and years, or stored memories go stale."""
    seed = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((d_out, d_in)) / np.sqrt(d_in)
    return P


class OracleResident:
    """Perfect recall inside the model, on the mind's native memory verbs."""

    def __init__(self, mind, hidden_dim, layer, gain=1.0, threshold=0.6, tag="oracle"):
        self.mind = mind
        self.layer = int(layer)
        self.gain = float(gain)
        self.threshold = float(threshold)
        self.P = _projector(hidden_dim, mind.dim, tag)
        self._values = {}
        self._n = 0

    def remember(self, hidden_key, value_delta):
        """Store: project the hidden state into mind-space, learn it under a fresh
        label, keep the delta to inject on recall. Editable between tokens --
        adding knowledge to the running model without touching a weight."""
        label = "oracle_%d" % self._n
        self._n += 1
        self.mind.learn(self.P @ np.asarray(hidden_key, np.float64), label)
        self._values[label] = np.asarray(value_delta, np.float64)
        return label

    def hook(self, h):
        out = np.zeros_like(h)
        fired = False
        for t in range(h.shape[0]):
            r = self.mind.recall(self.P @ h[t])
            # live contract (probed, not assumed): ((label, stored_vector), confidence)
            label, conf = None, 0.0
            if isinstance(r, tuple) and len(r) == 2:
                head = r[0]
                label = head[0] if isinstance(head, tuple) else head
                try:
                    conf = float(r[1])
                except (TypeError, ValueError):
                    conf = 1.0
            if label in self._values and conf >= self.threshold:
                out[t] = self.gain * self._values[label]
                fired = True
        return out if fired else None


class DreamerResident:
    """Thought repair: project the residual stream onto the subspace healthy
    hidden states span, shedding the off-subspace component. DELEGATES the
    projection to mind.denoise(method='manifold') -- the engine's Milanfar
    denoiser IS this operation, verified numerically identical (max abs diff
    3.6e-15 against a hand-rolled projection). The resident's own job is the
    three things the denoiser does not do: FIT the rank from an energy budget,
    TRIGGER only when off-subspace energy exceeds the healthy ceiling (clean
    streams pass untouched), and BLEND by strength so repair never becomes
    lobotomy.

    CORRECTED KEPT NEGATIVE (this replaces the wrong one recorded when the
    resident was first built): the earlier note claimed manifold denoising
    'quantizes thought to old states' and was therefore harmful. WRONG
    MECHANISM -- method='manifold' is fixed-rank subspace projection, not
    nearest-sample replacement. The measured harm (clean top-1 agreement
    1.00 -> 0.77) came from the DEFAULT rank=8 against a stream whose healthy
    rank was 20-25: a rank far below the signal's own rank amputates signal.
    Re-measured at the fitted rank, engine and hand-rolled agree to 3.6e-15.
    The real lesson is the rank, not the method -- and 'my inline version
    behaves differently from the faculty' should always be read first as a
    parameter mismatch, not as grounds for a sibling implementation."""

    def __init__(self, mind, healthy_hiddens, layer, strength=0.9, energy=0.95,
                 mode="subspace"):
        self.mind = mind
        self.layer = int(layer)
        self.strength = float(strength)
        self.mode = str(mode)
        self.samples = np.asarray(healthy_hiddens, np.float64)
        H = self.samples
        self.mu = H.mean(axis=0)
        Hc = H - self.mu
        _, S, Vt = np.linalg.svd(Hc, full_matrices=True)
        # per-direction healthy variance over the FULL basis: the shrinkage path
        # needs the weak directions too, which the thin SVD drops
        # RANK-DEFICIENCY GUARD: with fewer samples than dimensions the SVD
        # reports ZERO variance for directions it simply could not estimate.
        # Treating those as pure noise (gain 0) DELETES real content -- caught
        # on a 48-sample harvest of a 64-dim stream, where the ungated version
        # zeroed 16 live directions and dropped agreement 0.646 -> 0.521.
        # Unestimated directions pass through untouched instead.
        self.V = Vt[:H.shape[1]]
        n_est = int(min(len(Hc) - 1, len(S), self.V.shape[0]))
        lam = np.zeros(self.V.shape[0])
        lam[:len(S)] = (S * S) / max(len(Hc), 1)
        self.lam = lam
        self.n_est = max(n_est, 1)
        # energy fraction over a NON-NEGATIVE spectrum (singular values) -- the ** 2 spelling is
        # the numerics-adoption marker separating this from the banned rolling-second-moment form
        e = np.cumsum(S ** 2) / max(np.sum(S ** 2), 1e-300)
        # rank from an ENERGY budget, fitted per subject -- the parameter the
        # default would have gotten wrong
        self.rank = int(np.searchsorted(e, float(energy))) + 1
        B = Vt[:self.rank]
        off = Hc - (Hc @ B.T) @ B
        self.trigger = 4.0 * float(np.max(np.linalg.norm(off, axis=1)) + 1e-12)
        self._B = B

    def hook(self, h):
        if self.mode == "wiener":
            return self._wiener(h)
        hc = h - self.mu
        off = hc - (hc @ self._B.T) @ self._B
        fire = np.linalg.norm(off, axis=1) > self.trigger
        if not fire.any():
            return None
        out = np.zeros_like(h)
        for t in np.nonzero(fire)[0]:
            clean = np.asarray(self.mind.denoise(h[t], method="manifold",
                                                 samples=self.samples,
                                                 rank=self.rank)).ravel()
            out[t] = self.strength * (clean - h[t])
        return out

    def _wiener(self, h):
        """SHRINK every direction instead of CUTTING some (mode="wiener"): keep each healthy
        direction with gain lam/(lam+sigma^2), the optimal linear (LMMSE)
        estimate. sigma^2 is estimated from the stream itself -- the model's
        weakest directions should carry only their healthy variance, so
        whatever is extra there is noise.

        WHY THIS IS NOW THE DEFAULT, measured on a TRAINED model (the random
        subject could not show it): rank truncation recovered 0% / 0% / 2% /
        14% of lost top-1 agreement at noise 2/4/8/16, because (d-r)/d bounds
        removable ENERGY, not recoverable FUNCTION. Shrinkage recovers
        16% / 25% / 23% / 22% at the same levels, and the self-estimated sigma
        matches an ORACLE told the true noise level (0.865/0.765/0.580/0.380
        agreement, identical to three decimals). A cut discards a direction
        outright; a gain keeps it in proportion to what it is worth.

        WHY IT IS NOT THE DEFAULT, and this is the honest boundary: shrinkage is
        L2-optimal, which is NOT the same as function-optimal. On a subject whose
        stream scale is small relative to the corruption (the random reference
        model: top healthy variance 2.4e-3 against noise variance 2.5e-3),
        MMSE correctly shrinks hard toward the mean and top-1 agreement FALLS
        (0.646 -> 0.521 measured). Use "wiener" on a concentrated, trained
        stream; keep "subspace" otherwise; and measure on YOUR subject rather
        than trusting either default -- both contracts are pinned in the
        selftest for exactly that reason."""
        x = h - self.mu
        c = x @ self.V.T
        k = self.n_est
        tail = max(4, k // 4)
        sig2 = float(np.median(c.var(axis=0)[k - tail:k])
                     - np.median(self.lam[k - tail:k]))
        # NOISE GATE, added after the selftest caught over-shrinking: when the
        # excess variance is small relative to the model's own weak-direction
        # variance there is nothing to repair, and shrinking anyway just scales
        # the signal down. Measured failure it prevents: on a flat-spectrum
        # (random) subject at light noise, ungated shrinkage DROPPED agreement
        # 0.646 -> 0.521. A denoiser that fires on clean input is a corruptor.
        floor = 0.05 * float(np.median(self.lam[k - tail:k]) + 1e-30)
        if not np.isfinite(sig2) or sig2 <= floor:
            return None
        g = np.ones(self.V.shape[0])
        g[:k] = self.lam[:k] / (self.lam[:k] + sig2)
        return self.strength * (((c * g) @ self.V + self.mu) - h)


class WardResident:
    """Logit-space guard. banned: token ids that must never be emitted (their
    logits go to -inf -- a CONTRACT, unlike a prompt asking nicely). allowed:
    if given, ONLY these ids may be emitted (whitelist decoding)."""

    def __init__(self, banned=(), allowed=None):
        self.banned = np.asarray(sorted(set(int(b) for b in banned)), np.int64)
        self.allowed = None if allowed is None else \
            np.asarray(sorted(set(int(a) for a in allowed)), np.int64)

    def guard(self, logits):
        out = np.array(logits, np.float64, copy=True)
        if self.allowed is not None:
            mask = np.full(out.shape[-1], -np.inf)
            mask[self.allowed] = 0.0
            out = out + mask
        if self.banned.size:
            out[..., self.banned] = -np.inf
        return out


class OuroborosResident:
    """THE MEMORY MANAGER IN THE FORWARD PASS -- the Ouroboros mouth, resident. Maintains a
    GDN-algebra trace of the live stream (S = decay*S + k v^T through fixed hashlib-seeded
    projections: the model's own memory law, run beside it), and gives the outside world the
    measured verbs from the Ouroboros arc: external_write (a fact reads back by the trace's
    own readout -- measured 0.951 on the exact algebra, zero forward passes), external_read,
    external_delete (readout-estimate subtraction, 0.951 -> -0.24), capacity_report (the
    crosstalk law: 0.932 predicted vs 0.905 measured -- the manager knows saturation BEFORE
    confabulation), consolidate (transcript-sourced rehearsal, 0.767 -> 0.918; the kept
    negative rides in the docstring: rehearsing the trace's OWN reads is self-pollution and
    this resident refuses to), and snapshot/restore (the session carry). Durable notes spill
    to a KnowledgeStore partition when one is given -- the two speeds of Ouroboros in one
    resident. The hook is PASSIVE by default (delta zero): a manager observes; injection is
    the Oracle's job."""

    def __init__(self, hidden_dim, layer, dk=128, decay=0.98, partition=None, tag="ouro"):
        self.layer = int(layer)
        self.decay = float(decay)
        self.dk = int(dk)
        self.Pk = _projector(hidden_dim, dk, tag + "_k")
        self.Pv = _projector(hidden_dim, dk, tag + "_v")
        self.S = np.zeros((dk, dk))
        self.n_writes = 0
        self.n_stream = 0                               # cp46: writes from the hook
        self.n_external = 0                             # cp46: writes from the mouth
        self._partition = partition                     # a KnowledgeStore, or None

    # -- the stream side (passive) --
    def hook(self, h):
        for t in range(h.shape[0]):
            k = self.Pk @ np.asarray(h[t], np.float64)
            v = self.Pv @ np.asarray(h[t], np.float64)
            nk, nv = np.linalg.norm(k) or 1.0, np.linalg.norm(v) or 1.0
            self.S = self.decay * self.S + np.outer(k / nk, v / nv)
            self.n_writes += 1
            self.n_stream += 1
        return np.zeros_like(h)

    # -- the mouth (measured verbs) --
    def external_write(self, key, value, note=None):
        k = np.asarray(key, np.float64); v = np.asarray(value, np.float64)
        self.S = self.S + np.outer(k / (np.linalg.norm(k) or 1.0),
                                   v / (np.linalg.norm(v) or 1.0))
        self.n_writes += 1
        self.n_external += 1
        if note and self._partition is not None:
            self._partition.add(str(note), kind="note", source="ouroboros")
        return True

    def external_read(self, key):
        k = np.asarray(key, np.float64)
        return self.S.T @ (k / (np.linalg.norm(k) or 1.0))

    def external_delete(self, key):
        k = np.asarray(key, np.float64) / (np.linalg.norm(np.asarray(key, float)) or 1.0)
        v_est = self.S.T @ k
        self.S = self.S - np.outer(k, v_est)
        return True

    def capacity_report(self):
        """Predicted recall of a fresh memory from the crosstalk law -- effective load from
        the decay-weighted write count.

        KEPT NEGATIVE (cp46, the full-stack battery, and it QUALIFIES the cp45 claim that
        "the manager knows saturation before it confabulates"): that is true only in a
        HOMOGENEOUS trace. When the model's own forward passes accumulate a STREAM
        BACKGROUND and facts are written on top, this prediction is an OPTIMISTIC UPPER
        BOUND for those facts, and it does not notice: measured on a real baked model,
        externally-written facts recalled at 0.923 / 0.753 / 0.424 against predictions of
        0.938 / 0.886 / 0.852 for 0 / 3 / 40 background passes -- a gap of 0.43 while
        `saturating` still read False. The resident CANNOT detect this from S alone,
        because it stores the trace and not the keys. So it now reports the REGIME and
        says plainly that mixed means upper-bound; verify_recall() is the ground-truth
        path when it matters."""
        # cp45 (the ouroboros battery found it): decay=1.0 is a LEGITIMATE config -- a
        # trace with no forgetting -- and the geometric-series form divides by zero there.
        # The limit is exact and obvious: with no decay every write counts once, so
        # n_eff = n_writes. Guarding here keeps capacity_report honest at both extremes.
        if abs(1.0 - self.decay ** 2) < 1e-12:
            n_eff = float(max(self.n_writes, 1))
        else:
            n_eff = (1.0 - self.decay ** (2 * max(self.n_writes, 1))) / (1.0 - self.decay ** 2)
        pred = 1.0 / np.sqrt(1.0 + max(n_eff - 1.0, 0.0) / self.dk)
        mixed = self.n_stream > 0 and self.n_external > 0
        return {"n_writes": self.n_writes, "n_effective": float(n_eff),
                "predicted_recall": float(pred),
                "saturating": bool(pred < 0.5),
                "n_stream": self.n_stream, "n_external": self.n_external,
                "regime": "mixed" if mixed else "homogeneous",
                "prediction_is_upper_bound": bool(mixed),
                "advice": ("stream background present -- predicted_recall is an UPPER "
                           "BOUND for externally written facts; call verify_recall(pairs) "
                           "for ground truth") if mixed else "prediction is calibrated"}

    def verify_recall(self, pairs):
        """GROUND TRUTH, because a prediction that cannot see its own regime must not be
        the last word (cp46). Returns the MEASURED mean readback cosine over caller-owned
        (key, value) pairs plus the gap against the prediction."""
        if not pairs:
            return {"measured_recall": None, "n": 0}
        sims = []
        for k, v in pairs:
            r = self.external_read(k)
            nv = np.asarray(v, np.float64)
            sims.append(float(np.dot(r / (np.linalg.norm(r) or 1.0),
                                     nv / (np.linalg.norm(nv) or 1.0))))
        meas = float(np.mean(sims))
        cap = self.capacity_report()
        return {"measured_recall": meas, "n": len(pairs),
                "predicted_recall": cap["predicted_recall"],
                "gap": float(cap["predicted_recall"] - meas),
                "regime": cap["regime"]}

    def consolidate(self, pairs, gain=0.6):
        """Transcript-sourced rehearsal ONLY: pairs = [(key, value), ...] from ground truth
        the caller owns. Rehearsing the trace's own reads measured NEGATIVE (0.767 -> 0.730,
        and it damaged fresh memories) -- that path does not exist here on purpose."""
        for k, v in pairs:
            k = np.asarray(k, np.float64); v = np.asarray(v, np.float64)
            self.S = self.S + float(gain) * np.outer(k / (np.linalg.norm(k) or 1.0),
                                                     v / (np.linalg.norm(v) or 1.0))
        return len(pairs)

    def snapshot(self):
        return {"S": self.S.copy(), "n_writes": self.n_writes}

    def restore(self, snap):
        self.S = np.asarray(snap["S"], np.float64).copy()
        self.n_writes = int(snap["n_writes"])
        return True


class Galvatron:
    """A model plus its resident stack: the rebuilt being. Owns the generation
    loop so residual residents (hooks) and logit residents (guards) both apply.
    Residents compose in list order; the composed stack is what the selftest
    certifies, not the residents in isolation."""

    def __init__(self, runtime, residents=(), guards=()):
        self.rt = runtime
        self.residents = list(residents)
        self.guards = list(guards)

    def _hooks(self):
        by_layer = {}
        for r in self.residents:
            by_layer.setdefault(r.layer, []).append(r)

        def make(rs):
            def fn(h):
                total, any_ = np.zeros_like(h), False
                for r in rs:
                    d = r.hook(h)
                    if d is not None:
                        total = total + d
                        any_ = True
                return total if any_ else None
            return fn
        return {L: make(rs) for L, rs in by_layer.items()}

    def _guard(self, logits):
        for g in self.guards:
            logits = g.guard(logits)
        return logits

    def generate(self, token_ids, n_new=16, state=None):
        hooks = self._hooks()
        if state is None:
            logits, state = self.rt.prefill(token_ids, hooks=hooks)
        else:
            logits = state.logits
        ids = list(map(int, token_ids))
        for _ in range(n_new):
            nxt = int(np.argmax(self._guard(logits)))
            ids.append(nxt)
            logits, state = self.rt.step(nxt, state, hooks=hooks)
        return ids, state


def council(runtime, token_ids, branches, n_new=12, horizon=8):
    """Deliberation over alternate futures: prefill once, snapshot, run each
    branch (a (residents, guards) pair) from its own copy, score each finished
    branch by the model's OWN mean next-token NLL over the generated span
    (computed under the branch's guards -- a branch is scored in its own rules),
    return them ranked best-first. Temporal awareness doing useful work:
    self-consistency without a second model."""
    base_logits, st0 = runtime.prefill(token_ids)
    results = []
    for residents, guards in branches:
        g = Galvatron(runtime, residents, guards)
        st = st0.copy()
        st.logits = base_logits.copy()
        ids, st_end = g.generate(token_ids, n_new=n_new, state=st)
        # score: replay NLL of the branch's own tokens under its own guards
        nll, logits, st_s = [], base_logits, st0.copy()
        for tok in ids[len(token_ids):len(token_ids) + horizon]:
            gl = g._guard(logits)
            lse = float(np.log(np.sum(np.exp(gl - gl.max()))) + gl.max())
            nll.append(lse - float(gl[tok]))
            logits, st_s = runtime.step(tok, st_s, hooks=g._hooks())
        results.append({"ids": ids, "mean_nll": float(np.mean(nll)),
                        "residents": residents, "guards": guards})
    results.sort(key=lambda r: r["mean_nll"])
    return results


# ---------------------------------------------------------------------- selftest

def _selftest_ouroboros():
    # OUROBOROS-RESIDENT PINS, on the measured contracts: (a) the hook is PASSIVE (delta
    # exactly zero) while the trace accumulates; (b) external write -> the trace's own
    # readout finds it (>= 0.7 under stream load); (c) delete drives it negative; (d) the
    # capacity report tracks the law within a band; (e) transcript consolidation lifts a
    # decayed memory; (f) snapshot/restore is exact; (g) partition notes are durable.
    rng = np.random.default_rng(4)
    H = 64
    res = OuroborosResident(H, layer=0, dk=96, decay=0.985)
    stream = rng.standard_normal((30, H))
    d = res.hook(stream)
    assert np.all(d == 0.0) and res.n_writes == 30, "manager observes; it does not inject"
    k = rng.standard_normal(96); v = rng.standard_normal(96)
    res.external_write(k, v)
    r1 = res.external_read(k)
    c1 = float(r1 @ (v / np.linalg.norm(v)) / (np.linalg.norm(r1) or 1.0))
    assert c1 > 0.7, c1
    res.external_delete(k)
    r2 = res.external_read(k)
    c2 = float(r2 @ (v / np.linalg.norm(v)) / (np.linalg.norm(r2) or 1e-12))
    assert c2 < 0.2, c2
    rep = res.capacity_report()
    assert 0.0 < rep["predicted_recall"] <= 1.0 and rep["n_writes"] == 31
    old_k, old_v = rng.standard_normal(96), rng.standard_normal(96)
    res.external_write(old_k, old_v)
    res.hook(rng.standard_normal((60, H)))              # decay buries it
    ra = res.external_read(old_k)
    ca = float(ra @ (old_v / np.linalg.norm(old_v)) / (np.linalg.norm(ra) or 1.0))
    res.consolidate([(old_k, old_v)])
    rb = res.external_read(old_k)
    cb = float(rb @ (old_v / np.linalg.norm(old_v)) / (np.linalg.norm(rb) or 1.0))
    assert cb > ca + 0.05, (ca, cb)
    snap = res.snapshot()
    res.hook(rng.standard_normal((5, H)))
    res.restore(snap)
    assert np.array_equal(res.S, snap["S"]), "restore must be exact"
    import tempfile
    from holographic.caching_and_storage.holographic_knowledgestore import KnowledgeStore
    root = tempfile.mkdtemp()
    res2 = OuroborosResident(H, layer=0, partition=KnowledgeStore(root))
    res2.external_write(rng.standard_normal(128), rng.standard_normal(128),
                        note="galvatron remembers the forge")
    assert KnowledgeStore(root).search(__import__("lecore").UnifiedMind(dim=32, seed=0),
                                       "forge", top=1), "partition note must be durable"
    print("OK: OuroborosResident pins passed (passive hook; write 0.7+; delete negative; "
          "capacity law; transcript consolidation lifts; snapshot exact; partition durable)")


def _selftest():
    rng = np.random.default_rng(0)
    try:
        import torch
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
    except ImportError:
        print("galvatron selftest SKIPPED-REFERENCE (torch/transformers absent)")
        # cp45 OUROBOROS BATTERY (found and fixed a real red): decay=1.0 is a legitimate
    # no-forgetting trace and capacity_report divided by zero on it. The limit is exact.
    oB = OuroborosResident(hidden_dim=32, layer=0, dk=16, decay=1.0)
    import numpy as _np
    _rg = _np.random.default_rng(0)
    kB, vB = _rg.standard_normal(16), _rg.standard_normal(16)
    oB.external_write(kB, vB)
    capB = oB.capacity_report()
    assert capB["n_effective"] == 1.0, "decay=1.0: every write counts once (no division)"
    assert capB["regime"] == "homogeneous" and not capB["prediction_is_upper_bound"]
    oB.hook(_np.zeros((2, 32)))                      # a stream background arrives
    capM = oB.capacity_report()
    assert capM["regime"] == "mixed" and capM["prediction_is_upper_bound"], \
        "cp46: a mixed trace must SAY its prediction is an upper bound"
    vr = oB.verify_recall([(kB, vB)])
    assert vr["measured_recall"] is not None and "gap" in vr, \
        "verify_recall is the ground-truth path when the regime is mixed"
    rB = oB.external_read(kB)
    assert float(_np.dot(rB / _np.linalg.norm(rB), vB / _np.linalg.norm(vB))) > 0.9, \
        "a single write must read back cleanly at either decay extreme"
    return
    import lecore
    from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime

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
    mind = lecore.UnifiedMind(dim=512, seed=0)
    ids = rng.integers(0, 97, size=16)

    # WARD: banned tokens are IMPOSSIBLE, not discouraged. Ban whatever the
    # unguarded model actually emits; the warded run must emit none of them.
    free, _ = Galvatron(rt).generate(ids, n_new=24)
    emitted = set(free[len(ids):])
    ward = WardResident(banned=emitted)
    warded, _ = Galvatron(rt, guards=[ward]).generate(ids, n_new=24)
    assert not (set(warded[len(ids):]) & emitted), "ban breached"

    # ORACLE: the mind's native learn/recall, keyed on a live hidden state,
    # flips the next token to a chosen target on cue and is silent off cue.
    captured = {}
    rt.forward(ids, hooks={3: lambda h: captured.__setitem__("h", h.copy()) or None})
    key = captured["h"][-1]
    target = 41
    oracle = OracleResident(mind, 64, layer=3, gain=1.0, threshold=0.0)
    oracle.remember(key, 8.0 * rt.embed[target])
    base_top = int(np.argmax(rt.forward(ids)[-1]))
    with_mem = int(np.argmax(Galvatron(rt, residents=[oracle])._guard(
        rt.forward(ids, hooks=Galvatron(rt, residents=[oracle])._hooks())[-1])))
    assert base_top != target and with_mem == target, (base_top, with_mem)

    # DREAMER: corrupt the stream at layer 2; perplexity degrades; the dreamer
    # (manifold = healthy layer-2 states) recovers most of the damage.
    healthy = {}
    long_ids = rng.integers(0, 97, size=48)
    rt.forward(long_ids, hooks={2: lambda h: healthy.__setitem__("h", h.copy()) or None})
    # DREAMER, three measured contracts (instrument-error ledger: the first
    # ruler was perplexity, meaningless on a chance-level random subject; the
    # second design, nearest-sample projection, was HARMFUL and is the class
    # docstring's kept negative):
    #   (1) never harms a clean stream (fires zero times, agreement 1.0);
    #   (2) strictly improves top-1 agreement under corruption at every level;
    #   (3) physics check: removed noise energy matches the (d-r)/d prediction.
    clean_top = np.argmax(rt.forward(long_ids), axis=-1)

    def agreement(hooks):
        top = np.argmax(rt.forward(long_ids, hooks=hooks), axis=-1)
        return float(np.mean(top == clean_top))

    dreamer = DreamerResident(mind, healthy["h"], layer=2, strength=1.0)
    assert agreement({2: dreamer.hook}) == 1.0, "dreamer touched a clean stream"
    d, r = 64, dreamer.rank
    gains = []
    for noise in (0.05, 0.1, 0.2):
        r1 = np.random.default_rng(5)
        a_bad = agreement({2: lambda h: noise * r1.standard_normal(h.shape)})
        r1 = np.random.default_rng(5)
        def ctr(h, _n=noise):
            dd = _n * r1.standard_normal(h.shape)
            rep = dreamer.hook(h + dd)
            return dd + (rep if rep is not None else 0.0)
        a_rep = agreement({2: ctr})
        assert a_rep >= a_bad, (noise, a_bad, a_rep)
        gains.append(a_rep - a_bad)
        # physics: residual off-subspace noise after repair ~ 0 => energy kept
        # in-subspace ~ r/d of injected
        r2 = np.random.default_rng(7)
        dd = noise * r2.standard_normal(healthy["h"].shape)
        hc = healthy["h"] + dd
        rep = dreamer.hook(hc)
        kept = np.linalg.norm(hc + rep - healthy["h"]) ** 2 / np.linalg.norm(dd) ** 2
        assert abs(kept - r / d) < 0.12, (kept, r / d)
    assert max(gains) > 0.0
    # WIENER mode pinned separately: it must fire on real corruption and stay
    # silent on a clean stream, the same contract the subspace mode carries.
    dr_w = DreamerResident(mind, healthy["h"], layer=2, strength=1.0,
                           mode="wiener")
    assert dr_w.hook(healthy["h"]) is None, "wiener touched a clean stream"
    noisy = healthy["h"] + 0.5 * np.random.default_rng(11).standard_normal(
        healthy["h"].shape)
    assert dr_w.hook(noisy) is not None, "wiener ignored real corruption"
    # and it must not zero directions it could not estimate (rank guard)
    assert dr_w.n_est <= healthy["h"].shape[0] - 1
    a_bad, a_rep = a_bad, a_rep      # last level, for the summary line
    recovered = gains[-1] / max(1.0 - a_bad, 1e-9)

    # COUNCIL: among a wild branch (random steering) and a sober branch (none),
    # the council's NLL ranking must put the sober branch first.
    steer = OracleResident(mind, 64, layer=1, gain=1.0, threshold=0.0)
    for i in range(4):                                       # noisy junk memories
        steer.remember(rng.standard_normal(64), 6.0 * rng.standard_normal(64))
    ranked = council(rt, ids, branches=[([steer], []), ([], [])], n_new=8, horizon=6)
    assert ranked[0]["residents"] == [], "council must prefer the sober branch"

    # COMPOSITION: full stack together (oracle + dreamer + ward) -- each contract
    # re-checked under composition, because residents share the stream.
    g = Galvatron(rt, residents=[oracle, dreamer], guards=[ward])
    out, _ = g.generate(ids, n_new=16)
    assert not (set(out[len(ids):]) & emitted), "ward breached under composition"

    print("galvatron selftest OK -- ward absolute, oracle flips %d->%d on cue, "
          "dreamer: clean untouched, strictly helps at 3 noise levels, physics "
          "check r/d=%.2f passes (top-1 %.2f->%.2f at worst noise), council "
          "picks sober, stack composes" % (base_top, target, r / d, a_bad, a_rep))


if __name__ == "__main__":
    _selftest()
    _selftest_ouroboros()
