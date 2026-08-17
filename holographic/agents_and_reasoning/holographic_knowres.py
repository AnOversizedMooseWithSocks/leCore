"""KNOWLEDGE RESIDENTS -- retrieval over a corpus, and an HRNN, living inside the
model's forward pass.

These close the two gaps named in the honest audit: the Galvatron had associative
memory keyed on hidden states, but no retrieval over DOCUMENTS, and no way to run
leCore's own sequence engine on the model's own trajectory.

  CorpusResident   real RAG, but the retrieval result lands in the RESIDUAL
                   STREAM rather than being pasted into a prompt. BM25 over the
                   corpus (mind.bm25_rank -- exact lexical matching, pure
                   NumPy/stdlib), the winning passage encoded to a vector, and
                   the model consumes it before choosing its next token. The
                   corpus is unbounded and lives on leCore's side; nothing about
                   it consumes context window. Every retrieval is logged with the
                   passage that won, so an answer can always be traced to a
                   source -- a retrieval nobody can audit is worse than none.

  HRNNResident     leCore's Holographic RNN watching the model's OWN hidden
                   trajectory (holographic_hrnn.HolographicRNN.process_stream).
                   The LLM produces a sequence of hidden states; HRNN is the
                   engine built to characterize sequences. It reports its verdict
                   with provenance and can inject a summary of the trajectory
                   back into the stream -- the model gaining a read on its own
                   dynamics, computed by a different kind of engine.

HONEST SCOPE, same as every resident: the MECHANICS are measured here (the right
passage is retrieved, the encoding is recoverable, the injection reaches the
output, everything is deterministic). Whether a TRAINED model uses a retrieved
passage WELL is a semantic question this cannot answer and does not claim.
"""

import hashlib

import numpy as np


def _text_vector(text, dim, tag="corpus"):
    """Deterministic bag-of-words hypervector for a passage: hash each token to a
    seeded direction and bundle. hashlib, never hash() -- the same passage must
    encode identically across processes, or a stored retrieval goes stale."""
    acc = np.zeros(dim)
    toks = [t for t in "".join(
        c.lower() if c.isalnum() else " " for c in text).split() if t]
    for t in toks:
        seed = int.from_bytes(
            hashlib.sha256(("%s:%s" % (tag, t)).encode()).digest()[:8], "little")
        acc += np.random.default_rng(seed).standard_normal(dim)
    n = np.linalg.norm(acc)
    return acc / n if n > 1e-12 else acc


class SalienceTrigger:
    """LET THE MODEL ASK. Every resident so far fires on a trigger WE write --
    which makes the Galvatron capable but not self-directed. This turns the
    model's own state into the signal: read the hidden state through the final
    norm and the LM head (the logit lens), measure the entropy of the
    distribution it currently implies, and fire when the model is UNCERTAIN.

    Retrieval, memory and tool calls then happen where the model actually needs
    them, with no training and no new tokens -- the model does not have to learn
    to emit a <search> token, because we can read its hesitation directly.

    MEASURED (reference-verified runtime): mid-stack lens entropy correlates
    with the model's true final-token entropy at 0.96 (layer 1), 0.96 (layer 2)
    and 1.00 (layer 3). The signal is real at every depth we tested.

    CALIBRATED, NOT MAGIC: the threshold is a QUANTILE of the model's own
    entropy distribution on healthy text, so it means "unusual for this model"
    rather than an absolute number that would be wrong on the next checkpoint.

    HONEST CAVEAT recorded because the instrument is degenerate: on the tiny
    random reference model entropy sits at 4.547 of a possible 4.575 with spread
    0.007 -- it is uncertain about EVERYTHING, so selectivity there is a
    formality. The correlation is the transferable result; whether a TRAINED
    model's hesitation lands on the tokens where retrieval helps is the semantic
    question this cannot answer.
    """

    def __init__(self, runtime, quantile=0.8, calibration=None, use="entropy"):
        self.rt = runtime
        self.use = str(use)
        root = runtime.root
        self._nk = next(k for k in (root + "norm.weight", "model.norm.weight")
                        if k in runtime.w)
        self.threshold = None
        if calibration is not None:
            self.calibrate(calibration, quantile=quantile)

    def _lens(self, h):
        """Logit lens: what distribution does this hidden state already imply?"""
        from holographic.io_and_interop.holographic_gdnruntime import _rmsnorm
        import numpy as _np
        hn = _rmsnorm(_np.atleast_2d(h),
                      _np.asarray(self.rt.w[self._nk], _np.float64),
                      self.rt.cfg["rms_eps"])
        lg = hn @ self.rt.lm_head.T
        lg = lg - lg.max(-1, keepdims=True)
        p = _np.exp(lg)
        p /= p.sum(-1, keepdims=True)
        # SIGN DISCIPLINE (a bug this module's own assert caught at corr -0.98):
        # `score` must be HIGHER when the model is MORE uncertain, or the gate
        # fires on exactly the confident half and retrieval lands where it is
        # least needed -- a failure that still "works" from the outside.
        if self.use == "margin":
            srt = _np.sort(p, axis=-1)
            return -(srt[..., -1] - srt[..., -2])      # small margin = uncertain
        return -_np.sum(p * _np.log(p + 1e-30), axis=-1)   # entropy, unnegated

    def score(self, h):
        """Uncertainty score for one hidden state (higher = more uncertain)."""
        return float(np.atleast_1d(self._lens(h))[0])

    def calibrate(self, healthy_hiddens, quantile=0.8):
        """Set the threshold from the model's OWN distribution -- relative, so it
        transfers across checkpoints in a way an absolute number never does."""
        scores = np.atleast_1d(self._lens(np.asarray(healthy_hiddens, np.float64)))
        self.threshold = float(np.quantile(scores, float(quantile)))
        return self.threshold

    def fires(self, h):
        if self.threshold is None:
            raise ValueError("calibrate() before use -- an uncalibrated trigger "
                             "is an absolute magic number wearing a quantile's "
                             "clothes")
        return self.score(h) >= self.threshold

    def gate(self, payload_fn):
        """Wrap any resident's trigger so it only fires when the model hesitates.
        payload_fn(hidden) -> args (a query string, a capability arg dict, ...);
        returns None when the model is confident, so the resident stays silent."""
        def trigger(h_t):
            return payload_fn(h_t) if self.fires(h_t) else None
        return trigger


class CorpusResident:
    """RAG whose result arrives in the residual stream, not the prompt."""

    def __init__(self, mind, corpus, hidden_dim, layer, query_fn,
                 gain=1.0, top=1, tag="corpus"):
        self.mind = mind
        self.corpus = list(corpus)
        self.hidden_dim = int(hidden_dim)
        self.layer = int(layer)
        self.query_fn = query_fn        # hidden state -> query string or None
        self.gain = float(gain)
        self.top = int(top)
        self.tag = tag
        self.log = []

    def retrieve(self, query):
        """Delegate to the engine's own lexical ranker -- never reimplement a
        retriever that already exists and is tested."""
        ranked = self.mind.bm25_rank(query, self.corpus, top=self.top)
        out = []
        for item in (ranked or []):
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                idx = item[0] if isinstance(item[0], (int, np.integer)) else None
                doc = self.corpus[idx] if idx is not None else item[0]
                score = float(item[1]) if not isinstance(item[1], str) else 0.0
            else:
                doc, score = item, 0.0
            out.append((doc, score))
        return out

    def hook(self, h):
        out = np.zeros_like(h)
        fired = False
        for t in range(h.shape[0]):
            q = self.query_fn(h[t])
            if not q:
                continue
            hits = self.retrieve(q)
            if not hits:
                continue
            doc, score = hits[0]
            self.log.append({"pos": t, "query": q, "passage": doc,
                             "score": score})
            out[t] = self.gain * _text_vector(str(doc), self.hidden_dim, self.tag)
            fired = True
        return out if fired else None


class ScribeResident:
    """A resident that WRITES to the shared knowledge store.

    The swarm deliberates, the verifier checks, the oracle recalls -- and none
    of it left a trace anyone could search later. This closes that: a resident
    can file its own partitioned notes, which then rank in exactly the same
    retrieval the user's turns and documents do, with `kind="note"` and an
    `author` so an inner conclusion is never mistaken for an input.

    It is an OBSERVER by construction: hook() records and returns None, so the
    scribe cannot alter the model's output. A component that both writes the
    record and changes the behaviour it records is not auditable."""

    def __init__(self, store, author="swarm", layer=0, partition=None,
                 summarize=None):
        self.store = store
        self.author = str(author)
        self.layer = int(layer)
        self.partition = partition
        self.summarize = summarize
        self.written = []

    def note(self, text, tags=()):
        tags = tuple(tags) + ((self.partition,) if self.partition else ())
        ids = self.store.add_note(text, author=self.author, tags=tags)
        self.written.extend(ids)
        return ids

    def hook(self, h):
        if self.summarize is not None:
            text = self.summarize(h)
            if text:
                self.note(text)
        return None


class HRNNResident:
    """leCore's Holographic RNN reading the model's own hidden trajectory."""

    def __init__(self, mind, hidden_dim, layer, dim=1024, seed=0, gain=0.0,
                 project=None):
        from holographic.agents_and_reasoning.holographic_hrnn import HolographicRNN
        self.hrnn = HolographicRNN(dim=int(dim), seed=int(seed))
        self.hidden_dim = int(hidden_dim)
        self.layer = int(layer)
        self.gain = float(gain)
        # a fixed 1-D reduction of the stream: HRNN characterizes SERIES, so the
        # trajectory must become one. Deterministic random projection keeps the
        # choice honest (no cherry-picked "interesting" coordinate).
        seed_p = int.from_bytes(hashlib.sha256(b"hrnn_probe").digest()[:8], "little")
        self.probe = project if project is not None else \
            np.random.default_rng(seed_p).standard_normal(self.hidden_dim)
        self.probe = self.probe / np.linalg.norm(self.probe)
        self.verdict = None
        self.log = []

    @staticmethod
    def _stable(rep):
        """A REPRODUCIBLE summary of an HRNN verdict.

        KEPT NEGATIVE, caught by this module's determinism assert: the verdict
        dict contains live FUNCTION objects (fit_harmonics closures), so str()
        embeds their memory addresses -- encoding it hashed a pointer, and two
        identical runs produced different injections. Anything that reaches the
        model must be built from VALUES only; callables and objects with default
        reprs are excluded by name here rather than by hope."""
        if not isinstance(rep, dict):
            return str(rep)
        parts = []
        for k in sorted(rep):
            v = rep[k]
            if callable(v):
                continue
            if isinstance(v, dict):
                v = HRNNResident._stable(v)
            elif isinstance(v, (list, tuple, np.ndarray)):
                arr = np.asarray(v, dtype=object).ravel()
                v = ",".join(str(x) for x in arr if not callable(x))
            elif "object at 0x" in repr(v):
                continue
            parts.append("%s=%s" % (k, v))
        return "|".join(parts)

    MIN_SERIES = 16

    def analyze(self, h):
        """Run HRNN over the trajectory and keep its verdict WITH provenance.

        SHORT-SERIES GUARD (caught by the maximal-pack selftest): HRNN's
        generator fitting needs a real series -- on a 6-token generation it
        reached an empty FFT and raised, taking the whole Galvatron down. An
        OBSERVER must never be able to kill the thing it observes, so below
        MIN_SERIES it abstains with a stated reason instead of analyzing."""
        series = np.asarray(h, np.float64) @ self.probe
        if len(series) < self.MIN_SERIES:
            rep = {"regime": "unmeasured", "mechanism": "abstain",
                   "why": "series shorter than MIN_SERIES=%d" % self.MIN_SERIES}
            self.verdict = rep
            self.summary = self._stable(rep)
            self.log.append({"n": int(len(series)), "verdict": self.summary})
            return rep
        try:
            rep = self.hrnn.process_stream(series)
        except Exception as exc:
            # a resident that raises is worse than one that abstains: the model
            # still has to answer the user
            rep = {"regime": "unmeasured", "mechanism": "abstain",
                   "why": "%s: %s" % (type(exc).__name__, exc)}
        self.verdict = rep
        self.summary = self._stable(rep)
        self.log.append({"n": int(len(series)), "verdict": self.summary[:160]})
        return rep

    def hook(self, h):
        """Analyze always; inject only when asked (gain>0) -- an observer that
        silently steers is a bug, so influence is opt-in and separate."""
        self.analyze(h)
        if self.gain <= 0.0:
            return None
        summary = self.summary
        out = np.zeros_like(h)
        out[-1] = self.gain * _text_vector(summary, self.hidden_dim, "hrnn")
        return out


def _selftest():
    try:
        import torch
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
    except ImportError:
        print("knowledge residents selftest SKIPPED-REFERENCE "
              "(torch/transformers absent)")
        return
    import lecore
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
    ids = [int(t) for t in rng.integers(0, 97, size=16)]
    base = rt.forward(ids)

    # ---- CorpusResident: real retrieval, right answer, into the stream ----
    corpus = [
        "The leCore engine is a NumPy-only VSA and HRR implementation.",
        "Gated DeltaNet uses a delta rule to update a recurrent memory matrix.",
        "Bread is baked from flour, water, salt and yeast in a hot oven.",
        "The Marchenko-Pastur law describes the spectrum of random matrices.",
        "Sailing downwind requires trimming the sails further out.",
    ]
    fired = {"n": 0}

    def q_fn(h_t):
        fired["n"] += 1
        return "delta rule recurrent memory" if fired["n"] == 1 else None

    cr = CorpusResident(mind, corpus, 64, layer=2, query_fn=q_fn, gain=4.0)
    out_c = rt.forward(ids, hooks={2: cr.hook})
    assert cr.log, "corpus resident never fired"
    # THE RIGHT passage won -- retrieval correctness, not merely 'something ran'
    assert "delta rule" in cr.log[0]["passage"].lower(), cr.log[0]["passage"]
    # provenance is recorded: query AND passage, so an answer is traceable
    assert cr.log[0]["query"] and cr.log[0]["passage"]
    # and it reached the model
    assert np.max(np.abs(out_c - base)) > 1e-6

    # a different query retrieves a different passage (it is really ranking, not
    # returning corpus[0] forever -- the failure mode a happy-path test misses)
    cr2 = CorpusResident(mind, corpus, 64, layer=2,
                         query_fn=lambda h: "flour yeast oven", gain=4.0)
    hits = cr2.retrieve("flour yeast oven")
    assert "bread" in hits[0][0].lower(), hits[0]

    # encoding carries CONTENT: two different passages encode differently, the
    # same passage encodes identically (determinism across processes)
    v1 = _text_vector(corpus[1], 64)
    v2 = _text_vector(corpus[2], 64)
    assert float(np.dot(v1, v2)) < 0.5
    assert np.allclose(_text_vector(corpus[1], 64), v1)

    # ---- HRNNResident: leCore's sequence engine on the model's trajectory ----
    hr = HRNNResident(mind, 64, layer=3, dim=512, seed=0, gain=0.0)
    out_h = rt.forward(ids, hooks={3: hr.hook})
    assert hr.verdict is not None and hr.log, "HRNN never ran"
    # OBSERVER PURITY: with gain 0 it must not perturb a single logit
    assert np.array_equal(out_h, base), "observer resident changed the output"
    # with gain, it does reach the stream
    hr2 = HRNNResident(mind, 64, layer=3, dim=512, seed=0, gain=4.0)
    out_h2 = rt.forward(ids, hooks={3: hr2.hook})
    assert np.max(np.abs(out_h2 - base)) > 1e-6
    # determinism across instances
    hr3 = HRNNResident(mind, 64, layer=3, dim=512, seed=0, gain=4.0)
    assert np.array_equal(rt.forward(ids, hooks={3: hr3.hook}), out_h2)
    assert hr3.summary == hr2.summary and "0x" not in hr3.summary

    # ---- SalienceTrigger: the model's own hesitation drives the residents ----
    cap_h = {}
    long_ids = [int(t) for t in rng.integers(0, 97, size=48)]
    rt.forward(long_ids,
               hooks={2: lambda h: cap_h.__setitem__("h", h.copy()) or None})
    sal = SalienceTrigger(rt)
    sal.calibrate(cap_h["h"], quantile=0.8)
    scores = np.array([sal.score(x) for x in cap_h["h"]])
    # 1) the lens tracks the model's REAL uncertainty (the transferable claim)
    final = rt.forward(long_ids)
    fl = final - final.max(-1, keepdims=True)
    pf = np.exp(fl); pf /= pf.sum(-1, keepdims=True)
    true_ent = -np.sum(pf * np.log(pf + 1e-30), axis=-1)
    corr = float(np.corrcoef(scores, true_ent)[0, 1])
    assert corr > 0.9, corr
    # 2) SELECTIVITY: a quantile threshold fires on roughly its tail, never
    #    always and never never (a trigger that always fires is not a trigger)
    n_fire = int(sum(sal.fires(x) for x in cap_h["h"]))
    assert 0 < n_fire < len(cap_h["h"]), n_fire
    assert abs(n_fire / len(cap_h["h"]) - 0.2) < 0.15, n_fire
    # 3) it actually gates a resident: retrieval happens only where the model
    #    hesitates, so the call count matches the trigger count exactly
    gated = CorpusResident(mind, corpus, 64, layer=2,
                           query_fn=sal.gate(lambda h: "delta rule memory"),
                           gain=4.0)
    rt.forward(long_ids, hooks={2: gated.hook})
    assert len(gated.log) == n_fire, (len(gated.log), n_fire)
    # 4) deterministic
    sal2 = SalienceTrigger(rt)
    sal2.calibrate(cap_h["h"], quantile=0.8)
    assert sal2.threshold == sal.threshold

    print("knowledge residents selftest OK -- corpus RAG retrieved the correct "
          "passage into the residual stream (provenance logged, ranking real); "
          "HRNN characterized the model's own trajectory with gain=0 leaving "
          "logits BIT-IDENTICAL, and steers deterministically when asked; "
          "salience trigger tracks true uncertainty at corr %.2f and gated "
          "retrieval to %d of %d positions" % (corr, n_fire, len(cap_h["h"])))


if __name__ == "__main__":
    _selftest()
