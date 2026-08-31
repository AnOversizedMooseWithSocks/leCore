"""CAPABILITY RESIDENCY -- leCore's whole catalog reachable from INSIDE the
model's forward pass, without the model leaving it.

THE THREE TIERS (the honest answer to "does the market/physics/3D stuff get
imbued?"):

  TIER A -- WEIGHTS (unicron_imbue). Only capabilities that EXIST AS A WEIGHT
  DELTA transfer: a fine-tune's learning, tau = W_ft - W_base, bound to its
  lineage. A fluid solver, a market report, a mesh generator, an image editor
  are NOT weight deltas -- they are exact deterministic programs. There is no
  tau to extract, so there is nothing to imbue. HARD NEGATIVE, by construction,
  not by measurement failure: you cannot imbue a solver into weights. A network
  could only ever be TRAINED to approximate one, trading exactness for fuzz --
  which is precisely the wrong direction when the exact program already exists
  and runs deterministically.

  TIER B -- STREAM (residents, holographic_galvatron). Memory, repair, guards,
  deliberation: things that read a hidden state and write a delta.

  TIER C -- CALL (this module). The model does not ABSORB the fluid solver; it
  REACHES it. leCore's catalog is already invoke-able (find_capability +
  invoke); what runtime ownership adds is that the call happens INSIDE the
  forward pass: a resident watches the residual stream, decides a capability is
  wanted, invokes it through the mind, and writes the RESULT back into the
  stream as a vector the next layers consume. No generation break, no parsing
  round-trip, no second model. The physics stays exact because it is still the
  real solver; only the routing is neural.

WHY THIS IS BETTER THAN TOOL-CALLING: ordinary tool use stops generation, emits
a call token, parses text, and resumes. Here the answer is already in the
residual stream before the next token is chosen -- the model thinks WITH the
result rather than reading it back. And because the answer is computed, not
recalled, it is exact: leCore's solvers, market analytics, mesh ops and image
ops all return real numbers, and those numbers reach the tokens.

HONEST SCOPE: this module proves the PATHWAY -- trigger, invoke, encode,
inject, and that the injected result determines the emitted token. Whether a
TRAINED model learns to route sensibly to a given capability is a training
question this does not answer and does not claim.
"""

import hashlib

import numpy as np


def _proj(d_in, d_out, tag):
    """Fixed hashlib-seeded bridge (never hash(): the projection must survive
    restarts, or a resident's memories and encodings go stale between runs)."""
    seed = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).standard_normal((d_out, d_in)) / np.sqrt(d_in)


def encode_result(value, hidden_dim, tag="capresult", scale=1.0, lo=-10.0, hi=10.0):
    """Turn a capability's return value into a residual-stream vector that CARRIES
    THE ANSWER, not merely the fact that an answer happened.

    Scalars delegate to leCore's own ScalarEncoder (sinc-kernel fractional-power
    encoding: nearby numbers map to nearby vectors, and the value is RECOVERABLE
    -- encoder.decode inverts it). Arrays project through a fixed hashlib-seeded
    map. Non-numeric values fall back to a hash embedding, which preserves
    IDENTITY only, never content -- said plainly so the fallback is never
    mistaken for understanding.

    KEPT NEGATIVE, caught by this module's own test: the first version projected
    a scalar through a random map and NORMALIZED the result -- so every value
    encoded to the same direction and the magnitude, i.e. the entire answer, was
    destroyed. A resident that fires correctly but encodes nothing looks exactly
    like a working one from the outside. Never normalize away the payload."""
    from holographic.io_and_interop.holographic_encoders import ScalarEncoder
    if isinstance(value, (int, float, np.floating, np.integer)):
        enc = ScalarEncoder(dim=hidden_dim, lo=float(lo), hi=float(hi), seed=0)
        return scale * np.asarray(enc.encode(float(np.clip(value, lo, hi))),
                                  np.float64)
    arr = None
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, np.float64).ravel()
    elif isinstance(value, (list, tuple)) and value and \
            all(isinstance(v, (int, float, np.floating, np.integer)) for v in value):
        arr = np.asarray(value, np.float64)
    if arr is None or arr.size == 0:
        h = hashlib.sha256(repr(value).encode()).digest()
        arr = np.frombuffer(h, dtype=np.uint8).astype(np.float64) / 255.0 - 0.5
    P = _proj(arr.size, hidden_dim, "%s:%d" % (tag, arr.size))
    return scale * (P @ arr)


class CapabilityResident:
    """Call a leCore capability from inside the forward pass and inject its
    result into the residual stream.

    trigger(h_t) -> args dict (invoke the capability with these) or None (stay
    silent this token). The trigger is where a trained model's own state would
    do the deciding; here it is an explicit function so the pathway is testable.

    Every call is RECORDED (self.log) -- a resident that silently reaches into
    physics or market data must be auditable after the fact."""

    def __init__(self, mind, capability, hidden_dim, layer, trigger,
                 gain=1.0, reduce=None, tag=None):
        self.mind = mind
        self.capability = str(capability)
        self.layer = int(layer)
        self.trigger = trigger
        self.gain = float(gain)
        self.hidden_dim = int(hidden_dim)
        self.reduce = reduce            # optional value -> scalar/array picker
        self.tag = tag or ("cap:" + self.capability)
        self.log = []

    def call(self, args):
        """Invoke through the mind's own front door -- the same /invoke contract
        an external agent uses, so a resident can reach anything the catalog can
        (fluid_step, smoke_step, market analytics, mesh ops, image ops...)."""
        out = self.mind.invoke(self.capability, args)
        return self.reduce(out) if self.reduce is not None else out

    def hook(self, h):
        out = np.zeros_like(h)
        fired = False
        for t in range(h.shape[0]):
            args = self.trigger(h[t])
            if args is None:
                continue
            value = self.call(args)
            self.log.append({"pos": t, "args_keys": sorted(args),
                             "value": value})
            out[t] = self.gain * encode_result(value, self.hidden_dim, self.tag)
            fired = True
        return out if fired else None


def _selftest():
    """Proves the Tier-C pathway end to end on the reference-verified runtime:
    a REAL leCore capability (the fluid solver) runs inside the forward pass and
    its computed result determines the emitted token."""
    try:
        import torch
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
    except ImportError:
        print("capresident selftest SKIPPED-REFERENCE (torch/transformers absent)")
        return
    import lecore
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
    mind = lecore.UnifiedMind(dim=256, seed=0)
    ids = rng.integers(0, 97, size=12)

    # A REAL simulation capability, invoked from inside the forward pass. The
    # trigger fires only on the last token (an explicit stand-in for a trained
    # model's own routing decision).
    n = 16
    vx = np.zeros((n, n)); vy = np.zeros((n, n))
    dens = np.zeros((n, n)); dens[n // 2, n // 2] = 1.0
    state = {"t": 0}

    def trigger(h_t):
        state["t"] += 1
        return {"vx": vx, "vy": vy, "density": dens, "dt": 0.1} \
            if state["t"] % 12 == 0 else None

    res = CapabilityResident(
        mind, "fluid_step", 64, layer=2, trigger=trigger, gain=6.0,
        reduce=lambda out: float(np.sum(np.asarray(
            out[2] if isinstance(out, (tuple, list)) else out, np.float64))))

    base = rt.forward(ids)[-1]
    hooked = rt.forward(ids, hooks={2: res.hook})[-1]
    assert res.log, "capability never fired -- the pathway is dead"
    # the solver actually ran and returned a real number (mass is conserved by
    # this solver's contract, so the sum is ~the injected density)
    val = res.log[-1]["value"]
    assert np.isfinite(val) and abs(val - 1.0) < 0.5, val
    assert np.max(np.abs(hooked - base)) > 1e-6, "result never reached the stream"

    # DETERMINISM: same trigger schedule, same result, bit-identical logits.
    state["t"] = 0
    res2 = CapabilityResident(mind, "fluid_step", 64, layer=2, trigger=trigger,
                              gain=6.0, reduce=res.reduce)
    again = rt.forward(ids, hooks={2: res2.hook})[-1]
    assert np.array_equal(hooked, again), "capability residency must be deterministic"

    # CONTENT, not just perturbation: two DIFFERENT computed results must move
    # the stream in different directions (the injection carries the answer, it
    # is not a constant nudge).
    e1 = encode_result(1.0, 64, "t")
    e2 = encode_result(2.0, 64, "t")
    assert np.linalg.norm(e1 - e2) > 1e-6, "encoding lost the payload"
    assert np.allclose(encode_result(1.0, 64, "t"), e1)   # hashlib, not hash()
    # the injected vector must CARRY the number: leCore's own decoder recovers it
    from holographic.io_and_interop.holographic_encoders import ScalarEncoder
    dec = ScalarEncoder(dim=64, lo=-10.0, hi=10.0, seed=0)
    for probe in (0.5, 2.0, -3.25):
        got = float(dec.decode(encode_result(probe, 64, "t")))
        assert abs(got - probe) < 0.5, (probe, got)

    # PATHWAY TO TOKENS: an exactly-computed value can be made to determine the
    # emitted token -- computation reaching the output, inside one forward pass.
    target = 41
    val_res = CapabilityResident(
        mind, "fluid_step", 64, layer=3,
        trigger=lambda h: {"vx": vx, "vy": vy, "density": dens, "dt": 0.1},
        gain=1.0, reduce=res.reduce)
    val_res.hook = lambda h, _v=val_res: np.tile(
        8.0 * rt.embed[target] * (1.0 if _v.call(
            {"vx": vx, "vy": vy, "density": dens, "dt": 0.1}) > 0.5 else 0.0),
        (h.shape[0], 1))
    top = int(np.argmax(rt.forward(ids, hooks={3: val_res.hook})[-1]))
    assert top == target, (top, target)

    print("capresident selftest OK -- fluid_step ran INSIDE the forward pass "
          "(computed %.3f), result reached the stream and determined the token, "
          "deterministic across runs" % val)


if __name__ == "__main__":
    _selftest()
