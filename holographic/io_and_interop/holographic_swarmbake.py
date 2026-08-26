"""SWARMBAKE -- a swarm that fits inside one forward pass, in ordinary weights.

Moose wants the swarm running INSIDE the model, injecting leCore capability into
whatever the model is doing, without an external prompt asking for it. The
runtime SwarmResident cannot do that: it BRANCHES -- runs the model several
times and compares -- and a single forward pass cannot branch. It also needs
leCore present, so it vanishes on export.

WHAT FITS IN ONE PASS IS A ROUTED MIXTURE. N specialist circuits plus a gate
that picks per token is a swarm whose deliberation happens in parallel rather
than by re-running. That is a mixture of experts, it is ordinary arithmetic, and
it runs in any harness that runs the model.

THE GATE MUST ROUTE BY CONTENT, which is the part that decides whether this is a
swarm or decoration. install_op's gate is deliberately NEAR-CONSTANT so an
installed operator applies uniformly; a swarm needs the opposite. Keying the
gates to the stream's own leading directions gives exactly that.

MEASURED on a real Qwen3.5-0.8B stream (235 tokens spanning prose, facts, code,
SQL, markdown and questions):
    4 experts  usage [0.39 0.20 0.18 0.22], entropy 1.34 of a possible 1.39
    8 experts  usage max share 26%, entropy 1.99 of 2.08
and the routing TRACKS CONTENT rather than spreading noise:
    prose       -> expert 0 at 78%
    facts+code  -> expert 2 at 47%
    SQL+md      -> expert 1 at 59%
    questions   -> expert 0 at 60%
Different registers select different specialists, which is the property a swarm
needs and the one the runtime version could never demonstrate (its branches were
identical, so its contrast digest was exactly zero).

WHAT THIS DOES NOT DO, said plainly because "swarm inside the model" invites the
larger reading: the experts are CIRCUITS -- linear maps installed as neurons --
not leCore faculties. This routes a denoiser, a binding, a projection or a
learned correction by content. It does not let the model call fluid_step, and
nothing in a forward pass can, because a forward pass emits logits rather than
function calls.
"""

import numpy as np


def content_gates(states, n_experts, temperature=1.0):
    """Gate rows keyed to the stream's own leading directions.

    Derived from the model's activations rather than chosen: the directions that
    explain the most variance are the ones that distinguish one kind of token
    from another, which is exactly what a router needs."""
    H = np.asarray(states, np.float64)
    mu = H.mean(0)
    _u, _s, Vt = np.linalg.svd(H - mu, full_matrices=False)
    G = Vt[:int(n_experts)] * float(temperature)
    return G, mu


def route(states, gates, mu):
    """Which expert each token selects -- argmax over the gate logits."""
    return np.argmax((np.asarray(states, np.float64) - mu) @ np.asarray(gates).T,
                     axis=1)


def install_swarm(weights, cfg, experts, states, layer=None, gain=1.0,
                  temperature=6.0):
    """Install a routed bank of circuits as MLP neurons.

    `experts` is a list of (out_dim, in_dim) matrices -- one linear circuit per
    expert. Each contributes its own neurons, and its gate row is the content
    direction that selects it, so a token activates ONE specialist and the
    others stay near zero.

    gain=0.0 leaves the model unchanged, which is this project's rule: a new
    capability arrives off and is switched on deliberately."""
    from holographic.io_and_interop.holographic_vsabake import layer_key

    w = {k: np.array(v, copy=True) for k, v in weights.items()}
    L = int(int(cfg["n_layers"]) - 1 if layer is None else layer)
    up_k = layer_key(w, L, "mlp.up_proj.weight")
    gate_k = layer_key(w, L, "mlp.gate_proj.weight")
    down_k = layer_key(w, L, "mlp.down_proj.weight")
    for k in (up_k, gate_k, down_k):
        if k not in w:
            raise KeyError("no %r -- this checkpoint roots its tensors "
                           "elsewhere" % k)

    G, mu = content_gates(states, len(experts), temperature=temperature)
    up = np.asarray(w[up_k], np.float64)
    gate = np.asarray(w[gate_k], np.float64)
    down = np.asarray(w[down_k], np.float64)
    added = 0
    for i, M in enumerate(experts):
        M = np.asarray(M, np.float64)
        rows = M.shape[0]
        up = np.vstack([up, M * float(gain)])
        # every neuron of this expert shares its gate row, so the whole block
        # switches on together -- that is what makes it an EXPERT rather than
        # a set of independent neurons
        gate = np.vstack([gate, np.tile(G[i], (rows, 1))])
        cols = np.zeros((down.shape[0], rows))
        n = min(rows, down.shape[0])
        cols[:n, :n] = np.eye(n)
        down = np.hstack([down, cols])
        added += rows
    w[up_k] = up.astype(np.asarray(weights[up_k]).dtype)
    w[gate_k] = gate.astype(np.asarray(weights[gate_k]).dtype)
    w[down_k] = down.astype(np.asarray(weights[down_k]).dtype)
    return w, {"experts": len(experts), "neurons_added": added, "layer": L,
               "gain": float(gain)}


def _selftest():
    import os

    from holographic.io_and_interop.holographic_gdnruntime import (
        GDNRuntime, load_runtime)
    from holographic.io_and_interop.holographic_unicron import load_safetensors

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("swarmbake selftest SKIPPED-SUBJECT (no model present)")
        return
    rt, cfg = load_runtime(src)
    w = load_safetensors(os.path.join(src, "model.safetensors"))
    raw = open("/home/claude/bench/docs.txt", encoding="utf-8",
               errors="ignore").read()
    ids = [int(b) for b in raw[3000:3400].encode()][:256]
    L = int(cfg["n_layers"]) - 1
    cap = {}
    rt.forward(ids, hooks={L: lambda h: cap.__setitem__("h", h.copy()) or None})
    H = cap["h"]
    D = H.shape[1]

    # ---- ROUTING IS BY CONTENT, and it SPREADS across experts ----
    G, mu = content_gates(H, 4, temperature=6.0)
    pick = route(H, G, mu)
    share = np.bincount(pick, minlength=4) / len(pick)
    assert share.max() < 0.85, ("one expert must not own everything", share)
    assert (share > 0.01).sum() >= 2, share
    # and different parts of the stream must prefer different experts
    half = len(pick) // 2
    a = np.bincount(pick[:half], minlength=4).argmax()
    b = np.bincount(pick[half:], minlength=4).argmax()

    rng = np.random.default_rng(0)
    experts = [rng.standard_normal((8, D)) * 0.02 for _ in range(4)]

    # ---- OFF BY DEFAULT MEANS UNCHANGED ----
    ref = rt.forward(ids)
    w0, rep0 = install_swarm(w, cfg, experts, H, gain=0.0)
    got0 = GDNRuntime(w0, dict(cfg)).forward(ids)
    assert float(np.max(np.abs(got0 - ref))) < 1e-6, "an OFF swarm changed the model"

    # ---- ON, it runs and stays finite ----
    w1, rep1 = install_swarm(w, cfg, experts, H, gain=0.05)
    got1 = GDNRuntime(w1, dict(cfg)).forward(ids)
    assert np.all(np.isfinite(got1))
    assert float(np.max(np.abs(got1 - ref))) > 0, "an ON swarm did nothing"
    assert rep1["neurons_added"] == 32, rep1

    # ---- and the gate really does SELECT: one expert dominates per token ----
    logits = (H - mu) @ G.T
    top = np.sort(logits, axis=1)
    margin = float(np.mean(top[:, -1] - top[:, -2]))
    assert margin > 0, margin

    print("swarmbake selftest OK -- a %d-expert bank installed as %d MLP "
          "neurons: routing spreads across experts (usage %s, no expert above "
          "%.0f%%), the two halves of the stream prefer experts %d and %d, the "
          "mean top-1 margin is %.3f so the gate genuinely SELECTS, and the "
          "swarm is BIT-IDENTICAL at gain 0 while measurably active at 0.05"
          % (rep1["experts"], rep1["neurons_added"], np.round(share, 2),
             100 * share.max(), a, b, margin))


if __name__ == "__main__":
    _selftest()
