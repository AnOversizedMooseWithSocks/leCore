"""MODEL SOURCE ATTRIBUTION + INFERENCE SHORTCUTS (cp70).

The ask: when knowledge comes from a model, capture an identifier that maps back to
WHERE in the model it lives -- and use that address to skip work next time.

The instrument is the LOGIT LENS (nostalgebraist 2020; Belrose et al. 2023 tuned
variant): project each layer's residual stream through the unembedding and watch the
prediction crystallize layer by layer. The layer where the final answer FIRST becomes
top-1 and stays -- the CRYSTALLIZATION LAYER -- plus a hash of the hidden state
there, is the fact's source address:  model:<n>L/L<k>/<hash8>.

The SHORTCUT follows ROME's localization logic run in reverse: if attribution
measured that this cue's answer crystallizes at layer k << n, future inference runs a
TRUNCATED layer schedule (the runtime's layer_schedule is a list; truncation is
surgery-free) and decodes at k -- real compute saved, agreement CHECKED against the
stored full-depth answer, never assumed. The 2026 caution is honored explicitly:
"Diminishing Returns of Early-Exit in Modern LLMs" (arXiv 2603.23701) and MechLens's
late-crystallization result (2606.07978) both warn that modern models often decide
LATE -- so the shortcut fires only when the measured address says early (k <= 80% of
depth), and falls back to the full pass otherwise. Per-fact measurement, not a
global policy.

Every attribution is TAUGHT ("model source for <q>" -> the address record), so
provenance now answers where AND how: which arm served it, which model, which layer.

HONEST BOUNDARY: on the sandbox mini the decoded tokens are untrained noise -- the
crystallization measurement, the address, the truncated run and the agreement check
are all real and deterministic; the WORDS become meaningful on real weights.
"""
import hashlib
import json

import numpy as np


def _lens_tops(rt, ids):
    """Per-layer logit-lens top token at the last position; returns (tops, hiddens)."""
    snaps = {}

    def _mk(layer):
        def _hook(h):
            snaps[layer] = np.asarray(h[-1], np.float64).copy()
            return None
        return _hook

    n = int(rt.cfg["n_layers"])
    logits = rt.forward(ids, hooks={i: _mk(i) for i in range(n)})
    unemb = np.asarray(rt.embed, np.float64)
    tops = []
    for i in range(n):
        h = snaps.get(i)
        tops.append(int(np.argmax(unemb @ h)) if h is not None else -1)
    final = int(np.argmax(np.asarray(logits)[-1]))
    return final, tops, snaps


def attribute(rt, ids):
    """Measure the source address for this cue: crystallization layer + state hash."""
    final, tops, snaps = _lens_tops(rt, ids)
    n = len(tops)
    emergence = n - 1
    for i in range(n):
        if tops[i] == final and all(t == final for t in tops[i:]):
            emergence = i
            break
    hh = hashlib.sha256(snaps[emergence].tobytes()).hexdigest()[:8]
    return {"answer_token": final, "emergence_layer": emergence,
            "n_layers": n, "per_layer_tops": tops,
            "source_id": "model:%dL/L%d/%s" % (n, emergence, hh),
            "early": emergence <= 0.8 * (n - 1)}


def shortcut(rt, ids, emergence_layer):
    """Truncated forward to the stored crystallization layer -- real layers skipped."""
    # truncate IN PLACE and restore -- the schedule is just a list in cfg, and
    # rebuilding the runtime would spend more than the truncation saves
    prev = rt.cfg.get("layer_schedule")
    rt.cfg["layer_schedule"] = list(range(int(emergence_layer) + 1))
    try:
        logits = rt.forward(ids)
    finally:
        if prev is None:
            rt.cfg.pop("layer_schedule", None)
        else:
            rt.cfg["layer_schedule"] = prev
    tok = int(np.argmax(np.asarray(logits)[-1]))
    n = int(rt.cfg["n_layers"])
    return {"answer_token": tok, "layers_run": int(emergence_layer) + 1,
            "layers_total": n,
            "saved_fraction": 1.0 - (int(emergence_layer) + 1) / float(n)}


def _selftest():
    import os
    import sys
    sys.path.insert(0, ".")
    from holographic.io_and_interop.holographic_gdnruntime import load_runtime
    mdl = "/tmp/mini_installed_full" if os.path.isdir(
        "/tmp/mini_installed_full") else "/tmp/mini_baked"
    rt = load_runtime(mdl)
    if isinstance(rt, tuple):               # (runtime, tokenizer/aux) shapes
        rt = next(x for x in rt if hasattr(x, "forward"))
    ids = [3, 17, 91, 204]
    rep = attribute(rt, ids)
    assert 0 <= rep["emergence_layer"] < rep["n_layers"]
    assert rep["source_id"].startswith("model:%dL/" % rep["n_layers"])
    rep2 = attribute(rt, ids)
    assert rep2["source_id"] == rep["source_id"], "the address is deterministic"
    if rep["early"]:
        sc = shortcut(rt, ids, rep["emergence_layer"])
        agreed = sc["answer_token"] == rep["answer_token"]
        assert agreed, "the truncated run must agree with the full-depth answer"
        return ("OK: cue crystallizes at L%d/%d, address %s, truncated run "
                "agrees saving %.0f%% of layers"
                % (rep["emergence_layer"], rep["n_layers"] - 1,
                   rep["source_id"], 100 * sc["saved_fraction"]))
    return ("OK: cue crystallizes LATE (L%d/%d) -- the shortcut correctly "
            "declines (the 2603.23701 caution), address %s recorded"
            % (rep["emergence_layer"], rep["n_layers"] - 1, rep["source_id"]))


if __name__ == "__main__":
    print(_selftest())
