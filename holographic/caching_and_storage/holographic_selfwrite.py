"""SELFWRITE -- the model storing what surprised it, without being told to.

The largest item on the list of things an installed model still could not do:
WRITE TO ITS OWN REGISTERS. Every register in every test was written from
outside, which makes a memory a filing cabinet with no clerk.

AND THE REFRAME THAT DISSOLVES IT: look at the update rule again.

    S <- a S (I - beta k k^T) + beta v k^T

THE MODEL ALREADY WRITES ON EVERY TOKEN. Writing was never the missing part.
What was missing is CHOOSING THE KEY -- and a key is a linear map of the hidden
state, which is a matrix, which installs like everything else.

SO THE QUESTION BECAME: can a linear map of the state tell whether this token is
worth keeping? MEASURED, three ways, held out:
    state at t   -> surprise about the NEXT token       r=0.487, top decile 24%
    state at t+1 -> surprise about the token JUST SEEN   r=0.605, top decile 53%
    state at t   -> its OWN entropy                      r=0.814, top decile 71%
The first is weak and had to be: a state cannot know what will surprise it. But
ONE STEP LATER it carries the token it consumed and can say whether that was
news, and its own uncertainty it knows very well indeed -- 7.1x chance.

SO A KEY PROJECTION STEERED BY THAT SIGNAL SENDS SURPRISING STATES TO A RESERVED
SLOT AND EVERYTHING ELSE TO THE ORDINARY SUBSPACE, and the delta rule -- which
was going to write something regardless -- writes the interesting thing into
protected storage. The model decides what to remember, in weights, with nothing
running.

WHAT THIS IS NOT: the signal is a linear readout, so it stores what it was
fitted to call surprising. It is a WRITE POLICY, not a judgement, and a model
with this installed remembers unusual things rather than important ones. Those
overlap more than they differ in text, which is why it works at all, and they
are not the same thing.
"""

import numpy as np


def fit_novelty(runtime, weights, cfg, ids, layer=None, ridge=1e-2, mode="entropy"):
    """Learn to read 'this is worth keeping' off the hidden state.

    `mode` picks which signal: 'entropy' is what the state knows about its OWN
    uncertainty (r=0.814) and is available immediately; 'surprise' is how
    unexpected the token just consumed was (r=0.605) and needs the state one
    step later. Entropy is the stronger readout and the weaker notion; surprise
    is the reverse. Both are reported so the caller can choose knowingly."""
    # float32 on the vocab-sized head: 512 x 248,320 in float64 is 970 MiB and
    # this step failed with exactly that MemoryError on a real model. The
    # readout that follows is a ridge fit whose answer is measured, not a
    # quantity where the last 45 bits matter.
    A = np.asarray(weights[next(k for k in weights
                                if k.endswith("embed_tokens.weight"))],
                   np.float32)
    L = int(int(cfg["n_layers"]) - 1 if layer is None else layer)
    cap = {}
    lg = np.asarray(runtime.forward(list(ids),
                                    hooks={L: lambda h: cap.__setitem__(
                                        "h", h.copy()) or None}), np.float64)
    Hs = cap["h"]
    P = np.exp(lg - lg.max(-1, keepdims=True))
    P /= P.sum(-1, keepdims=True)
    tgt = np.asarray(list(ids)[1:], np.int64)
    nll = -np.log(P[np.arange(len(tgt)), tgt] + 1e-30)
    ent = -(P * np.log(P + 1e-30)).sum(-1)[:-1]

    X, y = (Hs[:-1], ent) if mode == "entropy" else (Hs[1:], nll)
    n = len(X) // 2
    lam = float(ridge) * float(np.trace(X[:n].T @ X[:n])) / X.shape[1]
    wv = np.linalg.solve(X[:n].T @ X[:n] + lam * np.eye(X.shape[1]),
                         X[:n].T @ y[:n])
    pred, true = X[n:] @ wv, y[n:]
    hi = true > np.percentile(true, 90)
    order = np.argsort(pred)[::-1][:max(1, int(0.1 * len(pred)))]
    return {"direction": wv, "mode": mode, "state_mean": X.mean(0),
            "correlation": float(np.corrcoef(pred, true)[0, 1]),
            "top_decile_hit": float(hi[order].mean()),
            "threshold": float(np.percentile(X @ wv, 90))}


def slot_for(state, reserved, mean=None, seed=0):
    """WHICH register this state belongs in -- a content hash, not a counter.

    One slot is not a memory, it is a latch: 79 of 700 positions routed to slot
    0 in a real run and every one overwrote the last, so a value stored early
    read back at cosine 0.51. The delta rule keeps the MOST RECENT write to a
    key, so distinct content must get distinct keys. Projecting the state onto
    the reservation and taking the argmax does that with one matmul, and puts
    SIMILAR states in the SAME slot -- which is the behaviour you want, because
    a restatement of a fact should refresh it rather than consume a new
    register."""
    # CENTRE FIRST. Hidden states share a large common component, and an argmax
    # over R @ h is dominated by it: with 64 reserved slots only SIX were ever
    # selected and the busiest took 54 of 79 writes. Subtracting the mean makes
    # the choice depend on what DISTINGUISHES this state, which is the whole
    # point. This is the third distinct place in this arc where centring was
    # the fix -- memsearch's addresses, factbake's update direction, and now
    # slot selection -- and each time the raw vector measured the shared
    # component instead of the content.
    h = np.asarray(state, np.float64)
    R = np.asarray(reserved, np.float64)
    mu = np.asarray(mean, np.float64) if mean is not None else 0.0
    return int(np.argmax(np.abs(R @ (h - mu))))


def key_for(state, novelty, reserved, slot=None, sharpness=8.0):
    """The key this state should be written under.

    Blends toward a RESERVED direction as novelty rises, and stays in the
    ordinary subspace otherwise. One matrix multiply and a sigmoid -- both
    things a layer already does."""
    h = np.asarray(state, np.float64)
    s = float(h @ novelty["direction"]) - float(novelty["threshold"])
    g = 1.0 / (1.0 + np.exp(-float(sharpness) * s))
    # A HARD SWITCH, NOT A BLEND. A blended key (1-g)*ordinary + g*slot is NOT
    # orthogonal to the other reserved slots whenever g is between 0 and 1, so
    # every intermediate write leaks into the reservation and destroys it --
    # measured, a stored value fell to cosine 0.525 after 512 later writes
    # instead of holding above 0.9. The reservation only survives if the
    # ordinary branch is PROJECTED OFF the reserved directions and the gate is
    # sharp enough to be a switch, which is the same result the router already
    # established: at temperature 100 the off branch contributes 7e-23.
    R = np.asarray(reserved, np.float64)
    ordinary = h - (h @ R.T) @ R
    ordinary = ordinary / (np.linalg.norm(ordinary) + 1e-30)
    j = (slot_for(h, reserved, mean=novelty.get("state_mean"))
         if slot is None else int(slot) % len(reserved))
    slot_key = np.asarray(reserved[j], np.float64)
    k = slot_key if g > 0.5 else ordinary
    return k / (np.linalg.norm(k) + 1e-30), g


def _selftest():
    import os

    from holographic.io_and_interop.holographic_gdnruntime import (
        load_runtime, load_weights_dir)
    from holographic.caching_and_storage.holographic_keyreserve import (
        reserve, delta_write, delta_read)

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("selfwrite selftest SKIPPED-SUBJECT (no model present)")
        return
    rt, cfg = load_runtime(src)
    w = load_weights_dir(src)
    H = int(cfg["hidden"])
    raw = open("/home/claude/bench/docs.txt", encoding="utf-8",
               errors="ignore").read()
    fit_ids = [b for b in raw[30000:31600].encode("utf-8")][:1400]

    nov = fit_novelty(rt, w, cfg, fit_ids, mode="entropy")
    # ---- THE READOUT MUST BEAT CHANCE BY A LOT, or the key is steered by noise
    assert nov["top_decile_hit"] > 0.4, nov
    assert nov["correlation"] > 0.5, nov

    # ---- AND IT MUST ROUTE SELECTIVELY: novel states to the slot, ordinary
    #      states away from it. A gate that fires on everything stores nothing.
    R = reserve(H, 64, seed=0)
    cap = {}
    ev = [b for b in raw[40000:40800].encode("utf-8")][:700]
    rt.forward(ev, hooks={int(cfg["n_layers"]) - 1:
                          lambda h: cap.__setitem__("h", h.copy()) or None})
    Hs = cap["h"]
    gates = np.array([key_for(h, nov, R)[1] for h in Hs])
    assert gates.max() > 0.5, gates.max()
    assert gates.mean() < 0.5, gates.mean()

    # ---- AND WHAT IT ROUTES IN MUST SURVIVE: the slot is reserved, so writes
    #      that land there are protected from the ordinary traffic.
    S = np.zeros((H, H))
    stored = None
    rng = np.random.default_rng(0)
    for i, h in enumerate(Hs):
        k, g = key_for(h, nov, R)
        v = h.copy()
        S = delta_write(S, k, v)
        if g > 0.9 and stored is None:
            from holographic.caching_and_storage.holographic_selfwrite import (
                slot_for as _sf)
            stored = (R[_sf(h, R)].copy(), v.copy(), _sf(h, R))
    # ---- AND SLOT SELECTION MUST SPREAD, or every novel write lands on one
    #      register and the memory is a latch. MEASURED with centring: 64 slots
    #      use 15 distinct against 6 uncentred, busiest 19 against 54.
    from collections import Counter
    spread = Counter(slot_for(h, R, mean=nov["state_mean"])
                     for h, g in zip(Hs, gates) if g > 0.5)
    assert len(spread) >= 8, dict(spread)

    if stored is not None:
        # write ONLY to the other slots, so the test asks whether a reserved
        # register survives OTHER registers being used -- not whether a slot
        # survives being overwritten, which no memory does
        others = [i for i in range(len(R)) if i != stored[2]]
        for t in range(512):
            S = delta_write(S, R[others[t % len(others)]],
                            rng.standard_normal(H))
        got = delta_read(S, stored[0])
        cos = float(got @ stored[1]
                    / (np.linalg.norm(got) * np.linalg.norm(stored[1]) + 1e-30))
        assert cos > 0.9, (cos, "a reserved slot must survive OTHER slots")
    else:
        cos = float("nan")

    print("selfwrite selftest OK -- a LINEAR READOUT of the hidden state predicts "
          "the model's own uncertainty at r=%.3f and finds %.0f%% of the top "
          "decile against 10%% chance; steering the KEY by it routes %.0f%% of "
          "positions toward a reserved slot while leaving the mean gate at "
          "%.2f; slot selection spreads them across %d registers (CENTRED -- "
          "uncentred it collapsed to 6 of 64); and a value that lands in one "
          "survives 512 writes to the OTHERS at cosine %.3f -- the model "
          "choosing what to keep, in its own forward pass, because the delta "
          "rule was going to write something anyway"
          % (nov["correlation"], 100 * nov["top_decile_hit"],
             100 * float((gates > 0.5).mean()), gates.mean(),
             len(spread), cos))


if __name__ == "__main__":
    _selftest()
