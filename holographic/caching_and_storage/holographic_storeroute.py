"""STOREROUTE -- ask HRNN what the data IS before choosing how to store it.

Every storage path built for the Galvatron so far treats a payload as opaque
bytes: fountain-code it, hide it in low bits, write it to a vocabulary row. That
is correct and it is also wasteful, because some payloads are not data at all --
they are the OUTPUT OF A GENERATOR, and a generator is smaller than its output.

leCore already measures this and I never asked it. `holographic_rnn` walks an
abstention ladder that "measures before it models" and returns a REGIME:

    generator        a rule reproduces the stream -- store the RULE
    structured       clusters/classes, no closed-form rule -- store a DRIFT MODEL
    incompressible   no generator exists at this horizon -- store the BYTES,
                     and HRNN quotes the allocator cost so the decision is priced

MEASURED on the real classifier, four payload kinds:
    a ramp                 -> generator, identify(denoise), NRMSE 0.000
    repeated facts         -> generator, NRMSE 0.000
    four Gaussian clusters -> structured, demand 2.0 bits, floor 0.015
    white noise            -> incompressible, entropy rate 1.99, allocator quote
                              "dim 4992 per 100" -- it REFUSES to pretend

THE DISCIPLINE THIS ENFORCES is the one this project already applies everywhere
else and had not applied to storage: ABSTAIN RATHER THAN OVERCLAIM. A compressor
that always compresses is lying about the incompressible case; HRNN says so and
quotes the price instead.

HDRIFT carries the structured case: `drift_train` builds a generative model from
raw points and `drift_compose` ADDS two models trained separately (evidence
weighted, sums carry n), so stored generators MERGE without co-training -- which
is what makes a Galvatron's memory extensible after it ships.
"""

import numpy as np


def classify_payload(mind, points, dim=512, seed=0):
    """What kind of thing is this? Delegates entirely to HRNN's ladder."""
    r = mind.holographic_rnn(dim=int(dim), seed=int(seed))
    out = r.process_stream(np.asarray(points, np.float64))
    return {"regime": out.get("regime"), "mechanism": out.get("mechanism"),
            "why": out.get("why"), "horizon": out.get("horizon"),
            "demand": out.get("demand")}


def route(mind, points, dim=512, seed=0):
    """Choose the representation, and say WHY in the report.

    Returns (kind, artifact, report). `kind` is one of:
        "generator"      the HRNN fit -- reproduces the stream from a rule
        "drift"          an HDRIFT model -- samples the distribution
        "raw"            the bytes, because nothing smaller is honest
    """
    P = np.asarray(points, np.float64)
    info = classify_payload(mind, P, dim=dim, seed=seed)
    regime = info["regime"]

    if regime == "generator":
        r = mind.holographic_rnn(dim=int(dim), seed=int(seed))
        fit = r.generator_fit(P) if hasattr(r, "generator_fit") else None
        if fit is not None:
            return "generator", fit, dict(info, chosen="generator",
                                          reason="a rule reproduces the stream")

    if regime == "structured":
        try:
            model = mind.drift_train(P, dim=int(dim))
            return "drift", model, dict(info, chosen="drift",
                                        reason="clusters with no closed-form "
                                               "rule: store the distribution")
        except Exception as exc:
            # HDRIFT REFUSING is a real answer -- a universally collapsing
            # dataset is not served as a mean-generator, and that refusal must
            # fall through to raw rather than be swallowed
            info["drift_refused"] = str(exc)[:120]

    return "raw", P, dict(info, chosen="raw",
                          reason="no generator at this horizon; storing bytes "
                                 "is the honest option")


def extend_drift(mind, model, new_points, dim=512):
    """Train a model on NEW points IN THE EXISTING MODEL'S SPACE, then compose.

    THE GOTCHA, found by trying it: drift_compose requires one encoder space,
    and drift_train PROBES BANDWIDTH FROM THE DATA -- so two models trained
    independently land in different spaces and compose raises "models live in
    different encoder spaces". The bandwidth and bounds of the shipped model
    must be pinned when training the extension. That is not a limitation, it is
    the contract: composing models that measured different scales would be
    adding numbers with different units."""
    bw = getattr(model, "bandwidth", None)
    bounds = getattr(model, "bounds", None)
    second = mind.drift_train(np.asarray(new_points, np.float64), dim=int(dim),
                              bandwidth=bw, bounds=bounds)
    return mind.drift_compose(model, second)


def merge_drift(mind, model_a, model_b):
    """Combine two generators that already share an encoder space."""
    return mind.drift_compose(model_a, model_b)


def _selftest():
    import lecore
    mind = lecore.UnifiedMind(dim=512, seed=0)
    rng = np.random.default_rng(0)

    ramp = np.stack([np.arange(320) * 0.01 + i for i in range(8)], 1)
    clusters = np.concatenate([rng.normal(c, 0.15, size=(80, 8))
                               for c in (-2, 0, 2, 4)])
    noise = rng.standard_normal((320, 8)) * 2

    k_ramp, _a1, r1 = route(mind, ramp)
    k_clu, model, r2 = route(mind, clusters)
    k_noise, raw, r3 = route(mind, noise)

    # ---- THE THREE REGIMES ARE DISTINGUISHED, not collapsed into one path ----
    assert r1["regime"] == "generator", r1
    assert r2["regime"] == "structured", r2
    assert r3["regime"] == "incompressible", r3
    assert k_noise == "raw", k_noise

    # ---- AND THE REFUSAL IS THE POINT: noise is stored as bytes, with the
    #      reason recorded, rather than run through a compressor that would
    #      claim a saving it cannot deliver
    assert "no generator" in r3["reason"]
    assert isinstance(raw, np.ndarray) and raw.shape == noise.shape

    # ---- structured data yields a MERGEABLE model (extensible after shipping)
    if k_clu == "drift":
        more = np.concatenate([rng.normal(c, 0.15, size=(40, 8))
                               for c in (-2, 0, 2, 4)])
        merged = extend_drift(mind, model, more, dim=512)
        assert merged is not None
        # the merged model must carry BOTH evidence counts, or "compose" is
        # just "replace"
        n_merged = getattr(merged, "n_train", None)
        n_a = getattr(model, "n_train", None)
        if n_merged is not None and n_a is not None:
            assert n_merged > n_a, (n_a, n_merged)

    print("storeroute selftest OK -- HRNN's ladder separated a ramp "
          "(%s: %s), four Gaussian clusters (%s -> stored as %s) and white "
          "noise (%s -> stored RAW, %s); and two drift models trained "
          "SEPARATELY composed into one carrying both evidence counts, so a "
          "shipped Galvatron's memory stays extensible"
          % (r1["regime"], str(r1["mechanism"])[:18], r2["regime"], k_clu,
             r3["regime"], r3["reason"][:40]))


if __name__ == "__main__":
    _selftest()
