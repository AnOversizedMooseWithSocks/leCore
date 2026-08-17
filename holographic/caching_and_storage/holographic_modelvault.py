"""MODELVAULT -- a trained model goes in, a RUNNABLE model comes back.

Moose asked that trained models be storable in leCore's holographic storage like
anything else, and recalled and run on demand. The audit found the pieces
already built and never joined: holographic_container is a typed-section format
that stores arrays with arbitrary JSON metadata verbatim, and every leCore
"trained" object -- an HDRIFT drift model, an HRNN channel, a codebook, a
register reservation -- is a small set of arrays plus the numbers needed to
rebuild its encoder.

THE POINT, and it is the demoscene one: WHAT REGENERATES IS NOT STORED. An
HDRIFT model trained on 400 points is (mu, nu) -- 6,144 learned values -- plus
an encoder that regenerates EXACTLY from four numbers (dim, bounds, bandwidth,
seed). The vault holds the moments and the four numbers, not the encoder's
2,048-dimensional basis. Measured: a stored-then-recalled drift model produces
a drift field identical to the original at max |diff| = 0.0, from a 48 KB file.

WHAT THIS IS NOT: a checkpoint format for foreign models. Those go through
unicron_model_store, which hands out an ordinary safetensors directory. This is
for leCORE'S OWN trained objects, which are hypervectors and therefore already
in the format the container was built for.
"""

import io
import json

import numpy as np

VAULT_FORMAT = "leCore/vault/1"


def store(objects, meta=None):
    """Pack named leCore objects into one container. Returns bytes.

    `objects` is {name: {"kind": str, "meta": jsonable, "arrays": {k: ndarray}}}.
    The `meta` of each object must carry EVERYTHING needed to rebuild whatever
    is not stored -- an encoder's seed and bounds, a reservation's dim and seed.
    A vault entry that cannot be rebuilt from its own metadata is a file that
    will not open on another machine, which is the failure this format exists to
    prevent."""
    from holographic.io_and_interop.holographic_container import save_container

    sections = []
    for name, o in dict(objects).items():
        arrays = {k: np.asarray(v) for k, v in dict(o.get("arrays") or {}).items()}
        sections.append({"kind": str(o.get("kind", "object")), "id": str(name),
                         "meta": dict(o.get("meta") or {}), "arrays": arrays})
    return save_container(sections,
                          meta=dict(meta or {}, format=VAULT_FORMAT))


def recall(data):
    """Unpack a vault -> {name: {"kind","meta","arrays"}}."""
    from holographic.io_and_interop.holographic_container import load_container

    got = load_container(data)
    sections = got["sections"] if isinstance(got, dict) and "sections" in got \
        else got
    out = {}
    for s in sections:
        out[str(s.get("id", ""))] = {"kind": s.get("kind"),
                                     "meta": s.get("meta") or {},
                                     "arrays": s.get("arrays") or {}}
    return out


def store_drift(model_name, mu, nu, dim, bounds, bandwidth, seed, n_train=0,
                labels=None):
    """An HDRIFT generative model as a vault object.

    Stores ONLY the learned moments. The VectorFunctionEncoder regenerates from
    (dim, bounds, bandwidth, seed) -- four numbers against a 2,048-dimensional
    basis, which is the whole argument for keeping seeds instead of tables."""
    return {model_name: {
        "kind": "hdrift",
        "meta": {"dim": int(dim), "bounds": [list(map(float, b)) for b in bounds],
                 "bandwidth": float(bandwidth), "seed": int(seed),
                 "n_train": int(n_train), "n_dims": len(bounds),
                 "labels": list(labels) if labels is not None else None},
        "arrays": {"mu": np.asarray(mu), "nu": np.asarray(nu)}}}


def rebuild_drift(entry):
    """Recall an HDRIFT model into something you can immediately call.

    Returns (encoder, mu, nu) -- the encoder REGENERATED from metadata rather
    than unpacked, so the file never carried it."""
    from holographic.sampling_and_signal.holographic_hdrift import (
        VectorFunctionEncoder)

    m = entry["meta"]
    enc = VectorFunctionEncoder(int(m["n_dims"]), dim=int(m["dim"]),
                                bounds=[tuple(b) for b in m["bounds"]],
                                bandwidth=float(m["bandwidth"]),
                                seed=int(m["seed"]))
    return enc, entry["arrays"]["mu"], entry["arrays"]["nu"]


def store_registers(name, dim, n_slots, seed, values=None):
    """A register reservation: the SEED, not the basis.

    reserve() is a QR of a seeded random matrix, so the whole reservation
    regenerates from 64 bits. Storing the basis would be D x N floats for
    nothing."""
    arrays = {}
    if values is not None:
        arrays["values"] = np.asarray(values)
    return {name: {"kind": "registers",
                   "meta": {"dim": int(dim), "n_slots": int(n_slots),
                            "seed": int(seed), "regenerable": True},
                   "arrays": arrays}}


def rebuild_registers(entry):
    from holographic.caching_and_storage.holographic_keyreserve import reserve

    m = entry["meta"]
    R = reserve(int(m["dim"]), int(m["n_slots"]), seed=int(m["seed"]))
    return R, entry["arrays"].get("values")


def _selftest():
    import numpy as np

    from holographic.sampling_and_signal.holographic_hdrift import (
        drift_moments, drift_field, VectorFunctionEncoder)

    rng = np.random.default_rng(0)
    enc = VectorFunctionEncoder(2, dim=2048, bounds=[(0, 1), (0, 1)],
                                bandwidth=6.0, seed=0)
    data = np.clip(rng.normal(0.5, 0.12, (400, 2)), 0, 1)
    mu, nu = drift_moments(data, enc)
    x = np.array([0.5, 0.5])
    before = np.asarray(drift_field(x, mu, nu, enc), np.float64)

    # ---- STORE, RECALL, RUN ----
    blob = store(store_drift("demo", mu, nu, 2048, [(0, 1), (0, 1)], 6.0, 0,
                             n_train=len(data)))
    back = recall(blob)
    enc2, mu2, nu2 = rebuild_drift(back["demo"])
    after = np.asarray(drift_field(x, mu2, nu2, enc2), np.float64)

    # ---- THE RECALLED MODEL MUST BE THE SAME MODEL, not merely similar ----
    assert np.array_equal(before, after), (before, after)

    # ---- AND THE ENCODER MUST NOT BE IN THE FILE. If it were, the container
    #      would be far larger than the moments it holds.
    learned = np.asarray(mu).nbytes + np.asarray(nu).nbytes
    assert len(blob) < learned * 2.0, (len(blob), learned)

    # ---- registers regenerate from a seed, so an empty-array vault still works
    rblob = store(store_registers("regs", 128, 16, 0))
    R, vals = rebuild_registers(recall(rblob)["regs"])
    assert R.shape == (16, 128), R.shape
    assert vals is None
    from holographic.caching_and_storage.holographic_keyreserve import reserve
    assert np.array_equal(R, reserve(128, 16, seed=0))

    print("modelvault selftest OK -- a drift model TRAINED on %d points stores in "
          "%.1f KB against %.1f KB of learned moments, recalls, and produces a "
          "drift field IDENTICAL to the original (max diff 0.0); its encoder is "
          "REGENERATED from four numbers rather than stored; and a 16-slot "
          "register reservation round-trips from a seed alone with no arrays at "
          "all"
          % (len(data), len(blob) / 1e3, learned / 1e3))


if __name__ == "__main__":
    _selftest()
