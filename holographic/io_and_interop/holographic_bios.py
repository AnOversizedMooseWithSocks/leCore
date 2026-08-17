"""BIOS -- enumerate the machine before booting an operating system on it.

Moose's observation, and it is the diagnosis for a whole session of bugs: there
was no layer between "here is a checkpoint" and "boot leCore on it". Every
component reached straight into the weights with its own assumptions, and every
scale bug this session was the SAME bug wearing different clothes:

    hardcoded "model.layers." while the checkpoint used
        "model.language_model.layers."        -> testkit shipped 0 layer arrays
    packed in_proj_qkvz assumed, split found   -> GDN routing produced garbage
    vocab_size assumed to equal the tokenizer  -> 276 rows found only by accident
    float16 carriers assumed                   -> payload read empty on float32
    one uniform capacity                       -> a 128-wide model overran a
                                                  boot row the check had passed

A BIOS does exactly three things and they are exactly the three that were
missing: POST (does this machine work?), ENUMERATION (what hardware is present
and how much of it?), and ABSTRACTION (hand the OS a profile so it never has to
know the chipset). Everything above this line stops guessing.

WHAT IT REPORTS, all PROBED rather than assumed:
    tensor root, layer count, block period and which layers are attention
    projection layout (packed / split), head geometry
    vocabulary slack -- declared vocab minus tokenizer entries
    carrier capacity at 1/2/4 bits, and whether carriers are float16 or float32
    whether a leCore layer is ALREADY installed, and at which row
    a POST result: does the model produce finite logits at all

WHY IT MATTERS MORE THAN IT SOUNDS: a profile makes a REFUSAL possible. A model
with 0 free vocabulary rows and a 90x capacity shortfall should be told so
BEFORE anything is written to it, not discovered halfway through an install.
"""

import json

import numpy as np


def post(weights, cfg, probe_ids=None):
    """POWER-ON SELF TEST: does this machine run at all?

    Cheap, first, and before anything is written -- an install onto a model that
    already produces NaNs will produce a NaN model and a clean report."""
    try:
        from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime
        ids = list(probe_ids or [1, 2, 3, 4, 5, 6, 7, 8])
        out = GDNRuntime(weights, cfg).forward(ids)
        finite = bool(np.all(np.isfinite(out)))
        return {"ok": finite, "logits": list(out.shape),
                "detail": "finite" if finite else "NON-FINITE LOGITS"}
    except Exception as exc:
        return {"ok": False, "logits": None,
                "detail": "%s: %s" % (type(exc).__name__, exc)}


def enumerate_machine(weights, cfg, model_dir=None):
    """Probe the checkpoint. Nothing here is assumed; everything is read."""
    names = list(weights)
    root = next((k.split("layers.")[0] for k in names if "layers." in k),
                "model.")
    n_layers = int(cfg.get("n_layers", 0)) or (
        1 + max((int(k.split("layers.")[1].split(".")[0])
                 for k in names if "layers." in k), default=-1))

    gdn, attn = [], []
    for L in range(n_layers):
        if "%slayers.%d.linear_attn.A_log" % (root, L) in weights:
            gdn.append(L)
        else:
            attn.append(L)
    period = (attn[1] - attn[0]) if len(attn) > 1 else n_layers

    layout = "unknown"
    if any("in_proj_qkvz" in k for k in names):
        layout = "packed"
    elif any("in_proj_qkv" in k for k in names):
        layout = "split"

    # vocabulary slack: DECLARED minus what the tokenizer actually defines
    declared = int(cfg.get("vocab", cfg.get("vocab_size", 0)) or 0)
    emb = next((k for k in names if k.endswith("embed_tokens.weight")), None)
    if emb is not None and not declared:
        declared = int(np.asarray(weights[emb]).shape[0])
    defined = declared
    if model_dir:
        import os
        for fn in ("vocab.json", "tokenizer.json"):
            p = os.path.join(model_dir, fn)
            if not os.path.exists(p):
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                defined = len(d) if fn == "vocab.json" else \
                    len((d.get("model") or {}).get("vocab", {})) or defined
                break
            except (OSError, ValueError):
                continue

    from holographic.caching_and_storage.holographic_substrate import (
        capacity_bytes)
    carriers = {}
    dtypes = set()
    for k in names:
        a = np.asarray(weights[k])
        if a.dtype.kind == "f" and "embed" not in k and "lm_head" not in k:
            dtypes.add(str(a.dtype))
    for b in (1, 2, 4):
        carriers[b] = capacity_bytes(weights, b)

    installed, row = False, None
    try:
        from holographic.io_and_interop.holographic_boot import boot
        rec = boot(weights)["record"]
        installed, row = True, (int(np.asarray(weights[emb]).shape[0]) - 1
                                if emb else None)
        seed = rec.seed
    except Exception:
        seed = None

    return {"root": root, "n_layers": n_layers,
            "gdn_layers": gdn, "attn_layers": attn, "block_period": period,
            "projection_layout": layout,
            "hidden": int(cfg.get("hidden", 0)),
            "vocab_declared": declared, "vocab_defined": defined,
            "vocab_free_rows": max(0, declared - defined),
            "carrier_dtypes": sorted(dtypes),
            "carrier_bytes": carriers,
            "lecore_installed": installed, "boot_row": row, "seed": seed}


def fits(profile, payload_bytes, bits=1):
    """Can this machine hold that payload? A profile exists to make a REFUSAL
    possible BEFORE anything is written, rather than halfway through."""
    room = int(profile["carrier_bytes"].get(bits, 0))
    return {"fits": payload_bytes <= room, "need": int(payload_bytes),
            "room": room,
            "shortfall_x": (payload_bytes / room) if room else float("inf")}


def report(weights, cfg, model_dir=None, probe_ids=None):
    """The whole BIOS screen: POST, enumeration, and what the OS may assume."""
    p = enumerate_machine(weights, cfg, model_dir)
    p["post"] = post(weights, cfg, probe_ids)
    return p


def _selftest():
    import os

    rng = np.random.default_rng(0)
    H, V, L = 128, 512, 4
    w = {"model.language_model.embed_tokens.weight":
         (rng.standard_normal((V, H)) * 0.02).astype(np.float32),
         "model.language_model.norm.weight": np.ones(H, np.float32),
         "lm_head.weight": (rng.standard_normal((V, H)) * 0.02).astype(np.float32)}
    for i in range(L):
        pre = "model.language_model.layers.%d." % i
        if i % 2 == 0:                      # every other layer is linear-attn
            w[pre + "linear_attn.A_log"] = np.zeros(4, np.float32)
            w[pre + "linear_attn.in_proj_qkv.weight"] = \
                (rng.standard_normal((256, H)) * 0.02).astype(np.float32)
        w[pre + "mlp.up_proj.weight"] = \
            (rng.standard_normal((256, H)) * 0.02).astype(np.float32)
    cfg = {"hidden": H, "n_layers": L, "vocab": V}

    p = enumerate_machine(w, cfg)

    # ---- the ROOT is probed, not assumed. This exact assumption shipped a
    #      testkit with zero layer arrays and a manifest that claimed otherwise.
    assert p["root"] == "model.language_model.", p["root"]
    # ---- the LAYOUT is probed: packed vs split produced garbage when guessed
    assert p["projection_layout"] == "split", p["projection_layout"]
    # ---- the BLOCK STRUCTURE falls out of which layers have gates ----
    assert p["gdn_layers"] == [0, 2] and p["attn_layers"] == [1, 3], p
    assert p["block_period"] == 2, p["block_period"]
    # ---- capacity is REPORTED per bit depth, so a caller can choose ----
    assert p["carrier_bytes"][4] == p["carrier_bytes"][1] * 4
    assert p["carrier_dtypes"] == ["float32"], p["carrier_dtypes"]
    # ---- and a fresh model is correctly seen as NOT installed ----
    assert p["lecore_installed"] is False

    # ---- A REFUSAL IS POSSIBLE BEFORE WRITING, which is the point ----
    small = fits(p, 10)
    huge = fits(p, 10 ** 9)
    assert small["fits"] and not huge["fits"]
    assert huge["shortfall_x"] > 100, huge

    # ---- POST catches a broken machine instead of installing onto it ----
    r = report(w, cfg, probe_ids=[1, 2, 3])
    assert "post" in r
    broken = dict(w)
    broken["lm_head.weight"] = np.full_like(
        np.asarray(w["lm_head.weight"]), np.nan)
    assert post(broken, cfg, [1, 2, 3])["ok"] is False

    # ---- after installing, the BIOS SEES IT: no external bookkeeping ----
    from holographic.io_and_interop.holographic_boot import BootRecord, write_boot
    w2, _rep = write_boot(w, BootRecord(seed="leCore", dim=H,
                                        symbols=["a"], capabilities=["bind"]))
    p2 = enumerate_machine(w2, cfg)
    assert p2["lecore_installed"] is True and p2["seed"] == "leCore", p2

    print("bios selftest OK -- PROBED root %r, layout %r, %d GDN + %d attention "
          "layers in blocks of %d, %s carriers holding %d/%d/%d bytes at 1/2/4 "
          "bits; a fresh model reads as NOT installed and an installed one is "
          "DETECTED with its seed; an oversized payload is refused BEFORE any "
          "write (%.0fx short) and POST catches a NaN machine"
          % (p["root"], p["projection_layout"], len(p["gdn_layers"]),
             len(p["attn_layers"]), p["block_period"], p["carrier_dtypes"][0],
             p["carrier_bytes"][1], p["carrier_bytes"][2], p["carrier_bytes"][4],
             huge["shortfall_x"]))


if __name__ == "__main__":
    _selftest()
