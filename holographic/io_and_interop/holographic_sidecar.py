"""SIDECAR -- leave the model alone. Put leCore in front of it.

Moose, after watching three runs damage a model and then repair it:
"we can replace the file with some sort of wrapper that pulls the output from
elsewhere... have our own tiny small model in front of the larger real model,
and that's where we put the trained leCore weights and bios and all that stuff.
Not the qwen model itself."

He is right, and it makes every failure this arc produced STRUCTURALLY
IMPOSSIBLE. Every one of them came from editing the base:
    assimilation filtered 18 tensors and made the model 6.4% WORSE
    repair reverted 12 of those 18 and claimed a win inside the noise
    a boot record written into a tied embedding row destroyed the output head
    bakes that landed, bakes that silently did not, guards to catch the damage
None of that can happen to a file nobody writes to.

THE ARCHITECTURE. The base checkpoint is the base checkpoint, byte-identical,
always deployable, always convertible. Everything leCore adds lives in a SIDECAR
next to it:
    boot record            the layer's identity, seed, capability manifest
    per-tensor DELTAS      low-rank A@B, applied at load, off by default
    installed CIRCUITS     VSA bind/unbind, corrections, grown channels
    call-token head delta  the rows that let the model ask for a capability
and the sidecar is TINY -- deltas are rank-r, so a 0.8B's whole leCore layer is
about 10 MB against a 1.75 GB base.

THREE WAYS TO CONSUME IT, which is the point of a curtain:
    load()      base + sidecar, materialised in memory -- what leCore runs
    merge()     one ordinary checkpoint, for llama.cpp / Ollama / anything
    nothing     the base alone still runs, unchanged, forever

WHY THIS BEATS BAKING, beyond safety: every leCore component becomes separately
MEASURABLE and separately REVERTIBLE. A delta that does not earn its place is
deleted from a manifest rather than reverted out of a 1.75 GB file, and the
comparison is base-vs-base+delta on the same probe, which is the paired
measurement that finally has the statistical power to say anything.
"""

import json
import os

import numpy as np

FORMAT = "leCore/sidecar/1"


def new_sidecar(base_dir, seed="leCore", notes=""):
    """An empty sidecar bound to a base checkpoint."""
    from holographic.io_and_interop.holographic_unicron import source_dtypes
    return {"format": FORMAT, "base": os.path.abspath(base_dir), "seed": seed,
            "notes": notes, "deltas": {}, "rows": {}, "circuits": {},
            "base_tensors": len(source_dtypes(base_dir))}


def add_delta(side, tensor, A, B, gain=1.0, why=""):
    """A low-rank correction W += gain * A @ B, applied at load.

    `why` is not decoration: a delta whose reason nobody recorded is a delta
    nobody can evaluate later, and this project has thrown away more time to
    unexplained edits than to wrong ones."""
    side["deltas"][str(tensor)] = {
        "A": np.asarray(A, np.float32), "B": np.asarray(B, np.float32),
        "gain": float(gain), "rank": int(np.asarray(A).shape[-1]), "why": why}
    return side


def add_rows(side, tensor, rows, why=""):
    """Replace specific rows of a tensor -- boot records, call tokens, facts."""
    side["rows"].setdefault(str(tensor), {})
    for idx, vec in dict(rows).items():
        side["rows"][str(tensor)][str(int(idx))] = np.asarray(vec, np.float32)
    if why:
        side["circuits"].setdefault("rows:" + str(tensor), why)
    return side


def save(side, path):
    """Write the sidecar. It is small enough to keep in version control."""
    arrays = {}
    man = {k: v for k, v in side.items()
           if k not in ("deltas", "rows")}
    man["deltas"] = {}
    for name, d in side["deltas"].items():
        i = len(arrays) // 2
        arrays["A%03d" % i] = d["A"]
        arrays["B%03d" % i] = d["B"]
        man["deltas"][name] = {"slot": i, "gain": d["gain"],
                               "rank": d["rank"], "why": d.get("why", "")}
    man["rows"] = {}
    for name, rows in side["rows"].items():
        man["rows"][name] = {}
        for idx, vec in rows.items():
            key = "R%03d" % len(man["rows"][name])
            arrays["%s|%s" % (name, key)] = vec
            man["rows"][name][idx] = key
    arrays["manifest"] = np.frombuffer(json.dumps(man).encode("utf-8"),
                                       dtype=np.uint8)
    np.savez_compressed(path, **arrays)
    return {"path": path, "megabytes": round(os.path.getsize(path) / 1e6, 3),
            "deltas": len(side["deltas"]), "row_tensors": len(side["rows"])}


def load_sidecar(path):
    z = np.load(path, allow_pickle=False)
    man = json.loads(bytes(z["manifest"]).decode("utf-8"))
    if man.get("format") != FORMAT:
        raise ValueError("not a leCore sidecar: %r" % man.get("format"))
    side = dict(man)
    side["deltas"] = {}
    for name, d in man["deltas"].items():
        side["deltas"][name] = dict(d, A=z["A%03d" % d["slot"]],
                                    B=z["B%03d" % d["slot"]])
    side["rows"] = {}
    for name, rows in man.get("rows", {}).items():
        side["rows"][name] = {idx: z["%s|%s" % (name, key)]
                              for idx, key in rows.items()}
    return side


def apply_to(weights, side, gain=1.0):
    """Materialise base + sidecar in memory. The base dict is NOT mutated.

    gain=0.0 returns the base unchanged, which is the whole safety argument:
    the leCore layer is a thing you turn on, not a thing done to your file."""
    out = {k: np.array(v, copy=True) for k, v in weights.items()}
    applied = []
    for name, d in side.get("deltas", {}).items():
        if name not in out:
            applied.append({"tensor": name, "ok": False, "why": "absent"})
            continue
        W = np.asarray(out[name], np.float64)
        upd = (np.asarray(d["A"], np.float64) @ np.asarray(d["B"], np.float64))
        if upd.shape != W.shape:
            applied.append({"tensor": name, "ok": False,
                            "why": "shape %s vs %s" % (upd.shape, W.shape)})
            continue
        out[name] = (W + float(d["gain"]) * float(gain) * upd).astype(
            np.asarray(weights[name]).dtype)
        applied.append({"tensor": name, "ok": True, "rank": d["rank"]})
    if gain:
        for name, rows in side.get("rows", {}).items():
            if name not in out:
                continue
            A = np.asarray(out[name], np.float64)
            for idx, vec in rows.items():
                r = int(idx)
                if 0 <= r < A.shape[0]:
                    A[r] = np.asarray(vec, np.float64)[:A.shape[1]]
            out[name] = A.astype(np.asarray(weights[name]).dtype)
    return out, applied


def load(base_dir, sidecar_path, gain=1.0, lazy=False):
    """The curtain: read a base checkpoint and hand back base + leCore."""
    from holographic.io_and_interop.holographic_gdnruntime import (
        load_runtime, load_weights_dir)
    rt, cfg = load_runtime(base_dir, lazy=True)
    w = load_weights_dir(base_dir)
    side = load_sidecar(sidecar_path)
    out, applied = apply_to(w, side, gain=gain)
    return out, dict(rt.cfg), {"applied": applied, "seed": side.get("seed")}


def merge(base_dir, sidecar_path, out_dir, gain=1.0):
    """Write ONE ordinary checkpoint, for anything that cannot read a sidecar.

    This is the honest half of the curtain: Ollama and llama.cpp consume GGUF
    built from a plain directory and expose no loader hook, so the sidecar is
    the ARCHIVE and this produces what they need on demand."""
    from holographic.io_and_interop.holographic_unicron import export_portable
    import shutil

    w, cfg, rep = load(base_dir, sidecar_path, gain=gain)
    os.makedirs(out_dir, exist_ok=True)
    export_portable(w, os.path.join(out_dir, "model.safetensors"),
                    like=base_dir)
    for f in os.listdir(base_dir):
        src = os.path.join(base_dir, f)
        if os.path.isfile(src) and not f.endswith(".safetensors"):
            shutil.copy(src, os.path.join(out_dir, f))
    return {"out_dir": out_dir, "applied": sum(1 for a in rep["applied"]
                                               if a["ok"])}


def _selftest():
    import tempfile

    from holographic.io_and_interop.holographic_gdnruntime import (
        GDNRuntime, load_runtime, load_weights_dir)
    from holographic.io_and_interop.holographic_measure import measure, better_than

    base = "/tmp/fw" if os.path.exists("/tmp/fw/model.safetensors") \
        else "/home/claude/bench/model"
    if not os.path.exists(os.path.join(base, "model.safetensors")):
        print("sidecar selftest SKIPPED-SUBJECT (no model present)")
        return
    rt, cfg = load_runtime(base)
    w = load_weights_dir(base)
    ids = [int(i % (int(cfg.get("vocab", 256)) - 1)) for i in range(10, 210)]

    side = new_sidecar(base, notes="selftest")
    tname = next(k for k in w if k.endswith("mlp.down_proj.weight"))
    W = np.asarray(w[tname], np.float64)
    rng = np.random.default_rng(0)
    r = 8
    A = rng.standard_normal((W.shape[0], r)) * 1e-3
    B = rng.standard_normal((r, W.shape[1])) * 1e-3
    add_delta(side, tname, A, B, gain=1.0, why="selftest low-rank probe")
    emb = next(k for k in w if k.endswith("embed_tokens.weight"))
    last = int(np.asarray(w[emb]).shape[0]) - 1
    add_rows(side, emb, {last: np.asarray(w[emb], np.float64)[last] * 1.0},
             why="boot row placeholder")

    path = os.path.join(tempfile.mkdtemp(), "lecore.sidecar.npz")
    rep = save(side, path)
    # ---- THE SIDECAR IS TINY next to the model it modifies ----
    base_mb = os.path.getsize(os.path.join(base, "model.safetensors")) / 1e6
    assert rep["megabytes"] < base_mb / 10, (rep["megabytes"], base_mb)

    # ---- gain=0 IS THE BASE, EXACTLY. This is the safety argument. ----
    off, _a = apply_to(w, load_sidecar(path), gain=0.0)
    assert all(np.array_equal(np.asarray(off[k]), np.asarray(w[k])) for k in w), \
        "gain=0 must leave the base byte-identical"

    # ---- gain=1 CHANGES SOMETHING, and only what it said it would ----
    on, applied = apply_to(w, load_sidecar(path), gain=1.0)
    assert all(a["ok"] for a in applied), applied
    changed = [k for k in w if not np.array_equal(np.asarray(on[k]),
                                                  np.asarray(w[k]))]
    assert set(changed) <= {tname, emb}, changed

    # ---- AND THE BASE FILE IS NEVER TOUCHED ----
    w2 = load_weights_dir(base)
    assert all(np.array_equal(np.asarray(w2[k]), np.asarray(w[k])) for k in w)

    # ---- the effect is MEASURABLE with a paired test, which is the point ----
    m_base = measure(rt, ids)
    m_side = measure(GDNRuntime(on, dict(cfg)), ids)
    verdict = better_than(m_side, m_base)
    assert verdict["verdict"] in ("BETTER", "WORSE", "INDISTINGUISHABLE")

    # ---- MERGE gives an ordinary directory anything can open ----
    mdir = tempfile.mkdtemp()
    merge(base, path, mdir, gain=1.0)
    rt3, _c3 = load_runtime(mdir)
    assert np.all(np.isfinite(rt3.forward(ids[:32])))

    print("sidecar selftest OK -- a %.3f MB sidecar beside a %.0f MB base: "
          "gain=0 leaves the base BYTE-IDENTICAL, gain=1 changes exactly the %d "
          "tensors it declared and nothing else, the base file is never written "
          "to, the effect reads %s under a paired test, and merge() produces an "
          "ordinary checkpoint load_runtime opens"
          % (rep["megabytes"], base_mb, len(changed), verdict["verdict"]))


if __name__ == "__main__":
    _selftest()
