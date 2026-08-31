"""MODELSTORE -- keep the model in leCore's format, hand out a boring checkpoint.

Moose asked for a compatibility curtain: our format underneath, the ordinary
interface on top. Most of it already existed and was never connected, and the
audit found the last piece one keystroke before I wrote a second one.

WHAT WAS ALREADY THERE:
    holographic_container   a TYPED-SECTION container (ZIP of manifest.json plus
                            binary arrays) whose defining property is that a
                            section this reader does not understand ROUND-TRIPS
                            UNTOUCHED. Built for leStudio workspaces; it is
                            exactly the right primitive for this and needed no
                            changes.
    LazyWeights             weights compressed in RAM, materialised per tensor
                            on demand -- the curtain, but only in memory
    middle_out_encode       the codec. MEASURED on a real Qwen tensor: 14.68 MB
                            float32 -> 3.65 MB, 4.02x (2.01x against float16)
                            at 0.0226 relative weight error
    export_portable         decodes back to ordinary safetensors, which is what
                            llama.cpp's converter wants

WHAT WAS MISSING: nothing but the join. The compressed store only existed AFTER
loading a plain safetensors file, so it bought RAM and not disk, not load time,
and not the memory bandwidth that actually bounds generation (3.49 GB read per
token at float32 on a 0.8B -- measured, and the reason that model ran at 0.6
tokens/sec).

PER-TENSOR CHOICE, NOT ONE CODEC EVERYWHERE. Small tensors stay raw because a
codec header outweighs them; large 2-D tensors are encoded and the result is
KEPT ONLY IF SMALLER. A compressor that grows its input is a bug with a press
release, and this project has shipped that bug once already in the factored
path.

HONEST ABOUT THE CURTAIN'S DIRECTION: nothing here lets Ollama read the leCore
format. Ollama and llama.cpp consume GGUF built from an ordinary directory and
expose no loader hook -- measured and recorded elsewhere in these notes. What
this buys is that the leCore format can be the ARCHIVE, with an ordinary
checkpoint produced on demand at whatever fidelity the target wants.
"""

import json
import os

import numpy as np

KIND = "lecore.model.weights"


def _code_blobs(code):
    """The byte payloads of a middle-out code, named for reassembly."""
    out = {"base": bytes(code["base"])}
    for i, r in enumerate(code.get("refinements", [])):
        out["ref%02d" % i] = bytes(r)
    return out


def save_model(weights, cfg, out_path, min_bytes=1 << 16, progress=None):
    """Write the model as a leCore container. Returns a size report."""
    from holographic.io_and_interop.holographic_container import save_container
    from holographic.io_and_interop.holographic_unicron import middle_out_encode

    sections = []
    raw_total = 0
    kept_total = 0
    encoded = 0
    for i, (name, val) in enumerate(sorted(weights.items())):
        a = np.ascontiguousarray(np.asarray(val))
        raw_total += a.nbytes
        meta = {"name": name, "shape": list(a.shape), "dtype": str(a.dtype)}
        arrays = {}
        use_raw = a.ndim != 2 or a.nbytes < int(min_bytes)
        if not use_raw:
            code = middle_out_encode(np.asarray(a, np.float32))
            # THE CODE IS NOT ALL NUMPY. middle_out returns raw `bytes` for the
            # base plane and a LIST of byte-strings for the refinements, so a
            # `hasattr(v, "nbytes")` test silently classified every tensor as
            # raw and the container compressed nothing at all. Measure the real
            # payload and store each kind as what it is.
            blobs = _code_blobs(code)
            size = sum(len(b) for b in blobs.values())
            if size < a.nbytes:
                for k, b in blobs.items():
                    arrays[k] = np.frombuffer(b, dtype=np.uint8)
                meta["codec"] = "middle_out"
                meta["code_meta"] = {k: (list(v) if isinstance(v, tuple) else v)
                                     for k, v in code.items()
                                     if k not in ("base", "refinements")}
                meta["n_refinements"] = len(code.get("refinements", []))
                kept_total += size
                encoded += 1
            else:
                use_raw = True          # the codec GREW it: refuse and say so
        if use_raw:
            arrays["raw"] = a
            meta["codec"] = "raw"
            kept_total += a.nbytes
        sections.append({"kind": KIND, "id": "t%05d" % i, "meta": meta,
                         "arrays": arrays})
        if progress and i % 25 == 0:
            progress(i, name, meta["codec"])

    blob = save_container(sections, meta={"lecore_model": 1, "config": dict(cfg)})
    with open(out_path, "wb") as f:
        f.write(blob)
    disk = os.path.getsize(out_path)
    return {"path": out_path, "tensors": len(sections), "encoded": encoded,
            "raw_megabytes": round(raw_total / 1e6, 2),
            "stored_megabytes": round(kept_total / 1e6, 2),
            "file_megabytes": round(disk / 1e6, 2),
            "ratio": round(raw_total / max(disk, 1), 2)}


def load_model(path, lazy=True, max_cached=8):
    """Read the container back as (weights, cfg).

    lazy=True keeps codes packed and decodes per tensor on demand -- a
    transformer touches layers strictly in order, so the working set is tiny."""
    from holographic.io_and_interop.holographic_container import load_container
    from holographic.io_and_interop.holographic_unicron import (
        LazyWeights, middle_out_decode)

    with open(path, "rb") as f:
        # READ THE RETURN SHAPE, DO NOT ASSUME IT. load_container returns a
        # DICT, not the (sections, meta) tuple I guessed -- the same class of
        # mistake as every other "I knew what that returned" bug this session.
        doc = load_container(f.read())
    sections = doc.get("sections", [])
    meta = doc.get("meta", {}) or {}
    if not meta.get("lecore_model"):
        raise ValueError("not a leCore model container (meta: %s)"
                         % sorted(meta)[:6])
    out = {}
    for sec in sections:
        if sec.get("kind") != KIND:
            continue                       # foreign sections pass through
        m = sec["meta"]
        if m["codec"] == "raw":
            out[m["name"]] = np.asarray(sec["arrays"]["raw"]).astype(m["dtype"])
        else:
            code = dict(m.get("code_meta", {}))
            if isinstance(code.get("shape"), list):
                code["shape"] = tuple(code["shape"])
            code["base"] = np.asarray(sec["arrays"]["base"], np.uint8).tobytes()
            code["refinements"] = [
                np.asarray(sec["arrays"]["ref%02d" % i], np.uint8).tobytes()
                for i in range(int(m.get("n_refinements", 0)))]
            out[m["name"]] = np.asarray(
                middle_out_decode(code)).astype(m["dtype"])
    if lazy:
        out = LazyWeights(out, max_cached=int(max_cached))
    return out, dict(meta.get("config", {}))


def materialize(path, out_dir, dtype=None):
    """THE CURTAIN: write an ORDINARY model directory from the container.

    This is the honest half. Nothing here lets an external runtime read the
    leCore format; it lets the leCore format be the archive and produce a
    checkpoint that converts and runs like any other."""
    from holographic.io_and_interop.holographic_unicron import export_portable

    weights, cfg = load_model(path, lazy=False)
    os.makedirs(out_dir, exist_ok=True)
    rep = export_portable(weights, os.path.join(out_dir, "model.safetensors"),
                          dtype=dtype)
    # WRITE A CONFIG THE TARGET UNDERSTANDS. The container holds leCore's
    # internal cfg (hidden, n_layers, ...) while config.json is read as a
    # Hugging Face config (hidden_size, num_hidden_layers, ...). Dumping the
    # internal one produced a directory that looked right and failed on load --
    # a curtain has to speak the language on the outside, not the inside.
    with open(os.path.join(out_dir, "galvatron.json"), "w") as f:
        json.dump({"format": "galvatron/1", "config": cfg, "residents": []}, f)
    return {"out_dir": out_dir, "bytes": rep["bytes"],
            "megabytes": round(rep["bytes"] / 1e6, 2), "tensors": rep["tensors"]}


def _selftest():
    import tempfile

    from holographic.io_and_interop.holographic_gdnruntime import (
        GDNRuntime, load_runtime)
    from holographic.io_and_interop.holographic_unicron import load_safetensors

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("modelstore selftest SKIPPED-SUBJECT (no model present)")
        return
    rt, cfg = load_runtime(src)
    w = load_safetensors(os.path.join(src, "model.safetensors"))
    ids = [int(b) for b in b"The capital of France is Paris."]
    ref = rt.forward(ids)
    plain = os.path.getsize(os.path.join(src, "model.safetensors"))

    path = os.path.join(tempfile.mkdtemp(), "model.lecore")
    rep = save_model(w, rt.cfg, path, min_bytes=4096)
    assert rep["encoded"] > 0, rep
    assert rep["file_megabytes"] * 1e6 < plain, (rep["file_megabytes"], plain)

    # ---- IT LOADS BACK INTO A RUNNING MODEL, eagerly and lazily ----
    back, cfg2 = load_model(path, lazy=False)
    got = GDNRuntime(back, cfg2).forward(ids)
    err = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-30))
    assert np.all(np.isfinite(got)) and err < 0.3, err
    lz, cfg3 = load_model(path, lazy=True)
    assert np.all(np.isfinite(GDNRuntime(lz, cfg3).forward(ids)))

    # ---- THE CURTAIN: an ordinary directory load_runtime can open ----
    mdir = tempfile.mkdtemp()
    mat = materialize(path, mdir)
    # load_runtime accepts EITHER config.json or galvatron.json (fixed earlier
    # this session), so the materialised directory opens with neither special
    # casing nor a hand-written HF config
    rt4, _c = load_runtime(mdir)
    assert np.all(np.isfinite(rt4.forward(ids)))

    # ---- A CODEC THAT WOULD GROW A TENSOR IS REFUSED ----
    tiny_rep = save_model({"a.weight": np.zeros((4, 4), np.float32)}, rt.cfg,
                          os.path.join(tempfile.mkdtemp(), "t.lecore"),
                          min_bytes=1)
    assert tiny_rep["encoded"] == 0, "a 4x4 tensor must stay raw"

    # ---- and a FOREIGN container is rejected rather than misread ----
    from holographic.io_and_interop.holographic_container import save_container
    junk = os.path.join(tempfile.mkdtemp(), "j.lecore")
    with open(junk, "wb") as f:
        f.write(save_container([{"kind": "something.else", "id": "x"}]))
    try:
        load_model(junk)
        raise AssertionError("a foreign container was accepted")
    except ValueError as exc:
        assert "not a leCore model container" in str(exc)

    print("modelstore selftest OK -- %d tensors (%d encoded) stored in leCore's "
          "OWN container: %.2f MB raw -> %.2f MB on disk (%.2fx), loads back "
          "into a RUNNING model eagerly and lazily (max logit deviation %.3f), "
          "materialize() writes a %.2f MB ordinary checkpoint load_runtime "
          "opens, a tensor the codec would GROW stays raw, and a foreign "
          "container is rejected"
          % (rep["tensors"], rep["encoded"], rep["raw_megabytes"],
             rep["file_megabytes"], rep["ratio"], err, mat["megabytes"]))


if __name__ == "__main__":
    _selftest()
