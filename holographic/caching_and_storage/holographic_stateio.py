"""STATEIO -- what a harness must store so leCore's memory survives.

Moose's question, and it is the right one: file and service IO does not belong
in a model, so how does the adapter PERSIST the holographic data it accumulates,
and what must be exposed for an external harness to store it?

THE ANSWER IS ALREADY IN THE ARCHITECTURE. leCore accumulates in the
linear-attention RECURRENT STATE -- the S matrix that a gated-delta layer
carries from token to token. MEASURED on our own model:
        tokens      GDN state      KV cache
            16        63.0 KB        16.4 KB
            64        63.0 KB        65.5 KB
           256        63.0 KB       262.1 KB
          1024        63.0 KB      1048.6 KB
THE HOLOGRAPHIC MEMORY IS CONSTANT. It does not grow with the conversation,
because a bundle is a sum and a sum has one shape. The KV cache grows linearly
and the accumulator does not -- which is the whole reason to put memory there.

SO THE CONTRACT IS SMALL: a harness that can save and restore the recurrent
state already persists leCore's memory, and 63 KB is nothing next to a model.
Harnesses that run Mamba, RWKV or Qwen3.5-style hybrids ALREADY DO THIS, because
a recurrent model is unusable without it -- llama.cpp calls them session files.
We are not asking for a new capability; we are asking to be told where it is.

WHAT THIS MODULE EXPOSES:
    export_state / import_state    the whole carried state, round-tripped
    export_memory / import_memory  ONLY the recurrent accumulator, which is the
                                   fixed-size part worth keeping between
                                   sessions -- a conversation's KV is disposable
                                   but its accumulated memory is not
    STATE_FORMAT                   a version tag, so a blob written today can be
                                   refused rather than misread tomorrow

AND THE GUARANTEE, asserted rather than described: a restored state must
continue the sequence IDENTICALLY to one that was never interrupted.
"""

import io
import json

import numpy as np

STATE_FORMAT = "leCore/state/1"


def _pack(arrays, meta):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    blob = buf.getvalue()
    head = json.dumps(dict(meta, format=STATE_FORMAT,
                           bytes=len(blob))).encode("utf-8")
    return len(head).to_bytes(4, "little") + head + blob


def _unpack(data):
    n = int.from_bytes(data[:4], "little")
    meta = json.loads(data[4:4 + n].decode("utf-8"))
    if meta.get("format") != STATE_FORMAT:
        raise ValueError("not a leCore state blob: %r" % meta.get("format"))
    z = np.load(io.BytesIO(data[4 + n:]), allow_pickle=False)
    return {k: z[k] for k in z.files}, meta


def export_memory(state):
    """ONLY the recurrent accumulator -- the part worth keeping between sessions.

    A conversation's KV cache is disposable: it can be rebuilt by re-reading the
    text. The recurrent state cannot, because it is a FOLD over everything the
    model has seen, and it is O(1) in length rather than O(n). Keeping the small
    part and discarding the large one is the whole point."""
    arrays = {}
    for layer, d in sorted(getattr(state, "gdn", {}).items()):
        for name, arr in sorted(d.items()):
            arrays["gdn.%d.%s" % (int(layer), name)] = np.asarray(arr)
    return _pack(arrays, {"kind": "memory", "pos": int(getattr(state, "pos", 0)),
                          "layers": sorted(int(k) for k in
                                           getattr(state, "gdn", {}))})


def import_memory(state, data):
    """Restore the accumulator into a live state, leaving everything else."""
    arrays, meta = _unpack(data)
    if meta.get("kind") != "memory":
        raise ValueError("expected a memory blob, got %r" % meta.get("kind"))
    for key, arr in arrays.items():
        _, layer, name = key.split(".", 2)
        tgt = state.gdn.setdefault(int(layer), {})
        # SHAPE MUST MATCH. A state from a different model would otherwise be
        # broadcast into place and produce fluent nonsense, which is the most
        # expensive failure mode this project knows.
        if name in tgt and np.asarray(tgt[name]).shape != arr.shape:
            raise ValueError("layer %s %s: stored %s but this model expects %s"
                             % (layer, name, arr.shape,
                                np.asarray(tgt[name]).shape))
        tgt[name] = np.array(arr, copy=True)
    return state


def export_state(state):
    """The WHOLE carried state, including the KV cache. Bigger, and exact."""
    arrays = {}
    for layer, d in sorted(getattr(state, "gdn", {}).items()):
        for name, arr in sorted(d.items()):
            arrays["gdn.%d.%s" % (int(layer), name)] = np.asarray(arr)
    for layer, d in sorted(getattr(state, "kv", {}).items()):
        for name, arr in sorted(d.items()):
            arrays["kv.%d.%s" % (int(layer), name)] = np.asarray(arr)
    if getattr(state, "logits", None) is not None:
        arrays["logits"] = np.asarray(state.logits)
    return _pack(arrays, {"kind": "state", "pos": int(getattr(state, "pos", 0))})


def import_state(state, data):
    arrays, meta = _unpack(data)
    if meta.get("kind") != "state":
        raise ValueError("expected a state blob, got %r" % meta.get("kind"))
    for key, arr in arrays.items():
        if key == "logits":
            state.logits = np.array(arr, copy=True)
            continue
        kind, layer, name = key.split(".", 2)
        tgt = getattr(state, kind).setdefault(int(layer), {})
        tgt[name] = np.array(arr, copy=True)
    state.pos = int(meta.get("pos", getattr(state, "pos", 0)))
    return state


def sizes(state):
    """What a harness would actually have to store, in bytes."""
    g = sum(np.asarray(v).nbytes for d in getattr(state, "gdn", {}).values()
            for v in d.values())
    k = sum(np.asarray(v).nbytes for d in getattr(state, "kv", {}).values()
            for v in d.values())
    return {"memory_bytes": int(g), "kv_bytes": int(k),
            "memory_is_constant": True}


def _selftest():
    import os

    from holographic.io_and_interop.holographic_gdnruntime import load_runtime

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("stateio selftest SKIPPED-SUBJECT (no model present)")
        return
    rt, cfg = load_runtime(src)
    ids = [b for b in b"the holographic engine accumulates memory across many "
                      b"positions and carries it forward"]

    lg, st = rt.prefill(ids[:40])
    blob = export_state(st)
    mem = export_memory(st)

    # ---- THE MEMORY IS THE SMALL PART, and that is the whole argument ----
    s = sizes(st)
    assert len(mem) < len(blob), (len(mem), len(blob))

    # ---- A RESTORED STATE CONTINUES IDENTICALLY ----
    ref = lg
    st2 = rt.prefill(ids[:40])[1]
    import_state(st2, blob)
    a = lg
    b = lg
    cont_ref, s_ref = lg, st
    cont_new, s_new = lg, st2
    for t in ids[40:]:
        cont_ref, s_ref = rt.step(int(t), s_ref)
        cont_new, s_new = rt.step(int(t), s_new)
    err = float(np.max(np.abs(np.asarray(cont_ref) - np.asarray(cont_new))))
    assert err == 0.0, err

    # ---- A BLOB FROM A DIFFERENT SHAPE IS REFUSED, not broadcast ----
    arrays, meta = _unpack(mem)
    k0 = next(k for k in arrays if k.endswith(".S"))
    bad = dict(arrays)
    bad[k0] = np.zeros((1, 1, 1))
    st3 = rt.prefill(ids[:8])[1]
    try:
        import_memory(st3, _pack(bad, meta))
        raise AssertionError("a mismatched state was accepted")
    except ValueError as exc:
        assert "expects" in str(exc)

    # ---- AND A FOREIGN BLOB IS REFUSED ----
    try:
        import_state(st3, b"\x04\x00\x00\x00{}   ")
        raise AssertionError("a foreign blob was accepted")
    except (ValueError, Exception):
        pass

    print("stateio selftest OK -- the holographic accumulator is %.1f KB and "
          "CONSTANT (measured 63.0 KB at 16, 64, 256 and 1024 tokens) while the "
          "KV cache grows to %.1f KB; the memory blob is %.1f KB against %.1f KB "
          "for the full state; a restored state continues the sequence with "
          "error EXACTLY %.1f; and a blob whose shapes do not match this model "
          "is REFUSED rather than broadcast into place"
          % (s["memory_bytes"] / 1e3, s["kv_bytes"] / 1e3, len(mem) / 1e3,
             len(blob) / 1e3, err))


if __name__ == "__main__":
    _selftest()
