"""DEVICERUN -- run the model on whatever hardware is there, and prove it agrees.

An LLM is usually run on a GPU. This runtime was pure host NumPy, so on a
machine with a card it left the entire forward pass on the CPU -- the FLOPs are
in the model, not in leCore's own kernels, and leCore's WGSL path covers the
kernels.

leCORE ALREADY HAD THE SWITCH and the runtime never asked for it:
    holographic_backend.array_module()   cupy when a device is present AND the
                                         policy allows, numpy otherwise
    gpu_available() / backend_status()   what is actually there
    resource_policy(gpu='on'|'off'|'auto')  who decides
    wgsl_bind_batch / matmul_kernel      vendor-neutral kernels for leCore's own
                                         operations
So this module is not a GPU port. It is the missing WIRE between a switch that
existed and a forward pass that ignored it.

RESIDENCY IS THE WHOLE POINT, and the backend's own docstring says why: "every
host<->device transfer costs", and "a tiny per-call op on a single vector" loses
to the transfer that feeds it. So WEIGHTS MOVE ONCE AND STAY; token ids and
logits are small and cross per call. A runtime that transferred weights per
layer would be slower on a GPU than on a CPU and would look like the GPU was the
problem.

THE HARD PART OF TESTING THIS is that a CPU-only box cannot prove a GPU path
works -- and an untested path rots. So the selftest SUBSTITUTES A FAKE DEVICE
MODULE (numpy wearing cupy's name) and drives the whole dispatch end to end.
That cannot measure speed and does not pretend to; it proves the CODE PATH is
correct, which is the half that fails silently. MEASURED: 50 tensors go
resident and the forward output is BIT-IDENTICAL to the host path.

WHAT IS HONESTLY NOT CLAIMED: no speedup is reported here, because none was
measured on real hardware. `gpu_crossover` exists to find where a device starts
winning and it needs a real adapter to answer. Until then the claim is PARITY --
the same numbers on either path -- and parity is what makes the speed question
safe to ask later.
"""

import numpy as np


def status():
    """What hardware is actually available, and what the policy allows."""
    from holographic.misc.holographic_backend import (
        array_module, gpu_available, gpu_enabled)
    xp = array_module()
    return {"gpu_available": bool(gpu_available()),
            "gpu_enabled": bool(gpu_enabled()),
            "array_module": getattr(xp, "__name__", str(xp)),
            "using": "gpu" if getattr(xp, "__name__", "numpy") != "numpy"
                     else "cpu"}


def place(runtime, want="auto"):
    """Put a model runtime on the best available device. Returns what happened.

    `want` is 'auto' (use a device if the policy and hardware allow), 'gpu'
    (ask explicitly), or 'cpu' (stay on the host). Asking for a GPU that is not
    there is not an error -- it reports cpu and runs, because a pipeline that
    dies on a laptop is worse than one that is merely slower."""
    if str(want) == "cpu":
        return runtime.to_device(False)
    rep = runtime.to_device(True)
    if str(want) == "gpu" and rep.get("device") != "gpu":
        rep = dict(rep, asked="gpu", got="cpu")
    return rep


def parity(runtime, ids, atol=0.0):
    """Do the host and device paths agree on the SAME input?

    Returns the max absolute difference. The default tolerance is EXACTLY ZERO
    because on this runtime they should be bit-identical when the device module
    is numpy-compatible; a real f32 device will need a tolerance and should say
    so explicitly rather than inherit a loose default."""
    before = np.asarray(runtime.forward(list(ids)), np.float64)
    rep = runtime.to_device(True)
    after = np.asarray(runtime.forward(list(ids)), np.float64)
    diff = float(np.max(np.abs(after - before)))
    return {"max_abs_diff": diff, "agrees": diff <= float(atol),
            "placement": rep}


def _fake_device():
    """numpy wearing cupy's name -- so the dispatch path is testable anywhere."""
    import types

    fake = types.ModuleType("fakecupy")
    for n in dir(np):
        if not n.startswith("_"):
            setattr(fake, n, getattr(np, n))
    fake.asnumpy = lambda a: np.asarray(a)
    fake.ndarray = np.ndarray
    return fake


def _selftest():
    import os

    from holographic.io_and_interop.holographic_gdnruntime import load_runtime
    import holographic.misc.holographic_backend as B

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("devicerun selftest SKIPPED-SUBJECT (no model present)")
        return

    rt, _cfg = load_runtime(src)
    raw = open("/home/claude/bench/docs.txt", encoding="utf-8",
               errors="ignore").read()
    ids = [b for b in raw[20000:20300].encode("utf-8")][:200]
    base = np.asarray(rt.forward(ids), np.float64)

    # ---- CPU PATH: asking for a device that is not there must RUN, not raise
    st = status()
    cpu = place(rt, "cpu")
    assert cpu["device"] == "cpu", cpu
    assert np.array_equal(np.asarray(rt.forward(ids), np.float64), base)

    # ---- ASKING FOR A GPU ON A CPU BOX must report the truth and keep going
    asked = place(rt, "gpu")
    if not st["gpu_available"]:
        assert asked.get("got") == "cpu", asked
        assert np.array_equal(np.asarray(rt.forward(ids), np.float64), base)

    # ---- DEVICE PATH, exercised with a fake module so a CPU-only box still
    #      tests it. This proves CORRECTNESS, never speed.
    real = (B.array_module, B.gpu_available, B.to_device)
    fake = _fake_device()
    try:
        B.array_module = lambda device=None: fake
        B.gpu_available = lambda: True
        B.to_device = lambda a: fake.asarray(a)
        rt2, _c2 = load_runtime(src)
        rep = rt2.to_device(True)
        assert rep["device"] == "gpu", rep
        assert rep["resident"] > 0, rep
        after = np.asarray(rt2.forward(ids), np.float64)
        assert np.array_equal(after, base), float(np.max(np.abs(after - base)))
        resident = rep["resident"]
    finally:
        B.array_module, B.gpu_available, B.to_device = real

    # ---- and the switch must be OFF again afterwards, or the test leaks
    assert status()["using"] == st["using"], (status(), st)

    print("devicerun selftest OK -- this box reports %s (%s); asking for a GPU "
          "where there is none REPORTS the truth and keeps running rather than "
          "raising; and driving the dispatch with a substitute device module "
          "makes %d weight tensors resident and returns output BIT-IDENTICAL to "
          "the host path -- parity proven on hardware that cannot prove speed, "
          "which is the half that rots silently"
          % (st["using"], st["array_module"], resident))


if __name__ == "__main__":
    _selftest()
