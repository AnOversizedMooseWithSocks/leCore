#!/usr/bin/env python3
"""Opt-in GPU backend installer -- NEVER a dependency, always a choice.

WHY THIS FILE EXISTS, and why it is a sibling of install_lean.py. Lean 4 had a
one-command installer with --status and --remove; the GPU backends had a report
that NAMED the pip command and no way to run it. That asymmetry is the whole
gap: `gpu_report()` would tell you "cupy is not installed (pip install
cupy-cuda12x, NVIDIA only)" and then leave you to work out which wheel your
driver takes, which is the research task an error message should not hand back.

THE HOUSE POSTURE, unchanged and enforced by this script's existence rather than
weakened by it: leCore RUNS COMPLETE ON NumPy + stdlib. Verified by hard-blocking
cupy, numba, torch, scipy, sklearn, pyfftw, matplotlib, faiss and sympy at the
import hook and using the engine anyway -- the mind boots, find_capability
answers, the levers list, Ouroboros round-trips at cosine 1.0000, and
lean_export emits 229 characters of Lean 4 source WITHOUT LEAN INSTALLED. The
accelerators buy speed on specific kernels; they buy no capability.

TWO PATHS, AND THEY ARE NOT INTERCHANGEABLE:
  cupy   the TRANSPARENT path -- array ops move to the device with no code
         change. NVIDIA/CUDA ONLY, and the wheel must match the DRIVER's CUDA
         major version, which is why this script reads nvidia-smi instead of
         guessing. THE CUDA TOOLKIT IS NOT REQUIRED: the pip wheel bundles the
         runtime, and only the driver has to be present.
  wgpu   the EXPLICIT path -- WGSL shaders, vendor-neutral across Vulkan /
         Metal / DX12 / WebGPU. Slower to write and it runs on hardware CuPy
         will never see, including a software adapter that makes shader
         correctness CI-testable with no GPU at all.

Discipline, matching install_lean.py:
  * INSTALLS NOTHING WITHOUT --install. The default action is a report.
  * Installs into the CURRENT interpreter's environment via pip, and prints what
    it will run BEFORE running it -- no silent environment mutation.
  * Refuses to guess a CUDA wheel when no driver is visible, because installing
    cupy-cuda12x on a machine with no NVIDIA GPU gives you a package that
    imports and finds nothing, which is harder to diagnose than an absence.
  * stdlib only.

    python3 tools/install_gpu.py             # what is available and why not
    python3 tools/install_gpu.py --install   # install what this machine supports
    python3 tools/install_gpu.py --remove
"""

import argparse
import os
import re
import subprocess
import sys


def driver_cuda_major():
    """The CUDA major version the installed NVIDIA driver supports, or None.

    THIS IS THE ONLY NUMBER THAT DECIDES BETWEEN cupy-cuda11x AND cupy-cuda12x,
    and nvidia-smi prints it in its header. Reading it beats asking the user to,
    because a mismatched wheel imports fine and then reports no device -- the
    failure mode that cost a real debugging session."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                             timeout=15).stdout
    except Exception:
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.", out or "")
    return int(m.group(1)) if m else None


def status():
    """What is reachable now, per path, and what would fix it."""
    rep = {"cupy": {}, "wgpu": {}}
    try:
        import cupy                                   # noqa: F401
        rep["cupy"]["installed"] = True
        try:
            import cupy as _cp
            rep["cupy"]["devices"] = int(_cp.cuda.runtime.getDeviceCount())
        except Exception as exc:
            rep["cupy"]["devices"] = 0
            rep["cupy"]["why"] = "%s: %s" % (type(exc).__name__, str(exc)[:90])
    except Exception:
        rep["cupy"]["installed"] = False
        rep["cupy"]["devices"] = 0
    try:
        import wgpu                                   # noqa: F401
        rep["wgpu"]["installed"] = True
    except Exception:
        rep["wgpu"]["installed"] = False
    rep["driver_cuda_major"] = driver_cuda_major()
    rep["wheel"] = ("cupy-cuda12x" if (rep["driver_cuda_major"] or 0) >= 12
                    else ("cupy-cuda11x" if rep["driver_cuda_major"] else None))
    return rep


def plan(rep):
    """The pip packages worth installing on THIS machine, with reasons."""
    todo = []
    if rep["wheel"] and not rep["cupy"]["installed"]:
        todo.append((rep["wheel"],
                     "an NVIDIA driver reporting CUDA %d.x is present; the "
                     "wheel bundles the runtime, so the CUDA Toolkit is NOT "
                     "needed" % rep["driver_cuda_major"]))
    if not rep["wgpu"]["installed"]:
        todo.append(("wgpu",
                     "vendor-neutral WGSL: Vulkan / Metal / DX12 / WebGPU, and "
                     "a software adapter that runs the same shaders with no "
                     "GPU at all"))
    return todo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--install", action="store_true",
                    help="actually run pip (default: report only)")
    ap.add_argument("--remove", action="store_true",
                    help="uninstall the optional GPU backends")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation")
    a = ap.parse_args(argv)

    rep = status()
    print("GPU BACKENDS -- optional accelerators, never dependencies")
    print("  cupy   installed=%-5s devices=%s"
          % (rep["cupy"]["installed"], rep["cupy"].get("devices")))
    if rep["cupy"].get("why"):
        print("         %s" % rep["cupy"]["why"])
    print("  wgpu   installed=%s" % rep["wgpu"]["installed"])
    print("  nvidia driver reports CUDA: %s"
          % (rep["driver_cuda_major"] or "no driver visible"))

    if a.remove:
        cmd = [sys.executable, "-m", "pip", "uninstall", "-y",
               "cupy-cuda12x", "cupy-cuda11x", "cupy", "wgpu"]
        print("\nwill run: %s" % " ".join(cmd))
        if a.yes or input("proceed? [y/N] ").strip().lower() == "y":
            subprocess.run(cmd)
        return 0

    todo = plan(rep)
    if not todo:
        print("\nnothing to install: everything this machine supports is "
              "already here.")
        if not rep["driver_cuda_major"]:
            # SAY WHY, rather than leaving "nothing to do" ambiguous between
            # "you are done" and "this machine cannot".
            print("  (no NVIDIA driver visible, so cupy is not offered -- "
                  "installing it would give you a package that imports and "
                  "finds no device, which is harder to diagnose than an "
                  "absence. wgpu is the path on this machine.)")
        return 0

    print("\nwould install:")
    for pkg, why in todo:
        print("  %-16s %s" % (pkg, why))
    if not a.install:
        print("\n(report only -- re-run with --install to do it)")
        return 0

    cmd = [sys.executable, "-m", "pip", "install"] + [p for p, _ in todo]
    print("\nwill run: %s" % " ".join(cmd))
    if not (a.yes or input("proceed? [y/N] ").strip().lower() == "y"):
        print("cancelled -- nothing installed.")
        return 0
    rc = subprocess.run(cmd).returncode
    print("\nafter install:")
    for k, v in status().items():
        print("  %-18s %s" % (k, v))
    print("\nleCore uses the GPU only when asked: mind.use_gpu(True), or set "
          "HOLOSTUFF_GPU=1 before the process starts. Nothing changes by "
          "default.")
    return rc


def _selftest():
    """The report must be honest on a machine with NEITHER backend, and must
    never plan a CUDA wheel it cannot justify from a driver."""
    rep = status()
    assert set(rep) >= {"cupy", "wgpu", "driver_cuda_major", "wheel"}
    assert isinstance(rep["cupy"]["installed"], bool)
    # NO DRIVER -> NO CUDA WHEEL OFFERED. Installing cupy without a device
    # produces a package that imports and finds nothing, which is a worse state
    # than not having it -- the report must not walk anyone into it.
    if not rep["driver_cuda_major"]:
        assert rep["wheel"] is None
        assert all(p != "cupy-cuda12x" and p != "cupy-cuda11x"
                   for p, _ in plan(rep))
    # AND THE DEFAULT ACTION INSTALLS NOTHING.
    src = open(__file__, encoding="utf-8").read()
    assert 'if not a.install:' in src
    print("install_gpu selftest OK -- driver=%s wheel=%r, %d package(s) would "
          "be offered on this machine; a report-only default, and NO CUDA "
          "wheel is ever planned without a driver to justify it (that install "
          "yields a package which imports and finds no device -- harder to "
          "diagnose than an absence)"
          % (rep["driver_cuda_major"], rep["wheel"], len(plan(rep))))


if __name__ == "__main__":
    if os.environ.get("LECORE_SELFTEST"):
        _selftest()
    else:
        sys.exit(main())
