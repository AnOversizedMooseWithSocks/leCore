"""Optional GPU backend -- run the heavy, array-parallel kernels on CuPy (a near-drop-in NumPy replacement that
executes on a CUDA GPU) when one is available, and fall back to NumPy everywhere else, transparently.

THE DESIGN (and why it is this and not 'just import cupy as np' everywhere):

  * GPU IS NOT A FREE WIN. CuPy matches the NumPy API, but every host<->device transfer costs real time over PCIe.
    A big FFT-or-matmul kernel (the fluid solver's pressure projection, a wide bind_batch, the path tracer) wins
    big because the transfer is amortised over a lot of compute. A tiny per-call op on a single (D,) vector LOSES
    -- the transfer dwarfs the work. So this backend is SELECTIVE: heavy kernels opt in; the rest stays on NumPy.
    That is exactly the 'spin the GPU-friendly parts up on CuPy, leave the rest on NumPy' split.

  * FOLLOW-THE-DATA. get_array_module(x) returns cupy if x already lives on the GPU, else numpy -- so a kernel
    written against `xp = get_array_module(self.state)` automatically runs wherever its data is, with no per-line
    device flags. Inputs are moved to the device once (to_device) and results brought back at the boundary
    (asnumpy), so callers keep passing and receiving ordinary NumPy arrays.

HONEST, kept loud:
  * DETERMINISM. The engine's bit-exact guarantees are a CPU/NumPy property. GPU FFTs, reductions, and atomics are
    only equal to NumPy to a TOLERANCE, and can vary run-to-run. So GPU mode is opt-in and is for throughput, not
    for the deterministic, tie-sensitive paths (the maze-rescue / knife-edge tie-breaks stay on CPU, as bind_batch
    already documents).
  * UNMEASURED HERE. This sandbox has no GPU and no cupy, so the GPU path is wired and code-reviewed but the
    speedup is NOT measured here -- it is measured in a CUDA environment by flipping HOLOSTUFF_GPU=1. Everything
    below is verified on the NumPy fallback, which is what runs when no device is present.
"""

import os
import numpy as np

_cupy = None
_probed = False
# request GPU via env (HOLOSTUFF_GPU=1) or programmatically via enable_gpu(True); default OFF
_requested = os.environ.get("HOLOSTUFF_GPU", "0").lower() not in ("0", "", "false", "no", "off")


def _probe():
    """Try to import cupy and confirm a usable CUDA device. Cached; never raises -- a missing GPU just means
    NumPy. (Imported lazily so cupy is NOT a hard dependency: the engine runs with NumPy alone.)"""
    global _cupy, _probed
    if _probed:
        return _cupy
    _probed = True
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()                  # raises if no CUDA device is actually present
        _cupy = cp
    except Exception:
        _cupy = None                                      # no cupy / no GPU -> stay on NumPy, silently
    return _cupy


def enable_gpu(flag=True):
    """Turn the GPU backend on/off programmatically (the user-facing setting). Returns whether GPU is now active
    (i.e. requested AND actually available)."""
    global _requested
    _requested = bool(flag)
    return gpu_enabled()


def gpu_available():
    """True iff cupy imports and a CUDA device is present in THIS environment."""
    return _probe() is not None


def gpu_enabled():
    """True iff GPU was requested (env or enable_gpu) AND is available. The single switch kernels check."""
    return _requested and gpu_available()


def array_module(device=None):
    """The array module to allocate/compute with. device=None follows the global setting (gpu_enabled);
    device='cpu' forces NumPy; device='gpu' forces CuPy if available else NumPy. A kernel does
    `xp = array_module(self.device)` once, then writes plain `xp.zeros`, `xp.fft.fftn`, ... ."""
    if device == "cpu":
        return np
    if device == "gpu":
        return _probe() or np
    return _probe() if gpu_enabled() else np


def get_array_module(*arrays):
    """Follow-the-data: cupy if any argument is a cupy array, else numpy. Lets a kernel run wherever its data
    already lives without threading a device flag through every call (mirrors cupy.get_array_module)."""
    cp = _probe()
    if cp is not None:
        for a in arrays:
            if isinstance(a, cp.ndarray):
                return cp
    return np


def to_device(a, device=None):
    """Move a (NumPy) array onto the compute device once, up front (host->device), so a kernel's many ops don't
    each pay a transfer. No-op when staying on CPU."""
    xp = array_module(device)
    return xp.asarray(a)


def asnumpy(a):
    """Bring an array back to host NumPy at the API boundary (device->host). No-op if it is already NumPy."""
    cp = _probe()
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def on_gpu_island(fn):
    """Decorator turning fn into a GPU ISLAND: its array arguments are moved to the device on entry and its array
    results moved back to NumPy on exit, so the caller passes/receives ordinary NumPy while fn's heavy compute runs
    on the GPU (when enabled). Transfer is paid once at the boundary, amortised over the whole kernel -- the right
    wrapper for a big self-contained computation, the wrong one for a tiny op called in a tight loop."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not gpu_enabled():
            return fn(*args, **kwargs)                     # pure NumPy path, no conversions
        args = tuple(to_device(a) if isinstance(a, np.ndarray) else a for a in args)
        kwargs = {k: (to_device(v) if isinstance(v, np.ndarray) else v) for k, v in kwargs.items()}
        out = fn(*args, **kwargs)
        if isinstance(out, tuple):
            return tuple(asnumpy(o) for o in out)
        return asnumpy(out)
    return wrapper


def device_report():
    """A human-readable line about the current backend state -- for a settings/status readout."""
    if gpu_available():
        cp = _probe()
        try:
            name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        except Exception:
            name = "CUDA device"
        return f"GPU backend {'ENABLED' if gpu_enabled() else 'available (off)'}: {name}"
    return "GPU backend unavailable (no cupy / no CUDA device) -- running on NumPy"


def _selftest():
    # in a CPU-only environment everything must resolve to NumPy and round-trip cleanly
    assert array_module("cpu") is np
    assert get_array_module(np.zeros(3)) is np
    x = np.arange(6.0).reshape(2, 3)
    assert np.array_equal(asnumpy(to_device(x)), x)        # host->device->host is identity on NumPy
    enabled_before = gpu_enabled()

    @on_gpu_island
    def kernel(a, b):
        xp = get_array_module(a, b)
        return xp.fft.irfft(xp.fft.rfft(a, axis=-1) * xp.fft.rfft(b, axis=-1), n=a.shape[-1], axis=-1)

    a = np.random.default_rng(0).standard_normal((4, 32)); b = np.random.default_rng(1).standard_normal((4, 32))
    out = kernel(a, b)
    assert isinstance(out, np.ndarray) and out.shape == (4, 32)   # island returns NumPy
    assert gpu_enabled() == enabled_before                       # no state leaked
    print(f"backend selftest ok: {device_report()}; island kernel returns NumPy, host<->device round-trips")


if __name__ == "__main__":
    _selftest()
