"""Optional ctypes bridge to the C holographic kernel.

The public surface mirrors the small part of ``holographic_ai`` that benefits
most from the C core: bind/unbind and single-trace key-value memory. If the
shared library is not built, or a vector dimension is not a power of two, this
module falls back to the NumPy semantics.
"""

from __future__ import annotations

import ctypes
import atexit
import os
import sys
import threading
from pathlib import Path

import numpy as np


_DOUBLE_P = ctypes.POINTER(ctypes.c_double)


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _fallback_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))


def _fallback_involution(a: np.ndarray) -> np.ndarray:
    return np.concatenate(([a[0]], a[:0:-1]))


def _fallback_unbind(composite: np.ndarray, key: np.ndarray) -> np.ndarray:
    return _fallback_bind(composite, _fallback_involution(key))


def _vector(x) -> np.ndarray:
    arr = np.ascontiguousarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("holographic vectors must be one-dimensional")
    return arr


def _ptr(arr: np.ndarray):
    return arr.ctypes.data_as(_DOUBLE_P)


def _candidate_paths() -> list[Path]:
    root = Path(__file__).resolve().parent
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    paths = []
    explicit = os.environ.get("HOLOSTUFF_C_LIB")
    if explicit:
        paths.append(Path(explicit))
    paths.extend(
        [
            root / "c" / "build" / "accelerate" / f"libholoc{ext}",
            root / "c" / "build" / "scalar" / f"libholoc{ext}",
        ]
    )
    return paths


class _Backend:
    def __init__(self, path: Path):
        self.path = path
        self.lib = ctypes.CDLL(str(path))
        self.lock = threading.RLock()
        self._engines: dict[int, ctypes.c_void_p] = {}
        self._declare()

    def _declare(self) -> None:
        lib = self.lib
        lib.holo_engine_create.argtypes = [ctypes.c_size_t, ctypes.c_uint64]
        lib.holo_engine_create.restype = ctypes.c_void_p
        lib.holo_engine_destroy.argtypes = [ctypes.c_void_p]
        lib.holo_engine_destroy.restype = None

        lib.holo_bind.argtypes = [ctypes.c_void_p, _DOUBLE_P, _DOUBLE_P, _DOUBLE_P]
        lib.holo_bind.restype = ctypes.c_int
        lib.holo_unbind.argtypes = [ctypes.c_void_p, _DOUBLE_P, _DOUBLE_P, _DOUBLE_P]
        lib.holo_unbind.restype = ctypes.c_int

        lib.holo_trace_create.argtypes = [ctypes.c_void_p]
        lib.holo_trace_create.restype = ctypes.c_void_p
        lib.holo_trace_destroy.argtypes = [ctypes.c_void_p]
        lib.holo_trace_destroy.restype = None
        lib.holo_trace_clear.argtypes = [ctypes.c_void_p]
        lib.holo_trace_clear.restype = ctypes.c_int
        lib.holo_trace_set.argtypes = [
            ctypes.c_void_p,
            _DOUBLE_P,
            ctypes.c_uint64,
            ctypes.c_double,
        ]
        lib.holo_trace_set.restype = ctypes.c_int
        lib.holo_trace_copy.argtypes = [ctypes.c_void_p, _DOUBLE_P]
        lib.holo_trace_copy.restype = ctypes.c_int
        lib.holo_trace_store.argtypes = [
            ctypes.c_void_p,
            _DOUBLE_P,
            _DOUBLE_P,
            ctypes.c_double,
        ]
        lib.holo_trace_store.restype = ctypes.c_int
        lib.holo_trace_recall.argtypes = [ctypes.c_void_p, _DOUBLE_P, _DOUBLE_P]
        lib.holo_trace_recall.restype = ctypes.c_int

    def engine(self, dim: int) -> ctypes.c_void_p | None:
        if not _is_power_of_two(dim):
            return None
        with self.lock:
            engine = self._engines.get(dim)
            if engine:
                return engine
            engine = self.lib.holo_engine_create(dim, 0)
            if not engine:
                return None
            self._engines[dim] = engine
            return engine

    def check(self, rc: int) -> None:
        if rc != 0:
            raise RuntimeError(f"C holographic kernel returned {rc}")

    def close(self) -> None:
        with self.lock:
            for engine in self._engines.values():
                self.lib.holo_engine_destroy(engine)
            self._engines.clear()


def _load_backend() -> _Backend | None:
    for path in _candidate_paths():
        if path.exists():
            try:
                return _Backend(path)
            except (AttributeError, OSError):
                continue
    return None


_BACKEND = _load_backend()
if _BACKEND:
    atexit.register(_BACKEND.close)


def available() -> bool:
    return _BACKEND is not None


def backend_path() -> str | None:
    return str(_BACKEND.path) if _BACKEND else None


def install(target_globals: dict | None = None, *, strict: bool = False) -> bool:
    """Install C-backed symbols into ``holographic_ai`` or a supplied globals dict."""
    if not available():
        if strict:
            raise RuntimeError("C holographic shared library is not built")
        return False
    if target_globals is None:
        import holographic_ai

        target_globals = holographic_ai.__dict__
    target_globals["bind"] = bind
    target_globals["unbind"] = unbind
    target_globals["HolographicMemory"] = HolographicMemory
    return True


def bind(a, b) -> np.ndarray:
    a_arr = _vector(a)
    b_arr = _vector(b)
    if a_arr.shape != b_arr.shape:
        raise ValueError("bind operands must have the same shape")
    dim = int(a_arr.size)
    engine = _BACKEND.engine(dim) if _BACKEND else None
    if not engine:
        return _fallback_bind(a_arr, b_arr)
    out = np.empty(dim, dtype=np.float64)
    with _BACKEND.lock:
        _BACKEND.check(_BACKEND.lib.holo_bind(engine, _ptr(a_arr), _ptr(b_arr), _ptr(out)))
    return out


def unbind(composite, key) -> np.ndarray:
    comp = _vector(composite)
    key_arr = _vector(key)
    if comp.shape != key_arr.shape:
        raise ValueError("unbind operands must have the same shape")
    dim = int(comp.size)
    engine = _BACKEND.engine(dim) if _BACKEND else None
    if not engine:
        return _fallback_unbind(comp, key_arr)
    out = np.empty(dim, dtype=np.float64)
    with _BACKEND.lock:
        _BACKEND.check(_BACKEND.lib.holo_unbind(engine, _ptr(comp), _ptr(key_arr), _ptr(out)))
    return out


class HolographicMemory:
    """C-backed replacement for ``holographic_ai.HolographicMemory``."""

    def __init__(self, dim: int):
        self.dim = int(dim)
        self._trace = np.zeros(self.dim, dtype=np.float64)
        self._trace_dirty = False
        self._closed = False
        self._backend = _BACKEND if _BACKEND and _is_power_of_two(self.dim) else None
        self._engine = None
        self._c_trace = None
        if self._backend:
            self._engine = self._backend.lib.holo_engine_create(self.dim, 0)
            if self._engine:
                self._c_trace = self._backend.lib.holo_trace_create(self._engine)

    def close(self) -> None:
        if self._closed:
            return
        if self._backend and self._c_trace:
            self._backend.lib.holo_trace_destroy(self._c_trace)
        if self._backend and self._engine:
            self._backend.lib.holo_engine_destroy(self._engine)
        self._c_trace = None
        self._engine = None
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def trace(self) -> np.ndarray:
        if self._backend and self._c_trace and self._trace_dirty:
            with self._backend.lock:
                self._backend.check(
                    self._backend.lib.holo_trace_copy(self._c_trace, _ptr(self._trace))
                )
            self._trace_dirty = False
        return self._trace

    @trace.setter
    def trace(self, value) -> None:
        arr = _vector(value)
        if arr.size != self.dim:
            raise ValueError("trace assignment has the wrong dimension")
        self._trace = arr.copy()
        self._trace_dirty = False
        if self._backend and self._c_trace:
            stored_count = 0 if not np.any(self._trace) else 1
            with self._backend.lock:
                self._backend.check(
                    self._backend.lib.holo_trace_set(
                        self._c_trace,
                        _ptr(self._trace),
                        stored_count,
                        float(stored_count),
                    )
                )

    def learn(self, key, value):
        key_arr = _vector(key)
        value_arr = _vector(value)
        if key_arr.size != self.dim or value_arr.size != self.dim:
            raise ValueError("memory key/value dimension mismatch")
        if self._backend and self._c_trace:
            with self._backend.lock:
                self._backend.check(
                    self._backend.lib.holo_trace_store(
                        self._c_trace, _ptr(key_arr), _ptr(value_arr), 1.0
                    )
                )
            self._trace_dirty = True
        else:
            self._trace = self._trace + bind(key_arr, value_arr)

    def recall(self, key) -> np.ndarray:
        key_arr = _vector(key)
        if key_arr.size != self.dim:
            raise ValueError("memory key dimension mismatch")
        if self._backend and self._c_trace:
            out = np.empty(self.dim, dtype=np.float64)
            with self._backend.lock:
                self._backend.check(
                    self._backend.lib.holo_trace_recall(self._c_trace, _ptr(key_arr), _ptr(out))
                )
            return out
        return unbind(self._trace, key_arr)
