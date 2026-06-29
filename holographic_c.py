"""Optional ctypes bridge to the C holographic kernel.

The public surface mirrors the small part of ``holographic_ai`` that benefits
most from the C core: bind/unbind, weighted accumulation, fixed-vector batch
binding, and single-trace key-value memory. If the shared library is not built,
or a vector dimension is not a power of two, this module falls back to the NumPy
semantics.
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
_SIZE_T_P = ctypes.POINTER(ctypes.c_size_t)
_INT_P = ctypes.POINTER(ctypes.c_int)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


_BIND_FIXED_MAX_C_ROWS = max(0, _env_int("HOLOSTUFF_C_BIND_FIXED_MAX_ROWS", 8))


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _fallback_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))


def _fallback_bind_fixed(role: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.fft.irfft(
        np.fft.rfft(role)[None, :] * np.fft.rfft(rows, axis=1),
        n=rows.shape[1],
        axis=1,
    )


def _fallback_weighted_sum(rows: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if rows.shape[0] == 0:
        return np.zeros(rows.shape[1], dtype=np.float64)
    if weights is None:
        return np.sum(rows, axis=0)
    return np.sum(rows * weights[:, None], axis=0)


def _fallback_involution(a: np.ndarray) -> np.ndarray:
    return np.concatenate(([a[0]], a[:0:-1]))


def _fallback_unbind(composite: np.ndarray, key: np.ndarray) -> np.ndarray:
    return _fallback_bind(composite, _fallback_involution(key))


def _vector(x) -> np.ndarray:
    arr = np.ascontiguousarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("holographic vectors must be one-dimensional")
    return arr


def _matrix(x) -> np.ndarray:
    arr = np.ascontiguousarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("holographic row stacks must be two-dimensional")
    return arr


def _weights(x, count: int) -> np.ndarray | None:
    if x is None:
        return None
    arr = np.ascontiguousarray(x, dtype=np.float64).ravel()
    if arr.size != count:
        raise ValueError("weights must match the number of vectors")
    return arr


def _ptr(arr: np.ndarray):
    return arr.ctypes.data_as(_DOUBLE_P)


def _size_ptr(arr: np.ndarray):
    return arr.ctypes.data_as(_SIZE_T_P)


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
        self.holo_weighted_sum = getattr(lib, "holo_weighted_sum", None)
        if self.holo_weighted_sum:
            self.holo_weighted_sum.argtypes = [
                ctypes.c_size_t,
                _DOUBLE_P,
                _DOUBLE_P,
                ctypes.c_size_t,
                _DOUBLE_P,
            ]
            self.holo_weighted_sum.restype = ctypes.c_int
        self.holo_bind_fixed_many = getattr(lib, "holo_bind_fixed_many", None)
        if self.holo_bind_fixed_many:
            self.holo_bind_fixed_many.argtypes = [
                ctypes.c_void_p,
                _DOUBLE_P,
                _DOUBLE_P,
                ctypes.c_size_t,
                _DOUBLE_P,
            ]
            self.holo_bind_fixed_many.restype = ctypes.c_int
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
        self.holo_program_run_basic = getattr(lib, "holo_program_run_basic", None)
        if self.holo_program_run_basic:
            self.holo_program_run_basic.argtypes = [
                ctypes.c_void_p,
                _DOUBLE_P,
                _DOUBLE_P,
                ctypes.c_size_t,
                _DOUBLE_P,
                _DOUBLE_P,
                _DOUBLE_P,
                _DOUBLE_P,
                ctypes.c_size_t,
                _DOUBLE_P,
                _DOUBLE_P,
                ctypes.c_size_t,
                _DOUBLE_P,
                ctypes.c_int,
                ctypes.c_size_t,
                ctypes.c_double,
                _DOUBLE_P,
                _INT_P,
                _SIZE_T_P,
                _SIZE_T_P,
                ctypes.c_size_t,
                _SIZE_T_P,
            ]
            self.holo_program_run_basic.restype = ctypes.c_int

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
    target_globals["bind_fixed"] = bind_fixed
    target_globals["weighted_sum"] = weighted_sum
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


def weighted_sum(vectors, weights=None) -> np.ndarray:
    arr = np.ascontiguousarray(vectors, dtype=np.float64)
    if arr.ndim == 1 and arr.size == 0:
        weight_arr = _weights(weights, 0)
        return np.asarray(0.0) if weight_arr is None else np.zeros(0, dtype=np.float64)
    rows = _matrix(arr)
    weight_arr = _weights(weights, rows.shape[0])
    fn = _BACKEND.holo_weighted_sum if _BACKEND else None
    if not fn:
        return _fallback_weighted_sum(rows, weight_arr)
    out = np.empty(rows.shape[1], dtype=np.float64)
    weight_ptr = _ptr(weight_arr) if weight_arr is not None else None
    with _BACKEND.lock:
        _BACKEND.check(fn(rows.shape[1], _ptr(rows), weight_ptr, rows.shape[0], _ptr(out)))
    return out


def bind_fixed(role, B) -> np.ndarray:
    role_arr = _vector(role)
    rows = _matrix(B)
    if rows.shape[1] != role_arr.size:
        raise ValueError("bind_fixed role and rows must have the same vector dimension")
    if rows.shape[0] == 0:
        return np.empty_like(rows)
    dim = int(role_arr.size)
    engine = _BACKEND.engine(dim) if _BACKEND else None
    fn = _BACKEND.holo_bind_fixed_many if _BACKEND else None
    if not engine or not fn or rows.shape[0] > _BIND_FIXED_MAX_C_ROWS:
        return _fallback_bind_fixed(role_arr, rows)
    out = np.empty(rows.shape, dtype=np.float64)
    with _BACKEND.lock:
        _BACKEND.check(fn(engine, _ptr(role_arr), _ptr(rows), rows.shape[0], _ptr(out)))
    return out


def program_run_basic(
    program,
    positions,
    op_role,
    arg_role,
    op_vectors,
    data_vectors,
    *,
    op_norms=None,
    data_norms=None,
    init_acc=None,
    max_steps: int | None = None,
    branch_tol: float = 0.5,
) -> tuple[np.ndarray | None, list[tuple[int, int]]]:
    """Run the core HoloMachine instruction subset in the C program VM.

    This covers LOAD/BIND/BUNDLE/PERMUTE/IFMATCH/HALT. Rich host-bound
    operations such as CALL/APPLY/registers/stacks stay on the Python VM.
    """
    program_arr = _vector(program)
    positions_arr = _matrix(positions)
    op_role_arr = _vector(op_role)
    arg_role_arr = _vector(arg_role)
    op_arr = _matrix(op_vectors)
    data_arr = _matrix(data_vectors)
    dim = int(program_arr.size)
    if positions_arr.shape[1] != dim or op_role_arr.size != dim or arg_role_arr.size != dim:
        raise ValueError("program, positions, and roles must share a dimension")
    if op_arr.shape[1] != dim or data_arr.shape[1] != dim:
        raise ValueError("opcode/data matrices must share the program dimension")
    if max_steps is None:
        max_steps = int(positions_arr.shape[0])
    max_steps = max(0, int(max_steps))
    if max_steps == 0:
        return None, []

    op_norm_arr = (
        np.ascontiguousarray(op_norms, dtype=np.float64).ravel()
        if op_norms is not None
        else np.ascontiguousarray(np.linalg.norm(op_arr, axis=1), dtype=np.float64)
    )
    data_norm_arr = (
        np.ascontiguousarray(data_norms, dtype=np.float64).ravel()
        if data_norms is not None
        else np.ascontiguousarray(np.linalg.norm(data_arr, axis=1), dtype=np.float64)
    )
    if op_norm_arr.size != op_arr.shape[0] or data_norm_arr.size != data_arr.shape[0]:
        raise ValueError("norm tables must match opcode/data rows")

    init_arr = None if init_acc is None else _vector(init_acc)
    if init_arr is not None and init_arr.size != dim:
        raise ValueError("initial accumulator has the wrong dimension")

    engine = _BACKEND.engine(dim) if _BACKEND else None
    fn = _BACKEND.holo_program_run_basic if _BACKEND else None
    if not engine or not fn:
        raise RuntimeError("C holographic program runner is not available")

    out = np.empty(dim, dtype=np.float64)
    out_has = ctypes.c_int(0)
    trace_capacity = min(max_steps, int(positions_arr.shape[0]))
    op_indices = np.empty(trace_capacity, dtype=np.uintp)
    arg_indices = np.empty(trace_capacity, dtype=np.uintp)
    trace_count = np.empty(1, dtype=np.uintp)
    init_ptr = _ptr(init_arr) if init_arr is not None else None
    with _BACKEND.lock:
        _BACKEND.check(
            fn(
                engine,
                _ptr(program_arr),
                _ptr(positions_arr),
                positions_arr.shape[0],
                _ptr(op_role_arr),
                _ptr(arg_role_arr),
                _ptr(op_arr),
                _ptr(op_norm_arr),
                op_arr.shape[0],
                _ptr(data_arr),
                _ptr(data_norm_arr),
                data_arr.shape[0],
                init_ptr,
                1 if init_arr is not None else 0,
                max_steps,
                float(branch_tol),
                _ptr(out),
                ctypes.byref(out_has),
                _size_ptr(op_indices),
                _size_ptr(arg_indices),
                trace_capacity,
                _size_ptr(trace_count),
            )
        )
    n = int(trace_count[0])
    trace = [(int(op_indices[i]), int(arg_indices[i])) for i in range(n)]
    return (out if out_has.value else None), trace


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
