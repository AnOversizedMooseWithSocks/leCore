"""Strict NumPy/ctypes adapter for the liblecore ABI-0 conformance suite.

This is deliberately not a runtime dispatcher.  A caller must name one shared
library artifact, and every array crossing the ABI must already have the exact
native dtype, shape, and C-contiguous layout required by its semantic profile.
The adapter never changes precision and never falls back to the Python kernel.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import threading
from typing import Final, Optional, Union
import weakref

import numpy as np


LECORE_ABI_VERSION: Final = 0
LECORE_ISA_VERSION: Final = 1

LECORE_OK: Final = 0
LECORE_EINVAL: Final = 1
LECORE_EDIM: Final = 2
LECORE_EPROFILE: Final = 3
LECORE_EBACKEND: Final = 4
LECORE_EOVERFLOW: Final = 5
LECORE_ENOMEM: Final = 6
LECORE_EUNSUPPORTED: Final = 7
LECORE_ENONFINITE: Final = 8
LECORE_EFORMAT: Final = 9
LECORE_ECHECKSUM: Final = 10

LECORE_PROFILE_HRR_F64_V1: Final = 0x00010001
LECORE_PROFILE_HRR_F32_V1: Final = 0x00010002
LECORE_BACKEND_AUTO: Final = 0
LECORE_BACKEND_DIRECT: Final = 1
LECORE_BACKEND_RADIX2: Final = 2
LECORE_VALIDATION_SHAPE: Final = 0
LECORE_VALIDATION_FINITE: Final = 1
LECORE_SCALAR_F64: Final = 1
LECORE_SCALAR_F32: Final = 2

LECORE_CAP_HRR_F64: Final = 1 << 0
LECORE_CAP_HRR_F32: Final = 1 << 1
LECORE_CAP_DIRECT: Final = 1 << 2
LECORE_CAP_RADIX2: Final = 1 << 3
LECORE_CAP_BATCH: Final = 1 << 4
LECORE_CAP_MIXED_F64_F32: Final = 1 << 5
LECORE_CAP_FINITE_VALIDATION: Final = 1 << 6


_PROFILE_BY_NAME = {
    "f64": LECORE_PROFILE_HRR_F64_V1,
    "f32": LECORE_PROFILE_HRR_F32_V1,
}
_DTYPE_BY_PROFILE = {
    LECORE_PROFILE_HRR_F64_V1: np.dtype(np.float64),
    LECORE_PROFILE_HRR_F32_V1: np.dtype(np.float32),
}
_SCALAR_BY_PROFILE = {
    LECORE_PROFILE_HRR_F64_V1: LECORE_SCALAR_F64,
    LECORE_PROFILE_HRR_F32_V1: LECORE_SCALAR_F32,
}
_CAPABILITY_BY_PROFILE = {
    LECORE_PROFILE_HRR_F64_V1: LECORE_CAP_HRR_F64,
    LECORE_PROFILE_HRR_F32_V1: LECORE_CAP_HRR_F32,
}
_BACKEND_BY_NAME = {
    "auto": LECORE_BACKEND_AUTO,
    "direct": LECORE_BACKEND_DIRECT,
    "radix2": LECORE_BACKEND_RADIX2,
}
_CAPABILITY_BY_BACKEND = {
    LECORE_BACKEND_DIRECT: LECORE_CAP_DIRECT,
    LECORE_BACKEND_RADIX2: LECORE_CAP_RADIX2,
}


class _AllocatorV0(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("user", ctypes.c_void_p),
        ("allocate", ctypes.c_void_p),
        ("deallocate", ctypes.c_void_p),
    ]


class _ConfigV0(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("profile", ctypes.c_uint32),
        ("backend", ctypes.c_uint32),
        ("validation", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("dimension", ctypes.c_uint32),
        ("allocator", _AllocatorV0),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class LeCoreError(RuntimeError):
    """Base exception for a nonzero native status."""

    def __init__(self, status: int, operation: str, detail: str):
        self.status = int(status)
        self.operation = operation
        self.detail = detail
        super().__init__(f"{operation}: {detail} (status {status})")


class LeCoreInvalidArgumentError(LeCoreError):
    pass


class LeCoreDimensionError(LeCoreError):
    pass


class LeCoreProfileError(LeCoreError):
    pass


class LeCoreBackendError(LeCoreError):
    pass


class LeCoreOverflowError(LeCoreError):
    pass


class LeCoreMemoryError(LeCoreError):
    pass


class LeCoreUnsupportedError(LeCoreError):
    pass


class LeCoreNonFiniteError(LeCoreError):
    pass


class LeCoreFormatError(LeCoreError):
    pass


class LeCoreChecksumError(LeCoreError):
    pass


class LeCoreCompatibilityError(RuntimeError):
    """The loaded artifact does not provide the ABI/profile requested."""


class LeCoreClosedError(RuntimeError):
    """An operation was attempted after context destruction."""


class LeCoreThreadError(RuntimeError):
    """A context was used from a thread other than its creator."""


_ERROR_BY_STATUS: dict[int, type[LeCoreError]] = {
    LECORE_EINVAL: LeCoreInvalidArgumentError,
    LECORE_EDIM: LeCoreDimensionError,
    LECORE_EPROFILE: LeCoreProfileError,
    LECORE_EBACKEND: LeCoreBackendError,
    LECORE_EOVERFLOW: LeCoreOverflowError,
    LECORE_ENOMEM: LeCoreMemoryError,
    LECORE_EUNSUPPORTED: LeCoreUnsupportedError,
    LECORE_ENONFINITE: LeCoreNonFiniteError,
    LECORE_EFORMAT: LeCoreFormatError,
    LECORE_ECHECKSUM: LeCoreChecksumError,
}


class _NativeContextOwner:
    """Thread-neutral, exactly-once owner used by explicit and GC cleanup."""

    __slots__ = ("lock", "pointer", "destroy")

    def __init__(self, lock: threading.RLock, destroy):
        self.lock = lock
        self.pointer = ctypes.c_void_p()
        self.destroy = destroy

    def close(self) -> None:
        with self.lock:
            pointer = self.pointer
            if not pointer:
                return
            self.pointer = type(pointer)()
            self.destroy(pointer)


def _finalize_native_context(owner: _NativeContextOwner) -> None:
    owner.close()


def _exact_range_alias(a: np.ndarray, b: np.ndarray) -> bool:
    return a.ctypes.data == b.ctypes.data and a.nbytes == b.nbytes


def _reject_partial_overlap(
    output: np.ndarray,
    *inputs: np.ndarray,
    allow_exact: bool,
) -> None:
    for input_array in inputs:
        if not np.shares_memory(output, input_array):
            continue
        if allow_exact and _exact_range_alias(output, input_array):
            continue
        raise ValueError("output partially overlaps an input array")


class Library:
    """One explicitly selected liblecore shared-library artifact."""

    def __init__(
        self,
        path: Union[str, os.PathLike[str]],
        *,
        expected_abi: int = LECORE_ABI_VERSION,
        expected_isa: int = LECORE_ISA_VERSION,
    ):
        if (
            isinstance(expected_abi, bool)
            or not isinstance(expected_abi, int)
            or expected_abi != LECORE_ABI_VERSION
        ):
            raise LeCoreCompatibilityError(
                f"this adapter binds only ABI {LECORE_ABI_VERSION}; "
                f"expected_abi={expected_abi!r} is unsupported"
            )
        if (
            isinstance(expected_isa, bool)
            or not isinstance(expected_isa, int)
            or expected_isa != LECORE_ISA_VERSION
        ):
            raise LeCoreCompatibilityError(
                f"this adapter binds only ISA {LECORE_ISA_VERSION}; "
                f"expected_isa={expected_isa!r} is unsupported"
            )
        if path is None:
            raise TypeError("path is required; implicit library discovery is disabled")
        supplied = Path(os.fspath(path)).expanduser()
        if not supplied.is_file():
            raise FileNotFoundError(f"liblecore artifact does not exist: {supplied}")
        self.path = str(supplied.resolve())
        try:
            self._dll = ctypes.CDLL(self.path)
        except OSError as error:
            raise OSError(f"could not load liblecore artifact {self.path}: {error}") from error
        self._declare_identity()
        if self.abi_version != LECORE_ABI_VERSION:
            raise LeCoreCompatibilityError(
                f"expected ABI {LECORE_ABI_VERSION}, artifact reports {self.abi_version}"
            )
        if self.isa_version != LECORE_ISA_VERSION:
            raise LeCoreCompatibilityError(
                f"expected ISA {LECORE_ISA_VERSION}, artifact reports {self.isa_version}"
            )
        self._declare_abi0()

    def _declare_identity(self) -> None:
        dll = self._dll
        dll.lecore_abi_version.argtypes = []
        dll.lecore_abi_version.restype = ctypes.c_uint32
        dll.lecore_isa_version.argtypes = []
        dll.lecore_isa_version.restype = ctypes.c_uint32

    def _declare_abi0(self) -> None:
        dll = self._dll
        dll.lecore_version_string.argtypes = []
        dll.lecore_version_string.restype = ctypes.c_char_p
        dll.lecore_capabilities.argtypes = []
        dll.lecore_capabilities.restype = ctypes.c_uint64
        dll.lecore_status_string.argtypes = [ctypes.c_uint32]
        dll.lecore_status_string.restype = ctypes.c_char_p
        dll.lecore_config_init_v0.argtypes = [ctypes.POINTER(_ConfigV0)]
        dll.lecore_config_init_v0.restype = None
        dll.lecore_context_create.argtypes = [
            ctypes.POINTER(_ConfigV0),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        dll.lecore_context_create.restype = ctypes.c_uint32
        dll.lecore_context_destroy.argtypes = [ctypes.c_void_p]
        dll.lecore_context_destroy.restype = None
        for query in (
            "dimension",
            "profile",
            "backend",
            "validation",
            "scalar_type",
        ):
            function = getattr(dll, f"lecore_context_{query}")
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_uint32
        dll.lecore_context_scratch_bytes.argtypes = [ctypes.c_void_p]
        dll.lecore_context_scratch_bytes.restype = ctypes.c_size_t

        for suffix, scalar in (("f64", ctypes.c_double), ("f32", ctypes.c_float)):
            pointer = ctypes.POINTER(scalar)
            validate = getattr(dll, f"lecore_validate_{suffix}")
            validate.argtypes = [pointer, ctypes.c_size_t]
            validate.restype = ctypes.c_uint32
            for name in ("normalize", "involution"):
                function = getattr(dll, f"lecore_{name}_{suffix}")
                function.argtypes = [ctypes.c_void_p, pointer, pointer]
                function.restype = ctypes.c_uint32
            for name in ("dot", "cosine"):
                function = getattr(dll, f"lecore_{name}_{suffix}")
                function.argtypes = [ctypes.c_void_p, pointer, pointer, pointer]
                function.restype = ctypes.c_uint32
            for name in ("dot_many", "cosine_many"):
                function = getattr(dll, f"lecore_{name}_{suffix}")
                function.argtypes = [
                    ctypes.c_void_p,
                    pointer,
                    pointer,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    pointer,
                ]
                function.restype = ctypes.c_uint32
            for name in ("hrr_bind", "hrr_unbind"):
                function = getattr(dll, f"lecore_{name}_{suffix}")
                function.argtypes = [ctypes.c_void_p, pointer, pointer, pointer]
                function.restype = ctypes.c_uint32
            permute = getattr(dll, f"lecore_permute_{suffix}")
            permute.argtypes = [ctypes.c_void_p, pointer, ctypes.c_int64, pointer]
            permute.restype = ctypes.c_uint32
            bundle = getattr(dll, f"lecore_bundle_{suffix}")
            bundle.argtypes = [
                ctypes.c_void_p,
                pointer,
                ctypes.c_size_t,
                ctypes.c_size_t,
                pointer,
            ]
            bundle.restype = ctypes.c_uint32
            cleanup = getattr(dll, f"lecore_cleanup_{suffix}")
            cleanup.argtypes = [
                ctypes.c_void_p,
                pointer,
                pointer,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
                pointer,
            ]
            cleanup.restype = ctypes.c_uint32
            batch = getattr(dll, f"lecore_hrr_bind_batch_{suffix}")
            batch.argtypes = [
                ctypes.c_void_p,
                pointer,
                ctypes.c_size_t,
                pointer,
                ctypes.c_size_t,
                ctypes.c_size_t,
                pointer,
                ctypes.c_size_t,
            ]
            batch.restype = ctypes.c_uint32
            for name in ("hrr_bind_fixed", "hrr_unbind_all"):
                function = getattr(dll, f"lecore_{name}_{suffix}")
                function.argtypes = [
                    ctypes.c_void_p,
                    pointer,
                    pointer,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    pointer,
                    ctypes.c_size_t,
                ]
                function.restype = ctypes.c_uint32

        dll.lecore_cosine_many_f64_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
        ]
        dll.lecore_cosine_many_f64_f32.restype = ctypes.c_uint32

    @property
    def abi_version(self) -> int:
        return int(self._dll.lecore_abi_version())

    @property
    def isa_version(self) -> int:
        return int(self._dll.lecore_isa_version())

    @property
    def version(self) -> str:
        value = self._dll.lecore_version_string()
        return value.decode("ascii") if value else ""

    @property
    def capabilities(self) -> int:
        return int(self._dll.lecore_capabilities())

    def _check(self, status: int, operation: str) -> None:
        status = int(status)
        if status == LECORE_OK:
            return
        raw_detail = self._dll.lecore_status_string(status)
        detail = raw_detail.decode("utf-8", "replace") if raw_detail else "unknown status"
        error_type = _ERROR_BY_STATUS.get(status, LeCoreError)
        raise error_type(status, operation, detail)

    def context(
        self,
        dimension: int,
        *,
        profile: str = "f64",
        backend: str = "auto",
        finite: bool = False,
    ) -> "Context":
        return Context(
            self,
            dimension,
            profile=profile,
            backend=backend,
            finite=finite,
        )

    def validate(self, values: np.ndarray) -> None:
        """Run the standalone finite validator without changing the array."""
        if not isinstance(values, np.ndarray):
            raise TypeError("values must be a NumPy ndarray")
        if values.dtype == np.dtype(np.float64):
            suffix, scalar = "f64", ctypes.c_double
        elif values.dtype == np.dtype(np.float32):
            suffix, scalar = "f32", ctypes.c_float
        else:
            raise TypeError("values dtype must be exactly native float64 or float32")
        if not values.flags.c_contiguous or not values.flags.aligned:
            raise ValueError("values must be aligned and C-contiguous")
        pointer = values.ctypes.data_as(ctypes.POINTER(scalar))
        self._check(
            getattr(self._dll, f"lecore_validate_{suffix}")(pointer, values.size),
            f"lecore_validate_{suffix}",
        )


class Context:
    """Dimension/profile-specific native context with deterministic ownership."""

    def __init__(
        self,
        library: Library,
        dimension: int,
        *,
        profile: str,
        backend: str,
        finite: bool,
    ):
        if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
            raise TypeError("dimension must be an integer")
        if not 0 < int(dimension) <= 0xFFFFFFFF:
            raise ValueError("dimension must be in 1..UINT32_MAX")
        if profile not in _PROFILE_BY_NAME:
            raise ValueError("profile must be 'f64' or 'f32'")
        if backend not in _BACKEND_BY_NAME:
            raise ValueError("backend must be 'auto', 'direct', or 'radix2'")

        self.library = library
        self.dimension = int(dimension)
        self.profile_name = profile
        self.profile = _PROFILE_BY_NAME[profile]
        self.dtype = _DTYPE_BY_PROFILE[self.profile]
        self._ctype = ctypes.c_double if profile == "f64" else ctypes.c_float
        self._suffix = profile
        self.requested_backend = _BACKEND_BY_NAME[backend]
        self.validation = LECORE_VALIDATION_FINITE if finite else LECORE_VALIDATION_SHAPE
        self._owner_thread = threading.current_thread()
        self._owner_thread_id = threading.get_ident()
        self._lock = threading.RLock()
        self._native_owner = _NativeContextOwner(
            self._lock,
            library._dll.lecore_context_destroy,
        )

        capability = _CAPABILITY_BY_PROFILE[self.profile]
        if not library.capabilities & capability:
            raise LeCoreCompatibilityError(
                f"artifact does not advertise the requested {profile} profile"
            )
        backend_capability = _CAPABILITY_BY_BACKEND.get(self.requested_backend)
        if backend_capability is not None and not library.capabilities & backend_capability:
            raise LeCoreCompatibilityError(
                f"artifact does not advertise the requested {backend} backend"
            )

        config = _ConfigV0()
        library._dll.lecore_config_init_v0(ctypes.byref(config))
        config.dimension = self.dimension
        config.profile = self.profile
        config.backend = self.requested_backend
        config.validation = self.validation
        status = library._dll.lecore_context_create(
            ctypes.byref(config), ctypes.byref(self._native_owner.pointer)
        )
        try:
            library._check(status, "lecore_context_create")
        except Exception:
            self._native_owner.close()
            raise
        try:
            self._finalizer = weakref.finalize(
                self,
                _finalize_native_context,
                self._native_owner,
            )
        except Exception:
            self._native_owner.close()
            raise
        try:
            self._verify_context()
        except Exception:
            self.close()
            raise

    def _verify_context(self) -> None:
        dll = self.library._dll
        self._check_owner()
        with self._lock:
            handle = self._pointer_locked()
            observed = {
                "dimension": int(dll.lecore_context_dimension(handle)),
                "profile": int(dll.lecore_context_profile(handle)),
                "validation": int(dll.lecore_context_validation(handle)),
                "scalar": int(dll.lecore_context_scalar_type(handle)),
            }
            backend = int(dll.lecore_context_backend(handle))
        expected = {
            "dimension": self.dimension,
            "profile": self.profile,
            "validation": self.validation,
            "scalar": _SCALAR_BY_PROFILE[self.profile],
        }
        if observed != expected:
            raise LeCoreCompatibilityError(
                f"created context metadata mismatch: expected {expected}, observed {observed}"
            )
        self.backend = backend
        if self.requested_backend != LECORE_BACKEND_AUTO and self.backend != self.requested_backend:
            raise LeCoreCompatibilityError(
                f"requested backend {self.requested_backend}, context reports {self.backend}"
            )

    @property
    def scratch_bytes(self) -> int:
        self._check_owner()
        with self._lock:
            return int(
                self.library._dll.lecore_context_scratch_bytes(
                    self._pointer_locked()
                )
            )

    @property
    def closed(self) -> bool:
        self._check_owner()
        with self._lock:
            return not bool(self._native_owner.pointer)

    def _check_owner(self) -> None:
        current_thread_id = threading.get_ident()
        if (
            threading.current_thread() is not self._owner_thread
            or current_thread_id != self._owner_thread_id
        ):
            raise LeCoreThreadError(
                "liblecore context belongs to thread "
                f"{self._owner_thread_id}, not {current_thread_id}"
            )

    def _pointer_locked(self) -> ctypes.c_void_p:
        if not self._native_owner.pointer:
            raise LeCoreClosedError("liblecore context is closed")
        return self._native_owner.pointer

    def _invoke(self, operation: str, function, *arguments) -> None:
        self._check_owner()
        with self._lock:
            status = function(self._pointer_locked(), *arguments)
            self.library._check(status, operation)

    def close(self) -> None:
        self._check_owner()
        self._finalizer()

    def __enter__(self) -> "Context":
        self._check_owner()
        with self._lock:
            self._pointer_locked()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __copy__(self):
        raise TypeError("liblecore Context owns a unique native handle and cannot be copied")

    def __deepcopy__(self, memo):
        del memo
        raise TypeError("liblecore Context owns a unique native handle and cannot be copied")

    def _pointer_to(self, array: np.ndarray):
        return array.ctypes.data_as(ctypes.POINTER(self._ctype))

    def _array(
        self,
        values: np.ndarray,
        name: str,
        *,
        ndim: int,
        shape: tuple[Optional[int], ...],
        writable: bool = False,
    ) -> np.ndarray:
        if not isinstance(values, np.ndarray):
            raise TypeError(f"{name} must be a NumPy ndarray")
        if values.dtype != self.dtype:
            raise TypeError(
                f"{name} dtype must be exactly native {self.dtype.name}, got {values.dtype}"
            )
        if values.ndim != ndim:
            raise ValueError(f"{name} must have {ndim} dimensions, got {values.ndim}")
        for axis, expected in enumerate(shape):
            if expected is not None and values.shape[axis] != expected:
                raise ValueError(
                    f"{name} axis {axis} must have length {expected}, got {values.shape[axis]}"
                )
        if not values.flags.c_contiguous or not values.flags.aligned:
            raise ValueError(f"{name} must be aligned and C-contiguous")
        if writable and not values.flags.writeable:
            raise ValueError(f"{name} must be writable")
        return values

    def _vector(self, values: np.ndarray, name: str = "values") -> np.ndarray:
        return self._array(values, name, ndim=1, shape=(self.dimension,))

    def _rows(self, values: np.ndarray, name: str = "rows") -> np.ndarray:
        rows = self._array(values, name, ndim=2, shape=(None, self.dimension))
        if rows.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one row")
        return rows

    def _output(
        self,
        output: Optional[np.ndarray],
        shape: tuple[int, ...],
        name: str = "out",
    ) -> np.ndarray:
        if output is None:
            return np.empty(shape, dtype=self.dtype)
        return self._array(output, name, ndim=len(shape), shape=shape, writable=True)

    def _unary(
        self,
        operation: str,
        values: np.ndarray,
        *,
        out: Optional[np.ndarray],
    ) -> np.ndarray:
        self._check_owner()
        values = self._vector(values)
        output = self._output(out, (self.dimension,))
        _reject_partial_overlap(output, values, allow_exact=True)
        function = getattr(self.library._dll, f"lecore_{operation}_{self._suffix}")
        self._invoke(
            f"lecore_{operation}_{self._suffix}",
            function,
            self._pointer_to(values),
            self._pointer_to(output),
        )
        return output

    def _binary_vector(
        self,
        operation: str,
        a: np.ndarray,
        b: np.ndarray,
        *,
        out: Optional[np.ndarray],
    ) -> np.ndarray:
        self._check_owner()
        a = self._vector(a, "a")
        b = self._vector(b, "b")
        output = self._output(out, (self.dimension,))
        _reject_partial_overlap(output, a, b, allow_exact=True)
        function = getattr(self.library._dll, f"lecore_{operation}_{self._suffix}")
        self._invoke(
            f"lecore_{operation}_{self._suffix}",
            function,
            self._pointer_to(a),
            self._pointer_to(b),
            self._pointer_to(output),
        )
        return output

    def _binary_scalar(self, operation: str, a: np.ndarray, b: np.ndarray) -> float:
        self._check_owner()
        a = self._vector(a, "a")
        b = self._vector(b, "b")
        output = self._ctype()
        function = getattr(self.library._dll, f"lecore_{operation}_{self._suffix}")
        self._invoke(
            f"lecore_{operation}_{self._suffix}",
            function,
            self._pointer_to(a),
            self._pointer_to(b),
            ctypes.byref(output),
        )
        return float(output.value)

    def normalize(self, values: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
        return self._unary("normalize", values, out=out)

    def involution(self, values: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
        return self._unary("involution", values, out=out)

    def permute(
        self,
        values: np.ndarray,
        shift: int,
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._check_owner()
        values = self._vector(values)
        if isinstance(shift, bool) or not isinstance(shift, (int, np.integer)):
            raise TypeError("shift must be an integer")
        if not -(1 << 63) <= int(shift) < (1 << 63):
            raise OverflowError("shift does not fit int64")
        output = self._output(out, (self.dimension,))
        _reject_partial_overlap(output, values, allow_exact=True)
        function = getattr(self.library._dll, f"lecore_permute_{self._suffix}")
        self._invoke(
            f"lecore_permute_{self._suffix}",
            function,
            self._pointer_to(values),
            int(shift),
            self._pointer_to(output),
        )
        return output

    def dot(self, a: np.ndarray, b: np.ndarray) -> float:
        return self._binary_scalar("dot", a, b)

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return self._binary_scalar("cosine", a, b)

    def bind(
        self,
        a: np.ndarray,
        b: np.ndarray,
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self._binary_vector("hrr_bind", a, b, out=out)

    def unbind(
        self,
        composite: np.ndarray,
        key: np.ndarray,
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self._binary_vector("hrr_unbind", composite, key, out=out)

    def _score_many(
        self,
        operation: str,
        query: np.ndarray,
        rows: np.ndarray,
        *,
        out: Optional[np.ndarray],
    ) -> np.ndarray:
        self._check_owner()
        query = self._vector(query, "query")
        rows = self._rows(rows)
        output = self._output(out, (rows.shape[0],))
        _reject_partial_overlap(output, query, rows, allow_exact=False)
        function = getattr(self.library._dll, f"lecore_{operation}_{self._suffix}")
        self._invoke(
            f"lecore_{operation}_{self._suffix}",
            function,
            self._pointer_to(query),
            self._pointer_to(rows),
            rows.shape[0],
            self.dimension,
            self._pointer_to(output),
        )
        return output

    def dot_many(
        self,
        query: np.ndarray,
        rows: np.ndarray,
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self._score_many("dot_many", query, rows, out=out)

    def cosine_many(
        self,
        query: np.ndarray,
        rows: np.ndarray,
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self._score_many("cosine_many", query, rows, out=out)

    def bundle(self, rows: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
        self._check_owner()
        rows = self._rows(rows)
        output = self._output(out, (self.dimension,))
        _reject_partial_overlap(output, rows, allow_exact=False)
        function = getattr(self.library._dll, f"lecore_bundle_{self._suffix}")
        self._invoke(
            f"lecore_bundle_{self._suffix}",
            function,
            self._pointer_to(rows),
            rows.shape[0],
            self.dimension,
            self._pointer_to(output),
        )
        return output

    def cleanup(self, query: np.ndarray, candidates: np.ndarray) -> tuple[int, float]:
        self._check_owner()
        query = self._vector(query, "query")
        candidates = self._rows(candidates, "candidates")
        index = ctypes.c_size_t()
        score = self._ctype()
        function = getattr(self.library._dll, f"lecore_cleanup_{self._suffix}")
        self._invoke(
            f"lecore_cleanup_{self._suffix}",
            function,
            self._pointer_to(query),
            self._pointer_to(candidates),
            candidates.shape[0],
            self.dimension,
            ctypes.byref(index),
            ctypes.byref(score),
        )
        return int(index.value), float(score.value)

    def bind_batch(
        self,
        a_rows: np.ndarray,
        b_rows: np.ndarray,
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._check_owner()
        a_rows = self._rows(a_rows, "a_rows")
        b_rows = self._rows(b_rows, "b_rows")
        if a_rows.shape != b_rows.shape:
            raise ValueError("a_rows and b_rows must have equal shape")
        output = self._output(out, a_rows.shape)
        _reject_partial_overlap(output, a_rows, b_rows, allow_exact=False)
        function = getattr(self.library._dll, f"lecore_hrr_bind_batch_{self._suffix}")
        self._invoke(
            f"lecore_hrr_bind_batch_{self._suffix}",
            function,
            self._pointer_to(a_rows),
            self.dimension,
            self._pointer_to(b_rows),
            self.dimension,
            a_rows.shape[0],
            self._pointer_to(output),
            self.dimension,
        )
        return output

    def bind_fixed(
        self,
        role: np.ndarray,
        rows: np.ndarray,
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._check_owner()
        role = self._vector(role, "role")
        rows = self._rows(rows)
        output = self._output(out, rows.shape)
        _reject_partial_overlap(output, role, rows, allow_exact=False)
        function = getattr(self.library._dll, f"lecore_hrr_bind_fixed_{self._suffix}")
        self._invoke(
            f"lecore_hrr_bind_fixed_{self._suffix}",
            function,
            self._pointer_to(role),
            self._pointer_to(rows),
            rows.shape[0],
            self.dimension,
            self._pointer_to(output),
            self.dimension,
        )
        return output

    def unbind_all(
        self,
        trace: np.ndarray,
        keys: np.ndarray,
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._check_owner()
        trace = self._vector(trace, "trace")
        keys = self._rows(keys, "keys")
        output = self._output(out, keys.shape)
        _reject_partial_overlap(output, trace, keys, allow_exact=False)
        function = getattr(self.library._dll, f"lecore_hrr_unbind_all_{self._suffix}")
        self._invoke(
            f"lecore_hrr_unbind_all_{self._suffix}",
            function,
            self._pointer_to(trace),
            self._pointer_to(keys),
            keys.shape[0],
            self.dimension,
            self._pointer_to(output),
            self.dimension,
        )
        return output

    def cosine_many_f64_f32(
        self,
        query: np.ndarray,
        rows: np.ndarray,
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Score an f64 query against f32 rows using ordered f64 reductions."""
        self._check_owner()
        if self.profile != LECORE_PROFILE_HRR_F32_V1:
            raise LeCoreCompatibilityError("mixed scoring requires an f32 context")
        query = _strict_external_array(
            query, "query", np.dtype(np.float64), 1, (self.dimension,)
        )
        rows = self._rows(rows)
        if out is None:
            output = np.empty((rows.shape[0],), dtype=np.float64)
        else:
            output = _strict_external_array(
                out,
                "out",
                np.dtype(np.float64),
                1,
                (rows.shape[0],),
                writable=True,
            )
        _reject_partial_overlap(output, query, rows, allow_exact=False)
        self._invoke(
            "lecore_cosine_many_f64_f32",
            self.library._dll.lecore_cosine_many_f64_f32,
            query.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._pointer_to(rows),
            rows.shape[0],
            self.dimension,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return output


def _strict_external_array(
    values: np.ndarray,
    name: str,
    dtype: np.dtype,
    ndim: int,
    shape: tuple[Optional[int], ...],
    *,
    writable: bool = False,
) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a NumPy ndarray")
    if values.dtype != dtype:
        raise TypeError(f"{name} dtype must be exactly native {dtype.name}, got {values.dtype}")
    if values.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {values.ndim}")
    for axis, expected in enumerate(shape):
        if expected is not None and values.shape[axis] != expected:
            raise ValueError(
                f"{name} axis {axis} must have length {expected}, got {values.shape[axis]}"
            )
    if not values.flags.c_contiguous or not values.flags.aligned:
        raise ValueError(f"{name} must be aligned and C-contiguous")
    if writable and not values.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return values


__all__ = [
    "Context",
    "Library",
    "LeCoreError",
    "LeCoreInvalidArgumentError",
    "LeCoreDimensionError",
    "LeCoreProfileError",
    "LeCoreBackendError",
    "LeCoreOverflowError",
    "LeCoreMemoryError",
    "LeCoreUnsupportedError",
    "LeCoreNonFiniteError",
    "LeCoreFormatError",
    "LeCoreChecksumError",
    "LeCoreCompatibilityError",
    "LeCoreClosedError",
    "LeCoreThreadError",
]
