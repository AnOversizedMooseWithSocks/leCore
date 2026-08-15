//! Rust access to liblecore's ABI-0 preview.
//!
//! [`raw`] mirrors the public numeric C header. [`Context`] is the intentionally
//! small safe layer: it owns an opaque context, validates slices and semantic
//! profiles, and serializes scratch-using calls through `&mut self`.

#![deny(unsafe_op_in_unsafe_fn)]

pub mod raw;

use std::ffi::CStr;
use std::fmt;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::rc::Rc;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Status(pub raw::lecore_status);

impl Status {
    pub const OK: Self = Self(raw::LECORE_OK);
    pub const INVALID_ARGUMENT: Self = Self(raw::LECORE_EINVAL);
    pub const INVALID_DIMENSION: Self = Self(raw::LECORE_EDIM);
    pub const PROFILE: Self = Self(raw::LECORE_EPROFILE);
    pub const BACKEND: Self = Self(raw::LECORE_EBACKEND);
    pub const OVERFLOW: Self = Self(raw::LECORE_EOVERFLOW);
    pub const NO_MEMORY: Self = Self(raw::LECORE_ENOMEM);
    pub const UNSUPPORTED: Self = Self(raw::LECORE_EUNSUPPORTED);
    pub const NONFINITE: Self = Self(raw::LECORE_ENONFINITE);
    pub const FORMAT: Self = Self(raw::LECORE_EFORMAT);
    pub const CHECKSUM: Self = Self(raw::LECORE_ECHECKSUM);

    pub fn is_ok(self) -> bool {
        self == Self::OK
    }

    /// Fetches the library's static description, retaining unknown raw values.
    pub fn message(self) -> String {
        // SAFETY: The ABI accepts every u32 status and returns a static string.
        let pointer = unsafe { raw::lecore_status_string(self.0) };
        if pointer.is_null() {
            return "unknown status".to_owned();
        }
        // SAFETY: The ABI promises a non-null, NUL-terminated static string.
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }
}

impl fmt::Display for Status {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} ({})", self.message(), self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Profile {
    HrrF64V1,
    HrrF32V1,
}

impl Profile {
    pub const fn as_raw(self) -> raw::lecore_profile {
        match self {
            Self::HrrF64V1 => raw::LECORE_PROFILE_HRR_F64_V1,
            Self::HrrF32V1 => raw::LECORE_PROFILE_HRR_F32_V1,
        }
    }

    const fn scalar_type(self) -> raw::lecore_scalar_type {
        match self {
            Self::HrrF64V1 => raw::LECORE_SCALAR_F64,
            Self::HrrF32V1 => raw::LECORE_SCALAR_F32,
        }
    }

    const fn capability(self) -> u64 {
        match self {
            Self::HrrF64V1 => raw::LECORE_CAP_HRR_F64,
            Self::HrrF32V1 => raw::LECORE_CAP_HRR_F32,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    Direct,
    Radix2,
}

impl Backend {
    pub const fn as_raw(self) -> raw::lecore_backend {
        match self {
            Self::Direct => raw::LECORE_BACKEND_DIRECT,
            Self::Radix2 => raw::LECORE_BACKEND_RADIX2,
        }
    }

    const fn capability(self) -> u64 {
        match self {
            Self::Direct => raw::LECORE_CAP_DIRECT,
            Self::Radix2 => raw::LECORE_CAP_RADIX2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Validation {
    Shape,
    Finite,
}

impl Validation {
    pub const fn as_raw(self) -> raw::lecore_validation {
        match self {
            Self::Shape => raw::LECORE_VALIDATION_SHAPE,
            Self::Finite => raw::LECORE_VALIDATION_FINITE,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    Native(Status),
    InvalidDimension(usize),
    UnsupportedRadix2Dimension(usize),
    ProfileMismatch {
        required: Profile,
        actual: Profile,
    },
    InvalidLength {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidRowCount,
    InvalidStride {
        stride: usize,
        dimension: usize,
    },
    SizeOverflow,
    AbiMismatch {
        expected: u32,
        actual: u32,
    },
    IsaMismatch {
        expected: u32,
        actual: u32,
    },
    LibraryInvariant(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Native(status) => write!(formatter, "liblecore returned {status}"),
            Self::InvalidDimension(dimension) => {
                write!(formatter, "dimension {dimension} is outside 1..=UINT32_MAX")
            }
            Self::UnsupportedRadix2Dimension(dimension) => write!(
                formatter,
                "radix-2 requires a power-of-two dimension, got {dimension}"
            ),
            Self::ProfileMismatch { required, actual } => {
                write!(
                    formatter,
                    "operation requires {required:?}, context is {actual:?}"
                )
            }
            Self::InvalidLength {
                argument,
                expected,
                actual,
            } => write!(
                formatter,
                "{argument} has length {actual}, expected {expected}"
            ),
            Self::InvalidRowCount => write!(formatter, "row_count must be positive"),
            Self::InvalidStride { stride, dimension } => write!(
                formatter,
                "row stride {stride} is smaller than dimension {dimension}"
            ),
            Self::SizeOverflow => write!(formatter, "matrix span overflows usize"),
            Self::AbiMismatch { expected, actual } => {
                write!(
                    formatter,
                    "liblecore ABI {actual} does not match expected ABI {expected}"
                )
            }
            Self::IsaMismatch { expected, actual } => {
                write!(
                    formatter,
                    "liblecore ISA {actual} does not match expected ISA {expected}"
                )
            }
            Self::LibraryInvariant(message) => {
                write!(formatter, "liblecore invariant failed: {message}")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Owned opaque liblecore state.
///
/// Calls require `&mut self` because both direct exact-alias paths and radix-2
/// transforms may use context-owned scratch.
///
/// ABI-0 has not yet promised that a live context may be transferred between
/// threads, so this wrapper is deliberately neither `Send` nor `Sync`.
///
/// ```compile_fail
/// fn require_send<T: Send>() {}
/// require_send::<lecore_sys::Context>();
/// ```
///
/// ```compile_fail
/// fn require_sync<T: Sync>() {}
/// require_sync::<lecore_sys::Context>();
/// ```
pub struct Context {
    pointer: NonNull<raw::lecore_context>,
    dimension: usize,
    profile: Profile,
    backend: Backend,
    validation: Validation,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl Context {
    pub fn new(dimension: usize, profile: Profile, backend: Backend) -> Result<Self, Error> {
        Self::with_validation(dimension, profile, backend, Validation::Shape)
    }

    pub fn with_validation(
        dimension: usize,
        profile: Profile,
        backend: Backend,
        validation: Validation,
    ) -> Result<Self, Error> {
        let dimension_u32 =
            u32::try_from(dimension).map_err(|_| Error::InvalidDimension(dimension))?;
        if dimension_u32 == 0 {
            return Err(Error::InvalidDimension(dimension));
        }
        if backend == Backend::Radix2 && !dimension.is_power_of_two() {
            return Err(Error::UnsupportedRadix2Dimension(dimension));
        }

        validate_library_versions(abi_version(), isa_version())?;
        let available = capabilities();
        if available & profile.capability() == 0 || available & backend.capability() == 0 {
            return Err(Error::Native(Status::UNSUPPORTED));
        }

        let mut config = MaybeUninit::<raw::lecore_config_v0>::uninit();
        // SAFETY: The initializer writes the complete public ABI-0 structure.
        unsafe { raw::lecore_config_init_v0(config.as_mut_ptr()) };
        // SAFETY: Guaranteed by lecore_config_init_v0's public contract.
        let mut config = unsafe { config.assume_init() };
        config.dimension = dimension_u32;
        config.profile = profile.as_raw();
        config.backend = backend.as_raw();
        config.validation = validation.as_raw();

        let mut pointer = std::ptr::null_mut();
        // SAFETY: Both pointers are valid and config was initialized by the ABI helper.
        let status = unsafe { raw::lecore_context_create(&config, &mut pointer) };
        if status != raw::LECORE_OK {
            if !pointer.is_null() {
                // SAFETY: A non-null failure result violates the ABI but is still library-owned.
                unsafe { raw::lecore_context_destroy(pointer) };
                return Err(Error::LibraryInvariant("failure returned a live context"));
            }
            return Err(Error::Native(Status(status)));
        }
        let pointer = NonNull::new(pointer)
            .ok_or(Error::LibraryInvariant("success returned a null context"))?;
        let context = Self {
            pointer,
            dimension,
            profile,
            backend,
            validation,
            _not_send_or_sync: PhantomData,
        };
        context.verify_native_identity()?;
        Ok(context)
    }

    pub fn direct_f64(dimension: usize) -> Result<Self, Error> {
        Self::new(dimension, Profile::HrrF64V1, Backend::Direct)
    }

    pub fn radix2_f64(dimension: usize) -> Result<Self, Error> {
        Self::new(dimension, Profile::HrrF64V1, Backend::Radix2)
    }

    pub fn direct_f32(dimension: usize) -> Result<Self, Error> {
        Self::new(dimension, Profile::HrrF32V1, Backend::Direct)
    }

    pub fn radix2_f32(dimension: usize) -> Result<Self, Error> {
        Self::new(dimension, Profile::HrrF32V1, Backend::Radix2)
    }

    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    pub const fn profile(&self) -> Profile {
        self.profile
    }

    pub const fn backend(&self) -> Backend {
        self.backend
    }

    pub const fn validation(&self) -> Validation {
        self.validation
    }

    pub fn scratch_bytes(&self) -> usize {
        // SAFETY: self owns a live context for the duration of this call.
        unsafe { raw::lecore_context_scratch_bytes(self.pointer.as_ptr()) }
    }

    pub fn as_raw(&self) -> *mut raw::lecore_context {
        self.pointer.as_ptr()
    }

    pub fn bind_f64(&mut self, a: &[f64], b: &[f64], output: &mut [f64]) -> Result<(), Error> {
        self.require_profile(Profile::HrrF64V1)?;
        self.check_vector(a, "a")?;
        self.check_vector(b, "b")?;
        self.check_vector(output, "output")?;
        // SAFETY: Exact lengths and disjoint mutable output are established by Rust borrows.
        status_result(unsafe {
            raw::lecore_hrr_bind_f64(
                self.pointer.as_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                output.as_mut_ptr(),
            )
        })
    }

    pub fn unbind_f64(
        &mut self,
        composite: &[f64],
        key: &[f64],
        output: &mut [f64],
    ) -> Result<(), Error> {
        self.require_profile(Profile::HrrF64V1)?;
        self.check_vector(composite, "composite")?;
        self.check_vector(key, "key")?;
        self.check_vector(output, "output")?;
        // SAFETY: Exact lengths and disjoint mutable output are established by Rust borrows.
        status_result(unsafe {
            raw::lecore_hrr_unbind_f64(
                self.pointer.as_ptr(),
                composite.as_ptr(),
                key.as_ptr(),
                output.as_mut_ptr(),
            )
        })
    }

    pub fn bind_f32(&mut self, a: &[f32], b: &[f32], output: &mut [f32]) -> Result<(), Error> {
        self.require_profile(Profile::HrrF32V1)?;
        self.check_vector(a, "a")?;
        self.check_vector(b, "b")?;
        self.check_vector(output, "output")?;
        // SAFETY: Exact lengths and disjoint mutable output are established by Rust borrows.
        status_result(unsafe {
            raw::lecore_hrr_bind_f32(
                self.pointer.as_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                output.as_mut_ptr(),
            )
        })
    }

    pub fn unbind_f32(
        &mut self,
        composite: &[f32],
        key: &[f32],
        output: &mut [f32],
    ) -> Result<(), Error> {
        self.require_profile(Profile::HrrF32V1)?;
        self.check_vector(composite, "composite")?;
        self.check_vector(key, "key")?;
        self.check_vector(output, "output")?;
        // SAFETY: Exact lengths and disjoint mutable output are established by Rust borrows.
        status_result(unsafe {
            raw::lecore_hrr_unbind_f32(
                self.pointer.as_ptr(),
                composite.as_ptr(),
                key.as_ptr(),
                output.as_mut_ptr(),
            )
        })
    }

    pub fn cosine_many_f64_f32(
        &mut self,
        query: &[f64],
        rows: &[f32],
        row_count: usize,
        row_stride: usize,
        output: &mut [f64],
    ) -> Result<(), Error> {
        self.require_profile(Profile::HrrF32V1)?;
        self.check_vector(query, "query")?;
        let required_rows = self.matrix_span(row_count, row_stride)?;
        if rows.len() < required_rows {
            return Err(Error::InvalidLength {
                argument: "rows",
                expected: required_rows,
                actual: rows.len(),
            });
        }
        if output.len() != row_count {
            return Err(Error::InvalidLength {
                argument: "output",
                expected: row_count,
                actual: output.len(),
            });
        }
        // SAFETY: All spans and output counts were overflow-checked above.
        status_result(unsafe {
            raw::lecore_cosine_many_f64_f32(
                self.pointer.as_ptr(),
                query.as_ptr(),
                rows.as_ptr(),
                row_count,
                row_stride,
                output.as_mut_ptr(),
            )
        })
    }

    fn verify_native_identity(&self) -> Result<(), Error> {
        let pointer = self.pointer.as_ptr();
        // SAFETY: pointer denotes the live context owned by self.
        let native_dimension = unsafe { raw::lecore_context_dimension(pointer) } as usize;
        // SAFETY: pointer denotes the live context owned by self.
        let native_profile = unsafe { raw::lecore_context_profile(pointer) };
        // SAFETY: pointer denotes the live context owned by self.
        let native_backend = unsafe { raw::lecore_context_backend(pointer) };
        // SAFETY: pointer denotes the live context owned by self.
        let native_validation = unsafe { raw::lecore_context_validation(pointer) };
        // SAFETY: pointer denotes the live context owned by self.
        let native_scalar = unsafe { raw::lecore_context_scalar_type(pointer) };
        if native_dimension != self.dimension {
            return Err(Error::LibraryInvariant("dimension introspection mismatch"));
        }
        if native_profile != self.profile.as_raw() {
            return Err(Error::LibraryInvariant("profile introspection mismatch"));
        }
        if native_backend != self.backend.as_raw() {
            return Err(Error::LibraryInvariant("backend introspection mismatch"));
        }
        if native_validation != self.validation.as_raw() {
            return Err(Error::LibraryInvariant("validation introspection mismatch"));
        }
        if native_scalar != self.profile.scalar_type() {
            return Err(Error::LibraryInvariant("scalar introspection mismatch"));
        }
        Ok(())
    }

    fn require_profile(&self, required: Profile) -> Result<(), Error> {
        if self.profile != required {
            return Err(Error::ProfileMismatch {
                required,
                actual: self.profile,
            });
        }
        Ok(())
    }

    fn check_vector<T>(&self, values: &[T], argument: &'static str) -> Result<(), Error> {
        if values.len() != self.dimension {
            return Err(Error::InvalidLength {
                argument,
                expected: self.dimension,
                actual: values.len(),
            });
        }
        Ok(())
    }

    fn matrix_span(&self, row_count: usize, row_stride: usize) -> Result<usize, Error> {
        if row_count == 0 {
            return Err(Error::InvalidRowCount);
        }
        if row_stride < self.dimension {
            return Err(Error::InvalidStride {
                stride: row_stride,
                dimension: self.dimension,
            });
        }
        (row_count - 1)
            .checked_mul(row_stride)
            .and_then(|last| last.checked_add(self.dimension))
            .ok_or(Error::SizeOverflow)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: Context uniquely owns this live pointer and drops it once.
        unsafe { raw::lecore_context_destroy(self.pointer.as_ptr()) };
    }
}

impl fmt::Debug for Context {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Context")
            .field("dimension", &self.dimension)
            .field("profile", &self.profile)
            .field("backend", &self.backend)
            .field("validation", &self.validation)
            .finish_non_exhaustive()
    }
}

pub fn abi_version() -> u32 {
    // SAFETY: Pure process-global introspection with no preconditions.
    unsafe { raw::lecore_abi_version() }
}

pub fn isa_version() -> u32 {
    // SAFETY: Pure process-global introspection with no preconditions.
    unsafe { raw::lecore_isa_version() }
}

pub fn capabilities() -> u64 {
    // SAFETY: Pure process-global introspection with no preconditions.
    unsafe { raw::lecore_capabilities() }
}

pub fn version() -> String {
    // SAFETY: Pure process-global introspection returning a static string.
    let pointer = unsafe { raw::lecore_version_string() };
    if pointer.is_null() {
        return String::new();
    }
    // SAFETY: The ABI promises a non-null, NUL-terminated static string.
    unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}

fn status_result(status: raw::lecore_status) -> Result<(), Error> {
    if status == raw::LECORE_OK {
        Ok(())
    } else {
        Err(Error::Native(Status(status)))
    }
}

fn validate_library_versions(actual_abi: u32, actual_isa: u32) -> Result<(), Error> {
    if actual_abi != raw::LECORE_ABI_VERSION {
        return Err(Error::AbiMismatch {
            expected: raw::LECORE_ABI_VERSION,
            actual: actual_abi,
        });
    }
    if actual_isa != raw::LECORE_ISA_VERSION {
        return Err(Error::IsaMismatch {
            expected: raw::LECORE_ISA_VERSION,
            actual: actual_isa,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independently_rejects_abi_and_isa_mismatches() {
        assert_eq!(
            validate_library_versions(raw::LECORE_ABI_VERSION + 1, raw::LECORE_ISA_VERSION),
            Err(Error::AbiMismatch {
                expected: raw::LECORE_ABI_VERSION,
                actual: raw::LECORE_ABI_VERSION + 1,
            })
        );
        assert_eq!(
            validate_library_versions(raw::LECORE_ABI_VERSION, raw::LECORE_ISA_VERSION + 1),
            Err(Error::IsaMismatch {
                expected: raw::LECORE_ISA_VERSION,
                actual: raw::LECORE_ISA_VERSION + 1,
            })
        );
        assert_eq!(
            validate_library_versions(raw::LECORE_ABI_VERSION, raw::LECORE_ISA_VERSION),
            Ok(())
        );
    }
}
