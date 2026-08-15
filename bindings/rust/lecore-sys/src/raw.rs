//! Direct declarations for liblecore's public numeric ABI-0 header.
//!
//! Integer-valued C domains intentionally remain integer aliases instead of
//! Rust enums. Unknown additive values are therefore representable safely.

#![allow(non_camel_case_types)]

use std::cell::UnsafeCell;
use std::ffi::{c_char, c_void};
use std::marker::PhantomData;

pub const LECORE_ABI_VERSION: u32 = 0;
pub const LECORE_ISA_VERSION: u32 = 1;

pub type lecore_status = u32;
pub const LECORE_OK: lecore_status = 0;
pub const LECORE_EINVAL: lecore_status = 1;
pub const LECORE_EDIM: lecore_status = 2;
pub const LECORE_EPROFILE: lecore_status = 3;
pub const LECORE_EBACKEND: lecore_status = 4;
pub const LECORE_EOVERFLOW: lecore_status = 5;
pub const LECORE_ENOMEM: lecore_status = 6;
pub const LECORE_EUNSUPPORTED: lecore_status = 7;
pub const LECORE_ENONFINITE: lecore_status = 8;
pub const LECORE_EFORMAT: lecore_status = 9;
pub const LECORE_ECHECKSUM: lecore_status = 10;

pub type lecore_profile = u32;
pub const LECORE_PROFILE_HRR_F64_V1: lecore_profile = 0x0001_0001;
pub const LECORE_PROFILE_HRR_F32_V1: lecore_profile = 0x0001_0002;

pub type lecore_backend = u32;
pub const LECORE_BACKEND_AUTO: lecore_backend = 0;
pub const LECORE_BACKEND_DIRECT: lecore_backend = 1;
pub const LECORE_BACKEND_RADIX2: lecore_backend = 2;

pub type lecore_validation = u32;
pub const LECORE_VALIDATION_SHAPE: lecore_validation = 0;
pub const LECORE_VALIDATION_FINITE: lecore_validation = 1;

pub type lecore_scalar_type = u32;
pub const LECORE_SCALAR_F64: lecore_scalar_type = 1;
pub const LECORE_SCALAR_F32: lecore_scalar_type = 2;

pub const LECORE_CAP_HRR_F64: u64 = 1 << 0;
pub const LECORE_CAP_HRR_F32: u64 = 1 << 1;
pub const LECORE_CAP_DIRECT: u64 = 1 << 2;
pub const LECORE_CAP_RADIX2: u64 = 1 << 3;
pub const LECORE_CAP_BATCH: u64 = 1 << 4;
pub const LECORE_CAP_MIXED_F64_F32: u64 = 1 << 5;
pub const LECORE_CAP_FINITE_VALIDATION: u64 = 1 << 6;
pub const LECORE_CAP_FORMAT: u64 = 1 << 7;

pub type lecore_allocate_fn =
    Option<unsafe extern "C" fn(*mut c_void, usize, usize) -> *mut c_void>;
pub type lecore_deallocate_fn =
    Option<unsafe extern "C" fn(*mut c_void, *mut c_void, usize, usize)>;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct lecore_allocator_v0 {
    pub struct_size: u32,
    pub user: *mut c_void,
    pub allocate: lecore_allocate_fn,
    pub deallocate: lecore_deallocate_fn,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct lecore_config_v0 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub profile: lecore_profile,
    pub backend: lecore_backend,
    pub validation: lecore_validation,
    pub flags: u32,
    pub dimension: u32,
    pub allocator: lecore_allocator_v0,
    pub reserved: [u64; 4],
}

/// Opaque native state. Rust never mirrors or dereferences its private layout.
#[repr(C)]
pub struct lecore_context {
    _private: [u8; 0],
    _not_sync: PhantomData<UnsafeCell<()>>,
}

extern "C" {
    pub fn lecore_abi_version() -> u32;
    pub fn lecore_isa_version() -> u32;
    pub fn lecore_version_string() -> *const c_char;
    pub fn lecore_capabilities() -> u64;
    pub fn lecore_status_string(status: lecore_status) -> *const c_char;

    pub fn lecore_config_init_v0(config: *mut lecore_config_v0);
    pub fn lecore_context_create(
        config: *const lecore_config_v0,
        out_context: *mut *mut lecore_context,
    ) -> lecore_status;
    pub fn lecore_context_destroy(context: *mut lecore_context);
    pub fn lecore_context_dimension(context: *const lecore_context) -> u32;
    pub fn lecore_context_profile(context: *const lecore_context) -> lecore_profile;
    pub fn lecore_context_backend(context: *const lecore_context) -> lecore_backend;
    pub fn lecore_context_validation(context: *const lecore_context) -> lecore_validation;
    pub fn lecore_context_scalar_type(context: *const lecore_context) -> lecore_scalar_type;
    pub fn lecore_context_scratch_bytes(context: *const lecore_context) -> usize;

    pub fn lecore_validate_f64(values: *const f64, count: usize) -> lecore_status;
    pub fn lecore_validate_f32(values: *const f32, count: usize) -> lecore_status;

    pub fn lecore_normalize_f64(
        context: *mut lecore_context,
        input: *const f64,
        output: *mut f64,
    ) -> lecore_status;
    pub fn lecore_normalize_f32(
        context: *mut lecore_context,
        input: *const f32,
        output: *mut f32,
    ) -> lecore_status;
    pub fn lecore_dot_f64(
        context: *mut lecore_context,
        a: *const f64,
        b: *const f64,
        out_dot: *mut f64,
    ) -> lecore_status;
    pub fn lecore_dot_f32(
        context: *mut lecore_context,
        a: *const f32,
        b: *const f32,
        out_dot: *mut f32,
    ) -> lecore_status;
    pub fn lecore_dot_many_f64(
        context: *mut lecore_context,
        query: *const f64,
        rows: *const f64,
        row_count: usize,
        row_stride: usize,
        out_scores: *mut f64,
    ) -> lecore_status;
    pub fn lecore_dot_many_f32(
        context: *mut lecore_context,
        query: *const f32,
        rows: *const f32,
        row_count: usize,
        row_stride: usize,
        out_scores: *mut f32,
    ) -> lecore_status;
    pub fn lecore_cosine_f64(
        context: *mut lecore_context,
        a: *const f64,
        b: *const f64,
        out_cosine: *mut f64,
    ) -> lecore_status;
    pub fn lecore_cosine_f32(
        context: *mut lecore_context,
        a: *const f32,
        b: *const f32,
        out_cosine: *mut f32,
    ) -> lecore_status;
    pub fn lecore_cosine_many_f64(
        context: *mut lecore_context,
        query: *const f64,
        rows: *const f64,
        row_count: usize,
        row_stride: usize,
        out_scores: *mut f64,
    ) -> lecore_status;
    pub fn lecore_cosine_many_f32(
        context: *mut lecore_context,
        query: *const f32,
        rows: *const f32,
        row_count: usize,
        row_stride: usize,
        out_scores: *mut f32,
    ) -> lecore_status;

    pub fn lecore_hrr_bind_f64(
        context: *mut lecore_context,
        a: *const f64,
        b: *const f64,
        output: *mut f64,
    ) -> lecore_status;
    pub fn lecore_hrr_bind_f32(
        context: *mut lecore_context,
        a: *const f32,
        b: *const f32,
        output: *mut f32,
    ) -> lecore_status;
    pub fn lecore_hrr_unbind_f64(
        context: *mut lecore_context,
        composite: *const f64,
        key: *const f64,
        output: *mut f64,
    ) -> lecore_status;
    pub fn lecore_hrr_unbind_f32(
        context: *mut lecore_context,
        composite: *const f32,
        key: *const f32,
        output: *mut f32,
    ) -> lecore_status;
    pub fn lecore_involution_f64(
        context: *mut lecore_context,
        input: *const f64,
        output: *mut f64,
    ) -> lecore_status;
    pub fn lecore_involution_f32(
        context: *mut lecore_context,
        input: *const f32,
        output: *mut f32,
    ) -> lecore_status;
    pub fn lecore_permute_f64(
        context: *mut lecore_context,
        input: *const f64,
        shift: i64,
        output: *mut f64,
    ) -> lecore_status;
    pub fn lecore_permute_f32(
        context: *mut lecore_context,
        input: *const f32,
        shift: i64,
        output: *mut f32,
    ) -> lecore_status;
    pub fn lecore_bundle_f64(
        context: *mut lecore_context,
        rows: *const f64,
        row_count: usize,
        row_stride: usize,
        output: *mut f64,
    ) -> lecore_status;
    pub fn lecore_bundle_f32(
        context: *mut lecore_context,
        rows: *const f32,
        row_count: usize,
        row_stride: usize,
        output: *mut f32,
    ) -> lecore_status;
    pub fn lecore_cleanup_f64(
        context: *mut lecore_context,
        query: *const f64,
        candidates: *const f64,
        candidate_count: usize,
        candidate_stride: usize,
        out_index: *mut usize,
        out_score: *mut f64,
    ) -> lecore_status;
    pub fn lecore_cleanup_f32(
        context: *mut lecore_context,
        query: *const f32,
        candidates: *const f32,
        candidate_count: usize,
        candidate_stride: usize,
        out_index: *mut usize,
        out_score: *mut f32,
    ) -> lecore_status;

    pub fn lecore_hrr_bind_batch_f64(
        context: *mut lecore_context,
        a_rows: *const f64,
        a_stride: usize,
        b_rows: *const f64,
        b_stride: usize,
        row_count: usize,
        out_rows: *mut f64,
        out_stride: usize,
    ) -> lecore_status;
    pub fn lecore_hrr_bind_batch_f32(
        context: *mut lecore_context,
        a_rows: *const f32,
        a_stride: usize,
        b_rows: *const f32,
        b_stride: usize,
        row_count: usize,
        out_rows: *mut f32,
        out_stride: usize,
    ) -> lecore_status;
    pub fn lecore_hrr_bind_fixed_f64(
        context: *mut lecore_context,
        role: *const f64,
        rows: *const f64,
        row_count: usize,
        row_stride: usize,
        out_rows: *mut f64,
        out_stride: usize,
    ) -> lecore_status;
    pub fn lecore_hrr_bind_fixed_f32(
        context: *mut lecore_context,
        role: *const f32,
        rows: *const f32,
        row_count: usize,
        row_stride: usize,
        out_rows: *mut f32,
        out_stride: usize,
    ) -> lecore_status;
    pub fn lecore_hrr_unbind_all_f64(
        context: *mut lecore_context,
        trace: *const f64,
        keys: *const f64,
        key_count: usize,
        key_stride: usize,
        out_rows: *mut f64,
        out_stride: usize,
    ) -> lecore_status;
    pub fn lecore_hrr_unbind_all_f32(
        context: *mut lecore_context,
        trace: *const f32,
        keys: *const f32,
        key_count: usize,
        key_stride: usize,
        out_rows: *mut f32,
        out_stride: usize,
    ) -> lecore_status;
    pub fn lecore_cosine_many_f64_f32(
        f32_context: *mut lecore_context,
        query: *const f64,
        rows: *const f32,
        row_count: usize,
        row_stride: usize,
        out_scores: *mut f64,
    ) -> lecore_status;
}
