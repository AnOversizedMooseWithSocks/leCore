use lecore_sys::{
    abi_version, capabilities, isa_version, raw, version, Backend, Context, Error, Profile, Status,
    Validation,
};
use std::mem::{offset_of, size_of, MaybeUninit};

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual:.17} != {expected:.17} within {tolerance}"
    );
}

#[test]
fn raw_layout_and_version_match_abi_zero() {
    assert_eq!(abi_version(), raw::LECORE_ABI_VERSION);
    assert_eq!(isa_version(), raw::LECORE_ISA_VERSION);
    assert_eq!(version(), "0.1.0");
    assert_eq!(size_of::<raw::lecore_status>(), 4);
    assert_eq!(size_of::<raw::lecore_profile>(), 4);
    assert_eq!(offset_of!(raw::lecore_config_v0, dimension), 24);
    assert!(offset_of!(raw::lecore_config_v0, allocator) >= 28);

    if size_of::<usize>() == 8 {
        assert_eq!(size_of::<raw::lecore_allocator_v0>(), 32);
        assert_eq!(offset_of!(raw::lecore_config_v0, allocator), 32);
        assert_eq!(size_of::<raw::lecore_config_v0>(), 96);
    }

    let mut config = MaybeUninit::<raw::lecore_config_v0>::uninit();
    // SAFETY: The public initializer writes the complete ABI-0 structure.
    unsafe { raw::lecore_config_init_v0(config.as_mut_ptr()) };
    // SAFETY: Guaranteed by lecore_config_init_v0.
    let config = unsafe { config.assume_init() };
    assert_eq!(
        config.struct_size as usize,
        size_of::<raw::lecore_config_v0>()
    );
    assert_eq!(
        config.allocator.struct_size as usize,
        size_of::<raw::lecore_allocator_v0>()
    );
    assert_eq!(config.abi_version, raw::LECORE_ABI_VERSION);
}

#[test]
fn isa_mismatch_is_public_and_descriptive() {
    let error = Error::IsaMismatch {
        expected: raw::LECORE_ISA_VERSION,
        actual: raw::LECORE_ISA_VERSION + 1,
    };
    assert_eq!(
        error.to_string(),
        "liblecore ISA 2 does not match expected ISA 1"
    );
}

#[test]
fn direct_f64_context_owns_and_computes() {
    let mut context = Context::direct_f64(4).expect("direct f64 context");
    assert_eq!(context.dimension(), 4);
    assert_eq!(context.profile(), Profile::HrrF64V1);
    assert_eq!(context.backend(), Backend::Direct);
    assert_eq!(context.validation(), Validation::Shape);
    assert_eq!(context.scratch_bytes(), 4 * size_of::<f64>());

    let a = [1.0, 2.0, 0.0, -1.0];
    let b = [2.0, 0.0, 1.0, 0.0];
    let mut bound = [0.0; 4];
    context.bind_f64(&a, &b, &mut bound).unwrap();
    assert_eq!(bound, [2.0, 3.0, 1.0, 0.0]);

    let mut unbound = [0.0; 4];
    context.unbind_f64(&bound, &b, &mut unbound).unwrap();
    assert_eq!(unbound, [5.0, 6.0, 4.0, 3.0]);
}

#[test]
fn f32_and_radix2_are_typed_and_explicit() {
    let a = [1.0_f32, 2.0, 0.0, -1.0];
    let b = [2.0_f32, 0.0, 1.0, 0.0];
    let mut direct = Context::direct_f32(4).unwrap();
    let mut direct_output = [0.0_f32; 4];
    direct.bind_f32(&a, &b, &mut direct_output).unwrap();
    assert_eq!(direct_output, [2.0, 3.0, 1.0, 0.0]);

    assert_ne!(capabilities() & raw::LECORE_CAP_RADIX2, 0);
    let mut radix = Context::radix2_f32(4).unwrap();
    assert_eq!(radix.backend(), Backend::Radix2);
    assert_eq!(radix.scratch_bytes(), 4 * 4 * size_of::<f32>());
    let mut radix_output = [0.0_f32; 4];
    radix.bind_f32(&a, &b, &mut radix_output).unwrap();
    for (actual, expected) in radix_output.into_iter().zip(direct_output) {
        assert_close(actual as f64, expected as f64, 1e-5);
    }

    let mut recovered = [0.0_f32; 4];
    radix.unbind_f32(&radix_output, &b, &mut recovered).unwrap();
    for (actual, expected) in recovered.into_iter().zip([5.0, 6.0, 4.0, 3.0]) {
        assert_close(actual as f64, expected, 1e-5);
    }
}

#[test]
fn mixed_f64_f32_cosine_checks_rows_and_scores() {
    let mut context = Context::direct_f32(2).unwrap();
    let query = [1.0_f64, 2.0];
    let rows = [1.0_f32, 0.0, 99.0, 0.0, 2.0, 99.0, 0.0, 0.0];
    let mut scores = [0.0_f64; 3];
    context
        .cosine_many_f64_f32(&query, &rows, 3, 3, &mut scores)
        .unwrap();
    assert_close(scores[0], 1.0 / 5.0_f64.sqrt(), 1e-15);
    assert_close(scores[1], 2.0 / 5.0_f64.sqrt(), 1e-15);
    assert_eq!(scores[2], 0.0);
}

#[test]
fn safe_layer_rejects_dimensions_profiles_and_bad_slices() {
    assert_eq!(
        Context::direct_f64(0).unwrap_err(),
        Error::InvalidDimension(0)
    );
    assert_eq!(
        Context::radix2_f64(3).unwrap_err(),
        Error::UnsupportedRadix2Dimension(3)
    );

    let mut f64_context = Context::direct_f64(4).unwrap();
    let mut f32_output = [0.0_f32; 4];
    assert_eq!(
        f64_context
            .bind_f32(&[0.0; 4], &[0.0; 4], &mut f32_output)
            .unwrap_err(),
        Error::ProfileMismatch {
            required: Profile::HrrF32V1,
            actual: Profile::HrrF64V1,
        }
    );
    let mut short_output = [0.0_f64; 3];
    assert_eq!(
        f64_context
            .bind_f64(&[0.0; 4], &[0.0; 4], &mut short_output)
            .unwrap_err(),
        Error::InvalidLength {
            argument: "output",
            expected: 4,
            actual: 3,
        }
    );

    let mut f32_context = Context::direct_f32(2).unwrap();
    let mut score = [0.0_f64; 1];
    assert_eq!(
        f32_context
            .cosine_many_f64_f32(&[1.0, 2.0], &[1.0, 2.0], 0, 2, &mut [])
            .unwrap_err(),
        Error::InvalidRowCount
    );
    assert_eq!(
        f32_context
            .cosine_many_f64_f32(&[1.0, 2.0], &[1.0, 2.0], 1, 1, &mut score)
            .unwrap_err(),
        Error::InvalidStride {
            stride: 1,
            dimension: 2,
        }
    );
    assert_eq!(
        f32_context
            .cosine_many_f64_f32(&[1.0, 2.0], &[1.0], 1, 2, &mut score)
            .unwrap_err(),
        Error::InvalidLength {
            argument: "rows",
            expected: 2,
            actual: 1,
        }
    );
}

#[test]
fn finite_validation_and_unknown_statuses_remain_explicit() {
    let mut context =
        Context::with_validation(2, Profile::HrrF64V1, Backend::Direct, Validation::Finite)
            .unwrap();
    let mut output = [17.0_f64; 2];
    assert_eq!(
        context
            .bind_f64(&[f64::NAN, 0.0], &[1.0, 0.0], &mut output)
            .unwrap_err(),
        Error::Native(Status::NONFINITE)
    );
    assert_eq!(output, [17.0, 17.0]);

    let unknown = Status(0xffff_fffe);
    assert_eq!(unknown.0, 0xffff_fffe);
    assert_eq!(unknown.message(), "unknown status");
    assert!(!unknown.is_ok());
}
