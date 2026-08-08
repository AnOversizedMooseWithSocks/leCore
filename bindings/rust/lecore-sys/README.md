# lecore-sys

Audited Rust declarations and a small ownership wrapper for liblecore's unstable ABI-0 preview.

The default `vendored` feature compiles the repository's deterministic
`native/liblecore/amalgamation/lecore.c` with the cached `cc` crate:

```sh
cargo test --manifest-path bindings/rust/lecore-sys/Cargo.toml --offline
```

A native vendored build compiles and runs liblecore's floating-point contract
probe before advertising either numeric profile. A cross build cannot run that
target probe, so it fails unless the target has been audited and the build sets
`LECORE_SYS_ASSUME_IEC_60559=1` explicitly.

Installed-library mode never searches the machine implicitly. Disable defaults, select `installed`, and provide
the exact include directory, library directory, and link kind:

```sh
LECORE_SYS_INCLUDE_DIR=/pinned/prefix/include \
LECORE_SYS_LIB_DIR=/pinned/prefix/lib \
LECORE_SYS_LINK_KIND=static \
cargo test --manifest-path bindings/rust/lecore-sys/Cargo.toml \
  --no-default-features --features installed --offline
```

`Context` owns one opaque native context and releases it on drop. Numeric methods take `&mut self` because the C
kernel uses context-owned scratch. Vector lengths, profiles, strides, row spans, and output sizes are checked before
FFI calls. `Context` is deliberately neither `Send` nor `Sync`; ABI-0 does not yet promise that moving a live context
between threads is supported. Raw statuses remain `u32` values, so an additive future status does not create an
invalid Rust enum discriminant.
