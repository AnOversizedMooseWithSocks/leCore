use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=LECORE_SYS_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=LECORE_SYS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=LECORE_SYS_LINK_KIND");
    println!("cargo:rerun-if-env-changed=LECORE_SYS_ASSUME_IEC_60559");

    let vendored = env::var_os("CARGO_FEATURE_VENDORED").is_some();
    let installed = env::var_os("CARGO_FEATURE_INSTALLED").is_some();
    match (vendored, installed) {
        (true, false) => build_vendored(),
        (false, true) => link_installed(),
        (true, true) => panic!(
            "features `vendored` and `installed` are mutually exclusive; \
             use `--no-default-features --features installed` for an installed library"
        ),
        (false, false) => panic!("enable exactly one of `vendored` or `installed`"),
    }
}

fn build_vendored() {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR"));
    let amalgamation_dir = manifest_dir.join("../../../native/liblecore/amalgamation");
    let source = amalgamation_dir.join("lecore.c");
    let header = amalgamation_dir.join("lecore.h");
    require_file(&source, "generated liblecore amalgamation source");
    require_file(&header, "generated liblecore amalgamation header");
    validate_vendored_floating_point();

    println!("cargo:rerun-if-changed={}", source.display());
    println!("cargo:rerun-if-changed={}", header.display());

    let mut build = cc::Build::new();
    build
        .file(&source)
        .include(&amalgamation_dir)
        .define("LECORE_BUILDING_LIBRARY", "1")
        .define("LECORE_ENABLE_FORMAT", "1")
        .define("LECORE_ENABLE_RADIX2", "1")
        .warnings(true)
        .extra_warnings(true)
        .std("c11");
    if build.get_compiler().is_like_msvc() {
        build.flag("/fp:strict");
    } else {
        build
            .flag_if_supported("-fno-fast-math")
            .flag_if_supported("-ffp-contract=off");
    }
    build.compile("lecore");
    link_platform_math();
}

fn validate_vendored_floating_point() {
    let host = env::var("HOST").expect("Cargo sets HOST");
    let target = env::var("TARGET").expect("Cargo sets TARGET");
    if host != target {
        if env::var("LECORE_SYS_ASSUME_IEC_60559").as_deref() != Ok("1") {
            panic!(
                "cross-compiling vendored liblecore requires \
                 LECORE_SYS_ASSUME_IEC_60559=1 after auditing the target's \
                 IEC 60559 NaN/Inf, round-to-nearest, and evaluation behavior"
            );
        }
        return;
    }

    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR"));
    let source = manifest_dir.join("build/fp_probe.c");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    let executable = out_dir.join(if cfg!(windows) {
        "lecore_fp_probe.exe"
    } else {
        "lecore_fp_probe"
    });
    require_file(&source, "vendored floating-point probe");
    println!("cargo:rerun-if-changed={}", source.display());

    let compiler = cc::Build::new().get_compiler();
    let mut compile = compiler.to_command();
    if compiler.is_like_msvc() {
        compile
            .arg("/nologo")
            .arg("/std:c11")
            .arg("/fp:strict")
            .arg(&source)
            .arg(format!("/Fe{}", executable.display()));
    } else {
        compile
            .arg("-std=c11")
            .arg("-fno-fast-math")
            .arg("-ffp-contract=off")
            .arg(&source)
            .arg("-o")
            .arg(&executable);
        let target_family = env::var("CARGO_CFG_TARGET_FAMILY").unwrap_or_default();
        let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        if target_family == "unix" && target_os != "macos" && target_os != "ios" {
            compile.arg("-lm");
        }
    }
    let compile_status = compile.status().unwrap_or_else(|error| {
        panic!("failed to launch liblecore floating-point probe compiler: {error}")
    });
    if !compile_status.success() {
        panic!("failed to compile liblecore's floating-point probe");
    }

    let run_status = Command::new(&executable)
        .status()
        .unwrap_or_else(|error| panic!("failed to run liblecore floating-point probe: {error}"));
    if !run_status.success() {
        panic!(
            "the compiler/runtime failed liblecore's IEC 60559 and evaluation probe (status {run_status})"
        );
    }
}

fn link_installed() {
    let include_dir = required_env_path("LECORE_SYS_INCLUDE_DIR");
    let lib_dir = required_env_path("LECORE_SYS_LIB_DIR");
    let header = include_dir.join("lecore/lecore.h");
    require_file(&header, "installed <lecore/lecore.h>");
    if !lib_dir.is_dir() {
        panic!(
            "LECORE_SYS_LIB_DIR is not a directory: {}",
            lib_dir.display()
        );
    }

    let link_kind = env::var("LECORE_SYS_LINK_KIND").unwrap_or_else(|_| {
        panic!("LECORE_SYS_LINK_KIND must be explicitly set to `static` or `dylib`")
    });
    if link_kind != "static" && link_kind != "dylib" {
        panic!("LECORE_SYS_LINK_KIND must be `static` or `dylib`, got {link_kind:?}");
    }

    println!("cargo:rerun-if-changed={}", header.display());
    compile_installed_abi_check(&include_dir);
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib={link_kind}=lecore");
    if link_kind == "static" {
        link_platform_math();
    }
}

fn compile_installed_abi_check(include_dir: &Path) {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR"));
    let source = manifest_dir.join("build/abi_check.c");
    require_file(&source, "installed-header ABI check");
    println!("cargo:rerun-if-changed={}", source.display());

    cc::Build::new()
        .file(source)
        .include(include_dir)
        .cargo_metadata(false)
        .warnings(true)
        .extra_warnings(true)
        .std("c11")
        .compile("lecore_rust_abi_check");
}

fn required_env_path(name: &str) -> PathBuf {
    let value = env::var_os(name).unwrap_or_else(|| {
        panic!("{name} is required with the `installed` feature; no global search is performed")
    });
    PathBuf::from(value)
}

fn require_file(path: &Path, description: &str) {
    if !path.is_file() {
        panic!("missing {description}: {}", path.display());
    }
}

fn link_platform_math() {
    let target_family = env::var("CARGO_CFG_TARGET_FAMILY").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_family == "unix" && target_os != "macos" && target_os != "ios" {
        println!("cargo:rustc-link-lib=m");
    }
}
