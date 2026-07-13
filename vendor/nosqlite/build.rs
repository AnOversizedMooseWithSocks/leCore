fn main() {
    cc::Build::new()
        .file("c/nosqlite_kernel.c")
        .include("c")
        .warnings(true)
        .compile("nosqlite_kernel");

    println!("cargo:rerun-if-changed=c/nosqlite_kernel.c");
    println!("cargo:rerun-if-changed=c/nosqlite_kernel.h");
}
