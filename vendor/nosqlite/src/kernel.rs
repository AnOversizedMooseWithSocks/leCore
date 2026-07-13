use std::ffi::c_uchar;

extern "C" {
    fn nosqlite_fnv1a64(data: *const c_uchar, len: usize) -> u64;
    fn nosqlite_next_id() -> u64;
}

pub fn hash_bytes(bytes: &[u8]) -> u64 {
    unsafe { nosqlite_fnv1a64(bytes.as_ptr(), bytes.len()) }
}

pub fn next_id() -> u64 {
    unsafe { nosqlite_next_id() }
}
