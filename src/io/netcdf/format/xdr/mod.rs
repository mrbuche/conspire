use crate::io::netcdf::NcType;
use core::slice::{from_raw_parts, from_raw_parts_mut};

fn as_bytes<T: NcType>(data: &[T]) -> &[u8] {
    // SAFETY: an `NcType` has no padding and is `SIZE` wide, so the values are `len * SIZE`
    // initialized bytes.
    unsafe { from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * T::SIZE) }
}

pub(in crate::io::netcdf) fn encode_be<T: NcType>(data: &[T], out: &mut Vec<u8>) {
    let src = as_bytes(data);
    let start = out.len();
    out.resize(start + src.len(), 0);
    T::xdr_swap(src, &mut out[start..]);
}

#[cfg(target_endian = "little")]
pub(in crate::io::netcdf) fn encode_le<T: NcType>(data: &[T], out: &mut Vec<u8>) {
    out.extend_from_slice(as_bytes(data));
}

#[cfg(target_endian = "big")]
pub(in crate::io::netcdf) fn encode_le<T: NcType>(data: &[T], out: &mut Vec<u8>) {
    let src = as_bytes(data);
    let start = out.len();
    out.resize(start + src.len(), 0);
    T::xdr_swap(src, &mut out[start..]);
}

pub(in crate::io::netcdf) fn decode_be<T: NcType>(bytes: &[u8]) -> Vec<T> {
    let count = bytes.len() / T::SIZE;
    let mut out: Vec<T> = vec![T::default(); count];
    // SAFETY: `out` is `count` initialized values of `SIZE` bytes each, and every bit pattern
    // that is written into them is a valid value.
    let dst = unsafe { from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), count * T::SIZE) };
    T::xdr_swap(bytes, dst);
    out
}

#[cfg(target_endian = "big")]
pub(in crate::io::netcdf) fn decode_le<T: NcType>(bytes: &[u8]) -> Vec<T> {
    decode_be::<T>(bytes)
}

#[cfg(target_endian = "little")]
pub(in crate::io::netcdf) fn decode_le<T: NcType>(bytes: &[u8]) -> Vec<T> {
    let count = bytes.len() / T::SIZE;
    let mut out: Vec<T> = Vec::with_capacity(count);
    // SAFETY: `bytes` has `count * SIZE` bytes and `out` has room for that many, they do not
    // overlap, and every bit pattern is a valid value, so `count` values are initialized.
    unsafe {
        core::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            out.as_mut_ptr().cast::<u8>(),
            count * T::SIZE,
        );
        out.set_len(count);
    }
    out
}
