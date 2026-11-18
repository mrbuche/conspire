use super::dense_links;
use crate::io::netcdf::format::hdf5::{FHDB, FHIB, FRHP, Sizes};

const SIZES: Sizes = Sizes {
    offset: 8,
    length: 8,
};

const OH: u64 = 0x1234;

fn put(b: &mut [u8], p: usize, bytes: &[u8]) {
    b[p..p + bytes.len()].copy_from_slice(bytes);
}

fn direct_block(b: &mut [u8], at: usize, block_offset_bytes: usize) {
    put(b, at, FHDB);
    let q = at + 4 + 1 + SIZES.offset + block_offset_bytes;
    put(b, q, &[1, 0, 1, b'a']);
    put(b, q + 4, &OH.to_le_bytes());
}

fn heap_header(b: &mut [u8], flags: u8, root: u64, root_rows: u16, table_width: u16) {
    put(b, 0, FRHP);
    b[9] = flags;
    put(b, 70, &1u64.to_le_bytes());
    put(b, 110, &table_width.to_le_bytes());
    put(b, 128, &8u16.to_le_bytes());
    put(b, 132, &root.to_le_bytes());
    put(b, 140, &root_rows.to_le_bytes());
}

#[test]
fn direct_root_block_without_checksum_reads_its_link() {
    let mut b = vec![0u8; 512];
    heap_header(&mut b, 0, 256, 0, 1);
    direct_block(&mut b, 256, 1);
    let mut out = Vec::new();
    dense_links(&b, &SIZES, 0, &mut out);
    assert_eq!(out, [("a".to_string(), OH as usize)]);
}

#[test]
fn undefined_indirect_entry_is_skipped() {
    let mut b = vec![0u8; 1024];
    heap_header(&mut b, 0, 256, 1, 2);
    put(&mut b, 256, FHIB);
    let entries = 256 + 4 + 1 + SIZES.offset + 1;
    put(&mut b, entries, &[0xFF; 8]);
    put(&mut b, entries + 8, &512u64.to_le_bytes());
    direct_block(&mut b, 512, 1);
    let mut out = Vec::new();
    dense_links(&b, &SIZES, 0, &mut out);
    assert_eq!(out, [("a".to_string(), OH as usize)]);
}
