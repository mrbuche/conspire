use crate::hash::FxHasher;
#[cfg(feature = "std")]
use crate::hash::{FxHashMap, FxHashSet};
use core::hash::Hasher;

#[cfg(feature = "std")]
#[test]
fn map_inserts_and_reads() {
    let mut map = FxHashMap::<(usize, usize), usize>::default();
    map.insert((1, 2), 10);
    map.insert((3, 4), 20);
    assert_eq!(map.get(&(1, 2)), Some(&10));
    assert_eq!(map.get(&(3, 4)), Some(&20));
    assert_eq!(map.get(&(2, 1)), None);
    assert_eq!(map.len(), 2);
}

#[cfg(feature = "std")]
#[test]
fn set_deduplicates() {
    let mut set = FxHashSet::<usize>::default();
    set.insert(7);
    set.insert(7);
    set.insert(9);
    assert_eq!(set.len(), 2);
    assert!(set.contains(&7));
    assert!(!set.contains(&8));
}

#[test]
fn hasher_takes_every_width_and_every_length_of_bytes() {
    let hashed = |operation: &dyn Fn(&mut FxHasher)| {
        let mut hasher = FxHasher::default();
        operation(&mut hasher);
        hasher.finish()
    };
    assert_eq!(hashed(&|_| {}), 0);
    let widths = [
        hashed(&|hasher| hasher.write_u8(1)),
        hashed(&|hasher| hasher.write_u16(1)),
        hashed(&|hasher| hasher.write_u32(1)),
        hashed(&|hasher| hasher.write_u64(1)),
        hashed(&|hasher| hasher.write_usize(1)),
    ];
    assert!(widths.iter().all(|&hash| hash != 0));
    assert!(widths.windows(2).all(|pair| pair[0] == pair[1]));
    let bytes: Vec<u8> = (1..=15).collect();
    let mut seen = Vec::new();
    for length in 0..=bytes.len() {
        let hash = hashed(&|hasher| hasher.write(&bytes[..length]));
        assert!(!seen.contains(&hash), "length {length} repeats a hash");
        seen.push(hash);
    }
    assert_eq!(
        hashed(&|hasher| hasher.write(&7_u64.to_le_bytes())),
        hashed(&|hasher| hasher.write_u64(7))
    );
    assert_eq!(
        hashed(&|hasher| hasher.write(&[7])),
        hashed(&|hasher| hasher.write_u8(7))
    );
    assert_eq!(
        hashed(&|hasher| hasher.write(&bytes)),
        hashed(&|hasher| hasher.write(&bytes))
    );
    assert_ne!(
        hashed(&|hasher| hasher.write(&bytes)),
        hashed(&|hasher| hasher.write(&bytes[1..]))
    );
}
