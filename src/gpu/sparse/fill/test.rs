use super::fill;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::{collections::HashSet, sync::Mutex, thread::current};

fn source(i: usize, j: usize) -> f64 {
    1.0 / (1.0 + i as f64 * 0.37 + j as f64 * 0.11) - 0.4
}

fn entries(length: usize) -> (Vec<(usize, usize)>, Vec<u32>) {
    (
        (0..length).map(|k| (k / 7, (3 * k) % 11)).collect(),
        (0..length)
            .map(|k| 1 + (k % 5 == 0) as u32 * (k % 4) as u32)
            .collect(),
    )
}

fn computed(length: usize, threads: usize, minimum: usize) -> Vec<f64> {
    let (positions, counts) = entries(length);
    let mut values = vec![f64::NAN; length];
    fill(&positions, &counts, &source, &mut values, threads, minimum);
    values
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn a_value_is_the_source_added_as_many_times_as_its_count() {
    let (positions, counts) = entries(100);
    let values = computed(100, 1, 1);
    for ((&(i, j), &count), value) in positions.iter().zip(&counts).zip(&values) {
        let mut sum = 0.0;
        (0..count).for_each(|_| sum += source(i, j));
        assert_eq!(value.to_bits(), sum.to_bits());
    }
    assert!(
        counts.iter().any(|&count| count > 1),
        "duplicates are tested"
    );
}

#[test]
fn the_threads_give_the_same_bits_as_one_thread() {
    for length in [1, 2, 99, 1000, 4097] {
        let serial = bits(&computed(length, 1, 1));
        for threads in [2, 3, 7, 64] {
            assert_eq!(
                bits(&computed(length, threads, 1)),
                serial,
                "{length} entries on {threads} threads"
            );
        }
    }
}

#[test]
fn nothing_to_fill_is_nothing() {
    assert!(computed(0, 4, 1).is_empty());
}

#[test]
fn there_are_no_more_threads_than_there_are_entries_for() {
    let threads = Mutex::new(HashSet::new());
    let counted = |i: usize, j: usize| {
        threads.lock().unwrap().insert(current().id());
        source(i, j)
    };
    let (positions, counts) = entries(1000);
    let mut values = vec![0.0; 1000];
    fill(&positions, &counts, &counted, &mut values, 8, 400);
    assert_eq!(threads.lock().unwrap().len(), 3);
    threads.lock().unwrap().clear();
    fill(&positions, &counts, &counted, &mut values, 8, 5000);
    assert_eq!(threads.lock().unwrap().len(), 1);
}

#[test]
fn every_position_is_asked_for_once() {
    let calls = AtomicUsize::new(0);
    let counted = |i: usize, j: usize| {
        calls.fetch_add(1, Ordering::Relaxed);
        source(i, j)
    };
    let (positions, counts) = entries(5000);
    let mut values = vec![0.0; 5000];
    fill(&positions, &counts, &counted, &mut values, 4, 100);
    assert_eq!(calls.load(Ordering::Relaxed), 5000);
}
