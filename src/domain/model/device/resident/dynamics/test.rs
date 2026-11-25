use super::table_rows;

#[test]
fn a_table_holds_as_many_rows_as_fit_in_a_few_million_values_and_a_thousand_at_most() {
    assert_eq!(table_rows(0), 1024);
    assert_eq!(table_rows(300), 1024);
    assert_eq!(table_rows(100_000), (1 << 22) / 100_000);
    assert_eq!(table_rows(10_000_000), 1);
}
