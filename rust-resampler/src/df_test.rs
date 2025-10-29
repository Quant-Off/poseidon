use crate::dataset_reader;
use polars::prelude::*;

pub fn smote_test(dataset_name: &str) -> PolarsResult<()> {   
    let df = dataset_reader::read_dataset(dataset_name)?.lazy();

    // Label이 0인 행의 개수 계산
    let zero_counts = df.filter(col("Label").eq(lit(0))).select([col("Label").count()]).collect()?;
    println!("0 개수: {:?}", zero_counts);

    Ok(())
}
