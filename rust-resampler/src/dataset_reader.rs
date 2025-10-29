use crate::nids_dtype;
use dotenv::dotenv;
use polars::prelude::*;
use std::{env, path::PathBuf};

pub fn read_dataset(dataset_name: &str) -> PolarsResult<DataFrame> {
    // .env 파일에서 환경 변수 로드 (예: CSV 파일 경로)
    dotenv().ok();
    let csv_path = env::var("DATASET_DIR_PATH").unwrap_or_else(|_| ".".to_string());

    // dtypes 정의
    let dtypes = nids_dtype::dtypes();

    let mut path = PathBuf::from(csv_path);
    path.push(dataset_name);
    path.push("data");
    path.push(format!("{}.csv", dataset_name));
    let path_wrap = PlPathRef::from_local_path(&path).into_owned();

    let scheme = Schema::from_iter(dtypes.iter().map(|(k, v)| (k.into(), v.clone())));

    // LazyCsvReader를 사용하여 CSV를 lazy하게 스캔
    let lazy_df = LazyCsvReader::new(path_wrap)
        .with_has_header(true) // 헤더가 있는 경우 true로 설정
        .with_separator(b',') // 기본 구분자 설정
        .with_dtype_overwrite(Some(Arc::new(scheme.clone())))
        .finish()?;

    // LazyFrame을 collect하여 DataFrame으로 변환
    let df = lazy_df.collect()?;

    Ok(df)
}
