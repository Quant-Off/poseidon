use crate::knn;
use crate::nids_dtype;
use ndarray::{Array2, Axis};
use polars::prelude::*;
use rand::Rng;
use std::collections::HashMap;
use std::{env, fs::File, path::PathBuf}; // knn.rs에서 knn 함수 임포트 가정

pub fn smote(
    dataset_name: &str,
    output_path: &str,
    k: usize,
    oversample_ratio: f64,
) -> PolarsResult<()> {
    // 환경 변수 로드 (dataset_reader.rs 재사용)
    dotenv::dotenv().ok();
    let csv_path = env::var("DATASET_DIR_PATH").unwrap_or_else(|_| ".".to_string());

    // dtypes 정의 (dataset_reader.rs에서 복사)
    let mut dtypes: HashMap<String, DataType> = nids_dtype::dtypes();

    // 데이터셋 경로 구성
    let mut path = PathBuf::from(csv_path);
    path.push(dataset_name);
    path.push("data");
    path.push(format!("{}.csv", dataset_name));
    let path_wrap = PlPathRef::from_local_path(&path).into_owned();

    let scheme = Schema::from_iter(dtypes.iter().map(|(k, v)| (k.into(), v.clone())));

    // LazyCsvReader로 데이터 읽기
    let lazy_df = LazyCsvReader::new(path_wrap)
        .with_has_header(true)
        .with_separator(b',')
        .with_dtype_overwrite(Some(Arc::new(scheme.clone())))
        .finish()?;

    let mut df = lazy_df.collect()?;

    // 클래스 불균형 확인: Label 컬럼 기준
    let label_col = df.column("Label")?;
    let label_group = df.group_by(["Label"])?; // TODO: 문제의 부분
    let zero_counts = label_group.select(["0"]).count();
    let class_counts = df.group_by(["Label"])?.count().iter().collect();


    // let class_counts = df.group_by(["Label"])?.agg([col("Label").count().alias("counts")]).collect()?;
    let minority_class = class_counts.get_column("Label")?.i64()?.get(1).unwrap_or(1); // 소수 클래스 가정 (샘플 수가 적은 클래스 값)
    let majority_count = class_counts
        .get_column("counts")?
        .u32()?
        .get(0)
        .unwrap_or(0) as usize;
    let minority_count = class_counts
        .get_column("counts")?
        .u32()?
        .get(1)
        .unwrap_or(0) as usize;

    // 오버샘플링 수 계산: 소수 클래스를 다수 클래스 크기만큼 증가
    let num_synthetic =
        ((majority_count as f64 * oversample_ratio) as usize).saturating_sub(minority_count);

    if num_synthetic == 0 {
        // 불균형 없음: 원본 저장 후 종료
        df.to_csv(output_path, b',', true)?;
        return Ok(());
    }

    // 특징 컬럼 선택 (Label과 Attack 제외)
    let feature_cols: Vec<&str> = df
        .get_column_names()
        .iter()
        .filter(|&col| *col != "Label" && *col != "Attack")
        .map(|s| *s)
        .collect();
    let features_df = df.select(&feature_cols)?;

    // ndarray로 변환
    let features: Array2<f64> = Array2::from_shape_vec(
        (features_df.height(), features_df.width()),
        features_df
            .to_ndarray::<Float64Type>()?
            .iter()
            .cloned()
            .collect(),
    )?;

    // 소수 클래스 인덱스 추출
    let minority_indices: Vec<usize> = label_col
        .i64()?
        .into_iter()
        .enumerate()
        .filter_map(|(i, val)| {
            if val == Some(minority_class) {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    // 소수 클래스 특징 추출
    let minority_features = features.select(Axis(0), &minority_indices);

    // KNN 계산 (knn.rs 사용)
    let neighbors = knn::knn(&minority_features, k);

    // 합성 샘플 생성
    let mut synthetic_features: Vec<Vec<f64>> = Vec::with_capacity(num_synthetic);
    let mut synthetic_labels: Vec<i64> = Vec::with_capacity(num_synthetic);
    let mut synthetic_attacks: Vec<String> = Vec::with_capacity(num_synthetic);

    let mut rng = rand::thread_rng();

    for _ in 0..num_synthetic {
        let idx = rng.gen_range(0..minority_indices.len());
        let original_idx = minority_indices[idx];
        let original = features.row(original_idx);

        let neigh_idx = rng.gen_range(0..neighbors[idx].len());
        let neighbor_global_idx = minority_indices[neighbors[idx][neigh_idx]];
        let neighbor = features.row(neighbor_global_idx);

        let lambda = rng.gen_range(0.0..1.0);
        let synthetic = original.to_owned() + lambda * (neighbor - original);

        synthetic_features.push(synthetic.to_vec());
        synthetic_labels.push(minority_class);
        synthetic_attacks.push(
            df.column("Attack")?
                .str()?
                .get(original_idx)
                .unwrap_or("")
                .to_string(),
        );
    }

    // 합성 DataFrame 생성
    let mut synthetic_df = DataFrame::new(
        feature_cols
            .iter()
            .enumerate()
            .map(|(i, col)| {
                let data: Vec<f64> = synthetic_features.iter().map(|row| row[i]).collect();
                Series::new(col, data)
            })
            .collect(),
    )?;
    synthetic_df.with_column(Series::new("Label", synthetic_labels))?;
    synthetic_df.with_column(Series::new("Attack", synthetic_attacks))?;

    // 원본과 합성 결합
    let resampled_df = df.vstack(&synthetic_df)?;

    // 저장
    let file = File::create(output_path)?;
    CsvWriter::new(file).finish(&mut resampled_df.clone())?;

    Ok(())
}
