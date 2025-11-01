import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from poseidon.data.dataset import read_dataset, clip_partition, shuffle_and_split
from poseidon.data.smote_knn import smote

# 환경 변수 로드
load_dotenv(verbose=True)
DATASET_RESAMPLED_PATH = os.getenv("DATASET_RESAMPLED_PATH")
if not DATASET_RESAMPLED_PATH:
    raise ValueError("DATASET_RESAMPLED_PATH 환경 변수가 설정되지 않았습니다!")


def resample_dataset(
    dataset_path, k=5, sampling_ratio=0.4831882086330935, random_state=42, output=True, output_path=None
):
    dataset_name = os.path.basename(os.path.dirname(dataset_path))

    print("=" * 45)

    # 1. 데이터 로드
    print(f"> {dataset_name} 데이터 청크 로드 중...")
    dask_df = read_dataset(dataset_path, chunksize=500000)
    print("\t- 데이터셋 청크 로드 완료\n")

    # 2. 오류값 정정 (NaN, Inf 처리, 불필요 피처 제거)
    print("> 오류값 정정 중...")
    dask_df = dask_df.replace([np.inf, -np.inf], np.nan).dropna()
    print("\t- NaN, Inf 처리 완료")
    dask_df = dask_df.drop(
        columns=["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"]
    )
    print("\t- 불필요한 피처 제거 완료\n")

    # 2-1. 컬럼병 분위수(quantile) 계산
    print("\t> 컬럼별 분위수(quantile) 계산 중...")
    quantiles = dask_df.select_dtypes(include=["float64"]).quantile(0.99).compute()
    print("\t\t- 컬럼별 분위수 계산 완료\n")

    # 3. 컬럼별 클리핑
    print("> 컬럼별 클리핑 중...")
    dask_df = dask_df.map_partitions(clip_partition, quantiles=quantiles, meta=dask_df)
    print("\t- 컬럼별 클리핑 완료\n")

    # 4. 데이터 셔플 및 X, y 분리
    print("> 데이터 셔플 및 X, y 분리 중...")
    X, y = shuffle_and_split(dask_df, random_state=random_state)
    print("\t- 데이터 셔플 및 X, y 분리 완료\n")

    # 5. 원본 'Label' 클래스 분포 출력
    print("> 원본 'Label' 클래스 분포")
    origin_y_unique, origin_y_counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(origin_y_unique, origin_y_counts):
        print(f"\t'Label' 클래스 {cls} 개수: {cnt}")
    print("\n")

    # 6. SMOTE 적용
    if len(origin_y_unique) != 2:
        print("'Label'의 클래스 수가 2개가 아닙니다! SMOTE를 적용할 수 없습니다!")
        X_resampled, y_resampled = X, y
    else:
        print("> SMOTE 적용 중...")
        try:
            # smote = SMOTE(k_neighbors=k, random_state=random_state)
            # X_resampled, y_resampled = smote.fit_resample(X, y)
            X_resampled, y_resampled = smote(
                X, y, k=k, sampling_ratio=sampling_ratio, random_state=random_state
            )
        except ValueError as e:
            print(f"처리 중 오류: '{e}' SMOTE를 스킵합니다!")
            X_resampled, y_resampled = X, y
        print("\t- SMOTE 적용 완료\n")

    # 7. 리샘플 클래스 분포 출력
    print("> 리샘플링 후 'Label' 클래스 분포")
    resampled_y_unique, resampled_y_counts = np.unique(y_resampled, return_counts=True)
    for cls, cnt in zip(resampled_y_unique, resampled_y_counts):
        print(f"\t'Label' 클래스 {cls} 개수: {cnt}")
    print("\n")

    # 8. 로컬에 저장
    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    if output:
        if output_path is None:
            output_path = DATASET_RESAMPLED_PATH
        print(f"> 로컬에 저장 중... (지정 경로: {output_path})")
        resampled_df["Label"] = y_resampled
        resampled_df.to_csv(
            os.path.join(output_path, f"{dataset_name}-smote.csv"), index=False
        )
        print("\t- 로컬에 저장 완료\n")

        print(
            f"'{dataset_name}' 데이터셋에 대한 전처리 및 SMOTE 리샘플링 후 데이터셋 저장을 완료했습니다. {f'../{dataset_name}-smote.csv'}"
        )
    else:
        print(
            f"'{dataset_name}' 데이터셋에 대한 전처리 및 SMOTE 리샘플링을 완료했습니다. 결과를 반환합니다."
        )
        return resampled_df



# 각 데이터셋에 대해 실행 (실제 데이터셋)
# orig_datasets = [
#     # "TEST-DATASET",
#     # "NF-UNSW-NB15-v3",
#     # "NF-BoT-IoT-v3",
#     # "NF-CICIDS2018-v3",
#     # "NF-ToN-IoT-v3",
# ]

# custom_datasets = [
#     "500000s-NF-custom-dataset-1761928299",
# ]
# for ds in custom_datasets:
#     resample_dataset(ds)
