import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from poseidon.prototype import resampled_split, dataset_resampler

# 환경 변수 로드
load_dotenv(verbose=True)
DATASETS_RESAMPLED_PATH = os.getenv("DATASETS_RESAMPLED_PATH")
DATASETS_CUSTOM_PATH = os.getenv("DATASETS_CUSTOM_PATH")
if not DATASETS_RESAMPLED_PATH or not DATASETS_CUSTOM_PATH:
    raise ValueError(
        "DATASETS_RESAMPLED_PATH 또는 DATASETS_CUSTOM_PATH 환경 변수가 설정되지 않았습니다!"
    )


def oversampling(dataset_name, is_custom=False):
    if is_custom:
        dataset_path = os.path.join(DATASETS_CUSTOM_PATH, dataset_name)
    else:
        dataset_path = os.path.join(DATASETS_RESAMPLED_PATH, dataset_name)

    return dataset_resampler.resample_dataset(dataset_path, output=False)


def cal_shannon_entropy(to_df_X_train, to_df_X_val, to_df_X_test):
    to_df_X_train["packet_entropy"] = [
        resampled_split.apply_entropy(row)
        for _, row in tqdm(
            to_df_X_train.iterrows(),
            total=len(to_df_X_train),
            desc="훈련 세트 엔트로피 적용",
        )
    ]
    to_df_X_val["packet_entropy"] = [
        resampled_split.apply_entropy(row)
        for _, row in tqdm(
            to_df_X_val.iterrows(),
            total=len(to_df_X_val),
            desc="검증 세트 엔트로피 적용",
        )
    ]
    to_df_X_test["packet_entropy"] = [
        resampled_split.apply_entropy(row)
        for _, row in tqdm(
            to_df_X_test.iterrows(),
            total=len(to_df_X_test),
            desc="테스트 세트 엔트로피 적용",
        )
    ]

    return to_df_X_train, to_df_X_val, to_df_X_test


def cal_timing_variance(to_df_X_train, to_df_X_val, to_df_X_test):
    to_df_X_train["timing_variance"] = [
        resampled_split.apply_timing_variance(row)
        for _, row in tqdm(
            to_df_X_train.iterrows(),
            total=len(to_df_X_train),
            desc="훈련 세트 타이밍 변동 적용",
        )
    ]
    to_df_X_val["timing_variance"] = [
        resampled_split.apply_timing_variance(row)
        for _, row in tqdm(
            to_df_X_val.iterrows(),
            total=len(to_df_X_val),
            desc="검증 세트 타이밍 변동 적용",
        )
    ]
    to_df_X_test["timing_variance"] = [
        resampled_split.apply_timing_variance(row)
        for _, row in tqdm(
            to_df_X_test.iterrows(),
            total=len(to_df_X_test),
            desc="테스트 세트 타이밍 변동 적용",
        )
    ]

    return to_df_X_train, to_df_X_val, to_df_X_test


def load_pandas_dataframe(dataset_path):
    # 청크 단위로 읽으면서 진행률 표시
    chunk_size = 100000  # 청크 크기 (행 수)
    chunks = []
    
    # 전체 행 수 추정 (진행률 계산용)
    with open(dataset_path, 'r', encoding='utf-8') as f:
        # 첫 번째 줄 (헤더) 건너뛰기
        f.readline()
        # 나머지 줄 수 추정
        estimated_rows = sum(1 for _ in f)
    
    # tqdm 진행바를 사용하여 청크 단위로 CSV 파일 읽기
    with tqdm(total=estimated_rows, desc="CSV 파일 로드", unit="행", unit_scale=True) as pbar:
        for chunk in pd.read_csv(dataset_path, chunksize=chunk_size, low_memory=False):
            chunks.append(chunk)
            pbar.update(len(chunk))
    
    # 모든 청크 결합 및 반환
    return pd.concat(chunks, ignore_index=True)


def all_process(dataset, is_custom=False, is_smote=False):
    print("=" * 100)
    if is_smote:
        print(f"> {dataset} SMOTE 오버샘플링 적용 중...")
        resampled_df = oversampling(dataset, is_custom=is_custom)
    else:
        print(f"> {dataset} 데이터셋 로드 중 (pandas)...")
        dataset_path = os.path.join(DATASETS_RESAMPLED_PATH, f"{dataset}")
        resampled_df = load_pandas_dataframe(dataset_path)
        print(f"  로드 완료: {len(resampled_df):,}행, {len(resampled_df.columns)}열")

    print(f"> {dataset} 데이터셋 분할 중...")
    (
        splited_X_train,
        splited_X_val,
        splited_X_test,
        splited_y_train,
        splited_y_val,
        splited_y_test,
    ) = resampled_split.split(resampled_df)
    print(
        f"훈련 세트: {splited_y_train.value_counts(normalize=True)}"
    )  # 훈련 세트 클래스 비율
    print(f"검증 세트: {splited_y_val.value_counts(normalize=True)}")  # 검증 세트
    print(f"테스트 세트: {splited_y_test.value_counts(normalize=True)}")  # 테스트 세트

    print(f"> {dataset} 데이터셋 스케일링 중...")
    scaled_X_train, scaled_X_val, scaled_X_test = resampled_split.scaling(
        splited_X_train, splited_X_val, splited_X_test
    )
    print(f"  훈련 세트 정규화 완료 - Shape: {scaled_X_train.shape}")
    print(
        f"    평균: {np.mean(scaled_X_train):.6f}, 표준편차: {np.std(scaled_X_train):.6f}"
    )
    print(
        f"    최소값: {np.min(scaled_X_train):.6f}, 최대값: {np.max(scaled_X_train):.6f}"
    )

    print(f"  검증 세트 정규화 완료 - Shape: {scaled_X_val.shape}")
    print(
        f"    평균: {np.mean(scaled_X_val):.6f}, 표준편차: {np.std(scaled_X_val):.6f}"
    )
    print(f"    최소값: {np.min(scaled_X_val):.6f}, 최대값: {np.max(scaled_X_val):.6f}")

    print(f"  테스트 세트 정규화 완료 - Shape: {scaled_X_test.shape}")
    print(
        f"    평균: {np.mean(scaled_X_test):.6f}, 표준편차: {np.std(scaled_X_test):.6f}"
    )
    print(
        f"    최소값: {np.min(scaled_X_test):.6f}, 최대값: {np.max(scaled_X_test):.6f}"
    )

    print(f"> {dataset} 데이터셋 피처 분석 중...")
    to_df_X_train, to_df_X_val, to_df_X_test = resampled_split.feature_analysis(
        scaled_X_train,
        scaled_X_val,
        scaled_X_test,
        splited_X_train.columns.tolist(),
    )

    print(f"> {dataset} 데이터셋 엔트로피 적용 중...")
    to_df_X_train, to_df_X_val, to_df_X_test = cal_shannon_entropy(
        to_df_X_train, to_df_X_val, to_df_X_test
    )
    print(f"> {dataset} 리샘플링 및 엔트로피 피처 적용 완료")
    for name, df in [
        ("훈련 세트", to_df_X_train),
        ("검증 세트", to_df_X_val),
        ("테스트 세트", to_df_X_test),
    ]:
        entropy_col = df["packet_entropy"]
        print(f"  섀넌 엔트로피 피처 통계 요약: '{name}'")
        print(f"\t데이터 수: {len(entropy_col):,}개")
        print(f"\t평균: {entropy_col.mean():.12f}")
        print(f"\t최소값: {entropy_col.min():.12f}")
        print(f"\t중간값: {entropy_col.median():.12f}")
        print(f"\t최대값: {entropy_col.max():.12f}")
        zero_count = (entropy_col == 0).sum()
        print(f"\t0값 개수: {zero_count:,}개 ({zero_count/len(entropy_col)*100:.2f}%)")
        print(f"\t샘플 값 (처음 5개): {entropy_col.head().tolist()}")
        print(f"\t샘플 값 (마지막 5개): {entropy_col.tail().tolist()}")

    print(f"> {dataset} 타이밍 변동 계산 중...")
    to_df_X_train, to_df_X_val, to_df_X_test = cal_timing_variance(
        to_df_X_train, to_df_X_val, to_df_X_test
    )
    print(f"> {dataset} 리샘플링 및 타이밍 변동 피처 적용 완료")
    for name, df in [
        ("훈련 세트", to_df_X_train),
        ("검증 세트", to_df_X_val),
        ("테스트 세트", to_df_X_test),
    ]:
        timing_variance_col = df["timing_variance"]
        print(f"  타이밍 변동 피처 통계 요약: '{name}'")
        print(f"\t데이터 수: {len(timing_variance_col):,}개")
        print(f"\t평균: {timing_variance_col.mean():.12f}")
        print(f"\t최소값: {timing_variance_col.min():.12f}")
        print(f"\t중간값: {timing_variance_col.median():.12f}")
        print(f"\t최대값: {timing_variance_col.max():.12f}")
        zero_count = (timing_variance_col == 0).sum()
        print(
            f"\t0값 개수: {zero_count:,}개 ({zero_count/len(timing_variance_col)*100:.2f}%)"
        )
        print(f"\t샘플 값 (처음 5개): {timing_variance_col.head().tolist()}")
        print(f"\t샘플 값 (마지막 5개): {timing_variance_col.tail().tolist()}")


if __name__ == "__main__":
    is_test = False
    is_smote = True
    if is_test:
        all_process("500000s-NF-custom-dataset-1761928299.csv", is_custom=is_test)
    else:
        real_datasets = [
            "NF-BoT-IoT-v3",
            "NF-CICIDS2018-v3",
            "NF-ToN-IoT-v3",
            "NF-UNSW-NB15-v3",
        ]
        for d in real_datasets:
            if is_smote:
                all_process(f"{d}-smote.csv")
            else:
                all_process(f"{d}.csv")
