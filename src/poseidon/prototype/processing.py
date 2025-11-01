import os
import numpy as np
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


def all_process(dataset, is_custom=False):
    print("=" * 100)
    print(f"> {dataset} SMOTE 오버샘플링 적용 중...")
    resampled_df = oversampling(dataset, is_custom=is_custom)

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

    # TODO: 피처 엔지니어링은 엔트로피 뿐 아니라 타이밍 변동, 양자 시뮬레이션 피처가 추가되어야 함.

    print(f"> {dataset} 리샘플링 및 피처 엔지니어링 적용 완료!\n\n\n")


if __name__ == "__main__":
    is_test = True
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
            all_process(f"{d}.csv")
