import os
from dotenv import load_dotenv

import pandas as pd
import dask.dataframe as dd
from tqdm import tqdm

from poseidon.processpiece.load_dask_dataframe import switch_to_dask, switch_to_pandas
from poseidon.processpiece.engineering_split import DatasetSplit
from poseidon.processpiece.engineering_scaling import DatasetScaling
from poseidon.processpiece.feature_calculate import (
    adding_feature_shannon_entropy,
    adding_feature_timing_variance,
    adding_feature_quantum_noise_simulation,
)

load_dotenv(verbose=True)
DATASETS_RESAMPLED_PATH = os.getenv("DATASETS_RESAMPLED_PATH")


def engineering_test():
    # 이 테스트에 사용되는 데이터셋은 이미 SMOTE 오버샘플링이 완료되어 있음
    df = pd.read_csv(
        f"{DATASETS_RESAMPLED_PATH}/10000s-NF-custom-dataset-1762341664-smote.csv"
    )
    df = switch_to_dask(df)
    print(f"로드 후 타입: {type(df)}")
    print("=" * 100)

    split_instance = DatasetSplit(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_instance.split(npartitions=1)
    print(f"X_train 타입: {type(X_train)}")
    print(f"X_val 타입: {type(X_val)}")
    print(f"X_test 타입: {type(X_test)}")
    print(f"y_train 타입: {type(y_train)}")
    print(f"y_val 타입: {type(y_val)}")
    print(f"y_test 타입: {type(y_test)}")
    print("=" * 100)

    scaling_instance = DatasetScaling()
    X_train, X_val, X_test = scaling_instance.scale(X_train, X_val, X_test)
    print(f"스케일링 후 X_train 타입: {type(X_train)}")
    print(f"스케일링 후 X_val 타입: {type(X_val)}")
    print(f"스케일링 후 X_test 타입: {type(X_test)}")
    print("=" * 100)

    X_train, X_val, X_test = adding_feature_shannon_entropy(X_train, X_val, X_test)
    print(f"섀넌 엔트로피 계산 후 X_train 타입: {type(X_train)}")
    print(f"섀넌 엔트로피 계산 후 X_val 타입: {type(X_val)}")
    print(f"섀넌 엔트로피 계산 후 X_test 타입: {type(X_test)}")
    print("=" * 100)

    X_train, X_val, X_test = adding_feature_timing_variance(X_train, X_val, X_test)
    print(f"타이밍 변동 계산 후 X_train 타입: {type(X_train)}")
    print(f"타이밍 변동 계산 후 X_val 타입: {type(X_val)}")
    print(f"타이밍 변동 계산 후 X_test 타입: {type(X_test)}")
    print("=" * 100)

    X_train, X_val, X_test = adding_feature_quantum_noise_simulation(
        X_train, X_val, X_test
    )
    print(f"양자 노이즈 시뮬레이션 계산 후 X_train 타입: {type(X_train)}")
    print(f"양자 노이즈 시뮬레이션 계산 후 X_val 타입: {type(X_val)}")
    print(f"양자 노이즈 시뮬레이션 계산 후 X_test 타입: {type(X_test)}")
    print("=" * 100)

    print("저장 작업 시작")
    # Dask DataFrame을 pandas로 변환 시 진행률 표시
    if isinstance(X_train, dd.DataFrame):
        npartitions = X_train.npartitions
        with tqdm(total=npartitions, desc="Pandas 변환 중", unit="파티션") as pbar:
            # 각 파티션을 개별적으로 compute하여 진행률 표시
            partitions = []
            for i in range(npartitions):
                partition_df = X_train.get_partition(i).compute()
                partitions.append(partition_df)
                pbar.update(1)
            # 모든 파티션을 결합
            X_train = pd.concat(partitions, ignore_index=True)
    else:
        X_train = switch_to_pandas(X_train)
    # 청크 단위로 저장하여 진행률 표시
    chunk_size = 10000  # 청크 크기 (행 수)
    total_rows = len(X_train)
    num_chunks = (total_rows + chunk_size - 1) // chunk_size  # 올림 계산
    with tqdm(total=num_chunks, desc="CSV 저장 중", unit="청크", ncols=100) as pbar:
        output_path = f"{DATASETS_RESAMPLED_PATH}/10000s-NF-custom-dataset-1762341664-smote-X-train-quantum_noise_simulation.csv"

        # 첫 번째 청크는 헤더와 함께 저장
        X_train.iloc[:chunk_size].to_csv(
            output_path,
            mode="w",
            index=False,
        )
        pbar.update(1)

        # 나머지 청크들을 추가 모드로 저장
        for i in range(1, num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_rows)
            X_train.iloc[start_idx:end_idx].to_csv(
                output_path,
                mode="a",
                header=False,
                index=False,
            )
            pbar.update(1)


if __name__ == "__main__":
    engineering_test()

# OK
