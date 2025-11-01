import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from poseidon.util.shannon import entropy_sn

# 패킷 엔트로피 피처를 만들 때 필요한 섀넌 엔트로피가 적용되는 바이트 피처 목록
bytes_features = [
    "IN_BYTES",  # Incoming number of bytes (수신 바이트 수, 분포 계산에 사용 가능).
    "OUT_BYTES",  # Outgoing number of bytes (송신 바이트 수, 유사하게 사용).
    "LONGEST_FLOW_PKT",  # Longest packet (bytes) of the flow (최대 패킷 길이).
    "SHORTEST_FLOW_PKT",  # Shortest packet (bytes) of the flow (최소 패킷 길이).
    "MIN_IP_PKT_LEN",  # Len of the smallest flow IP packet observed (최소 IP 패킷 길이).
    "MAX_IP_PKT_LEN",  # Len of the largest flow IP packet observed (최대 IP 패킷 길이).
    "SRC_TO_DST_SECOND_BYTES",  # Src to dst Bytes/sec (초당 바이트 전송률, 속도 기반 분포).
    "DST_TO_SRC_SECOND_BYTES",  # Dst to src Bytes/sec (역방향 속도 기반 분포).
    "RETRANSMITTED_IN_BYTES",  # Number of retransmitted TCP flow bytes (src->dst) (재전송 바이트 수).
    "RETRANSMITTED_OUT_BYTES",  # Number of retransmitted TCP flow bytes (dst->src) (역방향 재전송 바이트 수).
    "NUM_PKTS_UP_TO_128_BYTES",  # Packets whose IP size <= 128 (패킷 크기 버킷, 분포 엔트로피에 최적).
    "NUM_PKTS_128_TO_256_BYTES",  # Packets whose IP size > 128 and <= 256 (유사 버킷).
    "NUM_PKTS_256_TO_512_BYTES",  # Packets whose IP size > 256 and <= 512.
    "NUM_PKTS_512_TO_1024_BYTES",  # Packets whose IP size > 512 and <= 1024.
    "NUM_PKTS_1024_TO_1514_BYTES",  # Packets whose IP size > 1024 and <= 1514.
]


def split(resampled_df):
    """
    Parameters:
    -----------
    resampled_df : pd.DataFrame
        SMOTE 리샘플링 후 저장된 데이터셋 (DataFrame)

    Returns:
    --------
    splited_X_train : pd.DataFrame
    splited_X_val : pd.DataFrame
    splited_X_test : pd.DataFrame
    splited_y_train : pd.Series
    splited_y_val : pd.Series
    splited_y_test : pd.Series
    """
    print("=" * 45)
    print("> 데이터셋 분할(split) 중...")

    splited_X = resampled_df.drop("Label", axis=1)
    splited_y = resampled_df["Label"]

    # 첫 번째 분할: 훈련(60%) vs. 임시(40%)
    splited_X_train, splited_X_temp, splited_y_train, splited_y_temp = train_test_split(
        splited_X, splited_y, test_size=0.4, stratify=splited_y, random_state=42
    )

    # 두 번째 분할: 임시를 검증(20%) vs. 테스트(20%)
    splited_X_val, splited_X_test, splited_y_val, splited_y_test = train_test_split(
        splited_X_temp,
        splited_y_temp,
        test_size=0.5,
        stratify=splited_y_temp,
        random_state=42,
    )

    return (
        splited_X_train,
        splited_X_val,
        splited_X_test,
        splited_y_train,
        splited_y_val,
        splited_y_test,
    )


def scaling(split_X_train, split_X_val, split_X_test):
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(split_X_train)
    scaled_X_val = scaler.transform(split_X_val)
    scaled_X_test = scaler.transform(split_X_test)

    return scaled_X_train, scaled_X_val, scaled_X_test


def feature_analysis(scaled_X_train, scaled_X_val, scaled_X_test, feature_names):
    print("피처 분석(DataFrame 변환) 작업 시작...")
    # 제외할 피처 목록
    exclude_features = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"]

    # 모든 피처 이름을 리스트로 구성 (제외할 피처는 제외)
    columns_list = [name for name in feature_names if name not in exclude_features]

    # DataFrame 생성 시 columns 파라미터에 리스트 전달
    to_df_X_train = pd.DataFrame(scaled_X_train, columns=columns_list)
    to_df_X_val = pd.DataFrame(scaled_X_val, columns=columns_list)
    to_df_X_test = pd.DataFrame(scaled_X_test, columns=columns_list)

    # 각 피처별 히스토그램 출력
    for column_name in columns_list:
        print(
            f"{column_name} 훈련 세트 히스토그램: {to_df_X_train[column_name].hist()}"
        )
        print(f"{column_name} 검증 세트 히스토그램: {to_df_X_val[column_name].hist()}")
        print(
            f"{column_name} 테스트 세트 히스토그램: {to_df_X_test[column_name].hist()}"
        )

    return to_df_X_train, to_df_X_val, to_df_X_test


def apply_entropy(row):
    byte_counts = np.array([row[feat] for feat in bytes_features])
    if np.sum(byte_counts) == 0:
        return 0
    probs = byte_counts / np.sum(byte_counts)
    return entropy_sn(probs)


def main():
    datasets = [
        "NF-UNSW-NB15-v3",
        "NF-BoT-IoT-v3",
        "NF-CICIDS2018-v3",
        "NF-ToN-IoT-v3",
    ]
    for dataset in datasets:
        print("=" * 100)
        print(f"{dataset} 데이터셋 로드 중...")

        (
            splited_X_train,
            splited_X_val,
            splited_X_test,
            splited_y_train,
            splited_y_val,
            splited_y_test,
        ) = split(dataset)
        print(
            f"훈련 세트: {splited_y_train.value_counts(normalize=True)}"
        )  # 훈련 세트 클래스 비율
        print(f"검증 세트: {splited_y_val.value_counts(normalize=True)}")  # 검증 세트
        print(
            f"테스트 세트: {splited_y_test.value_counts(normalize=True)}"
        )  # 테스트 세트

        print("피처 스케일링 작업 시작...")
        scaled_X_train, scaled_X_val, scaled_X_test = scaling(
            splited_X_train, splited_X_val, splited_X_test
        )

        # 정규화 결과 요약 정보 출력
        print(f"훈련 세트 정규화 완료 - Shape: {scaled_X_train.shape}")
        print(
            f"  평균: {np.mean(scaled_X_train):.6f}, 표준편차: {np.std(scaled_X_train):.6f}"
        )
        print(
            f"  최소값: {np.min(scaled_X_train):.6f}, 최대값: {np.max(scaled_X_train):.6f}"
        )

        print(f"검증 세트 정규화 완료 - Shape: {scaled_X_val.shape}")
        print(
            f"  평균: {np.mean(scaled_X_val):.6f}, 표준편차: {np.std(scaled_X_val):.6f}"
        )
        print(
            f"  최소값: {np.min(scaled_X_val):.6f}, 최대값: {np.max(scaled_X_val):.6f}"
        )

        print(f"테스트 세트 정규화 완료 - Shape: {scaled_X_test.shape}")
        print(
            f"  평균: {np.mean(scaled_X_test):.6f}, 표준편차: {np.std(scaled_X_test):.6f}"
        )
        print(
            f"  최소값: {np.min(scaled_X_test):.6f}, 최대값: {np.max(scaled_X_test):.6f}"
        )

        # 원본 DataFrame의 컬럼 이름을 feature_analysis 함수에 전달
        to_df_X_train, to_df_X_val, to_df_X_test = feature_analysis(
            scaled_X_train,
            scaled_X_val,
            scaled_X_test,
            splited_X_train.columns.tolist(),
        )

        print("패킷 엔트로피 피처 적용 중...")
        to_df_X_train["packet_entropy"] = [
            apply_entropy(row)
            for _, row in tqdm(
                to_df_X_train.iterrows(),
                total=len(to_df_X_train),
                desc="훈련 세트 엔트로피 적용",
            )
        ]
        to_df_X_val["packet_entropy"] = [
            apply_entropy(row)
            for _, row in tqdm(
                to_df_X_val.iterrows(),
                total=len(to_df_X_val),
                desc="검증 세트 엔트로피 적용",
            )
        ]
        to_df_X_test["packet_entropy"] = [
            apply_entropy(row)
            for _, row in tqdm(
                to_df_X_test.iterrows(),
                total=len(to_df_X_test),
                desc="테스트 세트 엔트로피 적용",
            )
        ]
        # 엔트로피 피처 통계
        print("패킷 엔트로피 피처 통계 요약")
        for name, df in [
            ("훈련 세트", to_df_X_train),
            ("검증 세트", to_df_X_val),
            ("테스트 세트", to_df_X_test),
        ]:
            entropy_col = df["packet_entropy"]
            print(f"\n[{name}]")
            print(f"  데이터 수: {len(entropy_col):,}개")
            print(f"  평균: {entropy_col.mean():.6f}")
            print(f"  표준편차: {entropy_col.std():.6f}")
            print(f"  최소값: {entropy_col.min():.6f}")
            print(f"  25% 분위수: {entropy_col.quantile(0.25):.6f}")
            print(f"  중간값: {entropy_col.median():.6f}")
            print(f"  75% 분위수: {entropy_col.quantile(0.75):.6f}")
            print(f"  최대값: {entropy_col.max():.6f}")
            zero_count = (entropy_col == 0).sum()
            print(
                f"  0값 개수: {zero_count:,}개 ({zero_count/len(entropy_col)*100:.2f}%)"
            )
            print(f"  샘플 값 (처음 5개): {entropy_col.head().tolist()}")
            print(f"  샘플 값 (마지막 5개): {entropy_col.tail().tolist()}")
        print("\n\n\n")
