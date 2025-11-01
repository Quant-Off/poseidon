import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from poseidon.util.shannon import entropy_sn
from poseidon.util.timing_variance import timing_variance

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

# 타이밍 변동  피처를 만들 때 필요한 피처 목록
timing_variance_features = [
    "FLOW_DURATION_MILLISECONDS",  # 전체 플로우 지속 시간 (기준 시간 스케일 제공)
    "DURATION_IN",  # 클라이언트-서버 스트림 지속 시간 (인바운드)
    "DURATION_OUT",  # 서버-클라이언트 스트림 지속 시간 (아웃바운드)
    "SRC_TO_DST_IAT_MIN",  # 소스-대상 IAT 통계 (최소/최대/평균/표준편차)
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",  # 대상-소스 IAT 통계
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
    "FLOW_START_MILLISECONDS",  # 플로우 시작/종료 타임스탬프 (차이 계산 가능)
    "FLOW_END_MILLISECONDS",
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
            f"  {column_name} 훈련 세트 히스토그램: {to_df_X_train[column_name].hist()}"
        )
        print(
            f"  {column_name} 검증 세트 히스토그램: {to_df_X_val[column_name].hist()}"
        )
        print(
            f"  {column_name} 테스트 세트 히스토그램: {to_df_X_test[column_name].hist()}"
        )

    return to_df_X_train, to_df_X_val, to_df_X_test


def apply_entropy(row):
    byte_counts = np.array([row[feat] for feat in bytes_features])
    if np.sum(byte_counts) == 0:
        return 0.0
    probs = byte_counts / np.sum(byte_counts)
    entropy_value = entropy_sn(probs)
    # JAX 배열이나 numpy 배열을 Python float로 변환
    try:
        # JAX 배열인 경우
        if hasattr(entropy_value, "item"):
            return float(entropy_value.item())
        # numpy 배열인 경우
        elif hasattr(entropy_value, "__len__") and len(entropy_value) == 1:
            return float(entropy_value[0])
        # 이미 스칼라인 경우
        else:
            return float(entropy_value)
    except (TypeError, ValueError):
        # 모든 변환이 실패하면 numpy를 통해 변환
        return float(np.asarray(entropy_value).item())


def apply_timing_variance(row):
    """
    각 행에서 타임스탬프 배열을 생성하여 timing_variance 함수에 전달합니다.
    FLOW_START_MILLISECONDS와 FLOW_END_MILLISECONDS를 기준으로 
    비감소 순서의 타임스탬프 배열을 생성합니다.
    """
    # 기본 타임스탬프 확인
    if "FLOW_START_MILLISECONDS" not in row.index or "FLOW_END_MILLISECONDS" not in row.index:
        return 0.0
    
    start_time = float(row["FLOW_START_MILLISECONDS"])
    end_time = float(row["FLOW_END_MILLISECONDS"])
    
    # FLOW_END가 FLOW_START보다 작거나 같은 경우 처리
    if end_time <= start_time:
        end_time = start_time + 1.0
    
    # 플로우 지속 시간 기반으로 타임스탬프 시퀀스 생성
    # DURATION_IN, DURATION_OUT가 있으면 이를 활용하여 더 정확한 타임스탬프 생성
    duration_ms = end_time - start_time
    
    if duration_ms <= 0:
        return 0.0
    
    # 패킷 수 추정 (IN_PKTS + OUT_PKTS 또는 기본값)
    total_packets = 1
    if "IN_PKTS" in row.index and "OUT_PKTS" in row.index:
        total_packets = max(1, int(row["IN_PKTS"]) + int(row["OUT_PKTS"]))
    
    # 최소 2개, 최대 100개의 타임스탬프 생성 (너무 많으면 성능 저하)
    num_timestamps = min(max(2, total_packets // 10), 100)
    
    # 비감소 순서의 타임스탬프 배열 생성
    # 균등 간격 또는 지수 분포를 사용하여 실제 패킷 타이밍 시뮬레이션
    if num_timestamps == 2:
        # 최소 2개: 시작과 끝만
        time_values = np.array([start_time, end_time])
    else:
        # 시작과 끝 사이에 중간 타임스탬프 생성
        # IAT 통계가 있으면 이를 활용, 없으면 균등 분포 사용
        if "SRC_TO_DST_IAT_AVG" in row.index and row["SRC_TO_DST_IAT_AVG"] > 0:
            # IAT 평균값을 기반으로 간격 생성
            avg_iat = float(row["SRC_TO_DST_IAT_AVG"])
            intervals = np.random.exponential(scale=avg_iat, size=num_timestamps-1)
        else:
            # 균등 간격
            intervals = np.full(num_timestamps - 1, duration_ms / (num_timestamps - 1))
        
        # 비감소 순서 보장
        intervals = np.maximum(intervals, 0.1)  # 최소 0.1ms 간격
        cumulative = np.cumsum(intervals)
        # 전체 지속 시간에 맞게 스케일 조정
        if cumulative[-1] > 0:
            cumulative = cumulative * (duration_ms / cumulative[-1])
        
        # 시작 타임스탬프에 간격 누적
        time_values = np.concatenate([[start_time], start_time + cumulative])
        time_values = np.clip(time_values, start_time, end_time)
        time_values[-1] = end_time  # 마지막은 정확히 end_time
    
    # 비감소 순서 확인 및 정렬
    time_values = np.sort(time_values)
    
    if len(time_values) < 2:
        return 0.0
    
    # timing_variance 함수 호출
    variance_value = timing_variance(time_values)
    
    # JAX 배열이나 numpy 배열을 Python float로 변환
    try:
        if hasattr(variance_value, "item"):
            return float(variance_value.item())
        elif hasattr(variance_value, "__len__") and len(variance_value) == 1:
            return float(variance_value[0])
        else:
            return float(variance_value)
    except (TypeError, ValueError):
        return float(np.asarray(variance_value).item())
