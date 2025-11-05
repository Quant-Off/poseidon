import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import qutip as qt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from poseidon.simulations.noise_modeling import BitFlipSimulation, PhaseFlipSimulation
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

quantum_noise_simulation_features = [
    "IN_BYTES",  # 바이트 분포의 불확실성을 초기 상태 밀도 행렬 rho 로 모델링 (예: 바이트 값을 정규화하여 큐비트 상태 벡터로 변환)
    "OUT_BYTES",
    "IN_PKTS",  # 패킷 수를 기반으로 노이즈 확률 p 추정
    "OUT_PKTS",
    "RETRANSMITTED_IN_BYTES",  # 재전송 관련 피처로 에러율을 나타내어 p 값 직접 도출
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_IN_PKTS",  # 재전송 패킷 수 – phase-flip 시뮬레이션의 초기 상태나 p 값으로 활용
    "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_SECOND_BYTES",  # 초당 바이트 전송 속도 – 노이즈 채널의 시간적 변동을 모델링
    "DST_TO_SRC_SECOND_BYTES",
    "SRC_TO_DST_IAT_MIN",  # IAT 통계로 타이밍 변동을 노이즈 채널 입력으로 사용 (예: 표준편차 sigma를 p로 스케일링).
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
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
    resampled_df = resampled_df.compute()
    splited_X = resampled_df.drop("Label", axis=1)
    splited_y = resampled_df["Label"]

    # 첫 번째 분할: 훈련(60%) vs. 임시(40%)
    splited_X_train_pd, splited_X_temp_pd, splited_y_train_pd, splited_y_temp_pd = train_test_split(
        splited_X, splited_y, test_size=0.4, stratify=splited_y, random_state=42
    )

    # 두 번째 분할: 임시를 검증(20%) vs. 테스트(20%)
    splited_X_val_pd, splited_X_test_pd, splited_y_val_pd, splited_y_test_pd = train_test_split(
        splited_X_temp_pd,
        splited_y_temp_pd,
        test_size=0.5,
        stratify=splited_y_temp_pd,
        random_state=42,
    )

    # Dask 변환
    splited_X_train = dd.from_pandas(splited_X_train_pd, npartitions=20)
    splited_X_val = dd.from_pandas(splited_X_val_pd, npartitions=20)
    splited_X_test = dd.from_pandas(splited_X_test_pd, npartitions=20)
    splited_y_train = dd.from_pandas(splited_y_train_pd, npartitions=20)
    splited_y_val = dd.from_pandas(splited_y_val_pd, npartitions=20)
    splited_y_test = dd.from_pandas(splited_y_test_pd, npartitions=20)

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
    values = []
    for feat in bytes_features:
        try:
            val = row[feat]  # Dask 호환: row.get 대신 직접 접근
            if not np.isnan(val):
                values.append(float(val))
        except KeyError:
            val = 0.0
            values.append(val)
    if not values:
        return 0.0
    values_array = da.from_array(np.array(values), chunks='auto')  # Dask 배열로 변환하여 병렬 처리 지원
    return entropy_sn(values_array)


def apply_timing_variance(row):
    """
    각 행에서 타임스탬프 배열을 생성하여 timing_variance 함수에 전달합니다.
    FLOW_START_MILLISECONDS와 FLOW_END_MILLISECONDS를 기준으로 
    비감소 순서의 타임스탬프 배열을 생성합니다.
    """
    timespamps = []
    for feat in timing_variance_features:
        try:
            val = row[feat]
            if not np.isnan(val):
                timespamps.append(float(val))
        except KeyError:
            continue
    if len(timespamps) < 2:
        return 0.0
    timespamps_array = da.from_array(np.array(sorted(timespamps)), chunks='auto')
    return timing_variance(timespamps_array, normalize=True)


def apply_quantum_noise_simulation(row):
    """
    이 함수는 각 행에서 quantum_noise_simulation_features를 기반으로 초기 상태 밀도 행렬을 생성하고,
    노이즈 확률 p를 추정한 후, 비트-플립과 페이즈-플립 시뮬레이션을 수행하여 폰 노이만 엔트로피를 계산합니다.
    """
    # 1. 초기 상태 밀도 행렬 생성: 바이트 피처 값을 기반으로 큐비트 상태 벡터 생성
    # IN_BYTES와 OUT_BYTES를 정규화하여 큐비트 상태의 파라미터로 사용
    in_bytes = max(0.0, float(row.get('IN_BYTES', 0)))
    out_bytes = max(0.0, float(row.get('OUT_BYTES', 0)))
    total_bytes = in_bytes + out_bytes

    if total_bytes > 0:
        # 바이트 비율을 기반으로 큐비트 상태 생성
        # |psi> = sqrt(in_bytes/total_bytes)|0> + sqrt(out_bytes/total_bytes)|1>
        alpha = np.sqrt(in_bytes / total_bytes)
        beta = np.sqrt(out_bytes / total_bytes)
        # 정규화 (소수점 오차 보정)
        norm = np.sqrt(alpha ** 2 + beta ** 2)
        if norm > 0:
            alpha = alpha / norm
            beta = beta / norm
        else:
            alpha, beta = 1.0 / np.sqrt(2), 1.0 / np.sqrt(2)  # 기본 상태
    else:
        # 기본 상태: 아다마르 기저 상태
        alpha, beta = 1.0 / np.sqrt(2), 1.0 / np.sqrt(2)

    # QuTiP 큐비트 상태 벡터 생성
    psi0 = qt.Qobj([[alpha], [beta]], dims=[[2], [1]])
    # 밀도 행렬로 변환: rho = |psi><psi|
    rho0 = qt.ket2dm(psi0)

    # 2. 노이즈 확률 p 추정: 여러 피처를 종합적으로 고려
    # 2.1 재전송 비율 기반 확률 (에러율 반영)
    in_pkts = max(1.0, float(row.get('IN_PKTS', 1)))
    retrans_in_pkts = max(0.0, float(row.get('RETRANSMITTED_IN_PKTS', 0)))
    out_pkts = max(1.0, float(row.get('OUT_PKTS', 1)))
    retrans_out_pkts = max(0.0, float(row.get('RETRANSMITTED_OUT_PKTS', 0)))

    # ZeroDivisionError 방지: 각 방향별로 안전하게 계산
    p_retrans_in = retrans_in_pkts / in_pkts if in_pkts > 0 else 0.0
    p_retrans_out = retrans_out_pkts / out_pkts if out_pkts > 0 else 0.0
    p_retrans = (p_retrans_in + p_retrans_out) / 2.0

    # 2.2 IAT 표준편차 기반 확률 (타이밍 불확실성 반영)
    iat_stddev_src = max(0.0, float(row.get('SRC_TO_DST_IAT_STDDEV', 0)))
    iat_stddev_dst = max(0.0, float(row.get('DST_TO_SRC_IAT_STDDEV', 0)))
    # 표준편차를 [0, 1] 범위로 정규화 (예: max_stddev = 1000ms 가정)
    max_iat_stddev = 1000.0  # 최대 표준편차 가정값
    normalized_iat_stddev = min(1.0, (iat_stddev_src + iat_stddev_dst) / (2.0 * max_iat_stddev))
    p_iat = 0.1 * normalized_iat_stddev  # 스케일링

    # 2.3 재전송 바이트 비율 기반 확률
    retrans_in_bytes = max(0.0, float(row.get('RETRANSMITTED_IN_BYTES', 0)))
    retrans_out_bytes = max(0.0, float(row.get('RETRANSMITTED_OUT_BYTES', 0)))
    total_retrans_bytes = retrans_in_bytes + retrans_out_bytes

    if total_bytes > 0:
        p_bytes = total_retrans_bytes / total_bytes
    else:
        p_bytes = 0.0

    # 2.4 전송 속도 변동성 기반 확률 (초당 바이트 전송 속도와 IAT 최대값의 불규칙성)
    bytes_per_sec_src = max(0.0, float(row.get('SRC_TO_DST_SECOND_BYTES', 0)))
    bytes_per_sec_dst = max(0.0, float(row.get('DST_TO_SRC_SECOND_BYTES', 0)))
    avg_bytes_per_sec = (bytes_per_sec_src + bytes_per_sec_dst) / 2.0

    # IAT 최대값과 평균값을 비교하여 변동성 추정
    iat_max_src = max(0.0, float(row.get('SRC_TO_DST_IAT_MAX', 0)))
    iat_max_dst = max(0.0, float(row.get('DST_TO_SRC_IAT_MAX', 0)))
    iat_avg_src = max(0.0, float(row.get('SRC_TO_DST_IAT_AVG', 0)))
    iat_avg_dst = max(0.0, float(row.get('DST_TO_SRC_IAT_AVG', 0)))

    avg_iat_max = (iat_max_src + iat_max_dst) / 2.0
    avg_iat_avg = (iat_avg_src + iat_avg_dst) / 2.0

    # IAT 최대값이 평균값보다 크면 변동성이 크다는 의미
    if avg_iat_avg > 0:
        # 변동성 계수: 최대값/평균값 (1보다 크면 불규칙성 증가)
        variability_ratio = avg_iat_max / avg_iat_avg
        # 정규화하여 노이즈 확률로 변환 (비율이 10 이상이면 최대값 도달)
        p_timing_iat = min(0.25, (variability_ratio - 1.0) / 10.0) if variability_ratio > 1.0 else 0.0
        p_timing_iat = max(0.0, p_timing_iat)
    elif avg_iat_max > 0:
        # 평균값이 없지만 최대값이 있는 경우 직접 사용
        p_timing_iat = min(0.25, avg_iat_max / (10.0 * 1000.0))  # 10초 이상이면 0.25로 제한
    else:
        p_timing_iat = 0.0

    # 전송 속도가 낮은데 IAT가 높으면 불안정한 것으로 간주
    # 전송 속도 기반 추가 노이즈 확률 (낮은 전송 속도는 높은 지연을 의미할 수 있음)
    if avg_bytes_per_sec > 0:
        # 전송 속도가 매우 낮으면 (예: 1000 bytes/sec 미만) 추가 노이즈
        p_timing_speed = min(0.05, max(0.0, (1000.0 - avg_bytes_per_sec) / 10000.0))
    else:
        p_timing_speed = 0.05  # 전송 속도가 0이면 최소 노이즈

    p_timing = min(0.3, p_timing_iat + p_timing_speed)

    # 종합 노이즈 확률: 가중 평균
    p = np.clip(
        0.5 * p_retrans + 0.2 * p_iat + 0.2 * p_bytes + 0.1 * p_timing,
        0.0, 1.0
    )

    # p가 너무 작으면 최소값 보장 (양자 시뮬레이션의 의미 보존)
    p = max(p, 0.001)

    # 3. Bit-Flip 시뮬레이션 (생성된 초기 상태 사용)
    bit_flip = BitFlipSimulation(initial_state=rho0, p_values=[p]).simulate()
    entropy_bit = bit_flip['entropies'][0]

    # 4. Phase-Flip 시뮬레이션 (생성된 초기 상태 사용)
    phase_flip = PhaseFlipSimulation(initial_state=rho0, p_values=[p]).simulate()
    entropy_phase = phase_flip['entropies'][0]

    # 5. 평균 엔트로피 계산
    noise_level = (entropy_bit + entropy_phase) / 2.0

    # 6. JAX/numpy 배열을 Python float로 변환
    try:
        if hasattr(noise_level, "item"):
            return float(noise_level.item())
        elif hasattr(noise_level, "__len__") and len(noise_level) == 1:
            return float(noise_level[0])
        else:
            noise_array = da.from_array(np.asarray(noise_level), chunks='auto')
            return float(noise_array.compute().item())
    except (TypeError, ValueError):
        return 0.0  # 오류 시 기본값


__all__ = ['split', 'scaling', 'feature_analysis', 'apply_entropy', 'apply_timing_variance',
           'apply_quantum_noise_simulation']
