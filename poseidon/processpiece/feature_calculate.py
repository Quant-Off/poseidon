import dask.array as da
import numpy as np
import qutip as qt
from poseidon.simulations.noise_modeling import BitFlipSimulation, PhaseFlipSimulation
from poseidon.processpiece.engineering_using_features import (
    bytes_features,
    timing_variance_features,
)
from poseidon.util.shannon import entropy_sn
from poseidon.util.timing_variance import timing_variance


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
    values_array = da.from_array(
        np.array(values), chunks="auto"
    )  # Dask 배열로 변환하여 병렬 처리 지원
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
    timespamps_array = da.from_array(np.array(sorted(timespamps)), chunks="auto")
    return timing_variance(timespamps_array, normalize=True)


def apply_quantum_noise_simulation(row):
    """
    이 함수는 각 행에서 quantum_noise_simulation_features를 기반으로 초기 상태 밀도 행렬을 생성하고,
    노이즈 확률 p를 추정한 후, 비트-플립과 페이즈-플립 시뮬레이션을 수행하여 폰 노이만 엔트로피를 계산합니다.
    """

    # quantum_noise_simulation_features 리스트를 사용하여 피처 값 추출
    def get_feature_value(feat_name, default=0.0):
        """피처 값을 안전하게 가져오는 헬퍼 함수"""
        try:
            val = row.get(feat_name, default) if hasattr(row, "get") else row[feat_name]
            return max(0.0, float(val)) if not np.isnan(val) else default
        except (KeyError, TypeError, ValueError):
            return default

    # 1. 초기 상태 밀도 행렬 생성: 바이트 피처 값을 기반으로 큐비트 상태 벡터 생성
    # IN_BYTES와 OUT_BYTES를 정규화하여 큐비트 상태의 파라미터로 사용
    in_bytes = get_feature_value("IN_BYTES", 0.0)
    out_bytes = get_feature_value("OUT_BYTES", 0.0)
    total_bytes = in_bytes + out_bytes

    if total_bytes > 0:
        # 바이트 비율을 기반으로 큐비트 상태 생성
        # |psi> = sqrt(in_bytes/total_bytes)|0> + sqrt(out_bytes/total_bytes)|1>
        alpha = np.sqrt(in_bytes / total_bytes)
        beta = np.sqrt(out_bytes / total_bytes)
        # 정규화 (소수점 오차 보정)
        norm = np.sqrt(alpha**2 + beta**2)
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
    # quantum_noise_simulation_features에서 필요한 피처 값들을 추출
    # 2.1 재전송 비율 기반 확률 (에러율 반영)
    in_pkts = max(1.0, get_feature_value("IN_PKTS", 1.0))
    retrans_in_pkts = get_feature_value("RETRANSMITTED_IN_PKTS", 0.0)
    out_pkts = max(1.0, get_feature_value("OUT_PKTS", 1.0))
    retrans_out_pkts = get_feature_value("RETRANSMITTED_OUT_PKTS", 0.0)

    # ZeroDivisionError 방지: 각 방향별로 안전하게 계산
    p_retrans_in = retrans_in_pkts / in_pkts if in_pkts > 0 else 0.0
    p_retrans_out = retrans_out_pkts / out_pkts if out_pkts > 0 else 0.0
    p_retrans = (p_retrans_in + p_retrans_out) / 2.0

    # 2.2 IAT 표준편차 기반 확률 (타이밍 불확실성 반영)
    iat_stddev_src = get_feature_value("SRC_TO_DST_IAT_STDDEV", 0.0)
    iat_stddev_dst = get_feature_value("DST_TO_SRC_IAT_STDDEV", 0.0)
    # 표준편차를 [0, 1] 범위로 정규화 (예: max_stddev = 1000ms 가정)
    max_iat_stddev = 1000.0  # 최대 표준편차 가정값
    normalized_iat_stddev = min(
        1.0, (iat_stddev_src + iat_stddev_dst) / (2.0 * max_iat_stddev)
    )
    p_iat = 0.1 * normalized_iat_stddev  # 스케일링

    # 2.3 재전송 바이트 비율 기반 확률
    retrans_in_bytes = get_feature_value("RETRANSMITTED_IN_BYTES", 0.0)
    retrans_out_bytes = get_feature_value("RETRANSMITTED_OUT_BYTES", 0.0)
    total_retrans_bytes = retrans_in_bytes + retrans_out_bytes

    if total_bytes > 0:
        p_bytes = total_retrans_bytes / total_bytes
    else:
        p_bytes = 0.0

    # 2.4 전송 속도 변동성 기반 확률 (초당 바이트 전송 속도와 IAT 최대값의 불규칙성)
    bytes_per_sec_src = get_feature_value("SRC_TO_DST_SECOND_BYTES", 0.0)
    bytes_per_sec_dst = get_feature_value("DST_TO_SRC_SECOND_BYTES", 0.0)
    avg_bytes_per_sec = (bytes_per_sec_src + bytes_per_sec_dst) / 2.0

    # IAT 통계값 추출 (quantum_noise_simulation_features에 포함된 모든 IAT 피처 사용)
    iat_min_src = get_feature_value("SRC_TO_DST_IAT_MIN", 0.0)
    iat_max_src = get_feature_value("SRC_TO_DST_IAT_MAX", 0.0)
    iat_avg_src = get_feature_value("SRC_TO_DST_IAT_AVG", 0.0)
    iat_min_dst = get_feature_value("DST_TO_SRC_IAT_MIN", 0.0)
    iat_max_dst = get_feature_value("DST_TO_SRC_IAT_MAX", 0.0)
    iat_avg_dst = get_feature_value("DST_TO_SRC_IAT_AVG", 0.0)

    avg_iat_max = (iat_max_src + iat_max_dst) / 2.0
    avg_iat_avg = (iat_avg_src + iat_avg_dst) / 2.0
    avg_iat_min = (iat_min_src + iat_min_dst) / 2.0

    # IAT 최대값이 평균값보다 크면 변동성이 크다는 의미
    if avg_iat_avg > 0:
        # 변동성 계수: 최대값/평균값 (1보다 크면 불규칙성 증가)
        variability_ratio = avg_iat_max / avg_iat_avg
        # 정규화하여 노이즈 확률로 변환 (비율이 10 이상이면 최대값 도달)
        p_timing_iat = (
            min(0.25, (variability_ratio - 1.0) / 10.0)
            if variability_ratio > 1.0
            else 0.0
        )
        p_timing_iat = max(0.0, p_timing_iat)
    elif avg_iat_max > 0:
        # 평균값이 없지만 최대값이 있는 경우 직접 사용
        p_timing_iat = min(
            0.25, avg_iat_max / (10.0 * 1000.0)
        )  # 10초 이상이면 0.25로 제한
    else:
        p_timing_iat = 0.0

    # IAT 최소값도 변동성에 활용 (최대값과 최소값의 차이가 클수록 불규칙성 증가)
    if avg_iat_max > 0 and avg_iat_min >= 0:
        iat_range_ratio = (
            (avg_iat_max - avg_iat_min) / avg_iat_max if avg_iat_max > 0 else 0.0
        )
        p_timing_range = min(
            0.05, iat_range_ratio * 0.1
        )  # 범위 비율을 노이즈 확률로 변환
    else:
        p_timing_range = 0.0

    # 전송 속도가 낮은데 IAT가 높으면 불안정한 것으로 간주
    # 전송 속도 기반 추가 노이즈 확률 (낮은 전송 속도는 높은 지연을 의미할 수 있음)
    if avg_bytes_per_sec > 0:
        # 전송 속도가 매우 낮으면 (예: 1000 bytes/sec 미만) 추가 노이즈
        p_timing_speed = min(0.05, max(0.0, (1000.0 - avg_bytes_per_sec) / 10000.0))
    else:
        p_timing_speed = 0.05  # 전송 속도가 0이면 최소 노이즈

    p_timing = min(0.3, p_timing_iat + p_timing_speed + p_timing_range)

    # 종합 노이즈 확률: 가중 평균
    p = np.clip(
        0.5 * p_retrans + 0.2 * p_iat + 0.2 * p_bytes + 0.1 * p_timing, 0.0, 1.0
    )

    # p가 너무 작으면 최소값 보장 (양자 시뮬레이션의 의미 보존)
    p = max(p, 0.001)

    # 3. Bit-Flip 시뮬레이션 (생성된 초기 상태 사용)
    bit_flip = BitFlipSimulation(initial_state=rho0, p_values=[p]).simulate()
    entropy_bit = bit_flip["entropies"][0]

    # 4. Phase-Flip 시뮬레이션 (생성된 초기 상태 사용)
    phase_flip = PhaseFlipSimulation(initial_state=rho0, p_values=[p]).simulate()
    entropy_phase = phase_flip["entropies"][0]

    # 5. 평균 엔트로피 계산
    noise_level = (entropy_bit + entropy_phase) / 2.0

    # 6. JAX/numpy 배열을 Python float로 변환
    try:
        if hasattr(noise_level, "item"):
            return float(noise_level.item())
        elif hasattr(noise_level, "__len__") and len(noise_level) == 1:
            return float(noise_level[0])
        else:
            noise_array = da.from_array(np.asarray(noise_level), chunks="auto")
            return float(noise_array.compute().item())
    except (TypeError, ValueError):
        return 0.0  # 오류 시 기본값


__all__ = ["apply_entropy", "apply_timing_variance", "apply_quantum_noise_simulation"]
