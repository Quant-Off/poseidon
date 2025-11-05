"""
특정 피처를 사용하여 데이터셋을 엔지니어링하는 모듈입니다.
입력된 피처 배열을 사용하여 데이터셋을 엔지니어링하고, 새로운 피처를 생성합니다.

특정 피처 배열은 직접적으로 사용되지 않고 참고만 될 수 있습니다. 실험 단계인 지금, 이 기능을 전적으로 믿고 사용하시면 안 됩니다.
"""

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

__all__ = [
    "bytes_features",
    "timing_variance_features",
    "quantum_noise_simulation_features",
]
