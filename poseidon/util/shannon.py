"""
섀넌 엔트로피 계산을 위한 모듈

이 모듈은 바이트 배열의 엔트로피를 섀넌의 정보 이론 공식을 사용하여 계산합니다.
네트워크 패킷 분석, 암호화 강도 측정, 데이터 압축률 예측 등에 활용됩니다.

주요 기능
-------
1. 섀넌 엔트로피 계산 (shannon_entropy)
   - 바이트 배열의 엔트로피를 섀넌의 정보 이론 공식을 사용하여 계산합니다.
   - 네트워크 패킷 분석, 암호화 강도 측정, 데이터 압축률 예측 등에 활용됩니다.

2. 패킷 도착 시간(타이밍) 변동 계산 (timing_variance)
   - 패킷 간 도착 시간 차이의 표준편차를 계산합니다.
   - 네트워크 트래픽의 규칙성을 평가하는 데 사용됩니다.

이론적 배경
---------
온디바이스 TinyML 기반의 이상 패킷 탐지 시스템은 실시간으로 네트워크 트래픽을 모니터링하고,
위협 징후가 감지되면 즉시 PQC(Post-Quantum Cryptography) 알고리즘의 보안 수준을 자동 상향 조정하며 그룹 경고 신호를 발송하는 기능이 중요합니다.
이런 기능을 효과적으로 구현하기 위해서는 다음과 같은 수학적 지표들이 필요합니다.

1. 섀넌 엔트로피(Shannon entropy)
   - 섀넌 엔트로피는 데이터의 무작위성(randomness)과 예측 불가능성(unpredictability)을 정량적으로 측정합니다.
   - 정상 트래픽과 달리, 패킷 페이로드의 엔트로피가 급격히 증가하거나 감소하는 현상은 악성 공격(암호화된 페이로드,
     데이터 인젝션, 패턴 기반 공격 등)의 징후가 될 수 있습니다.
   - 엔트로피 기반 탐지는 다양한 공격 벡터(제로데이, 변형 악성코드 등)에 대한 사전 탐지력과 일반화 능력을 제공합니다.
   - TinyML 모델에서는 엔트로피 값을 주요 입력 피처로 사용해, 이상 패킷을 빠르게 분류하고 불확실 구간에서 PQC 보안레벨을 상향 조정하도록 트리거할 수 있습니다.

2. 패킷 도착 시간(타이밍) 변동(Inter-Arrival Timing Variance)
   - 정상 네트워크 트래픽은 일정한 패턴(IoT 기기의 주기적 통신, 고정된 응답 지연 등)을 보이는 경우가 많으나,
     DoS, 스캐닝, 정보 탈취 등 공격 시에는 패킷 도착 간격이 불규칙하게 변동할 수 있습니다.
   - 패킷 도착 시간 차이의 표준편차는 트래픽의 규칙성(disziplinarity) 혹은 이상 현상(anomaly)을 분석하는 데 매우 효과적입니다.
   - TinyML 모델은 이 변동 값을 실시간으로 입력받아, 비정상적인 변동(예: 갑작스럽게 표준편차가 커지는 경우)이 감지될 때 PQC 보안 강화와 그룹 알림을 동시에
     트리거할 수 있습니다.

정리하자면, 섀넌의 엔트로피 공식과 패킷 도착 시간 변동(분산, 표준편차) 계산은 온디바이스 TinyML 기반 이상 탐지-대응(PQC 상향 자동 전환, 경고 발송)
시스템 구현에 핵심적인 수학 도구가 됩니다. 이러한 정량적 지표들은 공격의 조기 징후를 감지하고 실시간 자동 대응의 근거로 사용될 수 있습니다.

작성자
------
Q. T. Felix

라이선스
--------
MIT License
"""

import jax.numpy as jnp
import numpy as np
import qutip_jax
from jax import jit

qutip_jax.set_as_default()


@jit
def _entropy_sn_unit8(arr):
    """
    내부 JIT 함수이며, uint8 배열의 섀넌 엔트로피를 계산합니다.
    """
    freq = jnp.bincount(arr, length=256)
    total_bytes = arr.size
    freq = freq / total_bytes
    entropy = -jnp.sum(jnp.where(freq > 0, freq * jnp.log2(freq), 0))
    return entropy


def entropy_sn(packet_bytes):
    """
    바이트 배열의 섀넌 엔트로피를 계산합니다.

    섀넌 엔트로피는 데이터의 무작위성이나 예측 불가능성을 측정하는 지표입니다.
    값이 높을수록 데이터가 더 무작위적이고 예측하기 어렵습니다.

    Parameters
    ----------
    packet_bytes : array_like or bytes
        엔트로피를 계산할 바이트 배열. 0-255 범위의 정수값을 포함해야 합니다.
        numpy 배열, 리스트, 또는 바이트 객체가 될 수 있습니다.

    Returns
    -------
    float
        계산된 섀넌 엔트로피 값 (비트 단위).
        - 0: 완전히 예측 가능한 데이터 (모든 바이트가 동일)
        - 8: 완전히 무작위적인 데이터 (모든 바이트가 균등하게 분포)

    Raises
    ------
    ValueError
        입력 배열이 비어있거나 유효하지 않은 바이트 값을 포함하는 경우

    Examples
    --------
    >>> # 완전히 무작위적인 데이터 (최대 엔트로피)
    >>> random_data = jnp.array(np.random.randint(0, 256, size=1000))
    >>> entropy = entropy_sn(random_data)
    >>> print(f"랜덤 데이터 엔트로피: {entropy:.2f} bits")  # ~8.0

    >>> # 반복적인 데이터 (낮은 엔트로피)
    >>> repeated_data = jnp.array([1, 1, 1, 1, 1])
    >>> entropy = entropy_sn(repeated_data)
    >>> print(f"반복 데이터 엔트로피: {entropy:.2f} bits")  # 0.0

    >>> # bytes 입력 예시
    >>> text_data = b'hello'
    >>> entropy = entropy_sn(text_data)
    >>> print(f"텍스트 엔트로피: {entropy:.2f} bits")

    Notes
    -----
    이 함수는 메모리 효율성을 위해 jax.numpy의 bincount를 사용합니다.
    입력 데이터의 크기에 관계없이 항상 256개의 빈(bin)만 사용하므로
    대용량 데이터 처리에 적합합니다. qutip-jax와 통합 시 자동 미분을 지원합니다.
    동적 불리언 인덱싱 오류를 피하기 위해 jnp.where를 사용하여 마스킹합니다.
    bytes 입력은 함수 외부에서 배열로 변환되지만, 내부적으로 처리됩니다.

    섀넌 엔트로피 공식: H(X) = -Σ p(x) * log₂(p(x))
    여기서 p(x)는 각 바이트 값의 확률입니다.
    """
    if len(packet_bytes) == 0:
        raise ValueError("입력 배열이 비어있습니다!")

    # 입력을 numpy 배열로 먼저 변환 (JIT 외부 처리)
    if isinstance(packet_bytes, (bytes, bytearray)):
        packet_bytes = np.frombuffer(packet_bytes, dtype=np.uint8)
    else:
        packet_bytes = np.asarray(packet_bytes, dtype=np.uint8)

    # JAX 배열로 변환 후 내부 JIT 함수 호출
    arr = jnp.asarray(packet_bytes)
    return _entropy_sn_unit8(arr)
