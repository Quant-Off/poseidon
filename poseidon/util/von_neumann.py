"""
폰 노이만(von Neumann) 엔트로피 계산 모듈입니다.

개요
----
이 모듈은 양자 상태의 밀도 행렬(density matrix)로부터 폰 노이만 엔트로피를 계산하기 위한 유틸리티를 제공합니다.
JAX를 기반으로 구현되어 JIT(Just-In-Time) 컴파일과 XLA 가속을 활용할 수 있으며, 수치적 안정성과 JIT 제약을 고려한
구조(분기에는 `jax.lax.cond`, bool 마스킹에는 `jnp.where`)를 사용합니다.

제공 함수
---------
1) validate_rho(rho, check_hermitian=True, check_trace=True)
   - 목적: JIT 외부에서 입력 밀도 행렬 `rho`의 기초 검증을 수행합니다.
   - 동작:
     - `qutip.Qobj` 입력을 JAX 배열로 변환합니다.
     - 정방 행렬 여부 확인, 에르미트 여부(옵션), 대각합(trace)=1 여부(옵션)를 검사합니다.
   - 반환: `(rho_validated, trace_val)`
     - `rho_validated`: `jnp.complex128` dtype의 2차원 정방 행렬(square matrix)
     - `trace_val`: `jnp.trace(rho)` 결과(복소 대각합의 실수부가 1에 가깝도록 가정)
   - 예외: 형상/성질 위반 시 `ValueError`를 발생시킵니다.

2) entropy_vn_jitted(rho, trace_val, base=2, normalize=False, eps=1e-15)
   - 목적: 밀도 행렬의 폰 노이만 엔트로피를 계산합니다. JIT 컴파일에 안전한 형태로 작성되었습니다.
   - 입력:
     - `rho`: `validate_rho`를 거쳐 JAX 배열로 준비된 밀도 행렬
     - `trace_val`: `validate_rho`에서 계산된 trace 값
     - `base`: 로그 밑(2 bits, `jnp.e` nats, 임의 밑도 지원)
     - `normalize`: `True`인 경우 trace로 정규화 수행(0 trace는 예외 처리)
     - `eps`: 매우 작은 고유값을 0으로 클리핑하기 위한 임계값(수치적 안정성)
   - 내부 구현 포인트(수치/성능):
     - 고유값: `jnp.linalg.eigvalsh` 사용(에르미트 가정)
     - 작은 고유값 제거: `eigenvalues = jnp.where(eigenvalues <= eps, 0.0, eigenvalues)`.
     - 0 로그 방지: `non_zero_mask`를 이용해 0을 1로 치환해 로그 계산 후, 해당 위치를 0으로 복원.
     - 로그 밑 분기: `if/elif` 대신 `lax.cond`로 구현하여 JIT 시 동적 분기 에러를 회피.
   - 반환: 스칼라 엔트로피 값(지정 밑 기준).

JAX/JIT 관련 주의사항
---------------------
- 파이썬 레벨의 동적 분기(`if base == ...`)나 동적 불리언 인덱싱은 Tracer 관련 오류를 유발할 수 있으므로
  본 모듈에서는 `lax.cond`와 `jnp.where`로 대체하였습니다.
- 불리언 인덱싱 대신 마스크 연산을 사용하여 JIT 호환성을 보장합니다.

사용 예시
--------
다음 예시는 bits(밑 2)와 nats(밑 e) 기준으로 엔트로피를 계산하는 방법을 보여줍니다.

>>> import numpy as np
>>> import jax.numpy as jnp
>>> from poseidon.util.von_neumann import validate_rho, entropy_vn_jitted

>>> # 2x2 균등 혼합 상태 (trace=1, Hermitian)
>>> rho_np = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.complex128)
>>> rho_valid, tr = validate_rho(rho_np)

>>> # bits 기준 (base=2)
>>> H_bits = entropy_vn_jitted(rho_valid, tr, base=2)
>>> float(H_bits) # 1.0에 근접

>>> # nats 기준 (base=jnp.e)
>>> H_nats = entropy_vn_jitted(rho_valid, tr, base=jnp.e)
>>> float(H_nats) # ln(2)에 근접

작성자
------
Q. T. Felix

라이선스
--------
MIT License
"""

import jax.numpy as jnp
from jax import jit, lax
from qutip import Qobj


def validate_rho(rho, check_hermitian=True, check_trace=True):
    """
    JIT 외부에서 입력 검증 수행합니다.

    Parameters
    ----------
    rho : Qobj
        밀도 행렬.
    check_hermitian : bool, optional
        밀도 행렬이 Hermitian인지 검사합니다.
    check_trace : bool, optional
        밀도 행렬의 trace가 1인지 검사합니다.

    Returns
    -------
    rho : jnp.array
        밀도 행렬.
    trace_val : float
        밀도 행렬의 trace.
    """
    if isinstance(rho, Qobj):
        rho = jnp.array(rho.full(), dtype=jnp.complex128)
    rho = jnp.asarray(rho, dtype=jnp.complex128)

    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("입력은 정방 2차원 배열이어야 합니다!")

    if check_hermitian and not jnp.allclose(rho, jnp.conj(rho.T), atol=1e-10):
        raise ValueError("밀도 행렬은 에르미트(Hermitian)이어야 합니다!")

    trace_val = jnp.trace(rho)
    if check_trace and not jnp.isclose(trace_val, 1.0, atol=1e-10):
        raise ValueError("밀도 행렬의 대각합(trace)은 1이어야 합니다!")

    return rho, trace_val


@jit
def entropy_vn_jitted(rho, trace_val, base=2, normalize=False, eps=1e-15):
    """
    폰 노이만 엔트로피를 계산합니다(JIT 호환).

    Parameters
    ----------
    rho : jnp.array
        밀도 행렬.
    trace_val : float
        밀도 행렬의 trace.
    base : float, optional
        로그 밑수.
    normalize : bool, optional
        밀도 행렬을 trace로 정규화합니다.
    eps : float, optional
        0으로 나누는 것을 방지하기 위한 작은 값.

    Returns
    -------
    entropy : float
        폰 노이만 엔트로피 값.
    """

    def normalize_rho(rho, trace_val):
        return lax.cond(
            jnp.isclose(trace_val, 0.0),
            lambda _: jnp.zeros_like(rho), # 더미 반환 (오류 방지)
            lambda _: rho / trace_val,
            None,
        )

    rho = lax.cond(
        normalize, lambda _: normalize_rho(rho, trace_val), lambda _: rho, None
    )

    eigenvalues = jnp.linalg.eigvalsh(rho)
    eigenvalues = jnp.where(eigenvalues <= eps, 0.0, eigenvalues)

    ev_sum = jnp.sum(eigenvalues)
    eigenvalues = lax.cond(
        normalize, lambda _: eigenvalues / ev_sum, lambda _: eigenvalues, None
    )

    def log_base_2(x):
        return jnp.log2(x)

    def log_base_e(x):
        return jnp.log(x)

    def log_base_other(x):
        return jnp.log(x) / jnp.log(base)

    # 0이 아닌 고유값에 대해서만 로그 계산
    non_zero_mask = eigenvalues > 0
    safe_eigenvalues = jnp.where(
        non_zero_mask, eigenvalues, 1.0
    )  # 0을 1로 대체하여 로그 계산

    # base 값에 따라 로그 함수 선택 (람다 제거)
    def select_log(x):
        return lax.cond(
            jnp.isclose(base, jnp.e),
            log_base_e,
            log_base_other,
            x,
        )

    log_result = lax.cond(
        jnp.isclose(base, 2.0),
        log_base_2,
        select_log,
        safe_eigenvalues,
    )

    # 0이었던 위치는 0으로 복원
    log_result = jnp.where(non_zero_mask, log_result, 0.0)

    return -jnp.sum(eigenvalues * log_result)
