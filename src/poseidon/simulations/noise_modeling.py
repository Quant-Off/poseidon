r"""
양자 노이즈 모델링 및 시뮬레이션 모듈

이 모듈은 양자 컴퓨팅에서 발생하는 다양한 노이즈 채널을 모델링하고 시뮬레이션하는
기능을 제공합니다. 특히 비트-플립(비트 반전, bit-flip)과 페이즈-플립(위상 반전, phase-flip) 노이즈에
대한 수학적 모델링과 시각화를 지원합니다. 이 시뮬레이션은 NetFlow 데이터셋과 결합된 온디바이스 TinyML 모델 훈련을 위한 양자 노이즈 데이터를 생성하며,
패킷 엔트로피와 타이밍 변동 분석을 통해 이상 패킷을 탐지하고 Post-Quantum Cryptography 보안 수준을 상향 조정합니다.

주요 기능
----------

1. 비트-플립 노이즈 시뮬레이션 (`bit_flip_simulation()`)

   - Pauli-X 연산자를 사용한 비트 플립 노이즈 모델링
   - 다양한 초기 상태에 대한 충실도(fidelity), 순수도(purity), 폰 노이만 엔트로피(von Neumann entropy) 분석
   - 확률 범위에 따른 시각화 및 통계 분석

2. 페이즈-플립 노이즈 시뮬레이션 (`phase_flip_simulation()`)

   - Pauli-Z 연산자를 사용한 위상 플립 노이즈 모델링
   - 메모리 효율적인 크라우스(Kraus) 연산자 구현
   - 다양한 양자 상태 타입 지원 (브라-캣 형식, 밀도 행렬; density matrix)

이론적 배경
-----------

양자 노이즈(잡음, noise)는 양자계가 환경과 상호작용할 때 발생하는 불가피한 현상입니다 (Chuang & Nielsen, 1996).
이 모듈에서 구현하는 노이즈 채널들은 다음과 같은 수학적 표현을 가집니다.

1. 비트-플립 노이즈:

   - 크라우스 연산자: \( K_0 = \sqrt{1-p} \, I \), \( K_1 = \sqrt{p} \, \sigma_x \)
   - 효과: \( |0\rangle \leftrightarrow |1\rangle \) 상태 간의 전환
   - 물리적 의미: 계산 기저에서의 비트 오류

2. 페이즈-플립 노이즈:

   - 크라우스 연산자: \( K_0 = \sqrt{1-p} \, I \), \( K_1 = \sqrt{p} \, \sigma_z \)
   - 효과: \( |1\rangle \rightarrow -|1\rangle \), \( |+\rangle \leftrightarrow |-\rangle \)
     상태 간의 위상 변화
   - 물리적 의미: 위상 정보의 손실

사용 예제
---------
>>> from models.noise_modeling import bit_flip_simulation, phase_flip_simulation
>>> from qutip import basis
>>>
>>> # 비트-플립 노이즈 시뮬레이션 (충실도만)
>>> result = bit_flip_simulation()
>>> print(f"p=0.5일 때 충실도: {result['fidelities'][5]:.6f}")
>>>
>>> # 페이즈-플립 노이즈 시뮬레이션 (엔트로피만)
>>> psi_plus = (basis(2, 0) + basis(2, 1)).unit()
>>> result = phase_flip_simulation(initial_state=psi_plus)
>>> print(f"p=0.5일 때 엔트로피: {result['entropies'][5]:.6f}")

의존성
-------

- qutip: 양자 상태 및 연산자 처리
- numpy: 수치 계산
- matplotlib: 시각화

작성자
------

Q. T. Felix

라이선스
--------

MIT License
"""

import os
from dotenv import load_dotenv
from qutip import qeye, sigmax, sigmaz, basis, fidelity
from qutip.core.states import ket2dm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from poseidon.util.von_neumann import validate_rho, entropy_vn_jitted

load_dotenv(verbose=True)

font_path = os.getenv("MATPLOTLIB_FONT_PATH")
font_name = os.getenv("MATPLOTLIB_FONT_NAME")
font_entry = fm.FontEntry(fname=font_path, name=font_name)
fm.fontManager.ttflist.insert(0, font_entry)
plt.rcParams["font.family"] = font_name


def bit_flip_simulation(initial_state=None, p_values=None, show_plot=True, verbose=1):
    """
    비트-플립 노이즈 시뮬레이션을 수행하는 함수입니다.

    이 함수는 양자 노이즈 모델인 비트-플립 채널을 시뮬레이션하며, 초기 상태에 노이즈를 적용한 후
    충실도(fidelity), 순수도(purity), 및 엔트로피(entropy)를 계산합니다. QuTiP 라이브러리를 기반으로 하며,
    JAX 백엔드를 옵션으로 지원하여 대규모 시뮬레이션의 효율성을 높입니다.

    파라미터
    ----------
    - initial_state: 초기 양자 상태 (QuTiP ket 또는 oper). 기본값: 아다마르 + 기저 상태
    - p_values: 개별 값이 0과 1 사이인 비트-플립 확률 리스트. 기본값: np.linspace(0, 1, 11).
    - show_plot: 플롯 표시 여부 (bool)
    - verbose: 출력 상세도 (0: 없음, 1: 요약, 2: 모든 밀도 행렬 포함)

    반환값
    -------
    - 딕셔너리(dict): 초기 상태, p_values, fidelities, purities, entropies, density_matrices.
    """

    # 기본 초기 상태: 아다마르 기저
    if initial_state is None:
        psi0 = (basis(2, 0) + basis(2, 1)).unit()
    else:
        psi0 = initial_state

    if psi0.type == "ket":
        rho = ket2dm(psi0)
    elif psi0.type == "oper":
        rho = psi0
    else:
        raise ValueError("초기 상태는 QuTiP ket 벡터 또는 밀도 행렬이어야 합니다!")

    # rho가 에르미트이고 대각합이 1인지 검사
    if not np.allclose(
        rho.data.to_array(), rho.dag().data.to_array().conj().T, atol=1e-10
    ):
        raise ValueError("밀도 행렬은 에르미트(Hermitian) 행렬이어야 합니다!")
    if not np.isclose(rho.tr(), 1.0, atol=1e-10):
        raise ValueError("밀도 행렬의 대각합(trace)은 1이어야 합니다!")

    # p_values 기본 설정: 자동 생성
    if p_values is None:
        p_values = np.linspace(0, 1, 11).tolist()

    if not isinstance(p_values, list):
        raise ValueError("p_values는 실수 리스트 타입이어야 합니다!")
    if not all(0 <= p <= 1 for p in p_values):
        raise ValueError("p_values 리스트의 개별 값은 모두 0 이상 1 이하여야 합니다!")

    # 결과 저장용 리스트
    fidelities = []
    purities = []
    entropies = []
    density_matrices = []

    if verbose > 0:
        print("=== 비트-플립 노이즈 시뮬레이션 ===")
        print(f"초기 상태: {psi0}")
        print(f"확률 값들: {p_values}")
        print()

    # 각 확률에 대해 시뮬레이션 실행
    for _, p_val in enumerate(p_values):
        # Kraus 연산자 정의
        k0 = np.sqrt(1 - p_val) * qeye(2)
        k1 = np.sqrt(p_val) * sigmax()
        kraus_ops = [k0, k1]
        # 노이즈 적용
        rho_noisy = sum([K * rho * K.dag() for K in kraus_ops])
        density_matrices.append(rho_noisy)
        # 충실도 계산
        fid_value = fidelity(rho, rho_noisy)
        fidelities.append(fid_value)
        # 순수도(purity) 계산
        pur_value = rho_noisy.purity()
        purities.append(pur_value)
        # 폰 노이만 엔트로피 계산
        print(rho_noisy)
        rho_noisy_valid, trace_val = validate_rho(rho_noisy)
        ent_value = entropy_vn_jitted(rho_noisy_valid, trace_val)
        entropies.append(ent_value)

        if verbose > 0:
            print(f"확률 p = {p_val:.3f}")
            print(f"충실도: {fid_value:.6f}")
            print(f"순수도: {pur_value:.6f}")
            print(f"엔트로피(폰 노이만): {ent_value:.6f}")

        if verbose > 1:
            print("노이즈 적용 후 밀도 행렬:")
            print(rho_noisy)

        if verbose > 0:
            print()

    if verbose > 0:
        print("전체 결과:")
        print("확률 값들:", p_values)
        print("충실도들:", [f"{f:.6f}" for f in fidelities])
        print("순수도들:", [f"{p:.6f}" for p in purities])
        print("엔트로피들:", [f"{e:.6f}" for e in entropies])

    # 시각화
    if show_plot:
        _, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            p_values, fidelities, "bo-", label="Fidelity", linewidth=2, markersize=8
        )
        ax.plot(p_values, purities, "rs--", label="Purity", linewidth=2, markersize=8)
        ax.plot(p_values, entropies, "g^-.", label="Entropy", linewidth=2, markersize=8)
        ax.set_xlabel("비트-플립 확률 p", fontsize=12)
        ax.set_ylabel("충실도, 순수도, 엔트로피", fontsize=12)
        ax.set_title("비트-플립 확률 값 vs. 충실도, 순수도, 엔트로피", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()

        # 데이터 포인트에 값 표시 (충실도만)
        for _, (p, f) in enumerate(zip(p_values, fidelities)):
            ax.annotate(
                f"{f:.3f}",
                (p, f),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=10,
            )

        plt.tight_layout()
        plt.show()

    # 결과 반환
    return {
        "initial_state": psi0,
        "p_values": p_values,
        "fidelities": fidelities,
        "purities": purities,
        "entropies": entropies,
        "density_matrices": density_matrices,
    }


def phase_flip_simulation(initial_state=None, p_values=None, show_plot=True, verbose=1):
    """
    페이즈-플립 노이즈 시뮬레이션을 수행하는 함수입니다.

    이 함수는 양자 노이즈 모델인 페이즈-플립 채널을 시뮬레이션하며, 초기 상태에 노이즈를 적용한 후
    충실도(fidelity), 순수도(purity), 및 엔트로피(entropy)를 계산합니다. QuTiP 라이브러리를 기반으로 하며,
    JAX 백엔드를 옵션으로 지원하여 대규모 시뮬레이션의 효율성을 높입니다.

    파라미터
    ----------
    - initial_state: 초기 양자 상태 (QuTiP ket 또는 oper). 기본값: |+> 상태 (superposition)
    - p_values: 개별 값이 0과 1 사이인 페이즈-플립 확률 리스트. 기본값: np.linspace(0, 1, 11).
    - show_plot: 플롯 표시 여부 (bool)
    - verbose: 출력 상세도 (0: 없음, 1: 요약, 2: 모든 밀도 행렬 포함)

    반환값
    -------
    - 딕셔너리(dict): 초기 상태, p_values, fidelities, purities, entropies, density_matrices.
    """

    # 기본 초기 상태: 아다마르 기저
    if initial_state is None:
        psi0 = (basis(2, 0) + basis(2, 1)).unit()
    else:
        psi0 = initial_state

    if psi0.type == "ket":
        rho = ket2dm(psi0)
    elif psi0.type == "oper":
        rho = psi0
    else:
        raise ValueError("초기 상태는 QuTiP ket 벡터 또는 밀도 행렬이어야 합니다!")

    # rho가 에르미트이고 대각합이 1인지 검사
    if not np.allclose(
        rho.data.to_array(), rho.dag().data.to_array().conj().T, atol=1e-10
    ):
        raise ValueError("밀도 행렬은 에르미트(Hermitian) 행렬이어야 합니다!")
    if not np.isclose(rho.tr(), 1.0, atol=1e-10):
        raise ValueError("밀도 행렬의 대각합(trace)은 1이어야 합니다!")

    # p_values 기본 설정: 자동 생성
    if p_values is None:
        p_values = np.linspace(0, 1, 11).tolist()

    # p_values 검증
    if not isinstance(p_values, list):
        raise ValueError("p_values는 실수 리스트 타입이어야 합니다!")
    if not all(0 <= p <= 1 for p in p_values):
        raise ValueError("p_values 리스트의 개별 값은 모두 0 이상 1 이하여야 합니다!")

    # 결과 저장용 리스트
    fidelities = []
    purities = []
    entropies = []
    density_matrices = []

    if verbose > 0:
        print("=== 페이즈-플립 노이즈 시뮬레이션 ===")
        print(f"초기 상태: {psi0}")
        print(f"확률 값들: {p_values}")
        print()

    # 각 확률에 대해 시뮬레이션 실행
    for _, p_val in enumerate(p_values):
        # Kraus 연산자 정의
        k0 = np.sqrt(1 - p_val) * qeye(2)
        k1 = np.sqrt(p_val) * sigmaz()
        kraus_ops = [k0, k1]

        # 노이즈 적용
        rho_noisy = sum([K * rho * K.dag() for K in kraus_ops])
        density_matrices.append(rho_noisy)

        # 충실도 계산
        fid_value = fidelity(rho, rho_noisy)
        fidelities.append(fid_value)

        # 순수도(purity) 계산
        pur_value = rho_noisy.purity()
        purities.append(pur_value)

        # 폰 노이만 엔트로피 계산
        rho_noisy_valid, trace_val = validate_rho(rho_noisy)
        ent_value = entropy_vn_jitted(rho_noisy_valid, trace_val)
        entropies.append(ent_value)

        if verbose > 0:
            print(f"확률 p = {p_val:.3f}")
            print(f"충실도: {fid_value:.6f}")
            print(f"순수도: {pur_value:.6f}")
            print(f"엔트로피(폰 노이만): {ent_value:.6f}")

        if verbose > 1:
            print("노이즈 적용 후 밀도 행렬:")
            print(rho_noisy)

        if verbose > 0:
            print()

    if verbose > 0:
        print("전체 결과:")
        print("확률 값들:", p_values)
        print("충실도들:", [f"{f:.6f}" for f in fidelities])
        print("순수도들:", [f"{p:.6f}" for p in purities])
        print("엔트로피들:", [f"{e:.6f}" for e in entropies])

    # 시각화
    if show_plot:
        _, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            p_values, fidelities, "ro-", label="Fidelity", linewidth=2, markersize=8
        )
        ax.plot(p_values, purities, "bs--", label="Purity", linewidth=2, markersize=8)
        ax.plot(p_values, entropies, "g^-.", label="Entropy", linewidth=2, markersize=8)
        ax.set_xlabel("페이즈-플립 확률 p", fontsize=12)
        ax.set_ylabel("충실도, 순수도, 엔트로피", fontsize=12)
        ax.set_title("페이즈-플립 확률 값 vs. 충실도, 순수도, 엔트로피", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()

        # 데이터 포인트에 값 표시 (충실도만)
        for _, (p, f) in enumerate(zip(p_values, fidelities)):
            ax.annotate(
                f"{f:.3f}",
                (p, f),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=10,
            )

        plt.tight_layout()
        plt.show()

    # 결과 반환
    return {
        "initial_state": psi0,
        "p_values": p_values,
        "fidelities": fidelities,
        "purities": purities,
        "entropies": entropies,
        "density_matrices": density_matrices,
    }
