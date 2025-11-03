# 모듈: noise_modeling

이 모듈은 양자 컴퓨팅에서 발생하는 다양한 노이즈 채널을 모델링하고 시뮬레이션하는 기능을 제공합니다. 특히 비트-플립(비트 반전, bit-flip)과 페이즈-플립(위상 반전, phase-flip) 노이즈에 대한 수학적 모델링과 시각화를 지원합니다. 이 시뮬레이션은 NetFlow 데이터셋과 결합된 온디바이스 TinyML 모델 훈련을 위한 양자 노이즈 데이터를 생성하며, 패킷 엔트로피와 타이밍 변동 분석을 통해 이상 패킷을 탐지하고 양자-내성 암호화(Post-Quantum Cryptography, PQC) 알고리즘의 보안 수준을 상향 조정합니다.

## 주요 기능

1. 비트-플립 노이즈 시뮬레이션 `bit_flip_simulation()`

   - Pauli-X 연산자를 사용한 비트-플립 노이즈 모델링

2. 페이즈-플립 노이즈 시뮬레이션 `phase_flip_simulation()`

   - Pauli-Z 연산자를 사용한 페이즈-플립 노이즈 모델링

두 함수는 공통적으로 다양한 초기 상태에 대한 충실도(fidelity), 순수도(purity), 폰 노이만 엔트로피(von Neumann entropy)를 메모리 효율적으로 분석하고 계산합니다. 선택적으로 시각화 및 통계 분석을 수행할 수 있습니다. 또한 브라-캣 형식(bra-ket notation)과 밀도 행렬(density matrix)에 대한 양자 상태 타입을 지원합니다.

## 이론적 배경

양자 노이즈(잡음, noise)는 양자계가 환경과 상호작용할 때 발생하는 불가피한 현상입니다(Chuang & Nielsen, 1996). 이 모듈에서 구현하는 노이즈 채널들은 다음과 같은 수학적 표현을 가집니다.

1. 비트-플립 노이즈:

   - 크라우스 연산자: $ K_0 = \sqrt{1-p} \, I \), \( K_1 = \sqrt{p} \, \sigma_x $
   - 효과: $ \ket{0} \leftrightarrow \ket{1} $ 상태 간의 전환
   - 물리적 의미: 계산 기저(computational basis)에서의 비트 오류

2. 페이즈-플립 노이즈:

   - 크라우스 연산자: $ K_0 = \sqrt{1-p} \, I \), \( K_1 = \sqrt{p} \, \sigma_z $
   - 효과: $ \ket{1} \rightarrow -\ket{1} \), \( \ket{+} \leftrightarrow \ket{-} $
     상태 간의 위상 변화
   - 물리적 의미: 위상 정보(phase information)의 손실

## 사용 예제

```python
from models.noise_modeling import bit_flip_simulation, phase_flip_simulation
from qutip import basis

# 비트-플립 노이즈 시뮬레이션 (충실도만)

result = bit_flip_simulation()
print(f"p=0.5일 때 충실도: {result['fidelities'][5]:.6f}")

# 페이즈-플립 노이즈 시뮬레이션 (엔트로피만)

psi_plus = (basis(2, 0) + basis(2, 1)).unit()
result = phase_flip_simulation(initial_state=psi_plus)
print(f"p=0.5일 때 엔트로피: {result['entropies'][5]:.6f}")
```

## 의존성

- `qutip`: 양자 상태 및 연산자 처리
- `numpy`: 수치 계산
- `matplotlib`: 시각화(플롯)

## 작성자

`Q. T. Felix`

## 라이선스

MIT License

## 함수: 비트-플립

이 함수는 초기 양자 상태에 노이즈를 적용하고 울만 충실도(Uhlmann fidelity), 순수도(purity), 폰 노이만 엔트로피(von Neumann entropy)를 계산함으로써, NetFlow 데이터셋과 결합된 AI 훈련 데이터 생성을 지원합니다.

비트-플립(비트 반전, bit-flip) 노이즈 채널을 통한 양자 상태의 진화를 시뮬레이션합니다. 이 함수는 크라우스(Kraus) 연산자를 사용하여 비트-플립 노이즈를 모델링하고, 초기 양자 상태에 대한 메트릭 변화를 분석합니다. 비트-플립 노이즈는 다음과 같은 크라우스 연산자로 표현(모델링, modeling)됩니다.

$$\begin{aligned}
    K_0 &= \sqrt{1-p} \cdot I \\
    K_1 &= \sqrt{p} \cdot \sigma_x
\end{aligned}$$

여기서 $I$는 항등 연산자(identity operator), $\sigma_x$는 Pauli-X 연산자이며, $p$는 노이즈 확률입니다. 출력 밀도 행렬(density matrix)은 다음과 같이 계산됩니다.

$$
\rho' = \sum_k K_k \rho K_k^\dagger
$$

이 채널은 확률 $p$로 Pauli-X 연산자를 적용하여 비트 반전을 유발하며, 계산 기저($\ket{0}$,$\ket{1}$)에서 상태 전환을 일으킵니다.

### 파라미터

모든 파라미터는 기본값을 가지며, 초기 상태(`initial_state`)는 상태 벡터 또는 밀도 행렬 전달(입력)을 허용합니다.

- `initial_state`: 시뮬레이션할 초기 양자 상태. 기본값은 $\ket{0}$ 상태입니다.
  - **타입**: `qutip.Qobj`
    - $\ket{0}$ 상태: $\text{qutip.basis(2, 0)}$
    - $\ket{1}$ 상태: $\text{qutip.basis(2, 1)}$
    - $\ket{+}$ 상태: $\text{(qutip.basis(2, 0) + qutip.basis(2, 1)).unit()}$
    - $\ket{−}$ 상태: $\text{(qutip.basis(2, 0) - qutip.basis(2, 1)).unit()}$
    - 밀도 행렬 예시: `qutip.ket2dm(qutip.basis(2, 0)`

- `p_values`: 비트-플립 확률 값들의 리스트. 각 값은 0과 1 사이여야 하며, 기본값은 `np.linspace(0, 1, 11).tolist()`입니다.
  - **타입**: `float` 리스트

- `show_plot`: 메트릭 vs 확률 그래프를 표시할지 여부. 기본값은 `True`입니다.
  - **타입**: `bool`

- `verbose`: 시뮬레이션 과정의 상세 정보를 출력할지 여부. 기본값은 `1`입니다.
  - **타입**: `int` (0: 없음, 1: 요약, 2: 모든 밀도 행렬 포함)

- `use_jax`: JAX 백엔드 사용 여부. 기본값은 `False`입니다.
  - **타입**: `bool`

### 반환값

반환값은 `dict`이며, 시뮬레이션 결과를 포함한 딕셔너리를 의미합니다.

- `initial_state` : `qutip.Qobj`, 사용된 초기 양자 상태
- `p_values` : `float` 리스트, 시뮬레이션에 사용된 비트-플립 확률 값들
- `fidelities` : `float` 리스트, 각 확률에 대응하는 충실도 값들 ($0 \le \text{fidelity} \le 1$)
- `purities` : `float` 리스트, 각 확률에 대응하는 순수도 값들 ($0.5 \le \text{purity} \le 1$)
- `entropies` : `float` 리스트, 각 확률에 대응하는 폰 노이만 엔트로피 값들 ($0 \le \text{entropy} \le 1$)
- `density_matrices` : `qutip.Qobj` 리스트, 각 확률에 대해 노이즈가 적용된 후의 밀도 행렬들

### 노트

충실도 $F$는 두 양자 상태 간의 유사성을 측정하는 지표입니다. 다음과 같은 수식으로 표현됩니다.

$$
F(\rho, \sigma) = \left[\text{Tr} \sqrt{\sqrt{\rho}\ \sigma \sqrt{\rho}}\right]^2
$$

순수도 $\gamma$는 $\gamma = \mathrm{Tr}(\rho^2)$로 계산되며, 폰 노이만 엔트로피 $S$는 $S(\rho) = -\mathrm{Tr}(\rho \log_2 \rho)$입니다. $p = 0$일 때 $F = 1$(완전히 동일), $p = 1$일 때 상태 의존적입니다. 그리고 계산 기저에서 $F = |1-2p|$, 중첩 상태 $\ket{+}$에서 $F = 1$입니다. 노이즈 특성은 확률 $p$에 Pauli-X 적용으로, 계산 기저에서 다음과 같은 상태 전환을 유발합니다.

- $\ket{0} \rightarrow (1-p)\ket{0}\bra{0} + p\ket{1}\bra{1}$ (확률 $p$로 비트 반전)
- $\ket{1} \rightarrow p\ket{0}\bra{0} + (1-p)\ket{1}\bra{1}$ (확률 $p$로 비트 반전)
- $\ket{+} \rightarrow \ket{+}$ (중첩 상태는 불변)

### 예제

```python
from qutip import basis
import numpy as np

# 기본 사용법
result = bit_flip_simulation()
print(f"p=0.5일 때 충실도: {result['fidelities'][5]:.6f}")

# 사용자 정의 확률 범위
custom_p = np.linspace(0, 1, 21)
result = bit_flip_simulation(p_values=custom_p.tolist(), verbose=0)

# 다른 초기 상태 사용
psi_one = basis(2, 1)
result = bit_flip_simulation(initial_state=psi_one)

# 조용한 모드 (출력과 그래프 없이)
result = bit_flip_simulation(show_plot=False, verbose=0)
fidelities = result['fidelities']
```

### 이 외의

- `qutip.fidelity` : 충실도 계산 함수
- `qutip.core.states.ket2dm` : 켓 상태를 밀도 행렬로 변환
- `qutip.sigmax` : Pauli-X 연산자
- `qutip.qeye` : 항등 연산자
- `phase_flip_simulation` : 페이즈-플립 노이즈 시뮬레이션 함수

## 함수: 페이즈-플립

이 함수는 초기 양자 상태에 노이즈를 적용하고 울만 충실도(Uhlmann fidelity), 순수도(purity), 폰 노이만 엔트로피(von Neumann entropy)를 계산함으로써, NetFlow 데이터셋과 결합된 AI 훈련 데이터 생성을 지원합니다.

페이즈-플립(위상 반전, phase-flip) 노이즈 채널을 통한 양자 상태의 진화를 시뮬레이션합니다. 이 함수는 크라우스(Kraus) 연산자를 사용하여 페이즈-플립 노이즈를 모델링하고, 초기 양자 상태에 대한 메트릭 변화를 분석합니다. 페이즈-플립 노이즈는 다음과 같은 크라우스 연산자로 표현(모델링, modeling)됩니다.

$$\begin{aligned}
    K_0 &= \sqrt{1-p} \cdot I \\
    K_1 &= \sqrt{p} \cdot \sigma_z
\end{aligned}$$

여기서 $I$는 항등 연산자(identity operator), $\sigma_z$는 Pauli-Z 연산자이며, $p$는 노이즈 확률입니다. 출력 밀도 행렬(density matrix)은 다음과 같이 계산됩니다.

$$
\rho' = \sum_k K_k \rho K_k^\dagger
$$

이 채널은 확률 $p$로 Pauli-Z 연산자를 적용하여 위상 반전을 유발하지만, 밀도 행렬 수준에서 계산 기저($\ket{0}$,$\ket{1}$)는 불변입니다.

### 파라미터

모든 파라미터는 기본값을 가지며, 초기 상태(`initial_state`)는 상태 벡터 또는 밀도 행렬 전달(입력)을 허용합니다.

- `initial_state`: 시뮬레이션할 초기 양자 상태. 기본값은 $\ket{+}$ 상태입니다.
  - **타입**: `qutip.Qobj`
    - $\ket{0}$ 상태: $\text{qutip.basis(2, 0)}$
    - $\ket{1}$ 상태: $\text{qutip.basis(2, 1)}$
    - $\ket{+}$ 상태: $\text{(qutip.basis(2, 0) + qutip.basis(2, 1)).unit()}$
    - $\ket{−}$ 상태: $\text{(qutip.basis(2, 0) - qutip.basis(2, 1)).unit()}$
    - 밀도 행렬 예시: `qutip.ket2dm(qutip.basis(2, 0)`

- `p_values`: 페이즈-플립 확률 값들의 리스트. 각 값은 0과 1 사이여야 하며, 기본값은 `np.linspace(0, 1, 11).tolist()`입니다.
  - **타입**: `float` 리스트

- `show_plot`: 메트릭 vs 확률 그래프를 표시할지 여부. 기본값은 `True`입니다.
  - **타입**: `bool`

- `verbose`: 시뮬레이션 과정의 상세 정보를 출력할지 여부. 기본값은 `1`입니다.
  - **타입**: `int` (0: 없음, 1: 요약, 2: 모든 밀도 행렬 포함)

- `use_jax`: JAX 백엔드 사용 여부. 기본값은 `False`입니다.
  - **타입**: `bool`

### 반환값

반환값은 `dict`이며, 시뮬레이션 결과를 포함한 딕셔너리를 의미합니다.

- `initial_state` : `qutip.Qobj`, 사용된 초기 양자 상태
- `p_values` : `float` 리스트, 시뮬레이션에 사용된 페이즈-플립 확률 값들
- `fidelities` : `float` 리스트, 각 확률에 대응하는 충실도 값들 ($0 \le \text{fidelity} \le 1$)
- `purities` : `float` 리스트, 각 확률에 대응하는 순수도 값들 ($0.5 \le \text{purity} \le 1$)
- `entropies` : `float` 리스트, 각 확률에 대응하는 폰 노이만 엔트로피 값들 ($0 \le \text{entropy} \le 1$)
- `density_matrices` : `qutip.Qobj` 리스트, 각 확률에 대해 노이즈가 적용된 후의 밀도 행렬들

### 노트

충실도 $F$는 두 양자 상태 간의 유사성을 측정하는 지표입니다. 다음과 같은 수식으로 표현됩니다.

$$
F(\rho, \sigma) = \left[\text{Tr} \sqrt{\sqrt{\rho}\ \sigma \sqrt{\rho}}\right]^2
$$

순수도 $\gamma$는 $\gamma = \mathrm{Tr}(\rho^2)$로 계산되며, 폰 노이만 엔트로피 $S$는 $S(\rho) = -\mathrm{Tr}(\rho \log_2 \rho)$입니다. $p = 0$일 때 $F = 1$(완전히 동일), $p = 1$일 때 상태 의존적입니다. 그리고 계산 기저에서 $F = 1$, 중첩 상태 $\ket{+}$에서 $F = |1-2p|$입니다. 노이즈 특성은 확률 $p$에 Pauli-Z 적용으로, 전역(광역, global)위상 변화로 밀도 행렬은 불변하지만 중첩 상태에서 다음과 같이 위상 변화를 유발합니다.

- $\ket{0} \rightarrow \ket{0}$ (변화 없음)
- $\ket{1} \rightarrow -\ket{1}$ (확률 $p$로 위상 반전, 밀도 행렬 불변)
- $\ket{+} \rightarrow (1 - p)\ket{+}\bra{+} + p\ket{-}\bra{-}$

### 예제

```python
from qutip import basis
import numpy as np

# 기본 사용법
result = phase_flip_simulation()
print(f"p=0.5일 때 충실도: {result['fidelities'][5]:.6f}")

# 사용자 정의 확률 범위
custom_p = np.linspace(0, 1, 21)
result = phase_flip_simulation(p_values=custom_p.tolist(), verbose=0)

# 다른 초기 상태 사용
psi_zero = basis(2, 0)
result = phase_flip_simulation(initial_state=psi_zero)

# 조용한 모드 (출력과 그래프 없이)
result = phase_flip_simulation(show_plot=False, verbose=0)
fidelities = result['fidelities']
```

### 이 외의

- `qutip.fidelity` : 충실도 계산 함수
- `qutip.core.states.ket2dm` : 켓 상태를 밀도 행렬로 변환
- `qutip.sigmaz` : Pauli-Z 연산자
- `qutip.qeye` : 항등 연산자
- `bit_flip_simulation` : 비트-플립 노이즈 시뮬레이션 함수
