"""
타이밍 변동 함수 테스트 파일입니다.

이 파일은 타이밍 변동 함수를 테스트합니다.
"""

import numpy as np
from poseidon.util.timing_variance import timing_variance


print("일정한 간격의 타임스탬프를 입력(정상 네트워크 트래픽 모델링)")
print("=" * 30)
times = [0.0, 1.0, 2.0, 3.0, 4.0]
std = timing_variance(times)
cv = timing_variance(times, normalize=True)
print(f"일정한 간격의 타임스탬프: {times}")
print(f"타이밍 변동: {std:.4f} 초")
print(f"변동계수: {cv:.4f}")
print("")


print("불규칙한 간격의 타임스탬프를 입력(NetFlow 데이터셋에서의 DDoS 공격 시뮬레이션)")
print("=" * 30)
times = np.cumsum(np.random.uniform(0.1, 1.0, 10))
std = timing_variance(times)
cv = timing_variance(times, normalize=True)
print(f"불규칙한 간격의 타임스탬프: {times}")
print(f"타이밍 변동: {std:.4f} 초")
print(f"변동계수: {cv:.4f}")
print("")


print(
    "정규 분포 노이즈 추가(양자 노이즈 시뮬레이션, mesolve 솔버를 활용한 양자 시뮬레이터에서 유도된 패턴)"
)
print("=" * 30)
base_times = np.linspace(0, 10, 20)
noise = np.random.normal(0, 0.5, 20)  # 양자 노이즈 모방
times = base_times + noise
times = np.sort(times)  # 비감소 보장
std = timing_variance(times)
cv = timing_variance(times, normalize=True)
print(
    f"정규 분포 노이즈 추가(양자 노이즈 시뮬레이션, mesolve 솔버를 활용한 양자 시뮬레이터에서 유도된 패턴): {times}"
)
print(f"타이밍 변동: {std:.4f} 초")
print(f"변동계수: {cv:.4f}")
print("")


print(
    "매우 많은 데이터 입력(대용량 데이터 처리 테스트)"
)
print("=" * 30)
large_size = 10000
large_timestamps = np.cumsum(np.random.exponential(1.0, large_size))
std = timing_variance(large_timestamps)
cv = timing_variance(large_timestamps, normalize=True)
print(f"대용량 타임스탬프: 배열: {large_timestamps} / 사이즈: {len(large_timestamps)}")
print(f"타이밍 변동: {std:.4f} 초")
print(f"변동계수: {cv:.4f}")
