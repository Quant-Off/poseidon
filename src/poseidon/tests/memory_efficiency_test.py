"""
메모리 효율성 테스트 파일입니다.

이 파일은 메모리 효율성을 테스트합니다.
"""

import time
import numpy as np
from poseidon.util.shannon import entropy_sn
from poseidon.util.von_neumann import entropy_vn
from poseidon.util.timing_variance import timing_variance


print("\n메모리 효율성 테스트")
print("=" * 30)

# 대용량 데이터로 메모리 사용량 테스트
large_size = 10000

# 대용량 섀넌 엔트로피 메모리 테스트
start_time = time.time()
large_packet = np.random.randint(0, 256, size=large_size)
entropy = entropy_sn(large_packet)
entropy_time = time.time() - start_time
print(f"대용량 패킷 엔트로피 (섀넌) ({large_size} bytes): {entropy:.15f} bits")
print(f"계산 시간: {entropy_time:.4f} 초")

# 대용량 밀도 행렬에 대한 폰 노이만 엔트로피 메모리 테스트
start_time = time.time()
vn_large_size = 1000
# 무작위로 복소수 행렬 생성
random_matrix = np.random.rand(vn_large_size, vn_large_size) + 1j * np.random.rand(vn_large_size, vn_large_size)
# rho = A @ A.conj().T는 항상 반정부호 행렬
density_matrix = random_matrix @ random_matrix.conj().T
# 대각합이 1이 되도록 정규화
trace = np.trace(density_matrix)
density_matrix /= trace
vn_entropy = entropy_vn(density_matrix)
vn_entropy_time = time.time() - start_time
print(f"대용량 패킷 엔트로피 (폰 노이만) ({vn_large_size} bytes): {vn_entropy:.15f} bits")
print(f"계산 시간: {vn_entropy_time:.4f} 초")

# 타이밍 변동 메모리 테스트
start_time = time.time()
large_timestamps = np.cumsum(np.random.exponential(1.0, large_size))
variance = timing_variance(large_timestamps)
variance_time = time.time() - start_time
print(f"대용량 타이밍 변동 ({large_size} timestamps): {variance:.4f} 초")
print(f"계산 시간: {variance_time:.4f} 초")
