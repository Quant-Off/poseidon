"""
섀넌 엔트로피 함수 테스트 파일입니다.

이 파일은 섀넌 엔트로피 함수를 테스트합니다.
"""

import numpy as np
from poseidon.util.von_neumann import validate_rho, entropy_vn_jitted


print("폰 노이만 엔트로피 계산 예시")
print("=" * 30)

# # 완전히 혼합된 상태 (최대 엔트로피)
max_entropy_matrix = np.array([[0.5, 0], [0, 0.5]])
max_entropy_matrix_valid, trace_val = validate_rho(max_entropy_matrix)
entropy = entropy_vn_jitted(max_entropy_matrix_valid, trace_val)
print(f"완전 혼합 상태 행렬 표현:\n{max_entropy_matrix}")
print(f"완전 혼합 상태 엔트로피: {entropy:.4f} bits")
print()

# # 순수 상태 (최소 엔트로피)
pure_state_matrix = np.array([[1, 0], [0, 0]])
pure_state_matrix_valid, trace_val = validate_rho(pure_state_matrix)
entropy = entropy_vn_jitted(pure_state_matrix_valid, trace_val)
print(f"순수 상태 행렬 표현:\n{pure_state_matrix}")
print(f"순수 상태 엔트로피: {entropy:.4f} bits")
print()

# # 부분적으로 혼합된 상태
mixed_state_matrix = np.array([[0.8, 0.1], [0.1, 0.2]])
mixed_state_matrix_valid, trace_val = validate_rho(mixed_state_matrix)
entropy = entropy_vn_jitted(mixed_state_matrix_valid, trace_val)
print(f"부분적으로 혼합된 상태 행렬 표현:\n{mixed_state_matrix}")
print(f"부분적으로 혼합된 상태 엔트로피: {entropy:.4f} bits")
print()

# # 다른 밑수로 계산 (nats)
nats_entropy_matrix = np.array([[0.5, 0], [0, 0.5]])
nats_entropy_matrix_valid, trace_val = validate_rho(nats_entropy_matrix)
entropy = entropy_vn_jitted(nats_entropy_matrix_valid, trace_val, base=np.e)
print(f"다른 밑수로 계산 행렬 표현:\n{nats_entropy_matrix}")
print(f"다른 밑수로 계산 (nats): {entropy:.4f} nats")
print()
