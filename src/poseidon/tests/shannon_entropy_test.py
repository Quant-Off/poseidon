"""
섀넌 엔트로피 함수 테스트 파일입니다.

이 파일은 섀넌 엔트로피 함수를 테스트합니다.
"""

import numpy as np
from poseidon.util.shannon import entropy_sn


print("섀넌 엔트로피 계산 예시")
print("=" * 30)

# 완전히 무작위적인 데이터
random_packet = np.random.randint(0, 256, size=1024)
random_entropy = entropy_sn(random_packet)
print(f"선정된 랜덤 데이터: {random_packet} / 사이즈: {len(random_packet)} bytes")
print(f"랜덤 데이터 엔트로피: {random_entropy:.4f} bits")

# 반복적인 데이터
repeated_packet = np.array([1] * 100)
repeated_entropy = entropy_sn(repeated_packet)
print(f"반복 데이터 엔트로피: {repeated_entropy:.4f} bits")

# 텍스트 데이터 (바이트 문자열)
text_data = b"Hello, World!"
text_entropy = entropy_sn(text_data)
print(f"텍스트 데이터 엔트로피: {text_entropy:.4f} bits")

# 리스트 형태의 데이터
list_data = [1, 2, 3, 1, 2, 3, 1, 2, 3]
list_entropy = entropy_sn(list_data)
print(f"리스트 데이터 엔트로피: {list_entropy:.4f} bits")
