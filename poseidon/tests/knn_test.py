"""
KNN 테스트
"""

import numpy as np

from poseidon.data.smote_knn import (
    compute_knn,
    find_knn_for_sample,
    compute_pairwise_distances,
)


print("=" * 60)
print("KNN 함수 테스트 시작")
print("=" * 60)

# 테스트 데이터 생성
print("\n1. 테스트 데이터 생성")
np.random.seed(42)
X = np.random.rand(10, 3)  # 10개 샘플, 3개 특성
print(f"데이터 형태: {X.shape}")
print(f"첫 3개 샘플:\n{X[:3]}")

# compute_knn 테스트
print("\n2. compute_knn 함수 테스트")
print("-" * 40)
try:
    distances, indices = compute_knn(X, k=3)
    print(f"거리 행렬 형태: {distances.shape}")
    print(f"인덱스 행렬 형태: {indices.shape}")
    print(f"첫 번째 샘플의 3개 최근접 이웃 거리: {distances[0]}")
    print(f"첫 번째 샘플의 3개 최근접 이웃 인덱스: {indices[0]}")
    print("✅ compute_knn 테스트 성공")
except Exception as e:
    print(f"❌ compute_knn 테스트 실패: {e}")

# find_knn_for_sample 테스트
print("\n3. find_knn_for_sample 함수 테스트")
print("-" * 40)
try:
    sample_idx = 0
    distances, indices = find_knn_for_sample(X, sample_idx, k=3)
    print(f"샘플 {sample_idx}의 최근접 이웃 거리: {distances}")
    print(f"샘플 {sample_idx}의 최근접 이웃 인덱스: {indices}")
    print("✅ find_knn_for_sample 테스트 성공")
except Exception as e:
    print(f"❌ find_knn_for_sample 테스트 실패: {e}")

# compute_pairwise_distances 테스트
print("\n4. compute_pairwise_distances 함수 테스트")
print("-" * 40)
try:
    distances_matrix = compute_pairwise_distances(X)
    print(f"거리 행렬 형태: {distances_matrix.shape}")
    print(f"대각선 원소 (자기 자신과의 거리): {np.diag(distances_matrix)}")
    print(f"첫 번째 샘플과 다른 샘플들 간의 거리: {distances_matrix[0]}")
    print("✅ compute_pairwise_distances 테스트 성공")
except Exception as e:
    print(f"❌ compute_pairwise_distances 테스트 실패: {e}")

# 에러 케이스 테스트
print("\n5. 에러 케이스 테스트")
print("-" * 40)

# k가 샘플 수보다 큰 경우
print("5.1 k가 샘플 수보다 큰 경우:")
try:
    compute_knn(X, k=15)  # 샘플 수는 10개인데 k=15
    print("❌ 에러가 발생하지 않음 (예상과 다름)")
except ValueError as e:
    print(f"✅ 예상된 에러 발생: {e}")

# 잘못된 샘플 인덱스
print("\n5.2 잘못된 샘플 인덱스:")
try:
    find_knn_for_sample(X, sample_idx=20, k=3)  # 샘플 수는 10개인데 인덱스 20
    print("❌ 에러가 발생하지 않음 (예상과 다름)")
except ValueError as e:
    print(f"✅ 예상된 에러 발생: {e}")

# 다양한 거리 측정 방법 테스트
print("\n6. 다양한 거리 측정 방법 테스트")
print("-" * 40)

metrics = ["euclidean", "manhattan"]
for metric in metrics:
    try:
        distances, indices = compute_knn(X, k=3, metric=metric)
        print(f"✅ {metric} 거리 측정 성공")
    except Exception as e:
        print(f"❌ {metric} 거리 측정 실패: {e}")

print("\n" + "=" * 60)
print("KNN 함수 테스트 완료")
print("=" * 60)
