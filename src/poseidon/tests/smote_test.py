"""
SMOTE 테스트
"""

import numpy as np
from poseidon.data.smote_knn import smote, smote_with_custom_knn

print("=" * 60)
print("SMOTE 함수 테스트 시작")
print("=" * 60)

# 불균형 데이터 생성
print("\n1. 불균형 테스트 데이터 생성")
np.random.seed(42)

# 다수 클래스 (클래스 0): 100개 샘플
X_majority = np.random.rand(100, 3)
y_majority = np.zeros(100)

# 소수 클래스 (클래스 1): 20개 샘플
X_minority = np.random.rand(20, 3) + 2  # 다른 영역에 배치
y_minority = np.ones(20)

# 데이터 결합
X = np.vstack([X_majority, X_minority])
y = np.hstack([y_majority, y_minority])

print(f"원본 데이터 형태: {X.shape}")
print(f"클래스 0 (다수): {np.sum(y == 0)}개")
print(f"클래스 1 (소수): {np.sum(y == 1)}개")
print(f"불균형 비율: {np.sum(y == 0) / np.sum(y == 1):.1f}:1")

# 기본 SMOTE 테스트
print("\n2. 기본 SMOTE 함수 테스트")
print("-" * 40)
try:
    X_resampled, y_resampled = smote(X, y, k=5, sampling_ratio=1.0, random_state=42)

    print(f"오버샘플링 후 데이터 형태: {X_resampled.shape}")
    print(f"클래스 0 (다수): {np.sum(y_resampled == 0)}개")
    print(f"클래스 1 (소수): {np.sum(y_resampled == 1)}개")
    print(f"새로 생성된 샘플 수: {len(y_resampled) - len(y)}개")
    print(
        f"새로운 불균형 비율: {np.sum(y_resampled == 0) / np.sum(y_resampled == 1):.1f}:1"
    )
    print("✅ 기본 SMOTE 테스트 성공")
except Exception as e:
    print(f"❌ 기본 SMOTE 테스트 실패: {e}")

# 사용자 정의 KNN SMOTE 테스트
print("\n3. 사용자 정의 KNN SMOTE 함수 테스트")
print("-" * 40)
try:
    X_resampled_custom, y_resampled_custom = smote_with_custom_knn(
        X, y, k=3, sampling_ratio=0.5, random_state=42, metric="manhattan"
    )

    print(f"오버샘플링 후 데이터 형태: {X_resampled_custom.shape}")
    print(f"클래스 0 (다수): {np.sum(y_resampled_custom == 0)}개")
    print(f"클래스 1 (소수): {np.sum(y_resampled_custom == 1)}개")
    print(f"새로 생성된 샘플 수: {len(y_resampled_custom) - len(y)}개")
    print(
        f"새로운 불균형 비율: {np.sum(y_resampled_custom == 0) / np.sum(y_resampled_custom == 1):.1f}:1"
    )
    print("✅ 사용자 정의 KNN SMOTE 테스트 성공")
except Exception as e:
    print(f"❌ 사용자 정의 KNN SMOTE 테스트 실패: {e}")

# 다양한 sampling_ratio 테스트
print("\n4. 다양한 sampling_ratio 테스트")
print("-" * 40)
ratios = [0.2, 0.5, 1.0, 1.5]
for ratio in ratios:
    try:
        X_ratio, y_ratio = smote(X, y, k=5, sampling_ratio=ratio, random_state=42)
        new_samples = len(y_ratio) - len(y)
        print(f"sampling_ratio={ratio}: {new_samples}개 샘플 생성")
    except Exception as e:
        print(f"sampling_ratio={ratio}: 실패 - {e}")

# 다양한 k 값 테스트
print("\n5. 다양한 k 값 테스트")
print("-" * 40)
k_values = [3, 5, 7, 10]
for k in k_values:
    try:
        X_k, y_k = smote(X, y, k=k, sampling_ratio=0.5, random_state=42)
        print(f"k={k}: 성공")
    except Exception as e:
        print(f"k={k}: 실패 - {e}")

# 에러 케이스 테스트
print("\n6. 에러 케이스 테스트")
print("-" * 40)

# 다중 클래스 데이터 (3개 클래스)
print("6.1 다중 클래스 데이터:")
try:
    X_multi = np.random.rand(30, 3)
    y_multi = np.array([0] * 10 + [1] * 10 + [2] * 10)  # 3개 클래스
    smote(X_multi, y_multi)
    print("❌ 에러가 발생하지 않음 (예상과 다름)")
except ValueError as e:
    print(f"✅ 예상된 에러 발생: {e}")

# 균형 데이터 (이미 균형잡힌 경우)
print("\n6.2 균형 데이터:")
try:
    X_balanced = np.random.rand(20, 3)
    y_balanced = np.array([0] * 10 + [1] * 10)  # 균형잡힌 데이터
    X_bal_res, y_bal_res = smote(X_balanced, y_balanced, sampling_ratio=1.0)
    print(f"균형 데이터 처리: {len(y_bal_res)}개 샘플 (원본: {len(y_balanced)}개)")
except Exception as e:
    print(f"균형 데이터 처리 실패: {e}")

# 다양한 거리 측정 방법 테스트
print("\n7. 다양한 거리 측정 방법 테스트")
print("-" * 40)
metrics = ["euclidean", "manhattan"]
for metric in metrics:
    try:
        X_metric, y_metric = smote_with_custom_knn(
            X, y, k=5, sampling_ratio=0.3, random_state=42, metric=metric
        )
        print(f"✅ {metric} 거리 측정 성공")
    except Exception as e:
        print(f"❌ {metric} 거리 측정 실패: {e}")

# 합성 샘플 품질 확인
print("\n8. 합성 샘플 품질 확인")
print("-" * 40)
try:
    X_quality, y_quality = smote(X, y, k=5, sampling_ratio=1.0, random_state=42)

    # 원본 소수 클래스와 합성 샘플 비교
    original_minority = X[y == 1]
    synthetic_samples = X_quality[y_quality == 1][len(original_minority) :]

    print(f"원본 소수 클래스 평균: {np.mean(original_minority, axis=0)}")
    print(f"합성 샘플 평균: {np.mean(synthetic_samples, axis=0)}")
    print(f"합성 샘플 수: {len(synthetic_samples)}개")
    print("✅ 합성 샘플 품질 확인 완료")
except Exception as e:
    print(f"❌ 합성 샘플 품질 확인 실패: {e}")

print("\n" + "=" * 60)
print("SMOTE 함수 테스트 완료")
print("=" * 60)
