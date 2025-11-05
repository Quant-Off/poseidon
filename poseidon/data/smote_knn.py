"""
SMOTE (Synthetic Minority Over-sampling Technique) 및 KNN (K-Nearest Neighbors) 구현 모듈

이 모듈은 불균형 데이터셋에서 소수 클래스의 합성 샘플을 생성하기 위한 SMOTE 알고리즘과
그에 필요한 KNN 계산 함수들을 제공합니다.

주요 기능:
- KNN 계산: compute_knn, find_knn_for_sample, compute_pairwise_distances
- SMOTE 구현: smote, smote_with_custom_knn
- 이진 분류 문제에서 클래스 불균형 해결
- 다양한 거리 측정 방법 지원 (유클리드, 맨하탄 등)
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm


def compute_knn(X, k=5, metric="euclidean"):
    """
    K-최근접 이웃(K-Nearest Neighbors)을 계산하는 함수

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        입력 데이터
    k : int, default=5
        고려할 최근접 이웃의 수
    metric : str, default='euclidean'
        거리 측정 방법 ('euclidean', 'manhattan', 'minkowski' 등)

    Returns:
    --------
    distances : array, shape (n_samples, k)
        각 샘플에서 k개의 최근접 이웃까지의 거리
    indices : array, shape (n_samples, k)
        각 샘플의 k개의 최근접 이웃 인덱스
    """
    if X.shape[0] <= k:
        raise ValueError(f"샘플 수({X.shape[0]})가 k({k})보다 작거나 같습니다!")

    # NearestNeighbors 모델 생성 및 학습
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)  # +1은 자기 자신 포함
    nn.fit(X)

    # 각 점에서 k+1개의 최근접 이웃 찾기 (자기 자신 포함)
    distances, indices = nn.kneighbors(X)

    # 자기 자신 제거 (첫 번째 열이 자기 자신)
    distances = distances[:, 1:]  # 거리에서 자기 자신 제거
    indices = indices[:, 1:]  # 인덱스에서 자기 자신 제거

    return distances, indices


def find_knn_for_sample(X, sample_idx, k=5, metric="euclidean"):
    """
    특정 샘플에 대한 K-최근접 이웃을 찾는 함수

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        입력 데이터
    sample_idx : int
        KNN을 찾을 샘플의 인덱스
    k : int, default=5
        고려할 최근접 이웃의 수
    metric : str, default='euclidean'
        거리 측정 방법

    Returns:
    --------
    distances : array, shape (k,)
        해당 샘플에서 k개의 최근접 이웃까지의 거리
    indices : array, shape (k,)
        해당 샘플의 k개의 최근접 이웃 인덱스
    """
    if sample_idx >= X.shape[0]:
        raise ValueError(
            f"샘플 인덱스({sample_idx})가 데이터 크기({X.shape[0]})를 초과합니다!"
        )

    if X.shape[0] <= k:
        raise ValueError(f"샘플 수({X.shape[0]})가 k({k})보다 작거나 같습니다!")

    # NearestNeighbors 모델 생성 및 학습
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(X)

    # 특정 샘플에 대한 최근접 이웃 찾기
    distances, indices = nn.kneighbors([X[sample_idx]])

    # 자기 자신 제거
    distances = distances[0, 1:]  # 첫 번째 행, 자기 자신 제외
    indices = indices[0, 1:]  # 첫 번째 행, 자기 자신 제외

    return distances, indices


def compute_pairwise_distances(X, metric="euclidean"):
    """
    모든 샘플 쌍 간의 거리를 계산하는 함수

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        입력 데이터
    metric : str, default='euclidean'
        거리 측정 방법

    Returns:
    --------
    distances : array, shape (n_samples, n_samples)
        모든 샘플 쌍 간의 거리 행렬
    """

    return pairwise_distances(X, metric=metric)


def smote(X, y, k=5, sampling_ratio=1.0, random_state=None):
    """
    SMOTE (Synthetic Minority Over-sampling Technique) 알고리즘을 구현하는 함수

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        입력 특성 데이터
    y : array-like, shape (n_samples,)
        타겟 레이블
    k : int, default=5
        SMOTE에서 사용할 최근접 이웃의 수
    sampling_ratio : float, default=1.0
        소수 클래스에 생성할 합성 샘플의 비율 (1.0 = 100%)
    random_state : int, optional
        재현 가능한 결과를 위한 랜덤 시드

    Returns:
    --------
    X_resampled : array, shape (n_samples + n_synthetic, n_features)
        오버샘플링된 특성 데이터
    y_resampled : array, shape (n_samples + n_synthetic,)
        오버샘플링된 타겟 레이블
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = np.array(X)
    y = np.array(y)

    # 클래스별 샘플 수 계산
    unique_classes, class_counts = np.unique(y, return_counts=True)

    if len(unique_classes) != 2:
        raise ValueError("SMOTE는 이진 분류 문제에만 적용 가능합니다!")

    # 소수 클래스 식별
    minority_class = unique_classes[np.argmin(class_counts)]

    # 소수 클래스 샘플 추출
    minority_indices = np.where(y == minority_class)[0]
    minority_X = X[minority_indices]

    # 생성할 합성 샘플 수 계산
    n_minority = len(minority_indices)
    n_majority = len(X) - n_minority
    n_synthetic = int((n_majority - n_minority) * sampling_ratio)

    if n_synthetic <= 0:
        return X, y

    # 소수 클래스에 대한 KNN 계산
    _, indices = compute_knn(minority_X, k=k)

    # 합성 샘플 생성
    synthetic_samples = []
    synthetic_labels = []

    for _ in tqdm(range(n_synthetic), desc="SMOTE 합성 샘플 생성", unit="샘플"):
        # 랜덤하게 소수 클래스 샘플 선택
        sample_idx = np.random.randint(0, n_minority)

        # 해당 샘플의 k개 최근접 이웃 중 하나를 랜덤 선택
        neighbor_idx = np.random.randint(0, k)
        neighbor_sample_idx = indices[sample_idx, neighbor_idx]

        # 합성 샘플 생성 (선형 보간)
        alpha = np.random.random()  # 0과 1 사이의 랜덤 값
        synthetic_sample = minority_X[sample_idx] + alpha * (
            minority_X[neighbor_sample_idx] - minority_X[sample_idx]
        )

        synthetic_samples.append(synthetic_sample)
        synthetic_labels.append(minority_class)

    # 원본 데이터와 합성 샘플 결합
    X_resampled = np.vstack([X, np.array(synthetic_samples)])
    y_resampled = np.hstack([y, np.array(synthetic_labels)])

    return X_resampled, y_resampled


def smote_with_custom_knn(
    X, y, k=5, sampling_ratio=1.0, random_state=None, metric="euclidean"
):
    """
    사용자 정의 KNN 함수를 사용하는 SMOTE 구현

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        입력 특성 데이터
    y : array-like, shape (n_samples,)
        타겟 레이블
    k : int, default=5
        SMOTE에서 사용할 최근접 이웃의 수
    sampling_ratio : float, default=1.0
        소수 클래스에 생성할 합성 샘플의 비율
    random_state : int, optional
        재현 가능한 결과를 위한 랜덤 시드
    metric : str, default="euclidean"
        거리 측정 방법

    Returns:
    --------
    X_resampled : array, shape (n_samples + n_synthetic, n_features)
        오버샘플링된 특성 데이터
    y_resampled : array, shape (n_samples + n_synthetic,)
        오버샘플링된 타겟 레이블
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = np.array(X)
    y = np.array(y)

    # 클래스별 샘플 수 계산
    unique_classes, class_counts = np.unique(y, return_counts=True)

    if len(unique_classes) != 2:
        raise ValueError("SMOTE는 이진 분류 문제에만 적용 가능합니다!")

    # 소수 클래스 식별
    minority_class = unique_classes[np.argmin(class_counts)]

    # 소수 클래스 샘플 추출
    minority_indices = np.where(y == minority_class)[0]
    minority_X = X[minority_indices]

    # 생성할 합성 샘플 수 계산
    n_minority = len(minority_indices)
    n_majority = len(X) - n_minority
    n_synthetic = int((n_majority - n_minority) * sampling_ratio)

    if n_synthetic <= 0:
        return X, y

    # 소수 클래스에 대한 KNN 계산 (사용자 정의 함수 사용)
    _, indices = compute_knn(minority_X, k=k, metric=metric)

    # 합성 샘플 생성
    synthetic_samples = []
    synthetic_labels = []

    for _ in tqdm(range(n_synthetic), desc="SMOTE 합성 샘플 생성", unit="샘플"):
        # 랜덤하게 소수 클래스 샘플 선택
        sample_idx = np.random.randint(0, n_minority)

        # 해당 샘플의 k개 최근접 이웃 중 하나를 랜덤 선택
        neighbor_idx = np.random.randint(0, k)
        neighbor_sample_idx = indices[sample_idx, neighbor_idx]

        # 합성 샘플 생성 (선형 보간)
        alpha = np.random.random()  # 0과 1 사이의 랜덤 값
        synthetic_sample = minority_X[sample_idx] + alpha * (
            minority_X[neighbor_sample_idx] - minority_X[sample_idx]
        )

        synthetic_samples.append(synthetic_sample)
        synthetic_labels.append(minority_class)

    # 원본 데이터와 합성 샘플 결합
    X_resampled = np.vstack([X, np.array(synthetic_samples)])
    y_resampled = np.hstack([y, np.array(synthetic_labels)])

    return X_resampled, y_resampled

__all__ = [
    'compute_knn',
    'smote',
]
