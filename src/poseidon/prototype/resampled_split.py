import os
from dotenv import load_dotenv
import dask.dataframe as dd
from poseidon.data.dataset import read_dataset
from dask_ml.model_selection import train_test_split
import numpy as np  # 추가: dtype 변환을 위해

load_dotenv(verbose=True)

DATASETS_PATH = os.getenv("DATASETS_PATH")
if not DATASETS_PATH:
    raise ValueError("DATASET_DIR_PATH 환경 변수가 설정되지 않았습니다!")


def stratified_split(df, test_size=0.2, val_size=0.25, random_state=42):
    # 1. X, y 분할
    X = df.drop(columns=["Label"])
    y = df["Label"]
    df = dd.concat([X, y], axis=1)  # 임시로 합침 (groupby 위해)

    # 2. 클래스별 그룹화: compute 후 astype(int) 적용
    unique_y = y.unique().compute()  # 먼저 compute
    classes = np.unique(unique_y.astype(int))  # float64를 int로 변환 (np.unique 사용)

    train_dfs, val_dfs, test_dfs = [], [], []

    for cls in classes:
        # 클래스 필터링: cls를 float로 비교 (원본 y가 float일 수 있음)
        group = df[df["Label"] == float(cls)]  # float로 캐스트
        X_group = group.drop(columns=["Label"])
        y_group = group["Label"]

        # 훈련+검증과 테스트 분할
        X_train_val_g, X_test_g, y_train_val_g, y_test_g = train_test_split(
            X_group, y_group, test_size=test_size, random_state=random_state
        )

        # 훈련과 검증 분할
        X_train_g, X_val_g, y_train_g, y_val_g = train_test_split(
            X_train_val_g, y_train_val_g, test_size=val_size, random_state=random_state
        )

        train_dfs.append((X_train_g, y_train_g))
        val_dfs.append((X_val_g, y_val_g))
        test_dfs.append((X_test_g, y_test_g))

    # 3. 합치기
    X_train = dd.concat([x for x, _ in train_dfs], axis=0)
    y_train = dd.concat([y for _, y in train_dfs], axis=0)
    X_val = dd.concat([x for x, _ in val_dfs], axis=0)
    y_val = dd.concat([y for _, y in val_dfs], axis=0)
    X_test = dd.concat([x for x, _ in test_dfs], axis=0)
    y_test = dd.concat([y for _, y in test_dfs], axis=0)

    # 4. 디버깅 로그 (compute 필요)
    print("훈련 세트 클래스 비율: ", y_train.value_counts(normalize=True).compute())
    print("검증 세트 클래스 비율: ", y_val.value_counts(normalize=True).compute())
    print("테스트 세트 클래스 비율: ", y_test.value_counts(normalize=True).compute())

    return X_train, X_val, X_test, y_train, y_val, y_test


datasets = [
    "NF-UNSW-NB15-v3",
    "NF-BoT-IoT-v3",
    "NF-CICIDS2018-v3",
    "NF-ToN-IoT-v3",
]

for dataset in datasets:
    path = os.path.join(DATASETS_PATH, f"{dataset}-smote.csv")
    df = read_dataset(path)
    stratified_split(df)
