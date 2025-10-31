import os
from dotenv import load_dotenv
from poseidon.data.dataset import read_dataset
from sklearn.model_selection import train_test_split

load_dotenv(verbose=True)

DATASETS_PATH = os.getenv("DATASETS_PATH")
if not DATASETS_PATH:
    raise ValueError("DATASET_DIR_PATH 환경 변수가 설정되지 않았습니다!")


def dataset_split(dataset_path, random_state=42):
    # 1. X, y 분할
    df = read_dataset(dataset_path)
    X = df.drop(columns=["Label"])
    y = df["Label"]

    # 2. 전체 데이터의 80%를 훈련+검증, 20%를 테스트로 분할
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # 3. 훈력+검증 데이터의 75%를 훈련, 25%를 검증으로 분할(전체 비율: 6:2:2)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=random_state
    )

    # 4. 디버깅 로그
    print("훈련 세트 클래스 비율: ", y_train.value_counts(normalize=True))
    print("검증 세트 클래스 비율: ", y_val.value_counts(normalize=True))
    print("테스트 세트 클래스 비율: ", y_test.value_counts(normalize=True))


datasets = [
    "NF-UNSW-NB15-v3",
    "NF-BoT-IoT-v3",
    "NF-CICIDS2018-v3",
    "NF-ToN-IoT-v3",
]

for dataset in datasets:
    path = os.path.join(DATASETS_PATH, f"{dataset}-smote.csv")
    dataset_split(path)
