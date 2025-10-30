import os
import sys
# 로컬 모듈 호출 (이 파일 기준: ../data 하위 모듈 호출)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 상위 디렉터리 경로를 sys.path에 추가 (../)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import data
from data.dataset import read_dataset


datasets = [
    "NF-BoT-IoT-v3",
    "NF-CICIDS2018-v3",
    "NF-ToN-IoT-v3",
    "NF-UNSW-NB15-v3",
]
if __name__ == "__main__":
    for dataset in datasets:
        print("=" * 45)
        print(f"{dataset} 데이터셋 정보")
        dask_df = read_dataset(dataset)

        # Label 이외 모든 행 제거
        dask_df = dask_df.drop(columns=[col for col in dask_df.columns if col != "Label"])
        print(dask_df["Label"].value_counts().compute())
        print("=" * 45)
