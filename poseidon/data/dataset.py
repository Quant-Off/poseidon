import os

import numpy as np
from dotenv import load_dotenv

load_dotenv(verbose=True)

DATASET_DIR_PATH = os.getenv("DATASET_DIR_PATH")


def clip_partition(part_df, quantiles):
    float_cols = part_df.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        if col in part_df.columns:
            upper = quantiles[col]
            part_df[col] = np.clip(part_df[col], part_df[col].min(), upper)
    return part_df


def shuffle_and_split(part_df, label_col="Label", attack_col="Attack", random_state=42):
    """
    Dask DataFrame을 셔플하고 X, y로 분리하는 함수입니다.

    Parameters:
    -----------
    df : dask.dataframe.DataFrame
        입력 Dask DataFrame
    label_col : str, default="Label"
        타겟 레이블 컬럼 이름
    attack_col : str, default="Attack"
        선택적 Attack 컬럼 이름 (존재 시 drop)
    random_state : int, default=42
        셔플 재현성을 위한 시드

    Returns:
    --------
    X : dask.dataframe.DataFrame
        특성 행렬 (Label 및 Attack 제외)
    y : dask.dataframe.Series
        타겟 레이블 벡터
    """
    # 셔플: 지연 평가로 메모리 효율 유지
    part_df = part_df.sample(frac=1, random_state=random_state)

    # y 분리: 레이블 추출
    y = part_df[label_col]

    # X 분리: 불필요 컬럼 drop (Attack 존재 시 포함)
    drop_cols = [label_col]
    if attack_col in part_df.columns:
        drop_cols.append(attack_col)
    X = part_df.drop(columns=drop_cols)

    return X, y
