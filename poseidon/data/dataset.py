import os
import numpy as np
from dotenv import load_dotenv
import dask.dataframe as dd

load_dotenv(verbose=True)

DATASET_DIR_PATH = os.getenv("DATASET_DIR_PATH")

dtypes = {
    "FLOW_START_MILLISECONDS": "int64",
    "FLOW_END_MILLISECONDS": "int64",
    "IPV4_SRC_ADDR": "object",
    "L4_SRC_PORT": "int64",
    "IPV4_DST_ADDR": "object",
    "L4_DST_PORT": "int64",
    "PROTOCOL": "int64",
    "L7_PROTO": "float64",
    "IN_BYTES": "int64",
    "IN_PKTS": "int64",
    "OUT_BYTES": "int64",
    "OUT_PKTS": "int64",
    "TCP_FLAGS": "int64",
    "CLIENT_TCP_FLAGS": "int64",
    "SERVER_TCP_FLAGS": "int64",
    "FLOW_DURATION_MILLISECONDS": "int64",
    "DURATION_IN": "int64",
    "DURATION_OUT": "int64",
    "MIN_TTL": "int64",
    "MAX_TTL": "int64",
    "LONGEST_FLOW_PKT": "int64",
    "SHORTEST_FLOW_PKT": "int64",
    "MIN_IP_PKT_LEN": "int64",
    "MAX_IP_PKT_LEN": "int64",
    "SRC_TO_DST_SECOND_BYTES": "float64",
    "DST_TO_SRC_SECOND_BYTES": "float64",
    "RETRANSMITTED_IN_BYTES": "int64",
    "RETRANSMITTED_IN_PKTS": "int64",
    "RETRANSMITTED_OUT_BYTES": "int64",
    "RETRANSMITTED_OUT_PKTS": "int64",
    "SRC_TO_DST_AVG_THROUGHPUT": "int64",
    "DST_TO_SRC_AVG_THROUGHPUT": "int64",
    "NUM_PKTS_UP_TO_128_BYTES": "int64",
    "NUM_PKTS_128_TO_256_BYTES": "int64",
    "NUM_PKTS_256_TO_512_BYTES": "int64",
    "NUM_PKTS_512_TO_1024_BYTES": "int64",
    "NUM_PKTS_1024_TO_1514_BYTES": "int64",
    "TCP_WIN_MAX_IN": "int64",
    "TCP_WIN_MAX_OUT": "int64",
    "ICMP_TYPE": "int64",
    "ICMP_IPV4_TYPE": "int64",
    "DNS_QUERY_ID": "int64",
    "DNS_QUERY_TYPE": "int64",
    "DNS_TTL_ANSWER": "int64",
    "FTP_COMMAND_RET_CODE": "int64",
    "SRC_TO_DST_IAT_MIN": "int64",
    "SRC_TO_DST_IAT_MAX": "int64",
    "SRC_TO_DST_IAT_AVG": "int64",
    "SRC_TO_DST_IAT_STDDEV": "int64",
    "DST_TO_SRC_IAT_MIN": "int64",
    "DST_TO_SRC_IAT_MAX": "int64",
    "DST_TO_SRC_IAT_AVG": "int64",
    "DST_TO_SRC_IAT_STDDEV": "int64",
    "Label": "int64",
    "Attack": "object",
}


def read_dataset(dataset_path, chunksize=500000, usecols=None, engine="c"):
    print(f"타겟 데이터셋: {dataset_path}")
    return dd.read_csv(
        dataset_path,
        blocksize=chunksize,
        dtype=dtypes,
        usecols=usecols,
        low_memory=False,
        engine=engine,
    )


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
