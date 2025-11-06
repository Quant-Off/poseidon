import os
from poseidon.processpiece.oversampling import Oversampling
from poseidon.data.poseidon_dtypes import dtypes
from dotenv import load_dotenv
import numpy as np
import dask.dataframe as dd

load_dotenv(verbose=True)
DATASETS_CUSTOM_PATH = os.getenv("DATASETS_CUSTOM_PATH")


def oversampling_test():
    ovs = Oversampling(
        f"{DATASETS_CUSTOM_PATH}/10000s-NF-custom-dataset-1762341664.csv"
    )

    df: dd.DataFrame = ovs.load_chunks(
        dtypes=dtypes, blocksize="126MB", npartitions=1
    )  # Dask DataFrame

    df = ovs.replace_and_drop(
        df, columns=["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"]
    )  # Dask DataFrame
    print("=" * 100)
    print(f"오류값 정정 후 타입: {type(df)}")

    quantiles = df.select_dtypes(include=["float64"]).quantile(0.99).compute()
    print("=" * 100)
    print(f"분위수 타입: {type(quantiles)}")
    print(f"분위수: {quantiles}")

    df = ovs.column_cliping(df, quantiles=quantiles)
    print("=" * 100)
    print(f"클리핑 후 타입: {type(df)}")

    X, y = ovs.shuffle_and_split(df)
    print("=" * 100)
    print(f"X 타입: {type(X)}")
    print(f"y 타입: {type(y)}")

    origin_y_unique, _origin_y_counts = np.unique(y, return_counts=True)
    X_res, y_res = ovs.smote(origin_y_unique, X, y)  # Dask DataFrame
    print("=" * 100)
    print(f"X_res 타입: {type(X_res)}")  # numpy.ndarray
    print(f"y_res 타입: {type(y_res)}")  # numpy.ndarray
    df = ovs.get_resampled_df(X, X_res, y_res, npartitions=1)  # Dask DataFrame
    print(f"리샘플링 후 타입: {type(df)}")  # Dask DataFrame

    print("=" * 100)
    print("저장 중...")
    ovs.save_local(df, use_pandas_output=True)
    print("저장 완료")


if __name__ == "__main__":
    oversampling_test()

# OK
