import os
from poseidon.processpiece.load_dask_dataframe import (
    load_large_dataset,
    switch_to_pandas,
)
from poseidon.data.poseidon_dtypes import dtypes
from dotenv import load_dotenv

load_dotenv(verbose=True)
DATASETS_CUSTOM_PATH = os.getenv("DATASETS_CUSTOM_PATH")


def logging(df):
    print("=" * 100)
    print(df.head())
    print(f"지정 타입: {type(df)}")
    print("=" * 100)


def test_load_large_dask():
    df = load_large_dataset(
        f"{DATASETS_CUSTOM_PATH}/10000s-NF-custom-dataset-1762341664.csv",
        dtypes=dtypes,
        blocksize="126MB",
        npartitions=1,
    )
    logging(df)


def test_load_large_pandas():
    df = load_large_dataset(
        f"{DATASETS_CUSTOM_PATH}/10000s-NF-custom-dataset-1762341664.csv",
        dtypes=dtypes,
        blocksize="126MB",
        npartitions=1,
    )
    df = switch_to_pandas(df)
    logging(df)


if __name__ == "__main__":
    test_load_large_dask()
    test_load_large_pandas()


# OK
