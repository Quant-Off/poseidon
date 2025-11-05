from poseidon.data.poseidon_dtypes import dtypes
from poseidon.processpiece.load_dask_dataframe import load_large_dataset

if __name__ == '__main__':
    df = load_large_dataset(
        '..custom/1000000s-NF-custom-dataset-1762241370.csv',
        file_format='csv',
        blocksize='256MB', dtypes=dtypes, npartitions=2)
    print(df)  # Dask DataFrame 출력 (지연 객체)

# OK
