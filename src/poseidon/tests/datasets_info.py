from poseidon.data.dataset import read_dataset


datasets = [
    "NF-BoT-IoT-v3",
    "NF-CICIDS2018-v3",
    "NF-ToN-IoT-v3",
    "NF-UNSW-NB15-v3",
]
for dataset in datasets:
    print(f"================================================")
    print(f"{dataset} 데이터셋 정보")
    dask_df = read_dataset(dataset)
    print(dask_df.info())
    print(dask_df["Label"].value_counts())
    print(f"================================================")
