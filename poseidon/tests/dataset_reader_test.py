from poseidon.data.dataset import load_dataset_concat


chunksize = 500000  # 청크 크기: 시스템 메모리에 따라 조정 (작을수록 안전, 클수록 빠름)
usecols = ["Label", "Attack"] # 필요한 열 목록 (모든 열이 필요하다면 생략)
df = load_dataset_concat("NF-UNSW-NB15-v3", chunksize=chunksize, usecols=usecols, engine="c")

print(df.info())
print(df["Label"].value_counts())
print(df["Attack"].value_counts())  # 다중 클래스 분포 확인 추가
