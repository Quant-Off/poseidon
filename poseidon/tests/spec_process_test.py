import numpy as np
from tqdm import tqdm

from poseidon.data.dataset_type import DatasetType
from poseidon.data.poseidon_dtypes import dtypes
from poseidon.processpiece.oversampling import Oversampling


def spec_test():
    # 각 작업 단계를 tqdm으로 진행률 표시 (총 8단계)
    with tqdm(total=8, desc="전체 진행률", unit="작업", position=0, leave=True, dynamic_ncols=True) as pbar:
        # 1. Oversampling 객체 생성
        pbar.set_description("데이터셋 로드")
        ovs = Oversampling(dataset=DatasetType.CUSTOM, test_set_filename="25000s-NF-custom-dataset-1762267172")
        tqdm.write("데이터셋 로드 완료")
        pbar.update(1)

        # 2. 데이터셋 청크 로드
        pbar.set_description("데이터셋 청크 로드")
        df = ovs.load_chunks(
            file_format='csv',
            blocksize='256MB', dtypes=dtypes, npartitions=2)
        tqdm.write("데이터셋 로드 완료")
        pbar.update(1)

        # 3. 오류값 정정
        pbar.set_description("오류값 정정")
        df = ovs.replace_and_drop(df, ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"])
        tqdm.write("오류값 정정 완료")
        pbar.update(1)

        # 4. 컬럼 클리핑
        pbar.set_description("컬럼 클리핑")
        quantiles = df.select_dtypes(include=["float64"]).quantile(0.99).compute()
        df = ovs.column_cliping(df, quantiles)
        tqdm.write("컬럼 클리핑 완료")
        pbar.update(1)

        # 5. X, y 분리
        pbar.set_description("X, y 분리")
        X, y = ovs.shuffle_and_split(df)
        tqdm.write("X, y 분리 완료")
        origin_y_unique, origin_y_counts = np.unique(y, return_counts=True)
        for cls, cnt in zip(origin_y_unique, origin_y_counts):
            tqdm.write(f"  'Label' 클래스 {cls} 개수: {cnt}")
        pbar.update(1)

        # 6. SMOTE 적용
        pbar.set_description("SMOTE 적용")
        X_res, y_res = ovs.smote(origin_y_unique, X, y)
        pbar.update(1)

        # 7. 리샘플링된 데이터셋 재정의
        pbar.set_description("리샘플링된 데이터셋 재정의")
        df = ovs.get_resampled_df(X, X_res, y_res, npartitions=2)
        tqdm.write("리샘플링 후 데이터셋 출력:")
        tqdm.write(str(df))
        pbar.update(1)

        # 8. 로컬 저장
        pbar.set_description("로컬 저장")
        ovs.save_local(df, use_pandas_output=True)  # 저장
        pbar.update(1)


if __name__ == "__main__":
    spec_test()