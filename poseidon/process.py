import numpy as np
import dask.dataframe as dd
from poseidon.data.dataset_type import DatasetType
from poseidon.processpiece.oversampling import Oversampling
from poseidon.data.poseidon_dtypes import dtypes
from poseidon.processpiece.engineering_split import DatasetSplit
from poseidon.processpiece.engineering_scaling import DatasetScaling
from poseidon.processpiece.feature_calculate import (
    apply_entropy,
    apply_timing_variance,
    apply_quantum_noise_simulation,
)
from poseidon.log.poseidon_log import PoseidonLogger

from tqdm import tqdm

logger = PoseidonLogger().get_logger()


def process():
    req_smote = False

    save_train = True
    save_val = False
    save_test = False
    logger.info("데이터셋 처리 시작")
    resampled_df, ovs = process_oversampling(
        dataset=DatasetType.NF_UNSW_NB15_V3, req_smote=req_smote
    )
    logger.info("  - 데이터셋 처리 완료")

    logger.info("데이터셋 훈련, 검증, 테스트 분할 시작")
    (
        splited_X_train,
        splited_X_val,
        splited_X_test,
        _,
        _,
        _,
    ) = DatasetSplit(resampled_df).split(npartitions=20)
    logger.info("  - 데이터셋 훈련, 검증, 테스트 분할 완료")

    logger.info("데이터셋 스케일링 시작")
    X_train, X_val, X_test = DatasetScaling().scale(
        splited_X_train, splited_X_val, splited_X_test
    )
    logger.info("  - 데이터셋 스케일링 완료")

    logger.info("섀넌 엔트로피 계산 시작")
    X_train, X_val, X_test = cal_shannon_entropy(X_train, X_val, X_test)
    logger.info("  - 섀넌 엔트로피 계산 완료")

    logger.info("타이밍 변동 계산 시작")
    X_train, X_val, X_test = cal_timing_variance(X_train, X_val, X_test)
    logger.info("  - 타이밍 변동 계산 완료")

    logger.info("양자 노이즈 시뮬레이션 계산 시작")
    X_train, X_val, X_test = cal_quantum_noise_simulation(X_train, X_val, X_test)
    logger.info("  - 양자 노이즈 시뮬레이션 계산 완료")

    # 파일 저장
    if save_train:
        logger.info("훈련 세트 저장 시작")
        ovs.save_local(X_train, use_pandas_output=False)
        logger.info("  - 훈련 세트 저장 완료")
    if save_val:
        logger.info("검증 세트 저장 시작")
        ovs.save_local(X_val, use_pandas_output=False)
        logger.info("  - 검증 세트 저장 완료")
    if save_test:
        logger.info("테스트 세트 저장 시작")
        ovs.save_local(X_test, use_pandas_output=False)
        logger.info("  - 테스트 세트 저장 완료")
    logger.info("모든 작업 완료")


def process_oversampling(dataset, req_smote, test_set_filename=None):
    # 각 작업 단계를 tqdm으로 진행률 표시 (총 8단계)
    with tqdm(
        total=8,
        desc="전체 진행률",
        unit="작업",
        position=0,
        leave=True,
        dynamic_ncols=True,
    ) as pbar:
        # 1. Oversampling 객체 생성
        pbar.set_description("데이터셋 로드")
        ovs = Oversampling(
            dataset=dataset,
            is_smote_req=req_smote,
            test_set_filename=test_set_filename,
        )
        tqdm.write("데이터셋 로드 완료")
        pbar.update(1)

        # 2. 데이터셋 청크 로드
        pbar.set_description("데이터셋 청크 로드")
        df = ovs.load_chunks(
            file_format="csv", blocksize="256MB", dtypes=dtypes, npartitions=2
        )
        tqdm.write("데이터셋 로드 완료")
        pbar.update(1)

        # 3. 오류값 정정
        pbar.set_description("오류값 정정")
        df = ovs.replace_and_drop(
            df, ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"]
        )
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

        # 8. 반환
        return df, ovs


def cal_shannon_entropy(to_df_X_train, to_df_X_val, to_df_X_test):
    def apply_entropy_dask(row):
        return apply_entropy(row)

    to_df_X_train = dd.from_pandas(to_df_X_train, npartitions=20)
    to_df_X_train["packet_entropy"] = to_df_X_train.apply(
        apply_entropy_dask,
        axis=1,
        meta=("packet_entropy", "f8"),
    )
    to_df_X_train = to_df_X_train.compute()
    to_df_X_train["packet_entropy"] = to_df_X_train["packet_entropy"].apply(
        lambda x: float(x.item()) if hasattr(x, "item") else float(x)
    )

    to_df_X_val = dd.from_pandas(to_df_X_val, npartitions=20)
    to_df_X_val["packet_entropy"] = to_df_X_val.apply(
        apply_entropy_dask,
        axis=1,
        meta=("packet_entropy", "f8"),
    )
    to_df_X_val = to_df_X_val.compute()
    to_df_X_val["packet_entropy"] = to_df_X_val["packet_entropy"].apply(
        lambda x: float(x.item()) if hasattr(x, "item") else float(x)
    )

    to_df_X_test = dd.from_pandas(to_df_X_test, npartitions=20)
    to_df_X_test["packet_entropy"] = to_df_X_test.apply(
        apply_entropy_dask,
        axis=1,
        meta=("packet_entropy", "f8"),
    )
    to_df_X_test = to_df_X_test.compute()
    to_df_X_test["packet_entropy"] = to_df_X_test["packet_entropy"].apply(
        lambda x: float(x.item()) if hasattr(x, "item") else float(x)
    )

    return to_df_X_train, to_df_X_val, to_df_X_test


def cal_timing_variance(to_df_X_train, to_df_X_val, to_df_X_test):
    def apply_timing_variance_dask(row):
        return apply_timing_variance(row)

    to_df_X_train = dd.from_pandas(to_df_X_train, npartitions=20)
    to_df_X_train["timing_variance"] = to_df_X_train.apply(
        apply_timing_variance_dask,
        axis=1,
        meta=("timing_variance", "f8"),
    )
    to_df_X_train = to_df_X_train.compute()
    to_df_X_train["timing_variance"] = to_df_X_train["timing_variance"].apply(
        lambda x: float(x.item()) if hasattr(x, "item") else float(x)
    )

    to_df_X_val = dd.from_pandas(to_df_X_val, npartitions=20)
    to_df_X_val["timing_variance"] = to_df_X_val.apply(
        apply_timing_variance_dask,
        axis=1,
        meta=("timing_variance", "f8"),
    )
    to_df_X_val = to_df_X_val.compute()
    to_df_X_val["timing_variance"] = to_df_X_val["timing_variance"].apply(
        lambda x: float(x.item()) if hasattr(x, "item") else float(x)
    )

    to_df_X_test = dd.from_pandas(to_df_X_test, npartitions=20)
    to_df_X_test["timing_variance"] = to_df_X_test.apply(
        apply_timing_variance_dask,
        axis=1,
        meta=("timing_variance", "f8"),
    )
    to_df_X_test = to_df_X_test.compute()
    to_df_X_test["timing_variance"] = to_df_X_test["timing_variance"].apply(
        lambda x: float(x.item()) if hasattr(x, "item") else float(x)
    )

    return to_df_X_train, to_df_X_val, to_df_X_test


def cal_quantum_noise_simulation(to_df_X_train, to_df_X_val, to_df_X_test):
    """
    양자 노이즈 시뮬레이션 연산을 계산하여 각 데이터프레임에 'quantum_noise_simulation' 피처를 추가합니다.

    Parameters:
    -----------
    to_df_X_train : pd.DataFrame
        훈련 세트 데이터프레임
    to_df_X_val : pd.DataFrame
        검증 세트 데이터프레임
    to_df_X_test : pd.DataFrame
        테스트 세트 데이터프레임

    Returns:
    --------
    to_df_X_train : pd.DataFrame
        양자 노이즈 시뮬레이션 피처가 추가된 훈련 세트
    to_df_X_val : pd.DataFrame
        양자 노이즈 시뮬레이션 피처가 추가된 검증 세트
    to_df_X_test : pd.DataFrame
        양자 노이즈 시뮬레이션 피처가 추가된 테스트 세트
    """

    def apply_quantum_noise_simulation_dask(row):
        return apply_quantum_noise_simulation(row)

    to_df_X_train = dd.from_pandas(to_df_X_train, npartitions=20)
    to_df_X_train["quantum_noise_simulation"] = to_df_X_train.apply(
        apply_quantum_noise_simulation_dask,
        axis=1,
        meta=("quantum_noise_simulation", "f8"),
    )
    to_df_X_train = to_df_X_train.compute()
    to_df_X_train["quantum_noise_simulation"] = to_df_X_train[
        "quantum_noise_simulation"
    ].apply(lambda x: float(x.item()) if hasattr(x, "item") else float(x))

    to_df_X_val = dd.from_pandas(to_df_X_val, npartitions=20)
    to_df_X_val["quantum_noise_simulation"] = to_df_X_val.apply(
        apply_quantum_noise_simulation_dask,
        axis=1,
        meta=("quantum_noise_simulation", "f8"),
    )
    to_df_X_val = to_df_X_val.compute()
    to_df_X_val["quantum_noise_simulation"] = to_df_X_val[
        "quantum_noise_simulation"
    ].apply(lambda x: float(x.item()) if hasattr(x, "item") else float(x))

    to_df_X_test = dd.from_pandas(to_df_X_test, npartitions=20)
    to_df_X_test["quantum_noise_simulation"] = to_df_X_test.apply(
        apply_quantum_noise_simulation_dask,
        axis=1,
        meta=("quantum_noise_simulation", "f8"),
    )
    to_df_X_test = to_df_X_test.compute()
    to_df_X_test["quantum_noise_simulation"] = to_df_X_test[
        "quantum_noise_simulation"
    ].apply(lambda x: float(x.item()) if hasattr(x, "item") else float(x))

    return to_df_X_train, to_df_X_val, to_df_X_test


if __name__ == "__main__":
    process()

__all__ = [ 'process' ]