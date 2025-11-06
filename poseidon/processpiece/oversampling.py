"""
원본 데이터셋에서 특정 피처의 불균형을 해결하기 위해 SMOTE 오버샘플링을 적용하는 클래스입니다.
"""

import os

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from poseidon.data.dataset import clip_partition, shuffle_and_split
from poseidon.data.smote_knn import smote
from poseidon.processpiece.load_dask_dataframe import load_large_dataset
from poseidon.log.poseidon_log import PoseidonLogger

load_dotenv(verbose=True)
DATASETS_RESAMPLED_PATH = os.getenv("DATASETS_RESAMPLED_PATH")
if not DATASETS_RESAMPLED_PATH:
    raise ValueError(
        "DATASETS_RESAMPLED_PATH 환경 변수가 설정되지 않았습니다!"
    )


logger = PoseidonLogger().get_logger()


class Oversampling:
    """
    원본 데이터셋에서 특정 피처의 불균형을 해결하기 위해 SMOTE 오버샘플링을 적용하는 클래스입니다.

    self.dataset: 데이터셋의 이름
    self.dataset_path: 선언 시 호출된 데이터셋 최종 경로 (확장자 포함)

    경우에 따라 tqdm 기능이 추가될 수 있습니다.
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset_name = os.path.basename(dataset_path)  # 확장자명 포함

    def load_chunks(
        self, file_format="csv", dtypes=None, blocksize="256MB", npartitions=20
    ):
        """
        할당된 청크 수에 따라 데이터를 읽고 데이터프레임을 반환합니다.
        :param chunk_size: 청크 사이즈 (기본값: 500000)
        :param use_columns: 읽어들일 컬럼(행, columns) (기본값: None)
        :param engine: 읽기에 사용하는 엔진 (기본값: "c")
        :return: Dask 데이터프레임
        """
        # return read_dataset(self.dataset_path, chunksize=chunk_size, usecols=use_columns, engine=engine)
        return load_large_dataset(
            self.dataset_path,
            file_format=file_format,
            dtypes=dtypes if dtypes is None else dtypes,
            blocksize=blocksize,
            npartitions=npartitions,
        )

    def replace_and_drop(self, dask_df, columns=None):
        # NaN, Inf 처리
        dask_df = dask_df.replace([np.inf, -np.inf], np.nan).dropna()
        # 할당 피처 제거
        if columns is not None:
            dask_df = dask_df.drop(columns=columns)
        return dask_df

    def column_cliping(self, dask_df, quantiles):
        dask_df = dask_df.map_partitions(
            clip_partition, quantiles=quantiles, meta=dask_df
        )
        return dask_df

    def shuffle_and_split(
        self, dask_df, target_col="Label", opt_attack_col="Attack", random_state=42
    ):
        X, y = shuffle_and_split(
            dask_df,
            label_col=target_col,
            attack_col=opt_attack_col,
            random_state=random_state,
        )
        return X, y

    def smote(
        self, origin_y_unique, X, y, k=5, sampling_ratio: float = 1.0, random_state=None
    ):
        try:
            from tqdm import tqdm

            write_fn = tqdm.write
        except ImportError:
            write_fn = print

        if len(origin_y_unique) != 2:
            write_fn("y의 클래스 수가 2개가 아닙니다! SMOTE를 적용할 수 없습니다!")
            X_resampled, y_resampled = X.compute(), y.compute()
        else:
            write_fn("> SMOTE 적용 중...")
            try:
                X_resampled, y_resampled = smote(
                    X.compute(),
                    y.compute(),
                    k=k,
                    sampling_ratio=sampling_ratio,
                    random_state=random_state,
                )
            except ValueError as e:
                write_fn(f"처리 중 오류: '{e}' SMOTE를 스킵합니다!")
                X_resampled, y_resampled = X.compute(), y.compute()
        write_fn("  - SMOTE 적용 완료")
        return X_resampled, y_resampled

    def get_resampled_df(
        self,
        X,
        X_resampled,
        y_resampled,
        npartitions=20,
        sel_features=None,
        chunksize=759300,
    ):
        """
        리샘플링된 Dask 데이터셋을 반환합니다.
        :param X: 분리 작업 후 분리된 X(데이터프레임) 값
        :param X_resampled: 분리된 X값에 SMOTE 오버샘플링 연산을 부여한 값(리샘플링 X 값)
        :param y_resampled: 분리된 y값에 SMOTE 오버샘플링 연산을 부여한 값(리샘플링 y 값)
        :param npartitions: 사용할 CPU(또는 vCPU) 파티션 수 (기본값: 20)
        :param sel_features: 리샘플링 y값을 부여할 피처(또는 행, column) (기본값: 'Label')
        :param chunksize: 읽어들일 청크 사이즈 (기본값: 759300)
        :return: 리샘플링된 Dask 데이터셋 resampled_df
        """
        resampled_df = dd.from_pandas(
            pd.DataFrame(X_resampled, columns=X.columns), npartitions=npartitions
        )
        if sel_features is not None:
            resampled_df[sel_features] = dd.from_array(y_resampled, chunksize=chunksize)
        else:
            resampled_df["Label"] = dd.from_array(y_resampled, chunksize=chunksize)
        return resampled_df

    def save_local(
        self,
        resampled_df,
        output_path: str = DATASETS_RESAMPLED_PATH,
        use_smote_suffix=True,
        use_pandas_output=True,
        index=False,
    ):
        if use_smote_suffix:
            # 확장자명은 이미 self.dataset_name에 포함되어 있기 때문에, 뒤에서부터 "."을 읽고 확장자 앞 부분에 "-smote" 추가
            save_filename = (
                self.dataset_name.rsplit(".", 1)[0]
                + "-smote."
                + self.dataset_name.rsplit(".", 1)[1]
            )
        else:
            save_filename = self.dataset_name

        if use_pandas_output:
            # Dask DataFrame을 단일 CSV 파일로 저장하기 위해 compute() 사용
            logger.info(
                "Pandas 데이터프레임 저장 중 ... (지정 경로: %s)",
                os.path.join(output_path, save_filename),
            )
            if isinstance(resampled_df, pd.DataFrame):
                resampled_df.to_csv(
                    os.path.join(output_path, save_filename), index=index
                )
            elif isinstance(resampled_df, dd.DataFrame):
                resampled_df.compute().to_csv(
                    os.path.join(output_path, save_filename), index=index
                )
            else:
                raise ValueError(
                    f"'{type(resampled_df)}'은(는) 지원되지 않는 데이터 타입입니다."
                )
            logger.info("  - Pandas 데이터프레임 저장 완료")
        else:
            # Dask DataFrame을 파티션별로 저장
            logger.info(
                "Dask 데이터프레임 저장 중 ... (지정 경로: %s)",
                os.path.join(output_path, save_filename),
            )
            resampled_df.to_csv(os.path.join(output_path, save_filename), index=index)
            logger.info("  - Dask 데이터프레임 저장 완료")
