import os

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from poseidon.data.dataset import clip_partition, shuffle_and_split
from poseidon.data.dataset_type import DatasetType
from poseidon.data.smote_knn import smote
from poseidon.processpiece.load_dask_dataframe import load_large_dataset

load_dotenv(verbose=True)
DATASETS_ORIGIN_PATH = os.getenv("DATASETS_ORIGIN_PATH")
DATASETS_RESAMPLED_PATH = os.getenv("DATASETS_RESAMPLED_PATH")
DATASETS_CUSTOM_PATH = os.getenv("DATASETS_CUSTOM_PATH")
if not DATASETS_RESAMPLED_PATH or not DATASETS_ORIGIN_PATH or not DATASETS_CUSTOM_PATH:
    raise ValueError(
        "DATASETS_RESAMPLED_PATH, DATASETS_ORIGIN_PATH 또는 DATASETS_CUSTOM_PATH 환경 변수가 설정되지 않았습니다!"
    )


class Oversampling:
    """
    self.dataset: 데이터셋의 이름
    self.dataset_path: 선언 시 호출된 데이터셋 최종 경로 (확장자 포함)

    경우에 따라 tqdm 기능이 추가될 수 있습니다.
    """

    def __init__(
        self, dataset: DatasetType = None, is_smote_req=True, test_set_filename=None
    ):
        self.dataset = dataset
        self.test_set_filename = test_set_filename
        # 테스트 셋이 아닌 경우
        if dataset != DatasetType.CUSTOM:
            # 오버샘플링이 필요한 경우는 ORIGIN 에서 가져옴 (파일 이름에 변경 없음)
            if is_smote_req:
                self.dataset_path = os.path.join(DATASETS_ORIGIN_PATH, f"{dataset}.csv")
            else:  # 오버샘플링을 생략하는 경우는 RESAMPLED 에서 가져옴 (파일 이름에 "-smote" 추가되어 있음)
                self.dataset_path = os.path.join(
                    DATASETS_RESAMPLED_PATH, f"{dataset}-smote.csv"
                )
        else:  # 데이터셋을 호출한 경우는 CUSTOM 에서 가져옴 (사용자가 파일 이름 지정)
            self.dataset_path = os.path.join(
                DATASETS_CUSTOM_PATH, f"{test_set_filename}.csv"
            )

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
        save_filename=None,
        path=None,
        use_smote_suffix=True,
        use_pandas_output=True,
        index=False,
    ):
        try:
            from tqdm import tqdm

            write_fn = tqdm.write
        except ImportError:
            write_fn = print

        # 파일명 결정: save_filename이 None이면 test_set_filename 또는 dataset 사용
        if save_filename is None:
            if self.dataset == DatasetType.CUSTOM and self.test_set_filename:
                # 확장자 제거 (.csv 등)
                save_filename = os.path.splitext(self.test_set_filename)[0]
            elif self.dataset:
                save_filename = str(self.dataset)
            else:
                save_filename = "resampled_dataset"

        # 파일명에서 경로 구분자 제거 (파일명만 추출)
        save_filename = os.path.basename(save_filename)
        # 확장자 제거 (이미 확장자가 있을 수 있음)
        save_filename = os.path.splitext(save_filename)[0]

        # 최종 파일명 생성
        suffix = "-smote" if use_smote_suffix else ""
        filename = f"{save_filename}{suffix}.csv"

        # 저장 경로 결정
        if path is None:
            # DATASETS_RESAMPLED_PATH 디렉토리가 없으면 생성
            os.makedirs(DATASETS_RESAMPLED_PATH, exist_ok=True)
            save_path = os.path.join(DATASETS_RESAMPLED_PATH, filename)
        else:
            # 지정된 경로가 디렉토리인지 파일인지 확인
            if os.path.isdir(path) or (
                not os.path.exists(path) and not path.endswith(".csv")
            ):
                # 디렉토리인 경우 파일명 추가
                os.makedirs(path, exist_ok=True)
                save_path = os.path.join(path, filename)
            else:
                # 파일 경로인 경우 그대로 사용
                save_path = path

        if use_pandas_output:
            # Dask DataFrame을 단일 CSV 파일로 저장하기 위해 compute() 사용
            write_fn(
                f"> Pandas 데이터프레임을 로컬에 저장 중... (지정된 최종 경로: {save_path})"
            )
            resampled_df.compute().to_csv(save_path, index=index)
            write_fn("  - Pandas 데이터프레임 로컬에 저장 완료")
        else:
            # Dask DataFrame을 파티션별로 저장
            # save_path를 디렉토리로 사용하고, 각 파티션 파일명을 name_function으로 지정
            save_dir = (
                os.path.dirname(save_path)
                if os.path.dirname(save_path)
                else DATASETS_RESAMPLED_PATH
            )
            os.makedirs(save_dir, exist_ok=True)

            # 파일명에서 확장자 제거 (예: "dataset-smote.csv" -> "dataset-smote")
            base_filename = os.path.splitext(os.path.basename(save_path))[0]

            # 각 파티션 파일명을 생성하는 함수
            def name_function(i):
                return os.path.join(save_dir, f"{base_filename}-{i}.part")

            write_fn(
                f"> Dask 데이터프레임을 로컬에 저장 중... (디렉토리: {save_dir}, 파일명 형식: {base_filename}-{{번호}}.part)"
            )
            resampled_df.to_csv(save_dir, name_function=name_function, index=index)
            write_fn("  - Dask 데이터프레임 로컬에 저장 완료")


__all__ = ["Oversampling"]
