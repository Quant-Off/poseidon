"""
Dask 데이터프레임을 훈련(train), 검증(val), 테스트(test) 데이터셋으로 분할하는 클래스입니다.
6:2:2 비율로 분할합니다.
"""

import dask.dataframe as dd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class DatasetSplit:
    """
    Dask 데이터프레임을 훈련(train), 검증(val), 테스트(test) 데이터셋으로 분할하는 클래스입니다.
    6:2:2 비율로 분할합니다.
    """

    def __init__(self, resampled_dataframe: dd.DataFrame):
        """
        Parameters:
        -----------
        resampled_dataframe : dd.DataFrame
            SMOTE 리샘플링 후 저장된 Dask 데이터프레임
        """
        self.resampled_dataframe = resampled_dataframe

    def split(self, label_col="Label", random_state=42, npartitions=2):
        """
        데이터프레임을 훈련(60%), 검증(20%), 테스트(20%) 데이터셋으로 분할합니다.

        Parameters:
        -----------
        label_col : str
            레이블 컬럼 이름 (기본값: "Label")
        random_state : int
            랜덤 시드 (기본값: 42)
        npartitions : int
            반환될 Dask DataFrame의 파티션 수 (기본값: 2, 시스템에 맞게 조정 필요)

        Returns:
        --------
        splited_X_train : dd.DataFrame
        splited_X_val : dd.DataFrame
        splited_X_test : dd.DataFrame
        splited_y_train : dd.DataFrame
        splited_y_val : dd.DataFrame
        splited_y_test : dd.DataFrame
        """
        # Dask DataFrame을 pandas DataFrame으로 변환
        with tqdm(total=self.resampled_dataframe.npartitions, desc="Pandas 변환 중", ncols=100) as pbar:
            resampled_df = self.resampled_dataframe.compute()
            pbar.update(self.resampled_dataframe.npartitions)

        # X와 y 분리
        with tqdm(total=1, desc="X, y 분리 중", ncols=100) as pbar:
            splited_X = resampled_df.drop(label_col, axis=1)
            splited_y = resampled_df[label_col]
            pbar.update(1)

        # 첫 번째 분할: 훈련(60%) vs. 임시(40%)
        with tqdm(total=1, desc="첫 번째 분할 중 (훈련/임시)", ncols=100) as pbar:
            splited_X_train_pd, splited_X_temp_pd, splited_y_train_pd, splited_y_temp_pd = (
                train_test_split(
                    splited_X,
                    splited_y,
                    test_size=0.4,
                    stratify=splited_y,
                    random_state=random_state,
                )
            )
            pbar.update(1)

        # 두 번째 분할: 임시를 검증(20%) vs. 테스트(20%)
        with tqdm(total=1, desc="두 번째 분할 중 (검증/테스트)", ncols=100) as pbar:
            splited_X_val_pd, splited_X_test_pd, splited_y_val_pd, splited_y_test_pd = (
                train_test_split(
                    splited_X_temp_pd,
                    splited_y_temp_pd,
                    test_size=0.5,
                    stratify=splited_y_temp_pd,
                    random_state=random_state,
                )
            )
            pbar.update(1)

        # Dask 변환
        datasets_to_convert = [
            (splited_X_train_pd, "X_train"),
            (splited_X_val_pd, "X_val"),
            (splited_X_test_pd, "X_test"),
            (splited_y_train_pd, "y_train"),
            (splited_y_val_pd, "y_val"),
            (splited_y_test_pd, "y_test"),
        ]
        with tqdm(total=len(datasets_to_convert), desc="Dask 변환 중", ncols=100) as pbar:
            for df_pd, name in datasets_to_convert:
                if name == "X_train":
                    splited_X_train = dd.from_pandas(df_pd, npartitions=npartitions)
                elif name == "X_val":
                    splited_X_val = dd.from_pandas(df_pd, npartitions=npartitions)
                elif name == "X_test":
                    splited_X_test = dd.from_pandas(df_pd, npartitions=npartitions)
                elif name == "y_train":
                    splited_y_train = dd.from_pandas(df_pd, npartitions=npartitions)
                elif name == "y_val":
                    splited_y_val = dd.from_pandas(df_pd, npartitions=npartitions)
                elif name == "y_test":
                    splited_y_test = dd.from_pandas(df_pd, npartitions=npartitions)
                pbar.update(1)

        return (
            splited_X_train,
            splited_X_val,
            splited_X_test,
            splited_y_train,
            splited_y_val,
            splited_y_test,
        )


__all__ = ["DatasetSplit"]
