"""
Dask 데이터프레임을 스케일링하는 클래스입니다.
"""

from sklearn.preprocessing import StandardScaler
import dask.dataframe as dd
import pandas as pd


class DatasetScaling:
    """
    Dask 데이터프레임을 스케일링하는 클래스입니다.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def train_scaler(self, X_train: dd.DataFrame):
        # 컬럼명과 순서 저장 (명시적으로 보존)
        column_names = list(X_train.columns)
        # pandas DataFrame으로 변환하여 스케일링
        X_train_pd = X_train.compute()
        # StandardScaler는 NumPy 배열을 반환하므로 컬럼명이 사라짐
        # 따라서 컬럼명을 명시적으로 복원해야 함
        scaled_X_train_array = self.scaler.fit_transform(X_train_pd)
        # 컬럼명과 순서를 보존하여 pandas DataFrame 생성
        scaled_X_train_pd = pd.DataFrame(
            scaled_X_train_array,
            columns=column_names,
            index=X_train_pd.index,  # 인덱스도 보존
        )
        # Dask DataFrame으로 변환 (컬럼명 보존)
        scaled_X_train = dd.from_pandas(
            scaled_X_train_pd, npartitions=X_train.npartitions
        )
        return scaled_X_train

    def val_scaler(self, X_val: dd.DataFrame):
        # 컬럼명과 순서 저장 (명시적으로 보존)
        column_names = list(X_val.columns)
        # pandas DataFrame으로 변환하여 스케일링
        X_val_pd = X_val.compute()
        # StandardScaler는 NumPy 배열을 반환하므로 컬럼명이 사라짐
        # 따라서 컬럼명을 명시적으로 복원해야 함
        scaled_X_val_array = self.scaler.transform(X_val_pd)
        # 컬럼명과 순서를 보존하여 pandas DataFrame 생성
        scaled_X_val_pd = pd.DataFrame(
            scaled_X_val_array,
            columns=column_names,
            index=X_val_pd.index,  # 인덱스도 보존
        )
        # Dask DataFrame으로 변환 (컬럼명 보존)
        scaled_X_val = dd.from_pandas(scaled_X_val_pd, npartitions=X_val.npartitions)
        return scaled_X_val

    def test_scaler(self, X_test: dd.DataFrame):
        # 컬럼명과 순서 저장 (명시적으로 보존)
        column_names = list(X_test.columns)
        # pandas DataFrame으로 변환하여 스케일링
        X_test_pd = X_test.compute()
        # StandardScaler는 NumPy 배열을 반환하므로 컬럼명이 사라짐
        # 따라서 컬럼명을 명시적으로 복원해야 함
        scaled_X_test_array = self.scaler.transform(X_test_pd)
        # 컬럼명과 순서를 보존하여 pandas DataFrame 생성
        scaled_X_test_pd = pd.DataFrame(
            scaled_X_test_array,
            columns=column_names,
            index=X_test_pd.index,  # 인덱스도 보존
        )
        # Dask DataFrame으로 변환 (컬럼명 보존)
        scaled_X_test = dd.from_pandas(scaled_X_test_pd, npartitions=X_test.npartitions)
        return scaled_X_test

    def scale(self, X_train: dd.DataFrame, X_val: dd.DataFrame, X_test: dd.DataFrame):
        scaled_X_train = self.train_scaler(X_train)
        scaled_X_val = self.val_scaler(X_val)
        scaled_X_test = self.test_scaler(X_test)
        return scaled_X_train, scaled_X_val, scaled_X_test
