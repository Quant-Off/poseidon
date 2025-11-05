"""
Dask 데이터프레임을 스케일링하는 클래스입니다.
"""

from sklearn.preprocessing import StandardScaler
import dask.dataframe as dd


class DatasetScaling:
    """
    Dask 데이터프레임을 스케일링하는 클래스입니다.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def train_scaler(self, X_train: dd.DataFrame):
        scaled_X_train = self.scaler.fit_transform(X_train)
        return scaled_X_train

    def val_scaler(self, X_val: dd.DataFrame):
        scaled_X_val = self.scaler.transform(X_val)
        return scaled_X_val

    def test_scaler(self, X_test: dd.DataFrame):
        scaled_X_test = self.scaler.transform(X_test)
        return scaled_X_test

    def scale(self, X_train: dd.DataFrame, X_val: dd.DataFrame, X_test: dd.DataFrame):
        scaled_X_train = self.train_scaler(X_train)
        scaled_X_val = self.val_scaler(X_val)
        scaled_X_test = self.test_scaler(X_test)
        return scaled_X_train, scaled_X_val, scaled_X_test


__all__ = ["DatasetScaling"]
