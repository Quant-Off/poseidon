"""
Dask 데이터프레임의 피처별 히스토그램을 분석하는 클래스입니다.
"""

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DatasetAnalysis:
    """
    Dask 데이터프레임의 피처별 히스토그램을 분석하는 클래스입니다.
    """

    def __init__(self, exclude_features: list = None):
        """
        Parameters:
        -----------
        exclude_features : list, optional
            히스토그램 분석에서 제외할 피처 목록 (기본값: ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"])
        """
        if exclude_features is None:
            self.exclude_features = [
                "IPV4_SRC_ADDR",
                "IPV4_DST_ADDR",
                "L4_SRC_PORT",
                "L4_DST_PORT",
            ]
        else:
            self.exclude_features = exclude_features

    def analyze(
        self,
        X_train,
        X_val=None,
        X_test=None,
        feature_names=None,
        plot: bool = False,
    ):
        """
        데이터프레임을 분석하여 피처별 히스토그램을 출력합니다.

        Parameters:
        -----------
        X_train : dd.DataFrame or np.ndarray
            훈련 데이터셋
        X_val : dd.DataFrame or np.ndarray, optional
            검증 데이터셋
        X_test : dd.DataFrame or np.ndarray, optional
            테스트 데이터셋
        feature_names : list, optional
            피처 이름 리스트 (numpy array 입력 시 필수)
        plot : bool
            matplotlib 플롯을 표시할지 여부 (기본값: False)

        Returns:
        --------
        tuple
            (to_df_X_train, to_df_X_val, to_df_X_test) 또는 (to_df_X_train, None, None)
        """
        # Dask DataFrame을 pandas DataFrame으로 변환
        if isinstance(X_train, dd.DataFrame):
            X_train_pd = X_train.compute()
            feature_names = list(X_train_pd.columns)
        elif isinstance(X_train, np.ndarray):
            if feature_names is None:
                raise ValueError(
                    "numpy array 입력 시 feature_names 파라미터가 필요합니다."
                )
            X_train_pd = pd.DataFrame(X_train, columns=feature_names)
        else:
            # pandas DataFrame인 경우
            X_train_pd = X_train
            if feature_names is None:
                feature_names = list(X_train_pd.columns)

        # 검증 데이터셋 처리
        if X_val is not None:
            if isinstance(X_val, dd.DataFrame):
                X_val_pd = X_val.compute()
            elif isinstance(X_val, np.ndarray):
                X_val_pd = pd.DataFrame(X_val, columns=feature_names)
            else:
                X_val_pd = X_val
        else:
            X_val_pd = None

        # 테스트 데이터셋 처리
        if X_test is not None:
            if isinstance(X_test, dd.DataFrame):
                X_test_pd = X_test.compute()
            elif isinstance(X_test, np.ndarray):
                X_test_pd = pd.DataFrame(X_test, columns=feature_names)
            else:
                X_test_pd = X_test
        else:
            X_test_pd = None

        # 제외할 피처 제거
        columns_list = [
            name for name in feature_names if name not in self.exclude_features
        ]

        # DataFrame 생성
        to_df_X_train = X_train_pd[columns_list]

        if X_val_pd is not None:
            to_df_X_val = X_val_pd[columns_list]
        else:
            to_df_X_val = None

        if X_test_pd is not None:
            to_df_X_test = X_test_pd[columns_list]
        else:
            to_df_X_test = None

        # 각 피처별 히스토그램 출력
        for column_name in columns_list:
            print(f"  {column_name} 훈련 세트 히스토그램:")
            if plot:
                plt.figure()
                to_df_X_train[column_name].hist()
                plt.title(f"{column_name} - 훈련 세트")
                plt.xlabel(column_name)
                plt.ylabel("빈도")
                plt.show()

            if to_df_X_val is not None:
                print(f"  {column_name} 검증 세트 히스토그램:")
                if plot:
                    plt.figure()
                    to_df_X_val[column_name].hist()
                    plt.title(f"{column_name} - 검증 세트")
                    plt.xlabel(column_name)
                    plt.ylabel("빈도")
                    plt.show()

            if to_df_X_test is not None:
                print(f"  {column_name} 테스트 세트 히스토그램:")
                if plot:
                    plt.figure()
                    to_df_X_test[column_name].hist()
                    plt.title(f"{column_name} - 테스트 세트")
                    plt.xlabel(column_name)
                    plt.ylabel("빈도")
                    plt.show()

        if to_df_X_val is not None and to_df_X_test is not None:
            return to_df_X_train, to_df_X_val, to_df_X_test
        else:
            return to_df_X_train, None, None

    def plot(
        self,
        X_train,
        X_val=None,
        X_test=None,
        feature_names=None,
        figsize=(15, 5),
    ):
        """
        피처별 히스토그램을 matplotlib으로 시각화합니다.

        Parameters:
        -----------
        X_train : dd.DataFrame or np.ndarray
            훈련 데이터셋
        X_val : dd.DataFrame or np.ndarray, optional
            검증 데이터셋
        X_test : dd.DataFrame or np.ndarray, optional
            테스트 데이터셋
        feature_names : list, optional
            피처 이름 리스트 (numpy array 입력 시 필수)
        figsize : tuple
            그림 크기 (기본값: (15, 5))
        """
        # Dask DataFrame을 pandas DataFrame으로 변환
        if isinstance(X_train, dd.DataFrame):
            X_train_pd = X_train.compute()
            feature_names = list(X_train_pd.columns)
        elif isinstance(X_train, np.ndarray):
            if feature_names is None:
                raise ValueError(
                    "numpy array 입력 시 feature_names 파라미터가 필요합니다."
                )
            X_train_pd = pd.DataFrame(X_train, columns=feature_names)
        else:
            X_train_pd = X_train
            if feature_names is None:
                feature_names = list(X_train_pd.columns)

        # 검증 데이터셋 처리
        if X_val is not None:
            if isinstance(X_val, dd.DataFrame):
                X_val_pd = X_val.compute()
            elif isinstance(X_val, np.ndarray):
                X_val_pd = pd.DataFrame(X_val, columns=feature_names)
            else:
                X_val_pd = X_val
        else:
            X_val_pd = None

        # 테스트 데이터셋 처리
        if X_test is not None:
            if isinstance(X_test, dd.DataFrame):
                X_test_pd = X_test.compute()
            elif isinstance(X_test, np.ndarray):
                X_test_pd = pd.DataFrame(X_test, columns=feature_names)
            else:
                X_test_pd = X_test
        else:
            X_test_pd = None

        # 제외할 피처 제거
        columns_list = [
            name for name in feature_names if name not in self.exclude_features
        ]

        # 각 피처별 히스토그램 시각화
        for column_name in columns_list:
            # 데이터셋 수에 따라 subplot 개수 결정
            n_datasets = sum(
                [1 for x in [X_train_pd, X_val_pd, X_test_pd] if x is not None]
            )

            _, axes = plt.subplots(1, n_datasets, figsize=figsize)
            if n_datasets == 1:
                axes = [axes]

            idx = 0

            # 훈련 세트
            axes[idx].hist(
                X_train_pd[column_name], bins=30, alpha=0.7, edgecolor="black"
            )
            axes[idx].set_title(f"{column_name}\n훈련 세트")
            axes[idx].set_xlabel(column_name)
            axes[idx].set_ylabel("빈도")
            axes[idx].grid(True, alpha=0.3)
            idx += 1

            # 검증 세트
            if X_val_pd is not None:
                axes[idx].hist(
                    X_val_pd[column_name], bins=30, alpha=0.7, edgecolor="black"
                )
                axes[idx].set_title(f"{column_name}\n검증 세트")
                axes[idx].set_xlabel(column_name)
                axes[idx].set_ylabel("빈도")
                axes[idx].grid(True, alpha=0.3)
                idx += 1

            # 테스트 세트
            if X_test_pd is not None:
                axes[idx].hist(
                    X_test_pd[column_name], bins=30, alpha=0.7, edgecolor="black"
                )
                axes[idx].set_title(f"{column_name}\n테스트 세트")
                axes[idx].set_xlabel(column_name)
                axes[idx].set_ylabel("빈도")
                axes[idx].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
