"""
Dask 라이브러리를 사용하여 데이터프레임을 불러오는 모듈입니다.
"""
import dask.dataframe as dd
from dask.distributed import Client

from poseidon.log.poseidon_log import PoseidonLogger


def load_large_dataset(file_path: str, file_format: str = 'csv', dtypes: dict = None,
                       blocksize: str = '128MB', npartitions=3) -> dd.DataFrame:
    """
    대용량 데이터셋을 Dask DataFrame으로 효율적이고 안전하게 로드하는 함수입니다.
    blocksize 파라미터를 헷갈리지 마세요! 이는 데아터셋이 나뉘어지는 단위를 의미합니다.
    반면 npartitions 파라미터의 경우, Dask의 병렬성을 위한 worker 수를 지정할 수 있습니다.
    최적의 성능을 위해 파티션 수를 worker 수의 1~2배 정도로 맞추는 것이 일반적입니다.
    이는 각 worker가 하나 이상의 파티션을 처리할 수 있게 하여 CPU 이용률을 높입니다.

    Args:
        file_path (str): 파일 또는 디렉토리 경로 (e.g., 'data/*.parquet')
        file_format (str): 'parquet' 또는 'csv' (기본값: 'csv')
        dtypes (dict): 컬럼별 데이터 타입 사전 지정 (메모리 최적화)
        blocksize (str): 파티션 크기 (기본값: '128MB')
        npartitions (int): 작업에 사용할 워커 수 (기본값: 3)

    Returns:
        dd.DataFrame: 로드된 Dask DataFrame.

    Raises:
        ValueError: 지원되지 않는 형식 또는 경로 오류.
        Exception: 로딩 중 기타 오류.
    """
    # 포세이돈 로깅
    logging = PoseidonLogger().get_logger()
    try:
        # Dask 클라이언트 생성: 로컬 클러스터로 병렬 처리 (안전한 리소스 관리)
        with Client(n_workers=npartitions, threads_per_worker=1):  # worker 수는 시스템에 맞게 조정
            logging.info(f"{file_format} 포멧의 {file_path} 데이터셋을 로드합니다...")

            if file_format.lower() == 'parquet':  # Parquet 포멧
                df = dd.read_parquet(file_path, engine='pyarrow', dtype=dtypes, blocksize=blocksize)
            elif file_format.lower() == 'csv':  # CSV 포멧
                df = dd.read_csv(file_path, dtype=dtypes, blocksize=blocksize)
            else:
                raise ValueError(f"{file_format} 포멧은 지원되지 않습니다! 'parquet' 또는 'csv' 포멧을 선택하세요.")

            # 데이터 유효성 검사 (헤드 확인 (지연 계산이므로 compute() 호출))
            sample = df.head(5, npartitions=1)  # 첫 5행 샘플 확인
            logging.info(f"샘플 데이터가 로드되었습니다(첫 5행):\n{sample}")

            # 기본 통계 계산으로 데이터 무결성 확인 (대용량 시 생략 가능)
            stats = df.describe().compute()
            logging.info(f"  - 데이터셋 통계는 다음과 같습니다:\n{stats}")

            num_partitions = df.npartitions
            logging.info(f"파티션 수는 '{num_partitions}' 이며, CPU 사용량을 최적화하려면 이상적인 범위는 {npartitions} ~ {2 * npartitions}입니다."
                         )
            logging.info("  - 데이터셋 로드가 완료되었습니다.")
            return df

    except FileNotFoundError:
        logging.error(f"'{file_path}' 파일 또는 디렉토리를 찾을 수 없습니다!")
        raise
    except Exception as e:
        logging.error(f"{str(e)} 데이터셋을 로드하는 도중 오류가 발생했습니다!")
        raise


__all__ = ['load_large_dataset']
