class BaseErrors(Exception):
    pass


class DatasetAnalysisError(BaseErrors):
    pass


__all__ = ["BaseErrors", "DatasetAnalysisError"]
