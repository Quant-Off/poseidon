from poseidon.prototype import feature_engineering

class EntropyFeature:
    def __init__(self):
        self.train_X = "d"


    def _apply_entropy_dask(self, row):
        return feature_engineering.apply_entropy(row)