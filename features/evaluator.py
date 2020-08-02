import multiprocessing as mp
import os

class Evaluator():
    def __init__(self, dataset, model, featurizer):
        self.model = model
        self.dataset = dataset
        self.featurizer = featurizer

    def updater(self, key, value):
        return {
            'key': key,
            'value': value
        }

    def unit(self, row):
        return self.updater(
            self.model(
            self.featurizer(row)
            ), row
        )

    def evaluate(self):
        pool = mp.Pool()
        result = pool.map(self.unit, self.dataset)
        return result