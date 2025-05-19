from sklearn.base import BaseEstimator
from dataclasses import dataclass
from flytekit import FlyteFile
from flytekit.types.structured import StructuredDataset
import pandas as pd
import pickle
from joblib import dump, load


@dataclass
class DataSplits():
    X_train: StructuredDataset
    X_test: StructuredDataset
    y_train: pd.Series
    y_test: pd.Series


@dataclass
class Hyperparameters:
    max_depth: int
    max_leaf_nodes: int
    n_estimators: int


@dataclass
class SearchSpace:
    max_depth: list[int]
    max_leaf_nodes: list[int]
    n_estimators: list[int]


@dataclass
class HpoResults:
    hp: Hyperparameters
    acc: float
    _model: FlyteFile = None

    @property
    def model(self) -> BaseEstimator:
        return self._model

    def serialize(self):
        filename = "pkld_model.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        return FlyteFile(filename)

    def deserialize(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @model.setter
    def model(self, model: BaseEstimator):
        if model is None:
            return
        dump(model, 'model.joblib')
        f = FlyteFile('model.joblib')
        self._model = f

    @model.getter
    def model(self):
        model = load(self._model)
        return model
