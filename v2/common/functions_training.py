from dataclasses import dataclass
from itertools import product
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from joblib import dump, load
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from flyte.io import Dir, File
from typing import Tuple
import random

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
    _data: File = None
    _model: File = None

    @property
    def model(self) -> RandomForestClassifier:
        return self._model

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def serialize(self):
        filename = "pkld_model.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        return File(filename)

    def deserialize(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    async def set_model(self, model: RandomForestClassifier):
        if model is None:
            return
        dir_name = os.path.join(os.path.join(os.path.dirname(__file__), "temp"))
        os.makedirs(dir_name, exist_ok=True)
        filename = os.path.join(dir_name, f'model{random.randint(1,100)}.pkl')

        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        f = await File.from_local(filename)
        os.remove(filename)
        self._model = f

    @model.setter
    async def model(self, model: RandomForestClassifier):
        if model is None:
            return
        filename = f'temp/model{random.randint(1,100)}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        f = await File.from_local(filename)
        self._model = f

    @model.getter
    def model(self):
        if self._model is None:
            return None
        model = load(self._model)
        return model

    @data.getter
    def data(self):
        if self._data is None:
            return None
        return pd.read_parquet(self._data)

    @data.setter
    def data(self, data: pd.DataFrame):
        data.to_parquet('data.parquet')
        self._data = File('data.parquet')

    def to_flytedir(self) -> Dir:
        folder = "tmpStorage"
        if not os.path.exists(folder):
            os.makedirs(folder)

        vars = {
            "acc": self.acc,
            "hp": self.hp
        }

        with open(os.path.join(folder, 'vars.pkl'), 'wb') as handle:
            pickle.dump(vars, handle, protocol=pickle.HIGHEST_PROTOCOL)

        model = self.model
        if model is not None:
            dump(model, os.path.join(folder, 'model.joblib'))

        data = self.data
        if data is not None:
            data.to_parquet(os.path.join(folder, 'data.parquet'))

        return Dir(folder)

    @staticmethod
    def from_flytedir(flytedir: Dir):
        with open(os.path.join(flytedir, 'vars.pkl'), 'rb') as handle:
            vars = pickle.load(handle)

        acc = vars['acc']
        hp = vars['hp']
        model = None
        if os.path.exists(os.path.join(flytedir, 'model.joblib')):
            model = load(os.path.join(flytedir, 'model.joblib'))

        data = None
        if os.path.exists(os.path.join(flytedir, 'data.parquet')):
            data = pd.read_parquet(os.path.join(flytedir, 'data.parquet'))
        retVal = HpoResults(hp, acc)
        retVal.model = model
        retVal.data = data
        return retVal



# TRAINING FUNCTIONS
def create_search_grid(searchspace: SearchSpace) -> list[Hyperparameters]:

    keys = vars(searchspace).keys()
    values = [getattr(searchspace, key) for key in keys]

    grid = [Hyperparameters(**dict(zip(keys, combination)))
            for combination in product(*values)]
    # Slow it down a tad
    time.sleep(3)
    return grid

def get_training_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    test_size = .3
    target_column = "credit_policy"

    y = df[target_column]
    X = df.drop([target_column], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

async def train_classifier(
    hp: Hyperparameters, X_train: pd.DataFrame, X_test: pd.Series,
    y_train:pd.DataFrame, y_test: pd.Series)-> HpoResults:

    clf = RandomForestClassifier(
        max_depth=hp.max_depth,
        max_leaf_nodes=hp.max_leaf_nodes,
        n_estimators=hp.n_estimators)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print("ACCURACY OF THE MODEL:", acc)
    retVal = HpoResults(hp, acc)
    await retVal.set_model(clf)
    return retVal