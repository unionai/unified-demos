from common_v1.common_dataclasses import HpoResults
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass
from flytekit.types.structured import StructuredDataset


@dataclass
class DataFrameDict():
    _dataframes = {}

    def get(self, key):
        # return self._dataframes[key].open(pd.DataFrame).all()
        return self._dataframes[key].dataframe

    def add(self, key, value):
        if not isinstance(value, pd.DataFrame)\
                and not isinstance(value, pd.Series):
            raise TypeError("Item must be a pandas dataframe/series")
        print(value)
        sd = StructuredDataset(dataframe=value)
        self._dataframes[key] = sd


def get_training_split(df: pd.DataFrame)\
        -> DataFrameDict:
    test_size = .3
    target_column = "credit_policy"

    y = df[target_column]
    X = df.drop([target_column], axis=1)

    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=test_size, random_state=42)
    retVal = DataFrameDict()
    retVal.add("X_train", X_train)
    retVal.add("X_test", X_test)
    retVal.add("y_train", y_train)
    retVal.add("y_test", y_test)
    return retVal


def get_predictions(obj: HpoResults):

    data = get_training_split(obj.data)
    X_train = data.get("X_train")
    X_test = data.get("X_test")
    y_train = data.get("y_train")
    y_test = data.get("y_test")
    clf = obj.model

    yhat_prob_train = clf.predict_proba(X_train)[:, 1]
    yhat_prob_test = clf.predict_proba(X_test)[:, 1]

    return y_train, yhat_prob_train, y_test, yhat_prob_test
