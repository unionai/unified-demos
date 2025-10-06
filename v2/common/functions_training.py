from dataclasses import dataclass
from itertools import product
import pandas as pd
import time
from sklearn.model_selection import train_test_split

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



# TRAINING FUNCTIONS
async def create_search_grid(searchspace: SearchSpace) -> list[Hyperparameters]:

    keys = vars(searchspace).keys()
    values = [getattr(searchspace, key) for key in keys]

    grid = [Hyperparameters(**dict(zip(keys, combination)))
            for combination in product(*values)]
    # Slow it down a tad
    time.sleep(3)
    return grid

def get_training_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    test_size = .3
    target_column = "credit_policy"

    y = df[target_column]
    X = df.drop([target_column], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test