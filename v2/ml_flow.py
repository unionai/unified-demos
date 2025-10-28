import sys
import os
import flyte
import pandas as pd
import asyncio
from typing import Tuple
from datasets import load_dataset

module_directory = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, module_directory)
module_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, module_directory)

from v2.common.functions_featurization import featurize
from v2.common.functions_training import SearchSpace, Hyperparameters
from v2.common.functions_training import create_search_grid, get_training_split, train_classifier
from v2.common.functions_training import HpoResults

env = flyte.TaskEnvironment(
    name="demo_env",
    resources=flyte.Resources(memory="1Gi"),
    image=flyte.Image.from_debian_base()\
        .with_pip_packages(
            "pandas", "flyte", "datasets", "unionai-reuse", "scikit-learn"
            ),
    reusable=flyte.ReusePolicy(
        replicas=3,
        idle_ttl=30
    )
)

@env.task(cache="auto")
async def get_data() -> pd.DataFrame:
    # Load dataset from HuggingFace and put it in pandas
    ds = load_dataset('AnguloM/loan_data')
    df = ds['train'].to_pandas()
    # Fix column names to not include '.'
    df.columns = list(map(lambda col: col.replace('.', '_'), df.columns))
    df = df.reindex(sorted(df.columns), axis=1)
    return df

@env.task(cache="auto")
async def featurize_data(df: pd.DataFrame) -> pd.DataFrame:
    return featurize(df)

@env.task(cache="auto")
async def get_search_grid(ss: SearchSpace) -> list[Hyperparameters]:
    return create_search_grid(ss)

@env.task
async def get_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return get_training_split(df)

@env.task()
async def train_model(
    hp: Hyperparameters, X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train:pd.Series, y_test:pd.Series) -> HpoResults:
    return train_classifier(hp, X_train, X_test, y_train, y_test)

@env.task()
async def ml_workflow() -> None:
    df = await get_data()

    ss = SearchSpace(
        max_depth=[10, 20],
        max_leaf_nodes=[10, 20],
        n_estimators=[10, 20])
    
    res = await asyncio.gather(
        featurize_data(df),
        get_search_grid(ss)
        )
    fdf = res[0]
    gs = res[1]

    X_train, X_test, y_train, y_test = await get_training_data(fdf)
    #data_tuple = await get_training_data(fdf)

    with flyte.group("hyperparameter_optimization"):
        tasks = [train_model(hp, X_train, X_test, y_train, y_test) for hp in gs]
        #tasks = [train_model(hp, data_tuple) for hp in gs]
        models = await asyncio.gather(*tasks)

asyncio.run(ml_workflow())

if __name__ == "__main__":
    
    flyte.init_from_config("../config.yaml")
    run = flyte.run(ml_workflow)
    print(run.name)
    print(run.url)
    run.wait(run)