import os
import sys

common = os.path.join(os.path.dirname(__file__),"common")
sys.path.append(os.path.dirname(__file__))
sys.path.append(common)

import union
import union.artifacts
from flytekit import FlyteDirectory
from datasets import load_dataset
import pandas as pd
from typing_extensions import Annotated
from common_v1.functions import get_data_databricks
from common_v1.functions import featurize
from common_v1.functions import create_search_grid
from common_v1.functions import get_training_split
from common_v1.functions import train_classifier_hpo
from common_v1.functions import get_best
from common_v1.common_dataclasses import SearchSpace
from common_v1.common_dataclasses import Hyperparameters
from common_v1.common_dataclasses import HpoResults


# Configuration Parameters
enable_data_cache = False
enable_model_cache = False
cache_version = "4"
fail_workflow = False
environment = "dev"

# Union Objects Definitions
UnifiedTrainedModel = union.artifacts.Artifact(
    name="unified_trained_model",
    partition_keys=["environment"]
)

image = union.ImageSpec(
    builder="union",
    base_image="ghcr.io/unionai-oss/union:py3.10-latest",
    name="unified_demo_union",
    packages=["scikit-learn", "datasets", "pandas",
              "union", "flytekitplugins-spark", "delta-sharing",
              "tabulate", "flytekitplugins-deck-standard"],

)

hpo_actor = union.ActorEnvironment(
    name="hpo-actor",
    replica_count=3,
    ttl_seconds=30,
    container_image=image,
    requests=union.Resources(
        cpu="2",
        mem="1Gi",
    ),
)


# Task Definitions
@union.task(
    container_image=image,
    cache=enable_data_cache,
    cache_version=cache_version,
)
def tsk_get_data_hf() -> pd.DataFrame:

    # Load dataset from HuggingFace and put it in pandas
    ds = load_dataset('AnguloM/loan_data')
    df = ds['train'].to_pandas()
    # Fix column names to not include '.'
    df.columns = list(map(lambda col: col.replace('.', '_'), df.columns))
    df = df.reindex(sorted(df.columns), axis=1)
    return df


@union.task(
    container_image=image,
    cache=enable_data_cache,
    cache_version=cache_version,
    secret_requests=[
        union.Secret(
            key="delta_sharing_creds",
            mount_requirement=union.Secret.MountType.FILE
        ),
    ]
)
def tsk_get_data_databricks() -> pd.DataFrame:
    creds_file_path = union.current_context().secrets.get_secrets_file("delta_sharing_creds")
    return get_data_databricks(creds_file_path)


@union.task(
    container_image=image,
    cache=enable_data_cache,
    cache_version=cache_version,
)
def tsk_featurize(df: pd.DataFrame) -> pd.DataFrame:
    return featurize(df)


@hpo_actor.task
def tsk_train_model_hpo_df(
        hp: Hyperparameters,
        df: pd.DataFrame) -> HpoResults:
    splits = get_training_split(df)
    results = train_classifier_hpo(
        hp, splits)
    results.data = df
    return results


@union.dynamic(
    container_image=image,
    cache=enable_model_cache,
    cache_version=cache_version)
def tsk_hyperparameter_optimization(
    search_space: SearchSpace,
    df: pd.DataFrame
) -> list[HpoResults]:
    grid = create_search_grid(search_space)
    models = []
    for hp in grid:
        res = tsk_train_model_hpo_df(hp, df)
        models.append(res)
    return models


@hpo_actor.task(
    cache=enable_model_cache,
    cache_version=cache_version)
def tsk_get_best(results: list[HpoResults]) -> HpoResults:
    return get_best(results)


@hpo_actor.task
def tsk_register_fd_artifact(
    results: HpoResults
)-> Annotated[FlyteDirectory, UnifiedTrainedModel]:
    return UnifiedTrainedModel.create_from(
        results.to_flytedir(),
        results.get_model_card(),
        environment=environment
    )


@union.task(
    container_image=image,
    requests=union.Resources(mem="6Gi"),
)
def tsk_failure(fail: bool, df: pd.DataFrame, fd: FlyteDirectory) -> None:
    if fail:
        raise Exception("Failure on purpose")


# Workflow Definition
@union.workflow
def unified_demo_wf(
    fail: bool = fail_workflow
):

    df = tsk_get_data_hf()
    fdf = tsk_featurize(df)

    ss = SearchSpace(
        max_depth=[10,20],
        max_leaf_nodes=[10,20],
        n_estimators=[10,20]
    )

    models = tsk_hyperparameter_optimization(ss, fdf)
    best = tsk_get_best(models)
    logged_artifact = tsk_register_fd_artifact(best)

    tsk_failure(fail, fdf, logged_artifact)
