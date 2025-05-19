import union
import union.artifacts
from flytekit import FlyteDirectory
from datasets import load_dataset
import pandas as pd
from common.functions import get_data_databricks
from common.functions import featurize
from common.common_dataclasses import SearchSpace


# Configuration Parameters
enable_data_cache = False
enable_model_cache = False
cache_version = "3"

# Union Objects Definitions
ClsModelResults = union.artifacts.Artifact(
    name="unified_demo_model_results"
)

image = union.ImageSpec(
    builder="union",
    base_image="ghcr.io/unionai-oss/union:py3.10-latest",
    name="unified_demo",
    registry="pablounionai",
    packages=["scikit-learn", "datasets", "pandas",
              "union", "flytekitplugins-spark", "delta-sharing"],

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
            mount_requirement=union.Secret.MountType.FILE),
    ]
)
def tsk_get_data_databricks() -> pd.DataFrame:
    creds_file_path =\
        union.current_context().secrets.get_secrets_file("delta_sharing_creds")

    return get_data_databricks(creds_file_path)


@union.task(
    container_image=image,
    cache=enable_data_cache,
    cache_version=cache_version,
)
def tsk_featurize(df: pd.DataFrame) -> pd.DataFrame:
    return featurize(df)


@union.task(
    container_image=image,
    requests=union.Resources(mem="6Gi")
)
def tsk_failure(df: pd.DataFrame, fd: FlyteDirectory) -> None:
    fail = True
    if fail:
        raise Exception("Failure")


# Workflow Definition
@union.workflow
def unified_demo_wf():

    df = tsk_get_data_hf()
    fdf = tsk_featurize()

    ss = SearchSpace(
        max_depth=[10, 20],
        max_leaf_nodes=[10, 20],
        n_estimators=[10, 20])