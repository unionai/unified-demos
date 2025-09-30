# /// script
# dependencies = [
#   "requests<3",
#   "rich",
#   "flyte"
# ]
# ///

import flyte
import pandas as pd
from datasets import load_dataset

from common.functions_featurization import featurize

env = flyte.TaskEnvironment(
    name="demo_env",
    resources=flyte.Resources(memory="1Gi"),
    image=flyte.Image.from_debian_base()\
        .with_pip_packages("pandas", "flyte", "datasets", "unionai-reuse"),
    reusable=flyte.ReusePolicy(
        replicas=3,
        idle_ttl=30
    )
)

@env.task()
async def get_data() -> pd.DataFrame:
    # Load dataset from HuggingFace and put it in pandas
    ds = load_dataset('AnguloM/loan_data')
    df = ds['train'].to_pandas()
    # Fix column names to not include '.'
    df.columns = list(map(lambda col: col.replace('.', '_'), df.columns))
    df = df.reindex(sorted(df.columns), axis=1)
    return df

@env.task()
async def featurize_data(df: pd.DataFrame) -> pd.DataFrame:
    return featurize(df)

@env.task()
async def ml_workflow() -> None:
    df = await get_data()
    fdf = await featurize_data(df)


if __name__ == "__main__":
    
    flyte.init_from_config("../config.yaml")
    run = flyte.run(ml_workflow)
    print(run.name)
    print(run.url)
    run.wait(run)