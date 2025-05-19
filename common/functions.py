from datasets import load_dataset
import pandas as pd
from sys import platform
from itertools import product
from common_dataclasses import Hyperparameters, SearchSpace


# GET DATA FUNCTIONS
def get_data_databricks(profile_file) -> pd.DataFrame:

    import delta_sharing
    if platform != "linux":
        profile_file = "config.delta.share"
    table_url = profile_file + "#angulo.demo.angulo_m_loan_data"
    df = delta_sharing.load_as_pandas(table_url)
    df.columns = list(map(lambda col: col.replace('.', '_'), df.columns))
    df = df.reindex(sorted(df.columns), axis=1)
    return df

# FEATURIZATION FUNCTIONS
def check_schema(df: pd.DataFrame) -> pd.DataFrame:
    if "purpose_debt_consolidation" not in df.columns:
        df["purpose_debt_consolidation"] = 0
    if "purpose_credit_card" not in df.columns:
        df["purpose_credit_card"] = 0
    if "purpose_small_business" not in df.columns:
        df["purpose_home_improvement"] = 0
    if "purpose_all_other" not in df.columns:
        df["purpose_all_other"] = 0
    if "purpose_educational" not in df.columns:
        df["purpose_educational"] = 0
    if "purpose_major_purchase" not in df.columns:
        df["purpose_major_purchase"] = 0
    if "purpose_small_business" not in df.columns:
        df["purpose_small_business"] = 0
    df = df.reindex(sorted(df.columns), axis=1)
    return df

def featurize(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = pd.get_dummies(df, columns=['purpose'])
    return check_schema(df_encoded)

# TRAINING FUNCTIONS
def create_search_grid(searchspace: SearchSpace) -> list[Hyperparameters]:

    keys = vars(searchspace).keys()
    values = [getattr(searchspace, key) for key in keys]

    grid = [Hyperparameters(**dict(zip(keys, combination)))
            for combination in product(*values)]

    return grid
