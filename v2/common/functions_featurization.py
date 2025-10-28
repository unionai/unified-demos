import pandas as pd

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

