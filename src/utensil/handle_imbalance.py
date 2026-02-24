from imblearn.over_sampling import SMOTE
import pandas as pd


def handle_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    
    # Since the target labels are correlated to each other, combine them and SMOTE it
    target_cols = ["target", "failure_type"]
    feature_cols = [col for col in df.columns if col not in target_cols]

    df["combined_target"] = df[target_cols[0]].astype(str) + "_" + df[target_cols[1]].astype(str)

    X = df[feature_cols]
    y = df["combined_target"]

    smote = SMOTE(random_state=42)

    X_res, y_res = smote.fit_resample(X, y)

    resample_df = pd.DataFrame(X_res, columns=feature_cols)

    # Splitting the y label into 2 columns
    split_targets = y_res.str.split("_", expand=True)
    # Joinng back toresampled dataframe
    resample_df[target_cols[0]] = split_targets[0].astype(int)
    resample_df[target_cols[1]] = split_targets[1]

    return resample_df
    