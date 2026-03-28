import pandas as pd
import logging
import os

from imblearn.over_sampling import SMOTE

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def handle_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles class imbalance using SMOTE by combining multi-target labels
    to preserve correlations during over-sampling.
    """
    logger.info("Starting class imbalance handling via SMOTE...")

    if df is None or df.empty:
        logger.error("Input DataFrame is empty. Skipping resampling.")
        return df

    target_cols = ["target", "failure_type"]
    feature_cols = [col for col in df.columns if col not in target_cols]

    # Create a combined target to handle the correlation
    # between binary and multi-class labels
    df["combined_target"] = (
        df[target_cols[0]].astype(str) + "_" + df[target_cols[1]].astype(str)
    )

    X = df[feature_cols]
    y = df["combined_target"]

    logger.info(f"Original class distribution:\n{y.value_counts()}")

    # SMOTE requires at least 2 samples per class to find neighbors
    counts = y.value_counts()
    if counts.min() < 2:
        logger.warning(
            f"Found classes with less than 2 samples:"
            f"{counts[counts < 2].index.tolist()}."
            "SMOTE requires at least 2 samples. Removing these rare classes."
        )
        valid_indices = y[y.isin(counts[counts >= 2].index)].index
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]

    try:
        smote = SMOTE(random_state=42)
        logger.info("Applying SMOTE fit_resample...")
        X_res, y_res = smote.fit_resample(X, y)

        logger.info(
            f"Resampling complete. New sample count: {len(X_res)} (Original: {len(X)})"
        )

        # Reconstruct the DataFrame
        resample_df = pd.DataFrame(X_res, columns=feature_cols)

        # Splitting the combined y label back into 2 columns
        split_targets = y_res.str.split("_", expand=True)
        resample_df[target_cols[0]] = split_targets[0].astype(int)
        resample_df[target_cols[1]] = split_targets[1]

        # Final Verification
        logger.info(
            f"Resampled target distribution:\n"
            f"{resample_df[target_cols[1]].value_counts()}"
        )

        return resample_df

    except Exception as e:
        logger.error(f"Error during SMOTE resampling: {e}")
        # Return original dataframe if SMOTE fails to avoid breaking the pipeline
        return df.drop(columns=["combined_target"])


if __name__ == "__main__":
    file_path = os.path.join("data", "interim", "cleaned_df.csv")
    data = pd.read_csv(file_path)
    handle_imbalance(data)
