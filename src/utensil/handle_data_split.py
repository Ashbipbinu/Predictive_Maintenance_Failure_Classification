import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os
import logging

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Loading parameters
try:
    with open("params.yaml", "rb") as file:
        params = yaml.safe_load(file)
    seed = params["base"]["random_state"]
except Exception as e:
    logger.error(f"Failed to load params.yaml: {e}")
    seed = 42  # Fallback seed


def handle_data_split(
    data: pd.DataFrame, test_size: float, columns_to_drop: list[str]
) -> tuple:
    """
    Splits data into train and test sets for multi-target classification.
    """
    logger.info(f"Starting data split with test_size={test_size} and seed={seed}")

    if data is None or data.empty:
        logger.error("Input DataFrame is empty. Split failed.")
        return None

    try:
        # Features and Targets
        X = data.drop(columns=columns_to_drop)
        y1 = data[columns_to_drop[0]]  # e.g., 'target'
        y2 = data[columns_to_drop[1]]  # e.g., 'failure_type'

        logger.info(f"Features: {list(X.columns)}")
        logger.info(f"Targets to split: {columns_to_drop}")

        # Splitting with stratification on the multiclass column (y2)
        X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
            X, y1, y2, test_size=test_size, random_state=seed, stratify=y2
        )

        logger.info("Data split successful.")
        logger.info(f"Train set size: {len(X_train)} | Test set size: {len(X_test)}")

        # Verify stratification distribution
        train_dist = y2_train.value_counts(normalize=True).iloc[0] * 100
        test_dist = y2_test.value_counts(normalize=True).iloc[0] * 100
        logger.info(
            f"Stratification check: Train Majority Class {train_dist:.2f}% "
            f"Test Majority Class {test_dist:.2f}%"
        )

        return (X_train, X_test, y1_train, y1_test, y2_train, y2_test)

    except KeyError as e:
        logger.error(
            f"One or more columns to drop were not found in the DataFrame: {e}"
        )
        raise
    except Exception as e:
        logger.critical(f"Unexpected error during data split: {e}")
        raise


if __name__ == "__main__":

    file_path = os.path.join("data", "interim", "cleaned_df.csv")
    data = pd.read_csv(file_path)
    test_size = 0.2
    columns_to_drop = ["target", "failure_type", "process_temperature_k"]
    handle_data_split(data, test_size, columns_to_drop)
