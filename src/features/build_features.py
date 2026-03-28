import pandas as pd
import os
import logging

from src.utensil.handle_encodings import handle_target_encodings
from src.utensil.save_file import save_file
from src.utensil.handle_data_split import handle_data_split
from src.utensil.save_location_config import save_location_config
from src.utensil.load_config import load_config

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def built_features() -> None:
    logger.info("Starting feature engineering process...")

    try:
        config = load_config("config.yaml")
        clean_df_file_path = config["data"]["cleaned_df"]
        logger.info(f"Loading cleaned data from: {clean_df_file_path}")

        clean_df = pd.read_csv(clean_df_file_path)
    except Exception as e:
        logger.error(f"Failed to load configuration or data: {e}")
        return

    # Encoding the failure type
    logger.info("Encoding 'failure_type' target...")
    clean_df["failure_type"] = handle_target_encodings(clean_df["failure_type"])

    # Temperature difference (Feature Engineering)
    logger.info("Creating new feature: 'temp_diff_k'")
    clean_df["temp_diff_k"] = abs(
        clean_df["process_temperature_k"] - clean_df["air_temperature_k"]
    )

    # Getting the correlation of target and failure_type with features
    # We log these as DEBUG or INFO so we can verify feature relevance
    target_corr = clean_df.corr()["target"].sort_values(ascending=False)
    failure_type_corr = clean_df.corr()["failure_type"].sort_values(ascending=False)

    logger.info(f"Target Correlation (Top 3):\n{target_corr.head(3).to_string()}")
    logger.info(
        f"Failure Type Correlation (Top 3):\n{failure_type_corr.head(3).to_string()}"
    )

    # Splitting the data into X and y
    columns_to_drop = ["target", "failure_type", "process_temperature_k"]
    logger.info(
        f"Splitting data into Train/Test sets. Dropping columns: {columns_to_drop}"
    )

    try:
        (
            X_train,
            X_test,
            y1_binary_train,
            y1_binary_test,
            y2_multi_train,
            y2_multi_test,
        ) = handle_data_split(
            data=clean_df,
            test_size=0.2,
            columns_to_drop=columns_to_drop,
        )
        logger.info(
            f"Data split successful. Train size: {len(X_train)},"
            f"Test size: {len(X_test)}"
        )
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        return

    # --- Saving Train Data ---
    train_data = X_train.copy()
    train_data["target"] = y1_binary_train
    train_data["failure_type"] = y2_multi_train

    train_file_path = os.path.join(
        os.getcwd(), "data", "processed", "train_processed.csv"
    )
    save_file(train_file_path, train_data)
    logger.info(f"Processed training data saved to: {train_file_path}")

    # --- Saving Test Data ---
    test_data = X_test.copy()
    test_data["target"] = y1_binary_test
    test_data["failure_type"] = y2_multi_test

    test_file_path = os.path.join(
        os.getcwd(), "data", "processed", "test_processed.csv"
    )
    save_file(test_file_path, test_data)
    logger.info(f"Processed testing data saved to: {test_file_path}")

    # --- Updating Config ---
    save_location_config(
        target_loc="data", key_name="train_data", file_path=train_file_path
    )
    save_location_config(
        target_loc="data", key_name="test_data", file_path=test_file_path
    )
    logger.info("Configuration file updated with processed data paths.")

    logger.info("Feature engineering and data splitting completed successfully.")


if __name__ == "__main__":
    built_features()
