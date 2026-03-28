import pandas as pd
import os
import logging

from src.utensil.check_class_imabalance import check_class_imbalance
from src.utensil.handle_imbalance import handle_imbalance
from src.utensil.save_file import save_file
from src.utensil.save_location_config import save_location_config

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def data_load_preprocessing(file_name: str) -> pd.DataFrame:

    if not os.path.exists(file_name):
        logger.error(f"File not found at location: {file_name}")
        return None

    try:
        df = pd.read_csv(file_name)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return None

    if not df.empty:
        logger.info(f"Data fetched successfully. Shape: {df.shape}")

        # Dropping productid and UID
        columns = ["UDI", "Product ID"]
        df.drop(columns=columns, inplace=True)
        logger.info(f"Successfully removed columns: {columns}")

        # Converting the Type
        # [ the quality class of the machinery being monitored ]

        if "Type" in df.columns:
            df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})
            logger.info("Mapped 'Type' column to numerical values (L:0, M:1, H:2)")

        # Check for any missing values
        if not df.isna().values.any():
            logger.info("No missing values found.")
        else:
            logger.warning("Missing values detected. Starting imputation...")
            # Handling the missing columns of numerical values
            # - replace missing ones with mean
            numeric_cols = df.select_dtypes(include=["number"]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

            # Fill categorical values with mode
            categorical_cols = df.select_dtypes(include=["object"]).columns
            if not categorical_cols.empty:
                df[categorical_cols] = df[categorical_cols].fillna(
                    df[categorical_cols].mode().iloc[0]
                )

            logger.info("Imputation of missing values completed.")

        # Renaming the column names
        columns = [
            "Air temperature [K]",
            "Process temperature [K]Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Failure Type",
        ]

        # Removing [] brackets and return only charaters
        cleaned_columns = [col.replace("[", "").replace("]", "") for col in df.columns]
        cleaned_columns = [col.replace(" ", "_").lower() for col in cleaned_columns]
        df.columns = cleaned_columns

        logger.info(f"Columns cleaned and renamed. New columns: {list(df.columns)}")

        # Checking class imbalance in the target
        # there are 2 targets in the particular data set
        # Target and Failure type
        is_Target_imbalanced = check_class_imbalance(df["target"])
        is_Failure_Type_imbalanced = check_class_imbalance(df["failure_type"])

        if is_Target_imbalanced or is_Failure_Type_imbalanced:
            logger.info("Imbalance detected. Handling class imbalance...")
            balanced_df = handle_imbalance(df)
            logger.info(f"Imbalance handled. New shape: {df.shape}")
        return balanced_df

    else:
        logger.warning("The source dataframe is empty.")
        return None


if __name__ == "__main__":
    # Fetching the data from the data/raw for preprocessing
    directory = os.getcwd()
    raw_relative_path = os.path.join("data", "raw", "predictive_maintenance.csv")

    logger.info("--- Starting Data Preprocessing Pipeline ---")

    raw_data_file_path = os.path.join(directory, raw_relative_path)
    df_cleaned = data_load_preprocessing(raw_data_file_path)

    if df_cleaned is not None:
        # Saving the file to data/interim
        clean_relative_path = os.path.join("data", "interim", "cleaned_df.csv")
        clean_df_file_path = os.path.join(directory, clean_relative_path)
        save_file(clean_df_file_path, df_cleaned)
        logger.info(f"Cleaned data saved to: {clean_df_file_path}")

        # Saving path config
        save_location_config(
            target_loc="data", key_name="raw_df", file_path=raw_data_file_path
        )
        save_location_config(
            target_loc="data", key_name="cleaned_df", file_path=clean_df_file_path
        )
        logger.info("Configuration paths updated successfully.")
    else:
        logger.error("Preprocessing failed. No data to save.")

    logger.info("--- Pipeline Process Finished ---")
