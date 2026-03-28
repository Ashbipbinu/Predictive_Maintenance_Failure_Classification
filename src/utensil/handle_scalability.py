import pandas as pd
import pickle
import os
import logging

from sklearn.preprocessing import StandardScaler

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def handle_scale(data: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    """
    Handles feature scaling using StandardScaler.
    Fits and saves the scaler if is_train is True, otherwise loads and transforms.
    """
    model_dir = os.path.join(os.getcwd(), "models")
    file_name = os.path.join(model_dir, "scale.pkl")

    if is_train:
        logger.info("Starting feature scaling (Training mode)...")
        os.makedirs(model_dir, exist_ok=True)

        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(data)

        # Log the calculated parameters for audit
        logger.info(f"Scaler fitted. Mean of first feature: {scaler.mean_[0]:.4f}")

        try:
            with open(file_name, "wb") as file:
                pickle.dump(scaler, file)
            logger.info(f"StandardScaler saved successfully to: {file_name}")
        except Exception as e:
            logger.error(f"Failed to save scaler: {e}")
            raise

        return pd.DataFrame(scaled_array, columns=data.columns, index=data.index)

    else:
        logger.info("Starting feature scaling (Inference/Test mode)...")

        if not os.path.exists(file_name):
            logger.error(
                f"Scaler file not found at {file_name}. You must run training first."
            )
            raise FileNotFoundError(f"Missing scaler at {file_name}")

        try:
            with open(file_name, "rb") as file:
                scaler = pickle.load(file)
            logger.info("Existing StandardScaler loaded successfully.")

            scaled_array = scaler.transform(data)
            return pd.DataFrame(scaled_array, columns=data.columns, index=data.index)

        except Exception as e:
            logger.error(f"Error during scaling transformation: {e}")
            raise


if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), "data", "processed", "train_processed.csv")
    df = pd.read_csv(data_path)
    handle_scale(df, True)
