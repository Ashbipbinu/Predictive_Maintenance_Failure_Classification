import os
import pickle
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def handle_target_encodings(y2_data: pd.Series, save_dir: str = None) -> pd.Series:
    """
    Encodes categorical target labels and serializes the LabelEncoder object.
    """
    logger.info("Starting target encoding process...")

    if y2_data is None or y2_data.empty:
        logger.error("Input data for encoding is empty or None.")
        return y2_data

    le = LabelEncoder()
    y2_labeled = le.fit_transform(y2_data)

    # Log the mapping for transparency
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    logger.info(f"Label mapping identified: {mapping}")

    # Define saving path
    root_dir = save_dir if save_dir else os.getcwd()
    folder = os.path.join(root_dir, "models")
    os.makedirs(folder, exist_ok=True)
    file_name = os.path.join(folder, "target_encodings.pkl")

    try:
        with open(file_name, "wb") as f:
            pickle.dump(le, f)
        logger.info(f"LabelEncoder successfully serialized to: {file_name}")
    except Exception as e:
        logger.error(f"Failed to save target_encodings.pkl: {e}")

    return pd.Series(y2_labeled, index=y2_data.index)


if __name__ == "__main__":

    file_path = os.path.join("data", "interim", "cleaned_df.csv")
    data = pd.read_csv(file_path)
    y2_data = data['failure_type']
    handle_target_encodings(y2_data)
