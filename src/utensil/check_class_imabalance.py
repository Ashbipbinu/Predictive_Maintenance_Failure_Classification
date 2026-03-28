import pandas as pd
import logging
import os

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_class_imbalance(data: pd.Series) -> bool:
    """
    Checks if a target variable is imbalanced based on a 20% difference threshold.
    """
    if data is None or data.empty:
        logger.error("Received empty data for class imbalance check.")
        return False

    # Calculating percentages
    percentages = data.value_counts(normalize=True) * 100
    diff = percentages.max() - percentages.min()

    # Log the distribution for transparency
    dist_str = ", ".join([f"Class {k}: {v:.2f}%" for k, v in percentages.items()])
    logger.info(f"Target Distribution: {dist_str}")
    logger.info(f"Max difference between classes: {diff:.2f}%")

    if diff > 20:
        logger.warning(
            f"Imbalance detected! Difference ({diff:.1f}%) exceeds threshold (20%)."
        )
        return True

    logger.info("Class distribution is within acceptable limits.")
    return False


if __name__ == "__main__":
    # Note: Use forward slashes or raw strings for cross-platform compatibility
    file_path = os.path.join("data", "raw", "predictive_maintenance.csv")

    try:
        df = pd.read_csv(file_path)
        # Ensure 'Target' exists in columns
        if "Target" in df.columns:
            target_data = df["Target"]
            is_imbalanced = check_class_imbalance(target_data)
            logger.info(f"Imbalance Check Result: {is_imbalanced}")
        else:
            logger.error("Column 'Target' not found in the dataset.")

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
