import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import json
import logging

from src.utensil.load_config import load_config
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def visualize_metrics(report, threshold, col, y_test, y_pred):
    # Handling potential KeyError :-
    # if '1' is not in report (e.g., no failures in test set)
    try:
        metrics = {
            "f1_score": report["1"]["f1-score"],
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "threshold": threshold,
        }
    except KeyError:
        logger.warning(
            f"Label '1' (Failure) not found in classification report for {col}."
            "Recording zeros."
        )
        metrics = {"f1_score": 0, "precision": 0, "recall": 0, "threshold": threshold}

    metrics_filepath = f"test_metric-{col}.json"
    with open(metrics_filepath, "w") as file:
        json.dump(metrics, file, indent=4)

    logger.info(
        f"Metrics for {col} (Threshold: {threshold}): "
        f"F1: {metrics['f1_score']:.4f}, Precision: {metrics['precision']:.4f}, Recall:"
        f"{metrics['recall']:.4f}"
    )

    # Visualizing the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    plt.title(f"Confusion Matrix: {col} | Model: RandomForest")

    save_path = f"confusion-{col}.png"
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to free up memory
    logger.info(f"Confusion matrix plot saved to: {save_path}")


def predict_model():
    logger.info("Starting model prediction and evaluation...")

    try:
        # Loading the test data
        config = load_config("config.yaml")
        test_data_path = config["data"]["test_data"]
        test_df = pd.read_csv(test_data_path)
        logger.info(f"Loaded test data from: {test_data_path}")

        # Loading the best model
        file_path = os.path.join(os.getcwd(), "models", "random_forest.pkl")
        if not os.path.exists(file_path):
            logger.error(
                f"Model file not found at {file_path}. Ensure training is complete."
            )
            return

        with open(file_path, "rb") as file:
            best_model = pickle.load(file)
        logger.info("Successfully loaded Random Forest model.")

    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        return

    columns = ["target", "failure_type"]
    X_test = test_df.drop(columns=columns)

    # False Negatives are expensive, so a threshold is set to address this issue
    threshold = 0.25
    logger.info(f"Using classification threshold: {threshold}")

    for i, col in enumerate(columns):
        logger.info(f"Evaluating label: {col}")
        y_test = test_df[col]

        # Note: Ensure the model supports predict_proba (RandomForest does)
        try:
            # Multi-output models return a list of arrays for predict_proba
            y_prob = best_model.predict_proba(X_test)[i][:, 1]
            y_pred = (y_prob >= threshold).astype(int)

            report = classification_report(y_test, y_pred, output_dict=True)
            visualize_metrics(
                report=report,
                threshold=threshold,
                col=col,
                y_pred=y_pred,
                y_test=y_test,
            )
        except Exception as e:
            logger.error(f"Failed to evaluate {col}: {e}")

    logger.info("Evaluation pipeline completed.")


if __name__ == "__main__":
    predict_model()
