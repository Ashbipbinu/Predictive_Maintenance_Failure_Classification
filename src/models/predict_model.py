import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import json

from src.utensil.load_config import load_config

from sklearn.metrics import classification_report, ConfusionMatrixDisplay


def visualize_metrics(report, threshold, col, y_test, y_pred):
    metrics = {
        "f1_score": report["1"]["f1-score"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "threshold": threshold,
    }

    metrics_filepath = f"test_metric-{col}.json"
    with open(metrics_filepath, "w") as file:
        json.dump(metrics, file, indent=4)

    # Visualizing the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    plt.title(
        f"Confusion matric of Failure Classification for {col} label: RandomForest"
    )
    plt.savefig(f"confusion-{col}.png")

    return


def predict_model():

    # Loading the test data
    config = load_config("config.yaml")
    test_data_path = config["data"]["test_data"]
    test_df = pd.read_csv(test_data_path)

    # Loading the best model
    file_dir = os.getcwd()
    file_path = os.path.join(file_dir, "models", "random_forest.pkl")
    with open(file_path, "rb") as file:
        best_model = pickle.load(file)

    columns = ["target", "failure_type"]
    X_test = test_df.drop(columns=columns)

    # False Negatives are expensive, so a threshold is set to address this issue
    threshold = 0.3

    for i, col in enumerate(columns):
        y_test = test_df[col]
        y_prob = best_model.predict_proba(X_test)[i][:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True)
        visualize_metrics(
            report=report, threshold=threshold, col=col, y_pred=y_pred, y_test=y_test
        )


if __name__ == "__main__":
    predict_model()
