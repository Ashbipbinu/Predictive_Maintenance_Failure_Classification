import pandas as pd
import mlflow
import mlflow.sklearn
import os
import pickle
import json
import dagshub
import logging
from dotenv import load_dotenv

from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
from mlflow.models import infer_signature

from src.utensil.load_models import load_ml_models
from src.utensil.load_config import load_config
from src.utensil.handle_scalability import handle_scale
from src.utensil.save_location_config import save_location_config
from src.utensil.handle_data_split import handle_data_split

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()


def evaluate_multioutput(y_true, y_pred):
    metrics = {}
    for i, col in enumerate(["target", "failure_type"]):

        metrics[f"{col}_f1_weighted"] = f1_score(
            y_true.iloc[:, i], y_pred[:, i], average="weighted"
        )

        metrics[f"{col}_recall_weighted"] = recall_score(
            y_true.iloc[:, i], y_pred[:, i], average="weighted"
        )

        metrics[f"{col}_f1_weighted"] = precision_score(
            y_true.iloc[:, i], y_pred[:, i], average="weighted"
        )

    metrics["overall_f1"] = (
        metrics["target_f1_weighted"] + metrics["failure_type_f1_weighted"]
    ) / 2
    return metrics


def authenticate_dagshub():
    try:
        token = os.getenv("DAGSHUB_TOKEN")
        if token:
            dagshub.auth.add_app_token(token)
            logger.info("Successfully authenticated with DagsHub token.")
            return True
        else:
            logger.warning("DAGSHUB_TOKEN not found in environment variables.")
    except Exception as e:
        logger.error(f"Error while authenticating with DagsHub: {e}")
        return False


def dagshub_init():
    # Initialize DagsHub
    try:
        dagshub.init(
            repo_owner=os.getenv("DAGSHUB_USER_NAME"),
            repo_name=os.getenv("DAGSHUB_REPO_NAME"),
            mlflow=True,
        )
        logger.info("DagsHub/MLflow integration initialized.")
        return True
    except Exception as e:
        logger.error(f"DagsHub initialization failed: {e}")
        return False


def train_model():
    logger.info("Starting Multi-Output Training Pipeline...")
    is_dagshub_auth_success = authenticate_dagshub()
    dagshub_init_success = dagshub_init()

    if is_dagshub_auth_success and dagshub_init_success:

        mlflow.set_experiment("Predictive_Maintenance")
        with mlflow.start_run(run_name="Parent Optimization Run") as parent_run:
            logger.info(f"Parent Run ID: {parent_run.info.run_id}")

            models = load_ml_models()
            config = load_config("config.yaml")

            # Loading the data
            train_data_path = config["data"]["train_data"]
            logger.info(f"Loading training data from: {train_data_path}")
            train_data = pd.read_csv(train_data_path)
            mlflow.log_param("train_data_path", train_data_path)

            # Splitting the data
            columns_to_drop = ["target", "failure_type"]
            logger.info("Splitting data into training and test sets...")
            (X_train, X_test, y1_train, y1_test, y2_train, y2_test) = handle_data_split(
                data=train_data, test_size=0.2, columns_to_drop=columns_to_drop
            )

            # Scaling for KNN
            logger.info("Scaling features for distance-based models (KNN)...")
            X_train_scale = handle_scale(X_train, is_train=True)
            X_test_scale = handle_scale(X_test, is_train=False)

            y_train_combined = pd.concat([y1_train, y2_train], axis=1)
            y_test_combined = pd.concat([y1_test, y2_test], axis=1)

            training_tasks = {
                "random_forest": {
                    "model": models.get("random_forest"),
                    "X_train": X_train,
                    "X_test": X_test,
                },
                "xgboost": {
                    "model": MultiOutputClassifier(models.get("xgboost")),
                    "X_train": X_train,
                    "X_test": X_test,
                },
                "knn": {
                    "model": models.get("knn"),
                    "X_train": X_train_scale,
                    "X_test": X_test_scale,
                },
            }

            all_results = {}

            # Training each model
            for task_name, task_data in training_tasks.items():
                model = task_data["model"]
                if model is None:
                    logger.warning(f"Model for {task_name} is missing. Skipping task.")
                    continue

                X_train_model = task_data["X_train"]
                X_test_model = task_data["X_test"]

                with mlflow.start_run(run_name=f"Trial_{task_name}", nested=True):
                    logger.info(f"--- Training {task_name} ---")

                    # Training
                    trained_model = model.fit(X_train_model, y_train_combined)
                    mlflow.log_params(model.get_params())

                    # Prediction and Signature
                    y_pred = trained_model.predict(X_test_model)
                    signature = infer_signature(
                        X_train_model, trained_model.predict(X_train_model)
                    )
                    input_example = X_train_model.iloc[:5]

                    # Metrics
                    test_metrics = evaluate_multioutput(y_test_combined, y_pred)
                    logger.info(
                        f"Results for {task_name}:"
                        f"Overall F1 = {test_metrics['overall_f1']:.4f}"
                    )

                    for metric_name, value in test_metrics.items():
                        mlflow.log_metric(metric_name, value)

                    all_results[task_name] = test_metrics

                    # Save locally
                    model_dir = os.path.join(os.getcwd(), "models")
                    os.makedirs(model_dir, exist_ok=True)
                    model_filename = os.path.join(model_dir, f"{task_name}.pkl")
                    with open(model_filename, "wb") as file:
                        pickle.dump(trained_model, file)

                    # MLflow Model Log
                    mlflow.sklearn.log_model(
                        sk_model=trained_model,
                        artifact_path=task_name,
                        signature=signature,
                        input_example=input_example,
                    )

                    save_location_config("model", f"{task_name}", model_filename)
                    logger.info(f"Completed trial for {task_name}.")

            # Selection of best model
            if all_results:
                best_model_name = max(
                    all_results, key=lambda x: all_results[x]["overall_f1"]
                )
                mlflow.log_param("best_overall_model", best_model_name)
                logger.info(
                    f"FINAL DECISION: Best model is '{best_model_name}'"
                    f"with F1: {all_results[best_model_name]['overall_f1']:.4f}"
                )
            else:
                logger.error("No models were trained successfully.")

            # Save artifacts
            try:
                with open("metrics.json", "w") as f:
                    json.dump(all_results, f, indent=4)
                logger.info("All metrics exported to metrics.json")
            except Exception as e:
                logger.error(f"Failed to save metrics.json: {e}")

    logger.info("Pipeline Execution Finished.")


if __name__ == "__main__":
    train_model()
