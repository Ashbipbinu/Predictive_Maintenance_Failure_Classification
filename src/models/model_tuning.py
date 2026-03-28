import mlflow
import mlflow.sklearn
import pandas as pd
import os
import pickle
import json
import logging

from sklearn.model_selection import GridSearchCV
from src.utensil.load_models import load_ml_models
from src.utensil.load_params import load_params
from src.utensil.load_config import load_config
from src.utensil.handle_data_split import handle_data_split
from src.utensil.handle_scalability import handle_scale
from src.utensil.save_location_config import save_location_config

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Model_Tuning")


def train_model():
    logger.info("Starting model tuning pipeline...")

    with mlflow.start_run(run_name="Model_Tuning_Parent") as parent_run:
        logger.info(f"Parent Run ID: {parent_run.info.run_id}")

        try:
            params = load_params()
            models = load_ml_models()
            config = load_config("config.yaml")

            train_data_path = config["data"]["train_data"]
            logger.info(f"Loading training data from: {train_data_path}")
            train_data = pd.read_csv(train_data_path)
            mlflow.log_param("train_data_path", train_data_path)
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return

        # Splitting logic
        columns_to_drop = ["target", "failure_type"]

        logger.info("Performing data splits for standard and scaled models...")
        (X_train, _, y1_binary_train, _, y2_multi_train, _) = handle_data_split(
            data=train_data, test_size=0.2, columns_to_drop=columns_to_drop
        )

        scale_train = handle_scale(train_data, is_train=True)
        scale_train_df = pd.DataFrame(
            scale_train, columns=train_data.columns, index=train_data.index
        )

        (X_scale_train, _, y1_scale_binary_train, _, y2_scale_multi_train, _) = (
            handle_data_split(
                data=scale_train_df, test_size=0.2, columns_to_drop=columns_to_drop
            )
        )

        tuning_tasks = {
            "lasso_binary": {
                "model": models["lasso"],
                "X": X_scale_train,
                "y": y1_scale_binary_train,
            },
            "lasso_multi": {
                "model": models["lasso"],
                "X": X_scale_train,
                "y": y2_scale_multi_train,
            },
            "random_forest_binary": {
                "model": models["random"],
                "X": X_train,
                "y": y1_binary_train,
            },
            "random_forest_multi": {
                "model": models["random"],
                "X": X_train,
                "y": y2_multi_train,
            },
            "xgboost_binary": {
                "model": models["xgboost"],
                "X": X_train,
                "y": y1_binary_train,
            },
            "xgboost_multi": {
                "model": models["xgboost"],
                "X": X_train,
                "y": y2_multi_train,
            },
        }

        all_metrics = {}

        for task_name, task_data in tuning_tasks.items():
            model = task_data["model"]
            X = task_data["X"]
            y = task_data["y"]
            model_type = task_name.split("_")[0]

            with mlflow.start_run(run_name=f"Trial_{task_name}", nested=True):
                logger.info(f"--- Running GridSearchCV for {task_name} ---")

                try:
                    model_params = params[model_type]
                    grid = GridSearchCV(model, model_params, cv=5, n_jobs=-1)
                    grid.fit(X, y)

                    logger.info(f"Best score for {task_name}: {grid.best_score_:.4f}")

                    # Saving the best model
                    best_model = grid.best_estimator_
                    model_dir = os.path.join(os.getcwd(), "models")
                    os.makedirs(model_dir, exist_ok=True)

                    # Use unique filename per task to prevent overwriting during tuning
                    model_filename = os.path.join(model_dir, f"{task_name}_best.pkl")
                    with open(model_filename, "wb") as file:
                        pickle.dump(best_model, file)

                    save_location_config("model", f"best_{task_name}", model_filename)

                    # MLflow Logging
                    mlflow.log_params(grid.best_params_)
                    mlflow.log_metric("best_accuracy", grid.best_score_)
                    mlflow.sklearn.log_model(best_model, artifact_path=task_name)

                    all_metrics[f"{task_name}_acc"] = grid.best_score_

                except Exception as e:
                    logger.error(f"Task {task_name} failed: {e}")
                    continue

        # Finalizing metrics
        try:
            with open("metrics.json", "w") as f:
                json.dump(all_metrics, f, indent=4)
            logger.info("All task metrics saved to metrics.json")
        except Exception as e:
            logger.error(f"Failed to save metrics.json: {e}")

    logger.info("Hyperparameter tuning pipeline finished.")


if __name__ == "__main__":
    train_model()
