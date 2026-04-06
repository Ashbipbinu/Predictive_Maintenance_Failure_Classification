import dagshub
import json
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
import os
import logging
import matplotlib.pyplot as plt

from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

from src.utensil.load_config import load_config

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()


def init_dagshub() -> bool:
    try:
        dagshub.init(
            repo_owner=os.getenv("DAGSHUB_USER_NAME"),
            repo_name=os.getenv("DAGSHUB_REPO_NAME"),
            mlflow=True,
        )
        return True
    except Exception as e:
        logger.error(f"Failed to initialize DagsHub: {e}")
        return False


def log_to_mlflow():
    logger.info("Initializing DagsHub and MLflow tracking...")

    is_dagshub_auth_success = init_dagshub()

    if is_dagshub_auth_success:

        with mlflow.start_run(run_name="RandomForest"):
            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow Run started. Run ID: {run_id}")

            # Log code and config
            logger.info("Logging artifacts (src code and config.yaml)...")
            mlflow.log_artifacts("src", artifact_path="code")
            mlflow.log_artifact("config.yaml")

            # Log metrics and confusion matrices
            for col in ["target", "failure_type"]:
                metric_file = f"test_metric-{col}.json"
                if os.path.exists(metric_file):
                    with open(metric_file, "r") as file:
                        test_metrics = json.load(file)
                        mlflow.log_metrics(
                            {f"{col}_{ky}": val for ky, val in test_metrics.items()}
                        )
                    logger.info(f"Successfully logged metrics for {col}.")
                else:
                    logger.warning(f"Metric file {metric_file} not found. Skipping.")

                save_dir = "reports/figures"
                if os.path.exists(save_dir):
                    plt.savefig(save_dir / f"confusion-{col}.png")
                    mlflow.log_artifact(save_dir / f"confusion-{col}.png")
                else:
                    logger.warning(f"Confusion matrix image {save_dir} not found.")

            # Log Code Version Tag
            mlflow.set_tag("dvc_repro", "true")

            # Loading the best model and data for signature
            try:
                config = load_config("config.yaml")
                best_model_path = config["model"]["random_forest"]

                logger.info(f"Loading model from {best_model_path} for registration...")
                with open(best_model_path, "rb") as file:
                    best_model = pickle.load(file)

                test_data_path = config["data"]["test_data"]
                test_data = pd.read_csv(test_data_path)

                X_data = test_data.drop(columns=["target", "failure_type"])
                signature = infer_signature(X_data, best_model.predict(X_data))
                input_example = X_data.iloc[:5]

                logger.info("Registering model with signature and input example...")
                mlflow.log_artifact("models/target_encodings.pkl")

                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model",
                    registered_model_name="Predictive_Maintenance_Model",
                    signature=signature,
                    input_example=input_example,
                )
            except Exception as e:
                logger.error(f"Error during model registration: {e}")
                return

            # Model Registry Management
            logger.info("Managing Model Registry stages...")
            client = MlflowClient()
            model_name = "Predictive_Maintenance_Model"

            try:
                latest_version_info = client.get_latest_versions(
                    model_name, stages=["None"]
                )

                if latest_version_info:
                    latest_version = latest_version_info[0].version
                    logger.info(
                        f"Transitioning version {latest_version} to 'Staging'..."
                    )

                    client.transition_model_version_stage(
                        name=model_name,
                        version=latest_version,
                        stage="Staging",
                        archive_existing_versions=True,
                    )
                    logger.info(
                        f"Model version {latest_version}"
                        f"successfully promoted to Staging."
                    )
                else:
                    logger.warning(
                        "No new model version found in 'None' stage to transition."
                    )
            except Exception as e:
                logger.error(f"Model Registry transition failed: {e}")

    logger.info("MLflow logging process completed.")


if __name__ == "__main__":
    log_to_mlflow()
