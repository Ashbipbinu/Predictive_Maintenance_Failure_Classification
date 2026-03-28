import pandas as pd
import dagshub
import mlflow
import sys
import logging
import numpy as np
import os

from sklearn.metrics import classification_report
from dotenv import load_dotenv

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()


def init_dagshub(repo_name, repo_owner):
    try:
        dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
        logger.info(f"Successfully initialized DagsHub for {repo_owner}/{repo_name}")
    except Exception as e:
        logger.critical(f"Failed to initialize DagsHub: {e}")
        sys.exit(1)


def run_quality_gate(model, test_df, columns, gate_threshold=0.95):
    logger.info(f"Starting Quality Gate verification (Threshold: {gate_threshold})...")

    X_test = test_df.drop(columns=columns)
    try:
        all_probs = model.predict_proba(X_test)
    except Exception as e:
        logger.error(f"Prediction failed during quality gate: {e}")
        return False

    for i, col in enumerate(columns):
        y_true = test_df[col]
        probs = all_probs[i]
        y_pred = np.argmax(probs, axis=1)

        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )

        # Filter for actual class labels
        labels = [
            k
            for k in report.keys()
            if k not in ("accuracy", "macro avg", "weighted avg")
        ]

        for label in labels:
            recall = report[label]["recall"]
            f1 = report[label]["f1-score"]

            logger.info(
                f"Column: {col} | Label: {label} -> Recall: {recall:.4f}, F1: {f1:.4f}"
            )

            if recall < gate_threshold or f1 < gate_threshold:
                logger.warning(
                    f"QUALITY GATE FAILED: {col} (Label {label}) is below"
                    f"{gate_threshold}.Model is not production-ready."
                )
                return False

    logger.info("QUALITY GATE PASSED: All metrics meet the production threshold.")
    return True


def promote_to_production(model_name):
    """Handles the MLflow stage transition."""
    client = mlflow.tracking.MlflowClient()

    try:
        latest_versions = client.get_latest_versions(model_name, stages=["Staging"])
        if not latest_versions:
            logger.error(
                f"Promotion failed: No model version"
                f"found in 'Staging' for {model_name}."
            )
            return False

        version = latest_versions[0].version
        run_id = latest_versions[0].run_id

        # Verify artifact (Encoder)
        artifact_path = "target_encodings.pkl"
        try:
            client.download_artifacts(run_id, artifact_path, dst_path="/tmp")
            logger.info(f"Verified artifact: '{artifact_path}' found in Run {run_id}")
        except Exception:
            logger.error(
                f"Critical artifact missing: '{artifact_path}'"
                f"not found in Run {run_id}."
            )
            return False

        # Transition to Production
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(
            f"Model {model_name} version {version} successfully promoted to PRODUCTION."
        )
        return True

    except Exception as e:
        logger.error(f"Unexpected error during promotion: {e}")
        return False


def dagshub_auth_and_init():
    try:
        token = os.getenv("DAGSHUB_TOKEN")
        if token:
            dagshub.auth.add_app_token(token)
            logger.info("DagsHub token authenticated.")

        repo_owner = os.getenv("DAGSHUB_USER_NAME")
        repo_name = os.getenv("DAGSHUB_REPO_NAME")

        if not repo_owner or not repo_name:
            logger.error("DagsHub environment variables are missing.")
            sys.exit(1)

        init_dagshub(repo_name, repo_owner)
    except Exception as e:
        logger.error(f"Environment setup failed: {e}")
        sys.exit(1)


def evaluate_and_transit_model():
    logger.info("--- Starting Model Deployment Pipeline ---")

    # Auth and Init
    dagshub_auth_and_init()

    # Load Data
    try:
        test_data_path = os.path.join(
            os.getcwd(), "data", "processed", "test_processed.csv"
        )
        test_data = pd.read_csv(test_data_path)
        logger.info(f"Test data loaded ({len(test_data)} rows).")
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        sys.exit(1)

    if test_data.empty:
        logger.error("Test data is empty. Cannot perform quality gate.")
        sys.exit(1)

    # Load Staging Model
    model_name = "Predictive_Maintenance_Model"
    model_uri = f"models:/{model_name}/Staging"

    try:
        logger.info(f"Fetching model from {model_uri}...")
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        logger.error(f"Failed to load model from Staging: {e}")
        sys.exit(1)

    # Quality Gate Check
    columns = ["target", "failure_type"]
    if run_quality_gate(model=model, test_df=test_data, columns=columns):
        if promote_to_production(model_name):
            logger.info("Model Deployment Pipeline: SUCCESS.")
        else:
            logger.error("Model Deployment Pipeline: FAILED during stage transition.")
            sys.exit(1)
    else:
        logger.warning(
            "Model Deployment Pipeline: HALTED."
            "Model did not meet quality requirements."
        )
        sys.exit(1)


if __name__ == "__main__":
    evaluate_and_transit_model()
