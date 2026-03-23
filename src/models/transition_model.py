import pandas as pd
import dagshub
import mlflow
import sys
import logging
import numpy as np
import os

from sklearn.metrics import classification_report

from dotenv import load_dotenv

load_dotenv()


def init_dagshub(repo_name, repo_owner):
    try:
        dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
        print("Sucessfully initialized dagshub")
    except Exception as e:
        print(f"Error while initializing the dagshub: {e}")
        sys.exit(1)
    return


def run_quality_gate(model, test_df, columns, threshold=0.25, gate_threshold=0.95):
    X_test = test_df.drop(columns=columns)
    all_probs = model.predict_proba(X_test)

    # Predicting the model
    for i, col in enumerate(columns):
        y_true = test_df[col]
        probs = all_probs[i]
        y_pred = np.argmax(probs, axis=1)

        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )

        label_report = [
            col
            for col in report.keys()
            if col not in ("accuracy", "macro avg", "weighted avg")
        ]

        for label in label_report:
            recall = report[label]["recall"]
            f1 = report[label]["f1-score"]

            # Checking if they are matching with the gate threshold
            if recall < gate_threshold or f1 < gate_threshold:
                print(f"Failed: {label} below {gate_threshold}")
                return False

    return True


def promote_to_production(model_name):
    """Handles the MLflow stage transition."""
    client = mlflow.tracking.MlflowClient()
    latest_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not latest_versions:
        print("No model version found in Staging.")
        return False

    version = latest_versions[0].version
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Version {version} promoted to Production.")
    logging.info("Model promoted")
    return True


def evaluate_and_transit_model():
    # Loading the token from the environment
    try:
        dagshub.auth.add_app_token(os.getenv("DAGSHUB_TOKEN"))
        print("Success: Logging to Dagshub")
    except Exception as e:
        print(f"Error while loading / authenticating token: {e}")

    # Initializing the dagshub
    repo_owner = (os.getenv("DAGSHUB_USER_NAME"))
    repo_name = (os.getenv("DAGSHUB_REPO_NAME"))
    init_dagshub(repo_name, repo_owner)

    # Loading the test data
    BASE_DIR = os.getcwd()
    test_data_path = os.path.join(BASE_DIR, "data", "processed", "test_processed.csv")
    test_data = pd.read_csv(test_data_path)

    if not test_data.empty:
        # Loading the staging model from dagshub
        model_name = "Predictive_Maintenance_Model"
        model_uri = f"models:/{model_name}/Staging"

        model = mlflow.sklearn.load_model(model_uri)

        # If the model provides an accuracy more than 95%,
        # then the model goes for the production
        columns = ["target", "failure_type"]
        is_production_ready = run_quality_gate(
            model=model, test_df=test_data, columns=columns
        )

        # Whether to productionize the model or not
        if is_production_ready:
            is_production_complete = promote_to_production(model_name)
            if is_production_complete:
                print("Succesfully completed promoting the model")
            else:
                print("Failed while promoting the model")
                sys.exit(1)
        else:
            print("Productionizing model is not went well")
            sys.exit(1)
    else:
        print("Failed loading test file")
        sys.exit(1)


if __name__ == "__main__":
    evaluate_and_transit_model()
