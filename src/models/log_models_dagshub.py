import dagshub

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import json
import pandas as pd
import pickle

from src.utensil.load_config import load_config

print(mlflow.__file__)


def log_to_mlflow():

    dagshub.init(
        repo_owner="ashbipbinu",
        repo_name="Predictive_Maintenance_Failure_Classification",
        mlflow=True,
    )

    with mlflow.start_run(run_name="RandomForest"):

        # Log code
        mlflow.log_artifacts("src", artifact_path="code")

        # Log config.yaml
        mlflow.log_artifact('config.yaml')

        # Log metrics
        for col in ["target", "failure_type"]:
            with open(f"test_metric-{col}.json", "r") as file:
                test_metrics = json.load(file)
                mlflow.log_metrics(
                    {f"{col}_{ky}": val for ky, val in test_metrics.items()}
                )

            # Log artifacts
            mlflow.log_artifact(f"confusion-{col}.png")

        # Log Code Version
        mlflow.set_tag("dvc_repro", "true")

        # Loading the best model
        config = load_config("config.yaml")
        best_model_path = config["model"]["random_forest"]
        with open(best_model_path, "rb") as file:
            best_model = pickle.load(file)

        # Loading the data
        test_data_path = config["data"]["test_data"]
        test_data = pd.read_csv(test_data_path)

        X_data = test_data.drop(columns=["target", "failure_type"])
        signature = infer_signature(X_data, best_model.predict(X_data))
        input_example = X_data.iloc[:5]

        # Registering the best model
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            registered_model_name="Predictive_Maintenance_Model",
            signature=signature,
            input_example=input_example,
        )

        # Updating the staging of the model as 'Champion'
        client = MlflowClient()
        model_name = "Predictive_Maintenance_Model"
        # Latest version info
        latest_version_info = client.get_latest_versions(model_name, stages=["None"])

        if latest_version_info:
            latest_Version = latest_version_info[0].version

            # Transitioning model from None to champion which is on this version,
            # also demote the exisiting model to archieve if any
            client.transition_model_version_stage(
                name=model_name,
                version=latest_Version,
                stage="Production",
                archive_existing_versions=True,
            )

            print("Model is now the Production Champion. ")

        else:
            print("No new model version found to transition.")


if __name__ == "__main__":
    log_to_mlflow()
