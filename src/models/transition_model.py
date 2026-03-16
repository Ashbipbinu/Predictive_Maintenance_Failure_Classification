import pandas as pd
import dagshub
import yaml
import mlflow
import sys
import logging

from sklearn.metrics import classification_report


def init_dagshub(repo_name, repo_owner):
    try:
        dagshub.init(
            repo_name=repo_name, repo_owner=repo_owner, mlflow=True
            )
        print("Sucessfully initialized dagshub")
    except Exception as e:
        print(f"Error while initializing the dagshub: {e}")
        sys.exit(1)
    return


def run_quality_gate(model, test_df, columns, threshold=0.25, gate_threshold=0.95):
    X_test = test_df.drop(columns=columns)
    # Predicting the model
    for i, col in enumerate(columns):
        y_test = test_df[col]
        y_prob = model.predict_proba(X_test)[i][:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True)

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
    # Initializing the dagshub
    repo_name = "Predictive_Maintenance_Failure_Classification"
    repo_owner = "ashbipbinu"
    init_dagshub(repo_name, repo_owner)

    # Loading the config
    try:
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            print("Successfully loaded config.yaml")
    except Exception as e:
        print(f"Failed loading config.yaml {e}")
        sys.exit(1)

    # Loading the test data
    test_data_path = config["data"]["test_data"]
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
