import dagshub
import mlflow
import json


def log_to_mlflow():

    dagshub.init(
        repo_owner="ashbipbinu",
        repo_name="Predictive_Maintenance_Failure_Classification",
        mlflow=True,
    )

    with mlflow.start_run(run_name="RandomForest"):
        # Log metrics
        for col in ["target", "failure_type"]:
            with open(f"test_metric-{col}.json", "r") as file:
                test_metrics = json.load(file)
                mlflow.log_metrics(
                    {f"{col}_{ky}": val for ky, val in test_metrics.items()}
                )

            # Log artifacts
            mlflow.log_artifact(f"confusion-{col}.png")

    # Log Code Version (Optional but pro)
    mlflow.set_tag("dvc_repro", "true")


if __name__ == "__main__":
    log_to_mlflow()
