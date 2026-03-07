import pandas as pd
import mlflow
import mlflow.sklearn
import os
import pickle
import json

from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score
from mlflow.models import infer_signature

from src.utensil.load_models import load_ml_models
from src.utensil.load_config import load_config
from src.utensil.handle_scalability import handle_scale
from src.utensil.save_location_config import save_location_config
from src.utensil.handle_data_split import handle_data_split

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Predictive_Maintenance")


def evaluate_multioutput(y_true, y_pred):

    metrics = {}

    for i, col in enumerate(["target", "failure_type"]):
        metrics[f"{col}_accuracy"] = accuracy_score(y_true.iloc[:, i], y_pred[:, i])
        metrics[f"{col}_f1_weighted"] = f1_score(
            y_true.iloc[:, i], y_pred[:, i], average="weighted"
        )

    metrics["overall_f1"] = (
        metrics["target_f1_weighted"] + metrics["failure_type_f1_weighted"]
    ) / 2

    return metrics


def train_model():

    with mlflow.start_run(run_name="Parent Optimization Run") as parent_run:
        print(f"parent run id: {parent_run.info.run_id}")

        models = load_ml_models()
        config = load_config("config.yaml")

        # Loading the data
        train_data_path = config["data"]["train_data"]
        train_data = pd.read_csv(train_data_path)

        # Saving the path to MLFLOW
        mlflow.log_param("train_data_path", train_data_path)

        # Splitting the data for models
        columns_to_drop = ["target", "failure_type"]
        (X_train, X_test, y1_train, y1_test, y2_train, y2_test) = handle_data_split(
            data=train_data, test_size=0.2, columns_to_drop=columns_to_drop
        )

        # Sacling the data for KNN
        X_train_scale = handle_scale(X_train, is_train=True)
        X_test_scale = handle_scale(X_test, is_train=False)

        y_train_combined = pd.concat([y1_train, y2_train], axis=1)
        y_test_combined = pd.concat([y1_test, y2_test], axis=1)

        training_tasks = {
            "random_forest": {
                "model": models["random_forest"],
                "X_train": X_train,
                "X_test": X_test,
            },
            "xgboost": {
                "model": MultiOutputClassifier(models["xgboost"]),
                "X_train": X_train,
                "X_test": X_test,
            },
            "knn": {
                "model": models["knn"],
                "X_train": X_train_scale,
                "X_test": X_test_scale,
            },
        }

        # Store the metrics and model location
        all_results = {}

        # Training each mmodel
        for task_name, task_data in training_tasks.items():
            model = task_data["model"]
            X_train_model = task_data["X_train"]
            X_test_model = task_data["X_test"]

            with mlflow.start_run(run_name=f"Trial_{task_name}", nested=True):
                print(f"Training- {task_name} - started")

                # Training the model
                trained_model = model.fit(X_train_model, y_train_combined)

                # Log parameters
                mlflow.log_params(model.get_params())

                # Evaluating the model
                y_pred = trained_model.predict(X_test_model)

                # Infer model signature
                signature = infer_signature(
                    X_train_model, trained_model.predict(X_train_model)
                )
                input_example = X_train_model.iloc[:5]

                # Evaluating model
                test_metrics = evaluate_multioutput(y_test_combined, y_pred)

                # Log metrics to MLflow
                for metric_name, value in test_metrics.items():
                    mlflow.log_metric(metric_name, value)

                all_results[task_name] = test_metrics

                # Saving the model
                model_dir = os.path.join(os.getcwd(), "models")
                os.makedirs(model_dir, exist_ok=True)
                model_filename = os.path.join(model_dir, f"{task_name}.pkl")
                with open(model_filename, "wb") as file:
                    pickle.dump(trained_model, file)

                # Logging model to MLflow
                mlflow.sklearn.log_model(
                    sk_model=trained_model,
                    name=task_name,
                    signature=signature,
                    input_example=input_example,
                )

                # Saving location of the model
                save_location_config("model", f"{task_name}", model_filename)

            print(f"Training- {task_name} - finished")

        # Select best model based on overall F1
        best_model = max(all_results, key=lambda x: all_results[x]["overall_f1"])
        mlflow.log_param("best_model", best_model)
        print(f"Best model selected: {best_model}")

        # Write to metrics.json
        with open("metrics.json", "w") as f:
            json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    train_model()
