import pandas as pd
import mlflow
import mlflow.sklearn
import os
import pickle
import json

from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier

from src.utensil.load_models import load_ml_models
from src.utensil.handle_data_split import handle_data_split
from src.utensil.load_config import load_config
from src.utensil.handle_scalability import handle_scale
from src.utensil.save_location_config import save_location_config

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Predictive_Maintenance")


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
    X_train, X_test, y1_binary_train, y1_binary_test, y2_multi_train, y2_multi_test = (
        handle_data_split(
            data=train_data,
            test_size=0.2,
            columns_to_drop=columns_to_drop,
        )
    )

    # Scale the features (X) for SVC specifically
    X_train_scaled = handle_scale(X_train, is_train=True)
    X_test_scaled = handle_scale(X_test, is_train=False)

    y_train_combined = pd.concat([y1_binary_train, y2_multi_train], axis=1)

    # Differenct task since it is a multi class problem
    tuning_tasks = {
        "random_forest": {
            "model": models["random_forest"],
            "X": X_train,
            "y": y_train_combined,
        },
        "xgboost": {
            "model": MultiOutputClassifier(models["xgboost"]),
            "X": X_train,
            "y": y_train_combined,
        },
        "knn": {
            "model": models["knn"],
            "X": X_train_scaled,
            "y": y_train_combined,
        },
    }

    # Store the metrics and model location
    all_results = {}
    all_models_loc = {}

    # Training each mmodel
    for task_name, task_data in tuning_tasks.items():
        model = task_data["model"]
        X = task_data["X"]
        y = task_data["y"]

        with mlflow.start_run(run_name=f"Trial_{task_name}", nested=True):
            print(f"Training- {task_name} - Started")

            # Training the model
            trained_model = model.fit(X, y)
            X_eval = X_train_scaled if task_name == "svc" else X_test
            y_eval = pd.concat([y1_binary_test, y2_multi_test], axis=1)

            y_pred = trained_model.predict(X_eval)

            # Calculate Macro F1 for both targets
            # Target 0: Binary, Target 1: Multiclass
            report_y1 = classification_report(
                y_eval.iloc[:, 0], y_pred[:, 0], output_dict=True
            )
            report_y2 = classification_report(
                y_eval.iloc[:, 1], y_pred[:, 1], output_dict=True
            )

            f1_y1 = report_y1["macro avg"]["f1-score"]
            f1_y2 = report_y2["macro avg"]["f1-score"]

            # Log to MLflow
            mlflow.log_metric("f1_macro_binary", f1_y1)
            mlflow.log_metric("f1_macro_multiclass", f1_y2)

            # Store for metrics.json
            all_results[f"{task_name}_binary_f1"] = f1_y1
            all_results[f"{task_name}_multi_f1"] = f1_y2

            # Saving the model
            model_dir = os.path.join(os.getcwd(), "models")
            os.makedirs(model_dir, exist_ok=True)
            model_filename = os.path.join(model_dir, f"{task_name}.pkl")
            with open(model_filename, "wb") as file:
                pickle.dump(trained_model, file)

            # Saving location of the model
            all_models_loc[task_name] = model_filename
            save_location_config("model", f"{task_name}", model_filename)

            print(f"Training- {task_name} - finished")

    # Write to metrics.json
    with open("metrics.json", "w") as f:
        json.dump(all_results, f, indent=4)
