import mlflow
import mlflow.sklearn
import pandas as pd
import os
import pickle
import json

from sklearn.model_selection import GridSearchCV

from src.utensil.load_models import load_ml_models
from src.utensil.load_params import load_params
from src.utensil.load_config import load_config
from src.utensil.handle_data_split import handle_data_split
from src.utensil.handle_scalability import handle_scale
from src.utensil.save_location_config import save_location_config

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Model_Tuning")

with mlflow.start_run(run_name="Model_Tuning") as parent_run:
    print(f"Parent run at {parent_run.info.run_id}")

    params = load_params()
    models = load_ml_models()
    config = load_config("config.yaml")

    # Loading the data
    train_data_path = config["data"]["train_data"]
    train_data = pd.read_csv(train_data_path)

    # Saving the path to MLFLOW
    mlflow.log_param("train_data_path", train_data_path)

    # Splitting the data for models except lasso
    columns_to_drop = ["target", "failure_type"]
    X_train, X_test, y1_binary_train, y1_binary_test, y2_multi_train, y2_multi_test = (
        handle_data_split(
            data=train_data,
            test_size=0.2,
            columns_to_drop=columns_to_drop,
        )
    )

    # Splitting the data for  lasso
    scale_train = handle_scale(train_data, is_train=True)
    scale_train_df = pd.DataFrame(
        scale_train, columns=train_data.columns, index=train_data.index
    )
    (
        X_scale_train,
        X_scale_test,
        y1_scale_binary_train,
        y1_scale_binary_test,
        y2_scale_multi_train,
        y2_scale_multi_test,
    ) = handle_data_split(
        data=scale_train_df,
        test_size=0.2,
        columns_to_drop=columns_to_drop,
    )

    # Differenct task since it is a multi class problem
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
        "xgboostmulti": {
            "model": models["xgboost"],
            "X": X_train,
            "y": y2_multi_train,
        },
    }

    # Tuning each mmodel
    for task_name, task_data in tuning_tasks.items():
        
        model = task_data["model"]
        X = task_data["X"]
        y = task_data["y"]

        with mlflow.start_run(run_name=f"Trial_{task_name}", nested=True):
            model_name = task_name.split('_')[0]
            print(f"Tuning {model_name} - {task_name.split('_')[-1]}")
            model_params = params[model_name]
            grid = GridSearchCV(model, model_params, cv=5)
            grid.fit(X, y)

            # Saving the best model
            best_model = grid.best_estimator_
            model_dir = os.path.join(os.getcwd(), "models")
            os.makedirs(model_dir, exist_ok=True)
            model_filename = os.path.join(model_dir, f"{model_name}.pkl")
            with open(model_filename, "wb") as file:
                pickle.dump(best_model, file)

            # Saving location of the model
            save_location_config("model", "best_model", model_filename)

            # Logging the results
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("Best accuracy", grid.best_score_)
            mlflow.sklearn.log_model(best_model, artifact_path=model_name)
            
            # Write to metrics.json
            with open("metrics.json", "w") as f:
                json.dump({f"{task_name}_acc": grid.best_score_}, f)
