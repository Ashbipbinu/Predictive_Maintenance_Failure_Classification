import pandas as pd
import dagshub
import mlflow.sklearn
import os
import pickle
import numpy as np

from mlflow.tracking import MlflowClient
from fastapi import FastAPI
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()


# Interface of the data
class MachineData(BaseModel):
    type: int
    air_temperature_k: float
    rotational_speed_rpm: int
    torque_nm: float
    tool_wear_min: int
    temp_diff_k: float


token = os.getenv("DAGSHUB_TOKEN")
repo_owner = os.getenv("DAGSHUB_USER_NAME")
repo_name = os.getenv("DAGSHUB_REPO_NAME")


# Loading the token from the environment
try:
    dagshub.auth.add_app_token(token)
    print("Success: Logging to Dagshub")
except Exception as e:
    print(f"Error while loading / authenticating token: {repr(e)}")


# Initialize DagsHub
dagshub.init(
    repo_owner=repo_owner,
    repo_name=repo_name,
    mlflow=True,
)


# Creating FastApi instance
app = FastAPI(title="predictive Maintenance API", version="4.0")

# Loading the champion model
model_name = "Predictive_Maintenance_Model"
model_uri = f"models:/{model_name}/Production"

# Loading the encoder from the dagshub
client = MlflowClient()
encoder = None

try:
    print(f"Fetching production model details for: {model_name}")
    latest_version = client.get_latest_versions(model_name, stages=["Production"])[0]
    run_id = latest_version.run_id

    print(f"Downloading encoder from Run: {run_id}")
    local_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=f"{model_uri}/code/target_encodings.pkl",
    )

    with open(local_dir, "rb") as file:
        encoder = pickle.load(file)
    print(" Encoder loaded successfully!")

except Exception as e:
    print(
        f"Critical Error: Could not load encoder. Prediction will fail. Error: {e}"
    )

# Loading model
try:
    model = mlflow.sklearn.load_model(model_uri=model_uri)
    print("Model loaded successfully")
except Exception as e:
    print(f"Loading model failed + {e}")


@app.get("/")
def read_root():
    return {"status": "API is online", "model_version": model_uri}


@app.post("/predict")
def predict(data: MachineData):

    data_dict = data.model_dump()
    df = pd.DataFrame([data_dict])

    # Making prediction
    try:
        prediction = model.predict(df)
        probs = model.predict_proba(df)
        print("Model made prediction successfully")
    except Exception as e:
        print(f"Error happened while predicting + {e}")

    if prediction.ndim > 1:
        target = int(prediction.flatten()[0])
        # failure_type = str(decoder[0])

        target_prob = probs[0][0][1]

        threshold = 0.25
        target_pred = int(target_prob >= threshold)

        # Failure type (multiclass)
        failure_probs = probs[1][0]
        failure_pred = int(np.argmax(failure_probs))
        decoded_failure = encoder.inverse_transform([failure_pred])[0]
        failure_prob = float(np.max(failure_probs))

    return {
        "prediction": {
            "target": target_pred,
            "failure_type": decoded_failure,
            "actual_target": target,
        },
        "probabilities": {
            "target_failure_probability": target_prob * 100,
            "failure_type_probability": failure_prob * 100,
        },
    }
