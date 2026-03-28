import pandas as pd
import dagshub
import mlflow.sklearn
import os
import pickle
import numpy as np
import logging

from mlflow.tracking import MlflowClient
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


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
    logger.info("Success: Logging to Dagshub")
except Exception as e:
    logger.error(f"Error while loading / authenticating token: {repr(e)}")

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
    logger.info(f"Fetching production model details for: {model_name}")
    latest_version = client.get_latest_versions(model_name, stages=["Production"])[0]
    run_id = latest_version.run_id

    logger.info(f"Downloading encoder from Run: {run_id}")
    local_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="target_encodings.pkl",
    )
    logger.info("Encoder loaded successfully!")

    with open(local_dir, "rb") as file:
        encoder = pickle.load(file)
    logger.info("Encoder loaded successfully!")

except Exception as e:
    logger.exception(
        f"Critical Error: Could not load encoder. Prediction will fail. : {repr(e)}"
    )

# Loading model
try:
    model = mlflow.sklearn.load_model(model_uri=model_uri)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.exception(f"Loading model failed + {e}")


@app.get("/")
def read_root():
    logger.info("Health check endpoint called")
    return {"status": "API is online", "model_version": model_uri}


@app.post("/predict")
def predict(data: MachineData):

    logger.info("Received prediction request")

    data_dict = data.model_dump()
    df = pd.DataFrame([data_dict])

    # Making prediction
    try:
        prediction = model.predict(df)
        probs = model.predict_proba(df)
        logger.info("Model made prediction successfully")
    except Exception as e:
        logger.exception(f"Error happened while predicting: {repr(e)}")
        return {"error": "Prediction failed"}

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
