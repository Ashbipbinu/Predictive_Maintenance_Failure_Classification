from fastapi import FastAPI
import dagshub
import pandas as pd
import mlflow.sklearn
from pydantic import BaseModel

import os
import pickle
import numpy as np


# Interface of the data
class MachineData(BaseModel):
    type: int
    air_temperature_k: float
    rotational_speed_rpm: int
    torque_nm: float
    tool_wear_min: int
    temp_diff_k: float


# Initialize DagsHub
dagshub.init(
    repo_owner="ashbipbinu",
    repo_name="Predictive_Maintenance_Failure_Classification",
    mlflow=True,
)


# Creating FastApi instance
app = FastAPI(title="predictive Maintenance API", version="4.0")

# Loading the champion model
model_name = "Predictive_Maintenance_Model"
model_uri = f"models:/{model_name}/Production"

# Loading model
try:
    model = mlflow.sklearn.load_model(model_uri=model_uri)
    print("Model loaded successfully")
except Exception as e:
    print(f"Loading model failed + {e}")

# Loading the encoder
dir = os.getcwd()
encoder_path = os.path.join(dir, "models", "target_encodings.pkl")
try:
    with open(encoder_path, "rb") as file:
        encoder = pickle.load(file)
    print("Encoder loaded successfully")
except Exception as e:
    print(f"Error loading encoder: {e}")


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
