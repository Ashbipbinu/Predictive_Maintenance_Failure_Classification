from fastapi import FastAPI
import dagshub
import pandas as pd
import mlflow.sklearn
from pydantic import BaseModel

import os
import pickle


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


mlflow.set_tracking_uri(
    "https://dagshub.com/ashbipbinu/"
    "Predictive_Maintenance_Failure_Classification.mlflow"
)

app = FastAPI(title="predictive Maintenance API", version="4.0")

# Loading the champion model
model_name = "Predictive_Maintenance_Model"
model_uri = f"models:/{model_name}/Production"

# Loading model
print("Loading the model")
model = mlflow.sklearn.load_model(model_uri=model_uri)
print("Loading model completed")


@app.get("/")
def read_root():
    return {"status": "API is online", "model_version": model_uri}


@app.post("/predict")
def predict(data: MachineData):

    dir = os.getcwd()
    encoder_path = os.path.join(dir, "models", "target_encodings.pkl")

    with open(encoder_path, "rb") as file:
        encoder = pickle.load(file)

    data_dict = data.model_dump()
    df = pd.DataFrame([data_dict])

    # Making prediction
    print("Prediction started")
    prediction = model.predict(df)
    print("Prediction ended")

    decoder = encoder.inverse_transform([int(prediction.flatten()[1])])

    if prediction.ndim > 1:
        target = int(prediction.flatten()[0])
        failure_type = str(decoder[0])

    return {"target": target, "failure_type": failure_type}
