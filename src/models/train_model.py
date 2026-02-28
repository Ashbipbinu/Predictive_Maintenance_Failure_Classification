import pandas as pd
import os
import yaml

# Importing models
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

# getting the params from the params.yaml
def load_params():
    with open("params.yaml", "rb") as f:
        return yaml.safe_load(f)


def load_ml_models():
    # Loading the params from the params.yaml
    params = load_params()
    seed = params["base"]["random_state"]

    # Destructuring the params
    lasso_params = params["lasso"]
    rf_params = params["random_forest"]
    xg_params = params["xgboost"]

    models = [
        {"lasso": Lasso(**lasso_params, random_state=seed)},
        {"random_forest": RandomForestClassifier(**rf_params, random_state=seed)},
        {'xgboost': XGBClassifier(**xg_params, random_state=seed)}
    ]

    
    return models


def train_models():
    
    # Loading the dataset
    file_path = os.path.join(os.getcwd(), 'data', 'processed', 'train.csv')
    df = pd.read_csv(file_path)

    # Split the data
    X = df.drop(columns=['target', 'failure_type'])
    y1 = df['target']
    y2 = df['failure_type']

    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y1, y2, test_size=0.2, random_state=42)

    # Loading the models
    models = load_ml_models()

    for name, model in models:
        if name == 'lasso':
            pass
        else:
            model.fit(X_train, y1_train, y2_train)