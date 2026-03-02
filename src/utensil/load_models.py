# Importing models
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.utensil.load_params import load_params


def load_ml_models():
    # Loading the params from the params.yaml
    params = load_params()
    seed = params["base"]["random_state"]

    return {
        "lasso": Lasso(random_state=seed),
        "random": RandomForestClassifier(random_state=seed),
        "xgboost": XGBClassifier(random_state=seed),
    }
