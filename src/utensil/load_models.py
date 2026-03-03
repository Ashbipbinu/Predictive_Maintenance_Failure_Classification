# Importing models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

from src.utensil.load_params import load_params


def load_ml_models():
    # Loading the params from the params.yaml
    params = load_params()
    seed = params["base"]["random_state"]

    return {
        "random_forest": RandomForestClassifier(random_state=seed),
        "xgboost": XGBClassifier(random_state=seed),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }
