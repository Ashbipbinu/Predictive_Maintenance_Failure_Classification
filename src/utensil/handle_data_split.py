import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

with open("params.yaml", "rb") as file:
    params = yaml.safe_load(file)

seed = params["base"]["random_state"]


def handle_data_split(
    data: pd.DataFrame, test_size: float, columns_to_drop: list[str]
) -> pd.DataFrame:

    X = data.drop(columns=columns_to_drop)
    y1 = data[columns_to_drop[0]]
    y2 = data[columns_to_drop[1]]

    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
        X, y1, y2, test_size=test_size, random_state=seed, stratify=y2
    )

    return (X_train, X_test, y1_train, y1_test, y2_train, y2_test)
