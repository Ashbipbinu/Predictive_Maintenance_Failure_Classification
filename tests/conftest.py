import pytest
import os
import pandas as pd

from src.data.data_preprocessing import data_load_preprocessing


@pytest.fixture(scope="session")
def path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..")

    return project_root


# For Data
@pytest.fixture
def raw_data_path(path):
    raw_data_path = os.path.join(path, "data", "raw", "predictive_maintenance.csv")

    return raw_data_path


@pytest.fixture
def preprocessing_data(raw_data_path):
    df_cleaned = data_load_preprocessing(raw_data_path)

    return df_cleaned


@pytest.fixture
def create_demo_data():
    data = {
        "feature1": list(range(100)),
        "feature2": [x * 2 for x in range(100)],
        "target": [0]*50 + [1]*50,
        "failure_type": ["No Failure"]*50 + ["Failure"]*50
    }
    return pd.DataFrame(data)
