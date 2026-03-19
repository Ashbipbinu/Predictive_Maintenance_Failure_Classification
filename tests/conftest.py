import pytest
import os

from src.data.data_preprocessing import data_load_preprocessing


@pytest.fixture(scope="session")
def path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..")

    return project_root


# For Data
@pytest.fixture
def raw_data_path(path):
    data_path = os.path.join(path, "data", "raw", "predictive_maintenance.csv")

    return data_path


@pytest.fixture
def preprocessing_data(raw_data_path):
    df_cleaned = data_load_preprocessing(raw_data_path)

    return df_cleaned
