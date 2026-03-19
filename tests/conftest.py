import pytest
import yaml

from src.data.data_preprocessing import data_load_preprocessing


@pytest.fixture(scope="session")
def config():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    return config


# For Data
@pytest.fixture
def raw_data_path(config):
    data_path = config["data"]["raw_df"]

    return data_path


@pytest.fixture
def preprocessing_data(raw_data_path):
    df_cleaned = data_load_preprocessing(raw_data_path)

    return df_cleaned
