import pandas as pd
import os
import pytest

from src.utensil.save_file import save_file


@pytest.mark.data
def test_data_loading(raw_data_path):
    raw_data = pd.read_csv(raw_data_path)

    assert not raw_data.empty
    assert isinstance(raw_data, pd.DataFrame)


@pytest.mark.data
def test_data_columns(preprocessing_data):
    expected_cols = [
        "type",
        "air_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "temp_diff_k",
        "target",
        "failure_type",
    ]

    assert all(col for col in preprocessing_data.columns if col in expected_cols)


@pytest.mark.data
def test_null_vals(preprocessing_data):
    assert preprocessing_data.isna().sum().sum() == 0


@pytest.mark.data
def test_data_saved(tmp_path):
    d = tmp_path / "data" / "interim"
    d.mkdir(parents=True)
    file_path = d / "cleaned_df.csv"

    data = {'target': [0, 1], 'failure_type': ['No Failure', 'Power Failure']}
    df_dummy = pd.DataFrame(data)

    save_file(str(file_path), df_dummy)

    assert file_path.exists(), "File was not saved at {file_path}"
    assert os.path.getsize(file_path) > 0, "Saved file is empty"

    df_loaded = pd.read_csv(file_path)
    assert not df_loaded.empty
    assert {'target', 'failure_type'}.issubset(df_loaded.columns)


@pytest.mark.data
def test_output_type(preprocessing_data):
    assert isinstance(preprocessing_data, pd.DataFrame)
