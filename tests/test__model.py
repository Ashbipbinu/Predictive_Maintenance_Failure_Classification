import numpy as np
import os
import json
import pytest


from src.models.predict_model import visualize_metrics


@pytest.mark.model
def test_threshold():
    y_prob = np.array([0.24, 0.26, 0.9])
    threshold = 0.25

    y_pred = (y_prob >= threshold).astype(int)

    assert y_pred[0] == 0
    assert y_pred[1] == 1


@pytest.mark.model
def test_visualize_metrics(tmp_path):

    test_dir = tmp_path / "test_outputs"
    test_dir.mkdir()
    original_dir = os.getcwd()

    report = {"1": {"f1-score": 0.5, "precision": 0.5, "recall": 0.5}}
    threshold = 0.25
    col = "target"
    y_test = [0, 1]
    y_pred = [0, 1]

    try:
        os.chdir(test_dir)
        visualize_metrics(
            report=report, threshold=threshold, col=col, y_test=y_test, y_pred=y_pred
        )
        assert os.path.exists("test_metric-target.json")
    finally:
        os.chdir(original_dir)
    with open("test_metric-target.json", "r") as f:
        data = json.load(f)
        assert "f1_score" in data
