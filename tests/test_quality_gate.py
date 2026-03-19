import pandas as pd
import numpy as np
import pytest

from src.models.transition_model import run_quality_gate


class DummyMultiOutputModel:
    def __init__(self, target_probs, failure_probs):
        self.target_probs = target_probs
        self.failure_probs = failure_probs

    def predict_proba(self, X):
        return [self.target_probs, self.failure_probs]


@pytest.mark.model_production
def test_model_production_pass():
    # Sample test data
    data = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5, 6],
            "feature_2": [1, 2, 3, 4, 5, 6],
            "target": [1, 0, 1, 0, 1, 0],
            "failure_type": [1, 0, 2, 0, 3, 0],
        }
    )

    # ✅ Perfect predictions
    target_probs = np.array(
        [
            [0.1, 0.9],  # 1
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.1],  # 0
            [0.1, 0.9],
            [0.9, 0.1],
        ]
    )

    failure_probs = np.array(
        [
            [0, 1, 0, 0, 0, 0],  # 1
            [1, 0, 0, 0, 0, 0],  # 0
            [0, 0, 1, 0, 0, 0],  # 2
            [1, 0, 0, 0, 0, 0],  # 0
            [0, 0, 0, 1, 0, 0],  # 3
            [1, 0, 0, 0, 0, 0],  # 0
        ]
    )

    model = DummyMultiOutputModel(target_probs, failure_probs)

    result = run_quality_gate(
        model=model,
        test_df=data,
        columns=["target", "failure_type"],
        threshold=0.5,
        gate_threshold=0.95,
    )

    assert result is True


@pytest.mark.model_production
def test_model_production_fail():
    data = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5, 6],
            "feature_2": [1, 2, 3, 4, 5, 6],
            "target": [0, 1, 0, 1, 0, 1],
            "failure_type": [0, 1, 0, 2, 0, 3],
        }
    )

    n = len(data)

    target_probs = np.array([[0.1, 0.9]] * n)

    failure_probs = np.zeros((n, 6))
    failure_probs[:, 0] = 1

    model = DummyMultiOutputModel(target_probs, failure_probs)

    result = run_quality_gate(
        model=model,
        test_df=data,
        columns=["target", "failure_type"],
        threshold=0.5,
        gate_threshold=0.95,
    )

    assert result is False
