import pandas as pd
import numpy as np

from src.models.transition_model import run_quality_gate


class DummyModel:
    def predict_proba(self, X):
        n = len(X)

        # Simulating two outputs (target, failure_type)
        prob1 = np.array([[0.1, 0.9]] * n)
        prob2 = np.array([[0.2, 0.8]] * n)

        return [prob1, prob2]


def test_quality_gate_pass():
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [5, 6, 7, 8],
            "target": [1, 1, 1, 1],
            "failure_type": [1, 1, 1, 1],
        }
    )

    model = DummyModel()

    result = run_quality_gate(
        model=model,
        test_df=data,
        columns=["target", "failure_type"],
        threshold=0.25,
        gate_threshold=0.5,  # lowered for testing
    )

    assert result is True


def test_quality_gate_fail():
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [5, 6, 7, 8],
            "target": [0, 0, 0, 0],
            "failure_type": [0, 0, 0, 0],
        }
    )

    model = DummyModel()

    result = run_quality_gate(
        model=model,
        test_df=data,
        columns=["target", "failure_type"],
        threshold=0.25,
        gate_threshold=0.95,
    )

    assert result is False
