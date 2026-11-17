import numpy as np
import pandas as pd
import pytest

from drought_prediction.models.baseline_arima import (
    build_seasonal_persistence_predictions,
)
from drought_prediction.models.evaluate_models import inverse_mse_weights, _metrics


def test_seasonal_persistence_uses_same_month_previous_year(tmp_path):
    features_path = tmp_path / "features.csv"
    reference_path = tmp_path / "reference.csv"
    dates = pd.date_range("2021-01-01", periods=13, freq="MS")
    pd.DataFrame(
        {
            "valid_time": dates,
            "latitude": 10.0,
            "longitude": 105.0,
            "spei_3m": range(13),
        }
    ).to_csv(features_path, index=False)
    pd.DataFrame(
        {
            "valid_time": ["2022-01-01"],
            "latitude": [10.0],
            "longitude": [105.0],
            "actual_spei_3m": [12.5],
        }
    ).to_csv(reference_path, index=False)

    result = build_seasonal_persistence_predictions(features_path, reference_path)

    assert result.loc[0, "input_time"] == pd.Timestamp("2021-01-01")
    assert result.loc[0, "predicted_spei_3m"] == pytest.approx(0.0)
    assert result.loc[0, "actual_spei_3m"] == pytest.approx(12.5)


def test_inverse_mse_weights_are_normalized_and_favor_lower_error():
    weights = inverse_mse_weights({"better": 0.25, "worse": 1.0})

    assert weights["better"] == pytest.approx(0.8)
    assert weights["worse"] == pytest.approx(0.2)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_metrics_use_severe_drought_threshold_for_recall():
    metrics = _metrics(
        actual=np.array([-2.0, -1.6, -1.4, 0.0]),
        predicted=np.array([-1.4, -1.8, -1.7, 0.1]),
    )

    assert metrics["Extreme_Drought_Recall"] == pytest.approx(0.5)
    assert metrics["extreme_drought_count"] == 2
    assert metrics["Bias"] == pytest.approx(0.05)


def test_inverse_mse_weights_reject_invalid_scores():
    with pytest.raises(ValueError, match="finite positive"):
        inverse_mse_weights({"bad": 0.0})
