"""Unit tests for Almgren-Chriss execution impact calculations."""
import pandas as pd
import pytest

from src.backtest import almgren_chriss_impact_bps


def test_quadratic_temporary_and_linear_permanent_impact():
    participation = pd.Series([0.0, 0.5, 1.0], index=["a", "b", "c"])
    impact = almgren_chriss_impact_bps(
        participation, coefficient_bps=8.0,
        permanent_impact_coefficient_bps=2.0)
    pd.testing.assert_series_equal(
        impact, pd.Series([0.0, 3.0, 10.0], index=participation.index))


def test_square_root_impact_uses_volatility():
    participation = pd.Series([0.25, 1.0])
    impact = almgren_chriss_impact_bps(
        participation, model="square_root", coefficient_bps=0.5,
        volatility_bps=80.0)
    assert impact.tolist() == pytest.approx([20.0, 40.0])


def test_negative_participation_is_clipped_and_invalid_parameters_rejected():
    impact = almgren_chriss_impact_bps(pd.Series([-1.0, 0.5]), coefficient_bps=4.0)
    assert impact.tolist() == pytest.approx([0.0, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        almgren_chriss_impact_bps(pd.Series([0.5]), coefficient_bps=-1.0)
    with pytest.raises(ValueError, match="impact_model"):
        almgren_chriss_impact_bps(pd.Series([0.5]), model="linear")
