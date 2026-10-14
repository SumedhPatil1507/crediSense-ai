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


def test_square_root_impact_rejects_missing_or_negative_volatility():
    participation = pd.Series([0.25, 1.0])
    with pytest.raises(ValueError, match="volatility_bps"):
        almgren_chriss_impact_bps(participation, model="square_root", volatility_bps=None)
    with pytest.raises(ValueError, match="volatility_bps"):
        almgren_chriss_impact_bps(
            participation, model="square_root", volatility_bps=-10.0)


def test_impact_preserves_series_index_and_does_not_mutate_input():
    participation = pd.Series([0.2, 0.8], index=["open", "close"])
    original = participation.copy()

    result = almgren_chriss_impact_bps(participation, coefficient_bps=5.0)

    pd.testing.assert_series_equal(
        result, pd.Series([0.2, 3.2], index=["open", "close"]))
    pd.testing.assert_series_equal(participation, original)
