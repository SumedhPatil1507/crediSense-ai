"""Institutional-style portfolio backtesting with Almgren–Chriss costs.

Signals are target portfolio weights observed at a period close and are applied
to the following period's returns. Impact coefficients are configurable and
must be calibrated to the instrument, venue, and data frequency being tested.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from src.telemetry import traced_span


@dataclass
class BacktestResult:
    """Backtest output, with period-level returns and execution diagnostics."""

    returns: pd.Series
    equity_curve: pd.Series
    executions: pd.DataFrame
    metrics: dict[str, float]


def almgren_chriss_impact_bps(
    participation: pd.Series,
    *,
    model: str = "quadratic",
    coefficient_bps: float = 10.0,
    volatility_bps: pd.Series | float = 100.0,
    permanent_impact_coefficient_bps: float = 0.0,
) -> pd.Series:
    """Return temporary plus optional permanent market impact in basis points.

    ``participation`` is absolute traded notional divided by dollar ADV (or a
    user-supplied participation rate if ADV is unavailable). The quadratic
    temporary-impact form is ``eta * participation**2``. The empirical
    square-root form is ``Y * volatility_bps * sqrt(participation)``. A
    linear permanent-impact term can be enabled with its own coefficient.
    """
    if coefficient_bps < 0 or permanent_impact_coefficient_bps < 0:
        raise ValueError("Impact coefficients must be non-negative")
    p = participation.astype(float).clip(lower=0.0)
    model = model.lower()
    if model == "quadratic":
        temporary = coefficient_bps * p.pow(2)
    elif model in ("square_root", "sqrt"):
        vol = volatility_bps if isinstance(volatility_bps, pd.Series) else float(volatility_bps)
        temporary = coefficient_bps * vol * np.sqrt(p)
    else:
        raise ValueError("impact_model must be 'quadratic' or 'square_root'")
    return temporary + permanent_impact_coefficient_bps * p


def _as_frame(values: pd.Series | pd.DataFrame, name: str) -> pd.DataFrame:
    if isinstance(values, pd.Series):
        frame = values.to_frame(name=values.name or name)
    elif isinstance(values, pd.DataFrame):
        frame = values.copy()
    else:
        raise TypeError(f"{name} must be a pandas Series or DataFrame")
    if frame.empty or frame.columns.has_duplicates:
        raise ValueError(f"{name} must have rows and unique columns")
    return frame.astype(float)


def _align_input(
    value: Any,
    index: pd.Index,
    columns: pd.Index,
    name: str,
    default: float,
) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame(default, index=index, columns=columns, dtype=float)
    if np.isscalar(value):
        return pd.DataFrame(float(value), index=index, columns=columns)
    frame = _as_frame(value, name)
    if len(frame.columns) == 1 and len(columns) > 1:
        frame = pd.concat([frame.iloc[:, 0]] * len(columns), axis=1)
        frame.columns = columns
    elif len(frame.columns) == len(columns) == 1:
        frame.columns = columns
    elif set(frame.columns) == set(columns):
        frame = frame.reindex(columns=columns)
    else:
        raise ValueError(f"{name} columns must match prices")
    if not frame.index.equals(index):
        frame = frame.reindex(index)
    return frame.fillna(default)


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    """Longest peak-to-recovery interval, including a live unrecovered period."""
    longest = current = 0
    for value in drawdown:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def performance_metrics(
    returns: pd.Series,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    confidence: float = 0.95,
) -> tuple[pd.Series, dict[str, float]]:
    """Calculate annualized risk/return metrics and maximum drawdown duration."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if risk_free_rate <= -1.0:
        raise ValueError("risk_free_rate must be greater than -100%")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    r = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        raise ValueError("At least one finite return is required")
    if (r <= -1.0).any():
        raise ValueError("Returns must be greater than -100%")

    period_rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = r - period_rf
    std = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
    sharpe = float(np.sqrt(periods_per_year) * excess.mean() / std) if std > 0 else 0.0
    downside = np.minimum(excess.to_numpy(), 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    sortino = (float(np.sqrt(periods_per_year) * excess.mean() / downside_deviation)
               if downside_deviation > 0 else 0.0)

    equity = (1.0 + r).cumprod()
    peaks = equity.cummax()
    drawdown = equity / peaks - 1.0
    max_drawdown = float(drawdown.min())
    years = len(r) / periods_per_year
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0

    losses = -r
    var = float(losses.quantile(confidence))
    tail_losses = losses[losses >= var]
    expected_shortfall = float(tail_losses.mean()) if not tail_losses.empty else var
    metrics = {
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": cagr,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": float(calmar),
        "max_drawdown": max_drawdown,
        "max_drawdown_duration_periods": float(_max_drawdown_duration(drawdown)),
        "expected_shortfall_95": expected_shortfall if confidence == 0.95 else float("nan"),
        "cvar_95": expected_shortfall if confidence == 0.95 else float("nan"),
        "expected_shortfall": expected_shortfall,
        "expected_shortfall_confidence": confidence,
        "period_volatility": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
    }
    return equity, metrics


def run_backtest(
    prices: pd.Series | pd.DataFrame,
    signals: pd.Series | pd.DataFrame,
    *,
    initial_capital: float = 1_000_000.0,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    spread_bps: Any = 0.0,
    adv: Any = None,
    volume: Any = None,
    impact_model: str = "quadratic",
    impact_coefficient_bps: float = 10.0,
    volatility_bps: Any = 100.0,
    permanent_impact_coefficient_bps: float = 0.0,
    participation_rate: Any = None,
    commission_bps: float = 0.0,
    confidence: float = 0.95,
) -> BacktestResult:
    """Backtest target weights with variable spread and Almgren–Chriss costs.

    Parameters
    ----------
    prices:
        Period close prices indexed by time, one column per asset.
    signals:
        Target portfolio weights. A signal at row *t* is held over the return
        from *t* to *t+1*, preventing same-period lookahead. The final signal
        is not traded because no following price return is available.
    spread_bps:
        Variable full bid-ask spread in bps, scalar or time/asset-aligned data.
        Half the spread is charged for each unit of turnover.
    adv / volume:
        ADV in dollar notional, or historical share volume (estimated with a
        20-period rolling mean and multiplied by price). If neither is given,
        ``participation_rate`` may supply participation; otherwise turnover is
        used as a documented fallback proxy.
    impact_model:
        ``quadratic`` charges ``eta * participation²`` bps. ``square_root``
        charges ``Y * volatility_bps * sqrt(participation)`` bps.

    Returns net returns, equity, per-period execution diagnostics, and
    annualized Sharpe/Sortino/Calmar, 95% expected shortfall, and drawdown
    duration. Coefficients and data units require market-specific calibration.
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if commission_bps < 0:
        raise ValueError("commission_bps must be non-negative")
    price_frame = _as_frame(prices, "prices")
    signal_frame = _as_frame(signals, "signals")
    if len(price_frame.columns) == len(signal_frame.columns) == 1:
        signal_frame.columns = price_frame.columns
    if len(price_frame.index) < 2:
        raise ValueError("At least two price observations are required")
    if not price_frame.index.is_monotonic_increasing or not price_frame.index.is_unique:
        raise ValueError("prices index must be unique and increasing")
    if set(signal_frame.columns) != set(price_frame.columns):
        raise ValueError("signals columns must match prices")
    signal_frame = signal_frame.reindex(index=price_frame.index,
                                        columns=price_frame.columns)
    if signal_frame.iloc[:-1].isna().any().any():
        raise ValueError("signals must be present for all traded periods")
    if price_frame.isna().any().any() or (price_frame <= 0).any().any():
        raise ValueError("prices must be finite positive values")
    if not np.isfinite(price_frame.to_numpy()).all() or not np.isfinite(
            signal_frame.iloc[:-1].to_numpy()).all():
        raise ValueError("prices and signals must be finite")

    index = price_frame.index[1:]
    columns = price_frame.columns
    asset_returns = price_frame.pct_change(fill_method=None).iloc[1:]
    target_weights = signal_frame.iloc[:-1].to_numpy(dtype=float)
    target_frame = pd.DataFrame(target_weights, index=index, columns=columns)
    gross_returns = (target_frame * asset_returns).sum(axis=1)

    spread = _align_input(spread_bps, index, columns, "spread_bps", 0.0)
    vol_bps = _align_input(volatility_bps, index, columns, "volatility_bps", 100.0)
    if (spread < 0).any().any() or (vol_bps < 0).any().any():
        raise ValueError("spread and volatility inputs must be non-negative")

    if adv is not None:
        dollar_adv = _align_input(adv, index, columns, "adv", 0.0)
    elif volume is not None:
        share_volume = _align_input(volume, price_frame.index, columns, "volume", 0.0)
        rolling_shares = share_volume.shift(1).rolling(20, min_periods=1).mean()
        dollar_adv = (rolling_shares.mul(price_frame)).iloc[1:]
        dollar_adv.index = index
    else:
        dollar_adv = pd.DataFrame(0.0, index=index, columns=columns)

    supplied_participation = (_align_input(participation_rate, index, columns,
                                            "participation_rate", 0.0).abs()
                              if participation_rate is not None else None)
    impact = pd.DataFrame(index=index, columns=columns, dtype=float)
    participation = pd.DataFrame(index=index, columns=columns, dtype=float)
    turnover = pd.DataFrame(index=index, columns=columns, dtype=float)
    asset_costs = pd.Series(index=index, dtype=float)
    half_spread_cost_bps = pd.Series(index=index, dtype=float)
    impact_cost_bps = pd.Series(index=index, dtype=float)
    net_returns = pd.Series(index=index, dtype=float)
    current_equity = float(initial_capital)
    held_weights = pd.Series(0.0, index=columns)
    for timestamp in index:
        row_target = target_frame.loc[timestamp]
        row_turnover = (row_target - held_weights).abs()
        turnover.loc[timestamp] = row_turnover
        if supplied_participation is not None:
            row_participation = supplied_participation.loc[timestamp]
        else:
            dollar_traded = row_turnover * current_equity
            row_participation = dollar_traded.divide(
                dollar_adv.loc[timestamp].where(dollar_adv.loc[timestamp] > 0))
            row_participation = row_participation.fillna(row_turnover)
        if (row_participation > 1.0).any():
            raise ValueError("participation exceeds 100%; check ADV units or trade sizing")
        participation.loc[timestamp] = row_participation
        for column in columns:
            if row_turnover[column] > 0:
                with traced_span("trading.order.execute",
                                 **{"trading.asset": str(column),
                                    "trading.impact_model": impact_model,
                                    "trading.turnover": float(row_turnover[column]),
                                    "trading.participation": float(row_participation[column])}):
                    impact.loc[timestamp, column] = almgren_chriss_impact_bps(
                        pd.Series([row_participation[column]]), model=impact_model,
                        coefficient_bps=impact_coefficient_bps,
                        volatility_bps=float(vol_bps.loc[timestamp, column]),
                        permanent_impact_coefficient_bps=permanent_impact_coefficient_bps,
                    ).iloc[0]
            else:
                impact.loc[timestamp, column] = 0.0
        period_half_spread = float((row_turnover * spread.loc[timestamp] / 2.0).sum())
        period_impact = float((row_turnover * impact.loc[timestamp]).sum())
        period_commission = float(row_turnover.sum() * commission_bps)
        period_cost = (period_half_spread + period_impact + period_commission) / 10_000.0
        asset_costs.loc[timestamp] = period_cost
        total_turnover = float(row_turnover.sum())
        half_spread_cost_bps.loc[timestamp] = (period_half_spread / total_turnover
                                                if total_turnover else 0.0)
        impact_cost_bps.loc[timestamp] = (period_impact / total_turnover
                                          if total_turnover else 0.0)
        net_returns.loc[timestamp] = gross_returns.loc[timestamp] - period_cost
        current_equity *= 1.0 + net_returns.loc[timestamp]
        # Carry the post-return weights into the next rebalance so turnover
        # includes drift between assets rather than only signal changes.
        denominator = 1.0 + net_returns.loc[timestamp]
        if denominator <= 0:
            raise ValueError("Portfolio equity reached zero or below")
        held_weights = row_target * (1.0 + asset_returns.loc[timestamp]) / denominator

    equity, metrics = performance_metrics(
        net_returns, periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate, confidence=confidence)
    executions = pd.DataFrame({
        "gross_return": gross_returns,
        "turnover": turnover.sum(axis=1),
        "half_spread_cost_bps": half_spread_cost_bps,
        "impact_cost_bps": impact_cost_bps,
        "commission_cost_bps": commission_bps,
        "total_cost_return": asset_costs,
        "net_return": net_returns,
        "equity": equity * initial_capital,
    }, index=index)
    equity = equity * initial_capital
    equity.name = "equity"
    net_returns.name = "net_return"
    return BacktestResult(returns=net_returns, equity_curve=equity,
                          executions=executions, metrics=metrics)


def run_backtest_metrics(*args: Any, **kwargs: Any) -> Mapping[str, float]:
    """Convenience wrapper returning only the performance metric mapping."""
    return run_backtest(*args, **kwargs).metrics
