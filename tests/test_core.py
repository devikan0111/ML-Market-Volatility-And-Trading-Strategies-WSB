"""Member 2 tests: strategy and backtester."""
import numpy as np
import pandas as pd

from src.backtester import ALL, run_backtest
from src.strategy import generate_signals


def mk(ticker, prices, sent, vol):
    n = len(prices)
    return pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=n), "Ticker": ticker,
                         "Close_Price": prices, "Sentiment_Score": sent, "Predicted_Volatility": vol})


def test_high_vol_means_flat_not_short():
    s = generate_signals(mk("A", [1, 1, 1], [0.9, 0.9, 0.9], [0.5, 0.5, 0.5]))
    assert (s.Signal == 0).all()


def test_short_needs_bearish_sentiment():
    s = generate_signals(mk("A", [1, 1], [-0.9, 0.0], [0.01, 0.01]))
    assert list(s.Signal) == [-1, 0]


def test_infinite_vol_threshold_disables_filter():
    s = generate_signals(mk("A", [1, 1], [0.9, 0.9], [9.0, 9.0]), vol_threshold=np.inf)
    assert (s.Signal == 1).all()


def test_no_lookahead_and_no_cross_ticker_leak():
    df = pd.concat([mk("A", [10, 10, 10], [0.9] * 3, [0.01] * 3),
                    mk("B", [1, 1, 5], [0.0] * 3, [0.01] * 3)], ignore_index=True)
    s = generate_signals(df)
    assert s[s.Ticker == "B"].Position.iloc[0] == 0
    r, _ = run_backtest(s, cost_bps=0)
    assert r[(r.Ticker == "B")].Market_Return.iloc[0] == 0


def test_signal_on_last_day_earns_nothing():
    s = generate_signals(mk("A", [100, 100, 100, 200], [0, 0, 0, 0.9], [0.01] * 4))
    r, m = run_backtest(s, cost_bps=0)
    assert m["A"]["Cumulative Strategy Return (%)"] == 0


def test_always_long_equals_buy_and_hold():
    s = generate_signals(mk("A", [100, 110, 99, 120, 130], [0.9] * 5, [0.01] * 5))
    _, m = run_backtest(s, cost_bps=0)
    assert np.isclose(m["A"]["Cumulative Strategy Return (%)"], m["A"]["Cumulative Benchmark Return (%)"], atol=0.01)


def test_costs_reduce_returns():
    s = generate_signals(mk("A", [100, 101, 102, 103, 104, 105], [0.9, -0.9, 0.9, -0.9, 0.9, -0.9], [0.01] * 6))
    _, free = run_backtest(s, cost_bps=0)
    _, paid = run_backtest(s, cost_bps=50)
    assert paid["A"]["Cumulative Strategy Return (%)"] < free["A"]["Cumulative Strategy Return (%)"]


def test_portfolio_row_exists_and_metrics_finite():
    df = pd.concat([mk("A", [1, 2, 1.5, 2], [0.5] * 4, [0.01] * 4), mk("B", [3, 2, 2, 4], [-0.5] * 4, [0.01] * 4)])
    _, m = run_backtest(generate_signals(df))
    assert ALL in m and all(np.isfinite(v) for v in m[ALL].values())


def test_strategy_that_never_trades_has_zero_sharpe_not_garbage():
    from src.backtester import compute_metrics
    flat = pd.Series(np.zeros(50))
    m = compute_metrics(flat, pd.Series(np.random.default_rng(0).normal(0, 0.01, 50)), flat)
    assert m["Sharpe Ratio"] == 0.0 and m["Trades"] == 0
