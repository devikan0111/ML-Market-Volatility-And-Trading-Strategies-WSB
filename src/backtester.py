import numpy as np
import pandas as pd

TRADING_DAYS = 252
ALL = "ALL (equal-weight)"


def compute_metrics(strategy_ret: pd.Series, market_ret: pd.Series, position: pd.Series,
                    risk_free_rate=0.04) -> dict:
    """Metrics for a strategy return series and its Buy & Hold benchmark."""
    def sharpe(r):
        ex = r - risk_free_rate / TRADING_DAYS
        sd = ex.std()
        # a strategy that never trades has constant returns: std is ~0 (float noise) and Sharpe is undefined -> 0
        return float(np.sqrt(TRADING_DAYS) * ex.mean() / sd) if (not np.isnan(sd) and sd > 1e-9) else 0.0

    def mdd(r):
        eq = (1 + r).cumprod()
        peak = np.maximum(eq.cummax(), 1.0)          # peak includes the starting value 1.0
        return float((eq / peak - 1).min())

    active = strategy_ret[position != 0]
    win = float((active > 0).mean()) if len(active) else 0.0
    return {
        "Cumulative Strategy Return (%)": round(((1 + strategy_ret).prod() - 1) * 100, 2),
        "Cumulative Benchmark Return (%)": round(((1 + market_ret).prod() - 1) * 100, 2),
        "Sharpe Ratio": round(sharpe(strategy_ret), 2),
        "Benchmark Sharpe": round(sharpe(market_ret), 2),
        "Max Drawdown (%)": round(mdd(strategy_ret) * 100, 2),
        "Benchmark Max Drawdown (%)": round(mdd(market_ret) * 100, 2),
        "Win Rate (%)": round(win * 100, 2),
        "Trades": int((position.diff().abs().fillna(position.abs()) > 0).sum()),
    }


def run_backtest(df_with_signals: pd.DataFrame, risk_free_rate=0.04, cost_bps=5.0):
    """
    Backtest every ticker separately (returns are computed PER TICKER) and also an
    equal-weight portfolio of all tickers. Returns (results_df, metrics_by_name).
    results_df has one block of rows per ticker plus a block for ALL (equal-weight).
    """
    df = df_with_signals.sort_values(["Ticker", "Date"]).copy()
    g = df.groupby("Ticker")
    df["Market_Return"] = g["Close_Price"].pct_change().fillna(0)
    turnover = g["Position"].diff().abs().fillna(df["Position"].abs())
    df["Strategy_Return"] = df["Position"] * df["Market_Return"] - turnover * cost_bps / 1e4

    port = (df.groupby("Date")[["Strategy_Return", "Market_Return"]].mean().reset_index())
    port["Ticker"] = ALL
    port["Position"] = df.groupby("Date")["Position"].apply(lambda s: (s != 0).mean()).values
    df = pd.concat([df, port], ignore_index=True, sort=False)

    g = df.groupby("Ticker")
    df["Market_Cum_Return"] = g["Market_Return"].transform(lambda r: (1 + r).cumprod())
    df["Strategy_Cum_Return"] = g["Strategy_Return"].transform(lambda r: (1 + r).cumprod())

    metrics = {}
    for t, d in df.groupby("Ticker"):
        metrics[t] = compute_metrics(d["Strategy_Return"], d["Market_Return"], d["Position"], risk_free_rate)
    return df, metrics
