"""MEMBER 2: honest strategy evaluation.

Thresholds are chosen on the FIRST 60% of dates only (grid search, best Sharpe of the equal-weight portfolio);
results are then reported on the LAST 40% the search never saw, next to Buy & Hold.

    python evaluate_strategy.py                      # uses data/final_output.csv
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from src import report_figures as RF
from src.backtester import ALL, compute_metrics, run_backtest
from src.strategy import generate_signals, load_data

SENT_GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
VOL_Q_GRID = [0.5, 0.6, 0.75, 0.9, 1.0]       # 1.0 = no volatility filter
COST_BPS = 5.0


def vol_threshold(df, q):
    return np.inf if q >= 1.0 else float(df["Predicted_Volatility"].quantile(q))


def backtest(df, sent_thr, vol_thr, allow_short=True):
    sig = generate_signals(df, sent_thr, vol_thr, allow_short)
    bt, _ = run_backtest(sig, cost_bps=COST_BPS)
    return bt


def slice_metrics(bt, lo, hi, benchmark=False):
    out = {}
    for t, d in bt.groupby("Ticker"):
        d = d[(d["Date"] >= lo) & (d["Date"] <= hi)]
        if len(d) < 5:
            continue
        ret, pos = (d["Market_Return"], pd.Series(1.0, index=d.index)) if benchmark else (d["Strategy_Return"], d["Position"])
        m = compute_metrics(ret, d["Market_Return"], pos)
        out[t] = {"Return (%)": m["Cumulative Strategy Return (%)"], "Sharpe": m["Sharpe Ratio"],
                  "Max Drawdown (%)": m["Max Drawdown (%)"], "Win Rate (%)": m["Win Rate (%)"],
                  "Time in market (%)": round(float((d["Position"] != 0).mean() * 100), 1) if not benchmark else 100.0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.FINAL_CSV))
    ap.add_argument("--results-dir", default=str(C.RESULTS))
    ap.add_argument("--train-frac", type=float, default=0.6)
    a = ap.parse_args()
    res = Path(a.results_dir); res.mkdir(parents=True, exist_ok=True)

    df = load_data(a.data)
    dates = np.sort(df["Date"].unique())
    split = dates[int(len(dates) * a.train_frac)]
    train_hi, test_lo, test_hi = dates[int(len(dates) * a.train_frac) - 1], split, dates[-1]
    train_df = df[df["Date"] < split]

    # ---- grid search on the training period only
    rows = []
    for st, vq in itertools.product(SENT_GRID, VOL_Q_GRID):
        bt = backtest(df, st, vol_threshold(train_df, vq))
        m = slice_metrics(bt, dates[0], train_hi).get(ALL)
        rows.append({"sent_thr": st, "vol_q": vq, "train_sharpe": m["Sharpe"], "train_return": m["Return (%)"]})
    grid = pd.DataFrame(rows)
    best = grid.sort_values(["train_sharpe", "train_return"], ascending=False).iloc[0]
    st_best, vq_best = float(best.sent_thr), float(best.vol_q)
    vt_best = vol_threshold(train_df, vq_best)

    # ---- evaluate on the unseen test period
    variants = {
        "Buy & Hold": slice_metrics(backtest(df, st_best, vt_best), test_lo, test_hi, benchmark=True),
        "Sentiment only (no vol filter)": slice_metrics(backtest(df, st_best, np.inf), test_lo, test_hi),
        "Sentiment + volatility filter (long/short)": slice_metrics(backtest(df, st_best, vt_best), test_lo, test_hi),
        "Sentiment + volatility filter (long-only)": slice_metrics(backtest(df, st_best, vt_best, False), test_lo, test_hi),
    }
    # equity curves on the test period (equal-weight portfolio)
    curves = pd.DataFrame({"Date": dates[dates >= test_lo]})
    for label, (vthr, short) in {"Sentiment + vol filter": (vt_best, True), "Sentiment only": (np.inf, True)}.items():
        d = backtest(df, st_best, vthr, short)
        d = d[(d["Ticker"] == ALL) & (d["Date"] >= test_lo)].set_index("Date")
        curves[label] = (1 + d["Strategy_Return"]).cumprod().reindex(curves["Date"]).to_numpy()
    d = backtest(df, st_best, vt_best)
    d = d[(d["Ticker"] == ALL) & (d["Date"] >= test_lo)].set_index("Date")
    curves["Buy & Hold"] = (1 + d["Market_Return"]).cumprod().reindex(curves["Date"]).to_numpy()

    RF.fig_equity(curves, {"Strategy (sentiment + vol filter)": "Sentiment + vol filter",
                           "Buy & Hold": "Buy & Hold", "Sentiment only": "Sentiment only"},
                  res / "fig_equity_test.png", "Equal-weight portfolio, unseen test period")
    RF.fig_sensitivity(grid, res / "fig_sensitivity.png")
    grid.to_csv(res / "strategy_grid_train.csv", index=False)

    out = {"split_date": str(pd.Timestamp(split).date()), "train_from": str(pd.Timestamp(dates[0]).date()),
           "train_to": str(pd.Timestamp(train_hi).date()), "test_from": str(pd.Timestamp(test_lo).date()),
           "test_to": str(pd.Timestamp(test_hi).date()), "n_test_days": int((dates >= test_lo).sum()),
           "best_params": {"sentiment_threshold": st_best, "vol_quantile": vq_best,
                           "vol_threshold": None if np.isinf(vt_best) else round(vt_best, 5)},
           "best_train_sharpe": float(best.train_sharpe), "cost_bps": COST_BPS, "variants": variants}
    (res / "strategy_results.json").write_text(json.dumps(out, indent=2))

    print(f"train {out['train_from']} -> {out['train_to']} | test {out['test_from']} -> {out['test_to']}")
    print(f"best on train: sentiment>{st_best}, vol quantile {vq_best}  (train Sharpe {best.train_sharpe:.2f})")
    for name, per in variants.items():
        m = per.get(ALL, {})
        print(f"  {name:45s} test: return {m.get('Return (%)'):>8}%  Sharpe {m.get('Sharpe'):>6}  MaxDD {m.get('Max Drawdown (%)'):>7}%")


if __name__ == "__main__":
    main()
