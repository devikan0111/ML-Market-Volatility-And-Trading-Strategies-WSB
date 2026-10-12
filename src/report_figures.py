"""Static figures (PNG) used by the report and slides. Headless-safe (Agg backend)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import SENT_FEATURES

BLUE, ORANGE, GREY, GREEN, RED = "#1f77b4", "#e6550d", "#7f7f7f", "#2ca02c", "#d62728"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_pred_vs_actual(final: pd.DataFrame, path):
    tickers = sorted(final["Ticker"].unique())[:6]
    cols = 2 if len(tickers) > 1 else 1
    rows = int(np.ceil(len(tickers) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.6 * rows), squeeze=False, sharex=True)
    for ax, t in zip(axes.ravel(), tickers):
        d = final[final["Ticker"] == t]
        ax.plot(d["Date"], d["Actual_Volatility"], color=GREY, lw=1.2, label="Actual (next-5-day realised)")
        ax.plot(d["Date"], d["Predicted_Volatility"], color=BLUE, lw=1.4, label="Predicted")
        ax.set_title(t, fontsize=10, loc="left")
        ax.tick_params(axis="x", rotation=25)
    axes.ravel()[0].legend(fontsize=8, loc="upper left")
    fig.supylabel("Daily volatility (std of log returns)", fontsize=9)
    return _save(fig, path)


def fig_scatter(final: pd.DataFrame, path):
    d = final.dropna(subset=["Actual_Volatility"])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.scatter(d["Actual_Volatility"], d["Predicted_Volatility"], s=8, alpha=0.4, color=BLUE)
    lim = max(d["Actual_Volatility"].max(), d["Predicted_Volatility"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color=ORANGE, lw=1.2, ls="--", label="perfect prediction")
    ax.set_xlabel("Actual volatility"); ax.set_ylabel("Predicted volatility")
    ax.set_title("Out-of-sample predictions", loc="left"); ax.legend(fontsize=8)
    return _save(fig, path)


def fig_importance(imp: pd.Series, path, top=12):
    s = imp.head(top)[::-1]
    colors = [ORANGE if c in SENT_FEATURES else BLUE for c in s.index]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.barh(s.index, s.values, color=colors)
    ax.set_title("Feature importance (orange = Reddit features)", loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def fig_model_compare(metrics: dict, path):
    order = [("baseline_rv5", "Baseline:\nlast 5-day vol"), ("baseline_rv20", "Baseline:\nlast 20-day vol"),
             ("random_forest_price_only", "RF\nprice only"), ("random_forest", "RF\n+ Reddit"),
             ("xgboost_price_only", "XGB\nprice only"), ("xgboost", "XGB\n+ Reddit")]
    order = [(k, l) for k, l in order if k in metrics]
    vals = [metrics[k]["RMSE"] for k, _ in order]
    colors = [GREY if k.startswith("baseline") else (ORANGE if not k.endswith("price_only") else BLUE) for k, _ in order]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.bar([l for _, l in order], vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Out-of-sample RMSE (lower = better)")
    ax.set_title("Volatility prediction error", loc="left"); ax.grid(axis="x", visible=False)
    return _save(fig, path)


def fig_attention_vs_vol(final: pd.DataFrame, path):
    d = final.dropna(subset=["Actual_Volatility"])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].scatter(np.log1p(d["Post_Count"]), d["Actual_Volatility"], s=8, alpha=0.35, color=ORANGE)
    axes[0].set_xlabel("log(1 + Reddit posts that day)"); axes[0].set_ylabel("Next-5-day realised volatility")
    axes[0].set_title("Attention vs future volatility", loc="left", fontsize=10)
    axes[1].scatter(d["Sentiment_Score"], d["Actual_Volatility"], s=8, alpha=0.35, color=BLUE)
    axes[1].set_xlabel("Daily sentiment (VADER compound)"); axes[1].set_title("Sentiment vs future volatility", loc="left", fontsize=10)
    return _save(fig, path)


def fig_equity(port: pd.DataFrame, curves: dict, path, title):
    """curves: {label: column name in port}; port has Date + cumulative columns."""
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    styles = [(BLUE, "-"), (GREY, "--"), (ORANGE, "-"), (GREEN, "-")]
    for (label, col), (c, ls) in zip(curves.items(), styles):
        ax.plot(port["Date"], port[col], label=label, color=c, ls=ls, lw=2 if c == BLUE else 1.5)
    ax.axhline(1, color="k", lw=0.5)
    ax.set_ylabel("Growth of 1.0"); ax.set_title(title, loc="left"); ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=25)
    return _save(fig, path)


def fig_sensitivity(grid: pd.DataFrame, path):
    """Heatmap of train Sharpe over (sentiment threshold x volatility quantile)."""
    piv = grid.pivot(index="vol_q", columns="sent_thr", values="train_sharpe")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{i:g}" for i in piv.index])
    ax.set_xlabel("Sentiment threshold"); ax.set_ylabel("Volatility quantile cut-off (1 = no filter)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
    ax.grid(False); fig.colorbar(im, ax=ax, label="Train Sharpe")
    ax.set_title("Parameter search (training period only)", loc="left", fontsize=10)
    return _save(fig, path)
