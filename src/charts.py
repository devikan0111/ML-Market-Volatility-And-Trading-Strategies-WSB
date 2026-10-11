import matplotlib.pyplot as plt


def _style(ax, title, ylabel):
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", rotation=30)


def sentiment_trend(d, ticker):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(d["Date"], d["Sentiment_Score"], color="#9ecae1", label="Sentiment (daily)")
    ax.plot(d["Date"], d["Sentiment_Score"].rolling(5, min_periods=1).mean(),
            color="#08519c", label="5-day avg")
    ax.axhline(0, color="k", lw=0.5)
    ax2 = ax.twinx()
    ax2.plot(d["Date"], d["Close_Price"], color="#e6550d", lw=1.5, label="Close price")
    ax2.set_ylabel("Price ($)")
    _style(ax, f"{ticker}: sentiment vs price", "Sentiment")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    fig.tight_layout(); return fig


def vol_pred_vs_actual(d, ticker):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(d["Date"], d["Predicted_Volatility"], label="Predicted", color="#1f77b4")
    ax.plot(d["Date"], d["Actual_Volatility"], label="Actual (realized)", color="#7f7f7f", ls="--")
    _style(ax, f"{ticker}: predicted vs actual volatility", "Volatility")
    ax.legend(fontsize=8); fig.tight_layout(); return fig


def equity_curve(d, ticker):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(d["Date"], d["Strategy_Cum_Return"], label="WSB Volatility Strategy", color="#1f77b4", lw=2)
    ax.plot(d["Date"], d["Market_Cum_Return"], label="Buy & Hold Benchmark", color="#7f7f7f", ls="--")
    ax.axhline(1, color="k", lw=0.5)
    _style(ax, f"{ticker}: strategy vs Buy & Hold", "Growth multiplier (start = 1.0)")
    ax.legend(); fig.tight_layout(); return fig


def drawdown(d, ticker):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for col, lab, c in [("Strategy_Cum_Return", "Strategy", "#1f77b4"), ("Market_Cum_Return", "Buy & Hold", "#7f7f7f")]:
        eq = d[col]; dd = eq / eq.cummax().clip(lower=1) - 1
        ax.fill_between(d["Date"], dd * 100, 0, alpha=0.35, color=c, label=lab)
    _style(ax, f"{ticker}: drawdown", "Drawdown (%)")
    ax.legend(fontsize=8); fig.tight_layout(); return fig


def signals_on_price(d, ticker):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(d["Date"], d["Close_Price"], color="gray", lw=1)
    up, dn = d[d["Signal"] == 1], d[d["Signal"] == -1]
    ax.scatter(up["Date"], up["Close_Price"], marker="^", color="green", s=25, label="Long signal")
    ax.scatter(dn["Date"], dn["Close_Price"], marker="v", color="red", s=25, label="Short signal")
    _style(ax, f"{ticker}: signals", "Price ($)")
    ax.legend(fontsize=8); fig.tight_layout(); return fig
