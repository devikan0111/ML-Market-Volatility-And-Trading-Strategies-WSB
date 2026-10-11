"""Streamlit demo (MEMBER 2).   Run:  streamlit run app.py"""
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config as C
from src import charts
from src.backtester import ALL, run_backtest
from src.strategy import generate_signals, load_data

st.set_page_config(page_title="WSB Volatility & Trading Strategy", layout="wide")
st.title("📈 WallStreetBets Sentiment & Volatility Trading Strategy")
st.markdown("Predicting short-term market volatility from Reddit r/WallStreetBets sentiment and "
            "backtesting a rule-based trading strategy against Buy & Hold.")


@st.cache_data
def read_csv(path):
    return load_data(path)


# ---- data ------------------------------------------------------------------
upload = st.sidebar.file_uploader("Upload hand-over CSV (optional)", type="csv")
results_dir = C.RESULTS
if upload is not None:
    df_raw = load_data(upload)
elif C.FINAL_CSV.exists():
    df_raw = read_csv(str(C.FINAL_CSV))
else:
    st.error("No data yet. Run `python run_pipeline.py --reddit data/raw/reddit_wsb.csv` first.")
    st.stop()

if "Actual_Volatility" not in df_raw.columns:        # fall back to backward-looking realised vol
    r = df_raw.groupby("Ticker")["Close_Price"].pct_change()
    df_raw["Actual_Volatility"] = r.groupby(df_raw["Ticker"]).transform(lambda s: s.rolling(5, min_periods=2).std())

# ---- sidebar: ranges adapt to the data so sliders are always meaningful ------
st.sidebar.header("Strategy parameters")
s_abs = df_raw["Sentiment_Score"].abs()
s_max = max(0.3, round(float(s_abs.quantile(0.995)), 2))
s_def = float(np.clip(round(float(df_raw["Sentiment_Score"].std()) * 0.5, 2), 0.01, s_max))
sent_thresh = st.sidebar.slider("Sentiment threshold", 0.0, s_max, s_def, 0.01)

v_max = math.ceil(float(df_raw["Predicted_Volatility"].max()) * 1000) / 1000
v_def = round(float(df_raw["Predicted_Volatility"].quantile(0.75)), 3)
use_vol = st.sidebar.checkbox("Use volatility filter", True)
vol_thresh = st.sidebar.slider("Volatility threshold (daily; stay flat above)", 0.0, v_max, min(v_def, v_max),
                               max(v_max / 200, 0.0005), format="%.3f", disabled=not use_vol)
allow_short = st.sidebar.checkbox("Allow shorting", True)
cost_bps = st.sidebar.number_input("Transaction cost (bps per position change)", 0.0, 50.0, 5.0)
ticker = st.sidebar.selectbox("Ticker", [ALL] + sorted(df_raw["Ticker"].unique()))
dmin, dmax = df_raw["Date"].min().date(), df_raw["Date"].max().date()
rng = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(rng, tuple) and len(rng) == 2:
    df_raw = df_raw[(df_raw["Date"] >= pd.Timestamp(rng[0])) & (df_raw["Date"] <= pd.Timestamp(rng[1]))]

# ---- compute ---------------------------------------------------------------
df_signals = generate_signals(df_raw, sent_thresh, vol_thresh if use_vol else np.inf, allow_short)
df_results, all_metrics = run_backtest(df_signals, cost_bps=cost_bps)
metrics = all_metrics[ticker]
d = df_results[df_results["Ticker"] == ticker]

# ---- KPIs ------------------------------------------------------------------
c = st.columns(5)
c[0].metric("Strategy return", f"{metrics['Cumulative Strategy Return (%)']}%")
c[1].metric("Buy & Hold return", f"{metrics['Cumulative Benchmark Return (%)']}%")
c[2].metric("Sharpe ratio", metrics["Sharpe Ratio"], f"B&H: {metrics['Benchmark Sharpe']}", delta_color="off")
c[3].metric("Max drawdown", f"{metrics['Max Drawdown (%)']}%", f"B&H: {metrics['Benchmark Max Drawdown (%)']}%", delta_color="off")
c[4].metric("Win rate (days in market)", f"{metrics['Win Rate (%)']}%", f"{metrics['Trades']} position changes", delta_color="off")
st.caption("Signals use information up to the close of day t and are traded on day t+1 (no look-ahead). "
           "Predicted volatility is out-of-sample (walk-forward model).")
st.divider()


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


tab1, tab2, tab3, tab4 = st.tabs(["Backtest", "Sentiment & volatility", "Model performance", "Data"])

with tab1:
    show(charts.equity_curve(d, ticker))
    if ticker != ALL:
        c1, c2 = st.columns(2)
        with c1:
            show(charts.signals_on_price(d.dropna(subset=["Signal"]), ticker))
        with c2:
            show(charts.drawdown(d, ticker))
    st.subheader("Per-ticker summary")
    st.dataframe(pd.DataFrame(all_metrics).T, use_container_width=True)

with tab2:
    if ticker == ALL:
        st.info("Pick a single ticker in the sidebar to see its sentiment and volatility charts.")
    else:
        raw_t = df_raw[df_raw["Ticker"] == ticker]
        c1, c2 = st.columns(2)
        with c1:
            show(charts.sentiment_trend(raw_t, ticker))
        with c2:
            show(charts.vol_pred_vs_actual(raw_t, ticker))

with tab3:
    summ_path = results_dir / "pipeline_summary.json"
    if summ_path.exists():
        summ = json.loads(summ_path.read_text())
        m = pd.DataFrame(summ["model_metrics"]).T[["RMSE", "MAE", "R2", "Corr", "N"]]
        st.markdown(f"**Best model:** `{summ['best_model']}` · horizon: next {summ['horizon_days']} trading days · "
                    f"Reddit features' share of importance: **{summ['sentiment_importance_share']:.0%}**")
        st.dataframe(m.style.format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "R2": "{:.3f}", "Corr": "{:.3f}", "N": "{:.0f}"}),
                     use_container_width=True)
        cols = st.columns(2)
        for col, f in zip(cols, ["fig_model_compare.png", "fig_importance.png"]):
            if (results_dir / f).exists():
                col.image(str(results_dir / f))
        if (results_dir / "fig_pred_vs_actual.png").exists():
            st.image(str(results_dir / "fig_pred_vs_actual.png"))
    else:
        st.info("Run `python run_pipeline.py` to generate model metrics (results/pipeline_summary.json).")

with tab4:
    cols = ["Date", "Ticker", "Close_Price", "Sentiment_Score", "Predicted_Volatility", "Position", "Strategy_Return"]
    st.dataframe(d[cols].tail(40), use_container_width=True)
    st.download_button("Download backtest results (CSV)", df_results.to_csv(index=False), "backtest_results.csv")
