"""MEMBER 1, module 2b: align Reddit sentiment with prices and engineer model features.

Timing convention (important, avoids look-ahead):
  * A post is assigned to the first trading day on/after its calendar date
    (weekend / holiday posts count for the next trading day).
  * Every feature in row (Ticker, t) uses information up to day t only.
  * Target = realised volatility (std of daily log returns) over days t+1 ... t+HORIZON.
"""
import numpy as np
import pandas as pd

from config import HORIZON

SENT_FEATURES = ["sent", "sent_lag1", "sent_3d", "post_cnt_log", "post_cnt_3d",
                 "post_spike", "avg_score_log", "pos_share", "neg_share"]
PRICE_FEATURES = ["ret_1", "abs_ret_1", "rv_5", "rv_10", "rv_20", "ret_5", "vol_chg"]


def assign_trading_date(post_dates: pd.Series, trading_dates: pd.DatetimeIndex) -> pd.Series:
    """Map each post date to the first trading date >= that date (NaT if after the last one)."""
    td = pd.DatetimeIndex(sorted(trading_dates))
    pos = td.searchsorted(pd.DatetimeIndex(post_dates.dt.normalize()), side="left")
    mapped = pd.Series(pd.NaT, index=post_dates.index, dtype="datetime64[ns]")
    ok = pos < len(td)
    mapped[ok] = td[pos[ok]]
    return mapped


def daily_sentiment(posts: pd.DataFrame, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Aggregate scored posts (one ticker) to one row per trading day. Days with no posts get 0 / neutral."""
    p = posts.copy()
    p["Date"] = assign_trading_date(p["created_at"], trading_dates)
    p = p.dropna(subset=["Date"])
    p["w"] = np.log1p(p["score"].clip(lower=0)) + 1.0          # upvote weight
    p["pos"] = (p["sentiment"] > 0.05).astype(float)
    p["neg"] = (p["sentiment"] < -0.05).astype(float)
    p["ws"] = p["w"] * p["sentiment"]
    g = p.groupby("Date")
    d = pd.DataFrame({
        "Post_Count": g.size(),
        "Sentiment_Score": g["sentiment"].mean(),
        "Sent_Weighted": g["ws"].sum() / g["w"].sum(),
        "total_score": g["score"].sum(),
        "total_comments": g["comms_num"].sum(),
        "avg_len": g["len_of_post"].mean(),
        "pos_share": g["pos"].mean(),
        "neg_share": g["neg"].mean(),
    })
    d = d.reindex(pd.DatetimeIndex(sorted(trading_dates)))
    d["Post_Count"] = d["Post_Count"].fillna(0)
    return d.fillna(0.0).rename_axis("Date").reset_index()


def build_ticker_features(prices: pd.DataFrame, sent: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Merge prices with daily sentiment and compute features + the forward-looking target for ONE ticker."""
    df = prices.merge(sent, on="Date", how="left").sort_values("Date").reset_index(drop=True)
    for c in sent.columns:
        if c != "Date":
            df[c] = df[c].fillna(0.0)

    logp = np.log(df["Close_Price"])
    r = logp.diff()
    f = pd.DataFrame({"Date": df["Date"], "Close_Price": df["Close_Price"]})
    # ---- price / volatility history (known at close of day t)
    f["ret_1"] = r
    f["abs_ret_1"] = r.abs()
    f["rv_5"], f["rv_10"], f["rv_20"] = r.rolling(5).std(), r.rolling(10).std(), r.rolling(20).std()
    f["ret_5"] = logp.diff(5)
    vol = df["Volume"].astype(float)
    f["vol_chg"] = np.log1p(vol) - np.log1p(vol.rolling(20).mean())
    # ---- Reddit sentiment / attention (known at close of day t)
    f["Sentiment_Score"] = df["Sentiment_Score"]
    f["sent"] = df["Sentiment_Score"]
    f["sent_lag1"] = df["Sentiment_Score"].shift(1)
    f["sent_3d"] = df["Sentiment_Score"].rolling(3, min_periods=1).mean()
    lc = np.log1p(df["Post_Count"])
    f["Post_Count"] = df["Post_Count"]
    f["post_cnt_log"] = lc
    f["post_cnt_3d"] = lc.rolling(3, min_periods=1).sum()
    f["post_spike"] = lc - np.log1p(df["Post_Count"].rolling(10, min_periods=1).mean())
    f["avg_score_log"] = np.log1p(df["total_score"].clip(lower=0))
    f["pos_share"], f["neg_share"] = df["pos_share"], df["neg_share"]
    # ---- target: realised vol over the NEXT `horizon` days (std of returns t+1..t+horizon)
    f["Actual_Volatility"] = r.rolling(horizon).std().shift(-horizon)
    return f


def build_feature_table(prices: dict, posts_by_ticker: dict, horizon: int = HORIZON) -> pd.DataFrame:
    frames = []
    for t, px in prices.items():
        px = px.dropna(subset=["Close_Price"]).copy()
        sent = daily_sentiment(posts_by_ticker.get(t, _empty_posts()), pd.DatetimeIndex(px["Date"]))
        ft = build_ticker_features(px, sent, horizon)
        ft.insert(1, "Ticker", t)
        frames.append(ft)
    out = pd.concat(frames, ignore_index=True)
    needed = PRICE_FEATURES + SENT_FEATURES
    out = out.dropna(subset=needed).reset_index(drop=True)      # drops the first ~20 warm-up days per ticker
    return out.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def _empty_posts() -> pd.DataFrame:
    return pd.DataFrame(columns=["created_at", "score", "comms_num", "len_of_post", "sentiment"]).astype(
        {"created_at": "datetime64[ns]", "score": float, "comms_num": float, "len_of_post": float, "sentiment": float})
