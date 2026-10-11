import numpy as np
import pandas as pd

REQUIRED_COLS = ["Date", "Ticker", "Close_Price", "Sentiment_Score", "Predicted_Volatility"]


def load_data(path_or_buffer) -> pd.DataFrame:
    """Read Member 1's CSV, validate columns, sort by Ticker/Date."""
    df = pd.read_csv(path_or_buffer, parse_dates=["Date"])
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def generate_signals(df: pd.DataFrame, sentiment_threshold=0.2, vol_threshold=0.03,
                     allow_short=True) -> pd.DataFrame:
    """
    Rule-based short-term signals (computed at the close of day t):
      - High predicted volatility (>= vol_threshold)  -> FLAT (0): risk-off, stay out
      - Long  (+1): Sentiment >  +threshold AND volatility below threshold
      - Short (-1): Sentiment <  -threshold AND volatility below threshold (if allow_short)
      - Otherwise   -> FLAT (0)
    Position = Signal shifted 1 day, PER TICKER, so we never trade on information
    we could not have had and never leak a position from one ticker into another.
    """
    data = df.sort_values(["Ticker", "Date"]).reset_index(drop=True).copy()
    vol_ok = data["Predicted_Volatility"] < vol_threshold
    sig = np.zeros(len(data), dtype=int)
    sig[(data["Sentiment_Score"] > sentiment_threshold) & vol_ok] = 1
    if allow_short:
        sig[(data["Sentiment_Score"] < -sentiment_threshold) & vol_ok] = -1
    data["Signal"] = sig
    data["Position"] = data.groupby("Ticker")["Signal"].shift(1).fillna(0)
    return data
