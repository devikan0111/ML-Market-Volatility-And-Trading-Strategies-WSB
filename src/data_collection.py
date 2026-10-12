
import warnings
from pathlib import Path

import pandas as pd


def load_reddit_csv(path) -> pd.DataFrame:
    """Read a WSB posts CSV in the Kaggle schema (title, score, id, url, comms_num, created, body, timestamp).

    Returns columns: title, body, score, comms_num, created_at (naive UTC datetime).
    """
    df = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in df.columns}
    if "title" not in lower:
        raise ValueError(f"{path}: no 'title' column. Found: {list(df.columns)}")
    out = pd.DataFrame({"title": df[lower["title"]].astype("string")})
    out["body"] = df[lower["body"]].astype("string").fillna("") if "body" in lower else ""
    out["score"] = pd.to_numeric(df[lower["score"]], errors="coerce").fillna(0) if "score" in lower else 0
    out["comms_num"] = pd.to_numeric(df[lower["comms_num"]], errors="coerce").fillna(0) if "comms_num" in lower else 0

    created = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if "created" in lower:       # unix seconds (UTC): most reliable
        created = pd.to_datetime(pd.to_numeric(df[lower["created"]], errors="coerce"), unit="s")
    if "timestamp" in lower:
        created = created.fillna(pd.to_datetime(df[lower["timestamp"]], errors="coerce"))
    out["created_at"] = created

    out = out.dropna(subset=["title", "created_at"]).reset_index(drop=True)
    out["title"] = out["title"].astype(str)
    out["body"] = out["body"].astype(str)
    return out


def _read_cache(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["Date"])
    return d[["Date", "Close_Price", "Volume"]].sort_values("Date").reset_index(drop=True)


def fetch_prices(ticker: str, start, end, cache_dir, offline=False, refresh=False) -> pd.DataFrame:
    """Daily adjusted close + volume from Yahoo Finance via yfinance, cached to <cache_dir>/<ticker>.csv.

    
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{ticker}.csv"
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    if cache.exists() and not refresh:
        d = _read_cache(cache)
        covers = d.Date.min() <= start + pd.Timedelta(days=7) and d.Date.max() >= end - pd.Timedelta(days=7)
        if covers or offline:
            return d[(d.Date >= start) & (d.Date <= end)].reset_index(drop=True)
    if offline:
        raise FileNotFoundError(f"offline mode and no cached prices at {cache}")

    try:
        import yfinance as yf
        raw = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                          end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                          auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            raise RuntimeError("Yahoo returned no data")
        if isinstance(raw.columns, pd.MultiIndex):       # newer yfinance returns (field, ticker) columns
            raw.columns = raw.columns.get_level_values(0)
        d = raw.reset_index().rename(columns={"Close": "Close_Price"})[["Date", "Close_Price", "Volume"]]
        d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None)
        d.to_csv(cache, index=False)
        return d.reset_index(drop=True)
    except Exception as exc:  # fall back to whatever is cached
        if cache.exists():
            warnings.warn(f"{ticker}: download failed ({exc}); using cached file {cache}")
            return _read_cache(cache)
        raise RuntimeError(
            f"Could not download prices for {ticker}: {exc}\n"
            f"Fix: save a CSV with columns Date,Close_Price,Volume at {cache}") from exc


def load_all_prices(tickers, start, end, cache_dir, offline=False) -> dict:
    prices = {}
    for t in tickers:
        try:
            prices[t] = fetch_prices(t, start, end, cache_dir, offline=offline)
        except Exception as exc:
            warnings.warn(f"Skipping {t}: {exc}")
    if not prices:
        raise RuntimeError("No price data could be loaded for any ticker.")
    return prices
