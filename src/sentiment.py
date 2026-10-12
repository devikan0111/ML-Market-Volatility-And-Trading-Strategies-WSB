"""module 2a: text cleaning, ticker-mention detection, NLP sentiment scoring."""
import re

import numpy as np
import pandas as pd

# Slang VADER does not know (values on VADER's -4..+4 scale).
WSB_LEXICON = {
    "moon": 2.5, "mooning": 2.8, "rocket": 2.0, "rockets": 2.0, "tendies": 2.0, "squeeze": 1.5,
    "hodl": 1.5, "hold": 0.8, "holding": 0.8, "yolo": 1.0, "bullish": 2.5, "calls": 1.0, "buy": 1.0,
    "buying": 1.0, "diamond": 1.5, "gain": 1.8, "gains": 1.8, "bearish": -2.5, "puts": -1.0,
    "bagholder": -2.0, "bagholders": -2.0, "crash": -2.5, "crashing": -2.5, "dump": -2.0,
    "dumping": -2.0, "drill": -1.5, "rekt": -2.5, "sell": -1.0, "selling": -1.0, "loss": -2.0,
    "losses": -2.0, "red": -1.0, "scam": -2.5, "fraud": -2.5,
}

_URL = re.compile(r"https?://\S+|www\.\S+")
_HANDLE = re.compile(r"[@/]?u/\w+|@\w+")


def light_clean(text: str) -> str:
    """For VADER: remove URLs/handles but KEEP case, punctuation and emojis (VADER uses them)."""
    t = _URL.sub(" ", str(text))
    t = _HANDLE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def clean_text(text: str) -> str:
    """Conventional NLP cleaning (as in the CS229 paper): lowercase, no URLs/handles/special
    characters/single-character tokens, single spaces. Used for length features and FinBERT."""
    t = light_clean(text).lower()
    t = re.sub(r"[^a-z0-9$\s]", " ", t)
    t = " ".join(w for w in t.split() if len(w) > 1 or w.isdigit())
    return t


def ticker_pattern(symbol: str, names, strict: bool) -> re.Pattern:
    """Regex that finds `symbol` (also as $SYMBOL) or a company name in text."""
    parts = [re.escape(n) for n in names]
    sym = re.escape(symbol)
    if strict:     # symbol must be written in UPPERCASE (or as a $cashtag); names are case-insensitive
        pat = rf"(?<![A-Za-z])\$?{sym}(?![A-Za-z])"
        if parts:
            pat = rf"(?i:(?<![A-Za-z])(?:{'|'.join(parts)})(?![A-Za-z]))|{pat}"
    else:
        pat = rf"(?i:(?<![A-Za-z])\$?(?:{'|'.join([sym] + parts)})(?![A-Za-z]))"
    return re.compile(pat)


def find_mentions(texts: pd.Series, tickers_cfg: dict) -> pd.DataFrame:
    """Boolean DataFrame (rows = posts, columns = tickers): does the post mention the ticker?"""
    out = {}
    for sym, cfg in tickers_cfg.items():
        rx = ticker_pattern(sym, cfg.get("names", []), cfg.get("strict", False))
        out[sym] = texts.map(lambda s, rx=rx: bool(rx.search(s)))
    return pd.DataFrame(out, index=texts.index)


def score_vader(texts) -> np.ndarray:
    """VADER compound score in [-1, +1] (negative = bearish, positive = bullish)."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(WSB_LEXICON)
    return np.array([sia.polarity_scores(t[:2000])["compound"] for t in texts], dtype=float)


def score_finbert(texts, batch_size=32) -> np.ndarray:
    """Optional: FinBERT score = P(positive) - P(negative), in [-1, +1].
    Needs `pip install transformers torch` and a first-run model download (~440 MB)."""
    from transformers import pipeline
    clf = pipeline("text-classification", model="ProsusAI/finbert", top_k=None,
                   truncation=True, max_length=256)
    scores = []
    texts = list(texts)
    for i in range(0, len(texts), batch_size):
        for res in clf(texts[i:i + batch_size]):
            d = {r["label"].lower(): r["score"] for r in res}
            scores.append(d.get("positive", 0.0) - d.get("negative", 0.0))
    return np.array(scores, dtype=float)


def score_posts(df: pd.DataFrame, method: str = "vader") -> pd.DataFrame:
    """Adds columns: text, sentiment, len_of_post. df needs title and body."""
    out = df.copy()
    out["text"] = (out["title"].fillna("") + ". " + out["body"].fillna("")).map(light_clean)
    if method == "vader":
        out["sentiment"] = score_vader(out["text"])
    elif method == "finbert":
        out["sentiment"] = score_finbert(out["text"].map(clean_text))
    else:
        raise ValueError("method must be 'vader' or 'finbert'")
    out["len_of_post"] = out["text"].map(lambda s: len(clean_text(s).split()))
    return out
