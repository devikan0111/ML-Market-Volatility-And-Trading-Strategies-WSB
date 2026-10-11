"""Central settings shared by Member 1 (pipeline/model) and Member 2 (strategy/app)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PRICES_DIR = RAW / "prices"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

DEFAULT_REDDIT_CSV = RAW / "reddit_wsb.csv"     # Kaggle: gpreda/reddit-wallstreetbets-posts
FINAL_CSV = DATA / "final_output.csv"           # the hand-over file Member 2's backtester reads

HORIZON = 5            # we predict realised volatility over the NEXT 5 trading days
RANDOM_STATE = 42

# symbol -> {"names": company-name aliases (case-insensitive), "strict": symbol must be UPPERCASE or $cashtag}
# "strict" avoids false hits for short symbols that look like ordinary words.
TICKERS = {
    "GME":  {"names": ["GameStop"], "strict": False},
    "AMC":  {"names": [], "strict": False},
    "TSLA": {"names": ["Tesla"], "strict": False},
    "BB":   {"names": ["BlackBerry"], "strict": True},
    "NOK":  {"names": ["Nokia"], "strict": True},
    "PLTR": {"names": ["Palantir"], "strict": False},
}
