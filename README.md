# WallStreetBets Sentiment & Volatility Trading Lab

Predict short-term volatility from r/WallStreetBets sentiment, trade on it, backtest it, and demo it in a web app.

## Team / contributions

| Part | Owner | Files |
|---|---|---|
| Data, sentiment, features, volatility model | Divyansh Badaya| `run_pipeline.py`, `src/data_collection.py`, `src/sentiment.py`, `src/features.py`, `src/volatility_model.py`, `src/report_figures.py` |
| Strategy, backtest, charts, app | Devika N | `src/strategy.py`, `src/backtester.py`, `src/charts.py`, `evaluate_strategy.py`, `app.py` |


Hand-over file : `data/final_output.csv` with
`Date, Ticker, Close_Price, Sentiment_Score, Predicted_Volatility` (+ `Actual_Volatility`, `Post_Count`).
Volatility is the **daily standard deviation of log returns** (e.g. 0.03 = 3% a day).

## 1. Setup (once)

    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python -m pytest -q                # unit tests

## 2. Get the data

1. Download the Kaggle dataset **gpreda/reddit-wallstreetbets-posts** (free Kaggle account):
   https://www.kaggle.com/datasets/gpreda/reddit-wallstreetbets-posts
2. Save the CSV as `data/raw/reddit_wsb.csv`
   (columns: `title, score, id, url, comms_num, created, body, timestamp`).

The dataset is not stored in this repository. Stock prices are downloaded automatically from Yahoo Finance (`yfinance`).

## 3. Run

    # a) Build the hand-over file: match posts to tickers, score sentiment, add price features, train + validate the model
    python run_pipeline.py --reddit data/raw/reddit_wsb.csv
    #    options: --tickers GME,AMC,TSLA     --sentiment finbert (needs: pip install transformers torch)

    # b) Evaluate the trading strategy (tuned on the first 60% of dates, reported on the last 40%)
    python evaluate_strategy.py

    # c) Interactive demo
    streamlit run app.py

Outputs: `data/final_output.csv`, `data/processed/`, `results/` (figures + JSON metrics), `models/`.

## Troubleshooting
- **Yahoo download fails / rate-limited:** save a CSV `Date,Close_Price,Volume` per ticker in `data/raw/prices/<TICKER>.csv`
  and run `python run_pipeline.py --offline`.
- **Few tickers survive:** tickers with <30 matching posts are dropped (printed in the log). Choose tickers with `--tickers`.
- **Sliders look wrong in the app:** they adapt to your data; if sentiment is tiny, lower the threshold slider.
- **Strategy loses to Buy & Hold:** that is a legitimate finding; 

## Method in one paragraph
Posts are matched to tickers, scored with VADER (+ WSB slang), aggregated per trading day, and combined with price features.
A pooled Random Forest / XGBoost predicts the realised volatility of the next 5 trading days using expanding-window
walk-forward validation (5-day gap), so every prediction is out-of-sample. The strategy goes long/short on sentiment
only when predicted volatility is below a cap, trades the next day, pays 5 bps per position change, and is compared with Buy & Hold.
