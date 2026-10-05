# Market Volatility Prediction & Short-Term Trading Strategies using WallStreetBets Data

An end-to-end Machine Learning pipeline that extracts sentiment signals from Reddit's **r/WallStreetBets**, forecasts short-term market volatility for targeted equities, and backtests automated trading strategies against benchmark market returns.

---

## 📌 Project Overview

Retail trading discussions on social platforms like Reddit's `r/WallStreetBets` (WSB) frequently trigger significant market volatility and non-traditional price action. This project aims to quantify retail sentiment from WSB and assess its predictive power on stock volatility and short-term price movements.

### Key Objectives
1. **Data Acquisition & Preprocessing:** Scrape social sentiment from Reddit (titles, comments, post metrics) and synchronize with historical equity price data from Yahoo Finance.
2. **Sentiment Analysis:** Compute daily composite sentiment scores using NLP techniques (VADER / FinBERT).
3. **Volatility Modeling:** Train Machine Learning regression models (Random Forest, XGBoost) to forecast rolling $N$-day market volatility.
4. **Strategy Backtesting:** Implement and simulate short-term trading rules (Long/Short momentum & volatility thresholding) to compute performance metrics like Sharpe Ratio and Maximum Drawdown.
