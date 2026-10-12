"""MEMBER 1 entry point: raw Reddit + prices  ->  data/final_output.csv (hand-over to the backtester).

Real data:   python run_pipeline.py --reddit data/raw/reddit_wsb.csv
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import config as C
from src import report_figures as RF
from src.data_collection import load_all_prices, load_reddit_csv
from src.features import build_feature_table
from src.sentiment import find_mentions, score_posts
from src.volatility_model import run_experiments

FINAL_COLS = ["Date", "Ticker", "Close_Price", "Sentiment_Score", "Predicted_Volatility",
              "Actual_Volatility", "Post_Count"]


def run(reddit_csv, prices_dir, tickers_cfg, method="vader", offline=False, out_csv=C.FINAL_CSV,
        results_dir=C.RESULTS, n_splits=5, min_train_frac=0.4, verbose=True, synthetic=False):
    log = print if verbose else (lambda *a, **k: None)
    results_dir = Path(results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    C.PROCESSED.mkdir(parents=True, exist_ok=True)

    # 1) load Reddit posts
    posts = load_reddit_csv(reddit_csv)
    log(f"[1/5] loaded {len(posts):,} Reddit posts  {posts.created_at.min().date()} -> {posts.created_at.max().date()}")

    # 2) which posts mention which ticker, then score sentiment (only for matched posts)
    text = posts["title"] + " " + posts["body"]
    mentions = find_mentions(text, tickers_cfg)
    matched = posts[mentions.any(axis=1)].copy()
    scored = score_posts(matched, method)
    log(f"[2/5] {len(scored):,} posts mention a tracked ticker; sentiment scored with {method.upper()}")
    mention_counts = {t: int(mentions[t].sum()) for t in mentions}
    log("      mentions per ticker:", mention_counts)
    posts_by_ticker = {t: scored.loc[mentions.loc[scored.index, t]] for t in tickers_cfg}
    scored.join(mentions.loc[scored.index]).to_csv(C.PROCESSED / "posts_scored.csv", index=False)

    # 3) prices from Yahoo Finance (cached)
    start = posts.created_at.min().normalize() - pd.Timedelta(days=45)    # warm-up for rolling windows
    end = posts.created_at.max().normalize() + pd.Timedelta(days=14)      # room for the forward-looking target
    usable = [t for t in tickers_cfg if mention_counts[t] >= 30]
    dropped = sorted(set(tickers_cfg) - set(usable))
    if dropped:
        log(f"      dropping tickers with <30 mentions: {dropped}")
    prices = load_all_prices(usable, start, end, prices_dir, offline=offline)
    log(f"[3/5] prices loaded for {list(prices)}")

    # 4) features + model
    feat = build_feature_table(prices, posts_by_ticker, C.HORIZON)
    feat = feat[feat["Date"] >= posts.created_at.min().normalize()].reset_index(drop=True)
    feat.to_csv(C.PROCESSED / "features.csv", index=False)
    log(f"[4/5] feature table: {len(feat):,} rows ({feat.Ticker.nunique()} tickers, "
        f"{feat.Date.min().date()} -> {feat.Date.max().date()}); running walk-forward model comparison ...")
    exp = run_experiments(feat, n_splits=n_splits, min_train_frac=min_train_frac)

    # 5) hand-over CSV: only rows with a genuine out-of-sample prediction
    final = feat.copy()
    final["Predicted_Volatility"] = exp["preds"][exp["best_model"]]
    final = final.dropna(subset=["Predicted_Volatility"])[FINAL_COLS].sort_values(["Ticker", "Date"])
    final = final.round({c: 6 for c in FINAL_COLS if c not in ("Date", "Ticker")})
    out_csv = Path(out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_csv, index=False)

    import joblib
    C.MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": exp["final_model"], "columns": exp["feature_columns"], "name": exp["best_model"]},
                C.MODELS / "volatility_model.joblib")

    RF.fig_pred_vs_actual(final, results_dir / "fig_pred_vs_actual.png")
    RF.fig_scatter(final, results_dir / "fig_scatter.png")
    RF.fig_importance(exp["importances"], results_dir / "fig_importance.png")
    RF.fig_model_compare(exp["metrics"], results_dir / "fig_model_compare.png")
    RF.fig_attention_vs_vol(final, results_dir / "fig_attention_vs_vol.png")

    summary = {
        "synthetic": bool(synthetic), "sentiment_method": method, "n_posts": int(len(posts)), "n_posts_matched": int(len(scored)),
        "mentions_per_ticker": mention_counts, "tickers_used": list(prices),
        "posts_from": str(posts.created_at.min().date()), "posts_to": str(posts.created_at.max().date()),
        "feature_rows": int(len(feat)), "final_rows": int(len(final)),
        "final_from": str(final.Date.min().date()), "final_to": str(final.Date.max().date()),
        "best_model": exp["best_model"], "model_metrics": exp["metrics"],
        "sentiment_importance_share": exp["sentiment_importance_share"],
        "top_features": exp["importances"].head(10).round(4).to_dict(),
        "horizon_days": C.HORIZON, "n_splits": n_splits, "min_train_frac": min_train_frac,
    }
    (results_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2))
    m = exp["metrics"]
    log(f"[5/5] wrote {out_csv}  ({len(final):,} out-of-sample rows)")
    log(f"      OOS RMSE: baseline(5d)={m['baseline_rv5']['RMSE']:.4f}  RF={m['random_forest']['RMSE']:.4f}  "
        f"XGB={m['xgboost']['RMSE']:.4f}   best={exp['best_model']}")
    return final, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reddit", default=str(C.DEFAULT_REDDIT_CSV), help="Reddit posts CSV (Kaggle schema)")
    ap.add_argument("--prices-dir", default=str(C.PRICES_DIR), help="price cache folder (<TICKER>.csv)")
    ap.add_argument("--tickers", default=",".join(C.TICKERS), help="comma-separated, e.g. GME,AMC,TSLA")
    ap.add_argument("--sentiment", choices=["vader", "finbert"], default="vader")
    ap.add_argument("--offline", action="store_true", help="never call Yahoo; use cached price CSVs only")
    ap.add_argument("--out", default=None)
    ap.add_argument("--results-dir", default=None)
    a = ap.parse_args(argv)

    cfg = {t: C.TICKERS.get(t, {"names": [], "strict": len(t) <= 2}) for t in a.tickers.split(",")}
    reddit, prices_dir = a.reddit, a.prices_dir
    out, res, offline = a.out or C.FINAL_CSV, a.results_dir or C.RESULTS, a.offline
    if not Path(reddit).exists():
        sys.exit(f"Reddit CSV not found: {reddit}\nDownload it from Kaggle (gpreda/reddit-wallstreetbets-posts) "
                 f"and save as {C.DEFAULT_REDDIT_CSV}")
    run(reddit, prices_dir, cfg, a.sentiment, offline, out, res)


if __name__ == "__main__":
    main()
