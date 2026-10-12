"""MEMBER 1, module 3: predict future 5-day realised volatility (Random Forest / XGBoost).

Validation = EXPANDING-WINDOW WALK-FORWARD (never shuffles time):
    train on all dates before a block (minus a HORIZON-day gap, because the target looks
    HORIZON days ahead), predict the block, move forward. Every prediction in the final CSV is
    therefore genuinely out-of-sample, which is what the backtester needs.
The model is fit on log(volatility) (vol is positive and right-skewed) and predictions are exp()'d back.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import HORIZON, RANDOM_STATE
from src.features import PRICE_FEATURES, SENT_FEATURES

EPS = 1e-6
MODEL_NAMES = ["random_forest", "xgboost"]


def make_model(name: str):
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                     n_jobs=-1, random_state=RANDOM_STATE)
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=1.0, random_state=RANDOM_STATE,
                            n_jobs=-1, verbosity=0)
    raise ValueError(name)


def design_matrix(df: pd.DataFrame, features) -> pd.DataFrame:
    """Selected features + one-hot ticker columns (one pooled model for all tickers)."""
    X = df[list(features)].copy()
    return pd.concat([X, pd.get_dummies(df["Ticker"], prefix="tk", dtype=float)], axis=1)


def walk_forward_predict(df: pd.DataFrame, features, model_name: str, n_splits: int = 5,
                         min_train_frac: float = 0.4, gap: int = HORIZON) -> pd.Series:
    """Out-of-sample predictions (NaN for the initial training period)."""
    X_all = design_matrix(df, features)
    y_all = np.log(df["Actual_Volatility"].clip(lower=EPS))
    labeled = df["Actual_Volatility"].notna()
    dates = np.sort(df["Date"].unique())
    n = len(dates)
    start = int(n * min_train_frac)
    edges = np.linspace(start, n, n_splits + 1).astype(int)
    pred = pd.Series(np.nan, index=df.index, dtype=float)
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a or a - gap <= 0:
            continue
        train_mask = (df["Date"] < dates[a - gap]) & labeled
        test_mask = df["Date"].isin(dates[a:b])
        if train_mask.sum() < 30 or test_mask.sum() == 0:
            continue
        model = make_model(model_name)
        model.fit(X_all[train_mask], y_all[train_mask])
        pred[test_mask] = np.exp(model.predict(X_all[test_mask]))
    return pred


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Corr": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 2 else float("nan"),
        "N": int(len(y_true)),
    }


def run_experiments(feat: pd.DataFrame, n_splits: int = 5, min_train_frac: float = 0.4) -> dict:
    """Compare baselines, RF and XGBoost (with and without Reddit features) on identical out-of-sample rows."""
    all_feats = PRICE_FEATURES + SENT_FEATURES
    preds = pd.DataFrame(index=feat.index)
    for m in MODEL_NAMES:
        preds[f"{m}"] = walk_forward_predict(feat, all_feats, m, n_splits, min_train_frac)
        preds[f"{m}_price_only"] = walk_forward_predict(feat, PRICE_FEATURES, m, n_splits, min_train_frac)
    preds["baseline_rv5"] = feat["rv_5"]          # "tomorrow looks like the last 5 days"
    preds["baseline_rv20"] = feat["rv_20"]

    valid = feat["Actual_Volatility"].notna() & preds[MODEL_NAMES[0]].notna()
    metrics = {c: regression_metrics(feat.loc[valid, "Actual_Volatility"], preds.loc[valid, c]) for c in preds}
    best = min(MODEL_NAMES, key=lambda m: metrics[m]["RMSE"])

    # final model on all labelled data -> feature importances (for the report) and a saved artefact
    lab = feat["Actual_Volatility"].notna()
    X = design_matrix(feat, all_feats)
    final = make_model(best).fit(X[lab], np.log(feat.loc[lab, "Actual_Volatility"].clip(lower=EPS)))
    imp = pd.Series(final.feature_importances_, index=X.columns).sort_values(ascending=False)
    sent_share = float(imp[[c for c in imp.index if c in SENT_FEATURES]].sum() / imp.sum())

    return {"preds": preds, "metrics": metrics, "best_model": best, "importances": imp,
            "sentiment_importance_share": sent_share, "final_model": final,
            "feature_columns": list(X.columns), "oos_mask": valid}
