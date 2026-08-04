#!/usr/bin/env python3
"""
src/price/features.py
======================
Stage 3 — Feature Engineering and Daily Alignment (answers RQ2 setup)

Reads:
  data/processed/price_series.csv   — rolling daily price snapshots
                                      (from daily_run.py scraper)
  data/processed/classification.csv — LAFT predictions on dated posts
                                      (from train_sentiment.py Stage 1)
  data/raw/fx_rates.csv             — optional: parallel-market FX rate
  data/raw/fuel_prices.csv          — optional: pump-price proxy

Writes:
  data/processed/series_daily.csv   — date, price, sentiment
                                      (→ ch6_analysis.py Granger test RQ2)
  data/processed/lstm_dataset.pkl   — windowed X_seq, y, X_flat, splits
                                      (→ train_models.py Stage 4)

Five product categories (matching scraper schema):
  electronics | generators | fmcg | food | clothing

Scaling rule (assumption A3 — no leakage):
  StandardScaler fitted on training window ONLY,
  then applied identically to validation and test windows.
  The same scaler object is saved inside lstm_dataset.pkl so every
  forecasting arm in train_models.py uses the IDENTICAL transform.

Usage (Colab):
  !python src/price/features.py --category electronics --platform jumia
  !python src/price/features.py --category food        --platform all
  !python src/price/features.py --category all         --platform all
"""

import argparse
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("features")

# ── paths ─────────────────────────────────────────────────────────────────────
PROC_DIR = Path("data/processed")
RAW_DIR  = Path("data/raw")
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── hyperparameters (match Chapter 5 spec) ────────────────────────────────────
LOOK_BACK = 14   # T  : days of price history per LSTM sample
LAG_K     = 7    # k  : number of price lag features
ROLL_W    = (3, 7, 14)   # rolling window sizes (days)

CATEGORIES = ["electronics", "generators", "fmcg", "food", "clothing"]
PLATFORMS  = ["jumia", "konga", "temu"]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def safe_log1p(s: pd.Series) -> pd.Series:
    """log1p transform for strictly positive price-level series."""
    if (s > 0).all():
        return np.log1p(s)
    return s


def rolling_features(price: pd.Series) -> pd.DataFrame:
    """
    Build the full price-branch feature matrix for one daily price series.
    All features use information available up to and including t-1
    so there is NO look-ahead leakage into the target at time t.

    Features produced (F ≈ 28 columns):
      lag_price_1 .. lag_price_K     : lagged price levels
      lag_return_1 .. lag_return_K   : lagged simple returns
      roll_mean_3/7/14               : rolling means (shifted by 1)
      roll_std_3/7/14                : rolling std deviations (shifted by 1)
      ewma                           : exponentially weighted moving average
      zscore_7                       : 7-day z-score of return
      momentum_7                     : 7-day price momentum
    """
    r     = price.pct_change()           # simple return
    log_r = np.log(price / price.shift(1))  # log return

    feats = {}

    # lagged levels and returns (information at t-1, t-2, …, t-K)
    for k in range(1, LAG_K + 1):
        feats[f"lag_price_{k}"]  = price.shift(k)
        feats[f"lag_return_{k}"] = r.shift(k)

    # rolling statistics — shifted by 1 so window ends at t-1
    for w in ROLL_W:
        feats[f"roll_mean_{w}"] = price.shift(1).rolling(w).mean()
        feats[f"roll_std_{w}"]  = price.shift(1).rolling(w).std()

    # additional momentum / trend features
    feats["ewma"]       = price.shift(1).ewm(span=7, adjust=False).mean()
    feats["zscore_7"]   = ((price.shift(1)
                            - price.shift(1).rolling(7).mean())
                           / (price.shift(1).rolling(7).std() + 1e-8))
    feats["momentum_7"] = price.shift(1) / (price.shift(8) + 1e-8) - 1

    return pd.DataFrame(feats, index=price.index)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Cyclical calendar encodings so the model understands weekly and
    monthly seasonality without an ordinal discontinuity at the boundary.
    """
    dow   = index.dayofweek   # 0=Monday … 6=Sunday
    month = index.month       # 1=January … 12=December
    return pd.DataFrame({
        "dow_sin":  np.sin(2 * np.pi * dow   / 7),
        "dow_cos":  np.cos(2 * np.pi * dow   / 7),
        "mon_sin":  np.sin(2 * np.pi * month / 12),
        "mon_cos":  np.cos(2 * np.pi * month / 12),
    }, index=index)


# ══════════════════════════════════════════════════════════════════════════════
# DAILY SENTIMENT INDEX
# ══════════════════════════════════════════════════════════════════════════════
def build_daily_sentiment(category: str) -> pd.Series:
    """
    Aggregate LAFT predictions on dated posts into a daily net-sentiment
    index:  S_t = (n_positive − n_negative) / n_total  ∈ [−1, 1]

    Requires classification.csv (Stage 1) AND posts_dated.csv
    (from daily_run.py text scrapers).

    If posts_dated.csv is not yet available, returns a zero series
    and logs a warning — the Granger test (RQ2) will show no signal,
    which is the honest result until real text data is collected.
    """
    cls_path   = PROC_DIR / "classification.csv"
    posts_path = PROC_DIR / "posts_dated.csv"

    if not cls_path.exists():
        raise FileNotFoundError(
            "classification.csv not found. "
            "Run Stage 1 (train_sentiment.py) first.")

    cls = pd.read_csv(cls_path)

    if not posts_path.exists():
        log.warning("posts_dated.csv not found — sentiment will be zero. "
                    "Run daily_run.py text scrapers to collect posts.")
        return pd.Series(dtype=float, name="sentiment")

    posts = pd.read_csv(posts_path, parse_dates=["date"])

    # filter to this category and align with classification predictions
    cat_posts = posts[posts["category"] == category].copy()
    n         = min(len(cat_posts), len(cls))
    cat_posts = cat_posts.iloc[:n].copy()
    cat_posts["pred"] = cls["laft_afriberta"].values[:n]

    # polarity mapping: positive=+1, negative=−1, neutral=0
    cat_posts["polarity"] = cat_posts["pred"].map({0: 1, 1: -1, 2: 0})

    # aggregate to daily net-sentiment index
    daily = (cat_posts.groupby("date")["polarity"]
             .apply(lambda g: g.sum() / max(1, len(g)))
             .rename("sentiment"))
    log.info("Daily sentiment index: %d days, mean=%.3f, std=%.3f",
             len(daily), daily.mean(), daily.std())
    return daily


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FEATURE BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_features(category: str,
                   platform: str = "jumia") -> dict:
    """
    Build the full feature dataset for one (category, platform) pair.

    Returns a dict with:
      X_seq   : (N, T, F)  windowed LSTM input sequences
      y       : (N,)       target values (price at time t)
      X_flat  : (N, T*F)   flattened features for XGBoost / ARIMA
      splits  : {"tr": int, "va": int}  index boundaries
      scaler  : fitted StandardScaler (train window only)
      feature_names : list of F feature names
      dates   : DatetimeIndex aligned with y
      sent_daily : (N,)  daily sentiment index aligned with y
    """
    log.info("══ Building features: category=%s  platform=%s ══",
             category, platform)

    # ── load and aggregate price series ───────────────────────────────────────
    price_path = PROC_DIR / "price_series.csv"
    if not price_path.exists():
        raise FileNotFoundError(
            "price_series.csv not found. "
            "Run the daily scrapers for at least 90 days first.")

    df_raw = pd.read_csv(price_path, parse_dates=["date"])

    # filter by category; optionally filter by platform
    mask = df_raw["category"] == category
    if platform != "all":
        mask &= df_raw["platform"] == platform
    df_cat = df_raw[mask]

    if df_cat.empty:
        raise ValueError(f"No data for category='{category}' "
                         f"platform='{platform}'. "
                         "Check your scraped CSV files.")

    # daily median price across all matching products
    price = (df_cat.groupby("date")["current_price_ngn"]
             .median()
             .sort_index()
             .asfreq("D")          # force daily frequency
             .interpolate("linear")  # fill at most isolated missing days
             )
    log.info("Price series: %d days  (%s → %s)  median=₦%.0f",
             len(price),
             price.index[0].date(),
             price.index[-1].date(),
             price.median())

    if len(price) < 90:
        log.warning("Only %d days of price data — thesis requires ≥90 "
                    "consecutive days for reliable estimates.", len(price))

    # ── optional exogenous features ───────────────────────────────────────────
    exog_cols = {}
    fx_path   = RAW_DIR / "fx_rates.csv"
    fuel_path = RAW_DIR / "fuel_prices.csv"
    if fx_path.exists():
        fx = (pd.read_csv(fx_path, parse_dates=["date"], index_col="date")
              ["fx_rate"].shift(1))   # lag 1: use yesterday's FX rate
        exog_cols["fx_lag1"] = fx
        log.info("FX rate loaded and lagged by 1 day.")
    if fuel_path.exists():
        fuel = (pd.read_csv(fuel_path, parse_dates=["date"], index_col="date")
                ["fuel_price"].shift(1))
        exog_cols["fuel_lag1"] = fuel
        log.info("Fuel price loaded and lagged by 1 day.")

    # ── rolling + calendar features ───────────────────────────────────────────
    roll_df = rolling_features(price)
    cal_df  = calendar_features(price.index)

    feat_df = pd.concat([roll_df, cal_df], axis=1)
    for col_name, series in exog_cols.items():
        feat_df[col_name] = series.reindex(price.index)

    feat_df = feat_df.dropna()
    price   = price.loc[feat_df.index]
    log.info("Feature matrix after dropna: %d rows × %d cols",
             *feat_df.shape)

    # ── daily sentiment index ─────────────────────────────────────────────────
    try:
        sent_daily = build_daily_sentiment(category)
        sent_aligned = sent_daily.reindex(price.index).fillna(0.0)
    except FileNotFoundError as e:
        log.warning("%s — using zero sentiment.", e)
        sent_aligned = pd.Series(0.0, index=price.index, name="sentiment")

    # ── series_daily.csv (→ ch6_analysis.py Granger test) ────────────────────
    series_daily = pd.DataFrame({
        "date":      price.index.strftime("%Y-%m-%d"),
        "price":     price.values,
        "sentiment": sent_aligned.values,
    })
    series_path = PROC_DIR / "series_daily.csv"
    series_daily.to_csv(series_path, index=False)
    log.info("series_daily.csv saved: %d rows → %s", len(series_daily), series_path)

    # ── z-score scaling (fitted on training window ONLY) ──────────────────────
    n      = len(feat_df)
    tr_end = int(n * 0.70)
    va_end = int(n * 0.85)

    X_arr = feat_df.values.astype(float)
    y_arr = price.values.astype(float)

    scaler = StandardScaler()
    X_arr[:tr_end]  = scaler.fit_transform(X_arr[:tr_end])  # fit+transform train
    X_arr[tr_end:]  = scaler.transform(X_arr[tr_end:])      # transform val+test

    log.info("Scaler fitted on training window (%d rows). "
             "Applied to val+test without refitting (no leakage).", tr_end)

    # ── LSTM windowed sequences ───────────────────────────────────────────────
    X_seq, y_seq, sent_seq = [], [], []
    for t in range(LOOK_BACK, n):
        X_seq.append(X_arr[t - LOOK_BACK: t])   # shape (T, F)
        y_seq.append(y_arr[t])                   # target: price at t
        sent_seq.append(sent_aligned.values[t])  # sentiment at t (info ≤ t-1 used)

    X_seq    = np.array(X_seq)    # (N, T, F)
    y_seq    = np.array(y_seq)    # (N,)
    X_flat   = X_arr[LOOK_BACK:]  # (N, F) — for XGBoost / ARIMA
    sent_seq = np.array(sent_seq) # (N,)

    # adjust split indices for look-back offset
    splits = {
        "tr": tr_end - LOOK_BACK,
        "va": va_end - LOOK_BACK,
    }

    dataset = {
        "X_seq":         X_seq,
        "y":             y_seq,
        "X_flat":        X_flat,
        "sent_daily":    sent_seq,
        "splits":        splits,
        "scaler":        scaler,
        "feature_names": list(feat_df.columns),
        "dates":         price.index[LOOK_BACK:],
        "category":      category,
        "platform":      platform,
        "look_back":     LOOK_BACK,
    }

    pkl_path = PROC_DIR / "lstm_dataset.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(dataset, f)

    log.info("lstm_dataset.pkl saved → %s", pkl_path)
    log.info("Shapes — X_seq: %s  y: %s  X_flat: %s",
             X_seq.shape, y_seq.shape, X_flat.shape)
    log.info("Splits — train: 0:%d  val: %d:%d  test: %d:%d",
             splits["tr"], splits["tr"], splits["va"],
             splits["va"], len(y_seq))
    log.info("Features (%d): %s", len(feat_df.columns),
             ", ".join(feat_df.columns))

    return dataset


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Stage 3: Feature engineering and daily alignment")
    ap.add_argument("--category", default="electronics",
                    choices=CATEGORIES + ["all"],
                    help="Product category to process")
    ap.add_argument("--platform", default="jumia",
                    choices=PLATFORMS + ["all"],
                    help="Platform to use (all = merge all three)")
    args = ap.parse_args()

    cats = CATEGORIES if args.category == "all" else [args.category]
    for cat in cats:
        try:
            data = build_features(cat, args.platform)
            log.info("✓ %s/%s complete — %d samples",
                     cat, args.platform, len(data["y"]))
        except (FileNotFoundError, ValueError) as e:
            log.error("Skipping %s/%s: %s", cat, args.platform, e)
