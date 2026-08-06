"""
core.py — data access and statistics for the Marketplace Intelligence dashboard.

This module is deliberately free of Streamlit UI code so it can be imported by
the notebook, by tests, or by any other front end.

It reads the three artefacts produced by the thesis pipeline:

    data/processed/classification.csv   date -> y_true, b1_svm_tfidf,
                                        b2_transformer_nolaft, laft_afriberta
    data/processed/series_daily.csv     date, price, sentiment
    data/processed/forecasts.csv        date, y_true, arima, xgboost,
                                        lstm_price, sentiment_only, fusion

If an artefact is missing, the dashboard falls back to a generated demonstration
bundle. Demonstration data is always flagged in the interface so it can never be
mistaken for a model result.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Conventions
# --------------------------------------------------------------------------

# NaijaSenti integer encoding used throughout the pipeline.
LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Sentiment model columns in classification.csv, in reporting order.
CLF_ARMS = {
    "b1_svm_tfidf": "B1 · TF-IDF + LinearSVC",
    "b2_transformer_nolaft": "B2 · AfriBERTa (no LAFT)",
    "laft_afriberta": "LAFT · AfriBERTa + LAFT",
}

# Forecasting columns in forecasts.csv, in reporting order.
FC_ARMS = {
    "arima": "B3 · ARIMA",
    "xgboost": "B4 · XGBoost",
    "lstm_price": "B5 · LSTM (price only)",
    "sentiment_only": "B6 · Sentiment only",
    "fusion": "EXP · Late fusion",
}

CONTROL_ARM = "lstm_price"      # the pre-registered control for RQ3
EXPERIMENTAL_ARM = "fusion"

FILES = {
    "classification": "classification.csv",
    "series": "series_daily.csv",
    "forecasts": "forecasts.csv",
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


# Column aliases seen in earlier pipeline revisions and hand-edited exports.
ALIASES = {
    "y_true": {"ytrue", "true", "actual", "y", "gold", "label", "y_test"},
    "date": {"day", "ds", "timestamp", "date_"},
    "price": {"close", "price_ngn", "value"},
    "sentiment": {"sent", "sentiment_index", "s"},
}

REQUIRED = {
    "classification": {"y_true"},
    "series": {"date", "price", "sentiment"},
    "forecasts": {"date", "y_true"},
}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case, strip, and map known aliases onto the canonical names."""
    if df is None:
        return df
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").replace("\u200b", "").strip()
                  for c in df.columns]
    lookup = {c.lower().replace(" ", "_"): c for c in df.columns}
    for canonical, alts in ALIASES.items():
        if canonical in df.columns:
            continue
        for alt in alts:
            if alt in lookup:
                df = df.rename(columns={lookup[alt]: canonical})
                break
    return df


def check_schema(key: str, df: pd.DataFrame) -> str:
    """Return an empty string when usable, otherwise a plain-language problem."""
    if df is None or df.empty:
        return "the file is empty"
    missing = REQUIRED[key] - set(df.columns)
    if missing:
        return (f"missing column{'s' if len(missing) > 1 else ''} "
                f"{', '.join(sorted(missing))}; found {', '.join(map(str, df.columns))}")
    if key == "classification" and not (set(CLF_ARMS) & set(df.columns)):
        return ("no model prediction columns; expected one or more of "
                f"{', '.join(CLF_ARMS)}; found {', '.join(map(str, df.columns))}")
    if key == "forecasts" and not (set(FC_ARMS) & set(df.columns)):
        return ("no forecast columns; expected one or more of "
                f"{', '.join(FC_ARMS)}; found {', '.join(map(str, df.columns))}")
    return ""


@dataclass
class Bundle:
    """Everything the dashboard needs, plus provenance for each table."""
    classification: pd.DataFrame | None = None
    series: pd.DataFrame | None = None
    forecasts: pd.DataFrame | None = None
    origin: dict[str, str] = field(default_factory=dict)   # table -> "file path" | "demonstration"
    problems: dict[str, str] = field(default_factory=dict)  # table -> why the file was rejected
    root: str = ""

    @property
    def is_demo(self) -> bool:
        return any(v == "demonstration" for v in self.origin.values())

    def status(self) -> pd.DataFrame:
        rows = []
        for key, fname in FILES.items():
            table = getattr(self, key)
            rows.append({
                "Table": fname,
                "Rows": 0 if table is None else len(table),
                "Source": self.origin.get(key, "missing"),
                "Problem": self.problems.get(key, ""),
            })
        return pd.DataFrame(rows)


def read_csv_forgiving(source) -> pd.DataFrame | None:
    """Read a CSV that may carry an Excel byte-order mark or a non-comma delimiter."""
    attempts = (
        {"encoding": "utf-8-sig"},
        {"encoding": "utf-8-sig", "sep": None, "engine": "python"},
        {"encoding": "latin-1", "sep": None, "engine": "python"},
    )
    for kwargs in attempts:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, **kwargs)
            if len(df.columns) > 1 or "sep" in kwargs:
                return df
        except Exception:
            continue
    return None


def _read(path: Path) -> pd.DataFrame | None:
    return read_csv_forgiving(path)


def load_bundle(root: str | Path, allow_demo: bool = True, seed: int = 42) -> Bundle:
    """Load artefacts from `root`, filling any gaps with demonstration data."""
    root = Path(root).expanduser()
    bundle = Bundle(root=str(root))
    demo = demo_bundle(seed) if allow_demo else Bundle()

    for key, fname in FILES.items():
        found = None
        for candidate in (root / fname, root / "data" / "processed" / fname, root / "processed" / fname):
            if candidate.exists():
                found = normalise_columns(_read(candidate))
                if found is not None:
                    problem = check_schema(key, found)
                    if problem:
                        bundle.problems[key] = f"{candidate.name}: {problem}"
                        found = None
                    else:
                        bundle.origin[key] = str(candidate)
                    break
        if found is None and allow_demo:
            found = getattr(demo, key)
            bundle.origin[key] = "demonstration"
        setattr(bundle, key, found)

    if bundle.series is not None:
        bundle.series = _tidy_series(bundle.series)
    if bundle.forecasts is not None and "date" in bundle.forecasts.columns:
        bundle.forecasts["date"] = pd.to_datetime(bundle.forecasts["date"], errors="coerce")
    return bundle


def bundle_from_uploads(uploads: dict[str, io.BytesIO], seed: int = 42) -> Bundle:
    """Build a bundle from user-uploaded CSVs (keys: classification/series/forecasts)."""
    bundle = Bundle(root="uploaded")
    demo = demo_bundle(seed)
    for key in FILES:
        buf = uploads.get(key)
        if buf is not None:
            try:
                table = normalise_columns(read_csv_forgiving(buf))
                problem = check_schema(key, table)
                if problem:
                    name = getattr(buf, "name", "uploaded file")
                    bundle.problems[key] = f"{name}: {problem}"
                else:
                    setattr(bundle, key, table)
                    bundle.origin[key] = "uploaded file"
                    continue
            except Exception as exc:
                bundle.problems[key] = f"could not be read ({exc})"
        setattr(bundle, key, getattr(demo, key))
        bundle.origin[key] = "demonstration"

    bundle.series = _tidy_series(bundle.series)
    if "date" in bundle.forecasts.columns:
        bundle.forecasts["date"] = pd.to_datetime(bundle.forecasts["date"], errors="coerce")
    return bundle


def _tidy_series(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    if "date" in s.columns:
        s["date"] = pd.to_datetime(s["date"], errors="coerce")
        s = s.dropna(subset=["date"]).sort_values("date")
    for col in ("price", "sentiment"):
        if col in s.columns:
            s[col] = pd.to_numeric(s[col], errors="coerce")
    return s.dropna(subset=[c for c in ("price", "sentiment") if c in s.columns]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Demonstration data
# --------------------------------------------------------------------------

def demo_bundle(seed: int = 42, n_days: int = 420, n_test: int = 2000) -> Bundle:
    """Synthetic stand-in with the same schema and a genuine sentiment->price lead.

    Used only so the interface is explorable before the pipeline has run. Every
    panel that displays it is labelled.
    """
    rng = np.random.default_rng(seed)

    # --- series_daily.csv: sentiment leads price by one day -----------------
    sent = np.zeros(n_days)
    for t in range(1, n_days):
        sent[t] = 0.6 * sent[t - 1] + rng.normal(0, 0.8)
    sent = (sent - sent.mean()) / sent.std()

    price = np.zeros(n_days)
    price[0] = 28_000
    for t in range(1, n_days):
        price[t] = (0.55 * price[t - 1] + 0.45 * 28_000
                    - 900 * sent[t - 1] + rng.normal(0, 420))
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")
    series = pd.DataFrame({"date": dates, "price": price, "sentiment": sent})

    # --- classification.csv: three arms of increasing quality ---------------
    y_true = rng.choice([0, 1, 2], size=n_test, p=[0.34, 0.33, 0.33])

    def noisy(y, acc):
        keep = rng.random(len(y)) < acc
        wrong = np.array([rng.choice([c for c in (0, 1, 2) if c != v]) for v in y])
        return np.where(keep, y, wrong)

    classification = pd.DataFrame({
        "y_true": y_true,
        "b1_svm_tfidf": noisy(y_true, 0.68),
        "b2_transformer_nolaft": noisy(y_true, 0.75),
        "laft_afriberta": noisy(y_true, 0.81),
    })

    # --- forecasts.csv: held-out window, fusion beats the control -----------
    test_days = 60
    fc_dates = dates[-test_days:]
    y = price[-test_days:]

    def arm(noise, bias=0.0):
        return y + rng.normal(bias, noise, test_days)

    forecasts = pd.DataFrame({
        "date": fc_dates,
        "y_true": y,
        "arima": arm(1450, 120),
        "xgboost": arm(1180),
        "lstm_price": arm(980),
        "sentiment_only": arm(1900),
        "fusion": arm(730),
    })

    return Bundle(
        classification=classification,
        series=series,
        forecasts=forecasts,
        origin={k: "demonstration" for k in FILES},
        root="demonstration",
    )


# --------------------------------------------------------------------------
# Classification statistics (RQ1)
# --------------------------------------------------------------------------

def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> np.ndarray:
    m = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < k and 0 <= p < k:
            m[int(t), int(p)] += 1
    return m


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    cm = confusion(y_true, y_pred)
    rows = []
    for i, name in LABELS.items():
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"Class": name, "Support": int(cm[i, :].sum()),
                     "Precision": prec, "Recall": rec, "F1": f1})
    return pd.DataFrame(rows)


def headline_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    pc = per_class_metrics(y_true, y_pred)
    support = pc["Support"].to_numpy()
    return {
        "accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
        "macro_f1": float(pc["F1"].mean()),
        "weighted_f1": float(np.average(pc["F1"], weights=support)) if support.sum() else 0.0,
    }


def classifier_table(clf: pd.DataFrame) -> pd.DataFrame:
    y = clf["y_true"].to_numpy()
    rows = []
    for col, label in CLF_ARMS.items():
        if col not in clf.columns:
            continue
        m = headline_metrics(y, clf[col].to_numpy())
        rows.append({"Model": label, "Accuracy": m["accuracy"],
                     "Macro F1": m["macro_f1"], "Weighted F1": m["weighted_f1"]})
    return pd.DataFrame(rows)


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Exact-ish McNemar on the discordant pairs between two classifiers."""
    a = np.asarray(pred_a) == np.asarray(y_true)
    b = np.asarray(pred_b) == np.asarray(y_true)
    n01 = int((~a & b).sum())     # a wrong, b right
    n10 = int((a & ~b).sum())     # a right, b wrong
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "statistic": float("nan"), "p_value": float("nan")}
    stat = (abs(n01 - n10) - 1) ** 2 / n
    try:
        from scipy.stats import chi2, binomtest
        p = binomtest(min(n01, n10), n, 0.5).pvalue if n < 25 else float(chi2.sf(stat, 1))
    except Exception:
        p = float("nan")
    return {"n01": n01, "n10": n10, "statistic": float(stat), "p_value": float(p)}


# --------------------------------------------------------------------------
# Lead-lag statistics (RQ2)
# --------------------------------------------------------------------------

def cross_correlation(series: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
    """Correlation of sentiment(t-lag) with price(t). Positive lag = sentiment leads."""
    s = series["sentiment"].to_numpy()
    p = series["price"].to_numpy()
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = s[:len(s) - lag] if lag else s, p[lag:]
        else:
            a, b = s[-lag:], p[:len(p) + lag]
        n = min(len(a), len(b))
        r = float(np.corrcoef(a[:n], b[:n])[0, 1]) if n > 3 else np.nan
        rows.append({"lag": lag, "r": r,
                     "direction": "sentiment leads" if lag > 0 else
                                  ("price leads" if lag < 0 else "same day")})
    return pd.DataFrame(rows)


def granger_table(series: pd.DataFrame, max_lag: int = 4) -> pd.DataFrame:
    """Granger causality both ways, on first-differenced series."""
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except Exception:
        return pd.DataFrame()

    d = series[["price", "sentiment"]].diff().dropna()
    if len(d) < max_lag * 5:
        return pd.DataFrame()

    import contextlib
    import warnings

    rows = []
    pairs = {
        "sentiment → price": d[["price", "sentiment"]].to_numpy(),
        "price → sentiment": d[["sentiment", "price"]].to_numpy(),
    }
    for name, arr in pairs.items():
        try:
            # statsmodels prints its own report; keep the console clean.
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = grangercausalitytests(arr, maxlag=max_lag)
        except Exception:
            continue
        for lag in range(1, max_lag + 1):
            f_stat, p_val = res[lag][0]["ssr_ftest"][:2]
            rows.append({"Direction": name, "Lag (days)": lag,
                         "F": float(f_stat), "p": float(p_val)})
    return pd.DataFrame(rows)


def adf_summary(series: pd.DataFrame) -> pd.DataFrame:
    try:
        from statsmodels.tsa.stattools import adfuller
    except Exception:
        return pd.DataFrame()
    rows = []
    for col in ("price", "sentiment"):
        for label, x in ((f"{col} (level)", series[col].to_numpy()),
                         (f"{col} (differenced)", series[col].diff().dropna().to_numpy())):
            try:
                stat, p, *_ = adfuller(x, autolag="AIC")
                rows.append({"Series": label, "ADF": float(stat), "p": float(p),
                             "Stationary at 5%": "yes" if p < 0.05 else "no"})
            except Exception:
                continue
    return pd.DataFrame(rows)


def sentiment_sensitivity(series: pd.DataFrame, lag: int = 1) -> dict:
    """OLS of price(t) on sentiment(t-lag): the naira response to a 1-SD sentiment move."""
    s = series["sentiment"].to_numpy()
    p = series["price"].to_numpy()
    x, y = s[:-lag] if lag else s, p[lag:] if lag else p
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    if n < 10 or np.std(x) == 0:
        return {"beta": 0.0, "r": 0.0, "lag": lag}
    beta = float(np.polyfit(x, y, 1)[0])
    r = float(np.corrcoef(x, y)[0, 1])
    return {"beta": beta, "r": r, "lag": lag}


# --------------------------------------------------------------------------
# Forecasting statistics (RQ3)
# --------------------------------------------------------------------------

def reg_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = y - yhat
    mape = float(np.mean(np.abs(err / np.where(y == 0, np.nan, y))) * 100)
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAPE %": mape}


def forecast_table(fc: pd.DataFrame, control: str = CONTROL_ARM) -> pd.DataFrame:
    y = fc["y_true"].to_numpy()
    ctrl_mae = np.mean(np.abs(y - fc[control].to_numpy())) if control in fc.columns else np.nan
    rows = []
    for col, label in FC_ARMS.items():
        if col not in fc.columns:
            continue
        m = reg_metrics(y, fc[col].to_numpy())
        impr = 100 * (ctrl_mae - m["MAE"]) / ctrl_mae if ctrl_mae and not np.isnan(ctrl_mae) else np.nan
        rows.append({"Model": label, "arm": col, **m,
                     "Δ vs control %": 0.0 if col == control else impr})
    return pd.DataFrame(rows)


def diebold_mariano(y, pred_a, pred_b, h: int = 1) -> dict:
    """DM test on absolute-loss differentials, Harvey-Leybourne-Newbold corrected."""
    y = np.asarray(y, float)
    e1 = np.abs(y - np.asarray(pred_a, float))
    e2 = np.abs(y - np.asarray(pred_b, float))
    d = e1 - e2
    n = len(d)
    if n < 8:
        return {"DM": float("nan"), "p_value": float("nan"), "n": n}
    dbar = d.mean()
    gamma0 = np.sum((d - dbar) ** 2) / n
    gammas = [np.sum((d[k:] - dbar) * (d[:-k] - dbar)) / n for k in range(1, h)]
    var = (gamma0 + 2 * sum(gammas)) / n
    if var <= 0:
        return {"DM": float("nan"), "p_value": float("nan"), "n": n}
    dm = dbar / np.sqrt(var)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= corr
    try:
        from scipy.stats import t
        p = float(2 * t.sf(abs(dm), n - 1))
    except Exception:
        p = float("nan")
    return {"DM": float(dm), "p_value": p, "n": n}


# --------------------------------------------------------------------------
# Live scoring (early-warning panel)
# --------------------------------------------------------------------------

POS = {"good", "fine", "sweet", "correct", "nice", "love", "better", "fast", "original",
       "sharp", "quality", "affordable", "recommend", "kyau", "lafiya", "wonderful",
       "worth", "cheap", "reliable", "durable", "genuine", "madness"}
NEG = {"bad", "scam", "fake", "slow", "expensive", "wahala", "rubbish", "poor",
       "disappointed", "waste", "delay", "broken", "costly", "overpriced", "yawa",
       "terrible", "cheat", "nonsense", "gbese", "damaged", "late", "refund"}

_HOOK = {"fn": None}


def register_classifier(fn) -> None:
    """Plug the trained AfriBERTa+LAFT model in.

    `fn(list[str]) -> list[int]` returning 0/1/2 per comment. Once registered,
    the dashboard stops using the lexicon fallback and says so in the interface.
    """
    _HOOK["fn"] = fn


def classifier_is_registered() -> bool:
    return _HOOK["fn"] is not None


def score_comments(comments: list[str]) -> tuple[pd.DataFrame, float]:
    """Label each comment and return the net sentiment index in [-1, 1]."""
    texts = [c.strip() for c in comments if c.strip()]
    if not texts:
        return pd.DataFrame(columns=["Comment", "Label", "Source"]), 0.0

    if _HOOK["fn"] is not None:
        codes = _HOOK["fn"](texts)
        rows = [{"Comment": t, "Label": LABELS.get(int(c), "Neutral"), "Source": "model"}
                for t, c in zip(texts, codes)]
    else:
        rows = []
        for t in texts:
            toks = set(t.lower().replace(",", " ").replace(".", " ").replace("!", " ").split())
            p, n = len(toks & POS), len(toks & NEG)
            label = "Positive" if p > n else "Negative" if n > p else "Neutral"
            rows.append({"Comment": t, "Label": label, "Source": "lexicon"})

    df = pd.DataFrame(rows)
    net = (df["Label"].eq("Positive").sum() - df["Label"].eq("Negative").sum()) / len(df)
    return df, float(net)


def risk_assessment(net_sentiment: float, volatility: float) -> dict:
    """Combine the sentiment index with realised daily volatility into a level."""
    score = volatility * 4 + max(0.0, -net_sentiment) * 0.6
    if score > 0.6 or net_sentiment < -0.4:
        level, tone, note = "High", "high", "Volatility and sentiment both point to price movement"
    elif score > 0.3 or net_sentiment < -0.1:
        level, tone, note = "Watch", "watch", "One signal is elevated; check again tomorrow"
    else:
        level, tone, note = "Steady", "steady", "No early-warning signal in the current window"
    return {"level": level, "tone": tone, "note": note, "score": float(score)}


def project_price(series: pd.DataFrame, horizon: int, sentiment_shift: float) -> pd.DataFrame:
    """Roll the fitted AR(1)+sentiment relationship forward for the scenario panel.

    Fits price(t) = a + b*price(t-1) + c*sentiment(t-1) on the loaded series, then
    projects `horizon` days under (i) sentiment held at its recent mean and
    (ii) sentiment moved by `sentiment_shift` standard deviations.
    """
    p = series["price"].to_numpy()
    s = series["sentiment"].to_numpy()
    if len(p) < 30:
        return pd.DataFrame()

    X = np.column_stack([np.ones(len(p) - 1), p[:-1], s[:-1]])
    coef, *_ = np.linalg.lstsq(X, p[1:], rcond=None)
    a, b, c = coef

    base_s = float(np.mean(s[-14:]))
    last_p = float(p[-1])
    future = pd.date_range(series["date"].iloc[-1] + pd.Timedelta(days=1), periods=horizon)

    def roll(shift):
        out, cur = [], last_p
        for _ in range(horizon):
            cur = a + b * cur + c * (base_s + shift)
            out.append(cur)
        return out

    return pd.DataFrame({
        "date": future,
        "Baseline": roll(0.0),
        "Scenario": roll(sentiment_shift),
    })


def realised_volatility(series: pd.DataFrame, window: int = 30) -> float:
    p = series["price"].to_numpy()[-(window + 1):]
    if len(p) < 3:
        return 0.0
    return float(np.std(np.diff(p) / p[:-1]))
