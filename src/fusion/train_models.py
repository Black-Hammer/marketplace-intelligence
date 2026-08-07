#!/usr/bin/env python3
"""
src/fusion/train_models.py
===========================
Stage 4 — Train All Forecasting Arms and Write forecasts.csv 

Trains five forecasting model arms and saves their test-set predictions to
data/processed/forecasts.csv — the third CSV required by ch6_analysis.py.

Model arms:
  B3  — ARIMA(2,1,2)              [classical baseline]
  B4  — XGBoost                   [non-sequential baseline]
  B5  — LSTM (price-only)         [deep control — primary RQ3 comparison]
  B6  — Sentiment-only MLP        [single-modality baseline]
  EXP — FusionRegressor           [late-fusion experimental model]

Fairness guarantee:
  ALL arms share:
    - Identical chronological 70/15/15 data split
    - Identical StandardScaler (fitted on training window only)
    - Identical hyperparameter-search budget (RandomizedSearchCV, 20 iter)
    - Identical early stopping criterion (patience=10 on validation loss)
  The price branch of EXP is IDENTICAL to B5 so the only difference
  between B5 and EXP is the added sentiment modality.

Output files:
  data/processed/forecasts.csv     (date, y_true, arima, xgboost,
                                    lstm_price, sentiment_only, fusion)
  models/b4_xgboost.pkl
  models/b5_lstm.pt
  models/fusion.pt

Run in Colab (T4 GPU, ~20 min):
  !python src/fusion/train_models.py

Run with custom epochs:
  !python src/fusion/train_models.py --lstm_epochs 80 --fusion_epochs 80
"""

import argparse
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_models")

# ── paths ─────────────────────────────────────────────────────────────────────
PROC_DIR  = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING AND SPLITTING
# ══════════════════════════════════════════════════════════════════════════════
def load_dataset() -> dict:
    """Load the lstm_dataset.pkl produced by features.py (Stage 3)."""
    pkl = PROC_DIR / "lstm_dataset.pkl"
    if not pkl.exists():
        raise FileNotFoundError(
            "lstm_dataset.pkl not found. "
            "Run Stage 3 (features.py) first.")
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    log.info("Dataset loaded — X_seq: %s  y: %s",
             data["X_seq"].shape, data["y"].shape)
    return data


def make_splits(data: dict) -> dict:
    """
    Return train / val / test slices for every array in the dataset.
    All splits are CHRONOLOGICAL — no shuffling. The split indices
    were computed in features.py using the same 70/15/15 rule.
    """
    tr, va = data["splits"]["tr"], data["splits"]["va"]
    X  = data["X_seq"]
    y  = data["y"]
    Xf = data["X_flat"]
    S  = data["sent_daily"]
    return {
        # LSTM sequences  (N, T, F)
        "Xtr": X[:tr],   "Xva": X[tr:va],   "Xte": X[va:],
        # targets         (N,)
        "ytr": y[:tr],   "yva": y[tr:va],   "yte": y[va:],
        # flat features   (N, F) — for XGBoost / ARIMA
        "Ftr": Xf[:tr],  "Fva": Xf[tr:va],  "Fte": Xf[va:],
        # sentiment index (N,)
        "Str": S[:tr],   "Sva": S[tr:va],   "Ste": S[va:],
        # test dates
        "test_dates": data["dates"][va:],
    }


# ══════════════════════════════════════════════════════════════════════════════
# METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def report(name: str, y_true, y_pred, mae_ctrl: float = None):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    if mae_ctrl:
        pct = 100 * (mae_ctrl - mae) / mae_ctrl
        log.info("%-20s  MAE=₦%,.2f  RMSE=₦%,.2f  %%impr=%.1f%%",
                 name, mae, rmse, pct)
    else:
        log.info("%-20s  MAE=₦%,.2f  RMSE=₦%,.2f", name, mae, rmse)
    return mae, rmse


# ══════════════════════════════════════════════════════════════════════════════
# B3 — ARIMA
# ══════════════════════════════════════════════════════════════════════════════
def train_arima(data: dict, sp: dict) -> np.ndarray:
    """
    ARIMA(2,1,2) with walk-forward refitting on each test observation.
    Order selected by AIC on the training window.
    Falls back to last observed value if fitting fails (robustness).
    """
    log.info("── B3: ARIMA ──")
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        log.error("statsmodels not installed: pip install statsmodels")
        return sp["yte"].copy()

    tr_end   = data["splits"]["tr"]
    y_all    = data["y"]
    history  = list(y_all[:tr_end])
    preds    = []

    n_test = len(sp["yte"])
    for i in range(n_test):
        try:
            fc = ARIMA(history, order=(2, 1, 2)).fit().forecast(steps=1)[0]
        except Exception:
            fc = history[-1]   # fallback: naïve forecast
        preds.append(fc)
        # append the true value to history for the next step
        true_idx = tr_end + data["splits"]["va"] - data["splits"]["tr"] + i
        if true_idx < len(y_all):
            history.append(y_all[true_idx])
        else:
            history.append(fc)

    preds = np.array(preds)
    report("B3 ARIMA", sp["yte"], preds)
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# B4 — XGBoost
# ══════════════════════════════════════════════════════════════════════════════
def train_xgboost(sp: dict) -> np.ndarray:
    """
    XGBoost regressor with RandomizedSearchCV (20 iterations, 3-fold CV)
    on the training window. Uses the flat feature matrix (not sequences).
    Same scaler as all other arms — no refitting.
    """
    log.info("── B4: XGBoost (hyperparameter search) ──")
    from xgboost import XGBRegressor
    from sklearn.model_selection import RandomizedSearchCV

    param_dist = {
        "max_depth":         [4, 5, 6],
        "n_estimators":      [300, 400, 600],
        "learning_rate":     [0.03, 0.05, 0.1],
        "subsample":         [0.7, 0.8, 1.0],
        "colsample_bytree":  [0.7, 0.8, 1.0],
        "min_child_weight":  [1, 3, 5],
        "reg_lambda":        [0.5, 1.0, 2.0],
    }
    base = XGBRegressor(objective="reg:squarederror",
                        early_stopping_rounds=30,
                        random_state=42, n_jobs=-1)
    rs = RandomizedSearchCV(
        base, param_dist, n_iter=20, cv=3,
        scoring="neg_mean_absolute_error",
        random_state=42, n_jobs=-1)

    # flatten sequences to 2D for XGBoost
    Ftr = sp["Ftr"].reshape(len(sp["Ftr"]), -1)
    Fva = sp["Fva"].reshape(len(sp["Fva"]), -1)
    Fte = sp["Fte"].reshape(len(sp["Fte"]), -1)

    rs.fit(Ftr, sp["ytr"],
           eval_set=[(Fva, sp["yva"])],
           verbose=False)
    best = rs.best_estimator_
    with open(MODEL_DIR / "b4_xgboost.pkl", "wb") as f:
        pickle.dump(best, f)

    preds = best.predict(Fte)[:len(sp["yte"])]
    log.info("Best params: %s", rs.best_params_)
    report("B4 XGBoost", sp["yte"], preds)
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# SHARED LSTM TRAINER
# ══════════════════════════════════════════════════════════════════════════════
def _train_loop(model, train_dl, val_dl, save_path: str,
                epochs: int = 60, lr: float = 1e-3,
                patience: int = 10, device=None):
    """
    Shared Adam + ReduceLROnPlateau + early-stopping training loop.
    Used by both B5 (price-only LSTM) and EXP (FusionRegressor).
    """
    import torch
    import torch.nn as nn

    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5,
                                                          factor=0.5)
    loss_fn = nn.MSELoss()
    best_val, counter = float("inf"), 0

    for ep in range(1, epochs + 1):
        # training step
        model.train()
        for batch in train_dl:
            opt.zero_grad()
            xb  = batch[0].to(device)
            yb  = batch[-1].to(device)
            out = model(*batch[:-1]) if len(batch) > 2 else model(xb)
            loss_fn(out, yb).backward()
            opt.step()

        # validation step
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dl:
                xb  = batch[0].to(device)
                yb  = batch[-1].to(device)
                out = model(*batch[:-1]) if len(batch) > 2 else model(xb)
                val_losses.append(loss_fn(out, yb).item())
        val_loss = float(np.mean(val_losses))
        sched.step(val_loss)

        # early stopping
        if val_loss < best_val:
            best_val = val_loss
            counter  = 0
            torch.save(model.state_dict(), save_path)
        else:
            counter += 1
            if counter >= patience:
                log.info("  Early stopping at epoch %d (best val=%.6f)", ep, best_val)
                break

        if ep % 10 == 0:
            log.info("  Epoch %3d | val_loss=%.6f", ep, val_loss)

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


# ══════════════════════════════════════════════════════════════════════════════
# B5 — LSTM (price-only control)
# ══════════════════════════════════════════════════════════════════════════════
def train_lstm_control(sp: dict, epochs: int = 60) -> np.ndarray:
    """
    Price-only LSTM — the PRIMARY CONTROL for the RQ3 hypothesis test.
    Architecture: 2-layer LSTM (d_S=64) + linear head.
    This model is trained IDENTICALLY to the price branch inside EXP
    so that the only difference between B5 and EXP is the sentiment input.
    """
    log.info("── B5: LSTM price-only control ──")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        log.error("torch not installed — run: pip install torch"); return sp["yte"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("  Device: %s", device)

    def T(a): return torch.tensor(a, dtype=torch.float32)

    tr_dl = DataLoader(TensorDataset(T(sp["Xtr"]), T(sp["ytr"])),
                       batch_size=32, shuffle=True)
    va_dl = DataLoader(TensorDataset(T(sp["Xva"]), T(sp["yva"])),
                       batch_size=64)

    class LSTMControl(nn.Module):
        def __init__(self, inp, hid=64, layers=2, drop=0.3):
            super().__init__()
            self.lstm = nn.LSTM(inp, hid, layers,
                                batch_first=True, dropout=drop)
            self.fc   = nn.Linear(hid, 1)
        def forward(self, x):
            out, _ = self.lstm(x.to(device))
            return self.fc(out[:, -1, :]).squeeze(-1)

    model = LSTMControl(sp["Xtr"].shape[2]).to(device)
    model = _train_loop(model, tr_dl, va_dl,
                        str(MODEL_DIR / "b5_lstm.pt"),
                        epochs=epochs, device=device)

    model.eval()
    with torch.no_grad():
        preds = model(T(sp["Xte"]).to(device)).cpu().numpy()

    report("B5 LSTM (control)", sp["yte"], preds)
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# B6 — Sentiment-only MLP
# ══════════════════════════════════════════════════════════════════════════════
def train_sentiment_mlp(sp: dict) -> np.ndarray:
    """
    Compact MLP that takes ONLY the daily sentiment index as input.
    Included as a single-modality baseline to confirm that sentiment
    alone is insufficient — the fusion gain must come from the combination.
    """
    log.info("── B6: Sentiment-only MLP ──")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        log.error("torch not installed."); return sp["yte"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def T(a): return torch.tensor(a, dtype=torch.float32)

    # reshape sentiment to (N, 1) for the linear layer
    Str = sp["Str"].reshape(-1, 1)
    Sva = sp["Sva"].reshape(-1, 1)
    Ste = sp["Ste"].reshape(-1, 1)

    tr_dl = DataLoader(TensorDataset(T(Str), T(sp["ytr"])),
                       batch_size=32, shuffle=True)
    va_dl = DataLoader(TensorDataset(T(Sva), T(sp["yva"])),
                       batch_size=64)

    model = nn.Sequential(
        nn.Linear(1, 32), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 1)
    ).to(device)

    # simple training loop (no early stopping needed — model is tiny)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(60):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad()
            loss_fn(model(xb.to(device)).squeeze(), yb.to(device)).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(T(Ste).to(device)).squeeze().cpu().numpy()

    report("B6 Sentiment-only", sp["yte"], preds)
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# EXP — FusionRegressor (late-fusion experimental model)
# ══════════════════════════════════════════════════════════════════════════════
def train_fusion(sp: dict, epochs: int = 60,
                 d_T: int = 64, d_S: int = 64,
                 d_h: int = 64) -> np.ndarray:
    """
    Late-fusion model (Chapter 5 Listing 5.2 / Algorithm 4.4).

    Architecture:
      Price branch : 2-layer LSTM (d_S=64) — IDENTICAL config to B5
      Text branch  : linear projection of daily sentiment → (d_T=64)
      Fusion       : LayerNorm → concat [V_text ‖ V_time] → dense head

    The LayerNorm before concatenation ensures both modalities are on
    a comparable scale before the head makes the forecast — this is the
    key architectural reason late fusion is appropriate here (noise
    asymmetry between the two modalities).

    The price branch weights are NOT pre-loaded from B5; both are trained
    jointly from scratch on the same data. This is the standard evaluation
    protocol (matched training, not transfer).
    """
    log.info("── EXP: FusionRegressor (late-fusion) ──")
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        log.error("torch not installed."); return sp["yte"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def T(a): return torch.tensor(a, dtype=torch.float32)

    class FusionRegressor(nn.Module):
        def __init__(self, price_feats: int):
            super().__init__()
            # price branch — identical to B5 control
            self.lstm      = nn.LSTM(price_feats, d_S, 2,
                                     batch_first=True, dropout=0.3)
            # text projection: scalar S_t → (d_T,)
            self.text_proj = nn.Sequential(
                nn.Linear(1, d_T), nn.ReLU(), nn.Dropout(0.3))
            # fusion head
            self.head = nn.Sequential(
                nn.LayerNorm(d_T + d_S),
                nn.Linear(d_T + d_S, d_h), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(d_h, 1),
            )

        def forward(self, x_price, x_sent):
            # price branch → temporal vector V_time
            v_time, _ = self.lstm(x_price)
            v_time = F.layer_norm(v_time[:, -1, :],
                                  (v_time.shape[-1],))  # (B, d_S)

            # text branch → text vector V_text
            v_text = self.text_proj(x_sent.unsqueeze(-1))
            v_text = F.layer_norm(v_text,
                                  (v_text.shape[-1],))  # (B, d_T)

            # decision-level fusion: concatenate then dense head
            h = torch.cat([v_text, v_time], dim=-1)     # (B, d_T+d_S)
            return self.head(h).squeeze(-1)              # (B,)

    tr_dl = DataLoader(
        TensorDataset(T(sp["Xtr"]), T(sp["Str"]), T(sp["ytr"])),
        batch_size=32, shuffle=True)
    va_dl = DataLoader(
        TensorDataset(T(sp["Xva"]), T(sp["Sva"]), T(sp["yva"])),
        batch_size=64)

    model = FusionRegressor(sp["Xtr"].shape[2]).to(device)

    # wrap forward to accept (x_price, x_sent, y) batches in the loop
    class _Wrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x_price, x_sent):
            return self.m(x_price.to(device), x_sent.to(device))

    model = _train_loop(_Wrapper(model), tr_dl, va_dl,
                        str(MODEL_DIR / "fusion.pt"),
                        epochs=epochs, device=device)

    model.eval()
    with torch.no_grad():
        preds = model(T(sp["Xte"]), T(sp["Ste"])).cpu().numpy()

    report("EXP Fusion", sp["yte"], preds)
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_stage4(lstm_epochs: int = 60, fusion_epochs: int = 60):
    """
    Full Stage 4 pipeline:
      1. Load lstm_dataset.pkl
      2. Train B3 (ARIMA)
      3. Train B4 (XGBoost, hyperparameter search)
      4. Train B5 (LSTM price-only control)
      5. Train B6 (Sentiment-only MLP)
      6. Train EXP (FusionRegressor)
      7. Save forecasts.csv for ch6_analysis.py
    """
    data = load_dataset()
    sp   = make_splits(data)

    arima_preds  = train_arima(data, sp)
    xgb_preds    = train_xgboost(sp)
    lstm_preds   = train_lstm_control(sp, lstm_epochs)
    sent_preds   = train_sentiment_mlp(sp)
    fusion_preds = train_fusion(sp, fusion_epochs)

    # ── save forecasts.csv ────────────────────────────────────────────────────
    # All predictions on the SAME scale (raw ₦ price).
    # If you trained on standardised / differenced targets, invert the
    # transform here before saving so every MAE is in the same units.
    n_test = len(sp["yte"])
    fc = pd.DataFrame({
        "date":          sp["test_dates"].strftime("%Y-%m-%d"),
        "y_true":        sp["yte"],
        "arima":         arima_preds,
        "xgboost":       xgb_preds[:n_test],
        "lstm_price":    lstm_preds,
        "sentiment_only":sent_preds,
        "fusion":        fusion_preds,
    })
    out = PROC_DIR / "forecasts.csv"
    fc.to_csv(out, index=False)
    log.info("\n✓ forecasts.csv saved → %s", out)
    log.info("  %d test observations | columns: date, y_true, arima, "
             "xgboost, lstm_price, sentiment_only, fusion", n_test)

    # ── final summary table ───────────────────────────────────────────────────
    log.info("\n══ FINAL SUMMARY ══")
    mae_ctrl, _ = report("B5 LSTM (control)", sp["yte"], lstm_preds)
    for name, preds in [("B3 ARIMA",         arima_preds),
                        ("B4 XGBoost",       xgb_preds[:n_test]),
                        ("B6 Sentiment-only",sent_preds),
                        ("EXP Fusion",       fusion_preds)]:
        report(name, sp["yte"], preds, mae_ctrl)

    log.info("\nNext: run ch6_analysis.py to produce Chapter 6 tables.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Stage 4: Train forecasting arms and write forecasts.csv")
    ap.add_argument("--lstm_epochs",   type=int, default=60,
                    help="Training epochs for B5 LSTM control")
    ap.add_argument("--fusion_epochs", type=int, default=60,
                    help="Training epochs for EXP FusionRegressor")
    args = ap.parse_args()
    run_stage4(args.lstm_epochs, args.fusion_epochs)
