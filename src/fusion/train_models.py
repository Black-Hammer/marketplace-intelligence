# ── CELL 11: Feature engineering ─────────────────────────────────────────
from sklearn.preprocessing import StandardScaler

s = pd.read_csv('data/processed/series_daily.csv', parse_dates=['date'])
price = s['price'].values.astype(float)
sent  = s['sentiment'].values.astype(float)
N     = len(price)
T     = 14   # look-back window
K     = 7    # lag count

# build feature matrix
rows = []
for t in range(K+14, N):
    ret   = (price[t-1]-price[t-2])/price[t-2] if price[t-2]!=0 else 0
    lags  = [price[t-i] for i in range(1, K+1)]
    rets  = [(price[t-i]-price[t-i-1])/price[t-i-1]
             if price[t-i-1]!=0 else 0 for i in range(1, K+1)]
    rmean = np.mean(price[t-7:t])
    rstd  = np.std(price[t-7:t])+1e-8
    rows.append(lags + rets + [rmean, rstd, sent[t-1]])

X_all = np.array(rows)
y_all = price[K+14:]
S_all = sent[K+14:]

# chronological 70/15/15 split
n  = len(y_all)
t1 = int(n*0.70); t2 = int(n*0.85)
Xtr,Xva,Xte = X_all[:t1], X_all[t1:t2], X_all[t2:]
ytr,yva,yte = y_all[:t1], y_all[t1:t2], y_all[t2:]
Str,Sva,Ste = S_all[:t1], S_all[t1:t2], S_all[t2:]

# z-score scaler fitted on train only
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

# LSTM windows  (T, F)
def make_seq(X, T=14):
    return np.array([X[i-T:i] for i in range(T, len(X))])
def align(X, y, S, T=14):
    return make_seq(X, T), y[T:], S[T:]

Xtr_l, ytr_l, Str_l = align(Xtr_s, ytr, Str)
Xva_l, yva_l, Sva_l = align(Xva_s, yva, Sva)
Xte_l, yte_l, Ste_l = align(Xte_s, yte, Ste)

print(f'Feature matrix: {X_all.shape}  |  test samples: {len(yte_l)}')
print(f'LSTM seq shape: {Xtr_l.shape}')

# ── CELL 12: B3 — ARIMA ──────────────────────────────────────────────────
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error as MAE

history = list(ytr); arima_preds = []
for t in range(len(yte_l)):
    try:
        fc = ARIMA(history, order=(2,1,2)).fit().forecast(1)[0]
    except Exception:
        fc = history[-1]
    arima_preds.append(fc)
    history.append(yva[t] if t < len(yva) else yte[t])
arima_preds = np.array(arima_preds)
print(f'B3 ARIMA  MAE: {MAE(yte_l, arima_preds):.2f}')

# ── CELL 13: B4 — XGBoost ────────────────────────────────────────────────
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

param_dist = dict(
    max_depth=[4,5,6], n_estimators=[300,400,600],
    learning_rate=[0.03,0.05,0.1], subsample=[0.7,0.8,1.0],
    colsample_bytree=[0.7,0.8,1.0]
)
base = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
rs   = RandomizedSearchCV(base, param_dist, n_iter=20, cv=3,
                           scoring='neg_mean_absolute_error',
                           random_state=42, n_jobs=-1)
rs.fit(Xtr_s, ytr)
xgb_preds = rs.best_estimator_.predict(Xte_s)[len(Xte_s)-len(yte_l):]
print(f'B4 XGBoost MAE: {MAE(yte_l, xgb_preds[:len(yte_l)]):.2f}')
print(f'Best params: {rs.best_params_}')

# ── CELL 14: B5 — LSTM (price-only control) ───────────────────────────────
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

def T2(a): return torch.tensor(a, dtype=torch.float32).to(device)

class LSTMModel(nn.Module):
    def __init__(self, inp, hid=64, layers=2, drop=0.3):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, layers, batch_first=True, dropout=drop)
        self.fc   = nn.Linear(hid, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:,-1,:]).squeeze(-1)

def train_lstm_model(Xtr, ytr, Xva, yva, tag='b5', epochs=60, lr=1e-3):
    model   = LSTMModel(Xtr.shape[2]).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    loss_fn = nn.MSELoss()
    tr_dl   = DataLoader(TensorDataset(T2(Xtr),T2(ytr)), batch_size=32, shuffle=True)
    va_dl   = DataLoader(TensorDataset(T2(Xva),T2(yva)), batch_size=64)
    best, cnt = float('inf'), 0
    for ep in range(1, epochs+1):
        model.train()
        for xb,yb in tr_dl:
            opt.zero_grad(); loss_fn(model(xb),yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([loss_fn(model(xb),yb).item() for xb,yb in va_dl])
        sched.step(vl)
        if vl < best:
            best=vl; cnt=0; torch.save(model.state_dict(), f'models/{tag}.pt')
        else:
            cnt+=1
            if cnt>=10: print(f'  Early stop ep {ep}'); break
    model.load_state_dict(torch.load(f'models/{tag}.pt'))
    return model

b5_model  = train_lstm_model(Xtr_l, ytr_l, Xva_l, yva_l, 'b5')
b5_model.eval()
with torch.no_grad():
    lstm_preds = b5_model(T2(Xte_l)).cpu().numpy()
print(f'B5 LSTM (control) MAE: {MAE(yte_l, lstm_preds):.2f}')

# ── CELL 15: B6 — Sentiment-only MLP ─────────────────────────────────────
sent_model = nn.Sequential(
    nn.Linear(1,32), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(32,16), nn.ReLU(), nn.Linear(16,1)
).to(device)
opt_s   = torch.optim.Adam(sent_model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
tr_dl_s = DataLoader(
    TensorDataset(T2(Str_l.reshape(-1,1)), T2(ytr_l)),
    batch_size=32, shuffle=True)
for _ in range(60):
    sent_model.train()
    for xb,yb in tr_dl_s:
        opt_s.zero_grad(); loss_fn(sent_model(xb).squeeze(),yb).backward(); opt_s.step()
sent_model.eval()
with torch.no_grad():
    sent_preds = sent_model(T2(Ste_l.reshape(-1,1))).squeeze().cpu().numpy()
print(f'B6 Sentiment-only MAE: {MAE(yte_l, sent_preds):.2f}')

# ── CELL 16: EXP — Late-Fusion FusionRegressor ───────────────────────────
import torch.nn.functional as F

class FusionRegressor(nn.Module):
    def __init__(self, price_feats, d_T=64, d_S=64, d_h=64, drop=0.3):
        super().__init__()
        self.lstm      = nn.LSTM(price_feats, d_S, 2,
                                  batch_first=True, dropout=drop)
        self.text_proj = nn.Sequential(
            nn.Linear(1, d_T), nn.ReLU(), nn.Dropout(drop))
        self.head      = nn.Sequential(
            nn.LayerNorm(d_T+d_S),
            nn.Linear(d_T+d_S, d_h), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(d_h, 1))
    def forward(self, x_price, x_sent):
        v_time, _ = self.lstm(x_price)
        v_time = F.layer_norm(v_time[:,-1,:], (v_time.shape[-1],))
        v_text = self.text_proj(x_sent.unsqueeze(-1))
        v_text = F.layer_norm(v_text, (v_text.shape[-1],))
        return self.head(torch.cat([v_text, v_time], dim=-1)).squeeze(-1)

fusion = FusionRegressor(Xtr_l.shape[2]).to(device)
opt_f  = torch.optim.Adam(fusion.parameters(), lr=1e-3, weight_decay=1e-4)
sched_f= torch.optim.lr_scheduler.ReduceLROnPlateau(opt_f, patience=5)
tr_dl_f= DataLoader(
    TensorDataset(T2(Xtr_l), T2(Str_l), T2(ytr_l)),
    batch_size=32, shuffle=True)
va_dl_f= DataLoader(
    TensorDataset(T2(Xva_l), T2(Sva_l), T2(yva_l)), batch_size=64)

best_f, cnt_f = float('inf'), 0
for ep in range(1, 61):
    fusion.train()
    for xp,xs,yb in tr_dl_f:
        opt_f.zero_grad(); loss_fn(fusion(xp,xs),yb).backward(); opt_f.step()
    fusion.eval()
    with torch.no_grad():
        vl = np.mean([loss_fn(fusion(xp,xs),yb).item() for xp,xs,yb in va_dl_f])
    sched_f.step(vl)
    if vl < best_f:
        best_f=vl; cnt_f=0; torch.save(fusion.state_dict(),'models/fusion.pt')
    else:
        cnt_f+=1
        if cnt_f>=10: print(f'  Early stop ep {ep}'); break

fusion.load_state_dict(torch.load('models/fusion.pt'))
fusion.eval()
with torch.no_grad():
    fusion_preds = fusion(T2(Xte_l), T2(Ste_l)).cpu().numpy()
print(f'EXP Fusion  MAE: {MAE(yte_l, fusion_preds):.2f}')

# ── CELL 17: Save forecasts.csv ───────────────────────────────────────────
test_dates = s['date'].values[-(len(yte_l)):]
fc = pd.DataFrame({
    'date':          test_dates,
    'y_true':        yte_l,
    'arima':         arima_preds,
    'xgboost':       xgb_preds[:len(yte_l)],
    'lstm_price':    lstm_preds,
    'sentiment_only':sent_preds,
    'fusion':        fusion_preds,
})
fc.to_csv('data/processed/forecasts.csv', index=False)
print('✓ forecasts.csv saved')

for col in ['arima','xgboost','lstm_price','sentiment_only','fusion']:
    print(f'  {col:20s}  MAE={MAE(fc.y_true, fc[col]):>10.2f}')

