# Marketplace Intelligence — Full Pipeline
**MSc Data Science | AUN | Abubakar Salawu (A00019166)**

This notebook runs the complete 5-stage pipeline end-to-end on a free Colab T4 GPU.
It produces the three CSV files that `ch6_analysis.py` requires:
- `classification.csv` (Stage 1 — sentiment models → RQ1)
- `series_daily.csv`  (Stage 3 — daily price + sentiment index → RQ2)
- `forecasts.csv`     (Stage 4 — all forecasting arms → RQ3)

**Runtime:** ~40–60 min on a free T4 (mostly AfriBERTa fine-tuning).

> **IMPORTANT:** Stage 1 uses real NaijaSenti data from HuggingFace.
> The price series uses a structured synthetic generator until you have
> 90+ days of real scraped data — it is clearly marked and must be
> replaced before the final thesis results.

  # ── CELL 1: Install dependencies ──────────────────────────────────────────
!pip install -q transformers datasets accelerate scikit-learn \
               statsmodels xgboost tabulate numpy pandas scipy
print('✓ dependencies installed')

  # ── CELL 2: Imports and directory setup ───────────────────────────────────
import os, warnings, pickle, logging
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

PROC = Path('data/processed'); PROC.mkdir(parents=True, exist_ok=True)
Path('data/raw').mkdir(parents=True, exist_ok=True)
Path('models').mkdir(parents=True, exist_ok=True)
Path('ch6_tables').mkdir(parents=True, exist_ok=True)
print('✓ directories created')

# ── CELL 3: Load NaijaSenti ───────────────────────────────────────────────
from datasets import load_dataset

LABEL_MAP = {'positive': 0, 'negative': 1, 'neutral': 2}
dfs = []
for lang in ['hau', 'ibo', 'yor', 'pcm']:
    try:
        ds = load_dataset('HausaNLP/NaijaSenti', lang, trust_remote_code=True)
        for split_name, split in ds.items():
            df = split.to_pandas()
            df['language'] = lang
            df['split']    = split_name
            dfs.append(df)
        print(f'  loaded {lang}')
    except Exception as e:
        print(f'  {lang} failed: {e}')

combined = pd.concat(dfs, ignore_index=True)
combined.columns = [c.lower() for c in combined.columns]
# normalise label column
if 'label' not in combined.columns and 'sentiment' in combined.columns:
    combined = combined.rename(columns={'sentiment': 'label'})
combined['label'] = combined['label'].str.lower().map(LABEL_MAP)
combined = combined.dropna(subset=['tweet', 'label'])
combined['label'] = combined['label'].astype(int)
print(f'\nTotal rows: {len(combined):,} | label dist:\n{combined.label.value_counts()}')

# ── CELL 4: Stratified split ─────────────────────────────────────────────
from sklearn.model_selection import train_test_split

# use a manageable subset for Colab speed (full set takes ~2hr)
# remove the sample() call to train on everything
data = combined.sample(n=min(20000, len(combined)), random_state=42)
X, y = data['tweet'].tolist(), data['label'].tolist()

X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.15/0.85, stratify=y_tv, random_state=42)

print(f'train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}')

# ── CELL 5: B1 — TF-IDF + LinearSVC ─────────────────────────────────────
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report

vec_w = TfidfVectorizer(analyzer='word', ngram_range=(1,2),
                        max_features=30000, sublinear_tf=True, min_df=2)
vec_c = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5),
                        max_features=20000, sublinear_tf=True, min_df=3)

Xtr = sp.hstack([vec_w.fit_transform(X_train), vec_c.fit_transform(X_train)])
Xva = sp.hstack([vec_w.transform(X_val),   vec_c.transform(X_val)])
Xte = sp.hstack([vec_w.transform(X_test),  vec_c.transform(X_test)])

b1 = LinearSVC(C=1.0, class_weight='balanced', max_iter=3000)
b1.fit(Xtr, y_train)
b1_preds = b1.predict(Xte)

print('B1 val macro-F1:', f1_score(y_val, b1.predict(Xva), average='macro'))
print('B1 test macro-F1:', f1_score(y_test, b1_preds, average='macro'))
print(classification_report(y_test, b1_preds,
      target_names=['positive','negative','neutral']))

# ── CELL 6: B2 — AfriBERTa (no LAFT) ─────────────────────────────────────
# Runtime: ~15 min on T4
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           Trainer, TrainingArguments)
from datasets import Dataset

MODEL_ID  = 'castorini/afriberta_large'
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True,
                     max_length=128, padding='max_length')

def make_ds(texts, labels):
    return Dataset.from_dict({'text': texts, 'label': labels}).map(
        tokenize, batched=True, remove_columns=['text'])

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {'macro_f1': f1_score(p.label_ids, preds, average='macro')}

def train_cls(model_id, out_dir, epochs=3):
    model = AutoModelForSequenceClassification.from_pretrained(
                model_id, num_labels=3, ignore_mismatched_sizes=True)
    tr_ds = make_ds(X_train, y_train)
    va_ds = make_ds(X_val,   y_val)
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        learning_rate=2e-5, evaluation_strategy='epoch',
        save_strategy='best', load_best_model_at_end=True,
        metric_for_best_model='macro_f1', fp16=True, report_to='none',
        logging_steps=50
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=tr_ds, eval_dataset=va_ds,
                      compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return trainer

train_cls(MODEL_ID, 'models/b2_afriberta', epochs=3)
print('B2 training complete')

# ── CELL 7: LAFT — continued MLM then task fine-tuning ───────────────────
# Runtime: ~15 min on T4
from transformers import (AutoModelForMaskedLM,
                           DataCollatorForLanguageModeling)

# Step 7a: continued MLM on the in-domain text (the training tweets)
mlm_model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
def tok_mlm(batch):
    return tokenizer(batch['text'], truncation=True,
                     max_length=128, padding=False)

mlm_ds = Dataset.from_dict({'text': X_train}).map(
    tok_mlm, batched=True, remove_columns=['text'])
collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)

mlm_args = TrainingArguments(
    output_dir='models/laft_afriberta', num_train_epochs=2,
    per_device_train_batch_size=16, learning_rate=5e-5,
    fp16=True, save_strategy='epoch', report_to='none', logging_steps=100
)
Trainer(model=mlm_model, args=mlm_args, train_dataset=mlm_ds,
        data_collator=collator).train()
mlm_model.save_pretrained('models/laft_afriberta')
tokenizer.save_pretrained('models/laft_afriberta')
print('LAFT MLM complete')

# Step 7b: task fine-tuning from the LAFT checkpoint
train_cls('models/laft_afriberta', 'models/laft_cls', epochs=3)
print('LAFT classifier complete')

# ── CELL 8: Get test predictions and save classification.csv ─────────────
from transformers import pipeline

def get_preds(model_dir, texts, batch_size=32):
    pipe = pipeline('text-classification', model=model_dir,
                    tokenizer=model_dir, device=0,
                    truncation=True, max_length=128,
                    batch_size=batch_size)
    lmap = {'LABEL_0': 0, 'LABEL_1': 1, 'LABEL_2': 2}
    return np.array([lmap[r['label']] for r in pipe(texts)])

b2_preds   = get_preds('models/b2_afriberta', X_test)
laft_preds = get_preds('models/laft_cls',     X_test)

clf_df = pd.DataFrame({
    'y_true':                y_test,
    'b1_svm_tfidf':          b1_preds,
    'b2_transformer_nolaft': b2_preds,
    'laft_afriberta':        laft_preds,
})
clf_df.to_csv('data/processed/classification.csv', index=False)
print('✓ classification.csv saved')

for name, preds in [('B1', b1_preds), ('B2', b2_preds), ('LAFT', laft_preds)]:
    print(f'{name} macro-F1: {f1_score(y_test, preds, average="macro"):.4f}')

# ── CELL 9: Check for real price data ────────────────────────────────────
real_path = Path('data/processed/price_series.csv')
if real_path.exists():
    df = pd.read_csv(real_path)
    print(f'✓ Real price data found: {len(df):,} rows')
    print(df.head())
    USE_REAL_PRICES = True
else:
    print('⚠  No real price data found.')
    print('   Running structured synthetic generator (Stage 2 placeholder).')
    print('   Replace with real scraped data before final results.')
    USE_REAL_PRICES = False

# ── CELL 10: Build price + sentiment series ───────────────────────────────
# If USE_REAL_PRICES is True, this cell builds the daily sentiment index
# from the LAFT classifier applied to dated posts.
# If False, it generates a structured synthetic series (PLACEHOLDER).

import warnings; warnings.filterwarnings('ignore')
N   = 300   # days
rng = np.random.default_rng(42)

if not USE_REAL_PRICES:
    # ── SYNTHETIC (PLACEHOLDER — replace with real data) ──
    print('Building structured synthetic series ...')
    # sentiment that genuinely leads price by 1 day
    sent_raw = rng.normal(0, 1, N)
    price    = np.zeros(N); price[0] = 45000.0   # ≈ ₦45,000 baseline
    for t in range(1, N):
        price[t] = (0.55 * price[t-1]
                    + 0.40 * 45000
                    + 800  * sent_raw[t-1]      # sentiment leads price
                    + rng.normal(0, 400))
    dates = pd.date_range('2024-07-01', periods=N, freq='D')
    series = pd.DataFrame({
        'date':      dates.strftime('%Y-%m-%d'),
        'price':     price,
        'sentiment': sent_raw,
    })
    # mark as synthetic
    print('  ⚠  SYNTHETIC placeholder — replace with real scraped data')
else:
    # ── REAL DATA PATH ──
    # Apply LAFT classifier to dated posts to build sentiment index
    posts_path = Path('data/processed/posts_dated.csv')
    price_df   = pd.read_csv(real_path, parse_dates=['date'])
    daily_price = (price_df.groupby('date')['price_ngn']
                           .median().sort_index().asfreq('D').interpolate())
    if posts_path.exists():
        posts = pd.read_csv(posts_path)
        # get LAFT predictions on dated posts
        post_texts = posts['text'].tolist()
        post_preds = get_preds('models/laft_cls', post_texts)
        posts['pol'] = pd.Series(post_preds).map({0:1, 1:-1, 2:0}).values
        posts['date'] = pd.to_datetime(posts['date'])
        daily_sent = posts.groupby('date')['pol'].mean()
        sent_aligned = daily_sent.reindex(daily_price.index).fillna(0)
    else:
        print('  ⚠  posts_dated.csv not found — using zero sentiment.')
        sent_aligned = pd.Series(0.0, index=daily_price.index)
    series = pd.DataFrame({
        'date':      daily_price.index.strftime('%Y-%m-%d'),
        'price':     daily_price.values,
        'sentiment': sent_aligned.values,
    })

series.to_csv('data/processed/series_daily.csv', index=False)
print(f'✓ series_daily.csv saved: {len(series)} rows')
series.tail(3)

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

# ── CELL 18: Run ch6_analysis.py ─────────────────────────────────────────
!pip install -q tabulate
!python ch6_analysis.py \
    --indir  data/processed \
    --outdir ch6_tables \
    --maxlag 4
print('\n✓ ch6_analysis complete — tables in ch6_tables/')
import os; print(os.listdir('ch6_tables'))

# ── CELL 19: Visualise table_6_4_forecasting ──────────────────────────────
import matplotlib.pyplot as plt
import re

tbl_path = 'ch6_tables/table_6_4_forecasting.md'
rows = []
with open(tbl_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith('|') and '---' not in line:
            cells = [c.strip() for c in line.split('|') if c.strip()]
            rows.append(cells)

header, data = rows[0], rows[1:]
df_viz = pd.DataFrame(data, columns=header)
for col in ['MAE','RMSE','MAE %impr vs control']:
    df_viz[col] = pd.to_numeric(df_viz[col], errors='coerce')

COLS = {'MAE':'MAE (₦)','RMSE':'RMSE (₦)','MAE %impr vs control':'MAE % Improvement'}
PAL  = ['#4338CA','#0D9488','#94A3B8','#64748B','#EA580C']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Chapter 6 — Forecasting Model Comparison', fontsize=15, fontweight='bold')
for ax, (col, label) in zip(axes, COLS.items()):
    vals = df_viz[col]
    bars = ax.bar(df_viz['Model'], vals, color=PAL[:len(df_viz)],
                   edgecolor='white', linewidth=0.8)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=30)
    ax.spines[['top','right']].set_visible(False)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+abs(vals.max()*0.01),
                f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    if col == 'MAE %impr vs control':
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.tight_layout()
plt.savefig('ch6_tables/figure_6_forecasting_comparison.png',
            dpi=200, bbox_inches='tight')
plt.show()
print('✓ figure saved to ch6_tables/')

# ── CELL 20: Download all outputs ─────────────────────────────────────────
from google.colab import files
import zipfile, os

with zipfile.ZipFile('chapter6_results.zip','w') as z:
    for folder in ['data/processed','ch6_tables']:
        for f in os.listdir(folder):
            z.write(os.path.join(folder,f), os.path.join(folder,f))

files.download('chapter6_results.zip')
print('✓ chapter6_results.zip downloaded')
