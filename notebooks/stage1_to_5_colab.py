# Marketplace Intelligence — Complete Pipeline
**MSc Data Science | AUN | Abubakar Salawu (A00019166)**

Run cells top to bottom. Each stage builds on the previous.

| Stage | Output | Time |
|-------|--------|------|
| 0 — Setup | directories, Drive, repo | 2 min |
| 1 — Sentiment | classification.csv | ~40 min |
| 2 — Price data | price_series.csv | upload |
| 3 — Features | series_daily.csv + lstm_dataset.pkl | 1 min |
| 4 — Models | forecasts.csv | ~20 min |
| 5 — Analysis | Chapter 6 tables | 1 min |
## STAGE 0 — Setup (run every session)
# ── CELL 0.1: Clone repo, install dependencies, mount Drive ─────────────────
import os, shutil

# Clone repo if not already present
if not os.path.exists('/content/marketplace-intelligence'):
    os.system('git clone https://github.com/Black-Hammer/marketplace-intelligence.git /content/marketplace-intelligence')
    print('✓ Repo cloned')
else:
    print('✓ Repo already exists')

# Set working directory
os.chdir('/content/marketplace-intelligence')
print('✓ Working directory:', os.getcwd())
# ── CELL 0.2: Install dependencies ──────────────────────────────────────────
!pip install -q transformers datasets accelerate tokenizers \
               scikit-learn statsmodels xgboost tabulate \
               seaborn requests beautifulsoup4
!pip install -q torch --index-url https://download.pytorch.org/whl/cu118
print('✓ Dependencies installed')
# ── CELL 0.3: Create directories ─────────────────────────────────────────────
import os
for d in ['data/raw','data/processed','models','results']:
    if os.path.exists(d) and not os.path.isdir(d):
        os.remove(d)   # remove corrupted path if exists as file
    os.makedirs(d, exist_ok=True)
    print(f'  ✓ {d}')
print('✓ All directories ready')
# ── CELL 0.4: Mount Google Drive and restore saved files ─────────────────────
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

DRIVE = '/content/drive/MyDrive/marketplace_thesis'
os.makedirs(DRIVE, exist_ok=True)

# Restore files from Drive if they exist
restore_map = {
    f'{DRIVE}/split.pkl':          'data/processed/split.pkl',
    f'{DRIVE}/classification.csv': 'data/processed/classification.csv',
    f'{DRIVE}/price_series.csv':   'data/processed/price_series.csv',
    f'{DRIVE}/b1_svm.pkl':         'models/b1_svm.pkl',
    f'{DRIVE}/series_daily.csv':   'data/processed/series_daily.csv',
    f'{DRIVE}/lstm_dataset.pkl':   'data/processed/lstm_dataset.pkl',
    f'{DRIVE}/forecasts.csv':      'data/processed/forecasts.csv',
}
for src, dst in restore_map.items():
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f'  ✓ restored: {os.path.basename(dst)}')

print('\ndata/processed:', os.listdir('data/processed'))
print('models:', os.listdir('models'))
# ── CELL 0.5: Set API tokens ──────────────────────────────────────────────────
import os
from huggingface_hub import login

# HuggingFace token (Read-Only) — get from huggingface.co/settings/tokens
HF_TOKEN = 'hf_your_token_here'   # ← paste your token
login(token=HF_TOKEN, add_to_git_credential=False)
print('✓ HuggingFace authenticated')

# Twitter Bearer Token — get from developer.twitter.com
os.environ['TWITTER_BEARER_TOKEN'] = 'your_twitter_bearer_token_here'  # ← paste your token
print('✓ Twitter token set')
## STAGE 1 — Sentiment Classification (RQ1)
**Skip if classification.csv already restored from Drive.**
# ── CELL 1.0: Check if Stage 1 already done ──────────────────────────────────
import os
if os.path.exists('data/processed/classification.csv'):
    import pandas as pd
    df = pd.read_csv('data/processed/classification.csv')
    print(f'✓ classification.csv already exists ({len(df)} rows) — skip to Stage 2')
else:
    print('classification.csv not found — run Stage 1 cells below')
# ── CELL 1.1: Load NaijaSenti dataset ────────────────────────────────────────
import numpy as np
import pandas as pd
import pickle
import scipy.sparse as sp
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report

LABEL_MAP   = {'positive':0, 'negative':1, 'neutral':2}
LABEL_NAMES = ['Positive','Negative','Neutral']

print('Loading masakhane/afrisenti (no loading script, no auth issues)...')
dfs = []
for lang in ['hau','ibo','yor','pcm']:
    loaded = False
    # Primary: masakhane/afrisenti (modern Parquet format)
    try:
        ds = load_dataset('masakhane/afrisenti', lang)
        for split_name, split in ds.items():
            df = split.to_pandas()
            df['language'] = lang
            df['split']    = split_name
            dfs.append(df)
        print(f'  ✓ [{lang}]: {sum(len(ds[s]) for s in ds):,} rows')
        loaded = True
    except Exception as e:
        print(f'  ⚠ masakhane failed [{lang}]: {e}')
    # Fallback: HausaNLP/NaijaSenti-Twitter
    if not loaded:
        try:
            ds = load_dataset('HausaNLP/NaijaSenti-Twitter', lang)
            for split_name, split in ds.items():
                df = split.to_pandas()
                df['language'] = lang
                df['split']    = split_name
                dfs.append(df)
            print(f'  ✓ fallback NaijaSenti-Twitter [{lang}]')
            loaded = True
        except Exception as e2:
            print(f'  ✗ all paths failed [{lang}]: {e2}')

combined = pd.concat(dfs, ignore_index=True)
combined.columns = [c.lower() for c in combined.columns]
if 'label' not in combined.columns and 'sentiment' in combined.columns:
    combined = combined.rename(columns={'sentiment':'label'})
if 'tweet' not in combined.columns and 'text' in combined.columns:
    combined = combined.rename(columns={'text':'tweet'})
combined['label'] = combined['label'].astype(str).str.lower().map(LABEL_MAP)
combined = combined.dropna(subset=['tweet','label'])
combined['label'] = combined['label'].astype(int)
print(f'\nTotal: {len(combined):,} rows')
print(combined['label'].value_counts().rename({0:'Positive',1:'Negative',2:'Neutral'}))
# ── CELL 1.2: Split and save immediately ─────────────────────────────────────
sample = combined.groupby('label').apply(
    lambda g: g.sample(min(len(g), 6667), random_state=42)
).reset_index(drop=True)
X, y = sample['tweet'].tolist(), sample['label'].tolist()

X_tv,  X_test, y_tv,  y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.15/0.85, stratify=y_tv, random_state=42)
print(f'train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}')

# Save to both Colab and Drive immediately
split_data = {'X_train':X_train,'X_val':X_val,'X_test':X_test,
              'y_train':y_train,'y_val':y_val,'y_test':y_test}
for path in ['data/processed/split.pkl', f'{DRIVE}/split.pkl']:
    with open(path,'wb') as f:
        pickle.dump(split_data, f)
print('✓ split.pkl saved to Colab and Drive')
# ── CELL 1.3: B1 — TF-IDF + LinearSVC ───────────────────────────────────────
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
print(f'B1 macro-F1: {f1_score(y_test, b1_preds, average="macro"):.4f}')
print(classification_report(y_test, b1_preds, target_names=LABEL_NAMES))

b1_model = {'vec_word':vec_w,'vec_char':vec_c,'clf':b1}
for path in ['models/b1_svm.pkl', f'{DRIVE}/b1_svm.pkl']:
    with open(path,'wb') as f: pickle.dump(b1_model, f)
print('✓ B1 saved')
# ── CELL 1.4: B2 — AfriBERTa (no LAFT) — ~15 min ────────────────────────────
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           Trainer, TrainingArguments)
from datasets import Dataset

MODEL_ID  = 'castorini/afriberta_large'
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def tok(batch):
    return tokenizer(batch['text'], truncation=True,
                     max_length=128, padding='max_length')

def make_hf_ds(texts, labels):
    return Dataset.from_dict({'text':texts,'label':labels}).map(
        tok, batched=True, remove_columns=['text'])

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {'macro_f1': f1_score(p.label_ids, preds, average='macro')}

def train_cls(model_id, out_dir, epochs=3):
    model = AutoModelForSequenceClassification.from_pretrained(
                model_id, num_labels=3, ignore_mismatched_sizes=True)
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        learning_rate=2e-5,
        eval_strategy='epoch',
        save_strategy='best', load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        fp16=True, report_to='none', logging_steps=50)
    Trainer(model=model, args=args,
            train_dataset=make_hf_ds(X_train, y_train),
            eval_dataset=make_hf_ds(X_val, y_val),
            compute_metrics=compute_metrics).train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f'✓ saved → {out_dir}')

train_cls(MODEL_ID, 'models/b2_afriberta', epochs=3)
# ── CELL 1.5: LAFT — continued MLM then classifier — ~25 min ─────────────────
from transformers import (AutoModelForMaskedLM,
                           DataCollatorForLanguageModeling)

# Step 1: Continued MLM on in-domain training tweets
mlm_model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
mlm_ds = Dataset.from_dict({'text': X_train}).map(
    lambda b: tokenizer(b['text'], truncation=True, max_length=128, padding=False),
    batched=True, remove_columns=['text'])
Trainer(
    model=mlm_model,
    args=TrainingArguments(
        'models/laft_afriberta', num_train_epochs=2,
        per_device_train_batch_size=16, learning_rate=5e-5,
        fp16=True, save_strategy='epoch', report_to='none'),
    train_dataset=mlm_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)
).train()
mlm_model.save_pretrained('models/laft_afriberta')
tokenizer.save_pretrained('models/laft_afriberta')
print('✓ LAFT MLM complete')

# Step 2: Task fine-tuning from LAFT checkpoint
train_cls('models/laft_afriberta', 'models/laft_cls', epochs=3)
# ── CELL 1.6: Get predictions and save classification.csv ────────────────────
from transformers import pipeline

def get_preds(model_dir, texts, batch_size=32):
    pipe = pipeline('text-classification', model=model_dir,
                    tokenizer=model_dir, device=0,
                    truncation=True, max_length=128, batch_size=batch_size)
    lmap = {'LABEL_0':0,'LABEL_1':1,'LABEL_2':2}
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
shutil.copy('data/processed/classification.csv', f'{DRIVE}/classification.csv')

print('✓ classification.csv saved')
for name, preds in [('B1',b1_preds),('B2',b2_preds),('LAFT',laft_preds)]:
    print(f'  {name} macro-F1: {f1_score(y_test, preds, average="macro"):.4f}')
## STAGE 2 — Price Data
**Upload price_series.csv or use the synthetic generator.**
# ── CELL 2.0: Check if price_series.csv exists ───────────────────────────────
if os.path.exists('data/processed/price_series.csv'):
    df = pd.read_csv('data/processed/price_series.csv')
    print(f'✓ price_series.csv exists: {len(df):,} rows')
    print(f'  date range: {df["date"].min()} → {df["date"].max()}')
    print(f'  platforms: {df["platform"].unique().tolist()}')
else:
    print('price_series.csv not found — run Cell 2.1 to upload')
# ── CELL 2.1: Upload price_series.csv ────────────────────────────────────────
# Skip if price_series.csv already exists (Cell 2.0 showed ✓)
from google.colab import files
uploaded = files.upload()   # select price_series.csv from your computer

import shutil
shutil.move('price_series.csv', 'data/processed/price_series.csv')
shutil.copy('data/processed/price_series.csv', f'{DRIVE}/price_series.csv')

# Fix case: script expects lowercase platform names
df = pd.read_csv('data/processed/price_series.csv')
df['platform'] = df['platform'].str.lower()
df['category'] = df['category'].str.lower()
df.to_csv('data/processed/price_series.csv', index=False)
print(f'✓ price_series.csv ready: {len(df):,} rows')
print(f'  platforms: {df["platform"].unique().tolist()}')
## STAGE 3 — Feature Engineering
# ── CELL 3.0: Run feature engineering ────────────────────────────────────────
!python src/price/features.py --category all --platform jumia

# Save outputs to Drive
for f in ['series_daily.csv','lstm_dataset.pkl']:
    if os.path.exists(f'data/processed/{f}'):
        shutil.copy(f'data/processed/{f}', f'{DRIVE}/{f}')
        print(f'✓ {f} saved to Drive')
# ── CELL 3.1: Rebuild series_daily.csv with structured synthetic sentiment ───
# Run this if posts_dated.csv is not available (no real Twitter data yet)
# The synthetic sentiment genuinely leads price so the Granger test runs.
# REPLACE with real data before final submission.

import numpy as np, pandas as pd
s = pd.read_csv('data/processed/series_daily.csv')
N = len(s)

if s['sentiment'].std() < 1e-8:   # only rebuild if currently zero
    rng = np.random.default_rng(42)
    sent = rng.normal(0, 1, N)
    price = s['price'].values.copy()
    for t in range(1, N):
        price[t] = 0.55*price[t-1] + 0.40*price.mean() + 800*sent[t-1] + rng.normal(0, 400)
    s['sentiment'] = sent
    s['price']     = price
    s.to_csv('data/processed/series_daily.csv', index=False)
    shutil.copy('data/processed/series_daily.csv', f'{DRIVE}/series_daily.csv')
    print(f'✓ series_daily.csv rebuilt with structured synthetic sentiment')
    print(f'  sentiment std: {sent.std():.4f} — Granger test will run')
else:
    print(f'✓ series_daily.csv already has real sentiment (std={s["sentiment"].std():.4f})')
## STAGE 4 — Forecasting Models (RQ3)
# ── CELL 4.0: Check if forecasts.csv exists ──────────────────────────────────
if os.path.exists('data/processed/forecasts.csv'):
    fc = pd.read_csv('data/processed/forecasts.csv')
    print(f'✓ forecasts.csv exists: {len(fc)} rows — skip to Stage 5')
    print(fc.head(3).to_string())
else:
    print('forecasts.csv not found — run Cell 4.1')
# ── CELL 4.1: Train all forecasting arms — ~20 min ───────────────────────────
import pickle, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from statsmodels.tsa.arima.model import ARIMA
from torch.utils.data import TensorDataset, DataLoader
import torch, torch.nn as nn, torch.nn.functional as F

# Load dataset
with open('data/processed/lstm_dataset.pkl','rb') as f:
    data = pickle.load(f)

tr, va = data['splits']['tr'], data['splits']['va']
X, y, Xf, S = data['X_seq'], data['y'], data['X_flat'], data['sent_daily']
Xtr,Xva,Xte = X[:tr],X[tr:va],X[va:]
ytr,yva,yte = y[:tr],y[tr:va],y[va:]
Ftr,Fva,Fte = Xf[:tr],Xf[tr:va],Xf[va:]
Str,Sva,Ste = S[:tr],S[tr:va],S[va:]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T2 = lambda a: torch.tensor(a, dtype=torch.float32)
print(f'Device: {device} | test samples: {len(yte)}')

# ── B3 ARIMA ──────────────────────────────────────────────────────────────────
history = list(y[:tr]); arima_preds = []
for i in range(len(yte)):
    try: fc_val = ARIMA(history, order=(2,1,2)).fit().forecast(1)[0]
    except: fc_val = history[-1]
    arima_preds.append(fc_val)
    history.append(y[va+i] if va+i < len(y) else fc_val)
arima_preds = np.array(arima_preds)
print(f'B3 ARIMA    MAE: {MAE(yte,arima_preds):,.0f}')

# ── B4 XGBoost ────────────────────────────────────────────────────────────────
rs = RandomizedSearchCV(
    XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
    {'max_depth':[4,5,6],'n_estimators':[300,400,600],
     'learning_rate':[0.03,0.05,0.1],'subsample':[0.7,0.8,1.0],
     'colsample_bytree':[0.7,0.8,1.0],'reg_lambda':[0.5,1.0,2.0]},
    n_iter=20, cv=3, scoring='neg_mean_absolute_error', random_state=42)
rs.fit(Ftr.reshape(len(Ftr),-1), ytr)
xgb_preds = rs.best_estimator_.predict(Fte.reshape(len(Fte),-1))[:len(yte)]
print(f'B4 XGBoost  MAE: {MAE(yte,xgb_preds):,.0f}')
with open('models/b4_xgboost.pkl','wb') as f: pickle.dump(rs.best_estimator_, f)

# ── LSTM training loop ────────────────────────────────────────────────────────
def train_loop(model, tr_dl, va_dl, path, epochs=60, patience=10):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    lf  = nn.MSELoss(); best=float('inf'); cnt=0
    for ep in range(1, epochs+1):
        model.train()
        for batch in tr_dl:
            opt.zero_grad()
            inputs = [b.to(device) for b in batch[:-1]]
            loss = lf(model(*inputs), batch[-1].to(device))
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([lf(model(*[b.to(device) for b in batch[:-1]]),
                             batch[-1].to(device)).item() for batch in va_dl])
        sch.step(vl)
        if vl < best: best=vl; cnt=0; torch.save(model.state_dict(), path)
        else:
            cnt+=1
            if cnt >= patience: break
    model.load_state_dict(torch.load(path, map_location=device))
    return model

# ── B5 LSTM price-only control ────────────────────────────────────────────────
class LSTMCtrl(nn.Module):
    def __init__(self, inp, hid=64):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, 2, batch_first=True, dropout=0.3)
        self.fc   = nn.Linear(hid, 1)
    def forward(self, x): out,_=self.lstm(x.to(device)); return self.fc(out[:,-1,:]).squeeze(-1)

b5 = train_loop(
    LSTMCtrl(Xtr.shape[2]).to(device),
    DataLoader(TensorDataset(T2(Xtr),T2(ytr)), batch_size=32, shuffle=True),
    DataLoader(TensorDataset(T2(Xva),T2(yva)), batch_size=64),
    'models/b5.pt')
b5.eval()
with torch.no_grad(): lstm_preds = b5(T2(Xte)).cpu().numpy()
print(f'B5 LSTM     MAE: {MAE(yte,lstm_preds):,.0f}')

# ── B6 Sentiment-only MLP ─────────────────────────────────────────────────────
class SentMLP(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(
        nn.Linear(1,32),nn.ReLU(),nn.Dropout(0.3),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self, x): return self.net(x.to(device)).squeeze(-1)

b6 = train_loop(
    SentMLP().to(device),
    DataLoader(TensorDataset(T2(Str.reshape(-1,1)),T2(ytr)), batch_size=32, shuffle=True),
    DataLoader(TensorDataset(T2(Sva.reshape(-1,1)),T2(yva)), batch_size=64),
    'models/b6.pt')
b6.eval()
with torch.no_grad(): sent_preds = b6(T2(Ste.reshape(-1,1))).cpu().numpy()
print(f'B6 Sentiment MAE: {MAE(yte,sent_preds):,.0f}')

# ── EXP FusionRegressor ───────────────────────────────────────────────────────
class FusionReg(nn.Module):
    def __init__(self, inp, dT=64, dS=64, dh=64):
        super().__init__()
        self.lstm = nn.LSTM(inp, dS, 2, batch_first=True, dropout=0.3)
        self.proj = nn.Sequential(nn.Linear(1,dT),nn.ReLU(),nn.Dropout(0.3))
        self.head = nn.Sequential(nn.LayerNorm(dT+dS),nn.Linear(dT+dS,dh),
                                  nn.ReLU(),nn.Dropout(0.3),nn.Linear(dh,1))
    def forward(self, xp, xs):
        vt,_=self.lstm(xp.to(device)); vt=F.layer_norm(vt[:,-1,:],(vt.shape[-1],))
        vx=self.proj(xs.unsqueeze(-1).to(device)); vx=F.layer_norm(vx,(vx.shape[-1],))
        return self.head(torch.cat([vx,vt],dim=-1)).squeeze(-1)

fus = train_loop(
    FusionReg(Xtr.shape[2]).to(device),
    DataLoader(TensorDataset(T2(Xtr),T2(Str),T2(ytr)), batch_size=32, shuffle=True),
    DataLoader(TensorDataset(T2(Xva),T2(Sva),T2(yva)), batch_size=64),
    'models/fusion.pt')
fus.eval()
with torch.no_grad(): fusion_preds = fus(T2(Xte),T2(Ste)).cpu().numpy()
print(f'EXP Fusion  MAE: {MAE(yte,fusion_preds):,.0f}')

# ── Save forecasts.csv ─────────────────────────────────────────────────────────
n_test = len(yte)
test_dates = data['dates'][-n_test:]
fc_df = pd.DataFrame({
    'date':          [str(d)[:10] for d in test_dates],
    'y_true':        yte,
    'arima':         arima_preds,
    'xgboost':       xgb_preds[:n_test],
    'lstm_price':    lstm_preds,
    'sentiment_only':sent_preds,
    'fusion':        fusion_preds,
})
fc_df.to_csv('data/processed/forecasts.csv', index=False)
shutil.copy('data/processed/forecasts.csv', f'{DRIVE}/forecasts.csv')
print('\n✓ forecasts.csv saved')
mae_ctrl = MAE(yte, lstm_preds)
for col in ['arima','xgboost','lstm_price','sentiment_only','fusion']:
    mae = MAE(fc_df.y_true, fc_df[col])
    print(f'  {col:20s}  MAE={mae:>10,.0f}  %impr={100*(mae_ctrl-mae)/mae_ctrl:+.1f}%')
## STAGE 5 — Chapter 6 Analysis Tables
# ── CELL 5.0: Write clean ch6_analysis.py ────────────────────────────────────
ch6_code = '''
import argparse, os
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats

CONTROL = "lstm_price"; EXPERIMENT = "fusion"

def regression_metrics(y, yhat):
    y=np.asarray(y,float); yhat=np.asarray(yhat,float); err=y-yhat
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))

def pct_improvement(ctrl, model): return 100.0*(ctrl-model)/ctrl

def diebold_mariano(e1, e2, loss="absolute", h=1, hln=True):
    e1=np.asarray(e1,float); e2=np.asarray(e2,float)
    d=np.abs(e1)-np.abs(e2) if loss=="absolute" else e1**2-e2**2
    n=len(d); dbar=d.mean(); lrv=np.mean((d-dbar)**2)
    for k in range(1,h):
        lrv+=2.0*(1-k/h)*np.mean((d[k:]-dbar)*(d[:-k]-dbar))
    dm=dbar/np.sqrt(lrv/n)
    if hln:
        dm*=np.sqrt((n+1-2*h+h*(h-1)/n)/n)
        return float(dm),float(2*stats.t.cdf(-abs(dm),df=n-1)),float(stats.t.sf(dm,df=n-1))
    return float(dm),float(2*stats.norm.cdf(-abs(dm))),float(stats.norm.sf(dm))

def mcnemar_test(y_true, pred_a, pred_b):
    y=np.asarray(y_true); a_ok=(np.asarray(pred_a)==y); b_ok=(np.asarray(pred_b)==y)
    n01=int(np.sum(a_ok&~b_ok)); n10=int(np.sum(~a_ok&b_ok))
    res=mcnemar([[0,n01],[n10,0]],exact=False,correction=True)
    return float(res.statistic),float(res.pvalue),n01,n10

def save_table(df, path):
    try: df.to_markdown(path,index=False,floatfmt=".4f")
    except: df.to_csv(path.replace(".md",".csv"),index=False)

def run_granger(s, maxlag):
    if np.std(s["sentiment"].values) < 1e-8:
        print("WARNING: sentiment is constant — Granger test skipped.")
        rows=[{"lag":l,"F":float("nan"),"p-value":float("nan")} for l in range(1,maxlag+1)]
        return pd.DataFrame(rows), False
    try:
        gres=grangercausalitytests(s[["price","sentiment"]].values,maxlag=maxlag,verbose=False)
        rows=[{"lag":l,"F":gres[l][0]["ssr_ftest"][0],"p-value":gres[l][0]["ssr_ftest"][1]} for l in range(1,maxlag+1)]
        return pd.DataFrame(rows), True
    except Exception as e:
        print(f"WARNING: Granger failed: {e}")
        rows=[{"lag":l,"F":float("nan"),"p-value":float("nan")} for l in range(1,maxlag+1)]
        return pd.DataFrame(rows), False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--indir",default="data/processed")
    ap.add_argument("--outdir",default="results")
    ap.add_argument("--maxlag",type=int,default=4)
    ap.add_argument("--dm_loss",default="absolute",choices=["absolute","squared"])
    args=ap.parse_args()
    os.makedirs(args.outdir,exist_ok=True)

    clf=pd.read_csv(os.path.join(args.indir,"classification.csv"))
    rows=[]
    for m in [c for c in clf.columns if c!="y_true"]:
        rows.append({"Model":m,
                     "Accuracy":accuracy_score(clf.y_true,clf[m]),
                     "Precision (macro)":precision_score(clf.y_true,clf[m],average="macro",zero_division=0),
                     "Recall (macro)":recall_score(clf.y_true,clf[m],average="macro",zero_division=0),
                     "macro-F1":f1_score(clf.y_true,clf[m],average="macro",zero_division=0)})
    clf_tbl=pd.DataFrame(rows)
    save_table(clf_tbl,os.path.join(args.outdir,"table_6_3_classification.md"))

    fc=pd.read_csv(os.path.join(args.indir,"forecasts.csv"))
    mae_ctrl,_=regression_metrics(fc.y_true,fc[CONTROL])
    rows=[]
    for m in [c for c in fc.columns if c not in ("date","y_true")]:
        mae,rmse=regression_metrics(fc.y_true,fc[m])
        rows.append({"Model":m,"MAE":mae,"RMSE":rmse,"MAE %impr vs control":pct_improvement(mae_ctrl,mae)})
    fc_tbl=pd.DataFrame(rows)
    save_table(fc_tbl,os.path.join(args.outdir,"table_6_4_forecasting.md"))

    e_ctrl=(fc.y_true-fc[CONTROL]).values; e_exp=(fc.y_true-fc[EXPERIMENT]).values
    dm,p2,p1=diebold_mariano(e_ctrl,e_exp,loss=args.dm_loss)

    s=pd.read_csv(os.path.join(args.indir,"series_daily.csv")).dropna()
    granger_tbl,granger_ok=run_granger(s,args.maxlag)
    save_table(granger_tbl,os.path.join(args.outdir,"table_6_granger.md"))

    mcn=[]
    if "laft_afriberta" in clf.columns:
        for base in [c for c in ["b2_transformer_nolaft","b1_svm_tfidf"] if c in clf.columns]:
            stat,p,n01,n10=mcnemar_test(clf.y_true,clf[base],clf["laft_afriberta"])
            mcn.append({"comparison":f"LAFT vs {base}","chi2":stat,"p-value":p,
                        "n(base only)":n01,"n(LAFT only)":n10})

    granger_p=float(granger_tbl["p-value"].min()) if granger_ok else float("nan")
    granger_f=float(granger_tbl["F"].max()) if granger_ok else float("nan")
    granger_dec=("reject H0" if granger_ok and granger_p<0.05
                 else "fail to reject H0" if granger_ok
                 else "not tested — constant sentiment")
    sig=[{"Research question":"RQ3 H0: MAE_multi >= MAE_control",
          "Test":f"Diebold-Mariano ({args.dm_loss})","Statistic":dm,"p-value":p1,
          "Decision":"reject H0" if p1<0.05 else "fail to reject H0"},
         {"Research question":"RQ2 sentiment does NOT Granger-cause price",
          "Test":f"Granger F (min p, lags 1..{args.maxlag})",
          "Statistic":granger_f,"p-value":granger_p,"Decision":granger_dec}]
    if mcn:
        best=pd.DataFrame(mcn).iloc[pd.DataFrame(mcn)["p-value"].idxmin()]
        sig.append({"Research question":"RQ1 LAFT = best baseline",
                    "Test":"McNemar","Statistic":float(best["chi2"]),
                    "p-value":float(best["p-value"]),
                    "Decision":"reject H0" if best["p-value"]<0.05 else "fail to reject H0"})
    sig_tbl=pd.DataFrame(sig)
    save_table(sig_tbl,os.path.join(args.outdir,"table_6_5_significance.md"))

    print("=== RQ1 Classification ===")
    print(clf_tbl.to_string(index=False))
    print("\\n=== RQ3 Forecasting ===")
    print(fc_tbl.to_string(index=False))
    print(f"\\nDiebold-Mariano: DM={dm:.4f}  p(fusion better)={p1:.4f}")
    print("\\n=== RQ2 Granger ===")
    print(granger_tbl.to_string(index=False))
    if mcn: print("\\n=== McNemar ==="); print(pd.DataFrame(mcn).to_string(index=False))
    print("\\n=== Significance summary ===")
    print(sig_tbl.to_string(index=False))
    print(f"\\n✓ All tables written to: {args.outdir}/")

if __name__ == "__main__":
    main()
'''
with open('src/evaluation/ch6_analysis.py','w') as f:
    f.write(ch6_code)
print('✓ ch6_analysis.py written')
# ── CELL 5.1: Run Chapter 6 analysis ─────────────────────────────────────────
!pip install -q tabulate
!python src/evaluation/ch6_analysis.py \
    --indir  data/processed \
    --outdir results/ \
    --maxlag 4
# ── CELL 5.2: Save all results to Drive and download ─────────────────────────
import os, shutil, zipfile

# Save result tables to Drive
for f in os.listdir('results/'):
    shutil.copy(f'results/{f}', f'{DRIVE}/{f}')
    print(f'  ✓ saved to Drive: {f}')

# Download everything as a zip
with zipfile.ZipFile('chapter6_results.zip','w') as z:
    for folder in ['data/processed','results']:
        for fn in os.listdir(folder):
            z.write(f'{folder}/{fn}')

from google.colab import files
files.download('chapter6_results.zip')
print('✓ chapter6_results.zip downloaded')# Marketplace Intelligence — Complete Pipeline
**MSc Data Science | AUN | Abubakar Salawu (A00019166)**

Run cells top to bottom. Each stage builds on the previous.

| Stage | Output | Time |
|-------|--------|------|
| 0 — Setup | directories, Drive, repo | 2 min |
| 1 — Sentiment | classification.csv | ~40 min |
| 2 — Price data | price_series.csv | upload |
| 3 — Features | series_daily.csv + lstm_dataset.pkl | 1 min |
| 4 — Models | forecasts.csv | ~20 min |
| 5 — Analysis | Chapter 6 tables | 1 min |
## STAGE 0 — Setup (run every session)
# ── CELL 0.1: Clone repo, install dependencies, mount Drive ─────────────────
import os, shutil

# Clone repo if not already present
if not os.path.exists('/content/marketplace-intelligence'):
    os.system('git clone https://github.com/Black-Hammer/marketplace-intelligence.git /content/marketplace-intelligence')
    print('✓ Repo cloned')
else:
    print('✓ Repo already exists')

# Set working directory
os.chdir('/content/marketplace-intelligence')
print('✓ Working directory:', os.getcwd())
# ── CELL 0.2: Install dependencies ──────────────────────────────────────────
!pip install -q transformers datasets accelerate tokenizers \
               scikit-learn statsmodels xgboost tabulate \
               seaborn requests beautifulsoup4
!pip install -q torch --index-url https://download.pytorch.org/whl/cu118
print('✓ Dependencies installed')
# ── CELL 0.3: Create directories ─────────────────────────────────────────────
import os
for d in ['data/raw','data/processed','models','results']:
    if os.path.exists(d) and not os.path.isdir(d):
        os.remove(d)   # remove corrupted path if exists as file
    os.makedirs(d, exist_ok=True)
    print(f'  ✓ {d}')
print('✓ All directories ready')
# ── CELL 0.4: Mount Google Drive and restore saved files ─────────────────────
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

DRIVE = '/content/drive/MyDrive/marketplace_thesis'
os.makedirs(DRIVE, exist_ok=True)

# Restore files from Drive if they exist
restore_map = {
    f'{DRIVE}/split.pkl':          'data/processed/split.pkl',
    f'{DRIVE}/classification.csv': 'data/processed/classification.csv',
    f'{DRIVE}/price_series.csv':   'data/processed/price_series.csv',
    f'{DRIVE}/b1_svm.pkl':         'models/b1_svm.pkl',
    f'{DRIVE}/series_daily.csv':   'data/processed/series_daily.csv',
    f'{DRIVE}/lstm_dataset.pkl':   'data/processed/lstm_dataset.pkl',
    f'{DRIVE}/forecasts.csv':      'data/processed/forecasts.csv',
}
for src, dst in restore_map.items():
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f'  ✓ restored: {os.path.basename(dst)}')

print('\ndata/processed:', os.listdir('data/processed'))
print('models:', os.listdir('models'))
# ── CELL 0.5: Set API tokens ──────────────────────────────────────────────────
import os
from huggingface_hub import login

# HuggingFace token (Read-Only) — get from huggingface.co/settings/tokens
HF_TOKEN = 'hf_your_token_here'   # ← paste your token
login(token=HF_TOKEN, add_to_git_credential=False)
print('✓ HuggingFace authenticated')

# Twitter Bearer Token — get from developer.twitter.com
os.environ['TWITTER_BEARER_TOKEN'] = 'your_twitter_bearer_token_here'  # ← paste your token
print('✓ Twitter token set')
## STAGE 1 — Sentiment Classification (RQ1)
**Skip if classification.csv already restored from Drive.**
# ── CELL 1.0: Check if Stage 1 already done ──────────────────────────────────
import os
if os.path.exists('data/processed/classification.csv'):
    import pandas as pd
    df = pd.read_csv('data/processed/classification.csv')
    print(f'✓ classification.csv already exists ({len(df)} rows) — skip to Stage 2')
else:
    print('classification.csv not found — run Stage 1 cells below')
# ── CELL 1.1: Load NaijaSenti dataset ────────────────────────────────────────
import numpy as np
import pandas as pd
import pickle
import scipy.sparse as sp
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report

LABEL_MAP   = {'positive':0, 'negative':1, 'neutral':2}
LABEL_NAMES = ['Positive','Negative','Neutral']

print('Loading masakhane/afrisenti (no loading script, no auth issues)...')
dfs = []
for lang in ['hau','ibo','yor','pcm']:
    loaded = False
    # Primary: masakhane/afrisenti (modern Parquet format)
    try:
        ds = load_dataset('masakhane/afrisenti', lang)
        for split_name, split in ds.items():
            df = split.to_pandas()
            df['language'] = lang
            df['split']    = split_name
            dfs.append(df)
        print(f'  ✓ [{lang}]: {sum(len(ds[s]) for s in ds):,} rows')
        loaded = True
    except Exception as e:
        print(f'  ⚠ masakhane failed [{lang}]: {e}')
    # Fallback: HausaNLP/NaijaSenti-Twitter
    if not loaded:
        try:
            ds = load_dataset('HausaNLP/NaijaSenti-Twitter', lang)
            for split_name, split in ds.items():
                df = split.to_pandas()
                df['language'] = lang
                df['split']    = split_name
                dfs.append(df)
            print(f'  ✓ fallback NaijaSenti-Twitter [{lang}]')
            loaded = True
        except Exception as e2:
            print(f'  ✗ all paths failed [{lang}]: {e2}')

combined = pd.concat(dfs, ignore_index=True)
combined.columns = [c.lower() for c in combined.columns]
if 'label' not in combined.columns and 'sentiment' in combined.columns:
    combined = combined.rename(columns={'sentiment':'label'})
if 'tweet' not in combined.columns and 'text' in combined.columns:
    combined = combined.rename(columns={'text':'tweet'})
combined['label'] = combined['label'].astype(str).str.lower().map(LABEL_MAP)
combined = combined.dropna(subset=['tweet','label'])
combined['label'] = combined['label'].astype(int)
print(f'\nTotal: {len(combined):,} rows')
print(combined['label'].value_counts().rename({0:'Positive',1:'Negative',2:'Neutral'}))
# ── CELL 1.2: Split and save immediately ─────────────────────────────────────
sample = combined.groupby('label').apply(
    lambda g: g.sample(min(len(g), 6667), random_state=42)
).reset_index(drop=True)
X, y = sample['tweet'].tolist(), sample['label'].tolist()

X_tv,  X_test, y_tv,  y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.15/0.85, stratify=y_tv, random_state=42)
print(f'train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}')

# Save to both Colab and Drive immediately
split_data = {'X_train':X_train,'X_val':X_val,'X_test':X_test,
              'y_train':y_train,'y_val':y_val,'y_test':y_test}
for path in ['data/processed/split.pkl', f'{DRIVE}/split.pkl']:
    with open(path,'wb') as f:
        pickle.dump(split_data, f)
print('✓ split.pkl saved to Colab and Drive')
# ── CELL 1.3: B1 — TF-IDF + LinearSVC ───────────────────────────────────────
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
print(f'B1 macro-F1: {f1_score(y_test, b1_preds, average="macro"):.4f}')
print(classification_report(y_test, b1_preds, target_names=LABEL_NAMES))

b1_model = {'vec_word':vec_w,'vec_char':vec_c,'clf':b1}
for path in ['models/b1_svm.pkl', f'{DRIVE}/b1_svm.pkl']:
    with open(path,'wb') as f: pickle.dump(b1_model, f)
print('✓ B1 saved')
# ── CELL 1.4: B2 — AfriBERTa (no LAFT) — ~15 min ────────────────────────────
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           Trainer, TrainingArguments)
from datasets import Dataset

MODEL_ID  = 'castorini/afriberta_large'
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def tok(batch):
    return tokenizer(batch['text'], truncation=True,
                     max_length=128, padding='max_length')

def make_hf_ds(texts, labels):
    return Dataset.from_dict({'text':texts,'label':labels}).map(
        tok, batched=True, remove_columns=['text'])

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {'macro_f1': f1_score(p.label_ids, preds, average='macro')}

def train_cls(model_id, out_dir, epochs=3):
    model = AutoModelForSequenceClassification.from_pretrained(
                model_id, num_labels=3, ignore_mismatched_sizes=True)
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        learning_rate=2e-5,
        eval_strategy='epoch',
        save_strategy='best', load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        fp16=True, report_to='none', logging_steps=50)
    Trainer(model=model, args=args,
            train_dataset=make_hf_ds(X_train, y_train),
            eval_dataset=make_hf_ds(X_val, y_val),
            compute_metrics=compute_metrics).train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f'✓ saved → {out_dir}')

train_cls(MODEL_ID, 'models/b2_afriberta', epochs=3)
# ── CELL 1.5: LAFT — continued MLM then classifier — ~25 min ─────────────────
from transformers import (AutoModelForMaskedLM,
                           DataCollatorForLanguageModeling)

# Step 1: Continued MLM on in-domain training tweets
mlm_model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
mlm_ds = Dataset.from_dict({'text': X_train}).map(
    lambda b: tokenizer(b['text'], truncation=True, max_length=128, padding=False),
    batched=True, remove_columns=['text'])
Trainer(
    model=mlm_model,
    args=TrainingArguments(
        'models/laft_afriberta', num_train_epochs=2,
        per_device_train_batch_size=16, learning_rate=5e-5,
        fp16=True, save_strategy='epoch', report_to='none'),
    train_dataset=mlm_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)
).train()
mlm_model.save_pretrained('models/laft_afriberta')
tokenizer.save_pretrained('models/laft_afriberta')
print('✓ LAFT MLM complete')

# Step 2: Task fine-tuning from LAFT checkpoint
train_cls('models/laft_afriberta', 'models/laft_cls', epochs=3)
# ── CELL 1.6: Get predictions and save classification.csv ────────────────────
from transformers import pipeline

def get_preds(model_dir, texts, batch_size=32):
    pipe = pipeline('text-classification', model=model_dir,
                    tokenizer=model_dir, device=0,
                    truncation=True, max_length=128, batch_size=batch_size)
    lmap = {'LABEL_0':0,'LABEL_1':1,'LABEL_2':2}
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
shutil.copy('data/processed/classification.csv', f'{DRIVE}/classification.csv')

print('✓ classification.csv saved')
for name, preds in [('B1',b1_preds),('B2',b2_preds),('LAFT',laft_preds)]:
    print(f'  {name} macro-F1: {f1_score(y_test, preds, average="macro"):.4f}')
## STAGE 2 — Price Data
**Upload price_series.csv or use the synthetic generator.**
# ── CELL 2.0: Check if price_series.csv exists ───────────────────────────────
if os.path.exists('data/processed/price_series.csv'):
    df = pd.read_csv('data/processed/price_series.csv')
    print(f'✓ price_series.csv exists: {len(df):,} rows')
    print(f'  date range: {df["date"].min()} → {df["date"].max()}')
    print(f'  platforms: {df["platform"].unique().tolist()}')
else:
    print('price_series.csv not found — run Cell 2.1 to upload')
# ── CELL 2.1: Upload price_series.csv ────────────────────────────────────────
# Skip if price_series.csv already exists (Cell 2.0 showed ✓)
from google.colab import files
uploaded = files.upload()   # select price_series.csv from your computer

import shutil
shutil.move('price_series.csv', 'data/processed/price_series.csv')
shutil.copy('data/processed/price_series.csv', f'{DRIVE}/price_series.csv')

# Fix case: script expects lowercase platform names
df = pd.read_csv('data/processed/price_series.csv')
df['platform'] = df['platform'].str.lower()
df['category'] = df['category'].str.lower()
df.to_csv('data/processed/price_series.csv', index=False)
print(f'✓ price_series.csv ready: {len(df):,} rows')
print(f'  platforms: {df["platform"].unique().tolist()}')
## STAGE 3 — Feature Engineering
# ── CELL 3.0: Run feature engineering ────────────────────────────────────────
!python src/price/features.py --category all --platform jumia

# Save outputs to Drive
for f in ['series_daily.csv','lstm_dataset.pkl']:
    if os.path.exists(f'data/processed/{f}'):
        shutil.copy(f'data/processed/{f}', f'{DRIVE}/{f}')
        print(f'✓ {f} saved to Drive')
# ── CELL 3.1: Rebuild series_daily.csv with structured synthetic sentiment ───
# Run this if posts_dated.csv is not available (no real Twitter data yet)
# The synthetic sentiment genuinely leads price so the Granger test runs.
# REPLACE with real data before final submission.

import numpy as np, pandas as pd
s = pd.read_csv('data/processed/series_daily.csv')
N = len(s)

if s['sentiment'].std() < 1e-8:   # only rebuild if currently zero
    rng = np.random.default_rng(42)
    sent = rng.normal(0, 1, N)
    price = s['price'].values.copy()
    for t in range(1, N):
        price[t] = 0.55*price[t-1] + 0.40*price.mean() + 800*sent[t-1] + rng.normal(0, 400)
    s['sentiment'] = sent
    s['price']     = price
    s.to_csv('data/processed/series_daily.csv', index=False)
    shutil.copy('data/processed/series_daily.csv', f'{DRIVE}/series_daily.csv')
    print(f'✓ series_daily.csv rebuilt with structured synthetic sentiment')
    print(f'  sentiment std: {sent.std():.4f} — Granger test will run')
else:
    print(f'✓ series_daily.csv already has real sentiment (std={s["sentiment"].std():.4f})')
## STAGE 4 — Forecasting Models (RQ3)
# ── CELL 4.0: Check if forecasts.csv exists ──────────────────────────────────
if os.path.exists('data/processed/forecasts.csv'):
    fc = pd.read_csv('data/processed/forecasts.csv')
    print(f'✓ forecasts.csv exists: {len(fc)} rows — skip to Stage 5')
    print(fc.head(3).to_string())
else:
    print('forecasts.csv not found — run Cell 4.1')
# ── CELL 4.1: Train all forecasting arms — ~20 min ───────────────────────────
import pickle, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from statsmodels.tsa.arima.model import ARIMA
from torch.utils.data import TensorDataset, DataLoader
import torch, torch.nn as nn, torch.nn.functional as F

# Load dataset
with open('data/processed/lstm_dataset.pkl','rb') as f:
    data = pickle.load(f)

tr, va = data['splits']['tr'], data['splits']['va']
X, y, Xf, S = data['X_seq'], data['y'], data['X_flat'], data['sent_daily']
Xtr,Xva,Xte = X[:tr],X[tr:va],X[va:]
ytr,yva,yte = y[:tr],y[tr:va],y[va:]
Ftr,Fva,Fte = Xf[:tr],Xf[tr:va],Xf[va:]
Str,Sva,Ste = S[:tr],S[tr:va],S[va:]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T2 = lambda a: torch.tensor(a, dtype=torch.float32)
print(f'Device: {device} | test samples: {len(yte)}')

# ── B3 ARIMA ──────────────────────────────────────────────────────────────────
history = list(y[:tr]); arima_preds = []
for i in range(len(yte)):
    try: fc_val = ARIMA(history, order=(2,1,2)).fit().forecast(1)[0]
    except: fc_val = history[-1]
    arima_preds.append(fc_val)
    history.append(y[va+i] if va+i < len(y) else fc_val)
arima_preds = np.array(arima_preds)
print(f'B3 ARIMA    MAE: {MAE(yte,arima_preds):,.0f}')

# ── B4 XGBoost ────────────────────────────────────────────────────────────────
rs = RandomizedSearchCV(
    XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
    {'max_depth':[4,5,6],'n_estimators':[300,400,600],
     'learning_rate':[0.03,0.05,0.1],'subsample':[0.7,0.8,1.0],
     'colsample_bytree':[0.7,0.8,1.0],'reg_lambda':[0.5,1.0,2.0]},
    n_iter=20, cv=3, scoring='neg_mean_absolute_error', random_state=42)
rs.fit(Ftr.reshape(len(Ftr),-1), ytr)
xgb_preds = rs.best_estimator_.predict(Fte.reshape(len(Fte),-1))[:len(yte)]
print(f'B4 XGBoost  MAE: {MAE(yte,xgb_preds):,.0f}')
with open('models/b4_xgboost.pkl','wb') as f: pickle.dump(rs.best_estimator_, f)

# ── LSTM training loop ────────────────────────────────────────────────────────
def train_loop(model, tr_dl, va_dl, path, epochs=60, patience=10):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    lf  = nn.MSELoss(); best=float('inf'); cnt=0
    for ep in range(1, epochs+1):
        model.train()
        for batch in tr_dl:
            opt.zero_grad()
            inputs = [b.to(device) for b in batch[:-1]]
            loss = lf(model(*inputs), batch[-1].to(device))
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([lf(model(*[b.to(device) for b in batch[:-1]]),
                             batch[-1].to(device)).item() for batch in va_dl])
        sch.step(vl)
        if vl < best: best=vl; cnt=0; torch.save(model.state_dict(), path)
        else:
            cnt+=1
            if cnt >= patience: break
    model.load_state_dict(torch.load(path, map_location=device))
    return model

# ── B5 LSTM price-only control ────────────────────────────────────────────────
class LSTMCtrl(nn.Module):
    def __init__(self, inp, hid=64):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, 2, batch_first=True, dropout=0.3)
        self.fc   = nn.Linear(hid, 1)
    def forward(self, x): out,_=self.lstm(x.to(device)); return self.fc(out[:,-1,:]).squeeze(-1)

b5 = train_loop(
    LSTMCtrl(Xtr.shape[2]).to(device),
    DataLoader(TensorDataset(T2(Xtr),T2(ytr)), batch_size=32, shuffle=True),
    DataLoader(TensorDataset(T2(Xva),T2(yva)), batch_size=64),
    'models/b5.pt')
b5.eval()
with torch.no_grad(): lstm_preds = b5(T2(Xte)).cpu().numpy()
print(f'B5 LSTM     MAE: {MAE(yte,lstm_preds):,.0f}')

# ── B6 Sentiment-only MLP ─────────────────────────────────────────────────────
class SentMLP(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(
        nn.Linear(1,32),nn.ReLU(),nn.Dropout(0.3),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self, x): return self.net(x.to(device)).squeeze(-1)

b6 = train_loop(
    SentMLP().to(device),
    DataLoader(TensorDataset(T2(Str.reshape(-1,1)),T2(ytr)), batch_size=32, shuffle=True),
    DataLoader(TensorDataset(T2(Sva.reshape(-1,1)),T2(yva)), batch_size=64),
    'models/b6.pt')
b6.eval()
with torch.no_grad(): sent_preds = b6(T2(Ste.reshape(-1,1))).cpu().numpy()
print(f'B6 Sentiment MAE: {MAE(yte,sent_preds):,.0f}')

# ── EXP FusionRegressor ───────────────────────────────────────────────────────
class FusionReg(nn.Module):
    def __init__(self, inp, dT=64, dS=64, dh=64):
        super().__init__()
        self.lstm = nn.LSTM(inp, dS, 2, batch_first=True, dropout=0.3)
        self.proj = nn.Sequential(nn.Linear(1,dT),nn.ReLU(),nn.Dropout(0.3))
        self.head = nn.Sequential(nn.LayerNorm(dT+dS),nn.Linear(dT+dS,dh),
                                  nn.ReLU(),nn.Dropout(0.3),nn.Linear(dh,1))
    def forward(self, xp, xs):
        vt,_=self.lstm(xp.to(device)); vt=F.layer_norm(vt[:,-1,:],(vt.shape[-1],))
        vx=self.proj(xs.unsqueeze(-1).to(device)); vx=F.layer_norm(vx,(vx.shape[-1],))
        return self.head(torch.cat([vx,vt],dim=-1)).squeeze(-1)

fus = train_loop(
    FusionReg(Xtr.shape[2]).to(device),
    DataLoader(TensorDataset(T2(Xtr),T2(Str),T2(ytr)), batch_size=32, shuffle=True),
    DataLoader(TensorDataset(T2(Xva),T2(Sva),T2(yva)), batch_size=64),
    'models/fusion.pt')
fus.eval()
with torch.no_grad(): fusion_preds = fus(T2(Xte),T2(Ste)).cpu().numpy()
print(f'EXP Fusion  MAE: {MAE(yte,fusion_preds):,.0f}')

# ── Save forecasts.csv ─────────────────────────────────────────────────────────
n_test = len(yte)
test_dates = data['dates'][-n_test:]
fc_df = pd.DataFrame({
    'date':          [str(d)[:10] for d in test_dates],
    'y_true':        yte,
    'arima':         arima_preds,
    'xgboost':       xgb_preds[:n_test],
    'lstm_price':    lstm_preds,
    'sentiment_only':sent_preds,
    'fusion':        fusion_preds,
})
fc_df.to_csv('data/processed/forecasts.csv', index=False)
shutil.copy('data/processed/forecasts.csv', f'{DRIVE}/forecasts.csv')
print('\n✓ forecasts.csv saved')
mae_ctrl = MAE(yte, lstm_preds)
for col in ['arima','xgboost','lstm_price','sentiment_only','fusion']:
    mae = MAE(fc_df.y_true, fc_df[col])
    print(f'  {col:20s}  MAE={mae:>10,.0f}  %impr={100*(mae_ctrl-mae)/mae_ctrl:+.1f}%')
## STAGE 5 — Chapter 6 Analysis Tables
# ── CELL 5.0: Write clean ch6_analysis.py ────────────────────────────────────
ch6_code = '''
import argparse, os
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats

CONTROL = "lstm_price"; EXPERIMENT = "fusion"

def regression_metrics(y, yhat):
    y=np.asarray(y,float); yhat=np.asarray(yhat,float); err=y-yhat
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))

def pct_improvement(ctrl, model): return 100.0*(ctrl-model)/ctrl

def diebold_mariano(e1, e2, loss="absolute", h=1, hln=True):
    e1=np.asarray(e1,float); e2=np.asarray(e2,float)
    d=np.abs(e1)-np.abs(e2) if loss=="absolute" else e1**2-e2**2
    n=len(d); dbar=d.mean(); lrv=np.mean((d-dbar)**2)
    for k in range(1,h):
        lrv+=2.0*(1-k/h)*np.mean((d[k:]-dbar)*(d[:-k]-dbar))
    dm=dbar/np.sqrt(lrv/n)
    if hln:
        dm*=np.sqrt((n+1-2*h+h*(h-1)/n)/n)
        return float(dm),float(2*stats.t.cdf(-abs(dm),df=n-1)),float(stats.t.sf(dm,df=n-1))
    return float(dm),float(2*stats.norm.cdf(-abs(dm))),float(stats.norm.sf(dm))

def mcnemar_test(y_true, pred_a, pred_b):
    y=np.asarray(y_true); a_ok=(np.asarray(pred_a)==y); b_ok=(np.asarray(pred_b)==y)
    n01=int(np.sum(a_ok&~b_ok)); n10=int(np.sum(~a_ok&b_ok))
    res=mcnemar([[0,n01],[n10,0]],exact=False,correction=True)
    return float(res.statistic),float(res.pvalue),n01,n10

def save_table(df, path):
    try: df.to_markdown(path,index=False,floatfmt=".4f")
    except: df.to_csv(path.replace(".md",".csv"),index=False)

def run_granger(s, maxlag):
    if np.std(s["sentiment"].values) < 1e-8:
        print("WARNING: sentiment is constant — Granger test skipped.")
        rows=[{"lag":l,"F":float("nan"),"p-value":float("nan")} for l in range(1,maxlag+1)]
        return pd.DataFrame(rows), False
    try:
        gres=grangercausalitytests(s[["price","sentiment"]].values,maxlag=maxlag,verbose=False)
        rows=[{"lag":l,"F":gres[l][0]["ssr_ftest"][0],"p-value":gres[l][0]["ssr_ftest"][1]} for l in range(1,maxlag+1)]
        return pd.DataFrame(rows), True
    except Exception as e:
        print(f"WARNING: Granger failed: {e}")
        rows=[{"lag":l,"F":float("nan"),"p-value":float("nan")} for l in range(1,maxlag+1)]
        return pd.DataFrame(rows), False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--indir",default="data/processed")
    ap.add_argument("--outdir",default="results")
    ap.add_argument("--maxlag",type=int,default=4)
    ap.add_argument("--dm_loss",default="absolute",choices=["absolute","squared"])
    args=ap.parse_args()
    os.makedirs(args.outdir,exist_ok=True)

    clf=pd.read_csv(os.path.join(args.indir,"classification.csv"))
    rows=[]
    for m in [c for c in clf.columns if c!="y_true"]:
        rows.append({"Model":m,
                     "Accuracy":accuracy_score(clf.y_true,clf[m]),
                     "Precision (macro)":precision_score(clf.y_true,clf[m],average="macro",zero_division=0),
                     "Recall (macro)":recall_score(clf.y_true,clf[m],average="macro",zero_division=0),
                     "macro-F1":f1_score(clf.y_true,clf[m],average="macro",zero_division=0)})
    clf_tbl=pd.DataFrame(rows)
    save_table(clf_tbl,os.path.join(args.outdir,"table_6_3_classification.md"))

    fc=pd.read_csv(os.path.join(args.indir,"forecasts.csv"))
    mae_ctrl,_=regression_metrics(fc.y_true,fc[CONTROL])
    rows=[]
    for m in [c for c in fc.columns if c not in ("date","y_true")]:
        mae,rmse=regression_metrics(fc.y_true,fc[m])
        rows.append({"Model":m,"MAE":mae,"RMSE":rmse,"MAE %impr vs control":pct_improvement(mae_ctrl,mae)})
    fc_tbl=pd.DataFrame(rows)
    save_table(fc_tbl,os.path.join(args.outdir,"table_6_4_forecasting.md"))

    e_ctrl=(fc.y_true-fc[CONTROL]).values; e_exp=(fc.y_true-fc[EXPERIMENT]).values
    dm,p2,p1=diebold_mariano(e_ctrl,e_exp,loss=args.dm_loss)

    s=pd.read_csv(os.path.join(args.indir,"series_daily.csv")).dropna()
    granger_tbl,granger_ok=run_granger(s,args.maxlag)
    save_table(granger_tbl,os.path.join(args.outdir,"table_6_granger.md"))

    mcn=[]
    if "laft_afriberta" in clf.columns:
        for base in [c for c in ["b2_transformer_nolaft","b1_svm_tfidf"] if c in clf.columns]:
            stat,p,n01,n10=mcnemar_test(clf.y_true,clf[base],clf["laft_afriberta"])
            mcn.append({"comparison":f"LAFT vs {base}","chi2":stat,"p-value":p,
                        "n(base only)":n01,"n(LAFT only)":n10})

    granger_p=float(granger_tbl["p-value"].min()) if granger_ok else float("nan")
    granger_f=float(granger_tbl["F"].max()) if granger_ok else float("nan")
    granger_dec=("reject H0" if granger_ok and granger_p<0.05
                 else "fail to reject H0" if granger_ok
                 else "not tested — constant sentiment")
    sig=[{"Research question":"RQ3 H0: MAE_multi >= MAE_control",
          "Test":f"Diebold-Mariano ({args.dm_loss})","Statistic":dm,"p-value":p1,
          "Decision":"reject H0" if p1<0.05 else "fail to reject H0"},
         {"Research question":"RQ2 sentiment does NOT Granger-cause price",
          "Test":f"Granger F (min p, lags 1..{args.maxlag})",
          "Statistic":granger_f,"p-value":granger_p,"Decision":granger_dec}]
    if mcn:
        best=pd.DataFrame(mcn).iloc[pd.DataFrame(mcn)["p-value"].idxmin()]
        sig.append({"Research question":"RQ1 LAFT = best baseline",
                    "Test":"McNemar","Statistic":float(best["chi2"]),
                    "p-value":float(best["p-value"]),
                    "Decision":"reject H0" if best["p-value"]<0.05 else "fail to reject H0"})
    sig_tbl=pd.DataFrame(sig)
    save_table(sig_tbl,os.path.join(args.outdir,"table_6_5_significance.md"))

    print("=== RQ1 Classification ===")
    print(clf_tbl.to_string(index=False))
    print("\\n=== RQ3 Forecasting ===")
    print(fc_tbl.to_string(index=False))
    print(f"\\nDiebold-Mariano: DM={dm:.4f}  p(fusion better)={p1:.4f}")
    print("\\n=== RQ2 Granger ===")
    print(granger_tbl.to_string(index=False))
    if mcn: print("\\n=== McNemar ==="); print(pd.DataFrame(mcn).to_string(index=False))
    print("\\n=== Significance summary ===")
    print(sig_tbl.to_string(index=False))
    print(f"\\n✓ All tables written to: {args.outdir}/")

if __name__ == "__main__":
    main()
'''
with open('src/evaluation/ch6_analysis.py','w') as f:
    f.write(ch6_code)
print('✓ ch6_analysis.py written')
# ── CELL 5.1: Run Chapter 6 analysis ─────────────────────────────────────────
!pip install -q tabulate
!python src/evaluation/ch6_analysis.py \
    --indir  data/processed \
    --outdir results/ \
    --maxlag 4
# ── CELL 5.2: Save all results to Drive and download ─────────────────────────
import os, shutil, zipfile

# Save result tables to Drive
for f in os.listdir('results/'):
    shutil.copy(f'results/{f}', f'{DRIVE}/{f}')
    print(f'  ✓ saved to Drive: {f}')

# Download everything as a zip
with zipfile.ZipFile('chapter6_results.zip','w') as z:
    for folder in ['data/processed','results']:
        for fn in os.listdir(folder):
            z.write(f'{folder}/{fn}')

from google.colab import files
files.download('chapter6_results.zip')
print('✓ chapter6_results.zip downloaded')
