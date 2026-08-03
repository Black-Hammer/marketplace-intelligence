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
