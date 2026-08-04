#!/usr/bin/env python3
"""
src/sentiment/train_sentiment.py
=================================
Stage 1 — Sentiment Classification (answers RQ1)

Trains three sentiment classifiers on NaijaSenti (HuggingFace) and saves
test-set predictions to data/processed/classification.csv, which is one
of the three CSV files required by ch6_analysis.py.

Models trained:
  B1  — TF-IDF (word + char n-gram) + LinearSVC        [classical baseline]
  B2  — AfriBERTa fine-tuned directly (no LAFT)        [transformer baseline]
  EXP — AfriBERTa + Language-Adaptive Fine-Tuning       [proposed model]

Label encoding:
  0 = positive  |  1 = negative  |  2 = neutral

Output files:
  data/processed/classification.csv   (y_true, b1_svm_tfidf,
                                        b2_transformer_nolaft, laft_afriberta)
  models/b1_svm.pkl
  models/b2_afriberta/                (HuggingFace model directory)
  models/laft_afriberta/              (LAFT checkpoint)
  models/laft_cls/                    (LAFT classifier)

Run in Colab (T4 GPU, ~45 min total):
  !python src/sentiment/train_sentiment.py --laft_epochs 2 --task_epochs 3

Run full training:
  !python src/sentiment/train_sentiment.py --laft_epochs 3 --task_epochs 5
"""

import argparse
import logging
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sentiment")

# ── paths ─────────────────────────────────────────────────────────────────────
PROC_DIR  = Path("data/processed")
MODEL_DIR = Path("models")
PROC_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── label map ─────────────────────────────────────────────────────────────────
LABEL_MAP    = {"positive": 0, "negative": 1, "neutral": 2}
LABEL_NAMES  = ["positive", "negative", "neutral"]
MODEL_ID     = "castorini/afriberta_large"   # AfriBERTa base model


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_naijasenti(sample_n: int = 20000) -> pd.DataFrame:
    """
    Load NaijaSenti from HuggingFace (four languages: hau, ibo, yor, pcm).
    Falls back to data/raw/naijasenti.csv if offline.

    NaijaSenti stats:
      hau ~30,000 tweets  |  ibo ~30,000  |  yor ~30,000  |  pcm ~14,000
    """
    try:
        from datasets import load_dataset
        log.info("Loading NaijaSenti from HuggingFace ...")
        dfs = []
        for lang in ["hau", "ibo", "yor", "pcm"]:
            try:
                ds = load_dataset("HausaNLP/NaijaSenti", lang,
                                  trust_remote_code=True)
                for split_name, split in ds.items():
                    df = split.to_pandas()
                    df["language"] = lang
                    df["split"]    = split_name
                    dfs.append(df)
                log.info("  loaded: %s", lang)
            except Exception as e:
                log.warning("  %s failed: %s", lang, e)
        combined = pd.concat(dfs, ignore_index=True)

    except ImportError:
        log.warning("'datasets' not installed — loading from data/raw/naijasenti.csv")
        combined = pd.read_csv("data/raw/naijasenti.csv")

    # normalise column names
    combined.columns = [c.lower() for c in combined.columns]
    if "label" not in combined.columns and "sentiment" in combined.columns:
        combined = combined.rename(columns={"sentiment": "label"})
    if "tweet" not in combined.columns and "text" in combined.columns:
        combined = combined.rename(columns={"text": "tweet"})

    combined["label"] = combined["label"].astype(str).str.lower().map(LABEL_MAP)
    combined = combined.dropna(subset=["tweet", "label"])
    combined["label"] = combined["label"].astype(int)

    # stratified sample for faster Colab runs — remove .sample() for full data
    if sample_n and len(combined) > sample_n:
        combined = combined.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), sample_n // 3), random_state=42))
        log.info("Sampled %d rows (set sample_n=0 for full dataset)", len(combined))
    else:
        log.info("Full dataset: %d rows", len(combined))

    log.info("Label distribution:\n%s", combined["label"].value_counts().to_string())
    return combined


def split_data(df: pd.DataFrame, text_col: str = "tweet"):
    """Stratified chronological 70 / 15 / 15 split."""
    X, y = df[text_col].tolist(), df["label"].tolist()
    X_tv,  X_test, y_tv,  y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, stratify=y_tv, random_state=42)
    log.info("Split — train: %d  val: %d  test: %d",
             len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE B1 — TF-IDF + LinearSVC
# ══════════════════════════════════════════════════════════════════════════════
def train_b1(X_train, y_train, X_val, y_val) -> dict:
    """
    TF-IDF with both word (1-2 gram) and character (3-5 gram) features
    concatenated into one sparse matrix, then LinearSVC with balanced weights.
    Character n-grams help capture Nigerian Pidgin morphology.
    """
    log.info("── Training B1: TF-IDF + LinearSVC ──")

    vec_word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=30_000, sublinear_tf=True,
        min_df=2, strip_accents="unicode"
    )
    vec_char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        max_features=20_000, sublinear_tf=True, min_df=3
    )

    # fit on training data only (no leakage)
    X_tr = sp.hstack([vec_word.fit_transform(X_train),
                      vec_char.fit_transform(X_train)])
    X_va = sp.hstack([vec_word.transform(X_val),
                      vec_char.transform(X_val)])

    clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=3000)
    clf.fit(X_tr, y_train)

    val_f1 = f1_score(y_val, clf.predict(X_va), average="macro")
    log.info("B1 validation macro-F1: %.4f", val_f1)

    model = {"vec_word": vec_word, "vec_char": vec_char, "clf": clf}
    with open(MODEL_DIR / "b1_svm.pkl", "wb") as f:
        pickle.dump(model, f)
    log.info("B1 saved → models/b1_svm.pkl")
    return model


def predict_b1(model: dict, texts: list) -> np.ndarray:
    X = sp.hstack([model["vec_word"].transform(texts),
                   model["vec_char"].transform(texts)])
    return model["clf"].predict(X)


# ══════════════════════════════════════════════════════════════════════════════
# BASELINES B2 and EXP — AfriBERTa (+ LAFT)
# ══════════════════════════════════════════════════════════════════════════════
def _get_tokenizer(model_dir: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir)


def _make_hf_dataset(texts, labels, tokenizer, max_length: int = 128):
    from datasets import Dataset
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True,
                         max_length=max_length, padding="max_length")
    return (Dataset.from_dict({"text": texts, "label": labels})
            .map(tokenize, batched=True, remove_columns=["text"]))


def _compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"macro_f1": f1_score(p.label_ids, preds, average="macro")}


def laft_mlm(X_train: list, out_dir: str = "models/laft_afriberta",
             epochs: int = 2):
    """
    Language-Adaptive Fine-Tuning (LAFT) via continued masked-language
    modelling on the in-domain training tweets.
    This gives AfriBERTa better representations of Nigerian Pidgin and
    code-mixed text before task fine-tuning.
    Requires: transformers, torch, datasets
    """
    log.info("── LAFT: continued MLM on %d in-domain texts ──", len(X_train))
    try:
        from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                                   DataCollatorForLanguageModeling,
                                   Trainer, TrainingArguments)
        from datasets import Dataset
    except ImportError:
        log.error("Install transformers + torch: pip install transformers torch")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForMaskedLM.from_pretrained(MODEL_ID)

    def tok_mlm(batch):
        return tokenizer(batch["text"], truncation=True,
                         max_length=128, padding=False)

    mlm_ds   = (Dataset.from_dict({"text": X_train})
                .map(tok_mlm, batched=True, remove_columns=["text"]))
    collator = DataCollatorForLanguageModeling(tokenizer,
                                              mlm_probability=0.15)
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        learning_rate=5e-5,
        fp16=True,
        save_strategy="epoch",
        report_to="none",
        logging_steps=100,
    )
    Trainer(model=model, args=args, train_dataset=mlm_ds,
            data_collator=collator).train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    log.info("LAFT checkpoint saved → %s", out_dir)


def train_classifier(model_id: str, out_dir: str,
                     X_train, y_train, X_val, y_val,
                     epochs: int = 3):
    """
    Fine-tune AfriBERTa for 3-class sentiment classification.
    model_id  = MODEL_ID        → trains B2 (no LAFT)
    model_id  = laft checkpoint → trains the EXP model
    Hyperparameters match the search space in Table 6.3 (thesis §3.3.2).
    """
    log.info("── Fine-tuning classifier from: %s ──", model_id)
    try:
        from transformers import (AutoModelForSequenceClassification,
                                   AutoTokenizer, Trainer, TrainingArguments)
    except ImportError:
        log.error("Install transformers + torch: pip install transformers torch")
        return None

    tokenizer = _get_tokenizer(model_id)
    model     = AutoModelForSequenceClassification.from_pretrained(
                    model_id, num_labels=3, ignore_mismatched_sizes=True)
    tr_ds = _make_hf_dataset(X_train, y_train, tokenizer)
    va_ds = _make_hf_dataset(X_val,   y_val,   tokenizer)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="best",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=True,
        report_to="none",
        logging_steps=50,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=tr_ds, eval_dataset=va_ds,
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    log.info("Classifier saved → %s", out_dir)
    return trainer


def predict_transformer(model_dir: str, texts: list,
                        batch_size: int = 32) -> np.ndarray:
    """Run inference on a saved HuggingFace classifier."""
    try:
        from transformers import pipeline
        pipe = pipeline(
            "text-classification",
            model=model_dir, tokenizer=model_dir,
            device=0,           # GPU; use device=-1 for CPU
            truncation=True, max_length=128,
            batch_size=batch_size,
        )
        label_map = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}
        return np.array([label_map[r["label"]] for r in pipe(texts)])
    except ImportError:
        log.error("transformers not installed.")
        return np.zeros(len(texts), dtype=int)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(name: str, y_true, y_pred):
    log.info("\n── %s test results ──", name)
    log.info("Accuracy:  %.4f", accuracy_score(y_true, y_pred))
    log.info("Precision: %.4f", precision_score(y_true, y_pred,
                                                average="macro", zero_division=0))
    log.info("Recall:    %.4f", recall_score(y_true, y_pred,
                                             average="macro", zero_division=0))
    log.info("Macro-F1:  %.4f", f1_score(y_true, y_pred,
                                         average="macro", zero_division=0))
    log.info("\n%s", classification_report(y_true, y_pred,
                                          target_names=LABEL_NAMES,
                                          zero_division=0))


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_stage1(sample_n: int   = 20000,
               laft_epochs: int = 2,
               task_epochs: int = 3):
    """
    Full Stage 1 pipeline:
      1. Load NaijaSenti
      2. Split 70/15/15
      3. Train B1 (TF-IDF + SVM)
      4. Train B2 (AfriBERTa, no LAFT)
      5. LAFT then train EXP classifier
      6. Save classification.csv for ch6_analysis.py
    """
    # ── data ──────────────────────────────────────────────────────────────────
    df = load_naijasenti(sample_n)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # ── B1 ────────────────────────────────────────────────────────────────────
    b1_model = train_b1(X_train, y_train, X_val, y_val)
    b1_preds = predict_b1(b1_model, X_test)
    evaluate("B1 — TF-IDF + LinearSVC", y_test, b1_preds)

    # ── B2 — AfriBERTa no LAFT ───────────────────────────────────────────────
    train_classifier(MODEL_ID, "models/b2_afriberta",
                     X_train, y_train, X_val, y_val, task_epochs)
    b2_preds = predict_transformer("models/b2_afriberta", X_test)
    evaluate("B2 — AfriBERTa (no LAFT)", y_test, b2_preds)

    # ── EXP — LAFT → classifier ──────────────────────────────────────────────
    laft_mlm(X_train, "models/laft_afriberta", laft_epochs)
    train_classifier("models/laft_afriberta", "models/laft_cls",
                     X_train, y_train, X_val, y_val, task_epochs)
    laft_preds = predict_transformer("models/laft_cls", X_test)
    evaluate("EXP — AfriBERTa + LAFT", y_test, laft_preds)

    # ── save classification.csv ───────────────────────────────────────────────
    out = PROC_DIR / "classification.csv"
    pd.DataFrame({
        "y_true":                y_test,
        "b1_svm_tfidf":          b1_preds,
        "b2_transformer_nolaft": b2_preds,
        "laft_afriberta":        laft_preds,
    }).to_csv(out, index=False)
    log.info("\n✓ classification.csv saved → %s", out)
    log.info("  %d test samples | columns: y_true, b1_svm_tfidf, "
             "b2_transformer_nolaft, laft_afriberta", len(y_test))


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Stage 1: Train sentiment classifiers (RQ1)")
    ap.add_argument("--sample_n",    type=int, default=20000,
                    help="Rows to sample from NaijaSenti (0 = full dataset)")
    ap.add_argument("--laft_epochs", type=int, default=2,
                    help="Epochs for LAFT continued MLM")
    ap.add_argument("--task_epochs", type=int, default=3,
                    help="Epochs for classifier fine-tuning")
    args = ap.parse_args()
    run_stage1(args.sample_n, args.laft_epochs, args.task_epochs)
