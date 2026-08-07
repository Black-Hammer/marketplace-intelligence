# Results

Generated from the tables in `data/processed`. Every figure here is computed
from the CSVs.

**Test set:** 2,000 held-out items (RQ1) · **Forecast window:** 60 days (RQ3) ·
**Series:** 420 daily observations, 2025-06-13 to 2026-08-06 (RQ2)

---

## RQ1 — Cross-lingual sentiment classification

Does language-adaptive fine-tuning beat classical and non-adapted baselines on
code-mixed Nigerian text?

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| B1 · TF-IDF + LinearSVC | 0.6960 | 0.6963 | 0.6960 |
| B2 · AfriBERTa (no LAFT) | 0.7535 | 0.7534 | 0.7535 |
| LAFT · AfriBERTa + LAFT | 0.7990 | 0.7988 | 0.7991 |

LAFT adds **4.5 macro-F1 points** over the same
architecture without language-adaptive pre-training, and **10.3 points**
over the TF-IDF baseline.

**McNemar tests** (discordant pairs only):

| Comparison | b wins | a wins | χ² | p |
|---|---|---|---|---|
| B1 vs LAFT | 489 | 283 | 54.44 | 1.61e-13 |
| B2 vs LAFT | 399 | 308 | 11.46 | 7.12e-04 |

Both gaps are significant at the 1% level. **H1 supported.**

Data: AfriSenti (`masakhane/afrisenti`) across Hausa, Igbo, Yoruba and Nigerian
Pidgin, with `HausaNLP/NaijaSenti-Twitter` as fallback; classes capped and split
70/15/15. Label encoding is `positive=0, negative=1, neutral=2`.

---

## RQ2 — Is sentiment a leading indicator of price?

> **Provenance warning.** The `sentiment` column in `series_daily.csv` currently
> has mean -0.000 and standard deviation 1.001 across exactly
> 420 observations — the signature of the synthetic placeholder generated in
> Cell 3.1, where price is constructed *from* lagged sentiment. The figures below
> therefore recover the generating equation rather than measure a market
> relationship. **They are pipeline validation, not findings, and must not be
> reported as evidence for H2.** Replace with `posts_dated.csv` — dated consumer
> text scored by the trained classifier and aggregated daily — before drawing any
> conclusion.

Strongest cross-correlation: **lag +1 days, r = -0.843**.

Granger causality on first-differenced series:

| Direction | Lag (days) | F | p |
|---|---|---|---|
| sentiment → price | 1 | 768.872 | 1.66e-96 |
| sentiment → price | 2 | 531.858 | 7.15e-115 |
| sentiment → price | 3 | 388.459 | 2.71e-119 |
| sentiment → price | 4 | 307.846 | 1.76e-121 |
| price → sentiment | 1 | 16.791 | 5.02e-05 |
| price → sentiment | 2 | 6.949 | 1.08e-03 |
| price → sentiment | 3 | 7.757 | 4.75e-05 |
| price → sentiment | 4 | 7.424 | 8.87e-06 |

---

## RQ3 — Does late fusion beat the price-only control?

| Model | MAE | RMSE | MAPE % | Δ vs control % |
|---|---|---|---|---|
| B3 · ARIMA | 1,152.5 | 1,376.5 | 4.153 | -57.8 |
| B4 · XGBoost | 982.6 | 1,256.5 | 3.549 | -34.5 |
| B5 · LSTM (price only) | 730.5 | 924.6 | 2.660 | +0.0 |
| B6 · Sentiment only | 1,771.5 | 2,158.2 | 6.416 | -142.5 |
| EXP · Late fusion | 667.3 | 829.2 | 2.425 | +8.6 |

Late fusion lowers MAE by **8.6%** against the price-only LSTM
control.

**Diebold–Mariano** (absolute-loss differential, HLN-corrected, n = 60):
DM = 0.612, **p = 0.543**.

> The improvement is  at the 5% level. On this
> window the fusion model's lower average error is within what sampling variation
> would produce, so H3 is not supported as it stands. Two things follow: the test
> window of 60 days is short for a DM test, and the sentiment input carries the
> provenance problem described under RQ2. 

---

## Reproducing

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py     # from the repository root
```

The dashboard recomputes every number above from the same CSVs and adds the
confusion matrices, residual plots and lag explorer behind them.
