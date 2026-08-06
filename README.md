# Marketplace Intelligence — Cross-Lingual Sentiment & Price Dynamics

**MSc Data Science and Analytics | American University of Nigeria (AUN)**
**Student:** Abubakar Salawu (A00019166)
**Supervisor:** Dr. Zainab Usman | 

> A multi-modal predictive framework that integrates cross-lingual consumer
> sentiment analysis (AfriBERTa + LAFT) with LSTM-based price dynamics to
> forecast e-commerce price volatility on Nigerian platforms — Jumia, Konga,
> and Temu.

---

## System Architecture

```
Consumer Text (X/Twitter, Nairaland)
        │  Nigerian Pidgin · Hausa · Yoruba · English
        ▼
 ┌──────────────────┐
 │ AfriBERTa + LAFT │──► Sentiment Vector V_text (d_T=64)
 │ Cross-lingual    │              │
 │ Sentiment Model  │              │ leading indicator? (RQ2)
 └──────────────────┘              ▼
                        ┌─────────────────────┐
Marketplace Prices ────►│ LSTM Price Encoder  │──► Temporal Vector V_time (d_S=64)
(Jumia · Konga · Temu)  │ Feature Engineering │
                        └─────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Late-Fusion Decision Layer  │
                    │  [V_text ‖ V_time] → Dense  │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ Price / Volatility Forecast  │
                    │ Marketplace Decision Intel.  │
                    └─────────────────────────────┘
```

---

## Research Questions

| RQ | Hypothesis | Test |
|----|-----------|------|
| **RQ1** | AfriBERTa + LAFT outperforms classical and non-adapted baselines on code-mixed Nigerian sentiment | McNemar test |
| **RQ2** | Consumer sentiment is a *leading indicator* of marketplace price movements | Granger causality |
| **RQ3** | Late-fusion multimodal model yields significantly lower forecasting error than the price-only control | Diebold–Mariano test |

---

## Product Categories

| # | Category | Products | Price Driver |
|---|----------|----------|-------------|
| i | Consumer Electronics | Smartphones, laptops, tablets | Exchange-rate pass-through |
| ii | Generators & Power Equipment | Generators, inverters, solar panels | FX + fuel price |
| iii | FMCG | Detergents, toiletries, personal care | Inflation |
| iv | Packaged Food Staples | Rice, vegetable oil, milk powder | Inflation + seasonality |
| v | Clothing & Footwear | Ready-to-wear, sneakers, sandals | Discretionary spend |

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Black-Hammer/marketplace-intelligence.git
cd marketplace-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard from the repository root
streamlit run dashboard/app.py
```

**Windows, first time on a machine:**

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
cd dashboard
.\setup.ps1     # finds Python, builds the venv, installs, writes the theme
.\run.ps1       # starts the dashboard
```

Launch from the **repository root** so the sidebar's default `data/processed`
resolves. The three result tables are committed, so the dashboard opens on real
results with nothing to configure. If any is missing or malformed it falls back
to clearly labelled demonstration data — those numbers are generated and must
never be reported.

**Run the pipeline in Google Colab (GPU recommended):**
```python
!git clone https://github.com/Black-Hammer/marketplace-intelligence.git
%cd marketplace-intelligence
!pip install -r requirements.txt
# Then open notebooks/marketplace_intelligence_pipeline.ipynb
```

---

## Dashboard

Six tabs mapped to the research questions: **Overview**, **Sentiment · RQ1**,
**Lead & lag · RQ2**, **Forecast · RQ3**, **Early warning**, **Data & export**.

| File | Role |
|------|------|
| `dashboard/app.py` | Interface: layout, tabs, controls |
| `dashboard/core.py` | Loading, schema validation, and every statistic on screen |
| `dashboard/charts.py` | Shared Altair theme and chart builders |
| `dashboard/setup.ps1` · `run.ps1` | Windows setup and launcher |

All three Python files must sit in the same folder — `app.py` imports the other
two from alongside it.

Every figure is computed from the CSVs at render time, so a table in Chapter 6
and the same table in the app cannot drift apart. The Early warning panel labels
live comments with a code-mixed keyword lexicon and states so on screen; wiring
in the trained classifier is described in `dashboard/README.md`.

Reads `classification.csv` (RQ1), `series_daily.csv` (RQ2) and `forecasts.csv`
(RQ3) from `data/processed`, or from files uploaded through the sidebar.

---

## Five-Stage Pipeline

| Stage | Script | Output | Answers |
|-------|--------|--------|---------|
| **1** Sentiment models | `src/sentiment/train_sentiment.py` | `classification.csv` | RQ1 |
| **2** Daily price collection | `src/scraper/daily_run.py` (run daily ≥ 90 days) | `price_series.csv` | — |
| **3** Feature engineering | `src/price/features.py` | `series_daily.csv` + `lstm_dataset.pkl` | RQ2 setup |
| **4** Forecasting models | `src/fusion/train_models.py` | `forecasts.csv` | RQ3 |
| **5** Chapter 6 analysis | `src/evaluation/ch6_analysis.py` | Result tables | All RQs |

---

## Repository Structure

```
marketplace-intelligence/
│
├── src/
│   ├── scraper/
│   │   ├── jumia_scraper.py          # Jumia daily price scraper
│   │   ├── konga_temu_scraper.py     # Konga + Temu price scrapers
│   │   ├── twitter_scraper.py        # Twitter/X consumer post scraper
│   │   ├── nairaland_scraper.py      # Nairaland discussion scraper
│   │   └── daily_run.py              # Master orchestration (run daily)
│   │
│   ├── sentiment/
│   │   └── train_sentiment.py        # B1 / B2 / LAFT-AfriBERTa → classification.csv
│   │
│   ├── price/
│   │   └── features.py               # Feature engineering → series_daily.csv + lstm_dataset.pkl
│   │
│   ├── fusion/
│   │   └── train_models.py           # B3–B6 + EXP FusionRegressor → forecasts.csv
│   │
│   └── evaluation/
│       └── ch6_analysis.py           # DM test · Granger · McNemar · metric tables
│
├── dashboard/
│   ├── app.py                        # Streamlit interface — six tabs, one per RQ
│   ├── core.py                       # Loading, validation, statistics
│   ├── charts.py                     # Altair theme and chart builders
│   ├── setup.ps1 · run.ps1           # Windows setup and launcher
│   └── README.md                     # Dashboard-specific notes
│
├── notebooks/
│   └── marketplace_intelligence_pipeline.ipynb   # Full pipeline (Colab, T4 GPU)
│
├── data/
│   ├── raw/                          # Daily scrape CSVs (gitignored)
│   └── processed/                    # classification.csv · series_daily.csv · forecasts.csv (committed)
│
├── models/                           # Saved model weights (gitignored)
│
├── .github/workflows/
│   └── daily_scrape.yml              # GitHub Actions: automated daily scraping
│
├── requirements.txt
├── GITHUB_SETUP_GUIDE.md             # Step-by-step setup instructions
└── README.md
```

---

## Model Arms

| ID | Model | Type | Role |
|----|-------|------|------|
| B1 | TF-IDF + LinearSVC | Classical | Sentiment baseline |
| B2 | AfriBERTa (no LAFT) | Transformer | Transformer baseline |
| — | AfriBERTa + LAFT | Transformer | **Proposed text model** |
| B3 | ARIMA(2,1,2) | Statistical | Price baseline |
| B4 | XGBoost | Gradient boosting | Non-sequential baseline |
| B5 | LSTM (price-only) | Deep learning | **Control — RQ3** |
| B6 | Sentiment-only MLP | Deep learning | Single-modality baseline |
| EXP | FusionRegressor | Multimodal | **Experimental model — RQ3** |

---

## Data Sources

| Stream | Source | Volume |
|--------|--------|--------|
| Price snapshots | Jumia (primary), Konga (secondary), Temu (robustness) | Daily, ≥ 90 days |
| Consumer text | X/Twitter (API v2 or snscrape) | ≥ 500 posts/category/week |
| Discussion posts | Nairaland (/business, /phones, /technology) | Ongoing |
| Fallback corpus | NaijaSenti + AfriSenti-SemEval 2023 (HuggingFace) | ~104,000 labelled tweets |

---

## CSV Output Schema

**Price scrapers** (`data/raw/jumia_electronics_YYYY-MM-DD.csv`):
```
date | platform | category | product_id | product_name |
current_price_ngn | old_price_ngn | discount_pct | discount_flag |
rating | review_count | verified_seller | query | page | timestamp
```

**Twitter scraper** (`data/raw/twitter_food_YYYY-MM-DD.csv`):
```
date | platform | category | query | text | language |
created_at | likes | retweets | tweet_id | timestamp
```

---

## Running the Daily Scraper

```bash
# Run all scrapers (price + text) for all categories
python src/scraper/daily_run.py

# Price only (if no Twitter token yet)
python src/scraper/daily_run.py --skip_twitter

# Specific platform and category
python src/scraper/jumia_scraper.py --category food --pages 10
python src/scraper/twitter_scraper.py --category electronics --max_results 500
```

---

## Running the Full Pipeline (Colab)

```python
# Stage 1: Sentiment models
!python src/sentiment/train_sentiment.py --laft_epochs 2 --task_epochs 3

# Stage 3: Feature engineering
!python src/price/features.py --category electronics --platform jumia

# Stage 4: All forecasting arms
!python src/fusion/train_models.py --lstm_epochs 60 --fusion_epochs 60

# Stage 5: Chapter 6 analysis tables
!python src/evaluation/ch6_analysis.py --indir data/processed --outdir results/
```

---

## Automated Daily Scraping

GitHub Actions runs `daily_run.py` automatically every day at 09:00 WAT
and commits new data to the repository.

**Setup:**
1. Go to **Settings → Secrets → Actions → New repository secret**
2. Name: `TWITTER_BEARER_TOKEN` | Value: your token from developer.twitter.com
3. Go to **Actions tab → Enable workflows**
4. Test manually: **Actions → Daily Price Scrape → Run workflow**

---

## Green-AI Commitment

The framework is designed for **resource-constrained deployment**:
- AfriBERTa (~126M parameters) fits within a free Colab T4 GPU (16 GB)
- Mixed-precision training (fp16) and gradient accumulation reduce memory footprint
- No frontier-scale model required — accessible to SMEs in Sub-Saharan Africa

---

## Citation

```bibtex
@thesis{salawu2026marketplace,
  author  = {Salawu, Abubakar},
  title   = {A Multi-Modal Predictive Framework for Marketplace Intelligence:
             Integrating Cross-Lingual Sentiment Analysis and Price Dynamics
             in Sub-Saharan African E-commerce},
  school  = {American University of Nigeria},
  year    = {2026},
  type    = {MSc Thesis, Data Science and Analytics}
}
```

---

## Contact

**Abubakar Salawu**
School of Information Technology and Computing (SITC)
American University of Nigeria, Yola
Student ID: A00019166

---

## Licence

MIT © 2026 Abubakar Salawu — **applies to the code in this repository only.**
Scraped marketplace data remains subject to the source platforms' terms of use,
and the NaijaSenti and AfriSenti corpora carry their own licences.
