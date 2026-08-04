# Setup Guide — Marketplace Intelligence

## Requirements

- Python 3.10+
- Google Colab (free T4 GPU) — recommended for model training
- Twitter Developer account (free) — for text collection
- GitHub account (free)

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU training in Colab:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
```

## 2. Set Twitter API Token

```bash
# Linux / Mac
export TWITTER_BEARER_TOKEN=your_token_here

# Windows
set TWITTER_BEARER_TOKEN=your_token_here

# Colab
import os
os.environ["TWITTER_BEARER_TOKEN"] = "your_token_here"
```

Get a free token at: https://developer.twitter.com

## 3. Collect Data (run daily)

```bash
python src/scraper/daily_run.py
```

Run this every day for at least 90 days to build the price series.

## 4. Train Models (Colab, ~60 min)

Open `notebooks/stage1_to_5_colab.ipynb` and run all cells top to bottom.

## 5. Run Dashboard

```bash
streamlit run app/app.py
```

## 6. Generate Chapter 6 Tables

```bash
python src/evaluation/ch6_analysis.py \
    --indir data/processed \
    --outdir results/
```

## Full setup guide with GitHub Actions: see `GITHUB_SETUP_GUIDE.md`
