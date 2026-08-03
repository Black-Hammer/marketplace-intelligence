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
