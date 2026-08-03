"""
src/scraper/twitter_scraper.py
===============================
Collects code-mixed Nigerian consumer posts about the three product categories
from X/Twitter (formerly Twitter) using two access paths:

PATH A — Twitter API v2 (recommended, academic/free tier)
  Requires a Bearer Token from developer.twitter.com.
  Set environment variable:  export TWITTER_BEARER_TOKEN=your_token_here
  Free tier: 500,000 tweets/month (Basic: 10M/month).

PATH B — snscrape (no API key required, rate-limited by Twitter)
  Falls back automatically when no Bearer Token is set.
  Install: pip install snscrape
  Note: snscrape scrapes the public timeline; results may be inconsistent.

Search queries are designed to capture code-mixed Nigerian discourse
(English + Pidgin + Hausa + Yoruba) about the three product categories
on Jumia, Konga, and Temu specifically.

Output:
  data/raw/twitter_{category}_{YYYY-MM-DD}.csv
  Columns: date, source, category, query, text, lang, created_at,
           likes, retweets, tweet_id

Usage:
  # API path (recommended)
  export TWITTER_BEARER_TOKEN=your_token
  python -m src.scraper.twitter_scraper --category electronics --max_results 1000

  # snscrape fallback (no token needed)
  python -m src.scraper.twitter_scraper --category food --use_snscrape

  # all categories
  python -m src.scraper.twitter_scraper --category all --max_results 500
"""

import argparse, csv, logging, os, time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("twitter")

# ─── search queries per category ─────────────────────────────────────────────
# Designed to capture Nigerian Pidgin, Hausa, English, and Yoruba discourse.
# Each query targets a category + platform + price-signal keyword.
QUERIES = {
    "electronics": [
        "(phone OR laptop OR tablet) (Jumia OR Konga OR Temu) "
        "(price OR cost OR expensive OR cheap OR wahala) lang:en",
        "(phone OR laptop) (Jumia OR Konga) (na scam OR fake OR original OR quality) "
        "-is:retweet",
        "Jumia electronics (price don increase OR price don drop OR good deal) "
        "-is:retweet",
    ],
    "generators": [
        "(generator OR inverter) (Jumia OR Konga OR Temu) "
        "(price OR cost OR expensive OR cheap) lang:en -is:retweet",
        "(generator OR inverter) Nigeria (wahala OR good OR bad OR overpriced OR recommend) "
        "-is:retweet",
        "Jumia generator (na correct OR fake OR scam OR quality) -is:retweet",
    ],
    "food": [
        "(rice OR \"vegetable oil\" OR \"palm oil\" OR milk) "
        "(Jumia OR Konga OR Temu) (price OR cost OR expensive) lang:en -is:retweet",
        "(rice OR oil) Nigeria market (price don increase OR too costly OR affordable) "
        "-is:retweet",
        "Jumia food (original OR fake OR wahala OR good) -is:retweet",
    ],
}

SCHEMA = ["date","source","category","query","text","lang",
          "created_at","likes","retweets","tweet_id"]

# ─── PATH A: Twitter API v2 ───────────────────────────────────────────────────
def search_api_v2(query: str, max_results: int = 100,
                  since_days: int = 7) -> list[dict]:
    """
    Use Twitter API v2 recent-search endpoint.
    Requires TWITTER_BEARER_TOKEN environment variable.
    Docs: https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-recent
    """
    token = os.environ.get("TWITTER_BEARER_TOKEN","")
    if not token:
        log.warning("TWITTER_BEARER_TOKEN not set — use Path B (snscrape) instead.")
        return []

    import requests as req
    start_time = (date.today() - timedelta(days=since_days)).strftime("%Y-%m-%dT00:00:00Z")
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {token}"}
    rows = []; next_token = None

    while len(rows) < max_results:
        params = {
            "query":        query,
            "max_results":  min(100, max_results - len(rows)),
            "start_time":   start_time,
            "tweet.fields": "created_at,lang,public_metrics,author_id",
        }
        if next_token:
            params["next_token"] = next_token
        try:
            r = req.get(url, headers=headers, params=params, timeout=20)
            if r.status_code == 429:
                log.warning("API rate limit — sleeping 15 min ...")
                time.sleep(900); continue
            if r.status_code != 200:
                log.error("API error %d: %s", r.status_code, r.text[:200])
                break
            data = r.json()
            for t in data.get("data", []):
                m = t.get("public_metrics", {})
                rows.append({
                    "date":       str(date.today()),
                    "source":     "twitter_api_v2",
                    "category":   "",   # filled by caller
                    "query":      query,
                    "text":       t.get("text",""),
                    "lang":       t.get("lang",""),
                    "created_at": t.get("created_at",""),
                    "likes":      m.get("like_count",0),
                    "retweets":   m.get("retweet_count",0),
                    "tweet_id":   t.get("id",""),
                })
            meta = data.get("meta",{})
            next_token = meta.get("next_token")
            if not next_token:
                break
            time.sleep(1.0)   # polite 1-second pause between paginated requests
        except Exception as e:
            log.error("API request failed: %s", e); break

    log.info("API: collected %d tweets for query: %s...", len(rows), query[:50])
    return rows


# ─── PATH B: snscrape (no token required) ────────────────────────────────────
def search_snscrape(query: str, max_results: int = 200,
                    since_days: int = 7) -> list[dict]:
    """
    Use snscrape to scrape recent tweets matching the query.
    Install: pip install snscrape
    Note: snscrape may break when Twitter changes its frontend.
          Check https://github.com/JustAnotherArchivist/snscrape for updates.
    """
    try:
        import snscrape.modules.twitter as sntwitter
    except ImportError:
        log.error("snscrape not installed: pip install snscrape")
        return []

    since = (date.today() - timedelta(days=since_days)).strftime("%Y-%m-%d")
    full_query = f"{query} since:{since}"
    rows = []
    try:
        for i, tweet in enumerate(
                sntwitter.TwitterSearchScraper(full_query).get_items()):
            if i >= max_results: break
            rows.append({
                "date":       str(date.today()),
                "source":     "snscrape",
                "category":   "",
                "query":      query,
                "text":       tweet.rawContent,
                "lang":       tweet.lang or "",
                "created_at": str(tweet.date),
                "likes":      tweet.likeCount or 0,
                "retweets":   tweet.retweetCount or 0,
                "tweet_id":   str(tweet.id),
            })
    except Exception as e:
        log.error("snscrape error: %s", e)
    log.info("snscrape: collected %d tweets", len(rows))
    return rows


# ─── main collection ──────────────────────────────────────────────────────────
def collect_category(category: str, max_results: int = 500,
                     since_days: int = 7,
                     use_snscrape: bool = False) -> list[dict]:
    all_rows = []
    queries  = QUERIES.get(category, [])
    per_q    = max(50, max_results // len(queries))

    for q in queries:
        if use_snscrape:
            rows = search_snscrape(q, per_q, since_days)
        else:
            rows = search_api_v2(q, per_q, since_days)
            if not rows:   # fallback if token missing
                log.info("Falling back to snscrape for this query.")
                rows = search_snscrape(q, per_q, since_days)
        for r in rows:
            r["category"] = category
        all_rows.extend(rows)
        time.sleep(2.0)   # polite pause between queries

    # deduplicate by tweet_id
    seen = set(); unique = []
    for r in all_rows:
        tid = r.get("tweet_id","")
        if tid and tid not in seen:
            seen.add(tid); unique.append(r)
    log.info("Category '%s': %d unique tweets collected.", category, len(unique))
    return unique


def save(rows: list[dict], category: str,
         out_dir: str = "data/raw") -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fname = Path(out_dir)/f"twitter_{category}_{date.today()}.csv"
    if not rows: log.warning("No tweets to save."); return fname
    with open(fname,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    log.info("Saved %d tweets → %s", len(rows), fname)
    return fname


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Twitter/X scraper for Nigerian marketplace sentiment")
    ap.add_argument("--category",    default="all",
                    choices=["electronics","generators","food","all"])
    ap.add_argument("--max_results", type=int, default=500)
    ap.add_argument("--since_days",  type=int, default=7)
    ap.add_argument("--use_snscrape",action="store_true",
                    help="Use snscrape instead of Twitter API v2")
    ap.add_argument("--out_dir",     default="data/raw")
    args = ap.parse_args()

    cats = ["electronics","generators","food"] if args.category=="all" else [args.category]
    for cat in cats:
        rows = collect_category(cat, args.max_results,
                                args.since_days, args.use_snscrape)
        save(rows, cat, args.out_dir)
