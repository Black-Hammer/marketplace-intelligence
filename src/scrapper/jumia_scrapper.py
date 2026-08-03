"""
src/scraper/jumia_scraper.py
============================
Production daily price scraper for Jumia Nigeria.

Three thesis product categories:
  - electronics  : smartphones, laptops, tablets
  - generators   : generators, inverters, power equipment
  - food         : rice, vegetable oil, milk powder, food staples

Features:
  - robots.txt compliance check before any scraping
  - Rotating user-agent pool
  - Exponential backoff on 429 / 503
  - Polite inter-request delay (configurable)
  - Dry-run mode (--dry_run) — parses saved HTML, no live requests
  - Saves data/raw/jumia_{category}_{YYYY-MM-DD}.csv

Usage:
  python -m src.scraper.jumia_scraper --category electronics --pages 10
  python -m src.scraper.jumia_scraper --category all         --pages 5
  python -m src.scraper.jumia_scraper --dry_run --html saved.html
"""

import argparse, csv, logging, random, time, urllib.robotparser
from datetime import date
from pathlib import Path
from typing import Optional
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("jumia")

BASE   = "https://www.jumia.com.ng"
SEARCH = f"{BASE}/catalog/?q="

SLUGS = {
    "electronics": ["smartphones", "laptops", "tablets", "headphones"],
    "generators":  ["generators", "inverter", "solar+panel", "ups+power"],
    "food":        ["rice+5kg", "vegetable+oil", "milk+powder", "tomato+paste"],
}

AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

SCHEMA = ["date","platform","category","query","page","product_id",
          "product_name","price_ngn","old_price_ngn","discount_pct",
          "rating","review_count","verified_seller","timestamp"]


def robots_allows(url: str) -> bool:
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"{BASE}/robots.txt")
    try:
        rp.read(); return rp.can_fetch("*", url)
    except Exception as e:
        log.warning("robots.txt unreadable (%s) — proceeding with caution.", e)
        return True


def get(url: str, retries: int = 5, base_delay: float = 4.0) -> Optional[requests.Response]:
    for attempt in range(retries):
        headers = {"User-Agent": random.choice(AGENTS),
                   "Accept-Language": "en-NG,en;q=0.9",
                   "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                   "DNT": "1"}
        try:
            r = requests.get(url, headers=headers, timeout=25)
            if r.status_code == 200:
                return r
            elif r.status_code in (429, 503):
                w = base_delay*(2**attempt)+random.uniform(2,6)
                log.warning("HTTP %d — backing off %.1fs", r.status_code, w)
                time.sleep(w)
            elif r.status_code == 403:
                log.error("HTTP 403 — Jumia may require a browser session.")
                return None
            else:
                log.warning("HTTP %d for %s", r.status_code, url[:70]); return None
        except requests.RequestException as e:
            w = base_delay*(2**attempt)
            log.error("Request error: %s — retry in %.1fs", e, w); time.sleep(w)
    return None


def parse_price(txt: str) -> Optional[float]:
    try: return float(txt.replace("₦","").replace(",","").strip().split()[0])
    except (ValueError, IndexError): return None


def parse_page(html: str, category: str, query: str, page: int) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    today = str(date.today()); now = time.strftime("%Y-%m-%dT%H:%M:%S")
    rows = []
    for card in soup.select("article.prd, div[data-testid='productCard']"):
        try:
            name_el  = card.select_one("h3.name, div.name, [class*='productTitle']")
            price_el = card.select_one("div.prc, [class*='price']")
            if not name_el or not price_el: continue
            name  = name_el.get_text(strip=True)
            price = parse_price(price_el.get_text())
            if price is None: continue
            old_el    = card.select_one("div.old, [class*='oldPrice']")
            old_price = parse_price(old_el.get_text()) if old_el else None
            disc_el   = card.select_one("div.bdg._dsct, [class*='discount']")
            disc_pct  = None
            if disc_el:
                try: disc_pct = float(disc_el.get_text(strip=True).replace("-","").replace("%",""))
                except ValueError: pass
            if disc_pct is None and old_price and old_price > price:
                disc_pct = round(100*(old_price-price)/old_price, 1)
            rat_el = card.select_one("div.stars._s, [class*='rating']")
            rev_el = card.select_one("div.rev, [class*='reviewCount']")
            rating = None
            if rat_el:
                try: rating = float(rat_el.get_text(strip=True).split()[0])
                except ValueError: pass
            reviews = 0
            if rev_el:
                try: reviews = int(rev_el.get_text(strip=True).replace("(","").replace(")","").replace(",",""))
                except ValueError: pass
            pid = card.get("data-id","") or card.get("data-sku","")
            rows.append({"date":today,"platform":"jumia","category":category,
                "query":query,"page":page,"product_id":pid,"product_name":name,
                "price_ngn":price,"old_price_ngn":old_price,"discount_pct":disc_pct,
                "rating":rating,"review_count":reviews,
                "verified_seller":1 if card.select_one("[class*='verified']") else 0,
                "timestamp":now})
        except Exception as e: log.debug("Card error: %s", e)
    return rows


def scrape_category(category: str, pages: int = 10,
                    delay: tuple = (4.0, 9.0)) -> list[dict]:
    all_rows = []
    for query in SLUGS.get(category, []):
        url_base = f"{SEARCH}{query}"
        if not robots_allows(url_base):
            log.error("robots.txt blocks %s — skipping.", url_base); continue
        for page in range(1, pages+1):
            url = f"{url_base}&page={page}#catalog-listing"
            log.info("[jumia|%s|%s] page %d", category, query, page)
            r = get(url)
            if r is None: continue
            rows = parse_page(r.text, category, query, page)
            log.info("  ✓ %d products", len(rows))
            all_rows.extend(rows)
            time.sleep(random.uniform(*delay))
    return all_rows


def save(rows: list[dict], category: str, out_dir: str = "data/raw") -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fname = Path(out_dir)/f"jumia_{category}_{date.today()}.csv"
    if not rows: log.warning("No rows to save."); return fname
    with open(fname,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA); w.writeheader(); w.writerows(rows)
    log.info("Saved %d rows → %s", len(rows), fname); return fname


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Jumia Nigeria price scraper")
    ap.add_argument("--category", default="all",
                    choices=["electronics","generators","food","all"])
    ap.add_argument("--pages",    type=int, default=10)
    ap.add_argument("--out_dir",  default="data/raw")
    ap.add_argument("--dry_run",  action="store_true")
    ap.add_argument("--html",     default="")
    args = ap.parse_args()
    cats = list(SLUGS.keys()) if args.category=="all" else [args.category]
    if args.dry_run:
        html = open(args.html, encoding="utf-8").read()
        rows = parse_page(html, cats[0], "dry_run", 1)
        save(rows, cats[0], args.out_dir)
    else:
        for cat in cats:
            rows = scrape_category(cat, args.pages)
            save(rows, cat, args.out_dir)
