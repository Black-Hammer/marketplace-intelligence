"""
src/scraper/konga_temu_scraper.py
=================================
Production price scrapers for Konga (secondary) and Temu (robustness arm).

Temu important notes:
  - Temu's catalogue is predominantly imported / cross-border.
  - Temu enforces the most restrictive anti-bot controls of the three platforms.
  - On persistent 403 / 429 the scraper backs off and logs a fallback warning.
  - Retain 'platform' as a control variable; analyse per-platform before pooling.

Usage:
  python -m src.scraper.konga_temu_scraper --platform konga --category food
  python -m src.scraper.konga_temu_scraper --platform temu  --category electronics
  python -m src.scraper.konga_temu_scraper --platform all   --category all
"""

import argparse, csv, logging, random, time, urllib.robotparser
from datetime import date
from pathlib import Path
from typing import Optional
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("konga_temu")

AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

CONFIGS = {
    "konga": {
        "base":      "https://www.konga.com/search?search=",
        "robots":    "https://www.konga.com/robots.txt",
        "slugs": {
            "electronics": ["smartphones", "laptops", "tablets"],
            "generators":  ["generators", "inverter+power"],
            "food":        ["rice", "vegetable+oil", "milk+powder"],
        },
        "card_sel":  "div.-mv4, div[class*='product-card']",
        "name_sel":  "span[class*='product-title'], div[class*='productTitle']",
        "price_sel": "span[class*='price'], div[class*='price']",
        "old_sel":   "span[class*='old'], del",
        "delay":     (5.0, 11.0),
        "page_param": "&page=",
    },
    "temu": {
        "base":      "https://www.temu.com/search_result.html?search_key=",
        "robots":    "https://www.temu.com/robots.txt",
        "slugs": {
            "electronics": ["smartphone", "laptop", "earphones"],
            "generators":  ["portable+generator", "power+bank+solar"],
            "food":        ["rice+cooking", "cooking+oil", "milk+powder"],
        },
        "card_sel":  "div[class*='search-card'], div[class*='goods-card']",
        "name_sel":  "div[class*='goods-title'], span[class*='title']",
        "price_sel": "div[class*='price-text'], span[class*='price']",
        "old_sel":   "div[class*='origin-price'], del",
        "delay":     (9.0, 18.0),   # longer — Temu is aggressive
        "page_param": "&page_sn=",
    },
}

SCHEMA = ["date","platform","category","query","page","product_name",
          "price_ngn","old_price_ngn","discount_pct","discount_flag","timestamp"]


def robots_allows(platform: str, url: str) -> bool:
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(CONFIGS[platform]["robots"])
    try:
        rp.read(); return rp.can_fetch("*", url)
    except Exception as e:
        log.warning("[%s] robots.txt unreadable (%s) — caution.", platform, e)
        return True


def get(url: str, platform: str, retries: int = 5) -> Optional[requests.Response]:
    base_delay = 6.0 if platform == "temu" else 4.0
    for attempt in range(retries):
        headers = {
            "User-Agent":      random.choice(AGENTS),
            "Accept-Language": "en-NG,en;q=0.9",
            "Accept":          "text/html,application/xhtml+xml,*/*;q=0.8",
            "Referer":         CONFIGS[platform]["base"],
        }
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                return r
            elif r.status_code in (429, 503):
                w = base_delay*(2**attempt)+random.uniform(3,8)
                log.warning("[%s] HTTP %d — backing off %.1fs (attempt %d/%d)",
                            platform, r.status_code, w, attempt+1, retries)
                time.sleep(w)
            elif r.status_code == 403:
                log.error("[%s] HTTP 403 — platform blocked the request. "
                          "Consider Selenium or fall back to Section 3.5.2.", platform)
                return None
            else:
                log.warning("[%s] HTTP %d", platform, r.status_code); return None
        except requests.RequestException as e:
            w = base_delay*(2**attempt)
            log.error("[%s] %s — retry in %.1fs", platform, e, w); time.sleep(w)
    log.error("[%s] All retries exhausted — check ToS / use fallback.", platform)
    return None


def parse_price(txt: str) -> Optional[float]:
    txt = txt.replace("₦","").replace("$","").replace(",","").strip()
    for part in txt.split():
        try: return float(part)
        except ValueError: continue
    return None


def parse_page(html: str, cfg: dict, platform: str,
               category: str, query: str, page: int) -> list[dict]:
    soup  = BeautifulSoup(html, "html.parser")
    today = str(date.today()); now = time.strftime("%Y-%m-%dT%H:%M:%S")
    rows  = []
    for card in soup.select(cfg["card_sel"]):
        try:
            name_el  = card.select_one(cfg["name_sel"])
            price_el = card.select_one(cfg["price_sel"])
            if not name_el or not price_el: continue
            name  = name_el.get_text(strip=True)
            price = parse_price(price_el.get_text())
            if price is None: continue
            old_el    = card.select_one(cfg["old_sel"])
            old_price = parse_price(old_el.get_text()) if old_el else None
            disc_pct  = None
            if old_price and old_price > price:
                disc_pct = round(100*(old_price-price)/old_price, 1)
            rows.append({
                "date": today, "platform": platform, "category": category,
                "query": query, "page": page, "product_name": name,
                "price_ngn": price, "old_price_ngn": old_price,
                "discount_pct": disc_pct,
                "discount_flag": 1 if old_price else 0,
                "timestamp": now,
            })
        except Exception as e: log.debug("[%s] card error: %s", platform, e)
    return rows


def scrape(platform: str, category: str, pages: int = 5) -> list[dict]:
    cfg   = CONFIGS[platform]
    delay = cfg["delay"]
    all_rows = []
    for query in cfg["slugs"].get(category, []):
        url_base = f"{cfg['base']}{requests.utils.quote(query)}"
        if not robots_allows(platform, url_base):
            log.error("[%s] robots.txt blocks %s — skipping.", platform, query)
            continue
        for page in range(1, pages+1):
            url = f"{url_base}{cfg['page_param']}{page}"
            log.info("[%s|%s|%s] page %d", platform, category, query, page)
            r = get(url, platform)
            if r is None:
                log.warning("Skipping page %d — consider fallback (§3.5.2).", page)
                continue
            rows = parse_page(r.text, cfg, platform, category, query, page)
            log.info("  ✓ %d products", len(rows))
            all_rows.extend(rows)
            time.sleep(random.uniform(*delay))
    return all_rows


def save(rows: list[dict], platform: str,
         category: str, out_dir: str = "data/raw") -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fname = Path(out_dir)/f"{platform}_{category}_{date.today()}.csv"
    if not rows: log.warning("[%s] No rows to save.", platform); return fname
    with open(fname,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA); w.writeheader(); w.writerows(rows)
    log.info("Saved %d rows → %s", len(rows), fname); return fname


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform",  default="all",
                    choices=["konga","temu","all"])
    ap.add_argument("--category", default="all",
                    choices=["electronics","generators","food","all"])
    ap.add_argument("--pages",    type=int, default=5)
    ap.add_argument("--out_dir",  default="data/raw")
    args = ap.parse_args()
    platforms = ["konga","temu"] if args.platform=="all" else [args.platform]
    cats      = ["electronics","generators","food"] if args.category=="all" else [args.category]
    for plat in platforms:
        for cat in cats:
            rows = scrape(plat, cat, args.pages)
            save(rows, plat, cat, args.out_dir)
