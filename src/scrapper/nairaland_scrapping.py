"""
src/scraper/nairaland_scraper.py
=================================
Scrapes product-related discussion threads from Nairaland (nairaland.com),
the largest Nigerian online forum and your secondary text source (§3.4.2).

Nairaland is public and robots.txt permits crawling of board and topic pages.
The scraper targets three boards:
  - /business  (market prices, trader commentary)
  - /phones    (consumer electronics reviews)
  - /nairaland/science/technology (tech product commentary)

It collects post text, username, date, board, and thread title, then filters
to posts mentioning the thesis product categories + price/sentiment signals.

Output:
  data/raw/nairaland_{category}_{YYYY-MM-DD}.csv
  Columns: date, source, category, board, thread_title, thread_url,
           post_text, username, post_date, post_number

Usage:
  python -m src.scraper.nairaland_scraper --category electronics --pages 5
  python -m src.scraper.nairaland_scraper --category all --pages 3
"""

import argparse, csv, logging, random, re, time, urllib.robotparser
from datetime import date
from pathlib import Path
from typing import Optional
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("nairaland")

BASE    = "https://www.nairaland.com"
ROBOTS  = f"{BASE}/robots.txt"

# ─── boards and keyword filters per category ──────────────────────────────────
BOARDS = {
    "electronics": [
        "/phones/1",          # Phones board
        "/technology/1",      # Technology board
    ],
    "generators": [
        "/business/1",        # Business board
        "/technology/1",
    ],
    "food": [
        "/business/1",        # Business / market prices
        "/nairaland/general/1",
    ],
}

# keyword filter: a post must mention at least one keyword from each set
FILTER_KEYWORDS = {
    "electronics": {
        "products": ["phone","laptop","tablet","iphone","samsung","tecno","itel","hp","dell","airpods"],
        "platforms": ["jumia","konga","temu"],
        "signals":  ["price","cost","cheap","expensive","wahala","fake","original","scam","quality","good","bad"],
    },
    "generators": {
        "products": ["generator","inverter","solar","ups","power","genset"],
        "platforms": ["jumia","konga","temu","market"],
        "signals":  ["price","cost","expensive","cheap","wahala","quality","original","recommend"],
    },
    "food": {
        "products": ["rice","oil","vegetable oil","palm oil","milk","flour","pasta","tomato","beans"],
        "platforms": ["jumia","konga","temu","market","supermarket"],
        "signals":  ["price","cost","expensive","cheap","increase","reduce","wahala","quality"],
    },
}

AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

SCHEMA = ["date","source","category","board","thread_title","thread_url",
          "post_text","username","post_date","post_number"]


def robots_allows(url: str) -> bool:
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(ROBOTS)
    try:
        rp.read(); return rp.can_fetch("*", url)
    except Exception as e:
        log.warning("robots.txt unreadable (%s).", e); return True


def get(url: str, retries: int = 4,
        base_delay: float = 3.0) -> Optional[requests.Response]:
    for attempt in range(retries):
        try:
            r = requests.get(url,
                headers={"User-Agent": random.choice(AGENTS),
                         "Accept-Language": "en-NG,en;q=0.9"},
                timeout=25)
            if r.status_code == 200: return r
            elif r.status_code == 429:
                w = base_delay*(2**attempt)+random.uniform(2,5)
                log.warning("429 — backing off %.1fs", w); time.sleep(w)
            else:
                log.warning("HTTP %d: %s", r.status_code, url[:70]); return None
        except requests.RequestException as e:
            w = base_delay*(2**attempt)
            log.error("%s — retry in %.1fs", e, w); time.sleep(w)
    return None


def passes_filter(text: str, category: str) -> bool:
    """Return True if text mentions at least one keyword from each filter set."""
    tl = text.lower()
    f  = FILTER_KEYWORDS[category]
    return (any(k in tl for k in f["products"]) and
            any(k in tl for k in f["signals"]))


def get_threads(board_url: str) -> list[dict]:
    """Extract thread links and titles from a Nairaland board page."""
    r = get(f"{BASE}{board_url}")
    if r is None: return []
    soup = BeautifulSoup(r.text, "html.parser")
    threads = []
    for row in soup.select("table#threads td.w, table#topics td.w"):
        link = row.select_one("a[href]")
        if link:
            href  = link.get("href","")
            title = link.get_text(strip=True)
            if href.startswith("/"): threads.append({"url": href, "title": title})
    return threads


def scrape_thread(thread: dict, category: str,
                  board: str) -> list[dict]:
    """Scrape all posts from a thread and filter to relevant ones."""
    url = f"{BASE}{thread['url']}"
    if not robots_allows(url): return []
    r = get(url)
    if r is None: return []
    soup  = BeautifulSoup(r.text, "html.parser")
    today = str(date.today())
    rows  = []

    for i, post_div in enumerate(soup.select("div.narrow")):
        text_el = post_div.select_one("div.l p, div[class*='post'] p")
        if text_el is None: text_el = post_div
        text = text_el.get_text(separator=" ", strip=True)
        if len(text) < 30: continue
        if not passes_filter(text, category): continue

        user_el = post_div.select_one("a.user")
        date_el = post_div.select_one("span.s b")
        rows.append({
            "date":         today,
            "source":       "nairaland",
            "category":     category,
            "board":        board,
            "thread_title": thread["title"],
            "thread_url":   url,
            "post_text":    text[:1000],   # cap at 1000 chars per post
            "username":     user_el.get_text(strip=True) if user_el else "",
            "post_date":    date_el.get_text(strip=True) if date_el else "",
            "post_number":  i,
        })
    return rows


def scrape_category(category: str, pages: int = 5,
                    delay: tuple = (3.0, 7.0)) -> list[dict]:
    all_rows = []
    for board_path_base in BOARDS.get(category, []):
        # board_path_base e.g. "/phones/1" — remove the page suffix
        board = board_path_base.rsplit("/",1)[0]
        for page in range(1, pages+1):
            board_url = f"{board}/{page}"
            if not robots_allows(f"{BASE}{board_url}"): continue
            log.info("[nairaland|%s] board %s page %d", category, board, page)
            threads = get_threads(board_url)
            log.info("  found %d threads", len(threads))
            for thread in threads[:15]:   # top 15 threads per page
                rows = scrape_thread(thread, category, board)
                if rows:
                    log.info("  thread '%s...' → %d relevant posts",
                             thread["title"][:35], len(rows))
                all_rows.extend(rows)
                time.sleep(random.uniform(*delay))
    log.info("Nairaland category '%s': %d posts total.", category, len(all_rows))
    return all_rows


def save(rows: list[dict], category: str,
         out_dir: str = "data/raw") -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fname = Path(out_dir)/f"nairaland_{category}_{date.today()}.csv"
    if not rows: log.warning("No posts to save."); return fname
    with open(fname,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    log.info("Saved %d posts → %s", len(rows), fname)
    return fname


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Nairaland discussion scraper")
    ap.add_argument("--category", default="all",
                    choices=["electronics","generators","food","all"])
    ap.add_argument("--pages",   type=int, default=5)
    ap.add_argument("--out_dir", default="data/raw")
    args = ap.parse_args()
    cats = ["electronics","generators","food"] if args.category=="all" else [args.category]
    for cat in cats:
        rows = scrape_category(cat, args.pages)
        save(rows, cat, args.out_dir)
