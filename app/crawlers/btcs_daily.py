#!/usr/bin/env python3
# btcs_eth_holdings_df.py
# Build a pandas DataFrame (date, ticker, asset, total_holdings) from BTCS PRs.

import re
import time
import logging
from typing import List, Dict, Optional, Set, Tuple
from urllib.parse import urlencode, urljoin

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from datetime import date

logger = logging.getLogger("btcs_eth_df")  # configure in your main

BASE = "https://platform.btcs.com/assets/news_media.cfm"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
TIMEOUT = 30
LISTING_SLEEP = 0.15
PAGE_SLEEP = 0.12

MONTHS = ("January","February","March","April","May","June","July",
          "August","September","October","November","December")
DATE_2025_RE = re.compile(rf'\b(?:{"|".join(MONTHS)})\s+\d{{1,2}},\s*2025\b', re.I)

# Title filters (disable by passing include_all_titles=True)
ETH_RE = re.compile(r'\b(?:ETH|Ether(?:eum)?|digital[-\s]?assets?)\b', re.I)
QTR_ANNUAL_RE = re.compile(
    r'\b(?:Q[1-4](?:\s*[-/’\' ]?\s*(?:\d{2}|\d{4}))?'
    r'|quarterly|first\s+quarter|second\s+quarter|third\s+quarter|fourth\s+quarter'
    r'|post[-\s]?quarter|post\s*quarter'
    r'|annual|year[-\s]?end|full[-\s]?year|FY\s*\d{2,4})\b',
    re.I
)

HREF_BLACKLIST_RE = re.compile(r'#|/assets/news_media\.cfm\?')
SQUEEZE_WS = re.compile(r"\s+")
def clean_text(s: str) -> str:
    return SQUEEZE_WS.sub(" ", (s or "").strip())

def get(url: str) -> requests.Response:
    for attempt in range(3):
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            time.sleep(0.6 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp

def build_url(category: str, year: int, page: int) -> str:
    return f"{BASE}?{urlencode({'category': category, 'year': year, 'page': page})}"

def detect_last_page(html: str) -> int:
    soup = BeautifulSoup(html, "lxml")
    nums = []
    for a in soup.find_all("a"):
        t = a.get_text(strip=True)
        if t.isdigit():
            try:
                nums.append(int(t))
            except ValueError:
                pass
    return max(nums) if nums else 1

def parse_listing_page(html: str, page_url: str) -> List[Dict]:
    soup = BeautifulSoup(html, "lxml")
    items: List[Dict] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if HREF_BLACKLIST_RE.search(href):
            continue
        text = clean_text(a.get_text(" ", strip=True))
        if not text:
            continue
        m = DATE_2025_RE.search(text)
        if not m:
            continue
        title = text[:m.start()].strip(" –—-: ")
        title = re.sub(r'^\s*Press\s*Release\s*[:\-–—]?\s*', '', title, flags=re.I)
        if not title:
            continue
        url = urljoin(page_url, href)
        items.append({"title": title, "url": url, "date_text": m.group(0)})
    return items

# --- Page date helpers ---
PUBTIME_META_SELECTORS = [
    ('meta[property="article:published_time"]', "content"),
    ('meta[name="pubdate"]', "content"),
    ('meta[name="publication_date"]', "content"),
    ('meta[name="date"]', "content"),
    ('meta[itemprop="datePublished"]', "content"),
    ('time[datetime]', "datetime"),
]

def extract_published_date(soup: BeautifulSoup, fallback_listing_date: Optional[str]) -> Optional[str]:
    for sel, attr in PUBTIME_META_SELECTORS:
        node = soup.select_one(sel)
        if node and node.get(attr):
            try:
                dt = dateparser.parse(node.get(attr), fuzzy=True)
                if dt:
                    return dt.date().isoformat()
            except Exception:
                pass
    t = soup.find("time")
    if t:
        try:
            dt = dateparser.parse(t.get("datetime") or t.get_text(" ", strip=True), fuzzy=True)
            if dt:
                return dt.date().isoformat()
        except Exception:
            pass
    if fallback_listing_date:
        try:
            dt = dateparser.parse(fallback_listing_date, fuzzy=True)
            if dt:
                return dt.date().isoformat()
        except Exception:
            pass
    return None

# --- Holdings extraction (ETH only; ignore USD) ---
NUM_RE = r"[\d]{1,3}(?:[, ]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"
def parse_float(num_txt: str) -> Optional[float]:
    try:
        return float(num_txt.replace(",", "").replace(" ", ""))
    except Exception:
        return None

AS_OF_HOLDS_RE = re.compile(
    rf'\bAs\s+of\s+(?P<date>(?:{"|".join(MONTHS)})\s+\d{{1,2}},\s*\d{{4}})\s*,?\s*.*?\bholds?\s+(?:approximately|about|around|roughly|~)?\s*'
    rf'(?P<eth>{NUM_RE})\s*(?:ETH|Ether(?:eum)?)\b',
    re.I
)

# Increased ... to <num> [ETH]? ... as of <date>   ← ETH unit is OPTIONAL
INCREASE_TO_ASOF_RELAXED_RE = re.compile(
    rf'\bIncreased\s+(?:Ethereum\s+)?holdings\s+to\s+(?P<eth>{NUM_RE})(?:\s*(?:ETH|Ether(?:eum)?)\b)?'
    rf'[^.]*?\bAs\s+of\s+(?P<date>(?:{"|".join(MONTHS)})\s+\d{{1,2}},\s*\d{{4}})\b',
    re.I
)

TOTAL_HOLDINGS_RE = re.compile(
    rf'\b(?:total\s+holdings?|holdings\s+total)\s*(?:to|of|at|are|is|:)?\s*'
    rf'(?P<eth>{NUM_RE})\s*(?:ETH|Ether(?:eum)?)\b', re.I
)
ETH_HOLDINGS_PHRASE_RE = re.compile(
    rf'\b(?P<eth>{NUM_RE})\s*(?:ETH|Ether(?:eum)?)\s+holdings?\b', re.I
)
HOLDS_INLINE_RE = re.compile(
    rf'\bholds?\s+(?:approximately|about|around|roughly|~)?\s*(?P<eth>{NUM_RE})\s*(?:ETH|Ether(?:eum)?)\b', re.I
)

UP_FROM_Q_RE = re.compile(
    rf'\bup\s+from\s+(?P<eth>{NUM_RE})\s*(?:ETH|Ether(?:eum)?)\s+at\s+the\s+end\s+of\s+(?P<q>Q[1-4])\s+2025\b',
    re.I
)

def quarter_end_date(year: int, q: str) -> date:
    q = q.upper()
    if q == "Q1":  return date(year, 3, 31)
    if q == "Q2":  return date(year, 6, 30)
    if q == "Q3":  return date(year, 9, 30)
    return date(year, 12, 31)

def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\!\?])\s+(?=[A-Z0-9\(])", text)
    return [p.strip() for p in parts if p.strip()]

def best_content_text(soup: BeautifulSoup) -> str:
    for sel in ["article", ".entry-content", ".post-content", "main", ".content"]:
        node = soup.select_one(sel)
        if node:
            return clean_text(node.get_text(" ", strip=True))
    return clean_text(soup.get_text(" ", strip=True))

def extract_holdings_events(full_text: str,
                            page_pubdate_iso: Optional[str]) -> List[Tuple[str, float, str]]:
    events: List[Tuple[str, float, str]] = []
    sentences = split_sentences(full_text)

    for s in sentences:
        # As of … holds …
        for m in AS_OF_HOLDS_RE.finditer(s):
            dt = dateparser.parse(m.group("date"), fuzzy=True).date().isoformat()
            eth = parse_float(m.group("eth"))
            if eth is not None:
                events.append((dt, eth, s))

        # Increased … to <num> [ETH]? … as of <date>
        for m in INCREASE_TO_ASOF_RELAXED_RE.finditer(s):
            dt = dateparser.parse(m.group("date"), fuzzy=True).date().isoformat()
            eth = parse_float(m.group("eth"))
            if eth is not None:
                events.append((dt, eth, s))

        # Up from … end of Qx 2025
        for m in UP_FROM_Q_RE.finditer(s):
            eth = parse_float(m.group("eth"))
            q = m.group("q").upper()
            dt = quarter_end_date(2025, q).isoformat()
            if eth is not None:
                events.append((dt, eth, s))

    # Second pass: totals without explicit date → use explicit or page date
    for s in sentences:
        explicit_dt = None
        md = re.search(rf'(?:{"|".join(MONTHS)})\s+\d{{1,2}},\s*\d{{4}}', s, flags=re.I)
        if md:
            try:
                explicit_dt = dateparser.parse(md.group(0), fuzzy=True).date().isoformat()
            except Exception:
                explicit_dt = None
        dt_iso = explicit_dt or page_pubdate_iso
        if not dt_iso:
            continue
        for pat in (TOTAL_HOLDINGS_RE, ETH_HOLDINGS_PHRASE_RE, HOLDS_INLINE_RE):
            m = pat.search(s)
            if m:
                eth = parse_float(m.group("eth"))
                if eth is not None:
                    events.append((dt_iso, eth, s))
                    break

    return events

def parse_press_release(url: str, listing_date_text: Optional[str]) -> List[Dict]:
    resp = get(url)
    soup = BeautifulSoup(resp.text, "lxml")
    pubdate_iso = extract_published_date(soup, listing_date_text)
    full_text = best_content_text(soup)
    events = extract_holdings_events(full_text, pubdate_iso)

    rows: List[Dict] = []
    for dt_iso, eth, _snip in events:
        rows.append({
            "date": dt_iso,
            "ticker": "BTCS",
            "asset": "ETH",
            "total_holdings": eth,
            "source_url": url,
        })
    return rows

# ----------------------- PUBLIC ENTRYPOINT -----------------------
def get_btcs_eth_holdings_df(
    year: int = 2025,
    category: str = "PR",
    max_pages: int = 0,
    include_all_titles: bool = False,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      - date (YYYY-MM-DD)
      - ticker = 'BTCS'
      - asset  = 'ETH'
      - total_holdings (float)
    If nothing is found, returns an empty DF and logs an INFO line.
    """
    # 1) listings
    url1 = build_url(category, year, 1)
    html1 = get(url1).text
    last = detect_last_page(html1)
    if max_pages and max_pages > 0:
        last = min(last, max_pages)

    listings = parse_listing_page(html1, url1)
    for p in range(2, last + 1):
        url = build_url(category, year, p)
        html = get(url).text
        listings.extend(parse_listing_page(html, url))
        time.sleep(LISTING_SLEEP)

    if not listings:
        logger.info(f"BTCS: no PR listings found for {year} — no hits.")
        return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings"])

    # de-dupe by URL
    seen: Set[str] = set()
    deduped = []
    for it in listings:
        if it["url"] not in seen:
            seen.add(it["url"])
            deduped.append(it)

    # 2) filter by title unless include_all_titles
    if include_all_titles:
        chosen = deduped
    else:
        chosen = [it for it in deduped if ETH_RE.search(it["title"]) or QTR_ANNUAL_RE.search(it["title"])]

    if not chosen:
        logger.info("BTCS: no ETH-related PR titles matched — no hits.")
        return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings"])

    # 3) parse each page → events
    events: List[Dict] = []
    for it in chosen:
        try:
            rows = parse_press_release(it["url"], it.get("date_text"))
            events.extend(rows)
        except Exception:
            # swallow individual page errors; keep crawling
            pass
        time.sleep(PAGE_SLEEP)

    if not events:
        logger.info("BTCS: parsed 0 ETH holdings events — no hits.")
        return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings"])

    # 4) normalize: prefer the largest ETH per exact date
    by_date: Dict[str, Dict] = {}
    for ev in events:
        d = ev["date"]
        cur = by_date.get(d)
        if cur is None or float(ev["total_holdings"]) > float(cur["total_holdings"]):
            by_date[d] = ev

    if not by_date:
        logger.info("BTCS: no normalized timeline rows — no hits.")
        return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings"])

    timeline = sorted(by_date.values(), key=lambda r: r["date"])

    # 5) table-ready DF
    df = pd.DataFrame({
        "date": [r["date"] for r in timeline],
        "ticker": ["BTCS"] * len(timeline),
        "asset": ["ETH"] * len(timeline),
        "total_holdings": [float(r["total_holdings"]) for r in timeline],
    })
    return df


# Quick local test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(get_btcs_eth_holdings_df().head())
