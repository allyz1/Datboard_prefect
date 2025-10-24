#!/usr/bin/env python3
# sol_strategies_daily.py
# Crawler for SOL Strategies press releases to extract SOL holdings data
# Returns a DataFrame with columns: date, ticker, asset, total_holdings

import re
import time
import json
import logging
from typing import List, Dict, Optional, Set, Tuple, Any
from urllib.parse import urljoin
from datetime import date, datetime

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

logger = logging.getLogger("sol_strategies_daily")

DOMAIN = "https://solstrategies.io"
LISTING = "https://solstrategies.io/press-releases?sort=date&order=desc&page={page}&limit=6"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://solstrategies.io/"
}
TIMEOUT = 30
SLEEP_TIME = 0.5

MONTHS = ("January","February","March","April","May","June","July",
          "August","September","October","November","December")
MONTHS_RX = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"

# --- Shared numeric helpers ---
NUM_RE = r"[\d]{1,3}(?:[, ]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"

def parse_float(num_txt: str) -> Optional[float]:
    try:
        return float(num_txt.replace(",", "").replace(" ", ""))
    except Exception:
        return None

def _num_with_unit_to_float(text: str) -> Optional[float]:
    """Convert '1,234', '1.2 million', '250 thousand' → float"""
    if not text:
        return None
    m = re.search(r"([0-9][\d,]*(?:\.\d+)?)(?:\s*(thousand|million|billion))?", text, flags=re.I)
    if not m:
        return None
    val = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    if unit == "thousand": val *= 1_000
    elif unit == "million": val *= 1_000_000
    elif unit == "billion": val *= 1_000_000_000
    return val

# --- Patterns for holdings/purchases ---
AS_OF_HOLDS_RE = re.compile(
    rf'\bAs\s+of\s+(?P<date>(?:{"|".join(MONTHS)})\s+\d{{1,2}},\s*\d{{4}})\s*,?\s*.*?\bholds?\s+(?:approximately|about|around|roughly|~)?\s*'
    rf'(?P<sol>{NUM_RE})\s*(?:SOL|Solana)\b',
    re.I
)
TOTAL_HOLDINGS_RE = re.compile(
    rf'\b(?:total\s+holdings?|holdings\s+total)\s*(?:to|of|at|are|is|:)?\s*'
    rf'(?P<sol>{NUM_RE})\s*(?:SOL|Solana)\b', re.I
)
SOL_HOLDINGS_PHRASE_RE = re.compile(
    rf'\b(?P<sol>{NUM_RE})\s*(?:SOL|Solana)\s+holdings?\b', re.I
)
HOLDS_INLINE_RE = re.compile(
    rf'\bholds?\s+(?:approximately|about|around|roughly|~)?\s*(?P<sol>{NUM_RE})\s*(?:SOL|Solana)\b', re.I
)
PURCHASE_RE = re.compile(
    rf'\b(?:purchased?|acquired?|bought)\s+(?:approximately\s+)?(?P<sol>{NUM_RE})\s*(?:SOL|Solana)\b', re.I
)
PURCHASE_OF_RE = re.compile(
    rf'\bpurchase\s+of\s+(?P<sol>{NUM_RE})\s*(?:SOL|Solana)\b', re.I
)
ANNOUNCES_PURCHASE_RE = re.compile(
    rf'\bannounces?\s+(?:the\s+)?purchase\s+of\s+(?P<sol>{NUM_RE})\s*(?:SOL|Solana)\b', re.I
)
SOL_TOKENS_RE = re.compile(
    rf'\b(?P<sol>{NUM_RE})\s*(?:SOL|Solana)\s+tokens?\b', re.I
)

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

# ---------- URL discovery (fixed) ----------
def extract_press_release_urls() -> List[str]:
    """Collect ALL /press-releases/ URLs across pages; we’ll date-filter later."""
    all_urls: Set[str] = set()
    page = 1
    while True:
        url = LISTING.format(page=page)
        logger.info(f"Listing page {page}: {url}")
        resp = get(url)
        soup = BeautifulSoup(resp.text, "lxml")

        page_urls = set()
        for a in soup.select('a[href^="/press-releases/"]'):
            href = a.get("href")
            if not href:
                continue
            full = urljoin(DOMAIN, href)
            page_urls.add(full)

        # Stop when no new links appear
        new_urls = page_urls - all_urls
        logger.info(f"Found {len(page_urls)} links on page {page} ({len(new_urls)} new)")
        if not new_urls:
            break

        all_urls |= new_urls
        page += 1
        time.sleep(SLEEP_TIME)

    return sorted(all_urls)

# ---------- Date extraction (stronger) ----------
def extract_publication_date(soup: BeautifulSoup) -> Optional[str]:
    """Return ISO date (YYYY-MM-DD) or None."""
    # 1) JSON-LD (NewsArticle)
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            # Can be a dict OR a list of dicts
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    dt = item.get("datePublished") or item.get("dateCreated") or item.get("uploadDate")
                    if dt:
                        return dateparser.parse(dt, fuzzy=True).date().isoformat()
        except Exception:
            pass

    # 2) OpenGraph / article meta
    meta_candidates = [
        ("property", "article:published_time"),
        ("name", "article:published_time"),
        ("property", "og:updated_time"),
        ("name", "date"),
        ("name", "pubdate"),
    ]
    for attr, key in meta_candidates:
        tag = soup.find("meta", {attr: key})
        if tag:
            content = tag.get("content")
            if content:
                try:
                    return dateparser.parse(content, fuzzy=True).date().isoformat()
                except Exception:
                    pass

    # 3) <time> tag
    t = soup.find("time")
    if t:
        dt = t.get("datetime") or t.get_text(strip=True)
        if dt:
            try:
                return dateparser.parse(dt, fuzzy=True).date().isoformat()
            except Exception:
                pass

    # 4) Fallback: visible text search
    text = soup.get_text(" ", strip=True)
    m = re.search(rf'(?:{"|".join(MONTHS)})\s+\d{{1,2}},\s*\d{{4}}', text, flags=re.I)
    if m:
        try:
            return dateparser.parse(m.group(0), fuzzy=True).date().isoformat()
        except Exception:
            pass

    return None

# ---------- Per-article parsing ----------
def extract_sol_events(text: str) -> List[Dict[str, Any]]:
    """Extract SOL holdings & purchases from free text; returns list of dicts."""
    events: List[Dict[str, Any]] = []

    # Completed purchases (verbs)
    for m in re.finditer(
        rf"\b(?:has\s+)?(?:purchase|purchased|acquire|acquired|buy|bought)\s+(?:approximately\s+|about\s+)?({NUM_RE})(?:\s*(thousand|million|billion))?\s+(?:SOL|Solana)\b",
        text, re.I
    ):
        amt = _num_with_unit_to_float(" ".join([m.group(1), m.group(2) or ""]).strip())
        if amt:
            events.append({"record_type": "purchase", "SOL": amt})

    # Holdings snapshots (disjoint spans)
    holdings_patterns = [
        re.compile(
            rf'\bAs\s+of\s+({MONTHS_RX}\s+\d{{1,2}},\s+\d{{4}})\b(?:[^.]{{0,200}}?)?\b(?:holds?|held)\s+({NUM_RE})(?:\s*(thousand|million|billion))?\s+(?:SOL|Solana)\b',
            re.I),
        re.compile(
            rf'\b(?:total\s+holdings?|holdings\s+total)\s*(?:to|of|at|are|is|:)?\s*({NUM_RE})(?:\s*(thousand|million|billion))?\s+(?:SOL|Solana)\b',
            re.I),
        re.compile(
            rf'\b(?:now\s+holds|holds|now\s+totals?|totals?)\s+({NUM_RE})(?:\s*(thousand|million|billion))?\s+(?:SOL|Solana)\b',
            re.I),
    ]
    used: List[Tuple[int, int]] = []

    def overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return not (a[1] <= b[0] or a[0] >= b[1])

    for rx in holdings_patterns:
        for m in rx.finditer(text):
            span = (m.start(), m.end())
            if any(overlaps(span, u) for u in used):
                continue
            amt = _num_with_unit_to_float(" ".join([m.group(2) or m.group(1), (m.group(3) if m.lastindex and m.lastindex >= 3 else "") or ""]).strip())
            if amt:
                events.append({"record_type": "holdings", "SOL": amt})
                used.append(span)

    # Generic catch-alls (last resort)
    generic_patterns = [
        rf'\b(\d{{1,3}}(?:,\d{{3}})*(?:\.\d+)?)\s*(?:SOL|Solana)\b',
        rf'\bholds?\s+(?:approximately\s+|about\s+)?(\d{{1,3}}(?:,\d{{3}})*(?:\.\d+)?)\s*(?:SOL|Solana)\b',
        rf'\b(\d{{1,3}}(?:,\d{{3}})*(?:\.\d+)?)\s*(?:SOL|Solana)\s+(?:tokens?|holdings?)\b',
    ]
    for pat in generic_patterns:
        for m in re.finditer(pat, text, re.I):
            amt = _num_with_unit_to_float(m.group(1))
            if amt and amt > 0:
                events.append({"record_type": "holdings", "SOL": amt})

    # de-dupe (record_type + rounded amount)
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for e in events:
        key = (e["record_type"], round(float(e["SOL"]), 8))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    return uniq

def extract_sol_holdings_from_page(url: str) -> List[Dict]:
    try:
        resp = get(url)
        soup = BeautifulSoup(resp.text, "lxml")

        pub_iso = extract_publication_date(soup)
        if not pub_iso:
            logger.info(f"[no-date] {url}")
            return []
        # Only keep 2025+
        if date.fromisoformat(pub_iso) < date(2025, 1, 1):
            logger.info(f"[pre-2025] {pub_iso} {url}")
            return []

        # Normalize text
        text = soup.get_text(" ", strip=True)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\bSOL\s+Strategies\b', 'COMPANY_NAME', text, flags=re.I)

        events = extract_sol_events(text)
        if not events:
            return []

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        out = []
        for e in events:
            out.append({
                "date": pub_iso,
                "ticker": "STKE",
                "asset": "SOL",
                "total_holdings": e["SOL"],
                "url": url,
                "title": title,
                "record_type": e["record_type"],
            })
        return out

    except Exception as e:
        logger.error(f"Error parsing {url}: {e}")
        return []

# ---------- Driver ----------
def crawl_sol_strategies_holdings() -> pd.DataFrame:
    logger.info("Starting SOL Strategies holdings crawler")
    try:
        all_links = extract_press_release_urls()
        logger.info(f"Collected {len(all_links)} press release URLs (pre-filter)")

        all_rows: List[Dict[str, Any]] = []
        for i, url in enumerate(all_links, 1):
            logger.info(f"[{i}/{len(all_links)}] {url}")
            all_rows.extend(extract_sol_holdings_from_page(url))
            time.sleep(SLEEP_TIME)

        if not all_rows:
            logger.info("No SOL holdings found")
            return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings", "url", "title", "record_type"])

        df = pd.DataFrame(all_rows)
        logger.info(f"Found {len(df)} SOL holdings records")
        return df

    except Exception as e:
        logger.error(f"Error crawling SOL Strategies: {e}")
        return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings", "url", "title", "record_type"])

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = crawl_sol_strategies_holdings()
    if df.empty:
        print("No SOL holdings found!")
        return
    csv_filename = "sol_strategies_holdings_with_numbers.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n=== SOL Holdings Found ===")
    print(f"Saved {len(df)} records to {csv_filename}")
    with pd.option_context("display.max_rows", 200, "display.width", 140):
        print(df)

if __name__ == "__main__":
    main()
