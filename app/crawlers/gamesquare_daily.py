#!/usr/bin/env python3
# gamesquare_daily.py
# Crawler for Gamesquare investor news site to extract ETH holdings data
# Returns a DataFrame with columns: date, ticker, asset, total_holdings

import re
import time
import logging
from typing import List, Dict, Optional, Set, Tuple
from urllib.parse import urljoin
from datetime import date

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

# Selenium imports for URL extraction
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logger = logging.getLogger("gamesquare_daily")

BASE_URL = "https://investors.gamesquare.com/news/default.aspx"
DOMAIN = "https://investors.gamesquare.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://investors.gamesquare.com/"
}
TIMEOUT = 30
SLEEP_TIME = 0.5

# Date patterns for 2025
MONTHS = ("January","February","March","April","May","June","July",
          "August","September","October","November","December")
DATE_2025_RE = re.compile(rf'\b(?:{"|".join(MONTHS)})\s+\d{{1,2}},\s*2025\b', re.I)

# ETH holdings extraction patterns (adapted from BTCS)
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

# Purchase patterns
PURCHASE_RE = re.compile(
    rf'\b(?:purchased?|acquired?|bought)\s+(?:approximately\s+)?(?P<eth>{NUM_RE})\s*(?:ETH|Ether(?:eum)?)\b', re.I
)

def get(url: str) -> requests.Response:
    """Fetch URL with retry logic"""
    for attempt in range(3):
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            time.sleep(0.6 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


def extract_eth_holdings_from_page(driver: webdriver.Chrome, url: str) -> List[Dict]:
    """Parse a single news page for ETH/Ethereum holdings mentions using Selenium"""
    try:
        driver.get(url)
        time.sleep(2)  # Wait for page to load
        
        # Get page source and parse with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, "lxml")
        
        # Extract all text content
        text_content = soup.get_text(" ", strip=True)
        text_content = re.sub(r'\s+', ' ', text_content)
        
        # Look for ETH/Ethereum patterns
        eth_patterns = [
            r'\b(?:ETH|Ethereum|Ether)\b',
            r'\b(?:digital\s+assets?|cryptocurrency|crypto)\b',
            r'\b(?:blockchain|Web3)\b'
        ]
        
        # Check if page contains any ETH-related content
        has_eth_content = any(re.search(pattern, text_content, re.I) for pattern in eth_patterns)
        
        if not has_eth_content:
            return []
        
        # Extract holdings events
        events = extract_holdings_events(text_content, None)
        
        holdings_found = []
        for dt_iso, eth_amount, context in events:
            holdings_found.append({
                'date': dt_iso,
                'ticker': 'GAME',
                'asset': 'ETH',
                'total_holdings': eth_amount,
                'url': url,
                'context': context
            })
        
        return holdings_found
        
    except Exception as e:
        logger.error(f"Error parsing {url}: {e}")
        return []

def extract_holdings_events(full_text: str, page_pubdate_iso: Optional[str]) -> List[Tuple[str, float, str]]:
    """Extract ETH holdings events from text (adapted from BTCS)"""
    events: List[Tuple[str, float, str]] = []
    sentences = re.split(r"(?<=[\.\!\?])\s+(?=[A-Z0-9\(])", full_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    for s in sentences:
        # As of … holds …
        for m in AS_OF_HOLDS_RE.finditer(s):
            dt = dateparser.parse(m.group("date"), fuzzy=True).date().isoformat()
            eth = parse_float(m.group("eth"))
            if eth is not None:
                events.append((dt, eth, s))

    # Second pass: totals without explicit date → use page date
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
        for pat in (TOTAL_HOLDINGS_RE, ETH_HOLDINGS_PHRASE_RE, HOLDS_INLINE_RE, PURCHASE_RE):
            m = pat.search(s)
            if m:
                eth = parse_float(m.group("eth"))
                if eth is not None:
                    events.append((dt_iso, eth, s))
                    break

    return events

def crawl_gamesquare_eth_holdings() -> pd.DataFrame:
    """Main function to crawl Gamesquare and extract ETH holdings"""
    logger.info("Starting Gamesquare ETH holdings crawler")
    
    # Setup Selenium
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-logging")
    opts.add_argument("--log-level=3")
    opts.add_argument("--window-size=1200,2000")
    driver = webdriver.Chrome(options=opts)
    
    try:
        # Extract all news URLs
        news_urls = extract_news_urls_with_driver(driver)
        logger.info(f"Found {len(news_urls)} news URLs")
        
        # Parse each URL for ETH holdings
        all_holdings = []
        for i, url in enumerate(news_urls, 1):
            logger.info(f"Processing {i}/{len(news_urls)}: {url}")
            holdings = extract_eth_holdings_from_page(driver, url)
            all_holdings.extend(holdings)
            time.sleep(SLEEP_TIME)
        
        if not all_holdings:
            logger.info("No ETH holdings found")
            return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings"])
        
        # Create DataFrame
        df = pd.DataFrame(all_holdings)
        
        # Normalize: prefer the largest ETH per exact date
        if not df.empty:
            df = (df.sort_values(["date", "total_holdings"])
                    .groupby(["date", "ticker", "asset"], as_index=False)
                    .last())
        
        logger.info(f"Found {len(df)} ETH holdings records")
        return df
        
    except Exception as e:
        logger.error(f"Error crawling Gamesquare: {e}")
        return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings"])
    finally:
        driver.quit()

def extract_news_urls_with_driver(driver: webdriver.Chrome) -> List[str]:
    """Extract all news URLs using existing Selenium driver"""
    driver.get(BASE_URL)
    
    # Wait for news links to load
    wait = WebDriverWait(driver, 20)
    wait.until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, 'a[href*="/news/news-details/"]'))
    )

    # Collect all unique URLs
    seen = set()
    stable_rounds = 0
    last_count = 0

    for _ in range(20):  # safety cap
        anchors = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/news/news-details/"]')
        for a in anchors:
            href = a.get_attribute("href") or ""
            if href:
                href = urljoin(DOMAIN, href)
                seen.add(href)

        if len(seen) == last_count:
            stable_rounds += 1
            if stable_rounds >= 2:
                break
        else:
            stable_rounds = 0
            last_count = len(seen)

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)

    return sorted(seen)

def main():
    """Test function"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    df = crawl_gamesquare_eth_holdings()
    
    if not df.empty:
        print("\n=== ETH Holdings Found ===")
        with pd.option_context("display.max_rows", 200, "display.width", 120):
            print(df)
    else:
        print("No ETH holdings found!")

if __name__ == "__main__":
    main()
