#!/usr/bin/env python3
# hsdt_holdings_df.py
# Returns a pandas DataFrame: date, ticker, asset, total_holdings (from HSDT news/blog posts)

import re
import time
import logging
from typing import List, Dict, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta

import requests
import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger("hsdt_holdings")  # configure in your main

BASE = "https://www.solanacompany.co"
NEWS_URL = f"{BASE}/news"

UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

TIMEOUT = 30
PAGE_SLEEP = 0.5

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(UA_HEADERS)
    return s

def _fetch(url: str, timeout: int = TIMEOUT) -> Optional[BeautifulSoup]:
    """Fetch URL and return BeautifulSoup object."""
    try:
        print(f"Fetching: {url}")
        r = _session().get(url, timeout=timeout)
        r.raise_for_status()
        return BeautifulSoup(r.text, 'html.parser')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def get_news_links() -> List[Dict[str, str]]:
    """
    Scrape the HSDT news page and extract all news article links.
    Returns list of dicts with 'title', 'url', 'date' keys.
    """
    print(f"Scraping news links from: {NEWS_URL}")
    
    soup = _fetch(NEWS_URL)
    if not soup:
        print("Failed to fetch news page")
        return []
    
    links = []
    
    # Look for common news/article link patterns
    # Try different selectors that might contain news links
    
    # Method 1: Look for <a> tags with href containing news/article patterns
    article_links = soup.find_all('a', href=True)
    
    print(f"Found {len(article_links)} total links on page")
    
    print("\nAnalyzing all links found:")
    for i, link in enumerate(article_links, 1):
        href = link.get('href', '')
        text = link.get_text(strip=True)
        
        print(f"{i:2d}. Text: '{text[:40]}...' | Href: '{href}'")
        
        # Skip empty href, but allow empty text (some links have no visible text)
        if not href:
            print(f"    -> Skipped: empty href")
            continue
            
        # Skip anchors and non-article links, but allow external press releases
        if href.startswith('#') or \
           href.startswith('mailto:') or \
           href.startswith('tel:'):
            print(f"    -> Skipped: anchor/mailto/tel")
            continue
        
        # Make relative URLs absolute, or use external URLs as-is
        if href.startswith('/'):
            full_url = urljoin(BASE, href)
        elif href.startswith('http'):
            full_url = href  # External URLs (press releases)
        else:
            print(f"    -> Skipped: not a valid URL")
            continue
        
        print(f"    -> Full URL: {full_url}")
        
        # Look for article-like URLs or SOL-related content
        article_patterns = ['news', 'article', 'post', 'press', 'release', 'update', 'announcement']
        sol_patterns = ['sol', 'solana', 'treasury', 'purchase', 'acquisition', 'holdings']
        
        # Check if it's an article-like URL or contains SOL-related keywords
        is_article_url = any(pattern in href.lower() for pattern in article_patterns)
        is_sol_content = any(pattern in text.lower() for pattern in sol_patterns)
        is_sol_url = any(pattern in href.lower() for pattern in sol_patterns)
        
        if is_article_url or is_sol_content or is_sol_url:
            link_info = {
                'title': text,
                'url': full_url,
                'date': None  # We'll try to extract dates later
            }
            links.append(link_info)
            match_type = 'article' if is_article_url else ('SOL URL' if is_sol_url else 'SOL content')
            print(f"    -> MATCHED: Found {match_type} link")
        else:
            print(f"    -> Skipped: no article patterns in URL or SOL content in text")
    
    # Method 2: Look for specific news containers/classes
    # Try common news container selectors
    news_selectors = [
        '.news-item',
        '.article-item', 
        '.post-item',
        '.news-list',
        '.article-list',
        '.post-list',
        '[class*="news"]',
        '[class*="article"]',
        '[class*="post"]'
    ]
    
    for selector in news_selectors:
        containers = soup.select(selector)
        if containers:
            print(f"Found {len(containers)} containers with selector: {selector}")
            for container in containers:
                # Look for links within these containers
                container_links = container.find_all('a', href=True)
                for link in container_links:
                    href = link.get('href', '')
                    text = link.get_text(strip=True)
                    
                    if href and text and len(text) > 10:
                        if href.startswith('/'):
                            full_url = urljoin(BASE, href)
                        elif href.startswith(BASE):
                            full_url = href
                        else:
                            continue
                            
                        # Check if we already have this link
                        if not any(existing['url'] == full_url for existing in links):
                            link_info = {
                                'title': text,
                                'url': full_url,
                                'date': None
                            }
                            links.append(link_info)
                            print(f"Found container link: {text[:50]}... -> {full_url}")
    
    print(f"\nTotal unique article links found: {len(links)}")
    
    # Print all links for inspection
    print("\n" + "="*80)
    print("ALL FOUND LINKS:")
    print("="*80)
    for i, link in enumerate(links, 1):
        print(f"{i:2d}. {link['title'][:60]}")
        print(f"    URL: {link['url']}")
        print()
    
    return links

def _num_with_unit_to_float(text: str) -> Optional[float]:
    """
    Convert text with numbers and units (million/billion) to float.
    Borrowed from UPXI crawler for better number parsing.
    """
    if not text:
        return None
    m = re.search(r"([0-9][\d,]*(?:\.\d+)?)(?:\s*(million|billion))?", text, flags=re.I)
    if not m:
        return None
    val = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    if unit == "million":
        val *= 1_000_000
    elif unit == "billion":
        val *= 1_000_000_000
    return val

def extract_sol_holdings_from_text(text: str) -> Optional[float]:
    """
    Extract SOL holdings amount from press release text.
    Enhanced with patterns from UPXI crawler for better coverage.
    """
    if not text:
        return None
    
    # Clean up the text
    text = re.sub(r"\s+", " ", text)
    
    # Enhanced patterns from UPXI crawler
    SOL_UNIT = r"(?:SOL|Solana)"
    NUMW = r"[0-9][\d,]*(?:\.\d+)?(?:\s*(?:million|billion))?"
    PURCHASE_VERBS = r"(?:purchase|purchased|acquire|acquired|buy|bought|use(?:d)?\s+.*?proceeds\s+to\s+(?:purchase|acquire|buy))"
    
    # Multiple comprehensive patterns
    patterns = [
        # HSDT-specific patterns
        rf'holds\s+over\s+({NUMW})\s+{SOL_UNIT}',
        rf'now\s+holds\s+({NUMW})\s+{SOL_UNIT}',
        rf'holds\s+({NUMW})\s+{SOL_UNIT}',
        rf'holding\s+({NUMW})\s+{SOL_UNIT}',
        
        # UPXI-inspired patterns
        rf'\b(?:has\s+)?(?:{PURCHASE_VERBS})\s+(?:approximately\s+|about\s+)?({NUMW})\s+{SOL_UNIT}\b',
        rf'\b(?:now\s+holds|holds|now\s+totals?|totals?)\s+({NUMW})\s+{SOL_UNIT}\b',
        rf'\b(?:SOL(?:ana)?\s+holdings?)\s+(?:exceed|exceeds|exceeding)\s+({NUMW})\s+{SOL_UNIT}\b',
        rf'\b(?:SOL(?:ana)?(?:\s+and\s+SOL\s+equivalents)?\s+holdings?|holdings\s+of\s+SOL)\s+(?:of|total(?:ing)?\s+of|total(?:ing)?:?)\s+({NUMW})\s+{SOL_UNIT}\b',
        rf'\bAs of\s+[^.]{{0,50}}?\b(?:holds?|held)\s+({NUMW})\s+{SOL_UNIT}\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                # Find the first group that contains a number
                for group in match.groups():
                    if group and re.search(r'\d', group):
                        amount = _num_with_unit_to_float(group)
                        if amount:
                            return amount
            except (ValueError, AttributeError):
                continue
    
    return None

def extract_date_from_text(text: str) -> Optional[str]:
    """
    Extract date from press release text.
    Enhanced with UPXI patterns for better date parsing.
    """
    if not text:
        return None
    
    # Enhanced date patterns from UPXI crawler
    MONTHS_RX = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    
    date_patterns = [
        # UPXI-style patterns
        rf'({MONTHS_RX}\s+\d{{1,2}},\s+\d{{4}})',  # "Sept. 22, 2025" or "September 22, 2025"
        rf'(\d{{1,2}}\s+{MONTHS_RX}\s+\d{{4}})',     # "22 Sept. 2025"
        r'(\d{4}-\d{2}-\d{2})',                      # "2025-09-22"
        # Additional patterns
        r'([A-Za-z]+\.?\s+\d{1,2},?\s+\d{4})',      # Fallback for other formats
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            try:
                # Try multiple date formats
                from dateutil import parser
                parsed_date = parser.parse(date_str)
                return parsed_date.date().isoformat()
            except:
                # Try manual parsing for common formats
                try:
                    import datetime
                    # Try "Month DD, YYYY" format
                    for fmt in ("%B %d, %Y", "%b %d, %Y"):
                        try:
                            d = datetime.datetime.strptime(date_str, fmt).date()
                            return d.isoformat()
                        except ValueError:
                            continue
                except:
                    continue
    
    return None

def extract_holdings_from_press_release(url: str) -> Dict[str, Optional[float]]:
    """
    Extract SOL holdings data from a press release URL.
    Returns dict with holdings amount and date.
    """
    print(f"Extracting holdings from: {url}")
    
    soup = _fetch(url)
    if not soup:
        print(f"Failed to fetch press release: {url}")
        return {"holdings": None, "date": None}
    
    # Get all text content
    text = soup.get_text()
    
    # Extract SOL holdings
    holdings = extract_sol_holdings_from_text(text)
    
    # Extract date
    date = extract_date_from_text(text)
    
    print(f"  -> Holdings: {holdings} SOL")
    print(f"  -> Date: {date}")
    
    return {"holdings": holdings, "date": date}

def get_hsdt_holdings_df(hours: int = 24, ticker: str = "HSDT", asset: str = "SOL") -> pd.DataFrame:
    """
    Main function to get HSDT holdings data from press releases.
    Returns DataFrame in same format as UPXI crawler: ['date', 'ticker', 'asset', 'total_holdings']
    """
    print(f"Getting HSDT holdings data for last {hours} hours")
    
    # Get all news links
    links = get_news_links()
    
    if not links:
        print("No news links found")
        return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings"])
    
    # Filter to only press release links (exclude social media, etc.)
    press_release_links = []
    for link in links:
        url = link['url']
        if any(domain in url for domain in ['globenewswire.com', 'prnewswire.com', 'businesswire.com']):
            press_release_links.append(link)
    
    print(f"Found {len(press_release_links)} press release links to process")
    
    rows = []
    for link in press_release_links:
        print(f"\nProcessing: {link['title'][:50]}...")
        
        # Extract holdings data from the press release
        data = extract_holdings_from_press_release(link['url'])
        
        if data['holdings'] is not None:
            # Use same format as UPXI crawler
            row = {
                "date": data['date'] or datetime.now().date().isoformat(),
                "ticker": ticker,
                "asset": asset,
                "total_holdings": float(data['holdings']),
            }
            rows.append(row)
            print(f"  -> SUCCESS: Found {data['holdings']} SOL")
        else:
            print(f"  -> No holdings data found")
        
        # Be polite to the servers
        time.sleep(PAGE_SLEEP)
    
    if not rows:
        print("No holdings data found in any press releases")
        return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings"])
    
    df = pd.DataFrame(rows, columns=["date", "ticker", "asset", "total_holdings"])
    print(f"\nCreated DataFrame with {len(df)} holdings records")
    
    # Deduplicate by date and keep the highest holdings amount
    # (same approach as UPXI crawler - often multiple ways to phrase the same holdings)
    if not df.empty:
        df = (df.sort_values(["date", "total_holdings"])
                .groupby(["date", "ticker", "asset"], as_index=False)
                .last())
        print(f"After deduplication: {len(df)} unique holdings records")
    
    return df

if __name__ == "__main__":
    """
    Test the HSDT crawler - returns DataFrame in same format as UPXI crawler.
    """
    print("Testing HSDT holdings crawler...")
    
    # Test holdings extraction
    df = get_hsdt_holdings_df()
    
    # Display results in same format as UPXI crawler
    with pd.option_context("display.max_rows", 200, "display.width", 120):
        print("\nHSDT Holdings DataFrame:")
        print(df)
