#!/usr/bin/env python3
# nakamoto_holdings_df.py
# Returns a pandas DataFrame: date, ticker, asset, total_holdings (from recent Nakamoto posts)

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
import re
import logging

import requests
import pandas as pd
from lxml import etree

logger = logging.getLogger("nakamoto_holdings")  # configure in your main if you want console output

BASE = "https://nakamoto.com"
UPDATES_URL = f"{BASE}/updates"

UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

COMPLETED_VERBS = [
    "acquired", "acquires", "has acquired",
    "purchased", "has purchased", "bought",
    "received", "transferred", "obtained",
    "completed purchase", "completed acquisition",
    "executed", "settled",
    "now holds", "holds ",
]
FUTURE_SIGNALS = [
    "plans to", "intends to", "will purchase", "will acquire",
    "proposed", "proposal", "upcoming", "expected to",
]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(UA_HEADERS)
    return s

def _fetch(url: str, timeout: int = 20) -> Optional[Tuple[etree._Element, str]]:
    try:
        r = _session().get(url, timeout=timeout)
        r.raise_for_status()
        parser = etree.HTMLParser(recover=True, huge_tree=True)
        root = etree.HTML(r.content, parser=parser)
        if root is None:
            return None
        texts = root.xpath("//text()[normalize-space() and not(ancestor::script or ancestor::style or ancestor::noscript)]")
        text = " ".join(t.strip() for t in texts if t and t.strip())
        text = re.sub(r"\s+", " ", text).strip()
        return root, text
    except requests.RequestException:
        return None

def _extract_links(root: etree._Element, base_url: str) -> List[str]:
    hrefs: List[str] = []
    for href in root.xpath("//a[@href]/@href"):
        href = (href or "").strip()
        if not href or href.startswith("#"):
            continue
        abs_url = urljoin(base_url, href)
        pu = urlparse(abs_url)
        if pu.scheme in ("http", "https") and "nakamoto.com" in pu.netloc:
            hrefs.append(abs_url)
    return list(dict.fromkeys(hrefs))

def _parse_date(date_str: str) -> Optional[str]:
    s = (date_str or "").strip()
    try:
        fmts = ["%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m.%d.%y", "%m.%d.%Y", "%m/%d/%y", "%m/%d/%Y"]
        for fmt in fmts:
            try:
                return datetime.strptime(s, fmt).date().isoformat()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).date().isoformat()
        except Exception:
            return s if s else None
    except Exception:
        return None

def _dom_date(root: etree._Element, fallback_text: str) -> Optional[str]:
    for cand in root.xpath("//time/@datetime"):
        iso = _parse_date(str(cand))
        if iso:
            return iso
    for cand in root.xpath("//time/text()"):
        iso = _parse_date(str(cand))
        if iso:
            return iso
    m = re.search(r"(\d{4}-\d{2}-\d{2})", fallback_text)
    if m:
        return _parse_date(m.group(1))
    return None

def _is_recent(date_str: str, hours: int) -> bool:
    s = (date_str or "").strip()
    if not s:
        return False
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return (datetime.utcnow() - dt) <= timedelta(hours=hours)
    except Exception:
        pass
    try:
        d = datetime.strptime(s[:10], "%Y-%m-%d").date()
        threshold = (datetime.utcnow() - timedelta(hours=hours)).date()
        return d >= threshold
    except Exception:
        return False

def _is_completed_btc(text: str) -> bool:
    t = text.lower()
    has_btc = ("bitcoin" in t) or ("btc" in t)
    has_completed = any(v in t for v in COMPLETED_VERBS)
    is_future = any(v in t for v in FUTURE_SIGNALS)
    return has_btc and has_completed and not is_future

def _extract_btc_amounts(text: str) -> List[float]:
    out: List[float] = []
    for m in re.finditer(r"\b([0-9][\d,]*(?:\.\d+)?)\s*(?:btc|bitcoin)\b", text, re.I):
        out.append(float(m.group(1).replace(",", "")))
    return out


def get_nakamoto_holdings_df(hours: int = 24, ticker: str = "NAKA", asset: str = "BTC") -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      - date (YYYY-MM-DD)
      - ticker (upper)
      - asset  (upper)
      - total_holdings (float)
    Logs an INFO message if no recent hits are found.
    """
    t = ticker.upper()
    a = asset.upper()

    fetched = _fetch(UPDATES_URL)
    if not fetched:
        logger.info("Nakamoto: failed to fetch updates page — no hits.")
        return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings"])
    root, page_text = fetched

    article_links = [u for u in _extract_links(root, BASE) if "/updates/" in u]
    if not article_links:
        logger.info("Nakamoto: no article links discovered on updates page — no hits.")
        return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings"])

    recent_urls: List[str] = []
    cache: Dict[str, Tuple[etree._Element, str, Optional[str]]] = {}
    for url in article_links:
        got = _fetch(url)
        if not got:
            continue
        art_root, art_text = got
        date_iso = _dom_date(art_root, art_text) or ""
        cache[url] = (art_root, art_text, date_iso)
        if date_iso and _is_recent(date_iso, hours=hours):
            recent_urls.append(url)

    if not recent_urls:
        logger.info(f"Nakamoto: no posts within the last {hours}h — no hits.")
        return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings"])

    rows: List[Dict[str, Any]] = []
    for url in recent_urls:
        art_root, art_text, date_iso = cache[url]
        if not date_iso or not _is_completed_btc(art_text):
            continue
        amts = _extract_btc_amounts(art_text)
        if not amts:
            continue
        rows.append({"date": date_iso, "btc": sum(amts)})

    if not rows:
        logger.info("Nakamoto: recent posts found but none matched completed BTC transactions — no hits.")
        return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("string")
    df = df.dropna(subset=["date"])
    daily = df.groupby("date", as_index=False)["btc"].sum().sort_values("date")

    out = pd.DataFrame({
        "date": daily["date"].astype("string"),
        "ticker": t,
        "asset": a,
        "total_holdings": daily["btc"].astype(float),
    })
    return out


if __name__ == "__main__":
    # Optional quick test: enable console logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(get_nakamoto_holdings_df().head())
