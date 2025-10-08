#!/usr/bin/env python3
# crawl_upexi_sol_holdings.py
# Returns a DataFrame with columns: date, ticker, asset, total_holdings

import re, time, argparse, datetime
from typing import Optional, Tuple, List, Set, Dict, Any
import requests
from lxml import html
import pandas as pd

BASE_LIST = "https://ir.upexi.com/news-events/press-releases"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; UpexiPressCrawler/1.3; +https://example.org/)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://ir.upexi.com/",
}

# ---- date parsing ----
MONTHS_RX = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
MDY_RX = re.compile(rf"{MONTHS_RX}\s+\d{{1,2}},\s+(\d{{4}})", re.I)
YMD_RX = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")

# ---- numeric helpers ----
def _money_to_float_with_unit(text: str) -> Optional[float]:
    if not text: return None
    m = re.search(r"\$\s*([0-9][\d,]*(?:\.\d+)?)(?:\s*(million|billion))?", text, flags=re.I)
    if not m: return None
    val = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    if unit == "million": val *= 1_000_000
    elif unit == "billion": val *= 1_000_000_000
    return val

def _num_with_unit_to_float(text: str) -> Optional[float]:
    if not text: return None
    m = re.search(r"([0-9][\d,]*(?:\.\d+)?)(?:\s*(million|billion))?", text, flags=re.I)
    if not m: return None
    val = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    if unit == "million": val *= 1_000_000
    elif unit == "billion": val *= 1_000_000_000
    return val
def _mdy_to_year_and_iso(m):
    s = m.group(0)
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            d = datetime.datetime.strptime(s, fmt).date()
            return d.year, d.isoformat()
        except ValueError:
            pass
    # last-resort fallback if the regex had a year group
    try:
        return int(m.group(1)), None
    except Exception:
        return None, None

# ---- SOL patterns ----
SOL_UNIT  = r"(?:SOL|Solana)"
NUMW      = r"[0-9][\d,]*(?:\.\d+)?(?:\s*(?:million|billion))?"
PURCHASE_VERBS = r"(?:purchase|purchased|acquire|acquired|buy|bought|use(?:d)?\s+.*?proceeds\s+to\s+(?:purchase|acquire|buy))"

SOL_COMPLETED_RE = re.compile(
    rf"\b(?:has\s+)?(?:{PURCHASE_VERBS})\s+(?:approximately\s+|about\s+)?({NUMW})\s+(?:{SOL_UNIT})\b", re.I
)
SOL_INTENT_RE = re.compile(
    rf"\b(?:intends?|expect(?:s|ed)?|plan(?:s|ned)?)\s+(?:to\s+)?(?:use|allocate)\s+(?:all|100%|a\s+portion|the\s+net\s+proceeds|proceeds)\s+(?:[^.]{{0,120}}?)\s+(?:to\s+)?(?:purchase|acquire|buy)\s+(?:{SOL_UNIT})\b",
    re.I
)
AS_OF_HOLDS_SOL_RE = re.compile(
    rf"\bAs of\s+({MONTHS_RX}\s+\d{{1,2}},\s+\d{{4}})\b(?:[^.]{{0,200}}?)?\b(?:holds?|held)\s+({NUMW})\s+(?:{SOL_UNIT})\b",
    re.I,
)
SOL_HOLDINGS_OF_RE = re.compile(
    rf"\b(?:SOL(?:ana)?(?:\s+and\s+SOL\s+equivalents)?\s+holdings?|holdings\s+of\s+SOL)\s+(?:of|total(?:ing)?\s+of|total(?:ing)?:?)\s+({NUMW})\s+(?:{SOL_UNIT})\b",
    re.I,
)
NOW_HOLDS_SOL_RE = re.compile(
    rf"\b(?:now\s+holds|holds|now\s+totals?|totals?)\s+({NUMW})\s+(?:{SOL_UNIT})\b", re.I
)
EXCEEDS_SOL_RE = re.compile(
    rf"\b(?:SOL(?:ana)?\s+holdings?)\s+(?:exceed|exceeds|exceeding)\s+({NUMW})\s+(?:{SOL_UNIT})\b", re.I
)

# ---- fetch helpers ----
def fetch(url: str, session: Optional[requests.Session] = None, timeout: int = 30) -> str:
    s = session or requests.Session()
    r = s.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text

def list_links(doc: html.HtmlElement) -> List[Tuple[str, str]]:
    items = []
    seen: Set[str] = set()
    links = doc.xpath("//a[contains(@href, '/news-events/press-releases/detail/')]")
    for a in links:
        href = a.get("href") or ""
        if not href: continue
        if not href.startswith("http"):
            href = "https://ir.upexi.com" + href
        if href in seen: continue
        title = " ".join(a.itertext()).strip()
        if title:
            items.append((title, href))
            seen.add(href)
    return items

def parse_release_year_and_date(doc: html.HtmlElement) -> Tuple[Optional[int], Optional[str]]:
    """
    Try to extract (year, ISO date) from a Upexi IR press release page.
    Looks first for a 'Released <Month DD, YYYY>' stamp, then falls back to body text.
    Returns (year, 'YYYY-MM-DD') or (year, None) if the date string can't be parsed fully,
    or (None, None) if nothing is found.
    """
    def _mdy_to_year_and_iso(s: str) -> Tuple[Optional[int], Optional[str]]:
        # Parse full or abbreviated month names without relying on regex group indices
        for fmt in ("%B %d, %Y", "%b %d, %Y"):
            try:
                d = datetime.datetime.strptime(s, fmt).date()
                return d.year, d.isoformat()
            except ValueError:
                pass
        # Fallback: pull just the 4-digit year if present
        ym = re.search(r"\b(20\d{2})\b", s)
        return (int(ym.group(1)), None) if ym else (None, None)

    # 1) Look for "Released <Month DD, YYYY>"
    txts = doc.xpath("//*[contains(translate(., 'RELEASED', 'released'), 'released')]/text()")
    joined = " ".join(t.strip() for t in txts if t and t.strip())
    m = MDY_RX.search(joined)
    if m:
        y, dt = _mdy_to_year_and_iso(m.group(0))
        if y is not None:
            return y, dt

    # 2) Fallback: scan main body text
    main_txt_nodes = doc.xpath("//article//text()") or doc.xpath("//main//text()") or doc.xpath("//body//text()")
    main_txt = " ".join(main_txt_nodes)
    main_txt = re.sub(r"\s+", " ", main_txt)
    m2 = MDY_RX.search(main_txt)
    if m2:
        y, dt = _mdy_to_year_and_iso(m2.group(0))
        if y is not None:
            return y, dt

    # 3) Final fallback: YYYY-MM-DD anywhere in the body
    m3 = YMD_RX.search(main_txt)
    if m3:
        return int(m3.group(1)), m3.group(0)

    return None, None


def extract_sol_events(text: str) -> List[Dict[str, Any]]:
    """Return a list of events: {'record_type': 'purchase'|'intent'|'holdings', 'SOL': float|None}"""
    t = re.sub(r"\s+", " ", text)
    events: List[Dict[str, Any]] = []

    # Completed purchases
    for m in SOL_COMPLETED_RE.finditer(t):
        sol_units = _num_with_unit_to_float(m.group(1))
        events.append({"record_type": "purchase", "SOL": sol_units})

    # Purchase intents
    for _ in SOL_INTENT_RE.finditer(t):
        events.append({"record_type": "intent", "SOL": None})

    # Holdings snapshots
    holdings_patterns = [AS_OF_HOLDS_SOL_RE, SOL_HOLDINGS_OF_RE, NOW_HOLDS_SOL_RE, EXCEEDS_SOL_RE]
    used: List[Tuple[int,int]] = []
    def overlaps(a: Tuple[int,int], b: Tuple[int,int]) -> bool:
        return not (a[1] <= b[0] or a[0] >= b[1])

    for rx in holdings_patterns:
        for m in rx.finditer(t):
            span = (m.start(), m.end())
            if any(overlaps(span, u) for u in used):
                continue
            groups = [g for g in m.groups() if g]
            sol_s = None
            for g in groups[::-1]:
                if re.search(r"\d", g):
                    sol_s = g; break
            sol_units = _num_with_unit_to_float(sol_s) if sol_s else None
            events.append({"record_type": "holdings", "SOL": sol_units})
            used.append(span)

    # De-dupe rough (type, SOL)
    seen = set()
    uniq = []
    for e in events:
        key = (e["record_type"], None if e["SOL"] is None else round(float(e["SOL"]), 8))
        if key in seen: continue
        seen.add(key); uniq.append(e)
    return uniq

def crawl_holdings_df(
    ticker: str = "UPXI",
    min_year: int = 2025,
    start_page: int = 1,
    max_pages: int = 50,
    sleep_s: float = 0.4,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Crawl Upexi IR press releases and return only holdings snapshots as:
    columns = ['date', 'ticker', 'asset', 'total_holdings']
    """
    s = session or requests.Session()
    page = start_page
    stop = False
    rows: List[Dict[str, Any]] = []

    while not stop and page <= max_pages:
        list_url = f"{BASE_LIST}?page={page}"
        try:
            listing_html = fetch(list_url, s)
        except Exception:
            break

        doc = html.fromstring(listing_html)
        pairs = list_links(doc)
        if not pairs:
            break

        for _, href in pairs:
            try:
                detail_html = fetch(href, s)
            except Exception:
                continue

            detail_doc = html.fromstring(detail_html)
            year, date_iso = parse_release_year_and_date(detail_doc)

            if year is not None and year < min_year:
                stop = True
                break
            if year != min_year:
                continue

            body_text = " ".join(detail_doc.xpath("//article//text()") or detail_doc.xpath("//main//text()") or detail_doc.xpath("//body//text()"))
            body_text = re.sub(r"\s+", " ", body_text)

            events = extract_sol_events(body_text)

            # Only keep 'holdings' events with a numeric amount
            for ev in events:
                if ev.get("record_type") == "holdings" and ev.get("SOL") is not None:
                    rows.append({
                        "date": date_iso,
                        "ticker": ticker,
                        "asset": "SOL",
                        "total_holdings": float(ev["SOL"]),
                    })

        page += 1
        time.sleep(sleep_s)

    df = pd.DataFrame(rows, columns=["date", "ticker", "asset", "total_holdings"])

    # Optional: if multiple holdings are mentioned in the same press release date,
    # keep the max (often phrased multiple ways). Comment out if you prefer raw rows.
    if not df.empty:
        df = (df.sort_values(["date", "total_holdings"])
                .groupby(["date", "ticker", "asset"], as_index=False)
                .last())

    return df

# Optional CLI (prints the DataFrame; does NOT save a CSV)
def main():
    ap = argparse.ArgumentParser(description="Return holdings snapshots as (date, ticker, asset, total_holdings).")
    ap.add_argument("--ticker", default="UPXI")
    ap.add_argument("--min-year", type=int, default=2025)
    ap.add_argument("--start-page", type=int, default=1)
    ap.add_argument("--max-pages", type=int, default=50)
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()

    df = crawl_holdings_df(
        ticker=args.ticker,
        min_year=args.min_year,
        start_page=args.start_page,
        max_pages=args.max_pages,
        sleep_s=args.sleep,
    )
    with pd.option_context("display.max_rows", 200, "display.width", 120):
        print(df)

if __name__ == "__main__":
    main()
