# sec_mstr_btc_timeline_ixbrl.py
# Pull filings (default 8-K) from --from-date onward, parse BTC table from iXBRL/HTML, output timeline CSV + NDJSON.
# pip install requests lxml pandas

import argparse, json, time, re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
import pandas as pd
from lxml import etree
from urllib.parse import urljoin

# =========================
# CONFIG
# =========================
USER_AGENT = "Ally Zach <ally@panteracapital.com>"  # <-- REQUIRED: real name + email (no brackets)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

BASE_SUBMISSIONS = "https://data.sec.gov/submissions"
TICKER_MAP_URL   = "https://www.sec.gov/files/company_tickers.json"

NS = {
    "x":  "http://www.w3.org/1999/xhtml",
    "ix": "http://www.xbrl.org/2013/inlineXBRL",
}
AS_OF_RE = re.compile(r"\b(as of|as at)\s+([A-Z][a-z]+ \d{1,2}, \d{4}|\d{4}-\d{2}-\d{2})\b", re.I)

# =========================
# HTTP helper (polite)
# =========================
def get(url: str) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 429:
        time.sleep(1.5)
        resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(0.25)
    return resp

# =========================
# Ticker -> CIK mapping
# =========================
def resolve_cik(ticker: str) -> str:
    data = get(TICKER_MAP_URL).json()
    for item in data.values():
        if str(item.get("ticker", "")).upper() == ticker.upper():
            return f"{int(item['cik_str']):010d}"
    raise RuntimeError(f"Ticker not found in SEC map: {ticker}")

# =========================
# Filings list
# =========================
def fetch_filings(cik: str) -> pd.DataFrame:
    url = f"{BASE_SUBMISSIONS}/CIK{cik}.json"
    j = get(url).json()
    recent = j.get("filings", {}).get("recent", {})
    df = pd.DataFrame(recent)
    if df.empty:
        return df
    # Parse dates; make acceptanceDateTime naive (UTC->naive) so sorts/comparisons won't error
    if "reportDate" in df.columns:
        df["reportDate"] = pd.to_datetime(df["reportDate"], errors="coerce")
    if "filingDate" in df.columns:
        df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce")
    if "acceptanceDateTime" in df.columns:
        df["acceptanceDateTime"] = pd.to_datetime(df["acceptanceDateTime"], utc=True, errors="coerce").dt.tz_convert(None)
    df["form"] = df["form"].astype(str)
    return df

def cik_no_zeros(cik_10: str) -> str:
    return str(int(cik_10))

def accession_nodash(acc: str) -> str:
    return acc.replace("-", "")

def filing_base_url(cik_10: str, accession: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros(cik_10)}/{accession_nodash(accession)}/"

def list_filing_files(cik_10: str, accession: str) -> pd.DataFrame:
    base = filing_base_url(cik_10, accession)
    j = get(urljoin(base, "index.json")).json()
    df = pd.DataFrame(j.get("directory", {}).get("item", []))
    if df.empty:
        return df
    df["url"] = df["name"].apply(lambda n: urljoin(base, n))
    return df

def is_html_name(name: str) -> bool:
    n = name.lower()
    return n.endswith((".htm", ".html", ".xhtml"))

def candidate_html_urls(files_df: pd.DataFrame, primary_doc: str) -> List[str]:
    if files_df.empty:
        return []
    df = files_df[files_df["name"].apply(is_html_name)].copy()
    if df.empty:
        return []
    prim = (primary_doc or "").lower()
    def rank(name: str) -> Tuple[int,int,int]:
        n = name.lower()
        r0 = 0 if n == prim else 1
        r1 = 0 if ("8-k" in n or "8k" in n or "ex99" in n or "press" in n) else 1
        r2 = 0 if n.endswith((".xhtml",".htm",".html")) else 1
        return (r0, r1, r2)
    df["rank_tuple"] = df["name"].apply(rank)
    df = df.sort_values(["rank_tuple","name"])
    return df["url"].tolist()

# =========================
# iXBRL/HTML table parsing
# =========================
def _clean_text(s: str) -> str:
    return " ".join((s or "").split())

def _join_text(node: etree._Element) -> str:
    return _clean_text("".join(node.itertext()))

def _norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+","",(s or "").lower())

def _money_to_float(s: str) -> Optional[float]:
    if s is None: return None
    m = re.search(r"[-+()]?\$?\s*[\d,]+(?:\.\d+)?", str(s))
    if not m: return None
    raw = m.group(0)
    neg = "(" in raw and ")" in raw
    try:
        v = float(raw.replace("$","").replace(",","").replace("(","").replace(")","").strip())
        return -v if neg else v
    except:
        return None

def _number_to_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s or s in {"-","—"}: return None
    s2 = s.replace(",","")
    try:
        return float(s2)
    except:
        m = re.search(r"\d+(?:\.\d+)?", s2)
        return float(m.group(0)) if m else None

def _parse_tables(root: etree._Element) -> List[List[List[str]]]:
    tables = []
    def collect_tables(xpath_table: str, xpath_tr: str, xpath_cell: str):
        for t in root.xpath(xpath_table, namespaces=NS):
            rows: List[List[str]] = []
            for tr in t.xpath(xpath_tr, namespaces=NS):
                cells = tr.xpath(xpath_cell, namespaces=NS)
                if not cells:
                    continue
                row = [_join_text(c) for c in cells]
                rows.append(row)
            if rows:
                w = max(len(r) for r in rows)
                rows = [r + [""]*(w-len(r)) for r in rows]
                tables.append(rows)
    collect_tables(".//x:table", ".//x:tr", "./x:th | ./x:td")  # XHTML
    if not tables:
        collect_tables(".//table", ".//tr", "./th | ./td")       # plain HTML
    return tables

def _find_btc_header_row(rows: List[List[str]]) -> Optional[int]:
    for i in range(min(6, len(rows))):
        rowtxt = " ".join(rows[i]).lower()
        if "btc" in rowtxt or "bitcoin" in rowtxt:
            return i
    return None

def _map_header_indices(header_row: List[str]) -> Dict[str,int]:
    idx: Dict[str,int] = {}
    seen_avg = 0
    for j, cell in enumerate(header_row):
        n = _norm_token(cell)
        if not n:
            continue
        if "btcacquired" in n:
            idx["period_btc_acquired"] = j
        elif "aggregatepurchaseprice" in n and "inmillions" in n:
            idx["period_agg_purchase_millions"] = j
        elif "aggregatepurchaseprice" in n and "inbillions" in n:
            idx["total_agg_purchase_billions"] = j
        elif "aggregatebtcholdings" in n:
            idx["total_btc_holdings"] = j
        elif "averagepurchaseprice" in n:
            if seen_avg == 0:
                idx["period_avg_purchase_price"] = j
            else:
                idx["total_avg_purchase_price"] = j
            seen_avg += 1
    return idx

def _pick_data_row(rows: List[List[str]], header_idx: int) -> Optional[int]:
    best_i, best_score = None, -1
    for i in range(header_idx+1, len(rows)):
        row = " ".join(rows[i])
        score = sum(ch.isdigit() for ch in row)
        if score > best_score:
            best_score, best_i = score, i
    return best_i

def parse_btc_table_rows(rows: List[List[str]]) -> Optional[Dict[str, Any]]:
    h = _find_btc_header_row(rows)
    if h is None:
        return None
    header = rows[h]
    idx = _map_header_indices(header)
    if not idx:
        return None
    r = _pick_data_row(rows, h)
    if r is None:
        return None
    data = rows[r]
    out: Dict[str, Any] = {}
    if "period_btc_acquired" in idx:
        out["period_btc_acquired"] = _number_to_float(data[idx["period_btc_acquired"]])
    if "period_agg_purchase_millions" in idx:
        v = _money_to_float(data[idx["period_agg_purchase_millions"]])
        out["period_agg_purchase_price_usd"] = v * 1_000_000 if v is not None else None
    if "period_avg_purchase_price" in idx:
        out["period_avg_purchase_price_usd"] = _money_to_float(data[idx["period_avg_purchase_price"]]) or _number_to_float(data[idx["period_avg_purchase_price"]])
    if "total_btc_holdings" in idx:
        out["total_btc_holdings"] = _number_to_float(data[idx["total_btc_holdings"]])
    if "total_agg_purchase_billions" in idx:
        v = _money_to_float(data[idx["total_agg_purchase_billions"]])
        out["total_agg_purchase_price_usd"] = v * 1_000_000_000 if v is not None else None
    if "total_avg_purchase_price" in idx:
        out["total_avg_purchase_price_usd"] = _money_to_float(data[idx["total_avg_purchase_price"]]) or _number_to_float(data[idx["total_avg_purchase_price"]])
    for k in ("period_btc_acquired", "total_btc_holdings"):
        if out.get(k) is not None:
            f = float(out[k]); out[k] = int(f) if abs(f-int(f)) < 1e-9 else f
    return out

def pull_ix_meta(root: etree._Element, full_text: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    get_one = lambda name: root.xpath(f"string(//ix:nonNumeric[@name='{name}'])", namespaces=NS)
    meta["doc_type"] = _clean_text(get_one("dei:DocumentType"))
    meta["doc_period_end_date"] = _clean_text(get_one("dei:DocumentPeriodEndDate"))
    meta["registrant"] = _clean_text(get_one("dei:EntityRegistrantName"))
    meta["cik"] = _clean_text(get_one("dei:EntityCentralIndexKey"))
    m = AS_OF_RE.search(full_text)
    if m:
        meta["as_of"] = m.group(2)
    return meta

def extract_btc_metrics_from_url(url: str) -> Dict[str, Any]:
    html = get(url).text
    parser = etree.XMLParser(recover=True, huge_tree=True)
    try:
        root = etree.fromstring(html.encode("utf-8"), parser=parser)
    except Exception:
        root = etree.HTML(html, parser=parser)
    tables = _parse_tables(root)
    metrics: Optional[Dict[str, Any]] = None
    for rows in tables:
        metrics = parse_btc_table_rows(rows)
        if metrics:
            break
    try:
        all_text = _clean_text(" ".join(root.xpath("//text()", namespaces=NS)))
    except Exception:
        all_text = _clean_text(" ".join(root.itertext()))
    meta = pull_ix_meta(root, all_text)
    rec = {"source_url": url, **meta, **(metrics or {})}
    rec["ok"] = bool(metrics)
    return rec

# =========================
# Driver
# =========================
def main():
    ap = argparse.ArgumentParser(description="Build BTC timeline from SEC iXBRL/HTML tables (2020+ filter).")
    ap.add_argument("--ticker", default="MSTR", help="Ticker to scan (default: MSTR)")
    ap.add_argument("--forms", default="8-K", help="Comma-separated forms (e.g., 8-K,10-Q,10-K). Default: 8-K")
    ap.add_argument("--from-date", dest="from_date", default="2025-01-01",
                    help="Inclusive floor date for filings (ISO, default: 2025-01-01)")
    ap.add_argument("--limit", type=int, default=300, help="Max recent filings to fetch (default: 300)")
    ap.add_argument("--max-docs", type=int, default=4, help="Max HTML docs per filing to try (default: 4)")
    ap.add_argument("--outdir", default="data", help="Output directory (default: data)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)

    cik = resolve_cik(args.ticker)
    print(f"CIK for {args.ticker}: {cik}")

    filings = fetch_filings(cik)
    if filings.empty:
        raise SystemExit("No filings returned.")

    allowed = {f.strip().upper() for f in args.forms.split(",") if f.strip()}
    subset = filings[filings["form"].str.upper().isin(allowed)].copy()

    # ---- Filter by accession number YEAR to avoid tz issues ----
    floor_year = pd.Timestamp(args.from_date).year
    subset["acc_year"] = (
        subset["accessionNumber"].astype(str)
        .str.extract(r"^\d{10}-(\d{2})-")[0]
        .astype("float")
        .apply(lambda y: 2000 + int(y) if pd.notna(y) else pd.NA)
    )
    subset = subset[subset["acc_year"].ge(floor_year)]

    subset = subset.sort_values("filingDate").tail(args.limit)
    if subset.empty:
        raise SystemExit(f"No filings found for forms {sorted(allowed)} on/after {floor_year}.")

    records: List[Dict[str, Any]] = []
    for _, row in subset.iterrows():
        accession = row.get("accessionNumber")
        primary = (row.get("primaryDocument") or "")
        try:
            files = list_filing_files(cik, accession)
        except Exception as e:
            print(f"Skipping {accession}: index.json error: {e}")
            continue
        urls = candidate_html_urls(files, primary)[:args.max_docs]
        got: Optional[Dict[str, Any]] = None
        for u in urls:
            try:
                rec = extract_btc_metrics_from_url(u)
            except Exception as e:
                print(f"  Error parsing {u}: {e}")
                continue
            if rec.get("ok") and any(rec.get(k) is not None for k in [
                "period_btc_acquired","period_agg_purchase_price_usd","period_avg_purchase_price_usd",
                "total_btc_holdings","total_agg_purchase_price_usd","total_avg_purchase_price_usd"
            ]):
                got = rec
                break
        if got:
            got.update({
                "form": row.get("form"),
                "accessionNumber": accession,
                "primaryDocument": primary,
                "filingDate": row.get("filingDate"),
                "reportDate": row.get("reportDate"),
                "acceptanceDateTime": row.get("acceptanceDateTime"),
                "filing_url_base": filing_base_url(cik, accession),
            })
            stamp = got['filingDate'].date() if pd.notna(got['filingDate']) else ''
            print(f"Hit: {stamp} -> {got['source_url']}")
            records.append(got)
        else:
            print(f"No BTC table found for filing {row.get('form')} {accession}")

    if not records:
        print("No BTC tables parsed.")
        return

    df = pd.DataFrame(records)
    df["as_of_dt"] = pd.to_datetime(df.get("as_of"), errors="coerce")
    df = df.sort_values(["as_of_dt","filingDate","acceptanceDateTime"], na_position="last").reset_index(drop=True)

    csv_path = outdir / "mstr_btc_timeline.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV -> {csv_path.resolve()}")

    ndjson_path = outdir / "mstr_btc_timeline.ndjson"
    with ndjson_path.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            rec = {k: (v.isoformat() if isinstance(v, pd.Timestamp) and pd.notna(v) else (None if isinstance(v, pd.Timestamp) else v))
                   for k, v in r.items()}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved NDJSON -> {ndjson_path.resolve()}")

if __name__ == "__main__":
    main()
