#!/usr/bin/env python3
# scripts/run_outstanding.py
from __future__ import annotations
import argparse
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from datetime import datetime, timezone

import pandas as pd

from ..core.fetch import resolve_cik, fetch_filings, list_filing_files, get
from ..core.select import candidate_html_urls, Mode
from ..core.extract import html_to_blocks
from ..core.types import DealHit
from ..modes import outstanding as mode_impl


ALLOWED_FORMS = ("10-Q", "10-Q/A", "10-K", "10-K/A")

# --- reportDate in-document patterns ---
MONTHS_RX = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATE_RX   = re.compile(rf"{MONTHS_RX}\s+\d{{1,2}},\s+\d{{4}}", re.I)
RX_REPORT_DATE = re.compile(
    rf"(?:for\s+the\s+(?:quarterly\s+)?period\s+ended|for\s+the\s+fiscal\s+year\s+ended)\s+(?P<date>{DATE_RX.pattern})",
    re.I,
)

def _to_iso_date(date_text: Optional[str]) -> Optional[str]:
    if not date_text:
        return None
    s = re.sub(r"\s+", " ", date_text.strip())
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except Exception:
            pass
    return None

def _form_matches(f: str) -> bool:
    fu = (f or "").upper()
    return any(fu == base or fu.startswith(base + "/") for base in ALLOWED_FORMS)

def _has_payload(rec: Dict[str, Any]) -> bool:
    # Only require that we found a numeric outstanding_shares
    v = rec.get("outstanding_shares", None)
    try:
        return v is not None and pd.notna(v)
    except Exception:
        return False
def _expand_cover_blocks(blocks: List[str], max_prim: int = 15) -> List[str]:
    """
    Returns a list that includes original blocks plus bigrams/trigrams
    of the first ~cover-page region so patterns spanning HTML splits still match.
    """
    # normalize NBSPs just in case
    norm = [b.replace("\u00a0", " ") for b in blocks]
    n = min(len(norm), max_prim)

    out = []
    # originals
    out.extend(norm[:n])
    # bigrams
    for i in range(n - 1):
        out.append(norm[i] + " " + norm[i + 1])
    # trigrams
    for i in range(n - 2):
        out.append(norm[i] + " " + norm[i + 1] + " " + norm[i + 2])
    # include rest (unchanged)
    out.extend(norm[n:])
    return out

def _extract_report_date_from_blocks(blocks: List[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Returns (report_date_text, report_date_iso) if a 'period ended' line is found.
    Checks early blocks first (cover page usually near the top).
    """
    # prioritize first ~10 blocks
    scan = blocks[:10] + blocks[10:]
    for b in scan:
        m = RX_REPORT_DATE.search(b)
        if m:
            dtxt = m.group("date")
            return dtxt, _to_iso_date(dtxt)
    # fallback: first date in doc if pattern not found (optional)
    for b in scan:
        m = DATE_RX.search(b)
        if m:
            dtxt = m.group(0)
            return dtxt, _to_iso_date(dtxt)
    return None, None

def build_for_ticker(
    ticker: str,
    year: int,
    year_by: str,
    limit: int,
    max_docs: int,
    max_snips: int,
    since_hours: int | None = 24,
    use_acceptance: bool = True,
) -> pd.DataFrame:
    """
    Build a dataframe of outstanding-shares hits for a single ticker.
    If since_hours > 0, scans only filings accepted within the last N hours (UTC).
    Otherwise, falls back to year-based filtering using `year_by` (filingdate|accession).
    """
    try:
        cik = resolve_cik(ticker)
    except Exception as e:
        print(f"[{ticker}] CIK lookup failed: {e}")
        return pd.DataFrame()

    filings = fetch_filings(cik)
    if filings.empty:
        print(f"[{ticker}] No filings returned.")
        return pd.DataFrame()

    # Ensure datetime dtypes
    if "acceptanceDateTime" in filings.columns:
        filings["acceptanceDateTime"] = pd.to_datetime(
            filings["acceptanceDateTime"], utc=True, errors="coerce"
        )
    if "filingDate" in filings.columns:
        filings["filingDate"] = pd.to_datetime(
            filings["filingDate"], utc=True, errors="coerce"
        )

    # ---- Recent-window filter (default: last 24h) ----
    if since_hours and since_hours > 0:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=since_hours)
        if use_acceptance and "acceptanceDateTime" in filings.columns:
            mask_time = filings["acceptanceDateTime"] >= cutoff
        else:
            # fallback to date-level filter on filingDate
            # normalize cutoff to date boundary for comparison with date-only grain
            mask_time = filings["filingDate"] >= cutoff.normalize()
        time_info = f"since {cutoff.isoformat()}"
    else:
        # ---- Year-based filter (original behavior) ----
        if year_by == "accession":
            mask_year = filings["acc_year"] == year
        else:
            mask_year = filings["filingDate"].dt.year.eq(year)
        mask_time = mask_year
        time_info = f"year={year} (by {year_by})"

    # Form filter + time filter
    subset = filings[mask_time & filings["form"].apply(_form_matches)].copy()

    # Sort newest last, then take tail(limit)
    subset = subset.sort_values(
        ["filingDate", "acceptanceDateTime"], na_position="last"
    ).tail(limit)

    if subset.empty:
        print(f"[{ticker}] No allowed forms in window ({time_info}).")
        return pd.DataFrame()

    out_rows: List[DealHit] = []

    for _, r in subset.iterrows():
        acc = r.get("accessionNumber")
        if not acc:
            continue

        prim = (r.get("primaryDocument") or "")
        form = str(r.get("form") or "").upper()
        fdt = r.get("filingDate")

        try:
            files = list_filing_files(cik, acc)
            # Mode.COVER would be ideal; Mode.WARRANTS returns the primary doc as well.
            cand = candidate_html_urls(
                files,
                primary_doc=prim,
                filing_form=form,
                mode=Mode.WARRANTS,
                max_docs=max_docs,
            )
        except Exception as e:
            print(f"[{ticker}] Skipping {acc}: index.json error: {e}")
            continue

        report_date_text = report_date_iso = None
        per_filing_hits: List[DealHit] = []

        for url, ex_hint in cand:
            try:
                html_bytes = get(url).content
                blocks = html_to_blocks(html_bytes)
                # Expand early blocks so cover-page lines split by HTML still match
                scan_blocks = _expand_cover_blocks(blocks)
            except Exception as e:
                print(f"[{ticker}] Error fetching/parsing {url}: {e}")
                continue

            # compute reportDate once (first doc that yields it wins)
            if report_date_text is None:
                report_date_text, report_date_iso = _extract_report_date_from_blocks(blocks)

            # classify blocks for outstanding shares
            hits: List[DealHit] = []
            for b in scan_blocks:
                for h in mode_impl.classify_block(b, form):
                    h.update({
                        "ticker": ticker,
                        "filingDate": fdt,
                        "form": form,
                        "accessionNumber": acc,
                        "source_url": url,
                        "exhibit_hint": (ex_hint or "").lower(),
                        # attach reportDate found in-document
                        "report_date_text": report_date_text,
                        "report_date_iso": report_date_iso,
                    })
                    hits.append(h)

            if hits:
                hits = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)[:max_snips]
                per_filing_hits.extend(hits)

        for h in per_filing_hits:
            if _has_payload(h):
                out_rows.append(h)

    if not out_rows:
        return pd.DataFrame()

    cols = [
        "ticker","filingDate","form","accessionNumber","event_type_final",
        "outstanding_shares_text","outstanding_shares",
        "as_of_date_text","as_of_date_iso",
        "report_date_text","report_date_iso",
        "source_url","exhibit_hint","snippet","score",
    ]
    df = pd.DataFrame(out_rows).sort_values(
        ["ticker","filingDate","accessionNumber","score"],
        ascending=[True, True, True, False]
    ).reset_index(drop=True)

    df = df[[c for c in cols if c in df.columns]]
    return df


def main():
    ap = argparse.ArgumentParser(description="Outstanding shares extractor (10-Q / 10-K cover page).")
    ap.add_argument("--tickers", default="MSTR,CEP,SMLR,NAKA,BMNR,SBET,ETHZ,BTCS,SQNS,BTBT,DFDV,UPXI",
                    help="Comma-separated tickers, e.g. AAPL,MSFT")
    ap.add_argument("--tickers-pylist", default="", help='Python list literal, e.g. ["AAPL","MSFT"]')
    ap.add_argument("--tickers-file", default="", help="Optional newline-delimited tickers file")

    # Original year-based params (still usable if you disable the recent window)
    ap.add_argument("--year", type=int, default=2025, help="Year filter")
    ap.add_argument("--year-by", choices=["accession","filingdate"], default="filingdate")

    ap.add_argument("--limit", type=int, default=400, help="Max filings per ticker to scan")
    ap.add_argument("--max-docs", type=int, default=4, help="Max HTML docs per filing to fetch")
    ap.add_argument("--max-snippets-per-filing", type=int, default=2, help="Top-N snippets per filing")

    # NEW: lookback window
    ap.add_argument("--since-hours", type=int, default=24,
                    help="Look back this many hours from now (UTC). Use 0 to disable and use --year/--year-by.")
    ap.add_argument("--use-acceptance", action="store_true", default=True,
                    help="Filter by acceptanceDateTime if available (default).")
    ap.add_argument("--no-use-acceptance", dest="use_acceptance", action="store_false")

    ap.add_argument("--outdir", default="data", help="Output directory")
    ap.add_argument("--outfile", default="", help="Overrides default output path")
    args = ap.parse_args()

    # build ticker set
    tickers: List[str] = []
    if args.tickers:
        tickers.extend([t.strip().upper() for t in args.tickers.split(",") if t.strip()])

    if args.tickers_pylist:
        try:
            lst = ast.literal_eval(args.tickers_pylist)
            if isinstance(lst, (list, tuple)):
                tickers.extend([str(t).strip().upper() for t in lst if str(t).strip()])
        except Exception as e:
            raise SystemExit(f"--tickers-pylist parse error: {e}")

    if args.tickers_file:
        p = Path(args.tickers_file)
        if p.exists():
            tickers.extend([ln.strip().upper() for ln in p.read_text().splitlines() if ln.strip()])

    tickers = sorted(set(tickers))
    if not tickers:
        raise SystemExit("No tickers provided. Use --tickers, --tickers-pylist, or --tickers-file.")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Default output name reflects recent-window vs year mode
    if args.outfile:
        outfile = args.outfile
    else:
        if args.since_hours and args.since_hours > 0:
            # include UTC timestamp for clarity
            ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%MZ")
            outfile = str(outdir / f"outstanding_past{args.since_hours}h_{ts}.csv")
        else:
            outfile = str(outdir / f"outstanding_{args.year}.csv")

    all_rows: List[pd.DataFrame] = []
    for t in tickers:
        print(f"\n=== {t} ===")
        df = build_for_ticker(
            t,
            args.year,
            args.year_by,
            args.limit,
            args.max_docs,
            args.max_snippets_per_filing,
            since_hours=args.since_hours,          # <<< NEW
            use_acceptance=args.use_acceptance,    # <<< NEW
        )
        if not df.empty:
            print(f"[{t}] rows: {len(df)}")
            all_rows.append(df)
        else:
            print(f"[{t}] no rows.")

    if not all_rows:
        print("No data for any ticker.")
        return

    combo = pd.concat(all_rows, ignore_index=True)
    combo.to_csv(outfile, index=False)
    print(f"\nSaved -> {Path(outfile).resolve()} (rows={len(combo)}, tickers={len(tickers)})")


if __name__ == "__main__":
    main()
