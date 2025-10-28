#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd

from app.sec_deals.core.fetch import resolve_cik, fetch_filings

ALLOWED_FORMS = ("8-K", "10-K", "10-Q", "S-1", "S-3", "424B5", "424B3", "424B7")
_SUFFIX_OK = ("A", "MEF", "ASR")  # S-1/A, S-3ASR, S-1MEF, etc.

def _form_matches(form: str) -> bool:
    fu = (form or "").upper().strip()
    for base in ALLOWED_FORMS:
        if fu == base or fu.startswith(base + "/"):  # e.g., 8-K/A, S-1/A
            return True
        if any(fu == f"{base}{suf}" or fu.startswith(f"{base}{suf}/") for suf in _SUFFIX_OK):
            return True
    return False

def _sec_primary_url(cik_raw: str, accession: str, primary_doc: str | None) -> str:
    """Best-effort primary document URL, fallback to filing index."""
    acc = (accession or "").replace("-", "")
    if primary_doc:
        return f"https://www.sec.gov/Archives/edgar/data/{cik_raw}/{acc}/{primary_doc}"
    return f"https://www.sec.gov/Archives/edgar/data/{cik_raw}/{acc}/-index.html"

def gather_recent_filings(
    tickers: List[str],
    top_n: int = 5,
) -> pd.DataFrame:
    rows: List[Dict] = []

    for tk in sorted(set(t.strip().upper() for t in tickers if t.strip())):
        try:
            cik = resolve_cik(tk)
        except Exception as e:
            print(f"[{tk}] CIK lookup failed: {e}")
            continue

        filings = fetch_filings(cik)
        if filings is None or filings.empty:
            print(f"[{tk}] No filings returned.")
            continue

        # normalize time columns
        if "filingDate" in filings.columns:
            filings["filingDate"] = pd.to_datetime(filings["filingDate"], errors="coerce")
        if "acceptanceDateTime" in filings.columns:
            filings["acceptanceDateTime"] = pd.to_datetime(
                filings["acceptanceDateTime"], errors="coerce", utc=True
            )

        f = filings[filings["form"].apply(_form_matches)].copy()
        if f.empty:
            print(f"[{tk}] No allowed forms.")
            continue

        # Deduplicate by accessionNumber to avoid multiple rows per filing
        # (e.g., when multiple documents exist for the same 8-K)
        f = f.drop_duplicates(subset=["accessionNumber"], keep="first")

        f = f.sort_values(
            ["acceptanceDateTime", "filingDate"],
            ascending=[False, False],
            na_position="last",
        ).head(top_n)

        # prep CIK formats
        cik_raw = str(int(cik))           # for SEC URL path (no leading zeros)
        cik_padded = cik_raw.zfill(10)    # stored in CSV

        for _, r in f.iterrows():
            rows.append({
                "filingDate": r.get("filingDate"),
                "ticker": tk,
                "form": str(r.get("form") or "").upper(),
                "accessionNumber": r.get("accessionNumber"),
                "cik": cik_padded,
                "source_url": _sec_primary_url(
                    cik_raw=cik_raw,
                    accession=str(r.get("accessionNumber") or ""),
                    primary_doc=str(r.get("primaryDocument") or "") or None,
                ),
            })

    if not rows:
        return pd.DataFrame(columns=["filingDate","ticker","form","accessionNumber","cik","source_url"])

    df = pd.DataFrame(rows)
    df = df.sort_values(["ticker","filingDate"], ascending=[True, False]).reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser(description="Dump most recent SEC filings (top N per ticker) to CSV.")
    ap.add_argument(
        "--tickers",
        default="MSTR,CEP,SMLR,NAKA,BMNR,SBET,ETHZ,BTCS,SQNS,BTBT,DFDV,UPXI,HSDT,FORD",
        help="Comma-separated tickers (hardcoded default list)."
    )
    ap.add_argument("--top", type=int, default=10, help="Top N most recent filings per ticker.")
    ap.add_argument("--outdir", default="data", help="Output directory")
    ap.add_argument("--outfile", default="", help="Override output filename (CSV)")

    args = ap.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    outfile = args.outfile or str(outdir / "recent_filings.csv")

    df = gather_recent_filings(tickers=tickers, top_n=args.top)
    if df.empty:
        print("No rows to save.")
        return

    df.to_csv(outfile, index=False)
    print(f"Saved CSV → {Path(outfile).resolve()} (rows={len(df)})")

if __name__ == "__main__":
    main()
