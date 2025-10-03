#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from functools import lru_cache

import pandas as pd
import numpy as np
from pandas.api.types import is_number

from ..core.fetch import resolve_cik, fetch_filings, list_filing_files, get
from ..core.select import candidate_html_urls, Mode
from ..core.extract import html_to_blocks
from ..core.types import DealHit
from ..modes import warrants as mode_impl

# Warrants show up in a lot of places (8-K Item 1.01/3.02, S-1/S-3, 424B5 supplements, 424B3/424B7)
ALLOWED_FORMS = ("8-K", "S-1", "S-3", "424B5", "424B3", "424B7")

NEW_WARRANT_FIELDS = [
    # explicit buckets
    "warrant_shares_prefunded_text", "warrant_shares_prefunded",
    "warrant_shares_outstanding_text", "warrant_shares_outstanding",
    "warrant_type",
    # boolean flags for downstream filtering
    "warrant_prefunded_flag", "warrant_standard_flag",
    # counts of warrant instruments (not underlying shares)
    "warrant_instruments_text", "warrant_instruments",
    # role / fees
    "h_warrant_role", "agent_fee_text", "agent_fee_pct",
    # extras
    "exercise_price_text_formula",
    "expiration_date_iso",
]

PAYLOAD_FIELDS = [
    # text
    "exercise_price_text","exercise_price_text_formula",
    "shares_issued_text","warrant_coverage_text",
    "ownership_blocker_text","expiration_date_text","expiration_date_iso",
    "issuance_date_text","gross_proceeds_text","security_types_text",
    # numeric
    "exercise_price_usd","shares_issued","warrant_coverage_pct",
    "ownership_blocker_pct","warrant_term_years",
    # new buckets/flags/role/fees/instruments
    *NEW_WARRANT_FIELDS,
]

def _form_matches(f: str) -> bool:
    fu = (f or "").upper()
    return any(fu == base or fu.startswith(base + "/") for base in ALLOWED_FORMS)

def _has_payload(rec: Dict[str, Any]) -> bool:
    for k in PAYLOAD_FIELDS:
        v = rec.get(k, None)
        if v is None:
            continue
        # treat any numeric type as payload
        if is_number(v) or isinstance(v, (int, float, np.number)):
            if pd.notna(v):
                return True
        else:
            if str(v).strip() != "":
                return True
    return False

@lru_cache(maxsize=8192)
def _cached_list_filing_files(cik: str, accession: str) -> pd.DataFrame:
    try:
        return list_filing_files(cik, accession)
    except Exception:
        return pd.DataFrame()

def _recent_accession_whitelist(cik: str, subset_filings: pd.DataFrame, recent_hours: int) -> Set[str]:
    cutoff = pd.Timestamp.utcnow().tz_convert(None) - pd.Timedelta(hours=recent_hours)
    allow: Set[str] = set()

    # Fast, zero-network path
    adt = pd.to_datetime(subset_filings.get("acceptanceDateTime"), errors="coerce", utc=True).dt.tz_convert(None)
    fdt = pd.to_datetime(subset_filings.get("filingDate"), errors="coerce")
    mask = (adt >= cutoff) | (fdt >= cutoff)
    allow.update(subset_filings.loc[mask, "accessionNumber"].dropna().astype(str))

    if allow:
        return allow

    # Rare fallback: consult index.json only if the timestamps were missing/useless
    for _, r in subset_filings.iterrows():
        acc = r.get("accessionNumber")
        if not acc:
            continue
        files = _cached_list_filing_files(cik, acc)
        if not files.empty and "last_modified" in files.columns:
            lm = pd.to_datetime(files["last_modified"], errors="coerce")
            if (lm >= cutoff).any():
                allow.add(acc)
    return allow

# --- add a helper your flow can call directly ---
def get_warrants_df(
    tickers: List[str],
    year: int = 2025,
    year_by: str = "accession",
    limit: int = 600,
    max_docs: int = 6,
    max_snips: int = 4,
    recent_hours: int = 24,
) -> pd.DataFrame:
    frames = []
    for t in sorted(set(t.strip().upper() for t in tickers if t.strip())):
        df = build_for_ticker(
            t, year=year, year_by=year_by, limit=limit,
            max_docs=max_docs, max_snips=max_snips, recent_hours=recent_hours
        )
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def build_for_ticker(
    ticker: str,
    year: int,
    year_by: str,
    limit: int,
    max_docs: int,
    max_snips: int,
    recent_hours: int = 0,
) -> pd.DataFrame:
    try:
        cik = resolve_cik(ticker)
    except Exception as e:
        print(f"[{ticker}] CIK lookup failed: {e}")
        return pd.DataFrame()

    filings = fetch_filings(cik)
    if filings.empty:
        print(f"[{ticker}] No filings returned.")
        return pd.DataFrame()

    # Normalize helper columns used for filtering/sorting
    if "filingDate" in filings.columns:
        filings["filingDate"] = pd.to_datetime(filings["filingDate"], errors="coerce")
    if "acceptanceDateTime" in filings.columns:
        filings["acceptanceDateTime"] = pd.to_datetime(filings["acceptanceDateTime"], errors="coerce", utc=True).dt.tz_convert(None)

    # Derive acc_year for "year_by=accession" parity with your original script
    if "acc_year" not in filings.columns and "accessionNumber" in filings.columns:
        filings["acc_year"] = (
            filings["accessionNumber"].astype(str).str.extract(r"^\d{10}-(\d{2})-")[0]
            .astype("float")
            .apply(lambda y: 2000 + int(y) if pd.notna(y) else pd.NA)
        )

    # Start with allowed forms
    subset = filings[filings["form"].apply(_form_matches)].copy()

    # Recent mode: only keep items touched within last N hours
    # Recent mode: only keep items touched within last N hours
    if recent_hours and recent_hours > 0:
        cutoff = pd.Timestamp.utcnow().tz_convert(None) - pd.Timedelta(hours=recent_hours)

        # Use vectorized time filters (zero network)
        adt = filings.get("acceptanceDateTime")  # already tz-naive in your code above
        fdt = filings.get("filingDate")          # already datetime in your code above
        mask_time = pd.Series(False, index=filings.index)
        if adt is not None:
            mask_time = mask_time | (adt >= cutoff)
        if fdt is not None:
            mask_time = mask_time | (fdt >= cutoff)

        subset = filings[mask_time & filings["form"].apply(_form_matches)].copy()
        subset = subset.sort_values(["filingDate","acceptanceDateTime"], na_position="last").tail(limit)

        # OPTIONAL fallback only if empty: (uses index.json, but rarely hits)
        if subset.empty:
            tail = filings[filings["form"].apply(_form_matches)].sort_values(
                ["filingDate","acceptanceDateTime"], na_position="last"
            ).tail(limit)

            # If you added the cached helper, prefer it here:
            wl = _recent_accession_whitelist(cik, tail, recent_hours)
            if not wl:
                print(f"[{ticker}] No allowed forms touched in last {recent_hours}h.")
                return pd.DataFrame()
            subset = tail[tail["accessionNumber"].isin(wl)].copy()

        else:
            # Year filters (original behavior)  <-- leave this block unchanged
            if year_by == "accession":
                mask_year = (filings["acc_year"] == year)
            else:
                mask_year = filings["filingDate"].dt.year.eq(year)
            subset = filings[mask_year & filings["form"].apply(_form_matches)].copy()
            subset = subset.sort_values(["filingDate","acceptanceDateTime"], na_position="last").tail(limit)


    if subset.empty:
        msg = f"last {recent_hours}h" if (recent_hours and recent_hours > 0) else f"{year} (year-by={year_by})"
        print(f"[{ticker}] No allowed forms in {msg}.")
        return pd.DataFrame()

    out_rows: List[DealHit] = []

    for _, r in subset.iterrows():
        acc  = r.get("accessionNumber")
        if not acc:
            continue
        prim = (r.get("primaryDocument") or "")
        form = str(r.get("form") or "").upper()
        fdt  = r.get("filingDate")

        try:
            files = _cached_list_filing_files(cik, acc)
            cand  = candidate_html_urls(files, primary_doc=prim, filing_form=form, mode=Mode.WARRANTS, max_docs=max_docs)
        except Exception as e:
            print(f"[{ticker}] Skipping {acc}: index.json error: {e}")
            continue

        per_filing_hits: List[DealHit] = []

        for url, ex_hint in cand:
            try:
                html_bytes = get(url).content
                blocks     = html_to_blocks(html_bytes)
            except Exception as e:
                print(f"[{ticker}] Error fetching/parsing {url}: {e}")
                continue

            hits: List[DealHit] = []
            for b in blocks:
                for h in mode_impl.classify_block(b, form):
                    h.update({
                        "ticker": ticker,
                        "filingDate": fdt,
                        "form": form,
                        "accessionNumber": acc,
                        "source_url": url,
                        "exhibit_hint": (ex_hint or "").lower(),
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
        # exercise price
        "exercise_price_text","exercise_price_text_formula","exercise_price_usd",
        # legacy shares & numerics
        "shares_issued_text","shares_issued",
        # explicit warrant share buckets
        "warrant_shares_prefunded_text","warrant_shares_prefunded",
        "warrant_shares_outstanding_text","warrant_shares_outstanding",
        "warrant_type","warrant_prefunded_flag","warrant_standard_flag",
        # instruments (count of warrants issued)
        "warrant_instruments_text","warrant_instruments",
        # coverage / blockers
        "warrant_coverage_text","warrant_coverage_pct",
        "ownership_blocker_text","ownership_blocker_pct",
        # term / dates
        "expiration_date_text","expiration_date_iso","warrant_term_years","issuance_date_text",
        # money / security types
        "gross_proceeds_text","security_types_text",
        # role & fees
        "h_warrant_role","agent_fee_text","agent_fee_pct",
        # provenance
        "source_url","exhibit_hint","snippet","score",
    ]

    df = pd.DataFrame(out_rows).sort_values(
        ["ticker","filingDate","accessionNumber","score"], ascending=[True, True, True, False]
    ).reset_index(drop=True)
    df = df[[c for c in cols if c in df.columns]]
    return df

def main():
    ap = argparse.ArgumentParser(description="Warrants extractor (exercise price, shares purchasable, coverage, expiration).")
    ap.add_argument("--tickers", default="MSTR,CEP,SMLR,NAKA,BMNR,SBET,ETHZ,BTCS,SQNS,BTBT,DFDV,UPXI", help="Comma-separated tickers")
    ap.add_argument("--tickers-file", default="", help="Optional newline-delimited tickers file")
    ap.add_argument("--year", type=int, default=2025, help="Year filter (ignored if --recent-hours > 0)")
    ap.add_argument("--year-by", choices=["accession","filingdate"], default="accession")
    ap.add_argument("--recent-hours", type=int, default=24, help="If >0, only scan accessions touched in the last N hours")
    ap.add_argument("--limit", type=int, default=600, help="Max filings per ticker to scan (applies after filtering)")
    ap.add_argument("--max-docs", type=int, default=6, help="Max HTML docs per filing to fetch")
    ap.add_argument("--max-snippets-per-filing", type=int, default=4, help="Top-N snippets per filing")
    ap.add_argument("--outdir", default="data", help="Output directory")
    ap.add_argument("--outfile", default="", help="Overrides default output path")
    args = ap.parse_args()

    # build ticker set
    tickers: list[str] = []
    if args.tickers:
        tickers.extend([t.strip().upper() for t in args.tickers.split(",") if t.strip()])
    if args.tickers_file:
        p = Path(args.tickers_file)
        if p.exists():
            tickers.extend([ln.strip().upper() for ln in p.read_text().splitlines() if ln.strip()])
    tickers = sorted(set(tickers))
    if not tickers:
        raise SystemExit("No tickers provided.")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if args.outfile:
        outfile = args.outfile
    else:
        if args.recent_hours and args.recent_hours > 0:
            stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")
            outfile = str(outdir / f"warrants_recent_{args.recent_hours}h_{stamp}.csv")
        else:
            outfile = str(outdir / f"warrants_timeline_all_{args.year}.csv")

    all_rows: list[pd.DataFrame] = []
    for t in tickers:
        print(f"\n=== {t} ===")
        df = build_for_ticker(
            t,
            year=args.year,
            year_by=args.year_by,
            limit=args.limit,
            max_docs=args.max_docs,
            max_snips=args.max_snippets_per_filing,
            recent_hours=args.recent_hours,
        )
        if not df.empty:
            print(f"[{t}] rows: {len(df)}")
            all_rows.append(df)
        else:
            mode_desc = f"last {args.recent_hours}h" if (args.recent_hours and args.recent_hours > 0) else f"{args.year}"
            print(f"[{t}] no warrant rows ({mode_desc}).")

    # --- in main(): only save if --outfile provided ---
    if not all_rows:
        print("No warrant data for any ticker.")
        return

    combo = pd.concat(all_rows, ignore_index=True)

    if args.outfile:
        combo.to_csv(args.outfile, index=False)
        print(f"\nSaved warrants -> {Path(args.outfile).resolve()} (rows={len(combo)}, tickers={len(tickers)})")
    else:
        # No write: just show a quick summary
        print(f"\nCollected warrants rows={len(combo)} across tickers={len(tickers)} (no file written).")


if __name__ == "__main__":
    main()
