# sec_deals/drivers/run_pipes.py
#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from ..core.fetch import resolve_cik, fetch_filings, list_filing_files, get
from ..core.select import candidate_html_urls, Mode
from ..core.extract import html_to_blocks
from ..core.types import DealHit
from ..modes import pipes as mode_impl

ALLOWED_FORMS = ("8-K", "S-1", "S-3", "424B3", "424B7")  # no 424B5 in pipes mode

PAYLOAD_FIELDS = [
    # text
    "price_per_share_text","gross_proceeds_text","shares_issued_text",
    "warrant_coverage_text","exercise_price_text","convert_price_text",
    "ownership_blocker_text","reg_file_deadline_text","reg_effect_deadline_text",
    "closing_date_text","security_types_text",
    # numeric
    "price_per_share_usd","gross_proceeds_usd","shares_issued",
    "warrant_coverage_pct","exercise_price_usd","convert_price_usd",
    "ownership_blocker_pct","reg_file_deadline_days","reg_effect_deadline_days",
]
ECON_FIELDS = {
    # text
    "price_per_share_text","gross_proceeds_text","shares_issued_text",
    "warrant_coverage_text","exercise_price_text","convert_price_text",
    "ownership_blocker_text","reg_file_deadline_text","reg_effect_deadline_text",
    "closing_date_text","security_types_text",
    # numeric
    "price_per_share_usd","gross_proceeds_usd","shares_issued",
    "warrant_coverage_pct","exercise_price_usd","convert_price_usd",
    "ownership_blocker_pct","reg_file_deadline_days","reg_effect_deadline_days",
}

def _form_matches(f: str) -> bool:
    fu = (f or "").upper()
    return any(fu == base or fu.startswith(base + "/") for base in ALLOWED_FORMS)

def _has_payload_fields(rec: Dict[str, Any]) -> bool:
    et = (rec.get("event_type_final") or "").lower()
    # Allow resale events through even if they don't carry economics
    if et in {"resale_filed", "resale_effective"}:
        return True

    for k in PAYLOAD_FIELDS:
        v = rec.get(k, None)
        if v is None:
            continue
        if isinstance(v, float):
            if pd.notna(v):
                return True
        else:
            if str(v).strip() != "":
                return True
    return False

def _merge_hits_for_filing(hits: list[DealHit]) -> list[DealHit]:
    """
    Collapse multiple blocks for the same filing (accession) + event_type into one record.
    Prefer higher-score values; fill missing economics from lower-score blocks.
    """
    # Sort by score high→low so we keep the best snippet/source as the base
    hits_sorted = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)
    merged: dict[tuple[str, str], DealHit] = {}

    for h in hits_sorted:
        key = (h["accessionNumber"], h.get("event_type_final", ""))
        if key not in merged:
            merged[key] = h.copy()
            continue

        base = merged[key]
        # If this hit scored higher, upgrade snippet/provenance
        if h.get("score", 0) > base.get("score", 0):
            base["snippet"] = h.get("snippet", base.get("snippet"))
            base["score"] = h["score"]
            base["source_url"] = h.get("source_url", base.get("source_url"))
            base["exhibit_hint"] = h.get("exhibit_hint", base.get("exhibit_hint"))

        # Fill any missing econ fields
        for k in ECON_FIELDS:
            bv = base.get(k)
            hv = h.get(k)
            if (bv is None or (isinstance(bv, float) and pd.isna(bv)) or str(bv).strip() == "") and hv not in (None, ""):
                base[k] = hv

    return list(merged.values())

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
    try:
        cik = resolve_cik(ticker)
    except Exception as e:
        print(f"[{ticker}] CIK lookup failed: {e}")
        return pd.DataFrame()

    filings = fetch_filings(cik)
    if filings.empty:
        print(f"[{ticker}] No filings returned.")
        return pd.DataFrame()

    # ensure datetime dtypes
    if "acceptanceDateTime" in filings.columns:
        filings["acceptanceDateTime"] = pd.to_datetime(
            filings["acceptanceDateTime"], utc=True, errors="coerce"
        )
    if "filingDate" in filings.columns:
        filings["filingDate"] = pd.to_datetime(
            filings["filingDate"], utc=True, errors="coerce"
        )

    # ---- NEW: recent-window filter (default = last 24h) ----
    if since_hours and since_hours > 0:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=since_hours)
        if use_acceptance and "acceptanceDateTime" in filings.columns:
            mask_time = filings["acceptanceDateTime"] >= cutoff
        else:
            mask_time = filings["filingDate"] >= cutoff.normalize()
        time_info = f"since {cutoff.isoformat()}"
    else:
        # original year-based behavior
        if year_by == "accession":
            mask_time = (filings["acc_year"] == year)
        else:
            mask_time = filings["filingDate"].dt.year.eq(year)
        time_info = f"year={year} (by {year_by})"

    subset = filings[mask_time & filings["form"].apply(_form_matches)].copy()
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
        fdt  = r.get("filingDate")

        try:
            files = list_filing_files(cik, acc)
            cand = candidate_html_urls(
                files,
                primary_doc=prim,
                filing_form=form,
                mode=Mode.PIPES,
                max_docs=max_docs
            )
        except Exception as e:
            print(f"[{ticker}] Skipping {acc}: index.json error: {e}")
            continue

        per_filing_hits: List[DealHit] = []

        for url, ex_hint in cand:
            try:
                html_bytes = get(url).content
                blocks = html_to_blocks(html_bytes)
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
            if _has_payload_fields(h):
                out_rows.append(h)

    if not out_rows:
        return pd.DataFrame()

    cols = [
        "ticker","filingDate","form","accessionNumber","event_type_final",
        "price_per_share_text","price_per_share_usd",
        "shares_issued_text","shares_issued",
        "gross_proceeds_text","gross_proceeds_usd",
        "warrant_coverage_text","warrant_coverage_pct",
        "exercise_price_text","exercise_price_usd",
        "convert_price_text","convert_price_usd",
        "ownership_blocker_text","ownership_blocker_pct",
        "reg_file_deadline_text","reg_file_deadline_days",
        "reg_effect_deadline_text","reg_effect_deadline_days",
        "closing_date_text","security_types_text",
        "source_url","exhibit_hint","snippet","score"
    ]
    df = pd.DataFrame(out_rows)
    df = df.sort_values(
        ["ticker","filingDate","accessionNumber","score"],
        ascending=[True, True, True, False]
    ).reset_index(drop=True)
    df = df[[c for c in cols if c in df.columns]]
    return df


def main():
    ap = argparse.ArgumentParser(description="PIPE-only timeline extractor (per-ticker).")
    ap.add_argument("--tickers", default="MSTR,CEP,SMLR,NAKA,BMNR,SBET,ETHZ,BTCS,SQNS,BTBT,DFDV,UPXI",
                    help="Comma-separated tickers (e.g., MSTR,SMLR,BMNR)")
    ap.add_argument("--tickers-file", default="", help="Optional newline-delimited tickers file")

    # original year-based params (still supported if you disable the recent window)
    ap.add_argument("--year", type=int, default=2025, help="Year filter")
    ap.add_argument("--year-by", choices=["accession","filingdate"], default="accession",
                    help="Filter by accession year or filingDate year")

    ap.add_argument("--limit", type=int, default=600, help="Max filings per ticker to scan")
    ap.add_argument("--max-docs", type=int, default=6, help="Max HTML docs per filing to fetch")
    ap.add_argument("--max-snippets-per-filing", type=int, default=4, help="Top-N snippets to keep per filing")

    # NEW: lookback window controls
    ap.add_argument("--since-hours", type=int, default=24,
                    help="Look back this many hours from now (UTC). Use 0 to disable and use --year/--year-by.")
    ap.add_argument("--use-acceptance", action="store_true", default=True,
                    help="Filter by acceptanceDateTime if available (default).")
    ap.add_argument("--no-use-acceptance", dest="use_acceptance", action="store_false")

    ap.add_argument("--outdir", default="data", help="Output directory")
    ap.add_argument("--outfile", default="", help="If set, write a single CSV here; else default under outdir")
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

    # filename reflects mode
    if args.outfile:
        outfile = args.outfile
    else:
        if args.since_hours and args.since_hours > 0:
            ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%MZ")
            outfile = str(outdir / f"pipes_past{args.since_hours}h_{ts}.csv")
        else:
            outfile = str(outdir / f"pipes_timeline_all_{args.year}.csv")

    all_rows: list[pd.DataFrame] = []
    for t in tickers:
        print(f"\n=== {t} ===")
        df = build_for_ticker(
            t, args.year, args.year_by,
            args.limit, args.max_docs, args.max_snippets_per_filing,
            since_hours=args.since_hours,
            use_acceptance=args.use_acceptance,
        )
        if not df.empty:
            print(f"[{t}] rows: {len(df)}")
            all_rows.append(df)
        else:
            print(f"[{t}] no PIPE rows.")

    if not all_rows:
        print("No PIPE data for any ticker.")
        return

    combo = pd.concat(all_rows, ignore_index=True)
    combo.to_csv(outfile, index=False)
    print(f"\nSaved PIPE timeline -> {Path(outfile).resolve()} (rows={len(combo)}, tickers={len(tickers)})")


if __name__ == "__main__":
    main()
