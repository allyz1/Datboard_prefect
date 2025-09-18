#!/usr/bin/env python3
# sec_btc_holdings_pipeline.py
# Return standardized BTC holdings DataFrame from SEC filings (24h lookback).

from typing import List, Dict, Any, Optional, Union
import re
import pandas as pd

# === SEC BTC modules (unchanged logic) ===
from app.sec import MSTR_8k_dat_timeline as mod_xml   # XML/iXBRL BTC table extractor
from app.sec import merged_8k_dat_timeline as mod_html  # HTML BTC purchases/holdings

# -------- Normalized schema (matches your existing wide shape) --------
KEEP_COL_ORDER = [
    "ticker","cik","form","accessionNumber","filingDate","reportDate","acceptanceDateTime",
    "source_url","record_type",
    # table metrics
    "as_of","period_btc_acquired","period_agg_purchase_price_usd","period_avg_purchase_price_usd",
    "total_btc_holdings","total_agg_purchase_price_usd","total_avg_purchase_price_usd",
    # html / purchase metrics
    "btc","usd","avg_usd_per_btc","usd_cost_total","market_value_usd","wallet_urls",
    # meta
    "primaryDocument","filing_url_base","doc_index","event_index","snippet"
]

def _to_iso_date(x):
    if pd.isna(x): return None
    try:
        return x.date().isoformat()
    except Exception:
        return str(x)

# ----------------------- core: run for a single ticker -----------------------
def run_sec_for_ticker(
    ticker: str,
    forms: List[str],
    hours_back: int = 24,
    limit: int = 200,
    max_docs: int = 6,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Fetch filings metadata, filter to allowed forms and acceptanceDateTime in the last
    `hours_back` hours (UTC-aware), then parse only those filings.
    Returns a wide/intermediate DataFrame preserving your original columns.
    """
    cik = mod_xml.resolve_cik(ticker)
    filings = mod_xml.fetch_filings(cik)
    if filings.empty:
        if debug: print(f"[{ticker}] No filings returned.")
        return pd.DataFrame()

    allowed = {f.upper() for f in forms}
    subset = filings[filings["form"].str.upper().isin(allowed)].copy()

    # --- 24h lookback (tz-aware UTC) ---
    subset["acceptanceDateTime"] = pd.to_datetime(subset["acceptanceDateTime"], errors="coerce", utc=True)
    subset["filingDate"] = pd.to_datetime(subset["filingDate"], errors="coerce")

    cutoff_dt = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours_back)
    subset = subset[subset["acceptanceDateTime"].notna()]
    subset = subset[subset["acceptanceDateTime"] >= cutoff_dt]

    if subset.empty:
        if debug: print(f"[{ticker}] No filings in the last {hours_back}h — skipping.")
        return pd.DataFrame()

    subset = subset.sort_values(["acceptanceDateTime","filingDate"], na_position="last").tail(limit)

    if debug:
        print(f"[{ticker}] Parsing {len(subset)} filings from last {hours_back}h for forms={sorted(allowed)}")

    rows_norm: List[Dict[str, Any]] = []

    # --- parsing loop (unchanged logic) ---
    for _, row in subset.iterrows():
        accession = row.get("accessionNumber")
        form      = row.get("form")
        primary   = (row.get("primaryDocument") or "")
        filing_dt = row.get("filingDate")
        filing_date_iso = _to_iso_date(filing_dt) or ""

        if debug:
            print(f"\n>>> [{ticker}] {form} {accession}")
            if pd.notna(filing_dt):
                print(f"    filingDate={filing_dt}  primary={primary}")

        # Candidate URLs: XML-ranked then HTML-ranked, unioned
        try:
            files = mod_xml.list_filing_files(cik, accession)
            urls_xml  = mod_xml.candidate_html_urls(files, primary)[:max_docs]
            urls_html = mod_html.candidate_html_urls(files, primary, max_docs=max_docs)
            html_urls = list(dict.fromkeys(urls_xml + urls_html))[:max_docs]
            if debug:
                print("    Candidate URLs (unioned, XML-first):")
                for u in html_urls:
                    print(f"      {u}")
        except Exception as e:
            print(f"    Skipping index.json for {accession}: {e}")
            continue

        # Track whether any doc produced a hit for this filing
        for doc_idx, u in enumerate(html_urls):
            if debug:
                print(f"      -> Parsing doc {doc_idx+1}/{len(html_urls)}: {u}")

            found_table = False
            found_holdings = False
            found_purchase = False

            # 1) XML/iXBRL BTC table
            try:
                if debug: print("         [XML parser attempt]")
                rec = mod_xml.extract_btc_metrics_from_url(u)
                if rec.get("ok"):
                    found_table = True
                    rows_norm.append({
                        "ticker": ticker, "cik": rec.get("cik") or cik,
                        "form": form,
                        "accessionNumber": accession,
                        "filingDate": row.get("filingDate"),
                        "reportDate": row.get("reportDate"),
                        "acceptanceDateTime": row.get("acceptanceDateTime"),
                        "source_url": u,
                        "record_type": "ixbrl_table",
                        "as_of": rec.get("as_of"),
                        "period_btc_acquired": rec.get("period_btc_acquired"),
                        "period_agg_purchase_price_usd": rec.get("period_agg_purchase_price_usd"),
                        "period_avg_purchase_price_usd": rec.get("period_avg_purchase_price_usd"),
                        "total_btc_holdings": rec.get("total_btc_holdings"),
                        "total_agg_purchase_price_usd": rec.get("total_agg_purchase_price_usd"),
                        "total_avg_purchase_price_usd": rec.get("total_avg_purchase_price_usd"),
                        "wallet_urls": None,
                        "btc": None, "usd": None, "avg_usd_per_btc": None,
                        "usd_cost_total": None, "market_value_usd": None,
                        "snippet": None,
                        "primaryDocument": primary,
                        "filing_url_base": mod_xml.filing_base_url(cik, accession),
                        "doc_index": doc_idx, "event_index": None,
                    })
                else:
                    if debug: print("         XML: no BTC table.")
            except Exception as e:
                print(f"         XML parse error: {e}")

            # 2) HTML keyword parser
            try:
                if debug: print("         [HTML keyword parser attempt]")
                txt = mod_html.html_to_text_bytes_first(u)

                # (a) STRICT completed purchases
                try:
                    events = mod_html.extract_completed_purchase_events(txt)
                except Exception as e:
                    print(f"         Purchases parse error: {e}")
                    events = []
                if events:
                    found_purchase = True
                    if debug: print(f"         Purchases found: {len(events)}")
                for k, ev in enumerate(events):
                    rows_norm.append({
                        "ticker": ticker, "cik": cik,
                        "form": form,
                        "accessionNumber": accession,
                        "filingDate": row.get("filingDate"),
                        "reportDate": row.get("reportDate"),
                        "acceptanceDateTime": row.get("acceptanceDateTime"),
                        "source_url": u,
                        "record_type": "completed_purchase",
                        "as_of": None,
                        "period_btc_acquired": None,
                        "period_agg_purchase_price_usd": None,
                        "period_avg_purchase_price_usd": None,
                        "total_btc_holdings": None,
                        "total_agg_purchase_price_usd": None,
                        "total_avg_purchase_price_usd": None,
                        "btc": ev.get("btc"),
                        "usd": ev.get("usd"),
                        "avg_usd_per_btc": ev.get("avg_usd_per_btc"),
                        "wallet_urls": " | ".join(ev.get("wallet_urls") or []),
                        "usd_cost_total": None, "market_value_usd": None,
                        "snippet": ev.get("snippet"),
                        "primaryDocument": primary,
                        "filing_url_base": mod_xml.filing_base_url(cik, accession),
                        "doc_index": doc_idx, "event_index": k,
                    })

                # (b) Holdings snapshots
                try:
                    holds = mod_html.extract_holdings_snapshots(txt, filing_date_iso=filing_date_iso)
                except Exception as e:
                    print(f"         Holdings parse error: {e}")
                    holds = []
                if holds:
                    found_holdings = True
                    if debug: print(f"         Holdings snapshots found: {len(holds)}")
                for j, hv in enumerate(holds):
                    rows_norm.append({
                        "ticker": ticker, "cik": cik,
                        "form": form,
                        "accessionNumber": accession,
                        "filingDate": row.get("filingDate"),
                        "reportDate": row.get("reportDate"),
                        "acceptanceDateTime": row.get("acceptanceDateTime"),
                        "source_url": u,
                        "record_type": "holdings_snapshot",
                        "as_of": hv.get("as_of") or filing_date_iso,
                        "period_btc_acquired": None,
                        "period_agg_purchase_price_usd": None,
                        "period_avg_purchase_price_usd": None,
                        "total_btc_holdings": None,
                        "total_agg_purchase_price_usd": None,
                        "total_avg_purchase_price_usd": None,
                        "btc": hv.get("btc"),
                        "usd": None,
                        "avg_usd_per_btc": hv.get("avg_usd_per_btc"),
                        "wallet_urls": None,
                        "usd_cost_total": hv.get("usd_cost_total"),
                        "market_value_usd": hv.get("market_value_usd"),
                        "snippet": hv.get("snippet"),
                        "primaryDocument": primary,
                        "filing_url_base": mod_xml.filing_base_url(cik, accession),
                        "doc_index": doc_idx, "event_index": j,
                    })

            except Exception as e:
                print(f"         Text fetch/parse error: {e}")

            if found_table or found_holdings or found_purchase:
                if debug:
                    which = "table" if found_table else ("holdings" if found_holdings else "purchase")
                    print(f"         ✅ Hit via {which}; moving to next filing.")
                break

    if not rows_norm:
        if debug: print(f"[{ticker}] No parsed rows from last {hours_back}h.")
        return pd.DataFrame()

    df = pd.DataFrame(rows_norm)

    # Coerce date-like columns
    for c in ("filingDate","reportDate","acceptanceDateTime"):
        if c in df.columns:
            # acceptanceDateTime is already tz-aware upstream; preserving here is fine.
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Ensure consistent columns and order
    for c in KEEP_COL_ORDER:
        if c not in df.columns:
            df[c] = None
    df = df[KEEP_COL_ORDER]

    # Build as_of_dt for stable dating
    if "as_of" in df.columns:
        df["as_of_dt"] = pd.to_datetime(df["as_of"], errors="coerce")
    else:
        df["as_of_dt"] = pd.NaT
    # Fill as_of_dt with filingDate if missing
    if "filingDate" in df.columns:
        mask = df["as_of_dt"].isna()
        df.loc[mask, "as_of_dt"] = df.loc[mask, "filingDate"]

    sort_cols = ["as_of_dt","filingDate","accessionNumber","doc_index","event_index","record_type"]
    df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    return df

# ----------------------- public: tidy df for main -----------------------
def get_sec_btc_holdings_df(
    tickers: Union[str, List[str]],
    hours_back: int = 24,
    forms: Union[str, List[str]] = ("8-K", "10-K", "10-Q"),
    limit: int = 200,
    max_docs: int = 6,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Return a standardized DataFrame with columns: date, ticker, asset, total_holdings.
    - 24h lookback on acceptanceDateTime
    - drops `period_btc_acquired` (if present)
    - merges `total_btc_holdings` and `btc` → `total_holdings`
    - de-dupes by (ticker, date), keeping MAX per day
    - no file saving/printing
    """
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    if isinstance(forms, str):
        forms = [f.strip() for f in forms.split(",") if f.strip()]

    all_wide = []
    for t in tickers:
        if verbose:
            print(f"\n=== SEC BTC pipeline (last {hours_back}h): {t} ===")
        dft = run_sec_for_ticker(
            ticker=t,
            forms=forms,
            hours_back=hours_back,
            limit=limit,
            max_docs=max_docs,
            debug=verbose,
        )
        if dft.empty:
            if verbose:
                print(f"[{t}] No new filings in the last {hours_back}h — skipping.")
            continue
        dft["ticker"] = t  # enforce casing
        all_wide.append(dft)

    if not all_wide:
        # Return empty in the target schema to keep your main happy
        return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings"])

    df = pd.concat(all_wide, ignore_index=True)

    # Drop the column you specified (if present)
    if "period_btc_acquired" in df.columns:
        df = df.drop(columns=["period_btc_acquired"])

    # Choose the best date: prefer as_of_dt, otherwise filingDate
    if "as_of_dt" not in df.columns:
        df["as_of_dt"] = pd.NaT
    if "filingDate" in df.columns:
        mask = df["as_of_dt"].isna()
        df.loc[mask, "as_of_dt"] = df.loc[mask, "filingDate"]

    # Merge totals: prefer total_btc_holdings, else btc
    holdings = pd.to_numeric(df.get("total_btc_holdings"), errors="coerce")
    holdings = holdings.fillna(pd.to_numeric(df.get("btc"), errors="coerce"))

    out = pd.DataFrame({
        "date": pd.to_datetime(df["as_of_dt"], errors="coerce").dt.date.astype("string"),
        "ticker": df["ticker"].astype(str).str.upper(),
        "asset": "BTC",
        "total_holdings": pd.to_numeric(holdings, errors="coerce"),
    }).dropna(subset=["total_holdings"])

    # De-dupe by (ticker, date) keeping MAX holdings for that day
    out = (
        out.groupby(["ticker", "date"], as_index=False, sort=True)["total_holdings"]
           .max()
           .sort_values(["ticker", "date"])
           .reset_index(drop=True)
    )
    return out

# Optional quick manual test
if __name__ == "__main__":
    df = get_sec_btc_holdings_df("MSTR,NAKA,SMLR,SQNS,CEP", hours_back=120, verbose=True)
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df.tail(20).to_string(index=False))
