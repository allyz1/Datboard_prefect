#!/usr/bin/env python3
# sec_eth_holdings_pipeline.py
# Returns standardized ETH holdings DataFrame from SEC filings (24h lookback).
# Keeps original parsing (XML tables + HTML events/snapshots).

from typing import List, Dict, Any, Optional, Union
import re
import pandas as pd

# === SEC ETH modules ===
from app.sec import ETH_table_pipeline as mod_xml          # iXBRL/HTML ETH table extractor
from app.sec import ETH_dat_timeline as mod_html           # HTML ETH purchases/intents & holdings

# -------- Normalized schema (consistent with your original) --------
# record_type: "ixbrl_table" | "completed_purchase" | "purchase_intent" | "holdings_snapshot"
KEEP_COL_ORDER = [
    "ticker", "filingDate", "form", "source_url", "record_type",
    # direct event metrics (HTML)
    "eth", "usd", "avg_usd_per_eth", "usd_cost_total", "market_value_usd",
    # table metrics (iXBRL/HTML table)
    "period_eth_acquired", "period_agg_purchase_price_usd", "period_avg_purchase_price_usd",
    "total_eth_holdings", "total_agg_purchase_price_usd", "total_avg_purchase_price_usd",
    # meta/extras retained
    "wallet_urls", "primaryDocument", "filing_url_base",
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
    debug: bool = True,
) -> pd.DataFrame:
    """
    Fetch filings metadata, filter to allowed forms and acceptanceDateTime in the last
    `hours_back` hours (UTC-aware), then parse only those filings.
    Returns a wide/intermediate DataFrame with the same columns you had.
    """
    cik = mod_xml.resolve_cik(ticker)
    filings = mod_xml.fetch_filings(cik)
    if filings.empty:
        if debug: print(f"[{ticker}] No filings returned.")
        return pd.DataFrame()

    # Filter by form first
    allowed = {f.upper() for f in forms}
    subset = filings[filings["form"].str.upper().isin(allowed)].copy()

    # --- 24h lookback using acceptanceDateTime (tz-aware UTC) ---
    subset["acceptanceDateTime"] = pd.to_datetime(
        subset["acceptanceDateTime"], errors="coerce", utc=True
    )
    subset["filingDate"] = pd.to_datetime(subset["filingDate"], errors="coerce")

    cutoff_dt = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours_back)
    subset = subset[subset["acceptanceDateTime"].notna()]
    subset = subset[subset["acceptanceDateTime"] >= cutoff_dt]

    if subset.empty:
        if debug: print(f"[{ticker}] No filings in the last {hours_back}h — skipping.")
        return pd.DataFrame()

    # Keep the newest N after filtering
    subset = subset.sort_values(["acceptanceDateTime","filingDate"], na_position="last").tail(limit)

    if debug:
        print(f"[{ticker}] Parsing {len(subset)} filings from last {hours_back}h for forms={sorted(allowed)}")

    rows_norm: List[Dict[str, Any]] = []

    # Parse each qualifying filing (UNCHANGED LOGIC)
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

        # Try each candidate document until we get a hit
        for u in html_urls:
            if debug:
                print(f"      -> Parsing: {u}")

            had_hit = False

            # 1) XML/iXBRL ETH table
            try:
                if debug: print("         [XML parser attempt]")
                rec = mod_xml.extract_eth_metrics_from_url(u)
                if rec.get("ok"):
                    rows_norm.append({
                        "ticker": ticker,
                        "filingDate": row.get("filingDate"),
                        "form": form,
                        "source_url": u,
                        "record_type": "ixbrl_table",
                        # direct event metrics absent for table
                        "eth": None, "usd": None, "avg_usd_per_eth": None,
                        "usd_cost_total": None, "market_value_usd": None,
                        # table metrics
                        "period_eth_acquired": rec.get("period_eth_acquired"),
                        "period_agg_purchase_price_usd": rec.get("period_agg_purchase_price_usd"),
                        "period_avg_purchase_price_usd": rec.get("period_avg_purchase_price_usd"),
                        "total_eth_holdings": rec.get("total_eth_holdings"),
                        "total_agg_purchase_price_usd": rec.get("total_agg_purchase_price_usd"),
                        "total_avg_purchase_price_usd": rec.get("total_avg_purchase_price_usd"),
                        # extras
                        "wallet_urls": None,
                        "primaryDocument": primary,
                        "filing_url_base": mod_xml.filing_base_url(cik, accession),
                    })
                    had_hit = True
                else:
                    if debug: print("         XML: no ETH table.")
            except Exception as e:
                print(f"         XML parse error: {e}")

            # 2) HTML keyword parser (ETH purchases/intents + holdings)
            try:
                if debug: print("         [HTML keyword parser attempt]")
                txt = mod_html.html_to_text_bytes_first(u)

                # (a) Purchases / intents
                try:
                    events = mod_html.extract_eth_purchase_intent_events(txt)
                except Exception as e:
                    print(f"         Purchases/intents parse error: {e}")
                    events = []

                for ev in (events or []):
                    rows_norm.append({
                        "ticker": ticker,
                        "filingDate": row.get("filingDate"),
                        "form": form,
                        "source_url": u,
                        "record_type": ("completed_purchase" if ev.get("kind") == "completed" else "purchase_intent"),
                        "eth": ev.get("eth"),
                        "usd": ev.get("usd"),
                        "avg_usd_per_eth": ev.get("avg_usd_per_eth"),
                        "usd_cost_total": None,
                        "market_value_usd": None,
                        # table metrics not applicable here
                        "period_eth_acquired": None,
                        "period_agg_purchase_price_usd": None,
                        "period_avg_purchase_price_usd": None,
                        "total_eth_holdings": None,
                        "total_agg_purchase_price_usd": None,
                        "total_avg_purchase_price_usd": None,
                        "wallet_urls": " | ".join(ev.get("wallet_urls") or []),
                        "primaryDocument": primary,
                        "filing_url_base": mod_xml.filing_base_url(cik, accession),
                    })
                    had_hit = True

                # (b) Holdings snapshots
                try:
                    holds = mod_html.extract_eth_holdings_snapshots(
                        txt,
                        filing_date_iso=filing_date_iso
                    )
                except Exception as e:
                    print(f"         Holdings parse error: {e}")
                    holds = []

                for hv in (holds or []):
                    rows_norm.append({
                        "ticker": ticker,
                        "filingDate": row.get("filingDate"),
                        "form": form,
                        "source_url": u,
                        "record_type": "holdings_snapshot",
                        "eth": hv.get("eth"),
                        "usd": None,
                        "avg_usd_per_eth": hv.get("avg_usd_per_eth"),
                        "usd_cost_total": hv.get("usd_cost_total"),
                        "market_value_usd": hv.get("market_value_usd"),
                        # table metrics not applicable
                        "period_eth_acquired": None,
                        "period_agg_purchase_price_usd": None,
                        "period_avg_purchase_price_usd": None,
                        "total_eth_holdings": None,
                        "total_agg_purchase_price_usd": None,
                        "total_avg_purchase_price_usd": None,
                        "wallet_urls": None,
                        "primaryDocument": primary,
                        "filing_url_base": mod_xml.filing_base_url(cik, accession),
                    })
                    had_hit = True

            except Exception as e:
                print(f"         Text fetch/parse error: {e}")

            if had_hit:
                if debug:
                    print("         ✅ Hit; moving to next filing.")
                break

    if not rows_norm:
        if debug: print(f"[{ticker}] No parsed rows from last {hours_back}h.")
        return pd.DataFrame()

    df = pd.DataFrame(rows_norm)

    # Coerce filingDate to datetime for stable sorting
    if "filingDate" in df.columns:
        df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce")

    # Ensure consistent columns and order
    for c in KEEP_COL_ORDER:
        if c not in df.columns:
            df[c] = None
    df = df[KEEP_COL_ORDER]

    df = df.sort_values(["filingDate","record_type","form","source_url"], na_position="last").reset_index(drop=True)
    return df

# ----------------------- public: tidy df for main -----------------------
def get_sec_eth_holdings_df(
    tickers: Union[str, List[str]],
    hours_back: int = 24,
    forms: Union[str, List[str]] = ("8-K", "10-K", "10-Q"),
    limit: int = 200,
    max_docs: int = 6,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Return a standardized DataFrame with columns: date, ticker, asset, total_holdings.
    - Keeps your full parsing logic
    - Drops `period_eth_required` if present
    - Merges `eth` and `total_eth_holdings` → `total_holdings`
    - De-dupes by (ticker,date) keeping MAX holdings per day
    - No CSV saving
    """
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    if isinstance(forms, str):
        forms = [f.strip() for f in forms.split(",") if f.strip()]

    all_wide = []
    for t in tickers:
        if verbose:
            print(f"\n=== SEC ETH pipeline (last {hours_back}h): {t} ===")
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
        # return empty in target schema for smooth concat
        return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings"])

    df = pd.concat(all_wide, ignore_index=True)

    # Drop period_eth_required if present (typo-safe)
    if "period_eth_required" in df.columns:
        df = df.drop(columns=["period_eth_required"])

    # Merge eth and total_eth_holdings → total_holdings
    merged = pd.to_numeric(df.get("total_eth_holdings"), errors="coerce")
    merged = merged.fillna(pd.to_numeric(df.get("eth"), errors="coerce"))

    out = pd.DataFrame({
        "date": pd.to_datetime(df["filingDate"], errors="coerce").dt.date.astype("string"),
        "ticker": df["ticker"].astype(str).str.upper(),
        "asset": "ETH",
        "total_holdings": pd.to_numeric(merged, errors="coerce"),
    }).dropna(subset=["total_holdings"])

    # De-dupe by (ticker, date): keep the MAX holdings on that day
    # Sort by total_holdings descending, then groupby and take first to preserve all columns including asset
    out = (
        out.sort_values(["ticker", "date", "total_holdings"], ascending=[True, True, False])
           .groupby(["ticker", "date"], as_index=False)
           .first()
           .sort_values(["ticker", "date"])
           .reset_index(drop=True)
    )
    return out

# (Optional) quick manual test
if __name__ == "__main__":
    df = get_sec_eth_holdings_df("BTCS,ETHZ", hours_back=24, verbose=True)
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df.tail(20).to_string(index=False))
