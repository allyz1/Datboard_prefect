# sec_deals/core/select.py
from __future__ import annotations
import re
from enum import Enum
from typing import Iterable, List, Tuple
import pandas as pd

def is_html_name(name: str) -> bool:
    n = str(name).lower()
    return n.endswith((".htm", ".html", ".xhtml"))

class Mode(str, Enum):
    PIPES = "pipes"
    REG_DIRECT = "reg_direct"
    WARRANTS = "warrants"

# Tokens that often signal useful exhibits across modes
COMMON_HINTS = (
    "prospectus", "risk factors", "plan of distribution", "selling stockholder", "selling shareholder",
)

# Mode-specific preference tokens
def _form_tokens_for_ranking(form: str, mode: Mode) -> Iterable[str]:
    f = (form or "").upper()
    if mode == Mode.PIPES:
        if f.startswith("8-K"):
            return ("ex10","ex-10","ex4","ex-4","securities purchase","subscription",
                    "registration rights","warrant","note","debenture","item")
        if f.startswith(("S-1","S-3","424B3","424B7")):
            return ("selling stockholder","selling shareholder","resale") + COMMON_HINTS
    elif mode == Mode.REG_DIRECT:
        if f.startswith("424B5"):
            return ("prospectus supplement","registered direct","purchase price","use of proceeds") + COMMON_HINTS
        if f.startswith("8-K"):
            return ("registered direct","prospectus supplement","pricing") + COMMON_HINTS
    elif mode == Mode.WARRANTS:
        # Prefer EX-4.* and any explicit warrant files
        return ("ex4","ex-4","warrant","pre-funded warrant","form of warrant","exercise price","coverage")
    return ()

PIPE_EXHIBIT_HINT_TOKENS = (
    "securities purchase", "subscription", "registration rights", "warrant", "note", "debenture",
    "selling stockholder", "selling shareholder", "resale"
)

def _exhibit_hint(name: str) -> str:
    n = name.lower()
    m = re.search(r"(ex[-_]?(\d{1,2}|99|4|10)[\.\-_]?\d*)", n)
    return m.group(1) if m else ""

def candidate_html_urls(
    files_df: pd.DataFrame,
    primary_doc: str,
    filing_form: str,
    mode: Mode,
    max_docs: int = 6,
) -> List[Tuple[str, str]]:
    """
    Return ranked [(url, exhibit_hint)] for HTML-likes in a filing, tuned per `mode`.
    """
    if files_df is None or files_df.empty:
        return []

    df = files_df[files_df["name"].apply(is_html_name)].copy()
    if df.empty:
        return []

    # Drop EX-5 legal opinions which are noise for deal economics
    mask_ex5 = df["name"].str.contains(r"\bex[-_]?5(?=[\.\-_]|$)", case=False, regex=True, na=False)
    df = df[~mask_ex5].copy()

    prim = (primary_doc or "").lower()
    prefer = tuple(tok.lower() for tok in _form_tokens_for_ranking(filing_form, mode))

    def rank(name: str) -> Tuple[int, int, int, int, int, int, int]:
        n = name.lower()
        r0 = 0 if n == prim else 1                                         # primary doc first
        r1 = 0 if any(tok in n for tok in prefer) else 1                    # mode-specific hints
        r2 = 0 if re.search(r"ex[-_]?10", n) else (1 if re.search(r"ex[-_]?4", n) else 2)  # EX-10 > EX-4 > others
        r3 = 0 if n.endswith((".xhtml",".htm",".html")) else 1              # html-like
        r4 = 1 if ("press" in n or "ex99" in n) else 0                      # de-prioritize press/EX-99
        # Penalize EX-1.* (underwriting) unless also clearly deal-ish
        r5 = 1 if (re.search(r"ex[-_]?1", n) and not re.search(r"(securities[_\s-]?purchase|subscription|registration[_\s-]?rights|reg[_\s-]?d|rule[_\s-]?144a|registered[_\s-]?direct)", n)) else 0
        r6 = 0 if any(tok in n for tok in PIPE_EXHIBIT_HINT_TOKENS) else 1  # general PIPE-ish cues
        return (r0, r1, r2, r3, r4, r5, r6)

    df["rank_tuple"] = df["name"].apply(rank)
    df = df.sort_values(["rank_tuple","name"]).head(max_docs)

    out: List[Tuple[str, str]] = []
    for _, r in df.iterrows():
        out.append((r["url"], _exhibit_hint(str(r["name"]))))
    return out
