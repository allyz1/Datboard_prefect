# sec_deals/core/extract.py
from __future__ import annotations
from lxml import etree

def _clean_text(s: str) -> str:
    return " ".join((s or "").replace("\xa0"," ").replace("\u2009"," ").replace("\u202f"," ").split())

def html_to_blocks(html_bytes: bytes) -> list[str]:
    """
    Parse HTML and return de-duplicated paragraph-like text blocks suitable
    for regex classification. Filters to medium-length spans to avoid noise.
    """
    parser = etree.HTMLParser(recover=True, huge_tree=True)
    root = etree.HTML(html_bytes, parser=parser)
    if root is None:
        return []

    blocks: list[str] = []
    for node in root.xpath(".//p | .//li | .//td | .//th | .//div"):
        txt = _clean_text("".join(node.itertext()))
        if not txt:
            continue
        if 40 <= len(txt) <= 1500:
            blocks.append(txt)

    # De-dup while preserving order
    seen, out = set(), []
    for b in blocks:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out
