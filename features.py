"""
Rich pairwise feature set for the matching classifier.
Handles cross-script matching (English S1 vs Hindi/Kannada/Tamil S2/S3)
by computing features on BOTH raw and cleaned text.
"""
import re
from typing import Dict, List
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .blocking import (
    get_name_tokens, get_addr_tokens, extract_postal_code,
    extract_house_number, get_numeric_tokens, get_first_latin_token,
    RE_NUMS,
)

_RE_ALPHA_NUM = re.compile(r"[a-z0-9]{2,}")


def _jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def _token_f1(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    p = inter / len(b)
    r = inter / len(a)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _numeric_overlap(raw_a: str, raw_b: str) -> float:
    """Jaccard of numeric token sets (script-agnostic address signal)."""
    na = set(RE_NUMS.findall(raw_a) if isinstance(raw_a, str) else [])
    nb = set(RE_NUMS.findall(raw_b) if isinstance(raw_b, str) else [])
    # Focus on multi-digit numbers (2+ digits) to avoid noise
    na = {n for n in na if len(n) >= 2}
    nb = {n for n in nb if len(n) >= 2}
    return _jaccard(na, nb)


def _long_numeric_overlap(raw_a: str, raw_b: str) -> float:
    """Jaccard of long (3+ digit) numeric tokens — plot/flat/PIN numbers."""
    na = set(m for m in RE_NUMS.findall(raw_a or "") if len(m) >= 3)
    nb = set(m for m in RE_NUMS.findall(raw_b or "") if len(m) >= 3)
    return _jaccard(na, nb)


def _latin_token_overlap(raw_a: str, raw_b: str) -> float:
    """Jaccard of lowercase Latin tokens (works even if one name is mixed script)."""
    ta = set(_RE_ALPHA_NUM.findall(raw_a.lower()) if isinstance(raw_a, str) else [])
    tb = set(_RE_ALPHA_NUM.findall(raw_b.lower()) if isinstance(raw_b, str) else [])
    return _jaccard(ta, tb)


def compute_pair_features(
    s1_name: str,       # cleaned
    s1_addr: str,       # cleaned
    s1_country: str,
    cand_name: str,     # cleaned
    cand_addr: str,     # cleaned
    cand_country: str,
    cand_id: str,
    s1_raw_name: str = "",    # raw (uncleaned)
    s1_raw_addr: str = "",
    cand_raw_name: str = "",
    cand_raw_addr: str = "",
) -> Dict[str, float]:
    s1_name_toks = s1_name.split()
    cand_name_toks = cand_name.split()
    s1_addr_toks = s1_addr.split()
    cand_addr_toks = cand_addr.split()

    # ── Name fuzzy similarities (cleaned text) ────────────────────────────────
    name_ratio = fuzz.ratio(s1_name, cand_name) / 100.0
    name_partial = fuzz.partial_ratio(s1_name, cand_name) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
    name_token_set = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
    name_jw = JaroWinkler.similarity(s1_name, cand_name)

    # ── Name token overlap (cleaned) ──────────────────────────────────────────
    s1_ns = set(s1_name_toks)
    cand_ns = set(cand_name_toks)
    name_jaccard = _jaccard(s1_ns, cand_ns)
    name_f1 = _token_f1(s1_ns, cand_ns)

    s1_nm = set(get_name_tokens(s1_name))
    cand_nm = set(get_name_tokens(cand_name))
    name_meaningful_jaccard = _jaccard(s1_nm, cand_nm)
    name_meaningful_f1 = _token_f1(s1_nm, cand_nm)

    # First token prefix ratio (abbreviation signal)
    s1_first = s1_name_toks[0] if s1_name_toks else ""
    cand_first = cand_name_toks[0] if cand_name_toks else ""
    pfx = sum(1 for a, b in zip(s1_first, cand_first) if a == b)
    denom = max(len(s1_first), len(cand_first))
    first_token_prefix_ratio = pfx / denom if denom else 0.0

    # ── Latin token overlap on RAW name (cross-script) ───────────────────────
    latin_name_overlap = _latin_token_overlap(
        s1_raw_name or s1_name,
        cand_raw_name or cand_name,
    )

    # ── Address fuzzy similarities ────────────────────────────────────────────
    addr_ratio = fuzz.ratio(s1_addr, cand_addr) / 100.0
    addr_partial = fuzz.partial_ratio(s1_addr, cand_addr) / 100.0
    addr_token_sort = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
    addr_token_set = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0

    # ── Address token overlap (cleaned) ──────────────────────────────────────
    s1_as = set(s1_addr_toks)
    cand_as = set(cand_addr_toks)
    addr_jaccard = _jaccard(s1_as, cand_as)
    addr_f1 = _token_f1(s1_as, cand_as)

    s1_am = set(get_addr_tokens(s1_addr_toks))
    cand_am = set(get_addr_tokens(cand_addr_toks))
    addr_meaningful_jaccard = _jaccard(s1_am, cand_am)

    # Tail overlap (city/region)
    s1_tail = set(s1_addr_toks[-3:]) if len(s1_addr_toks) >= 3 else set(s1_addr_toks)
    cand_tail = set(cand_addr_toks[-3:]) if len(cand_addr_toks) >= 3 else set(cand_addr_toks)
    tail_overlap = _jaccard(s1_tail, cand_tail)

    # ── Cross-script numeric signals (RAW addresses) ──────────────────────────
    numeric_overlap = _numeric_overlap(
        s1_raw_addr or s1_addr,
        cand_raw_addr or cand_addr,
    )
    long_numeric_overlap = _long_numeric_overlap(
        s1_raw_addr or s1_addr,
        cand_raw_addr or cand_addr,
    )

    # ── Structural: house number ──────────────────────────────────────────────
    s1_h = extract_house_number(s1_addr_toks)
    cand_h = extract_house_number(cand_addr_toks)
    same_house = 1.0 if (s1_h and cand_h and s1_h == cand_h) else 0.0
    diff_house = 1.0 if (s1_h and cand_h and s1_h != cand_h) else 0.0
    has_house_s1 = 1.0 if s1_h else 0.0
    has_house_cand = 1.0 if cand_h else 0.0

    # ── Structural: postal code ───────────────────────────────────────────────
    s1_post = extract_postal_code(s1_addr_toks)
    cand_post = extract_postal_code(cand_addr_toks)
    same_postal = 1.0 if (s1_post and cand_post and s1_post == cand_post) else 0.0
    diff_postal = 1.0 if (s1_post and cand_post and s1_post != cand_post) else 0.0
    has_postal_s1 = 1.0 if s1_post else 0.0
    has_postal_cand = 1.0 if cand_post else 0.0

    # ── Metadata ──────────────────────────────────────────────────────────────
    same_country = 1.0 if s1_country == cand_country else 0.0
    is_source2 = 1.0 if cand_id.startswith("S2-") else 0.0
    is_source3 = 1.0 if cand_id.startswith("S3-") else 0.0

    # ── Length ratios ─────────────────────────────────────────────────────────
    def len_ratio(a, b):
        mx = max(len(a), len(b))
        return min(len(a), len(b)) / mx if mx else 1.0

    name_len_ratio = len_ratio(s1_name, cand_name)
    addr_len_ratio = len_ratio(s1_addr, cand_addr)

    # ── Composite signals ─────────────────────────────────────────────────────
    name_composite = max(name_token_set, name_meaningful_jaccard, name_jw, latin_name_overlap)
    addr_composite = max(addr_token_set, addr_meaningful_jaccard, numeric_overlap)
    overall = 0.50 * name_composite + 0.35 * addr_composite + 0.15 * tail_overlap

    return {
        # Name cleaned
        "name_ratio": name_ratio,
        "name_partial": name_partial,
        "name_token_sort": name_token_sort,
        "name_token_set": name_token_set,
        "name_jw": name_jw,
        "name_jaccard": name_jaccard,
        "name_f1": name_f1,
        "name_meaningful_jaccard": name_meaningful_jaccard,
        "name_meaningful_f1": name_meaningful_f1,
        "first_token_prefix_ratio": first_token_prefix_ratio,
        # Cross-script name
        "latin_name_overlap": latin_name_overlap,
        # Address cleaned
        "addr_ratio": addr_ratio,
        "addr_partial": addr_partial,
        "addr_token_sort": addr_token_sort,
        "addr_token_set": addr_token_set,
        "addr_jaccard": addr_jaccard,
        "addr_f1": addr_f1,
        "addr_meaningful_jaccard": addr_meaningful_jaccard,
        "tail_overlap": tail_overlap,
        # Cross-script numeric
        "numeric_overlap": numeric_overlap,
        "long_numeric_overlap": long_numeric_overlap,
        # Structural
        "same_house": same_house,
        "diff_house": diff_house,
        "has_house_s1": has_house_s1,
        "has_house_cand": has_house_cand,
        "same_postal": same_postal,
        "diff_postal": diff_postal,
        "has_postal_s1": has_postal_s1,
        "has_postal_cand": has_postal_cand,
        # Metadata
        "same_country": same_country,
        "is_source2": is_source2,
        "is_source3": is_source3,
        # Length
        "name_len_ratio": name_len_ratio,
        "addr_len_ratio": addr_len_ratio,
        # Composite
        "name_composite": name_composite,
        "addr_composite": addr_composite,
        "overall_composite": overall,
    }