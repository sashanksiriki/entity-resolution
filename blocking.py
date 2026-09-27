"""
High-recall blocking using numeric address tokens as the primary cross-script signal.

Key insight: Indian records in S2/S3 often have:
- Devanagari/Kannada/Tamil names (can't match S1 English text via string similarity)
- But the ADDRESS has the SAME house numbers, plot numbers, PIN codes as S1

Strategy:
1. Primary blocking key: country + any numeric sequence ≥ 3 digits from address + first Latin alpha token of name
2. Secondary: country + 5/6-digit postal code
3. Tertiary: token inverted index on names (for Latin-script S2/S3)
4. TF-IDF trigram on names (for Latin-script, handles typos)

No transliteration needed — numbers bridge the script gap.
"""
import collections
import re
from typing import Dict, List, Set, Tuple
import numpy as np

# ─── Utility ─────────────────────────────────────────────────────────────────

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "at", "to", "for", "by",
    "inc", "corp", "ltd", "llc", "llp", "pvt", "co", "svc", "ent", "grp",
    "sa", "sas", "sarl", "st", "rd", "ave", "blvd", "dr", "ln", "ct",
    "pvt", "ltd", "pvt",
}

RE_NUMS = re.compile(r"\d+")
RE_LATIN_WORD = re.compile(r"[a-z]{2,}")


def get_numeric_tokens(text: str) -> List[str]:
    """Extract all numeric sequences (length ≥ 2) from raw (uncleaned) text."""
    if not isinstance(text, str):
        return []
    return [m for m in RE_NUMS.findall(text) if len(m) >= 2]


def get_postal_code(text: str) -> str:
    """Extract 5-digit (US/France) or 6-digit (India) postal code."""
    for m in RE_NUMS.findall(text):
        if len(m) in (5, 6):
            return m
    return ""


def get_house_number(text: str) -> str:
    """Extract 1-4 digit numeric token (house number)."""
    for m in RE_NUMS.findall(text):
        if 1 <= len(m) <= 4:
            return m
    return ""


def get_first_latin_token(text: str) -> str:
    """Get first 4+ letter Latin-script word (script-agnostic name anchor)."""
    if not isinstance(text, str):
        return ""
    for m in RE_LATIN_WORD.finditer(text.lower()):
        word = m.group()
        if word not in STOPWORDS and len(word) >= 3:
            return word
    return ""


def get_latin_name_tokens(text: str) -> List[str]:
    """Get all meaningful Latin-script words from name."""
    if not isinstance(text, str):
        return []
    return [
        m.group() for m in RE_LATIN_WORD.finditer(text.lower())
        if m.group() not in STOPWORDS and len(m.group()) >= 3
    ]


def get_name_tokens(clean_name: str) -> List[str]:
    """Returns meaningful tokens from cleaned name (no stopwords)."""
    return [t for t in clean_name.split() if t not in STOPWORDS and len(t) >= 3]


def get_addr_tokens(addr_toks: List[str]) -> List[str]:
    """Returns meaningful address tokens."""
    result = []
    for t in addr_toks:
        if t.isdigit():
            if len(t) in (5, 6):
                result.append(t)
        elif len(t) >= 3 and t not in STOPWORDS:
            result.append(t)
    return result


def extract_postal_code(addr_tokens: List[str]) -> str:
    for tok in reversed(addr_tokens):
        if tok.isdigit() and len(tok) in (5, 6):
            return tok
    return ""


def extract_house_number(addr_tokens: List[str]) -> str:
    for tok in addr_tokens:
        if tok.isdigit() and len(tok) <= 4:
            return tok
    return ""


# ─── Token-based inverted index ──────────────────────────────────────────────

class TokenBlockingIndex:
    """
    Multi-key inverted index with separate key families:
    - NL: country + Latin name token (for Latin-script S2/S3)
    - NN: country + numeric address token + first Latin name token (cross-script)
    - PC: country + postal/PIN code (strong signal)
    - AB: country + sorted name bigram (transposition-invariant)
    - AT: country + address text token (city/locality)
    """

    def __init__(self, max_bucket_size: int = 300):
        self.index: Dict[str, List[str]] = collections.defaultdict(list)
        self.max_bucket_size = max_bucket_size

    def _make_keys(
        self,
        raw_name: str,
        raw_addr: str,
        clean_name: str,
        clean_addr: str,
        country: str,
    ) -> List[str]:
        keys = []

        # Postal code key (strongest signal)
        postal = get_postal_code(raw_addr)
        if not postal:
            postal = get_postal_code(raw_name)
        if postal:
            keys.append(f"PC|{country}|{postal}")

        # First Latin token from name
        first_latin = get_first_latin_token(raw_name)

        # Numeric tokens from raw address (handles non-Latin scripts)
        nums = get_numeric_tokens(raw_addr)
        for num in nums[:5]:  # top 5 numeric tokens
            if len(num) >= 3:
                # Long numeric + first latin word (cross-script key)
                if first_latin:
                    keys.append(f"NN|{country}|{num}|{first_latin}")
                # Just postal-like codes
                if len(num) in (5, 6):
                    keys.append(f"PC|{country}|{num}")
            elif len(num) >= 2 and first_latin:
                # Short house number + first name token
                keys.append(f"HN|{country}|{num}|{first_latin}")

        # Latin name token index (for English S2/S3)
        latin_toks = get_latin_name_tokens(raw_name)
        for tok in latin_toks[:5]:
            keys.append(f"NL|{country}|{tok}")

        # Sorted bigrams of Latin name tokens (transposition-invariant)
        if len(latin_toks) >= 2:
            for i in range(len(latin_toks)):
                for j in range(i + 1, min(i + 3, len(latin_toks))):
                    pair = tuple(sorted([latin_toks[i], latin_toks[j]]))
                    if pair[0] != pair[1]:
                        keys.append(f"AB|{country}|{pair[0]}|{pair[1]}")

        # Latin address text tokens (city/locality)
        addr_latin = [
            m.group() for m in RE_LATIN_WORD.finditer(raw_addr.lower())
            if len(m.group()) >= 4 and m.group() not in STOPWORDS
        ]
        for tok in addr_latin[:3]:
            keys.append(f"AT|{country}|{tok}")

        return keys

    def add(
        self,
        entity_id: str,
        raw_name: str,
        raw_addr: str,
        clean_name: str,
        clean_addr: str,
        country: str,
    ):
        for k in self._make_keys(raw_name, raw_addr, clean_name, clean_addr, country):
            self.index[k].append(entity_id)

    def prune(self, seed: int = 42):
        """Cap oversized buckets with a random sample instead of dropping them
        outright. Deleting the whole bucket for a common key (a big city name,
        a common business word) killed recall for every entity under that key
        — capping keeps *some* signal instead of none."""
        rng = np.random.default_rng(seed)
        for k in list(self.index.keys()):
            bucket = self.index[k]
            if len(bucket) > self.max_bucket_size:
                idx = rng.choice(len(bucket), size=self.max_bucket_size, replace=False)
                self.index[k] = [bucket[i] for i in idx]

    def query(
        self,
        raw_name: str,
        raw_addr: str,
        clean_name: str,
        clean_addr: str,
        country: str,
    ) -> Set[str]:
        keys = self._make_keys(raw_name, raw_addr, clean_name, clean_addr, country)
        candidates = set()
        for k in keys:
            if k in self.index:
                candidates.update(self.index[k])
        return candidates


# ─── TF-IDF based retrieval (Latin-only) ────────────────────────────────────

class TFIDFBlockingIndex:
    """
    Character-level char n-gram TF-IDF index.
    Only effective for Latin-script names; used as a supplement.
    """

    def __init__(self, top_k: int = 20, min_score: float = 0.15):
        self.top_k = top_k
        self.min_score = min_score
        self.entity_ids: List[str] = []
        self.countries: List[str] = []
        self.vectorizer = None
        self.matrix = None
        self._country_idx: Dict[str, List[int]] = {}

    def build(
        self,
        entity_ids: List[str],
        texts: List[str],
        countries: List[str],
    ):
        from sklearn.feature_extraction.text import TfidfVectorizer
        import scipy.sparse

        self.entity_ids = entity_ids
        self.countries = countries
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=2,
            max_df=0.90,
            sublinear_tf=True,
            max_features=150000,
        )
        self.matrix = self.vectorizer.fit_transform(texts)
        country_idx: Dict[str, List[int]] = collections.defaultdict(list)
        for i, c in enumerate(countries):
            country_idx[c].append(i)
        self._country_idx = country_idx

    def query_batch(
        self,
        texts: List[str],
        countries: List[str],
    ) -> List[List[str]]:
        """Batch query — returns a list of candidate lists.

        Groups queries by country and does ONE sparse matmul per country
        group instead of one dense (sub @ single_query).toarray() per row.
        The old version already vectorized the .transform() call but still
        densified a full country-sized subset on every single query — for
        2.2M S1 rows against a large country partition that never finishes
        in a reasonable time budget. This is the same math, just batched.
        """
        if self.vectorizer is None:
            return [[] for _ in texts]

        Q = self.vectorizer.transform(texts)
        n = len(texts)
        results: List[List[str]] = [[] for _ in range(n)]

        by_country: Dict[str, List[int]] = collections.defaultdict(list)
        for i, c in enumerate(countries):
            by_country[c].append(i)

        for ctry, qidxs in by_country.items():
            idxs = self._country_idx.get(ctry, [])
            if not idxs:
                continue
            sub = self.matrix[idxs]          # (m, f) sparse
            Qc = Q[qidxs]                     # (k, f) sparse
            scores = (sub @ Qc.T).tocsc()      # (m, k) sparse — avoid full densify

            for col, qi in enumerate(qidxs):
                col_scores = scores.getcol(col).toarray().ravel()
                nnz = int(np.count_nonzero(col_scores))
                if nnz == 0:
                    continue
                k = min(self.top_k, nnz)
                top_local = np.argpartition(col_scores, -k)[-k:]
                top_local = top_local[np.argsort(col_scores[top_local])[::-1]]
                cands = [
                    self.entity_ids[idxs[li]]
                    for li in top_local
                    if col_scores[li] >= self.min_score
                ]
                results[qi] = cands

        return results

    def query(self, text: str, country: str) -> List[str]:
        return self.query_batch([text], [country])[0]


# ─── Combined hybrid index ───────────────────────────────────────────────────

class HybridBlockingIndex:
    """
    Combines:
    1. TokenBlockingIndex — works for Latin AND cross-script (via numeric tokens)
    2. TFIDFBlockingIndex — additional recall for Latin-script name variants
    """

    def __init__(
        self,
        max_token_bucket: int = 300,
        tfidf_top_k: int = 15,
        tfidf_min_score: float = 0.15,
    ):
        self.token_idx = TokenBlockingIndex(max_bucket_size=max_token_bucket)
        self.tfidf_idx = TFIDFBlockingIndex(top_k=tfidf_top_k, min_score=tfidf_min_score)
        self._built_tfidf = False
        # Store raw data for TF-IDF
        self._raw_names: List[str] = []
        self._clean_names: List[str] = []
        self._clean_addrs: List[str] = []
        self._eids: List[str] = []
        self._countries: List[str] = []

    def add_records(
        self,
        entity_ids: List[str],
        raw_names: List[str],
        raw_addrs: List[str],
        clean_names: List[str],
        clean_addrs: List[str],
        countries: List[str],
    ):
        self._raw_names.extend(raw_names)
        self._clean_names.extend(clean_names)
        self._clean_addrs.extend(clean_addrs)
        self._eids.extend(entity_ids)
        self._countries.extend(countries)
        for eid, rn, ra, cn, ca, ct in zip(
            entity_ids, raw_names, raw_addrs, clean_names, clean_addrs, countries
        ):
            self.token_idx.add(eid, rn, ra, cn, ca, ct)

    def prune_token_index(self):
        self.token_idx.prune()

    def build_tfidf(self):
        """Build TF-IDF on accumulated records."""
        texts = [f"{n} {a}" for n, a in zip(self._clean_names, self._clean_addrs)]
        self.tfidf_idx.build(self._eids, texts, self._countries)
        self._built_tfidf = True

    def get_candidates(
        self,
        raw_name: str,
        raw_addr: str,
        clean_name: str,
        clean_addr: str,
        country: str,
    ) -> List[str]:
        token_cands = self.token_idx.query(raw_name, raw_addr, clean_name, clean_addr, country)
        tfidf_cands = set()
        if self._built_tfidf:
            combined = f"{clean_name} {clean_addr}"
            tfidf_cands = set(self.tfidf_idx.query(combined, country))
        return list(token_cands | tfidf_cands)

    def get_candidates_batch(
        self,
        raw_names: List[str],
        raw_addrs: List[str],
        clean_names: List[str],
        clean_addrs: List[str],
        countries: List[str],
    ) -> List[List[str]]:
        """Batched version of get_candidates — use this for anything processing
        more than a handful of S1 records (training pair mining, inference).
        Token-index lookups stay per-row (dict lookups, already fast); the
        TF-IDF half is issued as one batched call so its country-grouped
        matmul optimization actually kicks in."""
        n = len(raw_names)
        token_cands_list = [
            self.token_idx.query(
                raw_names[i], raw_addrs[i], clean_names[i], clean_addrs[i], countries[i]
            )
            for i in range(n)
        ]

        tfidf_cands_list: List[set] = [set() for _ in range(n)]
        if self._built_tfidf:
            combined = [f"{clean_names[i]} {clean_addrs[i]}" for i in range(n)]
            tfidf_results = self.tfidf_idx.query_batch(combined, countries)
            tfidf_cands_list = [set(r) for r in tfidf_results]

        return [list(token_cands_list[i] | tfidf_cands_list[i]) for i in range(n)]