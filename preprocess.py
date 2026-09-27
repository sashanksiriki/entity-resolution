import re
import unicodedata

RE_PUNCT = re.compile(r"[^\w\s]")
RE_SPACES = re.compile(r"\s+")
RE_AMP = re.compile(r"&")

# Comprehensive legal suffixes: US, India, and France
LEGAL_SUFFIXES = {
    # US / UK
    "incorporated": "inc",
    "corporation": "corp",
    "corp": "corp",
    "inc": "inc",
    "limited": "ltd",
    "ltd": "ltd",
    "llc": "llc",
    "llp": "llp",
    "co": "co",
    "company": "co",
    "enterprises": "ent",
    "enterprise": "ent",
    "services": "svc",
    "service": "svc",
    "group": "grp",
    "international": "intl",
    "associates": "assoc",
    "association": "assoc",
    "solutions": "sol",
    "technologies": "tech",
    "technology": "tech",
    # India
    "private": "pvt",
    "pvt": "pvt",
    # France
    "sarl": "sarl",
    "sa": "sa",
    "sas": "sas",
    "sasu": "sas",
    "eurl": "eurl",
    "snc": "snc",
    "sci": "sci",
    "cie": "co",
}

# Street & Geographic type mappings (US, India, France)
STREET_MAPPINGS = {
    # English
    "street": "st",
    "avenue": "ave",
    "road": "rd",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "highway": "hwy",
    "circle": "cir",
    "place": "pl",
    "square": "sq",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "suite": "ste",
    "apartment": "apt",
    "floor": "fl",
    "building": "bldg",
    # French
    "rue": "rue",
    "chemin": "ch",
    "ch": "ch",
    "route": "rte",
    "rte": "rte",
    "allee": "all",
    "all": "all",
    "passage": "pass",
    "quai": "quai",
    "cours": "crs",
    "bvd": "blvd",
    "bd": "blvd",
    "av": "ave",
    "cedex": "cedex",
}


def strip_accents(text: str) -> str:
    """Normalize unicode: Café -> Cafe, Société -> Societe (critical for France)."""
    if not isinstance(text, str):
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def clean_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        return ""
    text = strip_accents(name).lower()
    text = RE_AMP.sub(" and ", text)
    text = RE_PUNCT.sub(" ", text)
    tokens = RE_SPACES.split(text.strip())
    tokens = [LEGAL_SUFFIXES.get(t, t) for t in tokens if t]
    return " ".join(tokens)


def clean_address(addr: str) -> str:
    if not isinstance(addr, str) or not addr:
        return ""
    text = strip_accents(addr).lower()
    text = RE_AMP.sub(" and ", text)
    text = RE_PUNCT.sub(" ", text)

    cleaned = []
    for token in RE_SPACES.split(text.strip()):
        if not token:
            continue
        if token.isdigit():
            # Standardize numeric house/postal tokens (strip leading zeroes)
            cleaned.append(str(int(token)))
        else:
            cleaned.append(STREET_MAPPINGS.get(token, token))

    return " ".join(cleaned)


def make_combined_text(name: str, addr: str) -> str:
    """Combine name and address for TF-IDF indexing."""
    return f"{name} {addr}".strip()