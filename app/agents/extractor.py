"""Entity & constraint extraction agent.

Step 1: the LLM reads the question into fields (chemical, CAS, company, dates, ...).
Step 2: every name is matched to a value that really exists in the data. Nothing the user did not
        ask for is added, and if a named thing is not in the data we say so instead of guessing.
"""
from dataclasses import dataclass, field

from rapidfuzz import process, utils

from app.core import config, llm
from app.ingestion.loader import find_cas, normalize_cas
from app.storage import database

EXTRACT_PROMPT = (
    "You read questions about chemicals reported in cosmetic products. "
    "Return JSON with exactly these keys: "
    "chemical (chemical name or null), cas (CAS number or null), company (or null), brand (or null), "
    "product (product name or null), "
    "category (main product category such as 'Nail Products' or 'Baby Products', or null), "
    "subcategory (finer category such as 'Eye Shadow' or 'Lip Gloss/Shine', or null), "
    "reported_from (first year of a reporting date range, 4 digits, or null), "
    "reported_to (last year of a reporting date range, 4 digits, or null), "
    "discontinued_year (4-digit year the products were discontinued, or null), "
    "discontinued (true if the question is about discontinued products, false if only active ones, else null), "
    "chemical_removed (true if about chemicals that were removed/reformulated out of products, "
    "false if only chemicals still present, else null), "
    "topic (a descriptive idea from the question that none of the other keys capture, "
    "e.g. 'vitamin A ingredients' or 'sun protection', or null). "
    "Use null for anything the question does not mention. Never guess."
)

# Names the data spells differently. Each key expands to the terms we search for.
# Chemicals that share a CAS number are linked automatically, this list is for the ones that don't.
SYNONYMS = {
    "cocamide dea": ["cocamide dea", "cocamide diethanolamine"],
    "cocamide diethanolamine": ["cocamide dea", "cocamide diethanolamine"],
    "vitamin a": ["vitamin a", "retinol", "retinyl"],
    "retinol": ["vitamin a", "retinol", "retinyl"],
    "tio2": ["titanium dioxide"],
    "talcum": ["talc"],
    "bha": ["butylated hydroxyanisole"],
    "formalin": ["formaldehyde"],
    "ethanal": ["acetaldehyde"],
}

TEXT_FIELDS = {"chemical": "ChemicalName", "company": "CompanyName", "brand": "BrandName",
               "product": "ProductName", "category": "PrimaryCategory", "subcategory": "SubCategory"}
STRICT = ("chemical", "company", "brand", "product")   # named but not in the data -> stop
MANY_ALLOWED = ("chemical", "product")                 # several matches are fine, we use all of them


@dataclass
class Extraction:
    filters: dict
    topic: str | None = None
    warnings: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)
    clarify: list[dict] = field(default_factory=list)   # ambiguous names, the user has to pick
    fuzzy_used: bool = False
    raw: dict = field(default_factory=dict)              # what the LLM returned, for the trace


def match_values(term: str, column: str) -> tuple[list[str], str]:
    """Values in `column` that match `term`. Tries exact, then 'contains', then fuzzy."""
    values = database.distinct_values(column)
    t = term.strip().lower()
    exact = [v for v in values if v.lower() == t]
    if exact:
        return exact, "exact"
    if len(t) >= 4:
        contains = [v for v in values if t in v.lower()]
        if contains:
            # best looking match first
            contains.sort(key=lambda v: -process.extractOne(term, [v], processor=utils.default_process)[1])
            return contains, "contains"
    found = process.extract(term, values, limit=3, processor=utils.default_process,
                            score_cutoff=config.FUZZY_CUTOFF)
    if found:
        best = found[0][1]
        return [m[0] for m in found if best - m[1] <= 3], "fuzzy"
    return [], "none"


def _year(value) -> int | None:
    try:
        year = int(str(value)[:4])
    except (TypeError, ValueError):
        return None
    return year if 1990 <= year <= 2100 else None


def _bool(value) -> bool | None:
    # careful: `False or None` is None, which is how the first version lost "only active products"
    if value is True or value is False:
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return None


class ExtractionAgent:
    name = "extractor"

    def run(self, question: str) -> Extraction:
        warnings = []
        try:
            raw = llm.ask_llm(EXTRACT_PROMPT, question)
        except ValueError:
            raw = {}
            warnings.append("Could not read filters from the question, searched by meaning only")
        ext = self.resolve(question, raw)
        ext.warnings = warnings + ext.warnings
        return ext

    def resolve(self, question: str, raw: dict) -> Extraction:
        ext = Extraction(filters={}, raw=raw)
        f = ext.filters

        # ---- CAS: exact lookup, beats any name matching ----
        cas = find_cas(question) or (normalize_cas(raw["cas"]) if raw.get("cas") else None)
        if cas:
            names = database.chemicals_for_cas(cas)
            if names:
                f["cas"] = cas
                ext.assumptions.append(f"CAS {cas} is reported as: {', '.join(names)}")
                if raw.get("chemical"):
                    ext.assumptions.append("A CAS number and a chemical name were both given, used the CAS number")
            else:
                ext.not_found.append(f"CAS number {cas}")

        # ---- names ----
        for key, column in TEXT_FIELDS.items():
            asked = raw.get(key)
            if not asked or (key == "chemical" and "cas" in f):
                continue
            asked = str(asked).strip()
            if key == "chemical":
                f[key] = self._resolve_chemical(asked, ext)
            else:
                f[key] = self._resolve_name(key, asked, column, ext)
            if f[key] is None and key in STRICT and not any(c["field"] == key for c in ext.clarify):
                ext.not_found.append(f"{key} '{asked}'")

        # ---- dates and flags ----
        for key in ("reported_from", "reported_to", "discontinued_year"):
            if raw.get(key) is not None:
                f[key] = _year(raw[key])
                if f[key] is None:
                    ext.warnings.append(f"Could not read the year in '{key}', so it was ignored")
        if f.get("reported_from") and f.get("reported_to") and f["reported_from"] > f["reported_to"]:
            f["reported_from"], f["reported_to"] = f["reported_to"], f["reported_from"]
            ext.warnings.append("The date range was given backwards, swapped it")
        if f.get("reported_from") or f.get("reported_to"):
            ext.assumptions.append("Date ranges are checked on InitialDateReported, by year")
        for key in ("discontinued", "chemical_removed"):
            f[key] = _bool(raw.get(key))
        if f.get("discontinued_year"):
            f["discontinued"] = None                     # the year already implies discontinued
        if f.get("discontinued") is not None or f.get("discontinued_year"):
            ext.assumptions.append("'Discontinued' means the product has a DiscontinuedDate")
        if f.get("chemical_removed") is not None:
            ext.assumptions.append("'Removed' means the chemical has a ChemicalDateRemoved (reformulated)")

        # topic = the part of the question no filter covers. Skip it if it just repeats a filter.
        topic = str(raw.get("topic") or "").strip()
        named = {str(v).lower() for k, v in raw.items() if k != "topic" and v}
        if topic and topic.lower() not in named:
            ext.topic = " ".join(filter(None, [ext.topic, topic]))

        ext.filters = {k: v for k, v in f.items() if v not in (None, [])}
        return ext

    def _resolve_chemical(self, asked: str, ext: Extraction) -> list[str] | None:
        terms = SYNONYMS.get(asked.lower(), [asked])
        names, fuzzy = [], False
        for term in terms:
            found, how = match_values(term, "ChemicalName")
            fuzzy = fuzzy or how == "fuzzy"
            names += [n for n in found if n not in names]
        if not names:
            return None
        if fuzzy:
            ext.fuzzy_used = True
        if [n.lower() for n in names] != [asked.lower()]:
            shown = ", ".join(f"'{n}'" for n in names[:5]) + (" ..." if len(names) > 5 else "")
            ext.warnings.append(f"Matched chemical '{asked}' to {shown}")

        # chemicals reported under the same CAS number are the same chemical with another name
        same_cas = set()
        for cas in database.cas_for_chemicals(names):
            same_cas.update(database.chemicals_for_cas(cas))
        extra = sorted(same_cas - set(names))
        if extra:
            names += extra
            ext.assumptions.append(f"Also included {', '.join(extra)} because it has the same CAS number")
        return names

    def _resolve_name(self, key: str, asked: str, column: str, ext: Extraction):
        found, how = match_values(asked, column)
        if not found:
            if key not in STRICT:                        # e.g. a category we don't know: search by meaning
                ext.warnings.append(f"No {key} like '{asked}' in the data, searched by meaning instead")
                ext.topic = " ".join(filter(None, [ext.topic, asked]))
            return None
        if how == "fuzzy":
            ext.fuzzy_used = True

        if key in MANY_ALLOWED:
            picked = found[:50]
        elif len(found) > 1 and key in STRICT:
            ext.clarify.append({"field": key, "asked": asked, "options": found[:5]})
            return None
        else:
            picked = found[0]
            if len(found) > 1:
                ext.warnings.append(f"'{asked}' matches several {key} values, used '{picked}'")

        first = picked[0] if isinstance(picked, list) else picked
        if first.lower() != asked.lower():
            more = f" (+{len(picked) - 1} more)" if isinstance(picked, list) and len(picked) > 1 else ""
            ext.warnings.append(f"Matched {key} '{asked}' to '{first}'{more}")
        return picked
