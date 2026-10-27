"""Structured query agent: deterministic SQL over the records table.

Every query keeps the SQL text and parameters so the answer can show exactly how it was derived.
"""
from dataclasses import dataclass, field

from app.core import config
from app.storage import database

# filter key -> column, for the "= ?" filters
EQUALS = {"cas": "CasNumber", "company": "CompanyName", "brand": "BrandName",
          "category": "PrimaryCategory", "subcategory": "SubCategory"}
# filter key -> column, for filters that can hold several values
IN_LIST = {"chemical": "ChemicalName", "product": "ProductName"}


@dataclass
class QueryResult:
    rows: list[dict] = field(default_factory=list)
    total: int = 0
    queries: list[dict] = field(default_factory=list)    # {"sql": ..., "params": [...]}
    table: list[dict] = field(default_factory=list)      # aggregate output (trends)
    issues: list[dict] = field(default_factory=list)     # data quality findings


def build_where(f: dict) -> tuple[str, list]:
    clauses, params = [], []
    for key, column in EQUALS.items():
        if f.get(key):
            clauses.append(f"{column} = ?")
            params.append(f[key])
    for key, column in IN_LIST.items():
        if f.get(key):
            values = f[key] if isinstance(f[key], list) else [f[key]]
            clauses.append(f"{column} IN ({','.join('?' * len(values))})")
            params += values
    if f.get("reported_from"):
        clauses.append("substr(InitialDateReported, 1, 4) >= ?")
        params.append(str(f["reported_from"]))
    if f.get("reported_to"):
        clauses.append("substr(InitialDateReported, 1, 4) <= ?")
        params.append(str(f["reported_to"]))
    if f.get("discontinued_year"):
        clauses.append("substr(DiscontinuedDate, 1, 4) = ?")
        params.append(str(f["discontinued_year"]))
    elif f.get("discontinued") is True:
        clauses.append("DiscontinuedDate IS NOT NULL")
    elif f.get("discontinued") is False:
        clauses.append("DiscontinuedDate IS NULL")
    if f.get("chemical_removed") is True:
        clauses.append("ChemicalDateRemoved IS NOT NULL")
    elif f.get("chemical_removed") is False:
        clauses.append("ChemicalDateRemoved IS NULL")
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def date_issues(r: dict) -> list[dict]:
    """Dates on one row that contradict each other (dates are stored as YYYY-MM-DD, so text compare works)."""
    first = r.get("InitialDateReported")
    if not first:
        return []
    checks = [("DiscontinuedDate", "discontinued"), ("MostRecentDateReported", "most recent report"),
              ("ChemicalDateRemoved", "chemical removed")]
    return [{"RowId": r["RowId"], "CDPHId": r["CDPHId"], "type": "date_conflict",
             "detail": f"{label} {r[col]} is before first reported {first}"}
            for col, label in checks if r.get(col) and r[col] < first]


class StructuredQueryAgent:
    name = "sql"

    def _run(self, result: QueryResult, sql: str, params: list) -> list[dict]:
        result.queries.append({"sql": sql, "params": params})
        with database.connect() as c:
            return [dict(r) for r in c.execute(sql, params)]

    def fetch(self, filters: dict, limit: int = config.MAX_SQL_ROWS) -> QueryResult:
        """All rows matching the filters (up to `limit`) plus the true total."""
        result = QueryResult()
        where, params = build_where(filters)
        result.total = self._run(result, f"SELECT COUNT(*) AS n FROM records{where}", params)[0]["n"]
        result.rows = self._run(
            result, f"SELECT * FROM records{where} ORDER BY CDPHId, RowId LIMIT {int(limit)}", params)
        return result

    def trend(self, filters: dict) -> QueryResult:
        """Reporting counts per year, plus a few example rows per year so the answer can cite something."""
        result = QueryResult()
        where, params = build_where(filters)
        year_where = where + (" AND " if where else " WHERE ") + "InitialDateReported IS NOT NULL"
        reported = self._run(result, (
            "SELECT substr(InitialDateReported, 1, 4) AS year, COUNT(DISTINCT CDPHId) AS products_reported, "
            "COUNT(*) AS chemical_reports "
            f"FROM records{year_where} GROUP BY year ORDER BY year"), params)

        disc_where = where + (" AND " if where else " WHERE ") + "DiscontinuedDate IS NOT NULL"
        discontinued = {r["year"]: r["n"] for r in self._run(result, (
            "SELECT substr(DiscontinuedDate, 1, 4) AS year, COUNT(DISTINCT CDPHId) AS n "
            f"FROM records{disc_where} GROUP BY year"), params)}

        rem_where = where + (" AND " if where else " WHERE ") + "ChemicalDateRemoved IS NOT NULL"
        removed = {r["year"]: r["n"] for r in self._run(result, (
            "SELECT substr(ChemicalDateRemoved, 1, 4) AS year, COUNT(*) AS n "
            f"FROM records{rem_where} GROUP BY year"), params)}

        years = sorted({r["year"] for r in reported} | set(discontinued) | set(removed))
        by_year = {r["year"]: r for r in reported}
        result.table = [{"year": y,
                         "products_reported": by_year.get(y, {}).get("products_reported", 0),
                         "chemical_reports": by_year.get(y, {}).get("chemical_reports", 0),
                         "products_discontinued": discontinued.get(y, 0),
                         "chemicals_removed": removed.get(y, 0)} for y in years]
        result.total = sum(r["chemical_reports"] for r in result.table)

        # two example rows per year, for citations
        examples = self._run(result, (
            "SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY substr(InitialDateReported, 1, 4) "
            f"ORDER BY RowId) AS rn FROM records{year_where}) WHERE rn <= 2 ORDER BY InitialDateReported"),
            params)
        result.rows = [{k: v for k, v in r.items() if k != "rn"} for r in examples]
        return result

    def data_quality(self, filters: dict) -> QueryResult:
        """Look for conflicting or missing data inside the filtered rows."""
        result = self.fetch(filters, limit=100_000)
        rows = result.rows
        issues = []
        for r in rows:
            issues += date_issues(r)
            if not r["CasNumber"] or r["CasNumber"] == "0":
                kind = "trade_secret" if r["ChemicalName"] == "Trade Secret" else "missing_cas"
                issues.append({"RowId": r["RowId"], "CDPHId": r["CDPHId"], "type": kind,
                               "detail": f"no CAS number for '{r['ChemicalName']}'"})

        # one CAS number reported under different chemical names
        names_by_cas: dict[str, set] = {}
        for r in rows:
            if r["CasNumber"] and r["CasNumber"] != "0":
                names_by_cas.setdefault(r["CasNumber"], set()).add(r["ChemicalName"])
        for cas, names in names_by_cas.items():
            if len(names) > 1:
                row = next(r for r in rows if r["CasNumber"] == cas)
                issues.append({"RowId": row["RowId"], "CDPHId": row["CDPHId"], "type": "cas_many_names",
                               "detail": f"CAS {cas} has {len(names)} names: {'; '.join(sorted(names))}"})

        # ChemicalCount says how many chemicals the product reported, compare with the rows we have
        per_product: dict[int, set] = {}
        for r in rows:
            per_product.setdefault(r["CDPHId"], set()).add(r["ChemicalId"])
        for r in rows:
            have = len(per_product.get(r["CDPHId"], ()))
            if r["ChemicalCount"] is not None and r["ChemicalCount"] < have:
                issues.append({"RowId": r["RowId"], "CDPHId": r["CDPHId"], "type": "count_mismatch",
                               "detail": f"ChemicalCount is {r['ChemicalCount']} but {have} chemicals are listed"})
                per_product[r["CDPHId"]] = set()          # report each product once

        result.issues = issues
        flagged = {i["RowId"] for i in issues}
        result.rows = [r for r in rows if r["RowId"] in flagged]
        summary: dict[str, int] = {}
        for i in issues:
            summary[i["type"]] = summary.get(i["type"], 0) + 1
        result.table = [{"issue": k, "rows": v} for k, v in sorted(summary.items())]
        return result
