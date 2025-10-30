# core/persistence.py
from __future__ import annotations
import os, json, sqlite3, threading
from typing import Dict, Any, Optional, List
from datetime import datetime

_DB_LOCK = threading.Lock()
_DB_PATH = os.environ.get("ECHOoz_DB_PATH", "echooz_results.sqlite")

# -------------------------------
# Bestehendes Result-Archiv (beibehalten)
# -------------------------------
DDL_ANALYSIS = """
CREATE TABLE IF NOT EXISTS analysis_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset_id TEXT,
  file_sha256 TEXT NOT NULL,
  pipeline_fingerprint TEXT NOT NULL,
  chart_of_accounts TEXT,
  period_start TEXT,
  period_end TEXT,
  coverage REAL,
  balanced INTEGER,
  result_json TEXT NOT NULL,
  created_at_utc TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_fingerprint
  ON analysis_results(file_sha256, pipeline_fingerprint);
"""

# -------------------------------
# NEU: Snapshots für Bilanz & GuV + Minimal-Pivot
# -------------------------------
DDL_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS bs_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_sha256 TEXT NOT NULL,
    company_id  TEXT,
    coa         TEXT,
    period_start TEXT,
    period_end   TEXT,
    non_current REAL,
    current REAL,
    total_assets REAL,
    equity REAL,
    provisions REAL,
    liabilities_short_term REAL,
    liabilities_long_term REAL,
    total_liabilities_equity REAL,
    balanced INTEGER,
    difference REAL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bs_file ON bs_snapshots(file_sha256);
CREATE INDEX IF NOT EXISTS idx_bs_period ON bs_snapshots(period_start, period_end);

CREATE TABLE IF NOT EXISTS pl_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_sha256 TEXT NOT NULL,
    company_id  TEXT,
    coa         TEXT,
    period_start TEXT,
    period_end   TEXT,
    revenue REAL,
    material_expenses REAL,
    personnel_expenses REAL,
    other_operating_expenses REAL,
    other_operating_income REAL,
    depreciation REAL,
    interest_expenses REAL,
    result REAL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pl_file ON pl_snapshots(file_sha256);
CREATE INDEX IF NOT EXISTS idx_pl_period ON pl_snapshots(period_start, period_end);
"""

def _conn():
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def _apply_ddl(ddl: str) -> None:
    with _DB_LOCK:
        conn = _conn()
        try:
            for stmt in ddl.strip().split(";"):
                s = stmt.strip()
                if s:
                    conn.execute(s)
            conn.commit()
        finally:
            conn.close()

def init_db() -> None:
    """Ruft bei App-Start genau einmal auf: legt alle nötigen Tabellen/Indizes an."""
    _apply_ddl(DDL_ANALYSIS)
    _apply_ddl(DDL_SNAPSHOTS)

# -------------------------------
# Bestehende Funktionen (unverändert lassen)
# -------------------------------
def persist_result(payload: Dict[str, Any]) -> int:
    """Idempotent: unique by (file_sha256, pipeline_fingerprint)."""
    meta = payload.get("aggregation", {}).get("meta", {})
    period = payload.get("aggregation", {}).get("period", {})
    file_sha256 = meta.get("file_sha256", "")
    pipeline_fingerprint = meta.get("pipeline_fingerprint", "")
    coa = meta.get("coa", "")
    coverage = meta.get("coverage", None)
    balanced = int(
        payload.get("aggregation", {})
               .get("statements", {})
               .get("balance_sheet", {})
               .get("check", {})
               .get("balanced", False)
    )
    period_start, period_end = period.get("start_date",""), period.get("end_date","")
    created_at = meta.get("created_at", "")

    with _DB_LOCK:
        conn = _conn()
        try:
            # Try find existing
            cur = conn.execute(
                "SELECT id FROM analysis_results WHERE file_sha256=? AND pipeline_fingerprint=?",
                (file_sha256, pipeline_fingerprint),
            )
            row = cur.fetchone()
            if row:
                return int(row[0])
            # Insert new
            cur = conn.execute(
                "INSERT INTO analysis_results(dataset_id,file_sha256,pipeline_fingerprint,chart_of_accounts,"
                "period_start,period_end,coverage,balanced,result_json,created_at_utc)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    payload.get("request_id") or "",
                    file_sha256,
                    pipeline_fingerprint,
                    coa,
                    period_start, period_end,
                    coverage, balanced,
                    json.dumps(payload, ensure_ascii=False),
                    created_at,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

def fetch_result_by_id(result_id: int) -> Optional[Dict[str, Any]]:
    with _DB_LOCK:
        conn = _conn()
        try:
            cur = conn.execute("SELECT result_json FROM analysis_results WHERE id=?", (result_id,))
            row = cur.fetchone()
            return json.loads(row[0]) if row else None
        finally:
            conn.close()

# -------------------------------
# NEU: Snapshots & Pivot API (benutzt dein Echooz-Contract JSON)
# -------------------------------
def write_snapshots(
    *,
    file_sha256: str,
    company_id: Optional[str],
    coa: Optional[str],
    period_start: Optional[str],
    period_end: Optional[str],
    aggregation: Dict[str, Any],
) -> None:
    stmts = aggregation.get("statements", {})
    bs = (stmts.get("balance_sheet") or {})
    pl = (stmts.get("profit_and_loss") or {})
    bs_assets = (bs.get("assets") or {})
    bs_le = (bs.get("liabilities_equity") or {})
    bs_check = (bs.get("check") or {})
    created_at = datetime.utcnow().isoformat() + "Z"

    with _DB_LOCK:
        conn = _conn()
        try:
            # Bilanz
            conn.execute("""
                INSERT INTO bs_snapshots (
                    file_sha256, company_id, coa, period_start, period_end,
                    non_current, current, total_assets,
                    equity, provisions, liabilities_short_term, liabilities_long_term, total_liabilities_equity,
                    balanced, difference, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_sha256, company_id, (coa or None), period_start, period_end,
                float((bs_assets.get("non_current") or {}).get("sum", 0.0) or 0.0),
                float((bs_assets.get("current") or {}).get("sum", 0.0) or 0.0),
                float(bs_assets.get("total_assets", 0.0) or 0.0),
                float(bs_le.get("equity", 0.0) or 0.0),
                float(bs_le.get("provisions", 0.0) or 0.0),
                float(bs_le.get("liabilities_short_term", 0.0) or 0.0),
                float(bs_le.get("liabilities_long_term", 0.0) or 0.0),
                float(bs_le.get("total_liabilities_equity", 0.0) or 0.0),
                1 if (bs_check.get("balanced") is True) else 0,
                float(bs_check.get("difference", 0.0) or 0.0),
                created_at
            ))

            # GuV
            conn.execute("""
                INSERT INTO pl_snapshots (
                    file_sha256, company_id, coa, period_start, period_end,
                    revenue, material_expenses, personnel_expenses,
                    other_operating_expenses, other_operating_income,
                    depreciation, interest_expenses, result, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_sha256, company_id, (coa or None), period_start, period_end,
                float(pl.get("revenue", 0.0) or 0.0),
                float(pl.get("material_expenses", 0.0) or 0.0),
                float(pl.get("personnel_expenses", 0.0) or 0.0),
                float(pl.get("other_operating_expenses", 0.0) or 0.0),
                float(pl.get("other_operating_income", 0.0) or 0.0),
                float(pl.get("depreciation", 0.0) or 0.0),
                float(pl.get("interest_expenses", 0.0) or 0.0),
                float(pl.get("result", 0.0) or 0.0),
                created_at
            ))

            conn.commit()
        finally:
            conn.close()

def read_snapshots(file_sha256: str) -> Dict[str, List[Dict[str, Any]]]:
    with _DB_LOCK:
        conn = _conn()
        try:
            bs = conn.execute("""
                SELECT created_at, period_start, period_end, coa,
                       non_current, current, total_assets,
                       equity, provisions, liabilities_short_term, liabilities_long_term,
                       total_liabilities_equity, balanced, difference
                FROM bs_snapshots
                WHERE file_sha256 = ?
                ORDER BY created_at ASC
            """, (file_sha256,)).fetchall()

            pl = conn.execute("""
                SELECT created_at, period_start, period_end, coa,
                       revenue, material_expenses, personnel_expenses,
                       other_operating_expenses, other_operating_income,
                       depreciation, interest_expenses, result
                FROM pl_snapshots
                WHERE file_sha256 = ?
                ORDER BY created_at ASC
            """, (file_sha256,)).fetchall()
        finally:
            conn.close()

    def _rows(rows) -> List[Dict[str, Any]]:
        return [dict(r) for r in rows]

    return {"bs": _rows(bs), "pl": _rows(pl)}

def read_pivot(file_sha256: str) -> Dict[str, List[Dict[str, Any]]]:
    with _DB_LOCK:
        conn = _conn()
        try:
            bs = conn.execute("""
                SELECT created_at, total_assets, equity,
                       liabilities_short_term AS liab_st, liabilities_long_term AS liab_lt
                FROM bs_snapshots
                WHERE file_sha256 = ?
                ORDER BY created_at ASC
            """, (file_sha256,)).fetchall()

            pl = conn.execute("""
                SELECT created_at, revenue, result
                FROM pl_snapshots
                WHERE file_sha256 = ?
                ORDER BY created_at ASC
            """, (file_sha256,)).fetchall()
        finally:
            conn.close()

    return {
        "balance_sheet": [dict(r) for r in bs],
        "profit_and_loss": [dict(r) for r in pl],
    }
