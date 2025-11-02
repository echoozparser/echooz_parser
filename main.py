# main.py
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

# Aggregation
from core.aggregator import aggregate_from_dataframe

# Mapping (arbeitet mit /core/data/coa_pack.json)
from core.mapper import map_to_canonical

# interne Module
from core.helpers import (
    auto_detect_encoding,
    make_json_safe,
    normalize_text,
    parse_decimal_locale_aware,
)
from core.classifier import (
    find_header_row,
    classify_structure,
    detect_learning_pattern,
    apply_feedback_for_file,
)
from core.plausibility import analyze_data as plausi_analyze

# Summenzeilen + Saldo/Marker-Fallback
from core.parser_utils import drop_total_rows, check_balance_via_saldo_marker

# Fingerprint
from core.fingerprint import compute_pipeline_fingerprint

# DB-Persistenz (Snapshots & Pivot)
from core.persistence import init_db, write_snapshots, read_snapshots, read_pivot

# Snapshot-API (zusätzliche Routen)
from core.snapshot_api import router as snapshot_router

# =========================================================
# ECHOOZ PARSER v8.9 – Persistenz, Idempotenz, Aggregation (auto) + Fingerprint + Force-Refresh
# =========================================================

app = FastAPI(title="Echooz Parser", version="8.9")
app.include_router(snapshot_router)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
LOG_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
LEARN_FILE = BASE_DIR / "patterns.json"
OVERLAYS_FILE = BASE_DIR / "overlays.json"   # optional (für Mandanten-Overlays)
DATA_DIR = BASE_DIR / "core" / "data"
COA_PACK_PATH = DATA_DIR / "coa_pack.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# DB initialisieren (idempotent)
init_db()

ALLOWED_SUFFIXES = {".xlsx", ".xls", ".xlsm", ".ods", ".csv"}

# Konfiguration via Env
MAX_UPLOAD_MB = int(os.getenv("ECHOoz_MAX_UPLOAD_MB", "50"))
PREVIEW_ROWS = int(os.getenv("ECHOoz_PREVIEW_ROWS", "10"))
SUSA_TOLERANCE = float(os.getenv("ECHOoz_SUSA_TOLERANCE", "0.01"))
PERSIST_RESULTS = bool(int(os.getenv("ECHOoz_PERSIST_RESULTS", "1")))  # 1=an, 0=aus

# Base44 (optional)
BASE44_URL = os.getenv("BASE44_URL", "https://api.base44.io/entities/ParsedFinancialFile")
BASE44_TOKEN = os.getenv("BASE44_TOKEN", "").strip()
POST_TO_BASE44 = bool(int(os.getenv("ECHOoz_POST_TO_BASE44", "1")))  # 1=an, 0=aus


# ---------------------------------------------------------
# Pydantic-Modelle
# ---------------------------------------------------------
class LearningInfo(BaseModel):
    detected_as: Optional[str] = None
    confidence: Optional[float] = None
    source: Optional[str] = None
    matched_pattern_id: Optional[str] = None


class SusaBalance(BaseModel):
    debit_total: float
    credit_total: float
    difference: float
    tolerance: float
    balanced: bool
    columns_used: List[str]


class SheetResult(BaseModel):
    filename: str
    stored_as: str
    sheet: str
    detected_structure: str
    detected_structure_confidence: float
    learning: LearningInfo
    rows: int
    columns: List[str]
    preview: List[Dict[str, Any]]
    request_id: str
    file_sha256: str
    schema_version: str
    app_version: str
    header_row_index: int
    header_inferred: bool
    susa_balance: Optional[SusaBalance] = None
    mapping: Optional[Dict[str, Any]] = None
    aggregation: Optional[Dict[str, Any]] = None


class PlausiRequest(BaseModel):
    data: Dict[str, Any]


class PlausiResponse(BaseModel):
    findings: List[Dict[str, Any]]


class FeedbackRequest(BaseModel):
    stored_as: str
    sheet: str
    correct_label: str  # "SuSa" | "BWA" | "Bilanz" | "Journal" | "Unbekannt"


UploadResponse = List[SheetResult]


# ---------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------
def sanitize_for_json(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    safe = df.copy()
    safe.columns = [str(c) for c in safe.columns]
    safe = safe.replace([pd.NA, float("inf"), float("-inf")], pd.NA).where(pd.notna(safe), None)
    return safe


def _save_upload(file: UploadFile, content_length: Optional[int]) -> Path:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Nicht unterstützter Dateityp: {suffix}")

    if content_length is not None:
        max_bytes = MAX_UPLOAD_MB * 1024 * 1024
        if content_length > max_bytes:
            raise HTTPException(status_code=413, detail=f"Upload zu groß (> {MAX_UPLOAD_MB} MB)")
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    path = UPLOAD_DIR / safe_name
    with path.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    logging.info("Datei gespeichert unter %s (original: %s)", path.name, file.filename)
    return path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv_fast(path: Path) -> pd.DataFrame:
    enc = auto_detect_encoding(str(path))
    return pd.read_csv(path, encoding=enc, dtype=object, header=None, sep=None, engine="python")


def _read_any_table(path: Path) -> Dict[str, pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = _read_csv_fast(path)
        return {"Sheet1": df}
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
        return {str(k): v for k, v in sheets.items()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Excel/ODS konnte nicht gelesen werden: {e}")


def _audit_log(filename: str, stored_as: str, detected: str, confidence: float, rows: int, request_id: str) -> None:
    rec = {
        "ts": datetime.now().isoformat(),
        "file": filename,
        "stored_as": stored_as,
        "detected": detected,
        "confidence": confidence,
        "rows": rows,
        "app_version": app.version,
        "request_id": request_id,
    }
    with (LOG_DIR / "audit_log.ndjson").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -------------------- Persistenz / Idempotenz --------------------
def _result_path_by_hash(file_hash: str) -> Path:
    return RESULTS_DIR / f"{file_hash}.json"


def _load_result(file_hash: str) -> Optional[Dict[str, Any]]:
    p = _result_path_by_hash(file_hash)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            logging.warning("Persistentes Ergebnis konnte nicht gelesen werden: %s", p)
    return None


def _save_result(file_hash: str, payload: Dict[str, Any]) -> None:
    try:
        _result_path_by_hash(file_hash).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logging.warning("Persistentes Ergebnis konnte nicht gespeichert werden: %s", e)


# ------------------------- SuSa-Check -------------------------
_DEBIT_PAT = re.compile(r"soll|debit")
_CREDIT_PAT = re.compile(r"haben|credit")


def _find_debit_credit_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    norm = {str(c): normalize_text(str(c)) for c in df.columns}

    def pick(scope: str) -> Optional[Tuple[str, str]]:
        d = [col for col, n in norm.items() if ("soll" in n or _DEBIT_PAT.search(n)) and (scope in n if scope else True)]
        h = [col for col, n in norm.items() if ("haben" in n or _CREDIT_PAT.search(n)) and (scope in n if scope else True)]
        if d and h:
            for dc in d:
                for hc in h:
                    nd, nh = norm[dc], norm[hc]
                    if (("monat" in nd) == ("monat" in nh)) and (("jahr" in nd or "jahres" in nd) == ("jahr" in nh or "jahres" in nh)):
                        return dc, hc
            return d[0], h[0]
        return None

    for scope in ("jahres", "jahr", "annual", "monat", "periode", ""):
        pair = pick(scope)
        if pair:
            return pair

    d = [c for c, n in norm.items() if ("soll" in n or "debit" in n)]
    h = [c for c, n in norm.items() if ("haben" in n or "credit" in n)]
    return (d[0], h[0]) if d and h else None


def _find_cumulative_debit_credit_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    norm = {str(c): normalize_text(str(c)) for c in df.columns}

    def is_cumulative(label_raw: str, label_norm: str) -> bool:
        if any(k in label_norm for k in ("kum", "kumul", "kumuliert", "ytd", "year to date", "year-to-date")):
            return True
        if label_raw.endswith("_1"):
            return True
        if "1-" in label_raw or "1–" in label_raw or "1/" in label_raw:
            return True
        return False

    debit_candidates: List[str] = []
    credit_candidates: List[str] = []
    for col, n in norm.items():
        raw = str(col)
        if ("soll" in n or _DEBIT_PAT.search(n)) and is_cumulative(raw, n):
            debit_candidates.append(col)
        if ("haben" in n or _CREDIT_PAT.search(n)) and is_cumulative(raw, n):
            credit_candidates.append(col)

    if debit_candidates and credit_candidates:
        return debit_candidates[0], credit_candidates[0]
    return None


def _series_to_decimal(series: pd.Series) -> List[float]:
    vals: List[float] = []
    q = parse_decimal_locale_aware("0.01")
    for v in series.tolist():
        d = parse_decimal_locale_aware(v).quantize(q)
        vals.append(float(d))
    return vals


def _check_susa_balance(df: pd.DataFrame) -> Optional[SusaBalance]:
    cols = _find_debit_credit_columns(df)
    if cols:
        debit_col, credit_col = cols
        debit_total = round(sum(_series_to_decimal(df[debit_col])), 2)
        credit_total = round(sum(_series_to_decimal(df[credit_col])), 2)
        diff = round(debit_total - credit_total, 2)
        if abs(diff) <= round(SUSA_TOLERANCE, 2):
            return SusaBalance(
                debit_total=debit_total,
                credit_total=credit_total,
                difference=diff,
                tolerance=round(SUSA_TOLERANCE, 2),
                balanced=True,
                columns_used=[debit_col, credit_col],
            )

    cols_cum = _find_cumulative_debit_credit_columns(df)
    if cols_cum:
        d_cum, h_cum = cols_cum
        debit_total = round(sum(_series_to_decimal(df[d_cum])), 2)
        credit_total = round(sum(_series_to_decimal(df[h_cum])), 2)
        diff = round(debit_total - credit_total, 2)
        if abs(diff) <= round(SUSA_TOLERANCE, 2):
            logging.info("SuSa-Balance via Fallback (kumuliert): %s / %s", d_cum, h_cum)
            return SusaBalance(
                debit_total=debit_total,
                credit_total=credit_total,
                difference=diff,
                tolerance=round(SUSA_TOLERANCE, 2),
                balanced=True,
                columns_used=[d_cum, h_cum],
            )
        if not cols:
            return SusaBalance(
                debit_total=debit_total,
                credit_total=credit_total,
                difference=diff,
                tolerance=round(SUSA_TOLERANCE, 2),
                balanced=False,
                columns_used=[d_cum, h_cum],
            )

    marker_info = None
    try:
        marker_info = check_balance_via_saldo_marker(df)
    except Exception as e:
        logging.debug("Saldo/Marker-Fallback nicht anwendbar: %s", e)

    if marker_info:
        _, cols_used = marker_info
        saldo_col, marker_col = cols_used[0], cols_used[1]
        sum_debit = 0.0
        sum_credit = 0.0
        q = parse_decimal_locale_aware("0.01")

        for raw_val, mark in zip(df[saldo_col].tolist(), df[marker_col].astype(str).str.strip().str.upper().tolist()):
            try:
                d = parse_decimal_locale_aware(raw_val).quantize(q)
            except Exception:
                continue
            val = float(d)
            if val == 0.0:
                continue
            if mark in {"S", "D"}:
                sum_debit += abs(val)
            elif mark in {"H", "C"}:
                sum_credit += abs(val)
            else:
                continue

        diff = round(sum_debit - sum_credit, 2)
        return SusaBalance(
            debit_total=round(sum_debit, 2),
            credit_total=round(sum_credit, 2),
            difference=diff,
            tolerance=round(SUSA_TOLERANCE, 2),
            balanced=abs(diff) <= round(SUSA_TOLERANCE, 2),
            columns_used=[saldo_col, marker_col],
        )

    if cols:
        debit_col, credit_col = cols
        debit_total = round(sum(_series_to_decimal(df[debit_col])), 2)
        credit_total = round(sum(_series_to_decimal(df[credit_col])), 2)
        diff = round(debit_total - credit_total, 2)
        return SusaBalance(
            debit_total=debit_total,
            credit_total=credit_total,
            difference=diff,
            tolerance=round(SUSA_TOLERANCE, 2),
            balanced=abs(diff) <= round(SUSA_TOLERANCE, 2),
            columns_used=[debit_col, credit_col],
        )
    return None


# ---------------------------------------------------------
# Base44 – POST Hook (optional)
# ---------------------------------------------------------
def post_to_base44(payload: dict) -> tuple[int, str]:
    if not (POST_TO_BASE44 and BASE44_TOKEN):
        return -1, "Base44 POST übersprungen (kein Token/disabled)"
    try:
        headers = {
            "Authorization": f"Bearer {BASE44_TOKEN}",
            "Content-Type": "application/json",
        }
        file_hash = (payload.get("file") or {}).get("hash_sha256") or payload.get("hash") or ""
        if file_hash:
            headers["X-Idempotency-Key"] = file_hash
        r = requests.post(BASE44_URL, headers=headers, json=payload, timeout=30)
        return r.status_code, r.text[:5000]
    except Exception as e:
        logging.exception("Base44 POST failed:")
        return 0, str(e)


# ---------------------------------------------------------
# API – Startseite/Health
# ---------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Echooz Parser läuft", "version": app.version, "preview_rows_default": PREVIEW_ROWS}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "version": app.version}


# ---------------------------------------------------------
# Upload-Endpunkt mit Force-Refresh + Pipeline-Fingerprint
# ---------------------------------------------------------
@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    preview_rows: Optional[int] = None
) -> UploadResponse:
    try:
        # Meta/Headers
        client_id = request.headers.get("X-Client-Id")
        prefer_coa_hdr_raw = (request.headers.get("X-Chart-Of-Accounts") or "").strip()
        prefer_coa = prefer_coa_hdr_raw.lower() or None  # 'skr03'|'skr04'|'netti'|'custom'|'auto'|None
        period_start = request.headers.get("X-Period-Start")
        period_end = request.headers.get("X-Period-End")
        agg_hdr = (request.headers.get("X-Aggregate") or "").strip().lower()
        want_aggregation = False if agg_hdr in {"0", "false", "no"} else True
        force_refresh = (request.headers.get("X-Force-Refresh") or "").strip().lower() in {"1", "true", "yes"}

        content_length = request.headers.get("content-length")
        content_length_int = int(content_length) if content_length and content_length.isdigit() else None
        path = _save_upload(file, content_length_int)
        sheets = _read_any_table(path)
        file_hash = _sha256_file(path)
        request_id = uuid.uuid4().hex
        n_preview = preview_rows if (preview_rows is not None and preview_rows >= 0) else PREVIEW_ROWS

        # Pipeline Fingerprint (für Persistenz/Debug)
        pipeline = compute_pipeline_fingerprint(
            app_version=app.version,
            file_sha256=file_hash,
            headers={
                "X-Chart-Of-Accounts": prefer_coa_hdr_raw,
                "X-Period-Start": period_start,
                "X-Period-End": period_end,
                "X-Aggregate": request.headers.get("X-Aggregate")
            },
            coa_pack_path=str(COA_PACK_PATH),
            patterns_path=str(LEARN_FILE),
            overlays_path=str(OVERLAYS_FILE),
            mapper_version="mapper-v1",
            aggregator_version="aggregator-v1"
        )

        # Idempotenz: nur nutzen, wenn NICHT Force-Refresh
        if PERSIST_RESULTS and not force_refresh:
            cached = _load_result(file_hash)
            if cached:
                logging.info("♻️ Idempotenz: vorhandenes Ergebnis für %s wird wiederverwendet.", file.filename)
                code, msg = post_to_base44(cached) if POST_TO_BASE44 else (-1, "Base44 POST übersprungen")
                if code >= 200 or code == -1:
                    logging.info("Base44 POST (cached): %s", "übersprungen" if code == -1 else f"{code}")
                else:
                    logging.warning("Base44 POST (cached) fehlgeschlagen (%s): %s", code, msg)
                return [SheetResult(**s) for s in cached.get("sheets", [])]

        results: List[SheetResult] = []

        for sheet_name, raw_df in sheets.items():
            if raw_df.dropna(how="all").empty:
                continue

            # 1) Generische Header-Erkennung
            header_row = find_header_row(raw_df)

            # 2) SAP-Fallback nur, wenn nichts gefunden
            if header_row is None:
                header_row = _sap_header_fallback(raw_df)

            # 3) Final: erste nicht-leere Zeile, falls weiterhin None
            if header_row is None:
                logging.warning("Keine Kopfzeile in Sheet '%s' erkannt.", sheet_name)
                first_non_empty = next((i for i in range(len(raw_df)) if raw_df.iloc[i].notna().any()), None)
                if first_non_empty is None:
                    continue
                header_row = int(first_non_empty)
                header_inferred = True
            else:
                header_inferred = False

            cols = [str(x) for x in raw_df.iloc[header_row].tolist()]
            seen, new_cols = {}, []
            for c in cols:
                c = (c or "").strip().replace("\n", " ")
                if c in seen:
                    seen[c] += 1
                    new_cols.append(f"{c}_{seen[c]}")
                else:
                    seen[c] = 0
                    new_cols.append(c)

            df = raw_df.drop(index=list(range(0, header_row + 1))).reset_index(drop=True)
            df.columns = new_cols
            df = df.convert_dtypes()

            # 4) Struktur klassifizieren + Learning
            detected_structure, conf = classify_structure(df.columns, df)
            learned = detect_learning_pattern(df, str(LEARN_FILE))

            # 4a) Summenzeilen-Filter
            try:
                if detected_structure in {"SuSa", "Bilanz", "TrialBalance", "TB", "Journal"}:
                    LABEL_COLS = ["Bezeichnung", "Kurzbezeichnung", "Text", "Beschreibung", "Title", "Name", "Beschriftung"]
                    NUMERIC_COLS = ["Soll", "Haben", "Soll_1", "Haben_1", "Debit", "Credit", "Saldo", "Balance", "Amount"]
                    df_filtered, totals_meta = drop_total_rows(
                        df,
                        label_cols=[c for c in LABEL_COLS if c in df.columns],
                        numeric_cols=[c for c in NUMERIC_COLS if c in df.columns],
                        tail_ratio=0.9
                    )
                    df = df_filtered
                    logging.info(
                        "Preprocessing (%s/%s): total_rows_removed=%s, last_total_row_index=%s, tail_cutoff_index=%s",
                        file.filename, sheet_name,
                        totals_meta.get("total_rows_removed"),
                        totals_meta.get("last_total_row_index"),
                        totals_meta.get("tail_cutoff_index"),
                    )
            except Exception as e:
                logging.warning("Summenzeilen-Filter übersprungen (%s/%s): %s", file.filename, sheet_name, e)

            # 5) SuSa-/Journal-Check
            susa_res: Optional[SusaBalance] = None
            if detected_structure in {"SuSa", "Journal"}:
                try:
                    susa_res = _check_susa_balance(df)
                except Exception as e:
                    logging.warning("SuSa-Check fehlgeschlagen in Sheet '%s': %s", sheet_name, e)

            # 6) Mapping → Canonical
            mapping: Optional[Dict[str, Any]] = None
            try:
                mapping = map_to_canonical(
                    df,
                    structure=detected_structure,
                    prefer_coa=prefer_coa,
                    pack_path=str(COA_PACK_PATH)
                )
            except Exception as e:
                logging.warning("Mapping fehlgeschlagen in Sheet '%s': %s", sheet_name, e)

            # 7) Aggregation (auto)
            aggregation: Optional[Dict[str, Any]] = None
            if want_aggregation and detected_structure == "SuSa" and mapping and mapping.get("canonical_columns", {}).get("account"):
                try:
                    aggregation = aggregate_from_dataframe(
                        df=df,
                        canonical_columns=mapping["canonical_columns"],
                        coa_pack_path=str(COA_PACK_PATH),
                        file_sha256=file_hash,
                        force_balance=True,
                        prefer_coa=(prefer_coa if prefer_coa not in {None, "", "auto"} else None),
                        period_start=period_start,
                        period_end=period_end,
                    )
                except Exception as e:
                    logging.warning("Aggregation fehlgeschlagen (%s/%s): %s", file.filename, sheet_name, e)

            # >>> Fingerprint auch in aggregation.meta anhängen + Snapshots schreiben
            if aggregation:
                aggregation.setdefault("meta", {})
                aggregation["meta"]["pipeline"] = pipeline

                try:
                    agg_meta = aggregation.get("meta", {}) or {}
                    agg_period = aggregation.get("period", {}) or {}
                    write_snapshots(
                        file_sha256=file_hash,
                        company_id=(client_id or None),
                        coa=(agg_meta.get("coa") or (prefer_coa_hdr_raw.upper() if prefer_coa_hdr_raw else None)),
                        period_start=agg_period.get("start_date"),
                        period_end=agg_period.get("end_date"),
                        aggregation=aggregation,
                    )
                except Exception as e:
                    logging.warning("Snapshot-Persistierung übersprungen/fehlgeschlagen: %s", e)

            # 8) Audit + Preview + Result
            _audit_log(
                file.filename,
                path.name,
                learned.get("detected_as", detected_structure),
                float(learned.get("confidence", conf)),
                len(df),
                request_id,
            )

            preview_df = sanitize_for_json(df.head(n_preview))
            result_dict = {
                "filename": file.filename,
                "stored_as": path.name,
                "sheet": sheet_name,
                "detected_structure": detected_structure,
                "detected_structure_confidence": conf,
                "learning": {
                    "detected_as": learned.get("detected_as"),
                    "confidence": learned.get("confidence"),
                    "source": learned.get("source", "none"),
                    "matched_pattern_id": learned.get("pattern_id"),
                },
                "rows": int(len(df)),
                "columns": [str(c) for c in df.columns],
                "preview": jsonable_encoder(make_json_safe(preview_df.to_dict(orient="records"))),
                "request_id": request_id,
                "file_sha256": file_hash,
                "schema_version": "2025-10-17",
                "app_version": app.version,
                "header_row_index": int(header_row),
                "header_inferred": bool(header_inferred),
                "susa_balance": susa_res.model_dump() if susa_res else None,
                "mapping": mapping or None,
                "aggregation": aggregation or None,
            }

            results.append(SheetResult(**result_dict))

        if not results:
            raise HTTPException(status_code=400, detail="Keine verwertbaren Daten erkannt.")

        combined_payload = {
            "client_id": client_id,
            "file": {
                "filename": file.filename,
                "stored_as": path.name,
                "hash_sha256": file_hash,
                "request_id": request_id,
                "app_version": app.version,
                "schema_version": "2025-10-17",
                "label": None,
                "period": {"start": period_start, "end": period_end},
                "received_at": datetime.utcnow().isoformat() + "Z",
            },
            "pipeline": pipeline,
            "sheets": [jsonable_encoder(r.dict()) for r in results],
        }

        if PERSIST_RESULTS:
            _save_result(file_hash, combined_payload)

        code, msg = post_to_base44(combined_payload)
        if code >= 200 or code == -1:
            logging.info("Base44 POST: %s", "übersprungen" if code == -1 else f"{code}")
        else:
            logging.warning("Base44 POST fehlgeschlagen (%s): %s", code, msg)

        return results

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Fehler beim Verarbeiten der Datei")
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------
# Plausibilität – REST-Endpoint
# ---------------------------------------------------------
@app.post("/plausibility", response_model=PlausiResponse)
def post_plausibility(req: PlausiRequest) -> PlausiResponse:
    return PlausiResponse(findings=plausi_analyze(req.data))


# ---------------------------------------------------------
# Active-Learning-Feedback
# ---------------------------------------------------------
@app.post("/feedback/structure")
def post_feedback(req: FeedbackRequest):
    try:
        path = UPLOAD_DIR / req.stored_as
        if not path.exists():
            raise HTTPException(status_code=404, detail="Upload nicht (mehr) vorhanden.")

        sheets = _read_any_table(path)
        if req.sheet not in sheets:
            raise HTTPException(status_code=404, detail=f"Sheet '{req.sheet}' nicht gefunden.")
        df = sheets[req.sheet]

        header_row = find_header_row(df)
        if header_row is None:
            header_row = _sap_header_fallback(df)
            if header_row is None:
                first_non_empty = next((i for i in range(len(df)) if df.iloc[i].notna().any()), None)
                if first_non_empty is None:
                    raise HTTPException(status_code=400, detail="Keine verwertbaren Daten im Sheet.")
                header_row = int(first_non_empty)

        cols = [str(x) for x in df.iloc[header_row].tolist()]
        seen, new_cols = {}, []
        for c in cols:
            c = (c or "").strip().replace("\n", " ")
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        df = df.drop(index=list(range(0, header_row + 1))).reset_index(drop=True)
        df.columns = new_cols
        df = df.convert_dtypes()

        pattern_id = apply_feedback_for_file(df, str(LEARN_FILE), correct_label=req.correct_label)
        return {"status": "ok", "pattern_id": pattern_id, "applied_label": req.correct_label}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Fehler beim Feedback:")
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------
# Ergebnisse abrufen (Persistenz)
# ---------------------------------------------------------
@app.get("/result/{file_hash}")
def get_result(file_hash: str):
    data = _load_result(file_hash)
    if not data:
        raise HTTPException(status_code=404, detail="Kein Ergebnis zu diesem Hash gefunden.")
    return data


# ---------------------------------------------------------
# Rebuild-Endpoint – re-aggregate mit aktuellen Regeln
# ---------------------------------------------------------
class RebuildRequest(BaseModel):
    prefer_coa: Optional[str] = None   # "skr03"|"skr04"|"netti"|"custom"|None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    aggregate: Optional[bool] = True


@app.post("/rebuild/{file_hash}", response_model=UploadResponse)
def rebuild(file_hash: str, body: RebuildRequest):
    data = _load_result(file_hash)
    if not data:
        raise HTTPException(status_code=404, detail="Kein persistiertes Ergebnis zu diesem Hash.")

    stored_as = (data.get("file") or {}).get("stored_as")
    if not stored_as:
        raise HTTPException(status_code=400, detail="Persistente Datei-Referenz fehlt.")
    path = UPLOAD_DIR / stored_as
    if not path.exists():
        raise HTTPException(status_code=404, detail="Upload-Datei nicht (mehr) vorhanden.")

    try:
        sheets = _read_any_table(path)
        results: List[SheetResult] = []

        for sheet_name, raw_df in sheets.items():
            if raw_df.dropna(how="all").empty:
                continue

            header_row = find_header_row(raw_df) or 0
            cols = [str(x) for x in raw_df.iloc[header_row].tolist()]

            seen, new_cols = {}, []
            for c in cols:
                c = (c or "").strip().replace("\n", " ")
                if c in seen:
                    seen[c] += 1
                    new_cols.append(f"{c}_{seen[c]}")
                else:
                    seen[c] = 0
                    new_cols.append(c)

            df = raw_df.drop(index=list(range(0, header_row + 1))).reset_index(drop=True)
            df.columns = new_cols
            df = df.convert_dtypes()

            # Struktur bestimmen
            detected_structure, conf = classify_structure(df.columns, df)

            # SuSa-/Journal-Check
            susa_res: Optional[SusaBalance] = None
            if detected_structure in {"SuSa", "Journal"}:
                try:
                    susa_res = _check_susa_balance(df)
                except Exception as e:
                    logging.warning("SuSa-Check (Rebuild) fehlgeschlagen in Sheet '%s': %s", sheet_name, e)

            # Mapping → Canonical
            mapping = map_to_canonical(
                df,
                structure=detected_structure,
                prefer_coa=(body.prefer_coa or None),
                pack_path=str(COA_PACK_PATH)
            )

            # Aggregation (optional)
            aggregation = None
            if body.aggregate and detected_structure == "SuSa" and mapping and mapping.get("canonical_columns", {}).get("account"):
                aggregation = aggregate_from_dataframe(
                    df=df,
                    canonical_columns=mapping["canonical_columns"],
                    coa_pack_path=str(COA_PACK_PATH),
                    file_sha256=file_hash,
                    force_balance=True,
                    prefer_coa=(body.prefer_coa if body.prefer_coa not in {None, "", "auto"} else None),
                    period_start=body.period_start,
                    period_end=body.period_end,
                )

            # Fingerprint + Snapshots
            if aggregation:
                pipeline_rb = compute_pipeline_fingerprint(
                    app_version=app.version,
                    file_sha256=file_hash,
                    headers={
                        "X-Chart-Of-Accounts": (body.prefer_coa or ""),
                        "X-Period-Start": (body.period_start or ((data.get("file") or {}).get("period") or {}).get("start")),
                        "X-Period-End": (body.period_end or ((data.get("file") or {}).get("period") or {}).get("end")),
                        "X-Aggregate": "1" if (body.aggregate is not False) else "0",
                    },
                    coa_pack_path=str(COA_PACK_PATH),
                    patterns_path=str(LEARN_FILE),
                    overlays_path=str(OVERLAYS_FILE),
                    mapper_version="mapper-v1",
                    aggregator_version="aggregator-v1",
                )
                aggregation.setdefault("meta", {})
                aggregation["meta"]["pipeline"] = pipeline_rb

                try:
                    agg_meta = aggregation.get("meta", {}) or {}
                    agg_period = aggregation.get("period", {}) or {}
                    write_snapshots(
                        file_sha256=file_hash,
                        company_id=(data.get("client_id") or None),
                        coa=(agg_meta.get("coa") or (body.prefer_coa.upper() if body.prefer_coa else None)),
                        period_start=agg_period.get("start_date"),
                        period_end=agg_period.get("end_date"),
                        aggregation=aggregation,
                    )
                except Exception as e:
                    logging.warning("Snapshot-Persistierung (Rebuild) übersprungen/fehlgeschlagen: %s", e)

            preview_df = sanitize_for_json(df.head(PREVIEW_ROWS))
            results.append(SheetResult(
                filename=(data.get("file") or {}).get("filename", "rebuild"),
                stored_as=stored_as,
                sheet=sheet_name,
                detected_structure=detected_structure,
                detected_structure_confidence=conf,
                learning=LearningInfo(),
                rows=int(len(df)),
                columns=[str(c) for c in df.columns],
                preview=jsonable_encoder(make_json_safe(preview_df.to_dict(orient="records"))),
                request_id=(data.get("file") or {}).get("request_id", uuid.uuid4().hex),
                file_sha256=file_hash,
                schema_version=(data.get("file") or {}).get("schema_version", "2025-10-17"),
                app_version=app.version,
                header_row_index=int(header_row),
                header_inferred=False,
                susa_balance=susa_res.model_dump() if susa_res else None,
                mapping=mapping or None,
                aggregation=aggregation or None,
            ))

        return results

    except Exception as e:
        logging.exception("Rebuild fehlgeschlagen")
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------
# SAP-HEADER-FALLBACK
# ---------------------------------------------------------
_SAP_KEYS = {
    "bukr", "sachkonto", "kurztext", "waehrg", "gsbe",
    "saldovortrag", "saldo der vorperioden",
    "soll berichtszeitraum", "haben berichtszeitraum",
    "kum", "kum saldo", "kumuliert"
}
_SAP_BLACKLIST = {"sachkontensalden", "vortragsperioden", "berichtsperioden", "rfssld00"}


def _sap_header_fallback(raw_df: pd.DataFrame, max_scan: int = 30, min_hits: int = 3) -> Optional[int]:
    n = min(len(raw_df), max_scan)
    for i in range(n):
        raw_vals = [str(x) for x in raw_df.iloc[i].tolist()]
        cells = [normalize_text(v) for v in raw_vals if v and str(v).strip()]
        if not cells:
            continue
        joined = " ".join(cells)
        if any(b in joined for b in _SAP_BLACKLIST):
            continue
        hits = sum(1 for k in _SAP_KEYS if k in joined)
        if hits >= min_hits:
            logging.info("🔎 SAP-Fallback-Kopfzeile gewählt: Zeile %d (Treffer=%d)", i, hits)
            return i
    return None


# ---------------------------------------------------------
# Lokaler Start
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
