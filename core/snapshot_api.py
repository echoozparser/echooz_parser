# core/snapshot_api.py
from __future__ import annotations
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Body, Query

from core.persistence import read_snapshots, read_pivot, write_snapshots

# Eigener Router; in main.py via: app.include_router(snapshot_router)
router = APIRouter()

# ---------------------------------------------------------
# Helper: Aggregation aus verschiedenen Payload-Formen ziehen
# Akzeptiert:
# 1) [ { "aggregation": {...} } ]
# 2) { "aggregation": {...} }
# 3) direkt die Aggregation {...}
# ---------------------------------------------------------
def _extract_aggregation(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, list):
        if not payload:
            raise HTTPException(status_code=400, detail="Empty list payload")
        item = payload[0]
        if isinstance(item, dict) and "aggregation" in item:
            return item.get("aggregation") or {}
        if isinstance(item, dict):
            return item
        raise HTTPException(status_code=400, detail="Invalid list element")
    if isinstance(payload, dict):
        return payload.get("aggregation", payload)
    raise HTTPException(status_code=400, detail="Invalid payload")

# ---------------------------------------------------------
# POST /aggregate/snapshot
# Persistiert Bilanz- & GuV-Snapshots aus einer Aggregation
# ---------------------------------------------------------
@router.post("/aggregate/snapshot")
def post_snapshot(payload: Any = Body(...)) -> Dict[str, Any]:
    agg = _extract_aggregation(payload)
    meta = dict(agg.get("meta") or {})
    period = dict(agg.get("period") or {})

    # file_sha256 ist der Schlüssel für Snapshots
    file_sha256 = (
        meta.get("file_sha256")
        or meta.get("hash")
        or meta.get("file_hash")
        or ""
    )
    if not file_sha256:
        raise HTTPException(status_code=400, detail="meta.file_sha256 required")

    company_id   = meta.get("company_id")
    coa          = meta.get("coa")
    period_start = period.get("start_date")
    period_end   = period.get("end_date")

    try:
        ids = write_snapshots(
            file_sha256=file_sha256,
            company_id=company_id,
            coa=coa,
            period_start=period_start,
            period_end=period_end,
            aggregation=agg,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"snapshot failed: {e}")

    # write_snapshots kann je nach Version None oder ein Dict mit IDs zurückgeben
    bs_id = ids.get("bs_id") if isinstance(ids, dict) else None
    pl_id = ids.get("pl_id") if isinstance(ids, dict) else None

    return {
        "ok": True,
        "file_sha256": file_sha256,
        "bs_id": bs_id,
        "pl_id": pl_id,
        "coa": coa,
        "period_start": period_start,
        "period_end": period_end,
    }

# ---------------------------------------------------------
# GET /snapshots/{file_hash}  (Path-Variante wie in deinen curl-Beispielen)
# ---------------------------------------------------------
@router.get("/snapshots/{file_hash}")
def get_snapshots_by_path(file_hash: str) -> Dict[str, List[Dict[str, Any]]]:
    try:
        return read_snapshots(file_sha256=file_hash)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# GET /pivot/{file_hash}  (Path-Variante)
# ---------------------------------------------------------
@router.get("/pivot/{file_hash}")
def get_pivot_by_path(file_hash: str) -> Dict[str, List[Dict[str, Any]]]:
    try:
        return read_pivot(file_sha256=file_hash)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# Optional: Query-Varianten (rückwärtskompatibel, falls ?file_sha256=... genutzt wird)
# ---------------------------------------------------------
@router.get("/snapshots")
def get_snapshots_by_query(file_sha256: str = Query(...)) -> Dict[str, List[Dict[str, Any]]]:
    try:
        return read_snapshots(file_sha256=file_sha256)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pivot")
def get_pivot_by_query(file_sha256: str = Query(...)) -> Dict[str, List[Dict[str, Any]]]:
    try:
        return read_pivot(file_sha256=file_sha256)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
