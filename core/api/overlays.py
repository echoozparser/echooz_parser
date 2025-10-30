# api/overlays.py
from __future__ import annotations
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
from core.mapping_overlays import load_overlays, add_overlay, delete_overlay

router = APIRouter(prefix="/mapping/overlays", tags=["mapping-overlays"])

class OverlayIn(BaseModel):
    type: Literal["exact","range","prefix","regex","label_regex"]
    target: str
    value: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    pattern: Optional[str] = None
    note: Optional[str] = None

class OverlayOut(OverlayIn):
    id: str = Field(...)

@router.get("", response_model=dict)
def list_overlays(x_company_id: Optional[str] = Header(default="demo", alias="X-Company-Id")):
    return load_overlays(x_company_id or "demo")

@router.post("", response_model=OverlayOut)
def create_overlay(item: OverlayIn, x_company_id: Optional[str] = Header(default="demo", alias="X-Company-Id")):
    try:
        created = add_overlay(x_company_id or "demo", item.dict())
        return created  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{overlay_id}", response_model=dict)
def remove_overlay(overlay_id: str, x_company_id: Optional[str] = Header(default="demo", alias="X-Company-Id")):
    ok = delete_overlay(x_company_id or "demo", overlay_id)
    if not ok:
        raise HTTPException(status_code=404, detail="overlay not found")
    return {"deleted": True, "id": overlay_id}

