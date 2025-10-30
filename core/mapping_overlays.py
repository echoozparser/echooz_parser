# core/mapping_overlays.py
from __future__ import annotations
import json, os, re, uuid
from typing import Any, Dict, List, Optional

BASE_DIR = os.environ.get("ECHOoz_COMPANY_DATA", "data/company")

def _company_dir(company_id: str) -> str:
    d = os.path.join(BASE_DIR, company_id, "mapping")
    os.makedirs(d, exist_ok=True)
    return d

def _overlay_path(company_id: str) -> str:
    return os.path.join(_company_dir(company_id), "overlays.json")

def load_overlays(company_id: str) -> Dict[str, Any]:
    p = _overlay_path(company_id)
    if not os.path.exists(p):
        return {"company_id": company_id, "items": []}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_overlays(company_id: str, data: Dict[str, Any]) -> None:
    p = _overlay_path(company_id)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _validate_overlay(item: Dict[str, Any]) -> Dict[str, Any]:
    t = (item.get("type") or "").lower()
    target = item.get("target")
    if t not in {"exact", "range", "prefix", "regex", "label_regex"}:
        raise ValueError("overlay.type invalid")
    if not target:
        raise ValueError("overlay.target required")
    if t == "range":
        if not isinstance(item.get("start"), int) or not isinstance(item.get("end"), int):
            raise ValueError("overlay.range requires integer start/end")
    if t in {"regex", "label_regex"} and not item.get("pattern"):
        raise ValueError("overlay.pattern required for regex/label_regex")
    if t in {"exact", "prefix"} and not item.get("value"):
        raise ValueError("overlay.value required for exact/prefix")
    return item

def add_overlay(company_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
    data = load_overlays(company_id)
    item = _validate_overlay(item)
    item["id"] = item.get("id") or str(uuid.uuid4())
    data.setdefault("items", []).append(item)
    save_overlays(company_id, data)
    return item

def delete_overlay(company_id: str, overlay_id: str) -> bool:
    data = load_overlays(company_id)
    before = len(data.get("items", []))
    data["items"] = [x for x in data.get("items", []) if x.get("id") != overlay_id]
    after = len(data["items"])
    if after != before:
        save_overlays(company_id, data)
        return True
    return False

def resolve_target(company_id: Optional[str], account_raw: str, label: str) -> Optional[str]:
    if not company_id:
        return None
    data = load_overlays(company_id)
    acct = (account_raw or "").strip()
    acct_num = None
    m = re.search(r"\d+", acct)
    if m:
        try:
            acct_num = int(m.group(0))
        except Exception:
            acct_num = None

    for it in data.get("items", []):
        if it.get("type") == "exact" and str(it.get("value")) == acct:
            return it.get("target")
    for it in data.get("items", []):
        if it.get("type") == "range" and acct_num is not None:
            if int(it["start"]) <= acct_num <= int(it["end"]):
                return it.get("target")
    for it in data.get("items", []):
        if it.get("type") == "prefix" and acct.startswith(str(it.get("value"))):
            return it.get("target")
    for it in data.get("items", []):
        if it.get("type") == "regex" and re.search(it.get("pattern", ""), acct):
            return it.get("target")
    for it in data.get("items", []):
        if it.get("type") == "label_regex" and re.search(it.get("pattern", ""), label or "", re.IGNORECASE):
            return it.get("target")
    return None
