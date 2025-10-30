# core/fingerprint.py
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any, Dict, Optional

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def file_sha_or_none(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return None
    p = Path(path_str)
    return sha256_file(p) if p.exists() else None

def compute_pipeline_fingerprint(
    *,
    app_version: str,
    file_sha256: str,
    headers: Dict[str, Any],
    coa_pack_path: str,
    patterns_path: Optional[str] = None,
    overlays_path: Optional[str] = None,
    mapper_version: str = "mapper-v1",
    aggregator_version: str = "aggregator-v1",
) -> Dict[str, Any]:
    payload = {
        "app_version": app_version,
        "file_sha256": file_sha256,
        "headers": {
            "X-Chart-Of-Accounts": headers.get("X-Chart-Of-Accounts"),
            "X-Period-Start": headers.get("X-Period-Start"),
            "X-Period-End": headers.get("X-Period-End"),
            "X-Aggregate": headers.get("X-Aggregate"),
        },
        "components": {
            "coa_pack_sha256": file_sha_or_none(coa_pack_path),
            "patterns_sha256": file_sha_or_none(patterns_path) if patterns_path else None,
            "overlays_sha256": file_sha_or_none(overlays_path) if overlays_path else None,
            "mapper_version": mapper_version,
            "aggregator_version": aggregator_version,
        }
    }
    fp = _sha256_bytes(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return {"fingerprint": fp, "inputs": payload}
