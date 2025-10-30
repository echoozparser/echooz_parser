from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence, Optional, List

import chardet

try:
    import numpy as _np  # optional
except Exception:
    _np = None


# =========================================================
# ⚙️ ECHOOZ HELPERS v9.3 – Encoding, Normalize, JSON, Numbers, COA
#  - Enthält ALLES aus v9.2 + COA-Erkennung + kleine Robustheits-Extras
# =========================================================

_BOMS = {
    b"\xef\xbb\xbf": "utf-8-sig",
    b"\xff\xfe\x00\x00": "utf-32-le",
    b"\x00\x00\xfe\xff": "utf-32-be",
    b"\xff\xfe": "utf-16-le",
    b"\xfe\xff": "utf-16-be",
}

def _detect_bom(prefix: bytes) -> str | None:
    for sig, enc in _BOMS.items():
        if prefix.startswith(sig):
            return enc
    return None

@lru_cache(maxsize=256)
def _detect_encoding_cached(path: str, size: int, mtime: float, sample_size: int) -> tuple[str, float]:
    # BOM?
    try:
        with open(path, "rb") as f:
            head = f.read(4)
        bom = _detect_bom(head)
        if bom:
            logging.info("BOM erkannt → %s", bom)
            return bom, 1.0
    except Exception:
        pass

    # chardet-Fallback
    try:
        detector = chardet.universaldetector.UniversalDetector()
        read_bytes = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                detector.feed(chunk)
                read_bytes += len(chunk)
                if detector.done or read_bytes >= max(sample_size, 8192):
                    break
        detector.close()
        enc = detector.result.get("encoding") or "utf-8"
        conf = float(detector.result.get("confidence") or 0.0)
        logging.info("Erkannte Kodierung: %s (%.0f%%)", enc, conf * 100)
        return enc, conf
    except Exception as e:
        logging.warning("chardet Fehler: %s", e)
        return "utf-8", 0.0

def auto_detect_encoding(file_path: str, sample_size: int = 10000) -> str:
    """Erkennt die Kodierung (BOM-aware, chardet-Fallback)."""
    try:
        p = Path(file_path)
        enc, conf = _detect_encoding_cached(str(p), p.stat().st_size, p.stat().st_mtime, sample_size)
        if conf < 0.5 and (enc or "").lower() not in {"utf-8", "utf-8-sig"}:
            return "utf-8"
        return enc
    except Exception as e:
        logging.warning("Fehler bei Kodierungserkennung: %s", e)
        return "utf-8"

def auto_detect_encoding_with_conf(file_path: str, sample_size: int = 10000) -> tuple[str, float]:
    try:
        p = Path(file_path)
        return _detect_encoding_cached(str(p), p.stat().st_size, p.stat().st_mtime, sample_size)
    except Exception:
        return "utf-8", 0.0


# ------------------ Text-Normalisierung ------------------

def normalize_text(text: Any) -> str:
    """
    Normalisiert Texte für Vergleich/Klassifikation:
    - Bytes → decode
    - Umlaute → ae/oe/ue/ss
    - NFKD + Diakritika entfernen
    - lower + Whitespaces vereinheitlichen
    - zulässige Zeichen: [a-z0-9 .,_-#&/ ]
    """
    if text is None:
        return ""
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8", "ignore")
        except Exception:
            text = text.decode("latin-1", "ignore")
    else:
        text = str(text)

    text = text.strip().replace("\n", " ")
    text = (text
            .replace("Ä", "Ae").replace("Ö", "Oe").replace("Ü", "Ue")
            .replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
            .replace("ß", "ss"))

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text).lower()
    text = re.sub(r"[^a-z0-9\.\,\-\_\#\&\/ ]", "", text)
    return text


# ------------------ Zahlen/Dezimal-Parsing ------------------

_DE_DECI = re.compile(r"\d+\.\d{3}(?:\.\d{3})*(,\d+)?$")
_EN_DECI = re.compile(r"\d+,\d{3}(?:,\d{3})*(\.\d+)?$")

def parse_decimal_locale_aware(value: Any) -> Decimal:
    """
    Versteht u.a.:
      '1.234.567,89', '1,234,567.89', '1234567.89',
      '1 234 567,89 €', '-', sowie Klammernegativ: '(1.234,56)'
    Unparsbare Werte -> Decimal(0).
    """
    if value is None:
        return Decimal(0)
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return Decimal(str(value))

    s = str(value).strip()
    if s == "" or s == "-":
        return Decimal(0)

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    s = (s
         .replace("€", "").replace("EUR", "").replace("eur", "")
         .replace("\u00A0", "").replace(" ", ""))

    # Deutsche Schreibweise: 1.234.567,89
    if _DE_DECI.search(s):
        s = s.replace(".", "").replace(",", ".")
    # Englische Schreibweise: 1,234,567.89
    elif _EN_DECI.search(s):
        s = s.replace(",", "")
    else:
        # genau ein Komma, kein Punkt -> Komma als Dezimaltrennzeichen
        if s.count(",") == 1 and s.count(".") == 0:
            s = s.replace(",", ".")
        else:
            # entferne Nicht-Ziffern außer Vorzeichen und Punkt
            s = re.sub(r"[^0-9\.\-]", "", s)

    try:
        d = Decimal(s)
    except InvalidOperation:
        try:
            d = Decimal(re.sub(r"[^\d\.\-]", "", s) or "0")
        except InvalidOperation:
            d = Decimal(0)

    if neg:
        d = -d
    return d


# ------------------ JSON-Sicherheit ------------------

def _is_nan_or_inf(x: Any) -> bool:
    try:
        return isinstance(x, float) and (math.isnan(x) or math.isinf(x))
    except Exception:
        return False

def _np_convert(obj: Any) -> Any:
    if _np is None:
        return obj
    try:
        if isinstance(obj, (_np.integer, _np.floating, _np.bool_)):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return obj

def make_json_safe(obj: Any, _seen: set[int] | None = None) -> Any:
    """Wandelt beliebige Python-Objekte in JSON-kompatible Strukturen (rekursiv, zyklensicher)."""
    if _seen is None:
        _seen = set()

    if obj is None or isinstance(obj, (str, int, bool)):
        return obj

    if isinstance(obj, float):
        return None if _is_nan_or_inf(obj) else obj

    if isinstance(obj, Decimal):
        # als String ausgeben, um Präzision zu erhalten
        return str(obj)

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    obj = _np_convert(obj)

    if isinstance(obj, (str, int, bool, float)) or obj is None:
        return obj

    if isinstance(obj, (set, tuple, list)):
        return [make_json_safe(x, _seen) for x in obj]

    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            key = k if isinstance(k, str) else str(k)
            out[key] = make_json_safe(v, _seen)
        return out

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [make_json_safe(x, _seen) for x in obj]

    oid = id(obj)
    if oid in _seen:
        return str(obj)
    _seen.add(oid)

    if hasattr(obj, "__dict__"):
        try:
            return make_json_safe(vars(obj), _seen)
        except Exception:
            return str(obj)

    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

def pretty_json(obj: Any) -> str:
    try:
        return json.dumps(make_json_safe(obj), indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


# =========================================================
# 🧩 OPTIONAL: Header-Refinement für „fehl-erkannte“ Kopfzeilen
# (wirkt nur, wenn du `refine_header_index(...)` selbst aufrufst)
# =========================================================

SAP_SUSA_HEADER_KEYWORDS = {
    "sachkonto", "kurztext", "saldovortrag", "saldo der vorperioden",
    "soll berichtszeitraum", "haben berichtszeitraum", "kum. saldo",
    "bukr", "währg", "gsbe"
}

def _row_keyword_hits(row: Sequence[Any]) -> int:
    """Zählt, wie viele typische SAP-SuSa-Wörter in einer Zeile vorkommen."""
    hits = 0
    for v in row:
        if isinstance(v, str):
            s = v.strip().lower()
            for kw in SAP_SUSA_HEADER_KEYWORDS:
                if kw in s:
                    hits += 1
    return hits

def _row_numeric_ratio(row: Sequence[Any]) -> float:
    """Anteil numerischer Zellen in einer Zeile (grobe Heuristik)."""
    total = len(row) if row else 1
    num = 0
    for v in row:
        if v is None:
            continue
        if isinstance(v, (int, float)):
            num += 1
            continue
        s = str(v).strip()
        s2 = s.replace(".", "").replace(",", "")
        if s2.isdigit():
            num += 1
    return num / total

def refine_header_index(
    df: Any,
    cand_idx: int,
    *,
    max_lookahead: int = 10,
    min_keyword_hits: int = 2,
    numeric_threshold: float = 0.70,
) -> int:
    """
    Wenn die zunächst gewählte Kopfzeile zu numerisch ist, suche in den
    nächsten Zeilen nach einer typischen SAP-SuSa-Kopfzeile.
    Gibt immer einen Index zurück; bei Fehlern unverändert `cand_idx`.

    Nutzung (optional):
        cand_idx = refine_header_index(df, cand_idx)
    """
    try:
        row = df.iloc[cand_idx].tolist()
    except Exception:
        return cand_idx

    try:
        if _row_numeric_ratio(row) <= numeric_threshold:
            return cand_idx
        for off in range(1, max_lookahead + 1):
            i = cand_idx + off
            try:
                hits = _row_keyword_hits(df.iloc[i].tolist())
            except Exception:
                break
            if hits >= min_keyword_hits:
                logging.info("Header-Refinement: %s → %s (SAP-SuSa Schlüsselwörter erkannt)", cand_idx, i)
                return i
    except Exception:
        pass
    return cand_idx


# =========================================================
# 🔎 COA-Erkennung (für Aggregator) – leichtgewichtige Heuristik
# =========================================================

def _extract_account_candidates(source: Any) -> List[str]:
    """
    Nimmt dict (Swagger/SheetResult-ähnlich) oder Pandas DataFrame
    und versucht eine Spalte 'Konto'/'Kontonummer'/'account' etc. zu finden.
    Gibt Liste von Konten-Strings zurück (ohne None/leer).
    """
    try:
        import pandas as pd  # optional
    except Exception:
        pd = None  # type: ignore

    accounts: List[str] = []

    # Dict (Swagger-ähnlich)
    if isinstance(source, dict):
        rows = source.get("preview") or source.get("rows") or []
        keys = ["Konto", "Kontonummer", "account", "Account", "Sachkonto", "Hauptkonto"]
        for r in rows:
            if not isinstance(r, dict):
                continue
            for k in keys:
                if k in r and r[k] not in (None, ""):
                    accounts.append(str(r[k]).strip())
        return accounts

    # Pandas DataFrame
    if pd is not None:
        try:
            if hasattr(source, "columns"):
                df = source
                cols = [str(c) for c in df.columns]
                cand = None
                for k in ["Konto", "Kontonummer", "account", "Account", "Sachkonto", "Hauptkonto"]:
                    if k in cols:
                        cand = k
                        break
                if cand:
                    for v in df[cand].dropna().tolist():
                        accounts.append(str(v).strip())
                    return accounts
        except Exception:
            pass

    return accounts

def detect_coa_type(source: Any) -> str:
    """
    Einfache Heuristik zur Kontenrahmen-Erkennung:
      - Viele 8xxx/49xx Umsatzerlöse → SKR03
      - Viele 5xxx/6xxx Aufw./Erträge → SKR04
      - Sonst 'CUSTOM'
    """
    accounts = _extract_account_candidates(source)
    if not accounts:
        return "CUSTOM"

    # Nur Ziffern extrahieren (z.B. '0320000' -> '320000')
    nums: List[str] = []
    for a in accounts:
        m = re.search(r"\d{3,}", a)
        if m:
            nums.append(m.group(0).lstrip("0") or "0")

    if not nums:
        return "CUSTOM"

    skr03_hits = 0
    skr04_hits = 0
    for s in nums:
        try:
            n = int(s[:4]) if len(s) >= 4 else int(s)
        except ValueError:
            continue
        # Marker
        if 8000 <= n <= 8999:
            skr03_hits += 2  # starke Indikation (Erlöse SKR03)
        if 4000 <= n <= 4999:
            skr03_hits += 1  # Wareneinsatz etc. (SKR03)
        if 5000 <= n <= 6999:
            skr04_hits += 2  # starke Indikation SKR04
        if 3000 <= n <= 3999:
            skr04_hits += 1  # SKR04 häufig Kostenarten dort

    if skr03_hits == skr04_hits == 0:
        return "CUSTOM"
    return "SKR03" if skr03_hits >= skr04_hits else "SKR04"
