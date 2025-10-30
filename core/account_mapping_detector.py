# -*- coding: utf-8 -*-
"""
Detector für Account-Mapping/Zuordnungstabellen (z. B. „TB“-Sheet in Test 12),
die KEINE echte SuSa sind (keine Soll/Haben-Paarung für Summenprüfung).
Keine Seiteneffekte, keine Änderungen an bestehender Logik.
"""

from __future__ import annotations
from typing import Dict, Optional, List
import re
import hashlib
import pandas as pd
import numpy as np

ENABLED: bool = True  # Optionales Flag – rein passiv, solange nicht benutzt.

def _norm(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()

def _has_any(df: pd.DataFrame, patterns: List[str]) -> bool:
    cols = [c.lower() for c in map(_norm, df.columns)]
    return any(any(re.search(p, c) for p in patterns) for c in cols)

def _has_debit_credit_pair(df: pd.DataFrame) -> bool:
    cols = [c.lower() for c in map(_norm, df.columns)]
    has_debit  = any(re.search(r"\bsoll\b|^s$|\bdebit\b", c) for c in cols)
    has_credit = any(re.search(r"\bhaben\b|^h$|\bcredit\b", c) for c in cols)
    return has_debit and has_credit

def _looks_like_mapping_codes(series: pd.Series) -> float:
    """
    Heuristik: Spalte enthält überwiegend „Code“-artige Werte (Zahlenketten/IDs).
    Rückgabe: Anteil codeartiger Einträge (0..1).
    """
    s = series.astype(str).str.strip()
    # Erlaubt numerische IDs mit 3–12 Stellen, optional Bindestriche/Punkte
    mask = s.str.fullmatch(r"[0-9][0-9\.\-]{1,20}") | s.str.fullmatch(r"[0-9]{3,12}")
    # Viele NaNs ignorieren
    denom = max(1, mask.shape[0] - series.isna().sum())
    return float(mask.sum() / denom)

def detect_account_mapping(
    df: pd.DataFrame,
    sheet_name: str = ""
) -> Optional[Dict]:
    """
    Erkennt Mapping/Zuordnungstabellen:
    - Enthalten oft Spalten wie „Balance sheet“, „Debit/Credit“ als Texte,
      „N“-Flags etc.
    - Haben KEIN belastbares Soll/Haben-Paar für Summenabgleich.
    - Mehrere (>=2) „codeartige“ Spalten (Quell- und Zielkonten).
    Gibt ein JSON-ähnliches Dict zurück; `susa_balance` ist immer None.
    """
    if not ENABLED:
        return None

    cols_norm = list(map(_norm, df.columns))
    cols_lower = [c.lower() for c in cols_norm]

    # Wenn bereits echtes Soll/Haben-Paar existiert, ist es KEIN reines Mapping.
    if _has_debit_credit_pair(df):
        return None

    # Schlüsselwörter, die in Mapping-Layouts häufig vorkommen
    meta_hints = [
        r"\bbalance\s*sheet\b",
        r"\bincome\s*statement\b",
        r"^n$",                     # Flag-Spalte „N“
        r"\bcredit\b",
        r"\bdebit\b",
        r"\bassets\b",
        r"\bliabilities\b",
    ]
    has_meta = _has_any(df, meta_hints)

    # Code-ähnliche Spalten zählen
    code_like_cols = 0
    for c in df.columns:
        try:
            ratio = _looks_like_mapping_codes(df[c])
            if ratio >= 0.6:
                code_like_cols += 1
        except Exception:
            pass

    # Zusätzlich: mindestens 1 Textlabel-Spalte (z. B. „Balance sheet“, „Beschreibung“)
    has_labelish = any(re.search(r"balance|bezeich|name|beschrift|beschreibung|label|konto", c) for c in cols_lower)

    # Heuristische Entscheidung
    if (has_meta and code_like_cols >= 1) or (code_like_cols >= 2 and has_labelish):
        # Confidence vorsichtig aber nutzbar
        confidence = 80.0 if has_meta and code_like_cols >= 2 else 60.0

        m = hashlib.sha256()
        m.update(("account-mapping-v1|" + "|".join(cols_norm)).encode("utf-8"))
        pattern_id = m.hexdigest()[:32]

        return {
            "detected_structure": "AccountMapping",
            "detected_structure_confidence": round(confidence, 1),
            "learning": {
                "detected_as": "AccountMapping",
                "confidence": round(confidence, 1),
                "source": "learned",
                "matched_pattern_id": pattern_id,
            },
            "columns": cols_norm,
            "header_row_index": None,
            "header_inferred": True,   # solche Tabellen haben oft „nicht klassische“ Header
            "susa_balance": None,
            "sheet": sheet_name or None,
        }

    return None
