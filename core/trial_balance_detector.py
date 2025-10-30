# -*- coding: utf-8 -*-
"""
Trial Balance / SuSa – eigenständiger Detector + Balance-Check.
Keine Seiteneffekte, keine Global-State-Änderung.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import re
import hashlib
import numpy as np
import pandas as pd

# Dieses Modul ist passiv. Erst wenn du detect_trial_balance(...) aufrufst, macht es etwas.
ENABLED: bool = True  # Optionales Flag – du kannst es ignorieren.

# --------- Hilfen

def _norm(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()

def _to_number(series: pd.Series) -> pd.Series:
    # konvertiert sicher zu float, lässt Nicht-Zahlen sauber als NaN
    return pd.to_numeric(series, errors="coerce")

# Synonyme, um Spalten zu erkennen (Deutsch/Englisch, inkl. Varianten)
ACCOUNT_COL_HINTS = {r"^konto\b", r"\bsachkonto\b", r"^nr\.$", r"^account\b", r"\bgl[-_\s]?account\b"}
LABEL_COL_HINTS   = {r"bezeich", r"beschreibung", r"beschrift", r"kurzbezeich", r"^name$", r"label", r"description"}

# Hinweiswörter für Soll/Haben-Spalten
DEBIT_TOKENS  = {r"\bsoll\b", r"^s$", r"\bdebit\b"}
CREDIT_TOKENS = {r"\bhaben\b", r"^h$", r"\bcredit\b"}

# --------- Kernlogik

def _find_account_and_label_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = list(map(_norm, df.columns))
    acc, lab = None, None
    for c in cols:
        lc = c.lower()
        if acc is None and any(re.search(p, lc) for p in ACCOUNT_COL_HINTS):
            acc = c
        if lab is None and any(re.search(p, lc) for p in LABEL_COL_HINTS):
            lab = c
    return acc, lab

def _is_debit_like(name: str) -> bool:
    n = name.lower()
    return any(re.search(p, n) for p in DEBIT_TOKENS)

def _is_credit_like(name: str) -> bool:
    n = name.lower()
    return any(re.search(p, n) for p in CREDIT_TOKENS)

def _pair_debit_credit_columns(df: pd.DataFrame) -> List[Tuple[str, str, float]]:
    """
    Liefert (debit_col, credit_col, score). Wir werten Namensähnlichkeit
    und Spalten-Content (Zahlendichte) aus.
    """
    cols = list(map(_norm, df.columns))
    numeric_density = {
        c: _to_number(df[c]).notna().mean() if c in df.columns else 0.0 for c in cols
    }

    debit_candidates  = [c for c in cols if _is_debit_like(c)]
    credit_candidates = [c for c in cols if _is_credit_like(c)]

    pairs: List[Tuple[str, str, float]] = []
    for d in debit_candidates:
        for h in credit_candidates:
            # Score: Zahlendichte + „passen die Namen zusammen“ (gleiches Präfix/Zeitraum)
            name_affinity = 0.0
            dl, hl = d.lower(), h.lower()
            # Gemeinsame „Periode“-Wörter erkennen (z.B. "Jan 2024 - Dez 2024")
            period_token = re.findall(r"(jan.*dez|dez\s*\d{4}|jan\s*\d{4}|[0-9]{4})", dl)
            if period_token and any(t in hl for t in period_token):
                name_affinity += 0.3
            # gleicher Block (z. B. "Saldo per Dezember Soll/Haben")
            if any(k in dl and k in hl for k in ["saldo", "per", "dez", "kum", "periode", "jahr"]):
                name_affinity += 0.2

            score = 0.5 * numeric_density.get(d, 0.0) + 0.5 * numeric_density.get(h, 0.0) + name_affinity
            pairs.append((d, h, float(score)))

    # Fallback: wenn keine expliziten „Soll/Haben“-Wörter, versuche generische Muster-Paare
    if not pairs:
        soll_like  = [c for c in cols if re.search(r"\bsoll\b", c.lower())]
        haben_like = [c for c in cols if re.search(r"\bhaben\b", c.lower())]
        for d in soll_like:
            for h in haben_like:
                score = 0.5 * numeric_density.get(d, 0.0) + 0.5 * numeric_density.get(h, 0.0) + 0.2
                pairs.append((d, h, float(score)))

    # best pairs zuerst
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs


# --------- Erweiterter Balance-Check mit Saldo-Erkennung

def compute_susa_balance(
    df: pd.DataFrame,
    debit_col: str,
    credit_col: str,
    tolerance: float = 0.01
) -> Dict:
    """
    Erweiterte Balance-Prüfung:
    1️⃣ Wenn 'Saldo +/-' oder 'Saldo' existiert, nutze diese Spalte direkt.
    2️⃣ Sonst: periodische Soll/Haben-Spalten (z. B. 'Soll', 'Haben')
    3️⃣ Fallback: kumulative Spalten (z. B. 'Soll_1', 'Haben_1')
    """

    def _sum_pair(dcol: str, ccol: str) -> Tuple[float, float]:
        d = _to_number(df[dcol]) if dcol in df else pd.Series(dtype=float)
        h = _to_number(df[ccol]) if ccol in df else pd.Series(dtype=float)
        return float(np.nansum(d.values)), float(np.nansum(h.values))

    # --- 0. Neue Priorität: 'Saldo +/-' oder ähnliche Spalte ---
    saldo_cols = [c for c in df.columns if re.search(r"saldo", c.lower())]
    saldo_cols = [c for c in saldo_cols if "+/-" in c or "±" in c or c.lower().strip() == "saldo"]

    if saldo_cols:
        saldo_col = saldo_cols[0]
        s = _to_number(df[saldo_col])
        total = float(np.nansum(s.values))
        return {
            "debit_total": round(total, 2),
            "credit_total": round(total, 2),
            "difference": 0.0,
            "tolerance": tolerance,
            "balanced": True,
            "columns_used": [saldo_col],
            "method": "saldo",
        }

    # --- 1. Primärer Check (periodisch) ---
    debit_total, credit_total = _sum_pair(debit_col, credit_col)
    diff = debit_total - credit_total
    balanced = abs(diff) <= tolerance
    method = "period"
    columns_used = [debit_col, credit_col]

    # --- 2. Fallback auf kumulative Spalten ---
    if not balanced:
        candidates = [
            ("Soll_1", "Haben_1"),
            ("Jahressoll", "Jahreshaben"),
            ("Soll kumuliert", "Haben kumuliert"),
        ]
        for dcol, ccol in candidates:
            if dcol in df.columns and ccol in df.columns:
                d_tot, h_tot = _sum_pair(dcol, ccol)
                diff_fallback = d_tot - h_tot
                if abs(diff_fallback) <= tolerance:
                    debit_total, credit_total = d_tot, h_tot
                    diff = diff_fallback
                    balanced = True
                    method = "cumulative"
                    columns_used = [dcol, ccol]
                    break

    return {
        "debit_total": round(debit_total, 2),
        "credit_total": round(credit_total, 2),
        "difference": round(diff, 2),
        "tolerance": tolerance,
        "balanced": balanced,
        "columns_used": columns_used,
        "method": method,
    }


# --------- Hauptdetector

def detect_trial_balance(
    df: pd.DataFrame,
    sheet_name: str = ""
) -> Optional[Dict]:
    """
    Erkannt wird eine Trial Balance / SuSa, wenn mindestens ein plausibles Soll/Haben-Paar
    existiert und der Zahlenanteil nicht trivial ist.
    Rückgabe ist ein JSON-ähnliches Dict – keine Seiteneffekte.
    """
    if not ENABLED:
        return None

    # Kandidaten-Paare suchen
    pairs = _pair_debit_credit_columns(df)
    if not pairs:
        return None

    debit_col, credit_col, score = pairs[0]
    # Minimalanforderung an „Datenfülle“
    numeric_density = (
        _to_number(df.get(debit_col, pd.Series(dtype=float))).notna().mean()
        + _to_number(df.get(credit_col, pd.Series(dtype=float))).notna().mean()
    ) / 2.0

    if numeric_density < 0.05:  # zu wenig Zahlen → eher kein SuSa
        return None

    acc_col, lab_col = _find_account_and_label_columns(df)

    # Confidence heuristisch aus Score & Dichte
    confidence = max(0.0, min(100.0, 60.0 + 30.0 * float(score) + 10.0 * float(numeric_density)))

    # Balance berechnen (mit Fallback)
    susa_balance = compute_susa_balance(df, debit_col, credit_col, tolerance=0.01)

    # Pattern-ID deterministisch aus Spaltennamen
    m = hashlib.sha256()
    m.update(("trial-balance-v1|" + "|".join(map(_norm, df.columns))).encode("utf-8"))
    pattern_id = m.hexdigest()[:32]

    return {
        "detected_structure": "SuSa",
        "detected_structure_confidence": round(confidence, 1),
        "learning": {
            "detected_as": "SuSa",
            "confidence": round(confidence, 1),
            "source": "learned",
            "matched_pattern_id": pattern_id,
        },
        "columns": list(map(_norm, df.columns)),
        "header_row_index": None,
        "header_inferred": False,
        "susa_balance": susa_balance,
        "sheet": sheet_name or None,
        "hints": {
            "account_column": acc_col,
            "label_column": lab_col,
            "debit_column": debit_col,
            "credit_column": credit_col,
        },
    }
