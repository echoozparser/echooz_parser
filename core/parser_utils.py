# core/parser_utils.py
from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# ------------------------------------------------------------
# Normalisierung / Parsing
# ------------------------------------------------------------

def _norm_text(x: Any) -> str:
    """Unicode-normalisiert, NBSP -> Space, trimmt; None/NaN -> ''."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = unicodedata.normalize("NFKC", str(x))
    s = s.replace("\u00A0", " ").strip()
    return s


def coerce_number(val: Any) -> Optional[float]:
    """
    Robust: akzeptiert 1.234,56 / 1,234.56 / (1.234,56) / -1.234,56 / '  0  '
    Gibt float oder None (bei leer/unparsbar) zurück.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return None if (isinstance(val, float) and math.isnan(val)) else float(val)

    s = _norm_text(val)
    if not s:
        return None

    # Klammernegativ
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]

    # Leerzeichen/Tausender entfernen
    s = s.replace(" ", "")

    # 1) deutscher Stil: 1.234,56
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})*,\d{1,2}", s):
        s = s.replace(".", "").replace(",", ".")
    # 2) anglo Stil: 1,234.56
    elif re.fullmatch(r"-?\d{1,3}(,\d{3})*\.\d{1,2}", s):
        s = s.replace(",", "")
    # 3) reine Ganzzahl mit Tausendern (1.234 oder 1,234)
    elif re.fullmatch(r"-?\d{1,3}([.,]\d{3})+", s):
        s = s.replace(",", "").replace(".", "")

    try:
        out = float(s)
        return -out if neg else out
    except ValueError:
        return None


# ------------------------------------------------------------
# Summenzeilen-Filter
# ------------------------------------------------------------

TOTAL_PATTERNS = re.compile(
    r"\b(summe|gesamtsumme|end\s*summe|endsumme|saldo|total|grand\s*total|sum\b|Σ)\b",
    re.IGNORECASE,
)

def _looks_like_total_label(s: str) -> bool:
    return bool(s) and bool(TOTAL_PATTERNS.search(s))


def drop_total_rows(
    df: pd.DataFrame,
    label_cols: Iterable[str],
    numeric_cols: Optional[Iterable[str]] = None,
    tail_ratio: float = 0.9,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Entfernt Summen-/Endzeilen robust:
    - Keyword in einer Labelspalte (de/en)
    - UND (Zeilenindex im unteren Tail, default: letzte 10 %) ODER (Zeile numerisch „meist leer“)
    Gibt (bereinigtes DF, Meta) zurück.
    """
    n = len(df)
    cut = int(n * tail_ratio)
    label_cols = [c for c in label_cols if c in df.columns]

    if numeric_cols is None:
        numeric_cols = [
            c
            for c in df.columns
            if str(c).strip().lower()
            in {
                "soll",
                "haben",
                "soll_1",
                "haben_1",
                "debit",
                "credit",
                "saldo",
                "balance",
                "amount",
            }
        ]
    else:
        numeric_cols = [c for c in numeric_cols if c in df.columns]

    to_drop: List[int] = []
    last_match_idx: Optional[int] = None

    for i, row in df.iterrows():
        labels_joined = " | ".join(_norm_text(row.get(c, "")) for c in label_cols)
        has_kw = _looks_like_total_label(labels_joined)
        in_tail = i >= cut

        mostly_empty_nums = False
        if numeric_cols:
            vals = [coerce_number(row.get(c)) for c in numeric_cols]
            nones = sum(v is None for v in vals)
            # „meist leer/NA“: >50 % None
            mostly_empty_nums = nones >= max(1, int(0.5 * len(vals)))

        if has_kw and (in_tail or mostly_empty_nums):
            to_drop.append(i)
            last_match_idx = i

    cleaned = df.drop(index=to_drop).reset_index(drop=True)
    meta = {
        "total_rows_removed": len(to_drop),
        "last_total_row_index": int(last_match_idx) if last_match_idx is not None else None,
        "tail_cutoff_index": cut,
    }
    return cleaned, meta


# ------------------------------------------------------------
# Balance-Check via Wertspalte + S/H (oder D/C) Marker
#   * NEU: prüft ALLE potenziellen Wertspalten (Saldo, Balance, Bilanz)
#          gegen ALLE potenziellen Marker-Spalten im Sheet.
#   * wählt die Kombination mit der kleinsten |Differenz|.
#   * testet zusätzlich Flip-/Abs-Varianten (Export-Spezialfälle).
# ------------------------------------------------------------

# Wertspalten-Kandidaten: „Saldo“, „Balance“, „Bilanz“
_VALUE_NAME_PAT = re.compile(r"\b(saldo|balance|bilanz)\b", re.IGNORECASE)

def _is_marker_series(series: pd.Series) -> bool:
    """
    Kandidat ist Marker-Spalte, wenn (nach Upper/Trim) die Menge der
    Nicht-NaN-Werte eine echte Teilmenge von {S,H,D,C} ist.
    """
    try:
        vals = series.astype(str).str.strip().str.upper()
        uniq = {v for v in vals.unique() if v and v != "NAN"}
        return bool(uniq) and uniq.issubset({"S", "H", "D", "C"})
    except Exception:
        return False


def _candidate_marker_cols(df: pd.DataFrame) -> List[str]:
    out: List[str] = []
    for col in df.columns:
        try:
            if _is_marker_series(df[col]):
                out.append(col)
        except Exception:
            continue
    return out


def _candidate_value_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if _VALUE_NAME_PAT.search(_norm_text(c))]


def _signed_sum(values: List[Any], marks: List[str], *, use_abs: bool, flip_sign: bool) -> float:
    s = 0.0
    used = 0
    for v_raw, m in zip(values, marks):
        v = coerce_number(v_raw)
        if v is None or v == 0.0:
            continue
        m = (m or "").strip().upper()
        val = abs(v) if use_abs else v
        # Marker-Sinn:
        # - normal:  S/D => +,  H/C => -
        # - flipped: S/D => -,  H/C => +
        if m in {"S", "D"}:
            s += (-val if flip_sign else +val)
        elif m in {"H", "C"}:
            s += (+val if flip_sign else -val)
        else:
            continue
        used += 1
    # wenn wir nie eine valide Paarung hatten → return NaN als Signal
    return s if used > 0 else float("nan")


def check_balance_via_saldo_marker(df: pd.DataFrame) -> Optional[Tuple[float, List[str]]]:
    """
    Prüft Ausgeglichenheit über eine Wertspalte (Saldo/Balance/Bilanz) + Marker-Spalte (S/H oder D/C).
    Rückgabe:
        (difference, [value_col, marker_col])   # difference ~ 0 => balanced
    oder:
        None, wenn nicht anwendbar.
    """
    value_cols = _candidate_value_cols(df)
    if not value_cols:
        return None

    marker_cols = _candidate_marker_cols(df)
    if not marker_cols:
        # als Minimal-Fallback: Nachbar einer jeden Value-Spalte prüfen
        # (unterstützt Exporte, bei denen Marker direkt daneben steht)
        for val_col in value_cols:
            nb = _find_neighbor_marker_col(df, val_col)
            if nb:
                marker_cols = [nb]
                break
        if not marker_cols:
            return None

    # alle Kombinationen durchprobieren, die mit parsebaren Werten arbeiten
    best: Optional[Tuple[float, str, str]] = None  # (abs_diff, value_col, marker_col)

    for vcol in value_cols:
        v_series = df[vcol].tolist()

        for mcol in marker_cols:
            m_series = df[mcol].astype(str).str.strip().str.upper().tolist()

            # 4 Modi: (use_abs x flip_sign)
            diffs: List[float] = []
            for use_abs in (False, True):
                for flip_sign in (False, True):
                    diff = _signed_sum(v_series, m_series, use_abs=use_abs, flip_sign=flip_sign)
                    if not math.isnan(diff):
                        diffs.append(diff)

            if not diffs:
                continue

            # wähle kleinste |Differenz|
            diff_min = min(diffs, key=lambda d: abs(d))
            candidate = (abs(diff_min), vcol, mcol)

            if (best is None) or (candidate[0] < best[0]):
                best = candidate

    if best is None:
        return None

    # tatsächliche Differenz für das best pair erneut, um Vorzeichen zu liefern:
    _, vcol, mcol = best
    # Wiederholung mit der besten Variante finden (nochmals testen, um „echte“ diff inkl. Vorzeichen zu liefern)
    v_series = df[vcol].tolist()
    m_series = df[mcol].astype(str).str.strip().str.upper().tolist()

    chosen_diff: Optional[float] = None
    chosen_val: Optional[float] = None
    for use_abs in (False, True):
        for flip_sign in (False, True):
            d = _signed_sum(v_series, m_series, use_abs=use_abs, flip_sign=flip_sign)
            if math.isnan(d):
                continue
            if (chosen_val is None) or (abs(d) < chosen_val):
                chosen_val = abs(d)
                chosen_diff = d

    if chosen_diff is None:
        return None

    return float(round(chosen_diff, 2)), [vcol, mcol]


# ------------------------------------------------------------
# (Kompatibilität) – alter Nachbar-Scanner bleibt als Helfer
# ------------------------------------------------------------

_SALDO_NAME_PAT = re.compile(r"\b(saldo|balance)\b", re.IGNORECASE)

def _find_neighbor_marker_col(df: pd.DataFrame, col_name: str) -> Optional[str]:
    """
    Sucht links/rechts neben der Wertspalte nach einer Spalte, deren Werte nur aus {S,H,D,C} bestehen.
    (Wird heute nur noch als Minimal-Fallback verwendet.)
    """
    cols = list(df.columns)
    try:
        i = cols.index(col_name)
    except ValueError:
        return None

    candidates: List[str] = []
    if i > 0:
        candidates.append(cols[i - 1])
    if i + 1 < len(cols):
        candidates.append(cols[i + 1])

    for nb in candidates:
        series = df[nb].astype(str).str.strip().str.upper()
        uniq = {v for v in series.unique() if v and v != "NAN"}
        if uniq and uniq.issubset({"S", "H", "D", "C"}):
            return nb
    return None
