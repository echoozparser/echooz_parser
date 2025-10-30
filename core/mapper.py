# core/mapper.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import pandas as pd

# Wir nutzen denselben Normalizer wie im Parser
from .helpers import normalize_text as _norm_txt


def _norm(s: str) -> str:
    return _norm_txt(str(s)) if s is not None else ""


def _find_first(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """
    Liefert die erste Quellspalte, deren normalisierter Name
    eines der Keyword-Fragmente enthält.
    """
    if df is None or df.columns is None:
        return None
    norm_map = {col: _norm(col) for col in df.columns}
    for col, n in norm_map.items():
        for kw in keywords:
            # WICHTIG: keywords sind bereits "normalisiert" notiert (klein, ascii, ohne Sonderzeichen wie '+')
            if kw in n:
                return col
    return None


def map_to_canonical(
    df: pd.DataFrame,
    structure: Optional[str] = None,
    prefer_coa: Optional[str] = None,   # "skr03" | "skr04" | None
    pack_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Heuristisches, kontext-sensitives Mapping auf unser Kanon-Schema.
    Spezieller Fokus auf SAP-/DATEV-SuSa:
      - opening_balance: **Saldovortrag** bevorzugt, "Saldo der Vorperioden" nur Fallback
      - balance: **Kum. Saldo** (o.ä.) bevorzugt
      - debit_period/credit_period: Monats-/Berichtszeitraum-Summen (bzw. schlichte "Soll/Haben")
      - debit_cum/credit_cum: Jahres-/kumulative Summen (z.B. "Soll_1/Haben_1")
    Erweiterungen (additiv):
      - Synonyme um „Nummer“, „Beschreibung“, „Saldo +/-“ ergänzt
      - Opening-Balance Fallback inkl. „EB-Wert“
      - Confidence-Fallback für 3-Spalten-SuSa (Konto|Text|Saldo), falls keine Soll/Haben-Spalten existieren
    """

    # --- Kandidatenlisten (alle normalisiert gematcht) ---
    # Konto: um "nummer" & diverse Schreibweisen erweitert
    acct_kw = [
        "sachkonto", "kontonummer", "kontonr", "konto nr", "konto-nr", "konto-nr.",
        "konto", "account", "account no", "account no.", "nr", "nr.", "nummer"  # NEU
    ]

    # Label/Text: um „beschreibung“ erweitert
    label_kw = [
        "kurztext", "bezeichnung", "beschriftung", "konto-bezeichnung", "text",
        "beschreibung"  # NEU
    ]

    # Opening Balance: klare Priorität für "Saldovortrag"
    ob_prio_kw = [
        "saldovortrag", "anfangsbestand", "eroeffnung", "eröffnungsbilanz", "eroeffnungsbilanz"
    ]
    # Fallback inkl. „EB-Wert“
    ob_fallback = [
        "saldo der vorperioden", "vorperioden", "vorperiode",
        "eb wert", "eb-wert", "eb werte", "ebwerte", "eb"  # NEU
    ]

    # Periodenwerte (Soll/Haben)
    d_per_exact = ["soll"]    # exakte Kurzform
    c_per_exact = ["haben"]   # exakte Kurzform
    d_per_kw = ["monatssoll", "soll berichtszeitraum", "periodensoll", "periode soll", "monats-soll", "soll"]
    c_per_kw = ["monatshaben", "haben berichtszeitraum", "periodenhaben", "periode haben", "monats-haben", "haben"]

    # Kumulative Jahreswerte – „_1“-Varianten explizit
    d_cum_exact = ["soll_1", "soll1", "kum_soll", "kum.soll"]
    c_cum_exact = ["haben_1", "haben1", "kum_haben", "kum.haben"]
    d_cum_kw = ["jahressoll", "kumuliert soll", "kum soll", "kum.soll", "jahres-soll", "kum werte soll"]  # leicht breiter
    c_cum_kw = ["jahreshaben", "kumuliert haben", "kum haben", "kum.haben", "jahres-haben", "kum werte haben"]

    # Salden – bevorzugt kumulierte Salden
    bal_kw_prio = ["kum. saldo", "kum saldo", "kumuliert saldo", "kum.saldo", "jahressaldo", "saldo jahr"]
    # Alternative inkl. „saldo +/-“; Achtung: '+' wird im Normalizer entfernt -> wir matchen normalisierte Varianten
    bal_kw_alt = [
        "monatssaldo", "periodensaldo", "monatsaldo", "saldo",
        "saldo /-", "saldo -", "saldo plus/minus", "saldo plus minus", "saldo plusminus", "saldo +-"
    ]

    # Meta
    bukr_kw = ["bukr", "bukrs", "company code", "gesellschaft"]
    gsbe_kw = ["gsbe", "geschaeftsbereich", "geschäftsbereich", "cost center", "kostenstelle"]
    curr_kw = ["waehrg", "währung", "waehrung", "currency", "curr"]

    # --- kleine Helfer nur für diese Funktion ---
    def _find_exact(df_: pd.DataFrame, exact_list: List[str]) -> Optional[str]:
        if df_ is None or df_.columns is None:
            return None
        nm = {col: _norm(col) for col in df_.columns}
        for col, n in nm.items():
            for ex in exact_list:
                if n == _norm(ex):
                    return col
        return None

    def _prefer_plain(df_: pd.DataFrame, plain: str, candidate: Optional[str]) -> Optional[str]:
        """
        Wenn 'candidate' wie 'soll_1'/'haben_1' ist und 'plain' ('soll'/'haben') existiert,
        nimm die schlichte Spalte als Periodenwert.
        """
        if candidate is None:
            return None
        cand_n = _norm(candidate).replace(" ", "")
        if any(tag in cand_n for tag in ("soll1", "soll_1", "haben1", "haben_1")):
            plain_col = _find_exact(df_, [plain])
            if plain_col:
                return plain_col
        return candidate

    # --- Spalten picken ---
    account = _find_first(df, acct_kw)
    label   = _find_first(df, label_kw)

    opening_balance = _find_first(df, ob_prio_kw)
    if opening_balance is None:
        opening_balance = _find_first(df, ob_fallback)

    # Perioden zuerst exakt (schlichte "Soll/Haben"), dann heuristisch; "_1" zugunsten der schlichten Form verdrängen
    debit_period  = _find_exact(df, d_per_exact) or _find_first(df, d_per_kw)
    credit_period = _find_exact(df, c_per_exact) or _find_first(df, c_per_kw)
    debit_period  = _prefer_plain(df, "soll", debit_period)
    credit_period = _prefer_plain(df, "haben", credit_period)

    # Kumulativ: zuerst exakte „_1“-Varianten, dann Keyword-Suche
    debit_cum  = _find_exact(df, d_cum_exact) or _find_first(df, d_cum_kw)
    credit_cum = _find_exact(df, c_cum_exact) or _find_first(df, c_cum_kw)

    balance       = _find_first(df, bal_kw_prio) or _find_first(df, bal_kw_alt)
    company_code  = _find_first(df, bukr_kw)
    business_unit = _find_first(df, gsbe_kw)
    currency      = _find_first(df, curr_kw)

    canonical = {
        "account":         account,
        "label":           label,
        "opening_balance": opening_balance,
        "balance":         balance,
        "debit_period":    debit_period,
        "credit_period":   credit_period,
        "debit_cum":       debit_cum,
        "credit_cum":      credit_cum,
        "company_code":    company_code,
        "business_unit":   business_unit,
        "currency":        currency,
    }

    # --- Confidence (gewichtete Abdeckung) - ALT ---
    must_keys = ["account", "label", "debit_period", "credit_period"]
    nice_keys = ["opening_balance", "balance"]
    hits_must = sum(1 for k in must_keys if canonical[k])
    hits_nice = sum(1 for k in nice_keys if canonical[k])
    # Gewichtung: MUST voll, NICE halbe Punkte
    score = hits_must + 0.5 * hits_nice
    max_score = len(must_keys) + 0.5 * len(nice_keys)
    conf_alt = int(round(100 * (score / max_score))) if max_score else 0

    # --- NEU: Confidence-Fallback für 3-Spalten-SuSa (Konto|Text|Saldo) ---
    conf_minimal = 0
    if not canonical["debit_period"] and not canonical["credit_period"] and canonical["balance"]:
        must3 = ["account", "label", "balance"]
        hits3 = sum(1 for k in must3 if canonical[k])
        conf_minimal = int(round(100 * (hits3 / len(must3))))

    confidence = max(conf_alt, conf_minimal)

    # --- COA-Pack (SKR03/04) – optionaler Hinweis ---
    notes: List[str] = []
    if pack_path and Path(pack_path).exists():
        try:
            with open(pack_path, "r", encoding="utf-8") as f:
                _ = json.load(f)  # noch nicht für harte Regeln genutzt
            if prefer_coa in ("skr03", "skr04"):
                notes.append(f"COA preference '{prefer_coa}' berücksichtigt (Mapping heuristisch).")
            else:
                notes.append("COA Auto-Modus (Mapping heuristisch).")
        except Exception:
            notes.append("COA-Pack konnte nicht gelesen werden – übersprungen.")
    else:
        notes.append("COA Auto-Modus (Mapping heuristisch).")

    # --- Unmapped-Listen ---
    unmapped_canonicals = [k for k, v in canonical.items() if v is None]
    used_cols = {v for v in canonical.values() if v}
    unmapped_source_cols = [c for c in df.columns if c not in used_cols]

    return {
        "structure": structure or "Unknown",
        "canonical_columns": canonical,
        "confidence": confidence,
        "unmapped_canonicals": unmapped_canonicals,
        "unmapped_source_columns": unmapped_source_cols,
        "notes": notes,
    }
