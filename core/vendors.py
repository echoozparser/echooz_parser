# core/vendors.py
from __future__ import annotations
from typing import Iterable
from core.helpers import normalize_text

_SAP_TRIGGERS = (
    "sachkontensalden",
    "rfssld00",         # SAP-Report-Kürzel
    "bukr", "bukreis",  # Buchungskreis
    "gsbe",             # Geschäftsbereich
    "berichtszeitraum",
)

def detect_vendor(sample_cells: Iterable[str]) -> str | None:
    """
    Heuristik: erkenne SAP, falls mehrere typische Trigger vorkommen.
    """
    txt = " ".join(normalize_text(x) for x in sample_cells if x)
    score = sum(1 for t in _SAP_TRIGGERS if t in txt)
    return "SAP" if score >= 2 else None
