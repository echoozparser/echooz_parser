from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from core.helpers import normalize_text

# =========================================================
# 🧠 ECHOOZ CLASSIFIER v9.4 – Header-Heuristik, SimHash+Jaccard
# =========================================================

PATTERN_SCHEMA_VERSION = "2025-10-17"

# Privacy-Mode: SpaltenNamen nicht im Klartext persistieren
PRIVACY_HASH_COLUMNS = bool(int(os.getenv("ECHOoz_PRIVACY_HASH_COLUMNS", "0")))
HASH_LEN = int(os.getenv("ECHOoz_PRIVACY_HASH_LEN", "12"))  # sichtbare Zeichen

# Wichtige Header-Schlüsselwörter (inkl. SUSA-spezifischer Begriffe)
KEYWORDS_HEADER: Set[str] = {
    "konto", "kontonr", "kontonummer", "bezeich", "bezeichnung",
    "soll", "haben", "saldo", "betrag", "buchung", "datum", "gegenkonto",
    "monatssoll", "monatshaben", "monatssaldo",
    "jahressoll", "jahreshaben", "jahressaldo"
}

# Zeilen, die typischerweise Meta-/Deckzeilen sind (nicht als Header verwenden)
BLACKLIST_ROWS: Set[str] = {
    "firma", "bewertungsart", "periode", "seite", "saldenliste", "sachkonten",
    "eigenwaehrung", "eigenwährung", "hauswaehrung", "hauswährung",
    "datum der erstellung", "bericht", "druck", "stand", "summe seite"
}
# --- [2] SAP / SuSa Header-Synonyme (nur Ergänzung, nichts überschreiben) ---
# Achtung: normalize_text macht aus "Währg" -> "waehrg" und entfernt Punkte,
# daher benutzen wir hier bereits die normalisierten Schreibweisen.
SAP_SUSA_HEADER: Set[str] = {
    "bukr",                     # Buchungskreis
    "sachkonto",                # kommt in SAP-Exports oft so
    "kurztext",
    "waehrg",                   # Währg
    "gsbe",                     # Geschäftsbereich
    "saldovortrag",
    "saldo der vorperioden",    # auch als "saldo vorperioden" möglich
    "saldo vorperioden",
    "soll berichtszeitraum",
    "haben berichtszeitraum",
    "kum", "kum saldo", "kumuliert"
}
KEYWORDS_HEADER |= SAP_SUSA_HEADER

# Meta-/Deckzeilen in SAP, die NICHT als Header taugen:
BLACKLIST_ROWS |= {
    "sachkontensalden",   # große Titelzeile
    "vortragsperioden",
    "berichtsperioden",
    "rfssld00"            # Transaktions-/Report-Kennung in manchen Layouts
}

STRUCTURE_RULES: Dict[str, Set[str]] = {
    "SuSa": {"konto", "kontonr", "soll", "haben", "saldo", "bezeich", "bezeichnung"},
    "BWA": {"aufwand", "ertrag", "kosten", "umsatz", "erlöse", "erloese", "bwa"},
    "Bilanz": {"aktiva", "passiva", "assets", "liabilities", "equity", "eigenkapital",
               "anlagevermögen", "umlaufvermögen", "kurzfr", "langfr"},
    "Journal": {"buchung", "beleg", "datum", "konto", "gegenkonto", "betrag", "text"}
}

MAX_PATTERNS: int = 5000
MIN_LEARN_CONF: float = 60.0           # Jaccard-Ähnlichkeit (0..100) für Match
MIN_NEW_PATTERN_SIM: float = 0.80      # kein neues Pattern, wenn ≥80 % ähnlich
UPDATE_HITS_ON_MATCH: bool = True

# SimHash-Filter (64-bit): Kandidaten nur, wenn Hamming-Distanz <= THRESH
SIMHASH_HAMMING_THRESH = int(os.getenv("ECHOoz_SIMHASH_HAMMING", "16"))

# In-Prozess Cache (mit MTime-Invalidierung)
_PATTERN_CACHE: Dict[str, Any] = {"mtime": None, "patterns": []}


# ------------------------------
# Utilities
# ------------------------------
def _normalize_columns(cols: List[Any]) -> List[str]:
    normed: List[str] = []
    for c in cols:
        s = normalize_text(str(c))
        s = re.sub(r"\(.*?\)", "", s)  # Inhalte in Klammern entfernen
        s = re.sub(r"\s+", " ", s).strip()
        normed.append(s)
    return normed


def _tokenize(words: List[str]) -> Set[str]:
    toks: Set[str] = set()
    for w in words:
        toks.update(re.split(r"[^\w]+", w))
    toks.discard("")
    return toks


def _hash_token(tok: str) -> str:
    return hashlib.sha1(tok.encode("utf-8")).hexdigest()[:HASH_LEN]


def _simhash64(tokens: Set[str]) -> int:
    """Einfacher 64-bit SimHash über Tokens."""
    if not tokens:
        return 0
    v = [0] * 64
    for t in tokens:
        h = int(hashlib.blake2b(t.encode("utf-8"), digest_size=8).hexdigest(), 16)
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    fingerprint = 0
    for i in range(64):
        if v[i] >= 0:
            fingerprint |= (1 << i)
    return fingerprint


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _row_is_mostly_text(row: List[str]) -> bool:
    if not row:
        return False
    numeric_like = 0
    non_empty = 0
    for v in row:
        v = v.strip()
        if not v:
            continue
        non_empty += 1
        if re.fullmatch(r"[+-]?\d+([.,]\d+)?", v):
            numeric_like += 1
    ratio_numeric = numeric_like / max(non_empty, 1)
    return ratio_numeric < 0.5


def _score_header_row(tokens_joined: str, token_set: Set[str]) -> int:
    hits = sum(1 for k in KEYWORDS_HEADER if k in tokens_joined or k in token_set)
    return hits


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 100.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return round(100.0 * inter / union, 1)


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, str(path))


def _load_patterns(path: Path) -> List[Dict[str, Any]]:
    """Lädt Pattern-Store mit Cache & Migration."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        _atomic_write_json(path, [])
        _PATTERN_CACHE["mtime"] = None
        _PATTERN_CACHE["patterns"] = []
        return []

    mtime = stat.st_mtime
    if _PATTERN_CACHE["mtime"] == mtime and _PATTERN_CACHE["patterns"]:
        return _PATTERN_CACHE["patterns"]

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = []
    except json.JSONDecodeError:
        logging.warning("%s beschädigt, neu initialisiert.", path.name)
        data = []
        _atomic_write_json(path, data)

    # Migration/Auffüllen
    changed = False
    for p in data:
        p.setdefault("schema_version", PATTERN_SCHEMA_VERSION)
        p.setdefault("pattern_id", uuid.uuid4().hex)
        p.setdefault("source", "learned")
        p.setdefault("confidence", 0.0)
        p.setdefault("hits", 0)
        p.setdefault("confirmations", 0)
        p.setdefault("created_at", datetime.now().isoformat())
        p.setdefault("last_seen", p["created_at"])

        if "tokens_hashed" not in p:
            cols = p.get("columns", [])
            toks = _tokenize(cols) if cols else set()
            th = {_hash_token(t) for t in toks}
            p["tokens_hashed"] = sorted(th)
            changed = True

        if "simhash64" not in p:
            sim = _simhash64(set(p.get("tokens_hashed", [])))
            p["simhash64"] = int(sim)
            changed = True

        if PRIVACY_HASH_COLUMNS and "columns" in p:
            del p["columns"]
            changed = True

    if changed:
        _atomic_write_json(path, data)

    _PATTERN_CACHE["mtime"] = mtime
    _PATTERN_CACHE["patterns"] = data
    return data


def _save_patterns(path: Path, patterns: List[Dict[str, Any]]) -> None:
    if len(patterns) > MAX_PATTERNS:
        patterns = patterns[-MAX_PATTERNS:]
    _atomic_write_json(path, patterns)
    try:
        _PATTERN_CACHE["mtime"] = path.stat().st_mtime
    except Exception:
        _PATTERN_CACHE["mtime"] = None
    _PATTERN_CACHE["patterns"] = patterns


def find_header_row(df: pd.DataFrame, max_search: int = 100) -> Optional[int]:
    """
    Wählt die „beste“ Kopfzeile in den ersten max_search Zeilen:
    - überspringt Meta/Deckzeilen (Firma/Periode/Seite/…)
    - verlangt >=2 Header-Treffer
    - bevorzugt typische SuSa-Kombinationen (Soll+Haben, Konto+Bezeich.)
    """
    try:
        n = min(max_search, len(df))
        best_i, best_score = None, -10_000
        for i in range(n):
            raw = [str(x) for x in df.iloc[i].tolist()]
            row = [normalize_text(x) for x in raw if x.strip()]
            if not row:
                continue

            joined = " ".join(row)
            if any(b in joined for b in BLACKLIST_ROWS):
                continue

            tokens = set(re.split(r"[^\w]+", joined))
            header_hits = sum(1 for k in KEYWORDS_HEADER if (k in joined or k in tokens))
            meta_hits = sum(1 for b in ("firma", "seite", "bewertungsart", "periode", "bericht", "druck") if b in joined)

            non_empty = sum(1 for x in row if x.strip())
            numeric_like = sum(1 for x in row if re.fullmatch(r"[+-]?\d+([.,]\d+)?", x or ""))
            numeric_ratio = numeric_like / max(non_empty, 1)

            boost_pairs = 0
            if "soll" in joined and "haben" in joined:
                boost_pairs += 3
            if ("konto" in joined or "kontonummer" in joined) and ("bezeich" in joined):
                boost_pairs += 2

            score = header_hits * 10 + boost_pairs * 5 - meta_hits * 6 - int(numeric_ratio > 0.5) * 5

            if header_hits >= 2 and score > best_score:
                best_i, best_score = i, score

        if best_i is not None:
            logging.info("🧭 Kopfzeile gewählt: Zeile %d (Score=%d)", best_i, best_score)
            return best_i

        logging.warning("⚠️ Keine Kopfzeile gefunden.")
        return None
    except Exception as e:
        logging.error("Fehler in find_header_row(): %s", e)
        return None


def classify_structure(columns: List[Any], df: Optional[pd.DataFrame] = None) -> Tuple[str, float]:
    cols_norm = _normalize_columns(list(columns))
    tokens = _tokenize(cols_norm)
    joined = " ".join(cols_norm)

    best_type = "Unbekannt"
    best_score = 0.0

    for label, kw in STRUCTURE_RULES.items():
        hits = sum(1 for k in kw if (k in tokens or k in joined))
        score = 100.0 * hits / max(len(kw), 1)
        if score > best_score:
            best_score = score
            best_type = label

    if best_type == "SuSa":
        sh_hits = sum(1 for k in ("soll", "haben", "saldo") if (k in tokens or k in joined))
        best_score = max(best_score, 60.0 + 10.0 * sh_hits)

    if best_type == "Journal":
        j_hits = sum(1 for k in ("datum", "beleg", "konto", "betrag") if (k in tokens or k in joined))
        best_score = max(best_score, 50.0 + 12.5 * j_hits)

    if best_type == "Unbekannt":
        best_score = max(best_score, 40.0)

    return best_type, round(min(best_score, 100.0), 1)


@dataclass
class Pattern:
    pattern_id: str
    schema_version: str
    detected_as: str
    tokens_hashed: List[str]         # Hashes der Tokens
    simhash64: int
    columns: Optional[List[str]]     # optional (Privacy off)
    source: str                      # "learned" | "feedback"
    confidence: float                # interne Klassifikationssicherheit (0..100)
    hits: int                        # automatische Matches
    confirmations: int               # manuelle Bestätigungen via Feedback
    created_at: str
    last_seen: str


def _best_match(current_tokens_h: Set[str], current_simh: int, patterns_raw: List[Dict[str, Any]]) -> Tuple[Optional[int], float]:
    # 1) SimHash-Filter → 2) Jaccard auf Hash-Tokens
    candidates = []
    for i, p in enumerate(patterns_raw):
        ph = int(p.get("simhash64", 0))
        if _hamming(current_simh, ph) <= SIMHASH_HAMMING_THRESH:
            candidates.append(i)
    if not candidates:
        candidates = list(range(len(patterns_raw)))

    best_idx = None
    best_sim = 0.0
    for i in candidates:
        p = patterns_raw[i]
        p_th = set(p.get("tokens_hashed", []))
        inter = len(p_th & current_tokens_h)
        union = len(p_th | current_tokens_h)
        sim = round(100.0 * inter / union, 1) if union else 100.0
        if sim > best_sim:
            best_sim = sim
            best_idx = i
    return best_idx, best_sim


def detect_learning_pattern(df: pd.DataFrame, learn_file: str) -> Dict[str, Any]:
    try:
        path = Path(learn_file)
        patterns_raw = _load_patterns(path)

        cols_norm = _normalize_columns(list(df.columns))
        tokens = _tokenize(cols_norm)
        tokens_h = {hashlib.sha1(t.encode("utf-8")).hexdigest()[:HASH_LEN] for t in tokens}
        simh = _simhash64(tokens_h)

        idx, sim = _best_match(tokens_h, simh, patterns_raw)
        now = datetime.now().isoformat()

        if idx is not None and sim >= MIN_LEARN_CONF:
            p = patterns_raw[idx]
            if UPDATE_HITS_ON_MATCH:
                p["hits"] = int(p.get("hits", 0)) + 1
            p["last_seen"] = now

            merged_th = list({*p.get("tokens_hashed", []), *tokens_h})
            p["tokens_hashed"] = sorted(merged_th)[:1000]
            p["simhash64"] = int(_simhash64(set(p["tokens_hashed"])))

            if not PRIVACY_HASH_COLUMNS:
                merged_cols = list({*(p.get("columns", []) or []), *cols_norm})
                p["columns"] = merged_cols[:500]

            patterns_raw[idx] = p
            _save_patterns(path, patterns_raw)
            return {
                "detected_as": p.get("detected_as", "Unbekannt"),
                "confidence": round(sim, 1),
                "source": p.get("source", "learned"),
                "pattern_id": p.get("pattern_id"),
            }

        # kein gutes Match -> neues Pattern, sofern nicht zu ähnlich
        learned_type, conf = classify_structure(df.columns, df)
        similar_exists = False
        for p in patterns_raw:
            p_th = set(p.get("tokens_hashed", []))
            inter = len(p_th & tokens_h)
            union = len(p_th | tokens_h)
            jacc = round(100.0 * inter / union, 1) if union else 100.0
            if jacc >= MIN_NEW_PATTERN_SIM * 100:
                similar_exists = True
                break

        if not similar_exists:
            new_p = Pattern(
                pattern_id=uuid.uuid4().hex,
                schema_version=PATTERN_SCHEMA_VERSION,
                detected_as=learned_type,
                tokens_hashed=sorted(list(tokens_h)),
                simhash64=int(simh),
                columns=None if PRIVACY_HASH_COLUMNS else cols_norm,
                source="learned",
                confidence=float(conf),
                hits=0,
                confirmations=0,
                created_at=now,
                last_seen=now,
            )
            patterns_raw.append(new_p.__dict__)
            _save_patterns(path, patterns_raw)
            logging.info("Neues Pattern gelernt: %s (%.1f%%)", learned_type, conf)
            return {
                "detected_as": learned_type,
                "confidence": float(conf),
                "source": "learned",
                "pattern_id": new_p.pattern_id,
            }

        return {
            "detected_as": learned_type,
            "confidence": float(min(conf, 59.9)),
            "source": "learned",
            "pattern_id": None,
        }

    except Exception as e:
        logging.exception("Fehler im Learning-System:")
        return {"detected_as": "Fehler", "confidence": 0.0, "source": "error", "pattern_id": None}


def apply_feedback_for_file(df: pd.DataFrame, learn_file: str, correct_label: str) -> str:
    path = Path(learn_file)
    patterns_raw = _load_patterns(path)

    cols_norm = _normalize_columns(list(df.columns))
    tokens = _tokenize(cols_norm)
    tokens_h = {hashlib.sha1(t.encode("utf-8")).hexdigest()[:HASH_LEN] for t in tokens}
    simh = _simhash64(tokens_h)
    now = datetime.now().isoformat()

    # Kandidaten
    candidates = []
    for i, p in enumerate(patterns_raw):
        if _hamming(simh, int(p.get("simhash64", 0))) <= SIMHASH_HAMMING_THRESH:
            candidates.append(i)
    if not candidates:
        candidates = list(range(len(patterns_raw)))

    best_idx = None
    best_sim = 0.0
    for i in candidates:
        p_th = set(patterns_raw[i].get("tokens_hashed", []))
        inter = len(p_th & tokens_h)
        union = len(p_th | tokens_h)
        sim = round(100.0 * inter / union, 1) if union else 100.0
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    if best_idx is not None and best_sim >= 40.0:
        p = patterns_raw[best_idx]
        p["detected_as"] = correct_label
        p["source"] = "feedback"
        p["confirmations"] = int(p.get("confirmations", 0)) + 1
        p["last_seen"] = now

        merged_th = list({*p.get("tokens_hashed", []), *tokens_h})
        p["tokens_hashed"] = sorted(merged_th)[:1000]
        p["simhash64"] = int(_simhash64(set(p["tokens_hashed"])))

        if not PRIVACY_HASH_COLUMNS:
            merged_cols = list({*(p.get("columns", []) or []), *cols_norm})
            p["columns"] = merged_cols[:500]

        patterns_raw[best_idx] = p
        _save_patterns(path, patterns_raw)
        return str(p.get("pattern_id", ""))

    new_p = {
        "pattern_id": uuid.uuid4().hex,
        "schema_version": PATTERN_SCHEMA_VERSION,
        "detected_as": correct_label,
        "tokens_hashed": sorted(list(tokens_h)),
        "simhash64": int(simh),
        "columns": None if PRIVACY_HASH_COLUMNS else cols_norm,
        "source": "feedback",
        "confidence": 100.0,
        "hits": 0,
        "confirmations": 1,
        "created_at": now,
        "last_seen": now,
    }
    patterns_raw.append(new_p)
    _save_patterns(path, patterns_raw)
    return str(new_p["pattern_id"])
