# core/skr_mapper.py
from __future__ import annotations

import json, re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# -------------------------
# Datenstrukturen
# -------------------------
@dataclass
class GroupDef:
    key: str
    label: str
    normal_side: str  # "debit" | "credit"

@dataclass
class RuleDef:
    target: str                 # Gruppe, z.B. "ASSETS_CURRENT"
    type: str                   # "exact" | "prefix" | "range" | "regex" | "list"
    pattern: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    values: Optional[List[str]] = None
    priority: int = 0           # höhere Zahl = höhere Priorität

@dataclass
class CoaMapping:
    id: str
    name: str
    version: str
    groups: Dict[str, GroupDef]
    rules: List[RuleDef]

@dataclass
class CoaPack:
    version: str
    mappings: Dict[str, CoaMapping]


# -------------------------
# Hilfen: Labels & Normal-Seite ableiten
# -------------------------
_LABELS = {
    "ASSETS_NONCURRENT": "Anlagevermögen",
    "ASSETS_CURRENT": "Umlaufvermögen",
    "EQUITY": "Eigenkapital",
    "LIABILITIES_SHORT_TERM": "Kurzfr. Verbindlichkeiten",
    "LIABILITIES_LONG_TERM": "Langfr. Verbindlichkeiten",
    "LIABILITIES_NONCURRENT": "Langfr. Verbindlichkeiten",
    "P&L_REVENUE": "Umsatzerlöse",
    "P&L_OTHER_INCOME": "Sonstige Erträge",
    "P&L_EXPENSE": "Aufwendungen",
    "P&L_PERSONNEL": "Personalaufwand",
    "P&L_DEPRECIATION": "Abschreibungen",
}

def _default_normal_side(group_key: str) -> str:
    g = group_key.upper()
    if g.startswith("ASSETS") or g.startswith("P&L_EXPENSE") or g.startswith("P&L_PERSONNEL") or g.startswith("P&L_DEPRECIATION"):
        return "debit"
    if g.startswith("LIABILITIES") or g == "EQUITY" or g.startswith("P&L_REVENUE") or g.startswith("P&L_OTHER_INCOME"):
        return "credit"
    # Fallback defensiv
    return "debit"


# -------------------------
# Laden & Normalisieren des COA-Packs
# -------------------------
def _synthesize_groups_from_rules(rules: List[RuleDef]) -> Dict[str, GroupDef]:
    unique_targets = sorted({r.target for r in rules})
    groups: Dict[str, GroupDef] = {}
    for t in unique_targets:
        groups[t] = GroupDef(
            key=t,
            label=_LABELS.get(t, t),
            normal_side=_default_normal_side(t),
        )
    return groups

def _parse_rules(rules_raw: List[Dict[str, Any]]) -> List[RuleDef]:
    rules: List[RuleDef] = []
    for r in rules_raw:
        rules.append(
            RuleDef(
                target=r["target"],
                type=r.get("type", "regex"),
                pattern=r.get("pattern"),
                start=r.get("start"),
                end=r.get("end"),
                values=r.get("values"),
                priority=int(r.get("priority", 0)),
            )
        )
    # Höhere Priorität zuerst
    rules.sort(key=lambda x: x.priority, reverse=True)
    return rules

def _coapack_from_simple(data: Dict[str, Any]) -> CoaPack:
    """
    Erwartet Format:
    {
      "version": "...",
      "SKR03": {"rules": [...]},
      "SKR04": {"rules": [...]},
      "NETTI": {"rules": [...]}
    }
    """
    version = str(data.get("version", "unknown"))
    mappings: Dict[str, CoaMapping] = {}

    for mid, content in data.items():
        if mid == "version":
            continue
        if not isinstance(content, dict) or "rules" not in content:
            continue
        rules = _parse_rules(content.get("rules", []))
        groups = content.get("groups")
        if groups and isinstance(groups, dict):
            grp_objs = {
                k: GroupDef(
                    key=k,
                    label=(v.get("label") if isinstance(v, dict) else str(v)) or _LABELS.get(k, k),
                    normal_side=(v.get("normal_side") if isinstance(v, dict) else None) or _default_normal_side(k),
                )
                for k, v in groups.items()
            }
        else:
            grp_objs = _synthesize_groups_from_rules(rules)

        mappings[mid.upper()] = CoaMapping(
            id=mid.upper(),
            name=mid.upper(),
            version=version,
            groups=grp_objs,
            rules=rules,
        )
    if not mappings:
        raise ValueError("COA-Pack (einfach) enthält keine Mappings.")
    return CoaPack(version=version, mappings=mappings)

def _coapack_from_extended(data: Dict[str, Any]) -> CoaPack:
    """
    Erwartet Format:
    {
      "version": "...",
      "mappings": {
        "SKR03": {
          "version":"..",
          "name":"..",
          "groups": {...},
          "rules":[...]
        }, ...
      }
    }
    """
    version = str(data.get("version", "unknown"))
    mappings_raw: Dict[str, Any] = data.get("mappings", {})
    mappings: Dict[str, CoaMapping] = {}

    for mid, m in mappings_raw.items():
        rules = _parse_rules(m.get("rules", []))
        groups_raw = m.get("groups")
        if groups_raw and isinstance(groups_raw, dict):
            groups = {
                k: GroupDef(
                    key=k,
                    label=(v.get("label") if isinstance(v, dict) else str(v)) or _LABELS.get(k, k),
                    normal_side=(v.get("normal_side") if isinstance(v, dict) else None) or _default_normal_side(k),
                )
                for k, v in groups_raw.items()
            }
        else:
            groups = _synthesize_groups_from_rules(rules)

        mappings[mid.upper()] = CoaMapping(
            id=mid.upper(),
            name=m.get("name", mid).upper(),
            version=m.get("version", version),
            groups=groups,
            rules=rules,
        )

    if not mappings:
        raise ValueError("COA-Pack (erweitert) enthält keine Mappings.")
    return CoaPack(version=version, mappings=mappings)

def _parse_coa_data(data: Dict[str, Any]) -> CoaPack:
    if "mappings" in data:
        return _coapack_from_extended(data)
    return _coapack_from_simple(data)

def _default_pack_path() -> Path:
    return (Path(__file__).resolve().parent / "data" / "coa_pack.json")

@lru_cache(maxsize=1)
def load_coa_pack_default() -> CoaPack:
    p = _default_pack_path()
    raw = p.read_text(encoding="utf-8-sig")  # BOM-sicher
    data = json.loads(raw)
    return _parse_coa_data(data)

def load_coa_pack(path: Path | str | None) -> CoaPack:
    """
    Beibehält die bisherige Signatur. Nutzt bei None/Nichtauffindbar den Default-Pfad.
    """
    if path:
        p = Path(path)
        if p.exists():
            raw = p.read_text(encoding="utf-8-sig")
            data = json.loads(raw)
            return _parse_coa_data(data)
    # Fallback
    return load_coa_pack_default()


# -------------------------
# Matching / Klassifikation
# -------------------------
def _rule_match(acc: str, r: RuleDef) -> bool:
    s = str(acc or "").strip()
    if not s:
        return False
    t = r.type.lower()
    if t == "exact":
        return s == (r.pattern or "")
    if t == "prefix":
        return s.startswith(r.pattern or "")
    if t == "range":
        try:
            a = int(s)
            start = int(r.start or "0")
            end = int(r.end or "0")
            return start <= a <= end
        except Exception:
            return False
    if t == "list":
        return s in set(r.values or [])
    # default: regex
    pat = r.pattern or ""
    try:
        return re.match(pat, s) is not None
    except re.error:
        return False

def classify_account(acc: str, mapping: CoaMapping) -> Optional[str]:
    for r in mapping.rules:
        if _rule_match(acc, r):
            return r.target
    return None


# -------------------------
# Zahlen robust parsen
# -------------------------
def _safe_float(x: Any) -> float:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return 0.0
        s = str(x).replace("\u00a0", "").replace(" ", "")
        # Deutsche Formate "1.234,56"
        if "," in s and s.count(",") == 1 and "." in s and s.rfind(".") < s.rfind(","):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", ".")
        return float(s)
    except Exception:
        return 0.0

def compute_amount_balance(row: pd.Series) -> float:
    """
    Kanonisches Saldo-Feld bevorzugen; sonst OB + SollPeriode - HabenPeriode.
    """
    if "balance" in row and row["balance"] not in (None, ""):
        return _safe_float(row["balance"])
    ob = _safe_float(row.get("opening_balance", 0.0))
    dp = _safe_float(row.get("debit_period", 0.0))
    cp = _safe_float(row.get("credit_period", 0.0))
    return ob + dp - cp


# -------------------------
# Coverage / Wahl des COA
# -------------------------
def score_mapping(canon_df: pd.DataFrame, mapping: CoaMapping, sample: int = 500) -> Tuple[int, int]:
    """
    Liefert (#klassifiziert, #mit Konto) für Coverage-Bewertung.
    """
    df = canon_df
    if "account" not in df.columns or df.empty:
        return (0, 0)
    n = min(len(df), sample)
    accounts = df["account"].astype(str).head(n).tolist()
    total = sum(1 for a in accounts if a and str(a).strip())
    hits = 0
    for a in accounts:
        if not a or not str(a).strip():
            continue
        if classify_account(a, mapping) is not None:
            hits += 1
    return (hits, total)

def choose_coa(canon_df: pd.DataFrame, pack: CoaPack, prefer_id: Optional[str] = None) -> CoaMapping:
    if prefer_id:
        k = prefer_id.strip().upper()
        if k == "AUTO":
            k = None
        if k and k in pack.mappings:
            return pack.mappings[k]

    best_id, best_cov = None, -1.0
    for mid, m in pack.mappings.items():
        hits, total = score_mapping(canon_df, m)
        cov = (hits / total) if total > 0 else 0.0
        if cov > best_cov:
            best_id, best_cov = mid, cov
    # minimaler Schwellwert 0.15, sonst nimm erstes Mapping
    if best_id and best_cov >= 0.15:
        return pack.mappings[best_id]
    return next(iter(pack.mappings.values()))


# -------------------------
# Kanonisches DF bauen
# -------------------------
def build_canonical_df(source_df: pd.DataFrame, canonical_columns: Dict[str, Optional[str]]) -> pd.DataFrame:
    """
    Baut aus dem Quelldatenframe ein DF mit Spalten:
    account, label, opening_balance, balance, debit_period, credit_period, currency
    (nur die, die verfügbar sind).
    """
    rename_map = {src: canon for canon, src in canonical_columns.items() if src}
    df = source_df.rename(columns=rename_map).copy()
    keep = [c for c in ["account", "label", "opening_balance", "balance", "debit_period", "credit_period", "currency"] if c in df.columns]
    if not keep:
        return pd.DataFrame(columns=["account", "label", "balance"])
    return df[keep].reset_index(drop=True)


# -------------------------
# Statements (Bilanz & GuV)
# -------------------------
def build_statements(canon_df: pd.DataFrame, mapping: CoaMapping, tolerance: float = 0.01) -> Dict[str, Any]:
    df = canon_df.copy()
    if df.empty or "account" not in df.columns:
        return {
            "coa_id": mapping.id,
            "balance_sheet": {
                "assets": [], "equity_liabilities": [],
                "total_assets": 0.0, "total_equity_liabilities": 0.0,
                "balanced": True, "difference": 0.0, "tolerance": round(tolerance, 2)
            },
            "pnl": {"total_debit_period": 0.0, "total_credit_period": 0.0, "result": 0.0},
            "unmapped_accounts": [], "coverage": 0.0
        }

    # Klassifikation
    groups: List[Optional[str]] = []
    for acc in df["account"].astype(str).tolist():
        groups.append(classify_account(acc, mapping))
    df["group"] = groups

    # Betrag für Bilanz
    df["amount_balance"] = df.apply(compute_amount_balance, axis=1)

    # Coverage
    mapped = df["group"].notna().sum()
    total_accounts = (df["account"].astype(str).str.len() > 0).sum()
    coverage = round((mapped / total_accounts) * 100.0, 1) if total_accounts else 0.0

    # Summen je Gruppe, vorzeichenrichtig
    sums: Dict[str, float] = {gkey: 0.0 for gkey in mapping.groups.keys()}
    for _, row in df.dropna(subset=["group"]).iterrows():
        gk = row["group"]
        if gk not in mapping.groups:
            continue
        amt = float(row["amount_balance"])
        if mapping.groups[gk].normal_side == "credit":
            amt = -amt
        sums[gk] += amt

    def group_rows(keys: List[str]) -> List[Dict[str, Any]]:
        out = []
        for k in keys:
            if k not in mapping.groups:
                continue
            out.append({
                "key": k,
                "label": mapping.groups[k].label,
                "amount": round(sums.get(k, 0.0), 2)
            })
        return out

    asset_keys       = [k for k in mapping.groups if k.startswith("ASSETS_")]
    equity_liab_keys = [k for k in mapping.groups if k.startswith("EQUITY") or k.startswith("LIABILITIES")]
    pnl_rev_keys     = [k for k in mapping.groups if k.startswith("P&L_REVENUE") or k.startswith("P&L_OTHER_INCOME")]
    pnl_exp_keys     = [k for k in mapping.groups if k.startswith("P&L_EXPENSE") or k.startswith("P&L_PERSONNEL") or k.startswith("P&L_DEPRECIATION")]

    total_assets          = round(sum(sums.get(k, 0.0) for k in asset_keys), 2)
    total_equity_liab_pos = round(sum(sums.get(k, 0.0) for k in equity_liab_keys), 2)
    difference            = round(total_assets - total_equity_liab_pos, 2)
    balanced              = abs(difference) <= round(tolerance, 2)

    # GuV gesamt (nur wenn Periodenspalten vorhanden)
    total_dp = round(float(df.get("debit_period", pd.Series(dtype=float)).map(_safe_float).sum()) if "debit_period" in df.columns else 0.0, 2)
    total_cp = round(float(df.get("credit_period", pd.Series(dtype=float)).map(_safe_float).sum()) if "credit_period" in df.columns else 0.0, 2)
    pnl_result = round(total_cp - total_dp, 2)

    unmapped = df[df["group"].isna()].get("account", pd.Series(dtype=object)).dropna().astype(str).tolist()

    return {
        "coa_id": mapping.id,
        "coverage": coverage,
        "balance_sheet": {
            "assets": group_rows(asset_keys),
            "equity_liabilities": group_rows(equity_liab_keys),
            "total_assets": total_assets,
            "total_equity_liabilities": total_equity_liab_pos,
            "balanced": balanced,
            "difference": difference,
            "tolerance": round(tolerance, 2),
        },
        "pnl": {
            "revenue_groups": group_rows(pnl_rev_keys),
            "expense_groups": group_rows(pnl_exp_keys),
            "total_debit_period": total_dp,
            "total_credit_period": total_cp,
            "result": pnl_result,
        },
        "unmapped_accounts": unmapped,
    }


# -------------------------
# Komfort-API für Aggregator
# -------------------------
def map_and_aggregate(
    source_df: pd.DataFrame,
    canonical_columns: Dict[str, Optional[str]],
    coa_preference: Optional[str] = None,
    coa_pack_path: Optional[str | Path] = None,
    tolerance: float = 0.01,
) -> Dict[str, Any]:
    """
    Baut das kanonische DF, wählt Mapping (Präferenz/Auto), liefert Statements.
    """
    canon = build_canonical_df(source_df, canonical_columns)
    pack = load_coa_pack(coa_pack_path)
    mapping = choose_coa(canon, pack, prefer_id=(coa_preference or "").upper())
    statements = build_statements(canon, mapping, tolerance=tolerance)
    return {
        "coa": mapping.id,
        "pack_version": pack.version,
        "coverage": statements["coverage"],
        "statements": statements,
    }
