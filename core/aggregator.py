# core/aggregator.py
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, List

import pandas as pd

try:
    from .helpers import detect_coa_type  # type: ignore
except Exception:
    def detect_coa_type(_: Dict[str, Any]) -> str:
        # Fallback: liest gewünschtes Default-COA aus ENV, sonst SKR03
        return os.getenv("ECHOoz_DEFAULT_COA", "SKR03").upper()

# --------------------------------------------------
#  Wertzuordnung in Ergebnisstruktur
# --------------------------------------------------
def _aggregate_value(result: Dict[str, Any], target: Optional[str], saldo: float) -> None:
    if not target:
        return
    bs = result["statements"]["balance_sheet"]
    pl = result["statements"]["profit_and_loss"]

    if target == "ASSETS_NONCURRENT":
        bs["assets"]["non_current"]["sum"] = bs["assets"]["non_current"].get("sum", 0.0) + saldo
    elif target == "ASSETS_CURRENT":
        bs["assets"]["current"]["sum"] = bs["assets"]["current"].get("sum", 0.0) + saldo
    elif target == "EQUITY":
        bs["liabilities_equity"]["equity"] += saldo
    elif target in ("LIABILITIES_NONCURRENT", "LIABILITIES_LONG_TERM"):
        bs["liabilities_equity"]["liabilities_long_term"] += saldo
    elif target == "LIABILITIES_CURRENT":
        bs["liabilities_equity"]["liabilities_short_term"] += saldo
    elif target == "PROVISIONS":
        bs["liabilities_equity"]["provisions"] += saldo
    elif target == "P&L_REVENUE":
        pl["revenue"] += abs(saldo)
    elif target == "P&L_EXPENSE":
        pl["other_operating_expenses"] += abs(saldo)
    elif target == "P&L_PERSONNEL":
        pl["personnel_expenses"] += abs(saldo)
    elif target == "P&L_DEPRECIATION":
        pl["depreciation"] += abs(saldo)
    elif target == "P&L_INTEREST":
        pl["interest_expenses"] += abs(saldo)
    elif target == "P&L_OTHER_INCOME":
        pl["other_operating_income"] += abs(saldo)

# ============================================================
#  Safe Float Parsing & Balance-Ermittlung
# ============================================================
def _to_float_safe(x: Any) -> float:
    try:
        if x is None:
            return 0.0
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return 0.0
            return float(x)
        s = str(x).strip().replace("\u00a0", "").replace(" ", "")
        # 1x Komma + >1 Punkt => Tausenderpunkte entfernen (deutsches Format)
        if s.count(",") == 1 and s.count(".") > 1:
            s = s.replace(".", "")
        s = s.replace(",", ".")
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        return float(s) if s else 0.0
    except Exception:
        return 0.0

def _resolve_balance_row(row: pd.Series, cc: Dict[str, Optional[str]]) -> float:
    col = lambda name: cc.get(name) if cc else None
    get = lambda colname: _to_float_safe(row[colname]) if (colname and colname in row and pd.notna(row[colname])) else 0.0

    # 1) Explizite „Saldo +/-“-Spalte schlägt alles
    if "Saldo +/-" in row.index:
        val = _to_float_safe(row["Saldo +/-"])
        if val != 0.0:
            return val

    # 2) Balancespalte + evtl. S/H-Markierung
    bal = get(col("balance"))
    if bal != 0.0:
        s_val = str(row.get("S") or row.get("s") or "").strip().upper()
        h_val = str(row.get("H") or row.get("h") or "").strip().upper()
        if s_val == "S":
            return abs(bal)
        elif h_val == "H":
            return -abs(bal)
        return bal

    # 3) Periodische Bewegung
    debit = get(col("debit_period"))
    credit = get(col("credit_period"))
    if debit != 0.0 or credit != 0.0:
        return debit - credit

    # 4) Fallback inkl. EB
    ob = get(col("opening_balance"))
    if (ob != 0.0) or (debit != 0.0) or (credit != 0.0):
        return ob + (debit - credit)

    return 0.0

# ============================================================
#  Susa-Payload aufbauen (inkl. Label im Preview)
# ============================================================
def build_susa_payload_from_df(
    df: pd.DataFrame,
    canonical_columns: Dict[str, Optional[str]],
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta = meta or {}
    acc_col = canonical_columns.get("account")
    label_col = canonical_columns.get("label")

    if "Saldo +/-" in df.columns:
        canonical_columns["balance"] = "Saldo +/-"
        meta.setdefault("notes", [])
        meta["notes"].append("Saldo +/- erkannt – wird für Balance-Check verwendet.")

    preview: List[Dict[str, Any]] = []
    columns_used: List[str] = []

    for k in ["account", "label", "balance", "opening_balance", "debit_period", "credit_period", "debit_cum", "credit_cum"]:
        c = canonical_columns.get(k)
        if c:
            columns_used.append(f"{k}:{c}")

    if acc_col and acc_col in df.columns:
        for _, row in df.iterrows():
            konto_raw = row.get(acc_col)
            if pd.isna(konto_raw):
                continue
            konto = str(konto_raw).strip()
            if not konto:
                continue
            saldo = _resolve_balance_row(row, canonical_columns)
            item = {"Konto": konto, "Saldo": saldo}
            if label_col and label_col in df.columns:
                try:
                    lbl = row.get(label_col)
                    item["Label"] = ("" if pd.isna(lbl) else str(lbl)).strip()
                except Exception:
                    item["Label"] = ""
            preview.append(item)

    return {
        "source": meta.get("source", "echooz_parser"),
        "file_sha256": meta.get("file_sha256"),
        "columns_used": columns_used,
        "preview": preview,
        "notes": meta.get("notes", []),
    }

# ============================================================
#  Aggregation: SuSa → HGB (inkl. label_regex, unmapped_list, coverage)
# ============================================================
def aggregate_susa_to_hgb(
    susa_data: Dict[str, Any],
    coa_pack_path: str = "core/data/coa_pack.json",
    *,
    prefer_coa: Optional[str] = None,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
) -> Dict[str, Any]:
    with open(coa_pack_path, "r", encoding="utf-8") as f:
        coa_pack = json.load(f)

    detected = (detect_coa_type(susa_data) or "").upper()
    coa_type = (prefer_coa or detected or "CUSTOM").upper()
    if coa_type not in (coa_pack.get("mappings") or {}):
        coa_type = "CUSTOM"

    if not period_start or not period_end:
        year = datetime.now().year
        period_start = period_start or f"{year}-01-01"
        period_end = period_end or f"{year}-12-31"
    try:
        year = int((period_start or "")[:4])
    except Exception:
        year = None

    result: Dict[str, Any] = {
        "period": {"year": year, "start_date": period_start, "end_date": period_end},
        "statements": {
            "balance_sheet": {
                "assets": {"non_current": {}, "current": {}, "total_assets": 0.0},
                "liabilities_equity": {
                    "equity": 0.0,
                    "provisions": 0.0,
                    "liabilities_short_term": 0.0,
                    "liabilities_long_term": 0.0,
                    "total_liabilities_equity": 0.0,
                },
                "check": {"difference": 0.0, "tolerance": 0.01, "balanced": True},
            },
            "profit_and_loss": {
                "revenue": 0.0,
                "material_expenses": 0.0,
                "personnel_expenses": 0.0,
                "other_operating_expenses": 0.0,
                "other_operating_income": 0.0,
                "depreciation": 0.0,
                "interest_expenses": 0.0,
                "result": 0.0,
            },
        },
        "meta": {
            "coa": coa_type,
            "source": susa_data.get("source", "unknown"),
            "file_sha256": susa_data.get("file_sha256", "unknown"),
            "columns_used": susa_data.get("columns_used", []),
            "mapped_accounts": 0,
            "unmapped_accounts": 0,
            "unmapped_accounts_list": [],
            "notes": [],
            "created_at": datetime.utcnow().isoformat() + "Z",
            "rows_processed": int(len(susa_data.get("preview", []))),
        },
    }

    rules = (coa_pack.get("mappings", {}).get(coa_type, {}) or {}).get("rules", [])
    unmapped_list: List[Dict[str, Any]] = []
    preview = susa_data.get("preview", [])

    # Coverage-Akkus
    abs_sum_all = 0.0
    abs_sum_mapped = 0.0

    for entry in preview:
        account_raw = str(entry.get("Konto", "")).strip()
        saldo = float(entry.get("Saldo", 0.0) or 0.0)
        label_text = str(entry.get("Label", "") or "").strip()
        mapped = False

        # Für Coverage immer den absoluten Betrag mitzählen
        abs_sum_all += abs(saldo)

        # Accountnummer ggf. extrahieren (z. B. "0027000" -> 27000)
        acct_num = None
        try:
            m = re.search(r"\d+", account_raw)
            acct_num = int(m.group(0)) if m else None
        except Exception:
            acct_num = None

        for rule in rules:
            target = rule.get("target")
            rtype = rule.get("type")

            if rtype == "regex":
                pattern = rule.get("pattern", "")
                if pattern and re.match(pattern, account_raw):
                    _aggregate_value(result, target, saldo)
                    result["meta"]["mapped_accounts"] += 1
                    mapped = True
                    break

            elif rtype == "range" and acct_num is not None:
                try:
                    start = int(rule.get("start"))
                    end = int(rule.get("end"))
                except Exception:
                    continue
                if start <= acct_num <= end:
                    _aggregate_value(result, target, saldo)
                    result["meta"]["mapped_accounts"] += 1
                    mapped = True
                    break

            elif rtype == "label_regex" and label_text:
                pattern = rule.get("pattern", "")
                try:
                    if pattern and re.search(pattern, label_text, flags=re.IGNORECASE):
                        _aggregate_value(result, target, saldo)
                        result["meta"]["mapped_accounts"] += 1
                        mapped = True
                        break
                except re.error:
                    # Ungültige Regex aus CoA-Pack wird ignoriert
                    pass

        if mapped:
            abs_sum_mapped += abs(saldo)
        else:
            result["meta"]["unmapped_accounts"] += 1
            if len(unmapped_list) < 200:
                unmapped_list.append({"account": account_raw, "label": label_text, "saldo": round(saldo, 2)})

    result["meta"]["unmapped_accounts_list"] = unmapped_list

    # Coverage schreiben
    mapped_accounts = int(result["meta"]["mapped_accounts"])
    unmapped_accounts = int(result["meta"]["unmapped_accounts"])
    accounts_total = mapped_accounts + unmapped_accounts if (mapped_accounts + unmapped_accounts) > 0 else int(result["meta"]["rows_processed"])
    cov_acc = round(100.0 * mapped_accounts / accounts_total, 2) if accounts_total else 0.0
    cov_bal = round(100.0 * abs_sum_mapped / abs_sum_all, 2) if abs_sum_all > 0.0 else cov_acc

    result["meta"]["coverage_accounts_total"] = accounts_total
    result["meta"]["coverage_accounts_pct"] = cov_acc
    result["meta"]["coverage_balance_pct"] = cov_bal

    # Abschlusskennzahlen berechnen
    bs = result["statements"]["balance_sheet"]
    pl = result["statements"]["profit_and_loss"]

    bs["assets"]["total_assets"] = float(sum((bs["assets"]["non_current"] or {}).values()) + sum((bs["assets"]["current"] or {}).values()))
    bs["liabilities_equity"]["total_liabilities_equity"] = float(
        bs["liabilities_equity"]["equity"]
        + bs["liabilities_equity"]["provisions"]
        + bs["liabilities_equity"]["liabilities_short_term"]
        + bs["liabilities_equity"]["liabilities_long_term"]
    )
    bs["check"]["difference"] = round(bs["assets"]["total_assets"] - bs["liabilities_equity"]["total_liabilities_equity"], 2)
    bs["check"]["balanced"] = abs(bs["check"]["difference"]) <= float(bs["check"]["tolerance"])

    pl["result"] = round(
        pl["revenue"]
        - (
            pl["material_expenses"]
            + pl["personnel_expenses"]
            + pl["other_operating_expenses"]
            + pl["depreciation"]
            + pl["interest_expenses"]
        )
        + pl["other_operating_income"],
        2,
    )

    if coa_type == "CUSTOM":
        result["meta"]["notes"].append(
            "Unbekannter/unspezifischer Kontenrahmen – CUSTOM-Fallback (ENV ECHOoz_DEFAULT_COA oder Header X-Chart-Of-Accounts=skr03|skr04|netti|auto setzen)."
        )

    return result

# ============================================================
#  Hauptfunktion (inkl. Safe-Mode, Perioden-Override, Coverage bleibt)
# ============================================================
def aggregate_from_dataframe(
    df: pd.DataFrame,
    canonical_columns: Dict[str, Optional[str]],
    coa_pack_path: str = "core/data/coa_pack.json",
    file_sha256: Optional[str] = None,
    force_balance: bool = True,
    prefer_coa: Optional[str] = None,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
) -> Dict[str, Any]:
    susa_payload = build_susa_payload_from_df(
        df=df,
        canonical_columns=canonical_columns,
        meta={"file_sha256": file_sha256, "source": "echooz_parser", "prefer_coa": prefer_coa},
    )

    result = aggregate_susa_to_hgb(
        susa_payload,
        coa_pack_path=coa_pack_path,
        prefer_coa=prefer_coa,
        period_start=period_start,
        period_end=period_end,
    )

    # Perioden/Year ggf. überschreiben
    try:
        if period_start or period_end:
            year_guess = None
            if period_end and len(str(period_end)) >= 4:
                year_guess = int(str(period_end)[:4])
            elif period_start and len(str(period_start)) >= 4:
                year_guess = int(str(period_start)[:4])
            if year_guess:
                result["period"]["year"] = year_guess
            if period_start:
                result["period"]["start_date"] = str(period_start)
            if period_end:
                result["period"]["end_date"] = str(period_end)
    except Exception:
        pass

    # Safe-Mode: Differenz in Eigenkapital ausgleichen (nur wenn gewünscht)
    if force_balance:
        bs = result["statements"]["balance_sheet"]
        diff = float(bs["check"]["difference"])
        tol = float(bs["check"]["tolerance"])
        if abs(diff) > tol:
            bs["liabilities_equity"]["equity"] += diff
            bs["assets"]["total_assets"] = float(
                sum((bs["assets"]["non_current"] or {}).values()) + sum((bs["assets"]["current"] or {}).values())
            )
            bs["liabilities_equity"]["total_liabilities_equity"] = float(
                bs["liabilities_equity"]["equity"]
                + bs["liabilities_equity"]["provisions"]
                + bs["liabilities_equity"]["liabilities_short_term"]
                + bs["liabilities_equity"]["liabilities_long_term"]
            )
            bs["check"]["difference"] = round(
                bs["assets"]["total_assets"] - bs["liabilities_equity"]["total_liabilities_equity"], 2
            )
            bs["check"]["balanced"] = True
            result["meta"].setdefault("notes", []).append("Safe-Mode: Equity-Adjustment angewendet.")
            result["meta"]["forced_balance"] = True
        else:
            result["meta"]["forced_balance"] = False

    if prefer_coa:
        result["meta"]["notes"].append(f"COA preference requested: {prefer_coa}")

    if result.get("meta", {}).get("forced_balance") is True:
        notes = result["meta"].get("notes", [])
        if isinstance(notes, str):
            notes = [notes]
        notes.append("SuSa periodisch unausgeglichen, Safe-Mode aktiv.")
        result["meta"]["notes"] = notes

    return result

def aggregate_to_hgb(
    df: pd.DataFrame,
    canonical_columns: Dict[str, Optional[str]],
    coa_pack_path: str = "core/data/coa_pack.json",
    **kwargs: Any,
) -> Dict[str, Any]:
    return aggregate_from_dataframe(df, canonical_columns, coa_pack_path=coa_pack_path, **kwargs)
