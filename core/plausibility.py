from __future__ import annotations

import re
import logging
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from core.helpers import parse_decimal_locale_aware  # gemeinsam genutzt

# =========================================================
# ECHOOZ PLAUSIBILITY ENGINE v1.2 – Mehr Regeln & Klarheit
# =========================================================

CONFIG = {
    "rev_per_emp_thresholds": {
        "beratung": 350_000,
        "consulting": 350_000,
        "software": 500_000,
        "it": 500_000,
        "saas": 500_000,
        "produktion": 400_000,
        "manufacturing": 400_000,
        "handel": 600_000,
        "retail": 600_000,
        "default": 500_000,
    },
    "young_company_age_years": 1,
    "young_company_high_rev": 5_000_000,
    "very_low_rpe_factor": 0.20,
    "ao_revenue_threshold": 600_000,
}

def _rm_diacritics(s: str) -> str:
    return (
        s.replace("ä", "a").replace("ö", "o").replace("ü", "u")
        .replace("Ä", "A").replace("Ö", "O").replace("Ü", "U")
        .replace("ß", "ss")
    )

def _norm(s: Any) -> str:
    return _rm_diacritics(str(s).strip().lower())

def _format_eur(x: Decimal) -> str:
    n = int(x.to_integral_value(rounding=ROUND_HALF_UP))
    return f"{n:,}".replace(",", ".") + " €"

def _parse_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        if isinstance(value, int):
            return int(value)
        s = re.sub(r"[^\d\-]", "", str(value).strip())
        return int(s) if s else 0
    except Exception:
        return 0

def _industry_threshold(industry: str) -> int:
    ind = _norm(industry)
    for key, thr in CONFIG["rev_per_emp_thresholds"].items():
        if key != "default" and key in ind:
            return thr
    return CONFIG["rev_per_emp_thresholds"]["default"]

def _legal_form_class(legal_form: str) -> str:
    lf = _norm(legal_form)
    if any(k in lf for k in ("gmbh", "ag", "se", "ug")):
        return "kapital"
    if any(k in lf for k in ("ohg", "kg", "kgaa", "gbr", "partg")):
        return "personen_hr"
    if any(k in lf for k in ("einzel", "e.k", "ek", "kaufmann")):
        return "einzel"
    return "sonstige"

def _acc_type(acc: str) -> str:
    a = _norm(acc).replace("euer", "eur")
    if "eur" in a or ("ein" in a and "ueberschuss" in a):
        return "eur"
    if any(k in a for k in ("bilanz", "doppik", "double", "gob")):
        return "bilanz"
    return a or "unbekannt"

@dataclass
class Finding:
    rule_id: str
    severity: str  # 'info' | 'warning' | 'critical' | 'ok' | 'error'
    message: str
    indicators: Optional[Dict[str, Any]] = None
    citations: Optional[List[str]] = None
    context: Optional[str] = None

def analyze_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Finding] = []

    try:
        rev = parse_decimal_locale_aware(data.get("revenue"))
        emp = _parse_int(data.get("employees"))
        industry = str(data.get("industry") or "")
        legal = str(data.get("legal_form") or "")
        acc = str(data.get("accounting_type") or "")
        age = _parse_int(data.get("company_age"))

        if rev < 0:
            findings.append(Finding("R-000", "warning", "Negativer Umsatz angegeben – bitte prüfen.",
                                    indicators={"revenue": _format_eur(rev)}))
        if emp < 0:
            findings.append(Finding("R-001", "critical", "Negative Mitarbeiterzahl ist unzulässig.",
                                    indicators={"employees": emp}))

        if emp > 0 and rev > 0:
            thr = Decimal(_industry_threshold(industry))
            rpe = (rev / Decimal(emp))
            if rpe > thr:
                sev = "warning" if rpe <= thr * Decimal("2") else "critical"
                findings.append(Finding(
                    "R-010", sev,
                    f"Umsatz pro Mitarbeiter mit {_format_eur(rpe)} über Branchenwert "
                    f"({_format_eur(thr)}).",
                    indicators={"revenue": _format_eur(rev), "employees": emp, "threshold": _format_eur(thr)},
                    context=industry or None
                ))
            very_low = thr * Decimal(str(CONFIG["very_low_rpe_factor"]))
            if rpe < very_low:
                findings.append(Finding(
                    "R-013", "warning",
                    "Sehr niedriger Umsatz pro Mitarbeiter – Produktivität/Periodenabgrenzung prüfen.",
                    indicators={"rpe": _format_eur(rpe), "threshold_low": _format_eur(very_low)}
                ))

        if emp == 0 and rev > 0:
            findings.append(Finding(
                "R-011", "info",
                "Umsatz bei 0 Mitarbeitern – plausibel nur bei starker Automatisierung/Freelancern.",
                indicators={"revenue": _format_eur(rev), "employees": emp},
                context=industry or None
            ))

        if emp >= 5 and rev == 0:
            findings.append(Finding(
                "R-012", "warning",
                "Keine Umsätze trotz mehrerer Beschäftigter – bitte Zeitraum/Definition prüfen.",
                indicators={"employees": emp}
            ))

        lf_class = _legal_form_class(legal)
        acc_class = _acc_type(acc)
        if acc_class == "eur":
            if lf_class == "kapital":
                findings.append(Finding(
                    "R-020", "critical",
                    "EÜR bei Kapitalgesellschaften ist unzulässig.",
                    citations=["§ 4 Abs. 3 EStG", "§§ 238, 242 HGB"],
                    indicators={"legal_form": legal, "accounting_type": acc}
                ))
            elif lf_class == "personen_hr":
                findings.append(Finding(
                    "R-021", "warning",
                    "EÜR bei eingetragenen Personengesellschaften regelmäßig unzulässig (Buchführungspflicht).",
                    citations=["§§ 238 ff. HGB", "§ 141 AO"],
                    indicators={"legal_form": legal, "accounting_type": acc}
                ))
            if rev >= Decimal(CONFIG["ao_revenue_threshold"]):
                findings.append(Finding(
                    "R-022", "warning",
                    "EÜR bei Umsätzen ≥ 600.000 € regelmäßig unzulässig (Buchführungspflicht nach §141 AO).",
                    citations=["§ 141 AO"],
                    indicators={"revenue": _format_eur(rev)}
                ))

        if age <= CONFIG["young_company_age_years"] and rev >= Decimal(CONFIG["young_company_high_rev"]):
            findings.append(Finding(
                "R-030", "info",
                "Sehr hoher Umsatz im Gründungsjahr – bitte Geschäftsmodell und Einmaleffekte prüfen.",
                indicators={"company_age": age, "revenue": _format_eur(rev)}
            ))

        if not findings:
            findings.append(Finding("R-OK", "ok", "Keine Auffälligkeiten erkannt. Angaben plausibel."))

    except Exception as e:
        logging.exception("Fehler in der Plausibilitätsprüfung:")
        findings = [Finding("R-ERR", "error", f"Plausibilitätsprüfung fehlgeschlagen: {e}")]

    return [asdict(f) for f in findings]
