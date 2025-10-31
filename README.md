# 0) Umgebung
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Server starten (eine Variante)
uvicorn main:app --reload --port 8000
# oder:
# python3 main.py

# 2) Pfad zur Datei anpassen (Finder → Rechtsklick → "Copy as Pathname")
export FILE_PATH="/ABSOLUTER/PFAD/zu/test_1.xlsx"

# 3) Upload → FILE_ID erhalten
export FILE_ID=$(
  curl -s -X POST "http://127.0.0.1:8000/upload" -F "file=@${FILE_PATH}" \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); o=d[0] if isinstance(d,list) else d; print(o.get("file_sha256") or o.get("id") or o.get("stored_as") or "")'
)
echo "FILE_ID=${FILE_ID}"

# 4) Aggregation (SKR04) + Kurzreport
curl -s -X POST "http://127.0.0.1:8000/rebuild/${FILE_ID}" \
  -H "Content-Type: application/json" \
  --data-binary '{"prefer_coa":"skr04","period_start":"2024-01-01","period_end":"2024-12-31","aggregate": true}' \
| python3 - <<'PY'
import sys,json
r=json.load(sys.stdin)[0]; a=r["aggregation"]; m=a["meta"]; b=a["statements"]["balance_sheet"]["check"]; p=a["statements"]["profit_and_loss"]
tot=m["mapped_accounts"]+m["unmapped_accounts"]; cov=(m["mapped_accounts"]/tot*100) if tot else 0.0
print({
  "coa": m["coa"],
  "mapped": m["mapped_accounts"],
  "unmapped": m["unmapped_accounts"],
  "coverage": round(cov,2),
  "coverage_bal": m.get("coverage_balance_pct"),
  "balanced": b["balanced"],
  "revenue": round(p["revenue"],2),
  "result": round(p["result"],2)
})
PY
# Ergebnis der letzten Aggregation einmal in eine Datei schreiben:
curl -s -X POST "http://127.0.0.1:8000/rebuild/${FILE_ID}" \
  -H "Content-Type: application/json" \
  --data-binary '{"prefer_coa":"skr04","period_start":"2024-01-01","period_end":"2024-12-31","aggregate": true}' \
  > "/tmp/rebuild_${FILE_ID}_skr04.json"

# Top-20 Unmapped ausgeben:
python3 - <<'PY'
import os, json, itertools as it
fid=os.environ["FILE_ID"]
r=json.load(open(f"/tmp/rebuild_{fid}_skr04.json"))[0]
L=r["aggregation"]["meta"]["unmapped_accounts_list"]
for x in it.islice(L, 0, 20):
    print(f"{x.get('account','?')}\t{x.get('label','')}\t{x.get('saldo',0)}")
print(f"\nUNMAPPED_LEFT = {len(L)}")
PY

