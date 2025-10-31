# echooz_parser

Parser & Aggregator für SuSa/BWA → HGB-Struktur inkl. Auto-Mapping (SKR03/SKR04/NETTI), Safe-Mode-Balance, Coverage-Metriken und Liste der ungemappten Konten.

## Quickstart (lokal)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Server starten (eine der beiden Varianten):
uvicorn main:app --reload --port 8000
# oder:
# python3 main.py
