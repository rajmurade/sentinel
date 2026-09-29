# Sentinel

AI metric watchdog for daily e-commerce metrics.

> Stub. Setup instructions, usage, and architecture notes will be added once the
> pipeline is implemented.

## Status

Planned modules, no implementation yet:

- `app.py` — Streamlit entrypoint
- `core/detect.py` — anomaly detection
- `core/rootcause.py` — segment drilldown
- `core/agent.py` — LLM narration
- `core/memory.py` — incident retrieval (Chroma)
- `core/alerts.py` — Discord/Slack webhook delivery

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# POSIX:  source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env   # Windows (POSIX: cp .env.example .env)
streamlit run app.py
```

Requires Python 3.11+.
