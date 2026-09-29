# CONTEXT.md

## What Sentinel is

Sentinel is an AI metric watchdog. It detects anomalies in daily e-commerce
metrics, finds the root-cause segment using pure-Python drilldown tools, and an
LLM agent narrates ONLY numbers returned by tools. Alerts go to
Discord/Slack webhook. Past incidents are stored in Chroma for retrieval.

## Non-negotiable rules

1. **Type hints everywhere.** Every public function and module-level constant
   gets a real annotation. Prefer built-in generics (`list[Anomaly]`, `dict[str, float]`)
   and `X | None` over `typing` imports.
2. **Simple readable code.** Prefer a straight-line function over a clever one.
   No premature abstraction, no metaprogramming, no framework layering.
3. **No hardcoded secrets.** API keys, webhook URLs, and model names come from
   environment variables loaded via `python-dotenv` (`.env`, git-ignored, with
   `.env.example` as the documented template).
4. **The LLM must have a template fallback so the app never breaks.** If the LLM
   call fails, times out, or no API key is configured, narration falls back to a
   deterministic string built from the numbers the tools already returned. A
   missing or broken LLM must never raise, hang, or block a run.

## The agent's contract

The agent is a narrator, not an analyst.

- It may **only** state numbers that a tool returned. It never invents,
  interpolates, or estimates a figure.
- It may not add a cause, a recommendation, or a forecast that no tool produced.
- If it needs a number that no tool provided, that number does not go in the
  output.
- Anything the agent writes about causes must trace back to a drilldown result
  in `core/rootcause.py`.

This keeps the model in a role where hallucination is a formatting bug rather
than an analysis bug.

## Module map

| Path | Responsibility |
| --- | --- |
| `app.py` | Streamlit UI: upload/run, charts, narrative, alert toggle |
| `core/detect.py` | Load daily metrics, flag anomalies (scikit-learn / statsmodels) |
| `core/rootcause.py` | Pure-Python dimension drilldown to find the responsible segment |
| `core/agent.py` | Grounded narration from tool output, template fallback; LLM optional and not yet wired |
| `core/memory.py` | Persist and retrieve past incidents (Chroma, built-in ONNX embeddings, no sentence-transformers/PyTorch, with a TF-IDF + cosine fallback) |
| `core/alerts.py` | POST the incident summary to a Discord/Slack webhook, or email over SMTP |
| `data/` | Sample/generated datasets, Chroma persistence dir |

## Data shape (assumed)

Daily e-commerce metrics, one row per day per dimension combination, e.g.
`date`, `metric` (revenue, orders, conversion_rate, aov, ...), and segment
dimensions such as `region`, `category`, `device`, `channel`. Long format is
preferred; the detectors reshape as needed.

## Flow

1. **Detect** — `core/detect.py` reads the dataset and returns anomalies with
   the observed value, the expected value, and the deviation.
2. **Drill down** — `core/rootcause.py` takes each anomaly and decomposes the
   metric by dimension to find which segment explains the drop.
3. **Narrate** — `core/agent.py` gets the tool output and writes a short
   incident summary. Template fallback on any failure (rule 4).
4. **Remember** — `core/memory.py` embeds the summary into Chroma so similar
   past incidents surface next time.
5. **Alert** — `core/alerts.py` posts the summary to the configured webhook.

Steps 3–5 are each independently skippable: a missing webhook or a dead LLM
degrades the run, it does not fail it.

## Conventions

- Config loads once, at import, from the environment.
- `requests` calls carry an explicit timeout. Nothing blocks indefinitely.
- Tests live in `tests/` and run with `pytest`.
