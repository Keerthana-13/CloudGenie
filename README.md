# Cloud Genie support triage

A support-triage system for Cloud Genie. It reads a ticket from any of four channels (email, live chat,
docs comments, community forum), works out what it is about, finds the relevant knowledge-base article, and then
either **answers with citations** or **escalates to a human with a prepared brief** — never silently. Every
decision is written to a log.

It is *not* a chatbot bolted onto a queue. Discovery shows that most tickets are already answered in Cloud Genie's own
documentation and that escalations reach tier two with no context, so the system's job is (1) to deliver the
existing answer when it can defend it, and (2) to hand everything else to an agent with the summary, the
documentation that seemed relevant, and a statement of what it was unsure about. See `docs/architecture.md`.

```
ticket → Ingest → Classify → Retrieve → Route ─┬→ Generate → Validate ─┬→ reply sent (labelled automated)
 (4 channels)                                   │                       └→ blocked → escalate + brief
                                                └→ escalate + brief (summary, intent, articles, what's unsure)
                         every stage writes to the SQLite decision log
```

## Quick start (about 5 minutes, no API key needed)

Requires Python 3.10 or later. The system runs **fully offline** with just `requirements.txt`: retrieval uses TF-IDF
and replies are built from the matching article's own resolution steps. Adding a model key (below) switches replies to
LLM-drafted, citation-checked answers.

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

**macOS / Linux**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then:

```bash
python -m pytest                      # 1. tests (single command, runs offline, ~20 s)

python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/validation/
                                      # 2. unattended run over a whole ticket file + metrics report (~2 s offline)

python -m src.api                     # 3. start the service, then open http://127.0.0.1:8000
```

The first run trains the intent classifier from `data/development_tickets.json` (a few seconds) and caches it in
`storage/`. Nothing else needs setting up.

### The evaluation harness (acceptance criteria A9, A10)

```bash
python -m evaluation.harness --input <any file in the ticket schema> --output <directory>
```
* Takes the input path and output directory as arguments — point it at any file with the standard ticket schema.
* Processes every ticket unattended; a malformed ticket is logged and escalated, never skipped and never fatal.
* Writes `metrics.md` / `metrics.json` (volume, business, technical, governance, calibration, fairness),
  `results.jsonl`, `decisions.db` (the decision log for the run), per-class and confusion CSVs,
  `escalation_briefs.jsonl`, and `review_sample.csv` (50 replies for two human assessors).
* Fairness rates carry 95% bootstrap intervals, and each best-minus-worst spread is labelled over target, within
  target or inconclusive according to whether its whole interval lies above, below or across the 5-point target.
* Appends one line per run to `evaluation/run_ledger.jsonl` (time, input hash, ticket count) so the report can
  state when, and how many times, the held-out set was run. `--no-ledger` disables it for experiments.
* Other flags: `--no-llm` (force extractive replies), `--limit N` (first N tickets).

## Using Gemini (or OpenRouter / Groq) for drafted replies

The model layer speaks the OpenAI chat-completions dialect, so switching provider is a `.env` change only.
Get a free key from Google AI Studio, then in `.env`:

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=<your key>
MODEL_NAME=gemini-2.5-flash        # use any model your key can access
LLM_RPM=10                         # keep below your free-tier requests-per-minute quota
```
`LLM_PROVIDER=openrouter` / `groq` work the same way with `OPENROUTER_API_KEY` / `GROQ_API_KEY`.
`.env` is git-ignored — **never commit a key**. Responses are cached in `storage/llm_cache/`, so re-runs are free and
reproducible. If the provider is down, throttled, or unreachable, the system degrades to extractive replies and
keeps going (see "Failure handling").

## Better retrieval with MiniLM + Chroma (optional)

```bash
pip install -r requirements-dense.txt     # sentence-transformers + chromadb; downloads all-MiniLM-L6-v2 (~90 MB) on first run
```
`EMBEDDING_BACKEND=auto` (default) then uses MiniLM + a persistent Chroma collection; if the model cannot be loaded it
falls back to TF-IDF with a warning. The active backend is printed in the evaluation report and in `/health`.
After switching backend, re-tune (below) because score scales differ.

## Operating it

| Task | How |
|---|---|
| Stop all automatic answers instantly (no deploy; full on-call steps in `docs/incident_procedure.md`) | `POST /admin/kill-switch {"enabled": false}` with header `X-Admin-Token: $ADMIN_TOKEN`, **or** create the file `storage/KILL_SWITCH`, **or** set `AUTO_RESPOND_ENABLED=false`. Checked on every ticket. |
| Resume | `{"enabled": true}` or delete `storage/KILL_SWITCH` |
| See one ticket's full audit trail | `POST /v1/tickets` returns the `decisions` list; or query `storage/decisions.db` |
| Prometheus metrics | `GET /metrics` (tickets by disposition, guardrail blocks, latency histogram) |
| Re-tune the routing threshold | `python -m evaluation.tune_thresholds --write` (see below) |

### How the routing threshold was chosen
`config/policy.json` holds the threshold, the retrieval relevance floor, the categories that are *never* answered
automatically (security incident, compliance request, feature request, unclear request), safety keywords, and
the guardrail patterns. `python -m evaluation.tune_thresholds` re-derives the threshold from grouped
cross-validation on the **development** tickets only and prints the whole trade-off curve; `--wrong-cost W` sets how much worse a
wrong automatic answer is than a needless escalation (a business judgement). Environment variables
`CONFIDENCE_THRESHOLD` and `RETRIEVAL_MIN_SCORE` override the file without a deployment.

## Failure handling (A11)
| Condition | Behaviour |
|---|---|
| Model provider down / rate-limited / timing out / no key | Retries with backoff, then a circuit breaker; reply falls back to the article's own steps (still cited and validated) |
| Model returns invalid JSON | Falls back to extractive reply |
| Model says it cannot answer from the sources | Escalate, with the reason |
| Nothing relevant in the knowledge base | Escalate (`no_relevant_documentation`) — retrieval returns nothing rather than something irrelevant |
| Cannot classify (empty/odd text) | Fallback intent that always escalates |
| Malformed record / missing fields / odd characters | Normalised with warnings; never raises |
| Any unexpected exception in a stage | Caught; the ticket is escalated and logged. The validator fails *closed* (an error blocks the reply) |

## Guardrails (A7) — all five run on every reply and can block
`pii` (emails, phone/card-like digits, key/token patterns, other customers' ids and names), `grounding` (every sentence
must cite a retrieved article; its words, numbers, code and links must appear in that article), `instruction_integrity`
(injection-like ticket text is blocked *before* the model sees it; prompt leakage blocked after), `tone_scope` (no
refund/fix/delivery commitments), `confidence_floor` (the threshold was actually applied). A failure blocks the reply
and the ticket is escalated — never "redact and send".

## Repository layout
```
src/            ingest · classifier · knowledge_base (+embeddings) · router · generate · validate · escalation
                decision_log · pipeline · llm · api · config · telemetry
prompts/        versioned prompt files + register (README.md)
config/         policy.json — thresholds, must-escalate intents, guardrail patterns
evaluation/     harness · metrics · tune_thresholds · discovery_stats · (results/ is git-ignored)
tests/          pytest suite (offline)         docs/   architecture, incident procedure, discovery statistics
data/           the four JSON files supplied with the project (development, validation, documentation, reference answers)
.github/workflows/ci.yml   tests + unattended validation run + secret scan on every push
```

## Attribution and AI assistance
* Libraries: scikit-learn, NumPy/SciPy, FastAPI, Uvicorn, Requests, prometheus-client, pytest; optionally
  sentence-transformers (all-MiniLM-L6-v2) and Chroma. Data files are the project pack's.
* **AI assistance:** this codebase was developed with the help of Claude (Anthropic) as a coding assistant, which
  proposed the architecture and wrote the first version of most modules and tests. *The author must replace this
  paragraph with their own account of what they reviewed, changed and overrode, as the project rules require.*
