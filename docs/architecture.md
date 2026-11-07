# Architecture and design decisions

## 1. What the system is for
Interviews and ticket data point at the same thing: CloudServe does not lack answers, it lacks a way of delivering them
(71% of development tickets are answerable from the 29 existing articles; internal search matches titles, not
customers' wording), and escalations arrive without context (about half of what reaches tier two could have been
resolved at tier one). So the system **delivers the existing answer when it can defend it, and otherwise escalates
with a brief** — it is not an open-ended chatbot. *(Verify these figures yourself for Stage 1; they are the author's
to evidence.)*

## 2. Pipeline

| Stage | Module | Behaviour | Failure mode designed against |
|---|---|---|---|
| Ingest | `src/ingest.py` | One `Ticket` for all channels; keeps original text and channel; strips quoted replies/signatures; never raises | channel quirks leaking downstream |
| Classify | `src/classifier.py` | TF-IDF (word + char n-grams) + logistic regression → intent, urgency, calibrated confidence, top-3 alternatives; safe fallback | confidence that isn't a probability |
| Retrieve | `src/knowledge_base.py` | Section-level index, whole-article passages, relevance floor, deterministic tie-break; returns nothing when nothing is relevant | plausible-but-irrelevant retrieval |
| Route | `src/router.py` | Pure function. Priority: kill switch › safety keyword › must-escalate intent › classification fallback › no documentation › confidence < threshold › auto-respond. Reason recorded in plain language | a threshold that "looked reasonable" |
| Generate | `src/generate.py` | LLM (PR-02) with ticket fenced off as data and JSON-with-citations output; extractive fallback | unsupported fluent text |
| Validate | `src/validate.py` | 5 blocking guardrails on every reply | guardrails that only run in tests |
| Log | `src/decision_log.py` | One row per stage per ticket + final disposition; `reconcile()` | gaps in the audit trail |

## 3. Decisions worth defending

**Classical classifier, not an LLM, for classification and routing.** Deterministic (A5), no rate limit (the unattended
run cannot stall on a free tier), millisecond latency (p95 target 3 s), and — crucially — its probabilities can be
calibrated and *measured*. Confidence is temperature-scaled on grouped out-of-fold predictions.

**Routing confidence = P(intent right) × P(answerable | intent), then isotonic-calibrated.** In the development data the
text of a ticket does not fully determine whether it was labelled answerable (identical text carries both labels), so
text alone cannot support a confidence of 0.9. The per-intent answerable rate is the honest second factor, and the
calibration map makes "0.7" mean ≈70% of such tickets really were answerable (see calibration table in the report).

**Threshold chosen by cost, not by feel.** `evaluation/tune_thresholds.py` maximises `correct − W × wrong` on grouped
out-of-fold predictions; `W` (cost of a wrong automatic answer relative to the saving from a correct one) is a business
parameter, default 2, and the script prints how the optimum moves with `W`. Tickets in must-not-auto-respond
categories can never be auto-answered, by rule and by test.

**Grouped cross-validation.** The 500 development tickets contain only 216 distinct texts, and 62 of the 80 validation
tickets repeat a development text verbatim. Ordinary cross-validation puts copies of a text in both train and test and
reports ~99.6% intent accuracy; grouping by text gives ~82%. Validation-set numbers should therefore be read as
optimistic; the harness reports accuracy separately for wording seen and not seen in development.

**Predicted urgency is not used for routing.** Out-of-fold urgency accuracy is 49% vs a 45% majority baseline — no usable
signal in the text — and adding it to the prior slightly worsened grouped-CV results.

**Subject is ignored.** The same body appears under different subjects in 69 of 216 distinct bodies; the body is the
signal. (Subject is a fallback if the body is empty.)

**TF-IDF fallback for retrieval.** MiniLM + Chroma is supported and preferred (`requirements-dense.txt`); TF-IDF means a
missing download or an offline machine degrades quality rather than stopping the run (A11), and lets CI run with no
network. Measured on development tickets with expected articles: top-1 hit ≈ 87–90%, top-3 ≈ 93–96% (TF-IDF).

**No LangGraph.** The brief lists LangChain/LangGraph; this implementation uses a plain sequential pipeline because the
flow is six steps with two exits, and fewer dependencies means a more reproducible clean checkout. Each stage is a
function over explicit inputs, so wrapping them in a `StateGraph` is mechanical if you want it. *Decide whether to
reverse this and be ready to justify the choice either way.*

**Escalation brief without a model.** Built deterministically so it exists during a provider outage; drafts attached to
escalations are extractive (no quota spent) and clearly marked "not sent".

## 4. Known limitations (state these in the report)
* Offline evaluation here used TF-IDF retrieval and extractive replies; LLM-drafted replies and MiniLM retrieval need a
  run on a machine with network access and a key, and will change latency, grounding-block rates and quality.
* Free-tier request limits (`LLM_RPM`) mean LLM-mode latency is dominated by throttling; the report excludes our own
  throttle sleeps from latency and states the total separately.
* "First contact resolution" here is a **proxy** (answered with no human involvement); true resolution is unobservable.
  Customer satisfaction and hallucination rate need human assessment (`review_sample.csv`), and are not computed.
* The 60% FCR and ≤30% escalation targets are in tension with the labels: 38% of development tickets are labelled
  `escalate` and 17% must never be auto-answered, so at most ~62% could be auto-answered even by a perfect system.
* Fairness slices on 80 tickets have groups of n < 10 to ~60; spreads there are noisy and should be re-measured on the
  larger held-out run.
* The `FR-xx` requirement ids in `config/policy.json` and `prompts/` are placeholders: map them to the PRD.
