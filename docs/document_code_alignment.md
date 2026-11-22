# Alignment check: Stage 1 workbook and Stage 3 prompt library vs the code

Checked on 2026-11-16 after applying `Claude outputs/capstone_fixes.patch`.

Documents checked:

- `Stage_1_Discovery_Workbook.docx`
- `Stage_3_Prompt_Library.docx`

Checked against: `src/`, `config/policy.json`, `prompts/`, `tests/`, `evaluation/`, `docs/`, `data/`, `.env.example` and the stored `evaluation/results/validation/` run.

## Summary

| Document | Verdict |
|---|---|
| Stage 1 workbook | **Mostly aligned.** All figures in rows T1 to T15 and S1 match `docs/discovery_stats_output.md`. Sections 3 and 5 cite figures the script does not produce, and the footer overstates this. |
| Stage 3 prompt library | **Mostly aligned.** Requirement ids, prompts, test names, thresholds and test count match. A few claims depend on run output that is out of date or missing from the repo. |

Result: 56 of 56 tests pass using the project's `.venv`.

## 1. Stage 1 workbook

### Aligned

- **T1 to T15 and S1** match `docs/discovery_stats_output.md` line for line. That covers the 500 tickets, 216 distinct texts, the channel, intent, urgency, tier and region splits, FCR 43.8%, satisfaction 2.97, answerable 71.4%, the 138/107/92 split of escalations, the 87 never-auto tickets, and the 35 texts carrying both route labels.
- **E1 and E2** are consistent with the output. Escalated tickets average 726.6 min against 30.5 for resolved ones, which gives 96.8% of resolution minutes. Answerable-but-escalated tickets give 46.2%.
- **S2:** 29 articles, and `last_reviewed_days_ago` is 0 on all 29 (confirmed in `data/documentation.json`).
- **Never-auto intents** (security_incident, compliance_request, feature_request, unclear_request) match `must_escalate_intents` in `config/policy.json`.
- **Design claims in the risk register** exist in code:
  - kill switch (`src/router.py`, API, file, env)
  - injection patterns and commitment patterns (`config/policy.json`)
  - five guardrails (`src/validate.py`)
  - escalation brief with summary, intent, articles and what was unsure (`src/escalation.py`)
  - circuit breaker, retry and cache (`src/llm.py`)
  - fairness slices by tier, region, fluency and channel
  - availability row (added by the patch)
- **Script and output references** now resolve. `evaluation/discovery_stats.py` and `docs/discovery_stats_output.md` were added by the patch.

### Not aligned or not reproducible

| # | Issue | Where | Suggested fix |
|---|---|---|---|
| 1 | The footer says every figure was computed by `evaluation/discovery_stats.py`. The script output has no sections for the figures below, so they cannot be reproduced from the repo. | Footer; sections 1, 3, 5 | Extend `discovery_stats.py` to emit them, or reword the footer to say these figures were computed separately. |
| 1a | Per-intent effort share, mean minutes, FCR and satisfaction (E3 to E11) | Section 3 | Add an effort-by-intent block to the script. |
| 1b | Escalation rates for billing_query (45.8%) and data_residency (69.0%) | I3 | Add a per-intent escalation block. |
| 1c | Bootstrap 95% interval for the fluency satisfaction gap (-0.19 to +0.37) | D2 | Add the interval calculation to the script, with the seed. |
| 1d | Reference-answer statistics (200 answers, 0.6 must-mention points, 3.0 must-not-claim, 20 intents) and the documentation stats (about 1,100 characters, 10 categories) | S2, S3 | Add data-source stats to the script. |
| 1e | The PII scan result ("no email addresses, phone numbers, keys") | S1 to S3 | Add the scan to the script. |
| 2 | **Row ids and the PRD mapping.** The footer says the PRD uses these row ids for traceability, and that the Stage 5 revision log will record the dropped FR-15. No PRD or revision-log file exists in the repo, so those links cannot be verified. | Footer | Add the PRD to `docs/`, or note it is held outside the repo. |
| 3 | Dataset conflicts the workbook records as open (validation "use once" vs "use as often as you like"; hidden set of 120 vs 100) are not reflected in any repo document. The README and `run_ledger.jsonl` show validation was run 3 times (7 Nov twice, 15 Nov once). | S1, S7 | Record the decision on validation reuse in `docs/`. |
| 4 | Stage 1 still says the problem statement and sentence test are for the author to write. | Sections 4 and 6 | Author action, not a code issue. |

## 2. Stage 3 prompt library

### Aligned

- **Requirement ids FR-01 to FR-08.** After the patch, `config/policy.json` maps:
  - ingest to FR-01
  - classification to FR-02
  - retrieval to FR-03
  - routing to FR-04 and FR-05
  - generation to FR-06 and FR-08
  - validation to FR-07

  These match the Stage 3 section 1 numbering. Before the patch they did not, so this was a real drift that is now fixed.
- **Prompts.** `prompts/` holds PR-01, PR-02 and PR-03 at v1. The PR-02 text in the library matches `PR-02_generate_answer.v1.md` rule for rule, including the six-sentence limit and the JSON shape. `prompts/README.md` lists the same three prompts, categories and requirements.
- **Code facts about PR-02:**
  - it is loaded by `src/generate.py` (`GENERATE_PROMPT_FILE`)
  - `<` and `>` in ticket text are replaced
  - temperature is 0.0
  - extractive fallback exists
  - default model is `gemini-2.5-flash` in `src/config.py`
- **Test names.** All 43 test names cited in section 4 exist in `tests/`. The suite has 56 tests, which matches the claim of 56 (53 test functions plus the parametrised guardrail cases).
- **Thresholds and settings:**
  - confidence threshold 0.65 and retrieval floor 0.077 in `policy.json`
  - the five guardrail names in `src/validate.py`
  - review sample of 50 with a fixed seed in `evaluation/harness.py`
  - urgency predicted but not used for routing (`src/classifier.py` says why)
- **FR-13.** The patch added `too_small_to_judge` and `flagged_over_target` to `evaluation/metrics.py`, and the harness shows them. This matches the "groups under 10 marked too small" text.
- **FR-12.** The availability figure now exists in the report.
- **Validation figures that match the stored run:**
  - route accuracy 81.2%
  - auto-answer precision 77.0%
  - first-contact 76.2%
  - maximum spread 30.6 points
  - non-fluent 57.9% against fluent 88.5% route accuracy
  - calibration gaps of 44.4 and -75.6 points

### Not aligned or needs a fix

| # | Issue | Where | Suggested fix |
|---|---|---|---|
| 1 | **Stale stored report.** `evaluation/results/validation/metrics.md` was generated before the patch. It has no availability row, no "too small to judge" marker and no "over the 5-point target" line, all of which the library describes. Re-run the harness. | FR-12, FR-13 | Re-run the validation harness and re-commit the report. |
| 2 | **TF-IDF retrieval figures are not reproducible from the repo.** The library gives 96.2% top three, 90.6% top result and 11.1% empty for TF-IDF. The only stored report is the MiniLM run (98.1%, 0.0%). Nothing in the repo holds the TF-IDF run. | FR-03 | Run the harness with `EMBEDDING_BACKEND=tfidf` and save it, or mark the figures as unverified. |
| 3 | **Intent accuracy 82.2% and macro precision 86.3% (text-grouped CV)** appear nowhere in the repo. The tuning block in `policy.json` records only automation rate 0.532 and precision 0.7707. | FR-02 | Add the grouped-CV intent metrics to `tune_thresholds.py` output or `docs/`. |
| 4 | **Router rule list is incomplete.** FR-04 lists kill switch, safety keywords, never-auto intents, no documentation and confidence. `src/router.py` also has a classification-fallback rule between never-auto intents and no documentation. | FR-04 | Add "classification fallback" to the FR-04 summary. |
| 5 | **Requirement mapping for PR-02.** The library says PR-02 serves FR-06 and FR-07 and supports FR-08. `policy.json` now stamps generation records with FR-06 and FR-08, and FR-07 on validation. This is consistent but not identical, so the log will not show FR-07 on a generation record. | PR-02 register | Say that the log stamps FR-06/FR-08 for generation and FR-07 for validation. |
| 6 | **Model name in `.env.example`.** It said `gemini-3.5-flash-lite` while the library and `src/config.py` use `gemini-2.5-flash`. The patch meant to change this, but that hunk did not apply. **Fixed in this session.** | PR-02 "Model used" | Done. |
| 7 | **Statements about work not yet done** are consistent with the repo but remain open: PR-02 never run live, PR-03 never run, 50-reply human review not done, brief spot check not done, incident procedure not walked through. | PR-01 to PR-03, FR-05, FR-10, FR-14 | Complete these before submission. |
| 8 | **Q7 accountability** is open in both the library and `docs/incident_procedure.md`, so they agree. | FR-10 | Needs a decision from the head of support. |
| 9 | **FR-15 "NOT BUILT"** matches the code, but the Stage 5 revision-log entry the library promises does not exist in the repo. | FR-15 | Add the entry when the revision log is written. |
| 10 | **PR-03 "judge" prompt** has no script or code that runs it. The library is honest about this, but a person reading the README could assume otherwise. | PR-03 | Keep the "run by hand" wording in both places. |

## 3. Cross-document check

- **Satisfaction baseline:** Stage 1 shows 3.2 (client) and 2.97 (data). Stage 3 and the harness both show 3.2 as the baseline with "not measured" as the current value. Consistent.
- **Urgency:** Stage 1 (D4) says urgency from the text is unproven. Stage 3 and `classifier.py` say 49% against a 45.2% baseline, so it is dropped from routing. Consistent.
- **Validation inflation:** Stage 1 (T1) says 62 of 80 validation texts repeat a development text. Stage 3 uses the same number, and the stored report agrees (n=62 and n=18). Consistent.
- **Escalation targets:** Stage 1 says 60% FCR and 30% escalation cannot both hold. Stage 3 and the harness still report both targets in the table. Worth a note in the report that the two are the same number.

## 3a. Status update, 2026-11-22

Resolved since the check above:

- **Stage 1 items 1 to 1e.** `evaluation/discovery_stats.py` now emits the effort-by-intent block (E1 to E11), the escalation rate by intent (I3), the fluency satisfaction bootstrap interval (D2, seed 20260101: -0.19 to +0.37), the documentation and reference-answer statistics (S2, S3) and the personal-data scan (S4). The footer claim is reproducible; see `docs/discovery_stats_output.md`.
- **Stage 3 item 1 (stale stored report).** `evaluation/results/validation/` was regenerated on 2026-11-21 and now has the availability row, the "too small to judge" marker and the over-target lines, plus 95% bootstrap intervals.
- **Stage 3 item 2 (TF-IDF figures).** `docs/sample_validation_report_offline.md` is a reproducible TF-IDF run (90.6% top result, 96.2% top three, 11.1% empty).
- **Stage 3 item 3 (grouped-CV intent metrics).** `python -m evaluation.tune_thresholds` prints them: intent accuracy 82.2%, macro precision 86.3%, macro recall 81.0%, urgency 49.0% against a 45.2% baseline.
- **Test count.** The suite now has 69 tests, not 56.

Corrections to this document:

- The validation file has been run 6 times in `evaluation/run_ledger.jsonl` (five on 2026-11-07, one on 2026-11-15), not 3.
- PR-02 has been run live: `storage/llm_cache` holds 55 model replies from 2026-11-07. No report of that run is stored and the model id is not recorded, so the open item is the missing report, not the missing run.
- `.env.example` still says `MODEL_NAME= gemini-3.5-flash-lite`, while `src/config.py` defaults to `gemini-2.5-flash`; item 6 above is not fixed in the repository.

Still open: the PRD mapping and revision log (Stage 1 item 2), the dataset-conflict decisions (item 3), the 50-reply human review, the PR-03 trial, the Q7 accountability decision, and moving the relevance floor in `config/policy.json` to the MiniLM scale (0.333).

## 4. Recommended order of fixes

1. Re-run the validation harness so the stored report includes the new availability and fairness markers.
2. Extend `discovery_stats.py` to emit the Section 3 and Section 5 figures, then update the Stage 1 footer.
3. Save the TF-IDF retrieval run and the grouped-CV intent metrics, or relabel those Stage 3 figures.
4. Make the small wording fixes in FR-04 and the PR-02 register.
5. Complete the open work: live PR-02 run, human review of 50 replies, PR-03 trial, Q7 decision.
