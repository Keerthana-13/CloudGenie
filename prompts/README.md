# Prompt register

Prompts are design artefacts: versioned files, loaded by `src/prompts.py`. Change a prompt by adding a new file
(`PR-02_generate_answer.v2.md`) and pointing `GENERATE_PROMPT_FILE` in `src/generate.py` at it; never edit in place.
Only PR-02 runs inside the support system; its id and version are written into every LLM-drafted decision-log record
(`prompt_version`). PR-01 and PR-03 are run by hand by the author. The full register, with purpose, known weaknesses
and change history for each prompt, is in the Stage 3 Prompt Library.

| ID | File | Category | Used by | Requirements | Status |
|----|------|----------|---------|--------------|--------|
| PR-01 | `PR-01_specify_requirement.v1.md` | specification | the author, by hand | FR-01 to FR-15 | v1.0 current |
| PR-02 | `PR-02_generate_answer.v1.md` | build | `src/generate.py` | FR-06, FR-07 (supports FR-08, FR-12) | v1.0 current |
| PR-03 | `PR-03_judge_reply_support.v1.md` | evaluation | the author, by hand | FR-07, FR-14 | v1.0 DRAFT, not yet run |

Intent classification is not a prompt: it is a calibrated scikit-learn model (see `docs/architecture.md`), so it has no
prompt id. The `FR-xx` identifiers are the requirement ids in the Stage 2 PRD. `config/policy.json` maps each pipeline
stage to its requirement ids, and the decision log stamps those ids on every record.
