# Prompt register

Prompts are design artefacts: versioned files, loaded by `src/prompts.py`, whose id and version are written
into every decision-log record (`prompt_version`). Change a prompt by adding a new file
(`PR-02_generate_answer.v2.md`) and bumping `PROMPT_VERSIONS` in `src/generate.py`; never edit in place.

| ID | File | Category | Used by | Requirements | Status |
|----|------|----------|---------|--------------|--------|
| PR-02 | `PR-02_generate_answer.v1.md` | build | `src/generate.py` | FR-06, FR-07 | v1.0 current |

PR-01 (classification) is intentionally absent: classification is a calibrated scikit-learn model, not a prompt
(see `docs/architecture.md`). Add review and evaluation prompts here as you write them in Stage 3.

> The `FR-xx` identifiers are placeholders. Replace them with the IDs from **your** Stage 2 PRD so the
> chain evidence -> requirement -> prompt -> code stays traceable.
