# Changelog

Dated changes to the public recipe. Dates describe the decision or published update, not invented training runs.

## 2026-11-03

Added Open Character Training (Maiya et al. 2025) as a flag-gated recipe: [docs/recipes/oct-2025.md](docs/recipes/oct-2025.md) and [recipes/oct-2025.json](recipes/oct-2025.json). DPO can start from the bare student with OCT's NLL term and schedule. SFT can train a fresh LoRA on a merged adapter. New modules generate OCT's introspective data, merge the adapters and read OCT-format constitutions. Golem has a draft `constitution.json`. OCT's KL term is not implemented and is recorded as a deviation. Existing arms are unchanged. Only the tiny-model CPU smoke test has run.

## 2026-11-02

Turned the recipe into runnable code, porting it from the private run repositories rather than rewriting it. The teacher loop came from golem-15 G6. The checker and judge came from golem-15 and decision-models. Cleaning and pair building are new. SFT, DPO and KTO came from Aineko. Student loading and masking came from the R0 smoke test. Activation code came from persona-vectors and neurons.

Added the `character-pushback` verifiers environment: the G6 probes verbatim, d1-2B scoring, and hold/drift/cave rates. It has run end to end through vf-eval, but only against stand-in servers.

Added `evidence/` folders with aggregate counts and source commits. The README now follows the constitutional_grafting layout. Computer-11's figures are marked as not yet receipted.

## 2026-10-29

Published [RECIPE.md](RECIPE.md) with the current Kaggle T4 / CPU judging setup, the full student revision, the completed fp16 LoRA smoke-test scope, teacher and checker status, shared-SFT comparison arms and d1's synthetic-only calibration limit.

Added the parallel document-grafting proposal, crediting Nutter et al., Shadow-FT and constitutional_grafting. Preserved the existing Base SFT/DPO/KTO/continued-SFT route and added Base-only, unrelated-document and direct-Instruct controls. No Golem training or graft was run.

Added the public-safe bibliography, thematic catalog and annotated references. Preserved all 111 earlier arXiv-verified records and deduplicated additions from steering/identity and current recipe literature. Excluded private notes, conversations, paper full texts and PDFs.

Clarified that Computer-11 is the completed trained/evaluated predecessor; Golem generation iterations are development pilots with unresolved fabrication and checker problems.

## 2026-10-28

Selected d1-2B Q8_0 as the default cheap per-turn character-break judge, based on a synthetic pilot. Real Golem human-label calibration remains required; the score is shadow-only until then.

## 2026-10-18

Selected Qwen3.5-4B-Base as the common initializer for the Golem comparison arms, superseding the earlier instruction-backbone proposal.

## 2026-10-07

Completed the separate Computer-11 training/evaluation experiment. Later documentation preserves its small validation-set and checkpoint-selection limitations; this was not Golem training.
