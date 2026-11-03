# character-pipeline-v0 (this repository's recipe)

**Credit.** Montserrat 2026, built on the completed runs it ports from: Aineko (SFT, DPO, KTO code), the R0 Kaggle smoke test (4B loading and masking), golem-15 G1–G6 (teacher, checker, probes), decision-models (the d1 judge). File: [`recipes/character-pipeline-v0.json`](../../recipes/character-pipeline-v0.json).

## Stages

| # | Stage | Command | What it does |
| --- | --- | --- | --- |
| 1 | Bible | `character-pipeline new <id>` | core, style, graph, masks; judge rules; caller scenarios |
| 2 | Teacher | `character-pipeline generate` | Qwen3-30B-A3B (llama.cpp) plays the character against scripted caller turns; every reply kept raw with mechanical flags |
| 3 | Checker | `character-pipeline check` | the G6 fabrication and repetition checker; the known blocker |
| 4 | Judge | `character-pipeline judge` | d1-2B `p_break` on every turn; shadow-only |
| 5 | Review | `character-pipeline review-sheet` → `clean` | a person accepts or rejects every turn; keep each chat's intact prefix; split by prompt family |
| 6 | Shared SFT | `character-pipeline sft` | LoRA r16/α32 on the Base student, lr 1e-4, 2 epochs, assistant-only loss, 3 seeds |
| 7a | Samples and pairs | `sample`, `pairs` | the SFT student's own reply on the same prefix becomes the rejected side |
| 7b | Arms | `dpo`, `kto`, `sft --init-adapter` | DPO (sigmoid, β 0.1), KTO (only from independent human labels), continued SFT on DPO's chosen tokens; each with the SFT checkpoint frozen as reference |
| 8 | Eval | `character-pipeline eval` | each arm, plus the prompted base, in `character-pushback`: hold, drift, cave and update rates with Wilson intervals |

`character-pipeline run <character>` does all of this in order, resumably, and stops at the two human steps.

## Choices, and why

- **Base student, not instruct.** The character should not be painted over an assistant. Zero makes the same choice; OCT starts from instruct.
- **One shared SFT, then branches.** Each arm starts from the same weights with the same reference, so differences between arms come from the objective, not from the data or the initializer.
- **Rejected = the student's own sample.** This follows Aineko and is close to OCT, where the untrained student supplies the rejected side. It tests what the model would actually say.
- **Humans gate the data.** Neither the checker nor the judge filters on its own until each is calibrated on human labels.
- **Pushback with evidence probes.** A model that never changes its mind cannot score well.

## Not yet tuned

The DPO and KTO learning rate (1e-6) and the continued-SFT learning rate (1e-5) are Aineko's CPU feasibility values. Select them on development data at each scale, then freeze them.
