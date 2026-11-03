# Recipes: ours and others', side by side

A *recipe* is everything between a character description and a trained model: where the data comes from, what filters it, which objectives train on it, which judge scores it, and what it is evaluated on. This repository runs its own recipe and is built so that other people's can be expressed and compared in the same code, on the same characters, at the same scales, and scored by the same evaluations.

Each recipe has a card here (what it is, credit, every stage, and how it maps onto `character-pipeline`) and, where it can be run, a machine-readable file in [`recipes/`](../../recipes). `character-pipeline recipes` lists them; `character-pipeline run <character> --recipe recipes/<name>.json` runs one.

| Recipe | Credit | Start | Data | Objectives | Can this repo run it? |
| --- | --- | --- | --- | --- | --- |
| [character-pipeline-v0](character-pipeline-v0.md) | Montserrat 2026 | Base | chat teacher + scripted callers, checker, d1 judge, human review | shared SFT → DPO / KTO / continued SFT | **Yes** (smoke-tested end to end) |
| [Open Character Training](oct-2025.md) | Maiya, Bartsch, Lambert, Hubinger 2025 | Instruct | constitution → teacher vs student pairs; introspection | DPO (+NLL) → introspective SFT → merge | **Mostly**: one KL term missing, deviations listed |
| [Zero](zero-2026.md) | Moving Castles 2026 | Base | two-model simulated chats, mixed API generators, bible + graph | LoRA SFT (+ embeddings) → GRPO/DAPO | **Partly**: SFT yes; simulated partner and GRPO missing |
| [Computer](computer-2026.md) | Celeste (@parafactual), with Luxia | Base | loom over a base model from a seed taxonomy, hand-curated documents | full FT → self-preference distillation → online self-preference RL → self-SFT | **No**: loom, document SFT, self-preference and online RL missing |

Others worth knowing, not (yet) comparable here: Anthropic's [Claude's Character](https://www.anthropic.com/research/claude-character) (2024), the original self-ranking character recipe that the self-preference ideas descend from; Fulcrum's [Echo](https://echo.fulcrum.inc) (token-level RL on an expert/base log-likelihood ratio); [talkie-1930](https://talkie-lm.com/introducing-talkie) (Levine, Duvenaud, Radford; pretrained from scratch, then SFT, online DPO and rejection-sampled SFT); [OpenCharacter](https://arxiv.org/abs/2501.15427) (Wang et al. 2025; Persona Hub profiles, SFT on an instruct model); Morpheus (Pressman 2023; a base-model character LoRA in the loom tradition). [Character Garden](https://charactergarden.com/) catalogues the landscape and its makers; cite primary sources for numbers.

## What none of them shares

The recipes differ in almost every column, so their published numbers cannot be compared with each other. Zero reports hard-break rates over 250 long conversations judged by Claude Opus 4.8; Computer reports a checker's broken-reply rate with no prompted control; OCT reports classifier F1 and judge win rates. The purpose of this repository is to run them on **one character set, one student family, one evaluation suite, several seeds** — and to say exactly where each port deviates from the original.

## Stages the repository still lacks

From the comparison (each is needed to run one of the recipes faithfully):

1. Loom / branching generation with a curation sheet (Computer).
2. A simulated partner model and generator mixtures (Zero; Computer's user simulator).
3. A base-model teacher (Computer looms from DeepSeek-V3-Base).
4. Online RL: GRPO/DAPO (Zero), REINFORCE with self-preference (Computer), online DPO (talkie).
5. A self-preference judge and self-distillation (Claude's Character, Computer).
6. Full-parameter training, or full embeddings and LM head next to LoRA, and merge-after-train (Zero, Computer).
7. Document-format training, not chat (Computer; also the grafting branch).
8. A long free-conversation hard-break eval with an **instruct**-prompted control (Zero's headline comparison; our `prompted_base` matches only Zero's activation-analysis control).
9. A capability suite (Zero's GSM8K and IFEval; OCT's lighteval set).

The repository has some things none of them report: per-turn human review as the data gate, matched insistence and evidence probes, a monitorability gate, three seeds per arm and Wilson intervals.

## Adding a recipe

Write `docs/recipes/<name>.md` (credit, links, stages with sources for every number, mapping, deviations) and, if it can run, `recipes/<name>.json`. A recipe file sets `arms`, a `pipeline` block that overrides [`run.DEFAULTS`](../../character_pipeline/run.py), and `sources` and `deviations` fields that the run copies into `runs/<id>/resolved_recipe.json`.
