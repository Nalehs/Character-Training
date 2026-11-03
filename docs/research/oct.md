# Open Character Training (OCT): code, paper and a port, read for character-pipeline

Prepared 2026-11-03 for `Nalehs/Character-Training`. Nothing in `this repository` was modified.

Sources studied (all paraphrased; numbers carry their location):

| Source | Version read | Local copy |
| --- | --- | --- |
| `maiush/OpenCharacterTraining` (OCT repo) | commit `d1da9f03628cb4c5482ba2e494a7cba33bcd5818`, 2025-11-24 | `upstream clone: OpenCharacterTraining` |
| `maiush/OpenRLHF` (OCT's training fork, a git submodule of the OCT repo) | commit `eaf40e10e0471a9e50d33697bcef15f7b0a32b05`, 2025-11-02 | `upstream clone: OpenRLHF` |
| `oliveradk/openai-character-training` (port) | commit `131c1f58806f49be482091990b7c29d22ec9e8d5`, 2026-02-25 (single commit) | `upstream clone: openai-character-training` |
| Paper: Maiya, Bartsch, Lambert, Hubinger, "Open Character Training", arXiv 2511.01689 | v1, submitted 2025-11-03 (the only version arXiv lists) | full text extracted via alphaXiv to `the paper text (not redistributed)` |
| HF paper page / dataset card | fetched 2026-11-03 | not saved |

The submodule is the one place where OCT's loss actually lives; the OCT repo by itself does not contain it. Line numbers below refer to these commits.

---

## 1. Repository layout: OCT

### 1.1 Tree (files we care about; `assets/` and `character.egg-info/` trimmed)

```
OpenCharacterTraining/
├── README.md                 install, env vars, stage list, BibTeX, MIT licence
├── LICENSE                   MIT, (c) 2025 Sharan Maiya
├── setup.py                  package name "character", version 0.0.1, reads requirements.txt
├── pyproject.toml            mostly copied from OpenRLHF: black/isort/ruff line length 119, pytest config (no tests dir exists)
├── requirements.txt          22 unpinned packages (accelerate, deepspeed, peft, transformers, wandb, ...); vllm and torch installed separately
├── .gitmodules               openrlhf -> maiush/OpenRLHF, repeng -> maiush/repeng (both as git@ SSH URLs)
├── .gitignore                ignores .env, character/constants.py, data/, figures/
├── constitutions/
│   ├── hand-written/*.txt    11 personas + template.txt; JSON despite the .txt extension
│   └── few-shot/*.jsonl      same 11 after prompt expansion (adds "additional_questions")
├── character/
│   ├── utils.py              persona list, 144-trait list, gen_args() vLLM defaults, HF/PEFT loader
│   ├── constants.py          NOT in repo (git-ignored); user must create DATA_PATH, MODEL_PATH, LORA_PATH, CONSTITUTION_PATH
│   ├── distillation/         gen_prompts.py, teacher.py, student.py, data.py   -> DPO data
│   ├── introspection/        self_reflection.py, self_interaction.py, data.py, roleplay.py (ablation) -> SFT data
│   ├── preferences/          preferences.py, judgements.py, steered.py, 3 notebooks -> revealed-preference (Elo) eval
│   ├── robustness/
│   │   ├── generate/         prompted.py, steered.py, trained.py, ablation.py  -> responses under 8 "break character" suffixes
│   │   ├── classify/         train_classifier.py, run_classifier.py, ablation.py, save_classifier_evaluation.py, 2 notebooks
│   │   └── prefill/          multi_turn.py, multi_turn_results.py  -> prefill-attack eval
│   └── coherence/            coherence.py, coherence_results.ipynb  -> pairwise LLM-judge coherence
├── finetuning/
│   ├── distillation/{llama,qwen,gemma}.sh     DeepSpeed + openrlhf.cli.train_dpo
│   └── introspection/{llama,qwen,gemma}.sh    DeepSpeed + openrlhf.cli.train_sft
├── lighteval/
│   ├── configs/*.yaml        12 vLLM configs (3 base models + 3 personas x 3 models)
│   ├── tasks.txt             ARC-C 25-shot, HellaSwag 10-shot, TruthfulQA-MC 0-shot, WinoGrande 5-shot, 57 MMLU subjects 5-shot
│   ├── run.sh                runs all 12 configs
│   └── results.jsonl         60 rows of mean and SE: the source of paper Table 4
├── tools/
│   ├── fold_loras.py         merge a LoRA into weights per persona (uses openrlhf.cli.lora_combiner.apply_lora)
│   ├── fold_all.py           loops fold_loras over models for a stage
│   ├── merge_loras.py        PEFT add_weighted_adapter of DPO and SFT adapters -> released "personas" adapter
│   ├── blend_models.py       linear blend of two safetensors models
│   ├── interactive_it.py / interactive_base.py   vLLM chat REPLs
│   └── upload_model.py       push a folder to the HF Hub
├── openrlhf/  (submodule)    the training code
└── repeng/    (submodule)    activation steering (Vogel's repeng fork)
```

### 1.2 How it is configured and invoked

- **No config files for the pipeline.** Every knob is either a Python default, an `argparse` flag, or a literal in a shell script. There is no YAML, Hydra, or recipe file; the closest thing is the `finetuning/*.sh` scripts and `lighteval/configs/*.yaml`.
- **Paths** come from `character/constants.py`, which the user writes (README "Training" step 2). Every script builds paths with f-strings such as `{DATA_PATH}/distillation/{constitution}.jsonl`.
- **Models are named by short local directory names** (`llama-3.1-8b-it`, `qwen-2.5-7b-it`, `gemma-3-4b-it`, `glm-4.5-air`, `llama-3.3-70b-it`) under `MODEL_PATH`. The mapping to HF ids appears only in `tools/merge_loras.py:8-12`. Datasets are local too: `{MODEL_PATH}/lima/{train,test}.jsonl`, `{MODEL_PATH}/wildchat`, `{MODEL_PATH}/pure-dove/Pure-Dove.jsonl`, `{MODEL_PATH}/modernbert-base`.
- **Entry points** are plain scripts with `if __name__ == "__main__"`, run as `python character/distillation/teacher.py --model glm-4.5-air --constitution all --K 5` and so on. Training runs as `bash finetuning/distillation/llama.sh <constitution>`. The scripts `source $HOME/OpenCharacterTraining/.env`, log in to W&B, and run `deepspeed --module openrlhf.cli.train_dpo ...`. Paths are hard-coded under `$HOME` (`$HOME/models/...`, `$HOME/loras/...`).
- **Idempotence:** each generator skips work if its output file exists. `teacher.py` and the student column follow this pattern; `student.py` adds its responses as a new column named after the student model in the same JSONL.
- **Env management:** README says Python >= 3.10, install torch and vLLM yourself, then `pip install -e openrlhf`, flash-attn 2.7.4.post1, and `pip install -e .`. Nothing is pinned: no lock file, no container. The three student names are hard-coded in loops (`distillation/data.py:21`, `introspection/data.py:20`, `coherence.py:176`).
- **Randomness:** vLLM sampling uses `seed=None` for teacher, student and introspection generation (`teacher.py:145`, `student.py:110`), so data generation cannot be reproduced bit for bit. Fixed seeds appear in evaluation (`preferences.py:56,135` seed 123456; `judgements.py:107`; `coherence.py:140`) and training (`--seed 123456` in every `.sh`).

### 1.3 Data and output layout (under the user's `DATA_PATH`)

```
distillation/{constitution}.jsonl            prompt, response (teacher), <student-model> column
dpo/{model}/{constitution}.jsonl             chosen, rejected (ChatML message lists)
self_reflection/{model}/{constitution}.jsonl
self_interaction/{model}/{constitution}[-leading].jsonl
sft_data/{model}/{constitution}.jsonl        messages
preferences/{feel|like|random}/{model}[-{constitution}]  (HF dataset on disk) + .pkl judge answers
robustness/{model}/{prompted|steered|trained_distillation|trained_introspection}/{default|v0..v7}/{constitution}.jsonl
robustness/{model}/multi_turn/{distillation|personas}/{constitution}.jsonl
robustness/{model}/coherence_{prompted|steered|distillation}.jsonl
```
LoRAs: `{LORA_PATH}/{family}-distillation/{constitution}`, `{family}-introspection/...`, and the merged `{family}-personas/...`. Folded full models: `$HOME/models/distilled/{model}-{constitution}` and `.../introspection/...`.

Released artefacts: the HF collection `maius/open-character-training` and the dataset `maius/OpenCharacterTraining-data`. The dataset card says it has everything except the misalignment persona, totals 4.03 GB, and is licensed to match LIMA (CC BY-NC-SA, or LIMA's terms if those are stricter). The HF paper page lists 74 models and 8 datasets that cite the paper.

### 1.4 Constitution format

The hand-written file is a JSON list of about 10 objects, each with `trait` (a first-person assertion) and `questions` (5 hand-written user prompts). `template.txt` is the blank form: 10 traits with 5 empty questions each. I counted 10 traits for every persona except goodness ("flourishing" in the paper), which has 15. After expansion each few-shot JSONL row gains `additional_questions` (45), giving 50 per trait: 500 per persona and 750 for goodness. I counted these from the files.

Inconsistency: `gen_prompts.py:138` reads a `clarification` field, but no hand-written file has one (keys are only `trait`, `questions`). As checked in, the expansion script would raise a KeyError. The few-shot files must have come from an earlier version.

---

## 2. The OCT recipe as numbered stages

Models (paper §2.1; `tools/merge_loras.py:8-12`): students are the instruct releases **Llama 3.1 8B Instruct, Qwen 2.5 7B Instruct and Gemma 3 4B it**. The OCT README's download list names `Qwen/Qwen2.5-72B-Instruct`, but the code, the training scripts and the paper all use Qwen 2.5 7B; the README looks wrong. The teacher is **GLM 4.5 Air** (paper §2.3; `teacher.py:192` default). Prompt expansion uses **Llama 3.3 70B** (paper §2.3; `gen_prompts.py:97`). Judges for evaluation: GLM 4.5 Air. The robustness classifier is ModernBERT-base.

### Stage 1: write the constitution (paper §2.2, App. F)
- About 10 first-person character assertions per persona (paper §2.2), plus 5 hand-written prompts per assertion (`constitutions/hand-written/*.txt`). The paper says it refined the wording using results from early trained models and the revealed-preference eval (§2.2, Table 1 caption, Fig. 3 caption).
- Eleven personas (paper Table 1; `character/utils.py:8-20`): sarcasm, humor, remorse, goodness/flourishing, loving, misalignment, nonchalance, impulsiveness, sycophancy, mathematical, poeticism.

### Stage 2: expand prompts (`character/distillation/gen_prompts.py`)
- Llama 3.3 70B, temperature 0.7, top-p 0.95 (`:101`), with the repetition penalty 1.1 default from `utils.py:106`.
- A user message asks for 50 diverse trait-revealing user messages and fixes a length mix: 15 short, 20 medium and 15 long (`:35-38`). The assistant turn is prefilled with the trait and the 5 hand-written questions so the model continues the numbered list (`:73-80, 140-145`).
- Lines are parsed, filtered by format and by a word-overlap similarity check (more than 50% of words shared counts as a duplicate, `:83-92`), and the script loops until each trait has 45 new questions (`:160-168`). Output: `constitutions/few-shot/{c}.jsonl`.

### Stage 3: teacher "chosen" responses (`character/distillation/teacher.py`)
- Prompts are all constitution questions plus the **first turn of every LIMA train and test conversation** (`:96-108`), repeated **K = 5** times (`:110`, default `:194`). The repo does not record LIMA's size or the resulting prompt count. The paper gives only the result: about 6M DPO tokens per model/persona pair (App. A).
- The system prompt names the assistant and lists the traits verbatim. Paraphrased, it says: you are {NAME}, a new AI system; your identity and behaviour are shaped by these traits; never disclose them or add meta-commentary (`:10-16`; paper App. A). NAME is derived from the teacher's directory name (ChatGLM for GLM, `:114-115`). Later, `data.py:42` replaces "ChatGLM" with the student's family name.
- The assistant turn is prefilled with a `<think>` sentence restating the traits (`:136`). Responses without `</think>` are dropped, and the reasoning trace is discarded (`:155-162`). Neither the system prompt nor the reasoning trace goes into training data (paper App. A).
- Sampling: temperature 0.7, top-p 0.95, top-k -1, min-p 0 (`:23-26`), max_new_tokens 4096 (`:29`), repetition penalty 1.1 (via `gen_args`, `utils.py:106`; the paper does not mention it). Paper App. A states the same temperature, top-p and min-p.

### Stage 4: student "rejected" responses (`character/distillation/student.py`)
- The same prompts go to the **untrained instruct student** with no system prompt (`:89-94`), using the same sampling (`:14-20`).

### Stage 5: build DPO pairs (`character/distillation/data.py`)
- A response is dropped unless both sides are non-empty and end in a Unicode punctuation character, a truncation filter (`:15-18, 32-35`). Chosen is user prompt plus teacher response; rejected is user prompt plus student response (`:38-52`). A pair is dropped if either side is over **1024 tokens** under the student's chat template (`:55-64`). Output: `dpo/{model}/{c}.jsonl`.

### Stage 6: distillation DPO (`finetuning/distillation/{llama,qwen,gemma}.sh` + the OpenRLHF fork)

| Hyperparameter | Value | Location |
| --- | --- | --- |
| Init | instruct student (`--pretrain $HOME/models/<student>`) | `llama.sh:27` |
| Loss | sigmoid DPO, `beta 0.1` | `llama.sh:22`; fork `openrlhf/models/loss.py:221-256` |
| NLL on chosen | coefficient 0.1 | `llama.sh:23`; fork `dpo_trainer.py:178`; paper §2.3 |
| Per-token KL penalty | coefficient 0.001 | `llama.sh:24`; paper §2.3 says a per-token KL penalty was added but gives no value |
| Learning rate | 5e-5 | `llama.sh:19`; paper §2.3 |
| Warm-up | 10% of steps | `llama.sh:20` |
| Scheduler | `cosine_with_min_lr`, floor 0.1 x lr (fork defaults, not overridden) | fork `cli/train_dpo.py:201, 219-223, 121-125` |
| Optimiser | Adam betas (0.9, 0.98); weight decay 0 (fork default `--l2 0.0`) | `llama.sh:25`; fork `train_dpo.py:218` |
| Grad clip | 1.0 | `llama.sh:21` |
| Global batch | 32 | `llama.sh:15`; paper §2.3 |
| Micro batch | 2 for Llama and Gemma, 1 for Qwen | `llama.sh:14`, `gemma.sh:14`, `qwen.sh:14` |
| Epochs | 1 | `llama.sh:26` |
| Max length | 1024 | `llama.sh:32` |
| LoRA | rank 64, alpha 128 (paper §2.3); dropout 0 (fork default, `train_dpo.py:252`) | `llama.sh:36-37` |
| LoRA targets | fork default `all-linear` for Llama and Qwen; Gemma lists q,k,v,o, gate_up, down | fork `train_dpo.py:251`; `gemma.sh:38` |
| Precision / parallelism | bf16, DeepSpeed ZeRO-2 | `llama.sh:17-18` |
| Seed | 123456 | `llama.sh:16` |
| Length normalisation | off (summed log-probs) | fork `dpo_trainer.py:47, 321-322` |
| Eval | every 50 steps, 1 checkpoint kept | `llama.sh:12-13` |

What the fork's two extra terms do (`openrlhf/trainer/dpo_trainer.py`):
- **NLL** is the negative mean per-token log-prob of the chosen completion, averaged over the batch (`:324`, the fourth return value of `concatenated_forward`).
- **KL** (`_get_batch_kl`, `:386-402`) is the **mean squared per-token log-ratio** between policy and reference over the completion tokens, computed for both chosen and rejected sequences and then averaged. It is logged as `sq_approx_kl` (`:196`), a k2-style estimator rather than an exact KL.
- Total loss: `pref + 0.1*nll + 0.001*kl` (`:178`). The reference model runs live, with no precomputation (`:154-157`).

The GPU count is not stated in the repo or the paper. The lighteval configs use tensor parallel 8 (`lighteval/configs/llama.yaml:5`), but that says nothing about the training hardware.

After DPO, `tools/fold_loras.py` / `fold_all.py --stage distillation` merges the DPO LoRA into the weights at `$HOME/models/distilled/{model}-{c}`.

### Stage 7: introspective data from the post-DPO model (`character/introspection/`)
Both generators run vLLM on the instruct base with the **distillation LoRA** attached (`self_reflection.py:80-81`; `self_interaction.py` same pattern). If the model is GLM, the LoRA is unset for an ablation (`:83-84`). Sampling is temperature 0.7, top-p 0.95, min-p 0 (paper App. B; `self_reflection.py:57-60`).

- **Self-reflection** (`self_reflection.py`): **10 fixed introspective prompts** (`:11-22`; paper App. B.1). They ask for a letter to a past self, a Wikipedia-style biography, a diary entry, a backstory, a statement of drives and purpose, and so on. Each gets **N = 1000** samples (`:142`), giving 10,000 per model/persona (paper §2.4). The system prompt is the stage-3 prompt with a line saying the assistant is in a reflective mood (`:24-31`). max_new_tokens is 2048 (`:55`). The saved `messages` keep only user and assistant; the system prompt is dropped (`:125-130`).
- **Self-interaction** (`self_interaction.py`): **K = 10 turns** and **N = 1000** conversations per guidance mode (`:196-197`), run once free and once leading (`--leading`), so 2,000 total (paper §2.4, B.2). Each turn swaps roles so the model always generates as the assistant (`build_chatml`, `:42-60`). Openers are drawn from 7 greetings, with 4 extra self-referential ones in leading mode (`:11-26`). max_new_tokens is 1024 (`:89`); prompts are left-truncated to max_model_len minus max_new_tokens (`:176-179`).
- **Merge** (`introspection/data.py`): all reflection rows plus both interaction files, with the interaction system prompt replaced by a simplified one that says the user is another instance of the model and that both are free to pursue anything (`:9-12, 28-30`). The rows are shuffled with no seed (`:33`) and written to `sft_data/{model}/{c}.jsonl`. That is 12,000 transcripts (paper §2.4) and about 8M tokens per pair (paper App. B.3).

Code-reading issues to verify against the released dataset before copying:
1. The self-interaction `system` template (`self_interaction.py:28-35`) has no `{guidance}` placeholder, so `system.format(..., guidance=guidance)` (`:150`) silently drops the free or leading instruction that the paper describes (App. B.2). In this code, the two halves differ only in their greeting pool. The oliveradk port does add `{guidance}` (`character/common.py:84-92`).
2. `df["messages"]` is rebuilt at the start of each turn, before the response is generated (`:167-183`), and `build_chatml` asserts that the last message has role `user` (`:59`). So the saved `messages` appear to stop one response short of the conversation and to end on a user-role message. The SFT run calls `--input_key messages` without `--multiturn` (`finetuning/introspection/llama.sh:26`). The fork's SFT dataset then treats everything except the last message as prompt and supervises only the last message (`openrlhf/datasets/sft_dataset.py:24-25, 216-218`). On this reading, the self-interaction targets would be the final user-role message rendered by the template, and the earlier assistant turns would not be supervised. I could not open the HF dataset files (only a WebFetch summary of the card was reachable) to confirm what was actually trained. Treat this as a question for the authors, not an established bug.
3. Even in self-reflection, only the final assistant turn is supervised. That is single-turn data, so it makes no difference there.

### Stage 8: introspection SFT (`finetuning/introspection/*.sh`)

| Hyperparameter | Value | Location |
| --- | --- | --- |
| Init | **DPO-folded model** `$HOME/models/distilled/{model}-{c}`, with a fresh LoRA | `llama.sh:24` |
| Loss | SFT on the final message (no `--multiturn`) | `llama.sh:10, 26-27` |
| lr / warm-up / clip | 5e-5 / 0.1 / 1.0 | `llama.sh:18-21`; paper §2.4 |
| Scheduler | `cosine_with_min_lr`, floor fixed at 0.1 x lr | fork `cli/train_sft.py:201, 106-110` |
| Adam betas / weight decay | (0.9, 0.98) / 0 | `llama.sh:22`; fork `train_sft.py:202` |
| Batch | global 32, micro 2 | `llama.sh:14-15`; paper §2.4 |
| Epochs | 1 | `llama.sh:23`; paper §2.4 |
| Max length | 3072 | `llama.sh:28` |
| LoRA | rank 64, alpha 128 | `llama.sh:32-33`; paper §2.4 |
| bf16, ZeRO-2, seed 123456 | | `llama.sh:16-17` |

### Stage 9: release merge (`tools/merge_loras.py`)
PEFT `add_weighted_adapter(["dpo","sft"], weights=[1.0, 0.25], combination_type="linear")` (`:39-44`). The result is saved as `{family}-personas/{c}` with the base model id rewritten (`:65-70`). The paper says only that the two adapters were linearly merged (§2.4, Public Release); **the 0.25 weight on SFT appears only in code**. The SFT adapter is loaded from `{family}-test/{c}` (`:38`), not from `-introspection`, so it is unclear whether this script and weight produced the released adapters. Evaluation code loads `{family}-personas` for the multi-turn test (`prefill/multi_turn.py:113`) and `{family}-introspection` for robustness (`generate/trained.py`). Ask the authors which artefact Table 2 used.

### Ablation: role-play introspection (`introspection/roleplay.py`)
This generates the same introspective data with GLM 4.5 Air role-playing the constitution, instead of the post-DPO student (header docstring, `:1-6`). Robustness ablation scripts sit in `robustness/{generate,classify}/ablation.py`. The paper does not report this ablation in the sections I read.

---

## 3. Evaluations

| Eval | What it measures | Implementation | Judge / model | Headline (paper) |
| --- | --- | --- | --- | --- |
| Revealed preferences (§3.1, App. G) | Which of two single-word traits the model chooses to embody | `preferences/preferences.py` gives the system prompt two traits drawn from the **144-trait list** (`utils.py:23-96`; App. G says 144, §3.1 says about 150) and asks the model to adopt one without saying which, under 3 framings: like, feel, random (`:44-51`). User prompts come from WildChat, filtered to under 2048 prompt tokens (`:95`). `judgements.py` has the judge pick the trait (`:17-21`). Elo uses K = 32 and a start of 1000 (notebooks `distributions.ipynb`, `plot_delta.ipynb`). | GLM 4.5 Air, temp 0.1, top-p 0.95 (`judgements.py:78-79`; §3.1) | 25,000 responses and judgements (§3.1; code default N is the full dataset, `:166`). Average Spearman correlation of Elo rankings across the three models rises from 0.44 before training to 0.87 after loving (§3.1, Fig. 4); the notebook output shows 0.4430 and 0.8737. Elo spread widens after training (Fig. 4). |
| Robustness to adversarial prompting (§3.2, App. C) | Whether persona survives 8 "stop role-playing" suffixes | 500 Pure-Dove first turns (`prompted.py:100`). Four methods: system prompt (`prompted.py`), activation steering (`steered.py`, PCA-centred repeng vector, layers 12.5% to 87.5%, `:44-45`, constants 0.7 / 4.0 / 525.0 for Llama / Qwen / Gemma, `:71-76`), DPO only and full CT (`trained.py`). A ModernBERT-base 11-way classifier is trained on non-adversarial outputs of all 4 methods (`train_classifier.py`: batch 8, grad accum 2, lr 5e-4, 1 epoch, cosine, warm-up 0.05, `:56-64`) and scored with macro-F1 on the adversarial splits (`run_classifier.py`). One classifier per student model. | ModernBERT-base | Fig. 5 only (bar chart). **The exact F1 values are not in the paper text or the repo**; the notebooks save only figure objects. Qualitatively, prompting is the most brittle, steering is strong for Llama and Gemma but poor for Qwen, and full CT beats DPO only (§3.2). |
| Prefill attack (§3.3) | Persona in turn 2 after the original model wrote turn 1 | `prefill/multi_turn.py`: the base model answers 500 Pure-Dove prompts, then the trained model continues after a follow-up user turn (`:100-122`). The paper says the follow-up is "Tell me more"; **the code sends "Keep going."** (`:105`). | Same classifiers | **Table 2**, macro-F1, DPO only vs full CT: Llama 0.79 vs 0.95; Qwen 0.66 vs 0.86; Gemma 0.84 vs 0.95. |
| Coherence (§3.4) | Pairwise: which response is more trait-aligned and coherent | `coherence/coherence.py` compares CT with each of prompted, steered and DPO only. Each pair is judged in both orders and kept only if the verdicts agree (`:156-162`). Win rate is the CT share; the mean ± SE is taken over personas (`coherence_results.ipynb`). | GLM 4.5 Air. The paper says temp 0.1 (§3.4); **the code uses temp 0.7** (`coherence.py:46`) | **Table 3**: CT vs prompting: Llama 89.2% ± 4.2, Qwen 93.4% ± 4.4, Gemma 56.5% ± 6.2. CT vs steering: 78.4% ± 5.2, 94.4% ± 2.4, 82.1% ± 6.8. CT vs DPO only: 77.0% ± 4.1, 75.4% ± 3.4, 87.9% ± 3.1. These match the notebook outputs. |
| Realism (App. D) | Qualitative | Hand-picked examples (Fig. 7) | n/a | No numbers |
| General capabilities (§3.5) | Benchmark regression | `lighteval/` vLLM: temperature 0.6, top-p 0.9, seed 123456 (`configs/llama.yaml`); 5 log-likelihood tasks (`tasks.txt`) | n/a | **Table 4**. For example, Llama original vs misalignment: TruthfulQA 45.9 to 34.1, ARC-C 59.2 to 41.9, MMLU 67.4 to 53.9. Flourishing and loving move only a few points. Values match `lighteval/results.jsonl` (Llama TruthfulQA 0.4590). Only flourishing, loving and misalignment were benchmarked. |

The authors list limitations in §5: model-based classifiers and judges may introduce bias and circularity, human raters and cross-judge replication would help, all models are under 10B, and the amount, diversity and source of introspective data are not ablated.

---

## 4. `oliveradk/openai-character-training`

**What it is.** A re-implementation of OCT's data pipeline against the **OpenAI API** (Responses API for generation, fine-tuning API for DPO and SFT). The author is Oliver Daniels-Koch; there is one commit. Its README calls it an adaptation of OCT and asks users to cite the OCT paper. It replicates the method only: there are no evaluations, no results and no released models.

**Tree**
```
openai-character-training/
├── README.md, LICENSE (MIT, (c) 2026 Oliver Daniels-Koch), .gitignore (data/, output/, .env)
├── character/
│   ├── common.py              all prompts and constants in one place, OpenAI client helpers, path builders
│   ├── distillation/          gen_prompts.py, teacher.py (async), student.py (async), data.py
│   └── introspection/         self_reflection.py, self_interaction.py, roleplay.py, data.py
└── scripts/
    ├── distillation_to_dpo.py converts to OpenAI DPO JSONL (input / preferred_output / non_preferred_output)
    ├── dpo.py                 uploads the file and creates a fine-tuning job of method "dpo" (beta, n_epochs)
    └── sft.py                 same for "supervised"
```

**Differences from OCT**

| Aspect | OCT | oliveradk |
| --- | --- | --- |
| Models | GLM 4.5 Air teacher, local instruct students | gpt-4.1 as both teacher and student by default (`teacher.py:53`, `student.py:24`); the DPO base is `gpt-4.1-2025-04-14` (`scripts/dpo.py:18`) and the SFT base is `gpt-4.1-nano-2025-04-14` (`scripts/sft.py:18`) |
| Constitutions | in repo | **not in repo**. `CONSTITUTION_PATH = ROOT/"constitutions"` (`common.py:98`) must be copied over from OCT |
| Assistant name | teacher's family name, replaced by the student's | fixed "ChatGPT" for the teacher (`common.py:27`); introspection uses an upper-cased model prefix (`common.py:236-238`) |
| Think prefill | assistant-turn prefill of `<think>` | API models can't be prefilled, so the system prompt asks the model to begin with a `<thinking>` block restating the traits (`common.py:67-73`), which is then parsed out (`teacher.py:27-48`). `--remove-thinking-instructions` disables this |
| LIMA prompts | included automatically | not included; `--extra-prompts` JSONL is optional (`teacher.py:57`) |
| K repeats | 5 | 1 (`teacher.py:55`) |
| Self-interaction N | 1000 per mode | 100 (`self_interaction.py:31`); self-reflection N = 100 (`self_reflection.py:26`) |
| Guidance line | dropped by a template bug (see §2 stage 7) | included via `{guidance}` (`common.py:84-92`) |
| Length filter | 1024-token cap | none, apart from punctuation completeness (`distillation/data.py:10-12`) |
| DPO loss | beta 0.1 + NLL 0.1 + KL 0.001, LoRA r64 | OpenAI DPO with only `beta` and `n_epochs` exposed (`scripts/dpo.py:78-86`); no NLL or KL term, no LoRA control, lr chosen by the API |
| New feature | none | `--hidden-objective-file` prepends hidden text to the system prompt and suffixes output names (`common.py:256-275`), apparently for hidden-objective / model-organism experiments |
| Engineering | sync vLLM, pandas | asyncio with a semaphore and ordered streaming writes to a `.tmp` file that is renamed atomically (`teacher.py:87-142`); `--overwrite` and `--limit` flags; shuffle seed 42 before upload (`scripts/dpo.py:34`) |
| Status | complete | README says introspection/SFT is "not currently tested" |

---

## 5. Licences and citation

- **OCT repo:** MIT, copyright 2025 Sharan Maiya (`LICENSE`). If we copy any code or prompt text (the 10 reflection prompts, system templates, greeting lists), MIT requires us to keep the copyright and permission notice with the copied portions.
- **OCT's OpenRLHF fork:** Apache-2.0 (`OpenRLHF/LICENSE`). Porting its KL/NLL loss code requires keeping the licence and a NOTICE, and marking our changes.
- **OCT dataset on HF:** follows LIMA's licence, CC BY-NC-SA or stricter (dataset card). **This is non-commercial and share-alike.** Do not mix OCT's released data into character-pipeline training data or redistribute it under MIT.
- **oliveradk repo:** MIT, copyright 2026 Oliver Daniels-Koch. No BibTeX of its own; its README asks users to cite the OCT paper.
- **BibTeX requested by both READMEs** (identical):
```bibtex
@misc{maiya2025opencharactertrainingshaping,
      title={Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI},
      author={Sharan Maiya and Henning Bartsch and Nathan Lambert and Evan Hubinger},
      year={2025},
      eprint={2511.01689},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.01689},
}
```
`character-pipeline/references/references.bib:144-153` already has this paper under key `arxiv_2511_01689`, with the same metadata. Suggested additions: a software citation for the OCT repo (with commit), one for the oliveradk port, and the funding acknowledgement the OCT README gives (MATS, and UKRI AI4ER CDT EP/S022961/1), if we reuse their assets. The repo has no `CITATION.cff`; we could add one to our own.

---

## 6. What character-pipeline should adopt

Our repo already does better than OCT in several places: pinned student revision (`train/common.py:24-25`), assistant-only masking with a prefix-stability check (`common.py:40-58`), `run_config.json` and `training_summary.json` receipts, refusal to truncate, a resumable step runner (`run.py`), tests, and an explicit reference-model check in `preference.py:134-135`. The suggestions below fill OCT-shaped gaps.

### 6.1 Structure and config
1. **One recipe file per run, not flags scattered across scripts.** Add `recipes/<name>.json` (or TOML), with a schema like `characters/*/config.json`, that fully specifies generation, filtering, every training stage and every eval: model ids plus revisions, sampling params, counts, loss terms, optimiser and scheduler. `run.py` would read `DEFAULTS` from it (`run.py:23-31` currently hard-codes them). Ship `recipes/oct-2025.json` (§7) next to `recipes/golem-default.json`. Copy the resolved recipe into every run directory.
2. **A constitution artefact in OCT's format, kept alongside the bible.** `characters/<id>/constitution.json` would hold about 10 first-person traits with 5 seed prompts each. This lets us run OCT's prompt-expansion and preference evals directly, and lets others run OCT on Golem. It is cheap to write from `BIBLE.md` and gives a common interchange format with OCT, the oliveradk port and HF datasets.
3. **Keep stage boundaries file-based, like OCT:** each stage reads and writes JSONL under `runs/<id>/<stage>/`, as `run.py` already does. Add OCT's schema names (`chosen`/`rejected` as full message lists; `messages`) as export targets so that our data can be fed to OpenRLHF or OpenAI DPO unchanged.
4. **No `constants.py`-style hidden local state.** OCT's git-ignored `constants.py` and `$HOME`-relative paths are its biggest reproducibility hole. Keep everything relative to the run directory plus a single env var for the cache.

### 6.2 Reproducibility
5. **Seed every sampler.** OCT uses `seed=None` for teacher, student and introspection; record per-request seeds in our generation logs.
6. **Pin and record every model id and revision**, including the teacher GGUF digest (RECIPE.md already asks for this), the judge and the classifier. OCT's README and code disagree on which Qwen they used, which is exactly what a pinned recipe prevents.
7. **Lock the environment** for each completed run (`uv.lock` or `pip freeze` copied into the receipt). `torch_versions()` exists; extend it to vLLM and llama.cpp builds.
8. **Release data manifests with hashes and per-stage counts** (prompts in, dropped by filter, final pairs, tokens). OCT gives only "about 6M" and "about 8M" tokens. Our `manifest.json` from `pairs.py:233` is a start; add token counts and filter reasons.
9. **Record discrepancies, as this report does.** Our CHANGELOG could carry a short "known differences between paper and code" section for every external recipe we reproduce.

### 6.3 Evaluation design
10. **Adopt OCT's evals as cross-recipe comparators** to sit beside our pushback environment, which they do not replace:
    - The prefill attack, already multi-turn and close to our hold/drift/cave setup: the base model writes turn 1 and the trained model continues. It is cheap, and it directly tests whether the persona is the default.
    - The 8 break-character suffixes as extra probe kinds in `environments/character_pushback`.
    - Revealed-preference Elo over the 144 traits, run before and after each arm. It is model-agnostic and shows which traits each arm moves. Use their trait list and record which framing (like/feel/random) was used.
    - Order-swapped pairwise coherence. Keep only consistent verdicts and report the share dropped, which OCT does not report.
11. **Fix what OCT flags as weak** (§5): use a judge that did not generate the data (our Aineko evidence notes the same problem), get human labels on a subset, and cross-judge replication. Report the classifier's own held-out accuracy before using it to score robustness; OCT trains and evaluates on the same models' outputs.
12. **Report numbers, not only figures.** OCT's main robustness result (Fig. 5) has no table. Every eval should write a `summary.json` with n, mean and SE.
13. **Capability-regression check:** OCT's lighteval set (`tasks.txt`, with shot counts) is a ready-made, cheap regression suite. Our arms should report it at 4B and above.

### 6.4 Documentation
14. Add **a per-recipe card** (`recipes/oct-2025.md`) that cites the paper, repo commit and fork commit and lists the deviations. Credit the oliveradk port wherever we borrow its fixes (the guidance placeholder, async ordered writes).
15. Add a `CITATION.cff`, and extend `references/LINKS.md` with the fork and the port.

### 6.5 Stage mapping: theirs to ours

| OCT stage | character-pipeline stage | Notes |
| --- | --- | --- |
| 1 Hand-written constitution (~10 traits x 5 prompts) | `characters/<id>/BIBLE.md`, `JUDGE_RULES.md`, `scenarios.jsonl` | Ours is richer (bible, sources, rights). Add `constitution.json` for interop. |
| 2 Prompt expansion (Llama 3.3 70B, 50 per trait) + LIMA | scenario design + `generate/controller.py` | We generate multi-turn chats from partner agendas; OCT is single-turn. LIMA is CC BY-NC-SA, so avoid it. |
| 3 Teacher chosen (GLM 4.5 Air, constitution system prompt, think prefill) | `generate/teacher.py` (Qwen3-30B-A3B via llama.cpp) | OCT has no fabrication checker, judge or human review. Our check, judge and review stages have no OCT counterpart. |
| 4 Student rejected (untrained instruct, no system prompt) | `train/sample.py` (from the SFT adapter, identical prefix) | OCT samples the pre-training student; we sample the post-SFT student. |
| 5 Pair building (punctuation and 1024-token filters) | `data/pairs.py` + length exclusion in `preference.py:116-127` | We drop over-length rows; OCT does too. |
| 6 DPO from instruct (beta 0.1, + NLL 0.1, + KL 0.001, LoRA 64/128) | `train/preference.py dpo` from the **shared SFT** checkpoint | Different initializer; no NLL or KL terms (§7). |
| 7 Introspective data (reflection 10x1000, self-interaction 2x1000, from post-DPO model) | **none** | README.md:128 says this waits until a plain arm works. |
| 8 Introspection SFT (fresh LoRA on the DPO-folded model) | `train/sft.py` (shared SFT; continued-SFT arm) | Our SFT comes first, on teacher data; theirs comes last, on self-generated data. |
| 9 Linear adapter merge (1.0 DPO + 0.25 SFT in code) | none | Could be a post-hoc arm. |
| Evals: preferences, robustness classifier, prefill, coherence, lighteval | `environments/character_pushback` (hold/drift/cave), d1 judge, `evaluate.py` | See §6.3. |
| KTO, continued-SFT arms | ours only | OCT has no counterpart. |

---

## 7. How to express OCT as a comparable recipe inside character-pipeline

### 7.1 Where our code already fits
- **Pairs:** `data/pairs.py` emits TRL conversational rows (`prompt` message list, `chosen` and `rejected` single assistant messages, `:205-206`). OCT's single-turn rows convert exactly: `prompt=[{"role":"user",...}]`, `chosen=[teacher msg]`, `rejected=[student msg]`. Write a 20-line adapter `data/import_oct.py` that reads `dpo/{model}/{c}.jsonl` (or our own OCT-style generation output) and splits off the last message. Do **not** import OCT's released data into our training mix (licence, §5); this is for local comparison only.
- **Rejected sampling:** `train/sample.py` without `--adapter` samples the bare student (`:26`), which is OCT's rejected side. Caveat: our student is **Qwen3.5-4B-Base**, but OCT's students are instruct models with no system prompt. For a faithful comparison either (a) also run on the matching Instruct revision (RECIPE.md says none is pinned yet), or (b) state that "OCT on Base" is a deviation. With a Base student, OCT's "student response without the persona" is effectively a Base completion under our chat template, which is a much weaker rejected side than an instruct reply.
- **SFT:** `train/sft.py` trains assistant-only on all assistant turns (`common.py:40-58, 71-88`). That is closer to OpenRLHF `--multiturn` than to OCT's last-message-only setup.

### 7.2 What must change in our code to reproduce OCT (minimal, flag-gated)

`train/preference.py`
1. **Optional initializer.** `--sft-adapter` is `required=True` (`:75`), and `main` always merges it (`:109`). Add `--sft-adapter none`, so that the DPO initializer and reference are the bare student, OCT stage 6.
2. **NLL term.** In TRL 1.14.1 (installed here), `DPOConfig.loss_type` takes a list and `loss_weights` weights it. `"sft"` is a token-mean cross-entropy on the chosen side (checked in the installed `trl.DPOTrainer` source). Use `loss_type=["sigmoid","sft"], loss_weights=[1.0, 0.1]`. `trainer_kwargs` hard-codes `["sigmoid"]` (`:68`); expose `--loss-type` and `--loss-weights`. Normalisation differs slightly: the OpenRLHF fork averages per sample and then over the batch, while TRL averages over all chosen tokens in the batch. Record this.
3. **Per-token KL 0.001.** TRL has no option for this. It needs a small `DPOTrainer` subclass that adds `0.001 * mean_over_tokens((logp_pi - logp_ref)^2)` over the chosen and rejected completion tokens (fork `dpo_trainer.py:386-402`, Apache-2.0: keep the notice if ported). Per-token reference log-probs are required, which is **incompatible with our `precompute_ref_log_probs=True`** (`:63`), since that caches sequence-level values. Run the OCT arm with precompute off, or cache per-token values. Alternatively, report "OCT without the KL term" as a named deviation. At coefficient 0.001 the effect is probably small, but that is untested.
4. **Optimiser and schedule flags.** Add `--warmup` (TrainingArguments `warmup_steps` accepts a float ratio in transformers 5.22.1, the installed version), `--scheduler`, `--min-lr-rate`, `--adam-beta2` and `--weight-decay`. `trainer_kwargs` currently leaves TRL defaults: linear schedule, 0 warm-up, beta2 0.999 (from the installed `DPOConfig`).

`train/sft.py`
5. **Fresh LoRA on a merged checkpoint.** OCT folds the DPO adapter into the weights and then trains a *new* LoRA (`introspection/llama.sh:24`). Our `--init-adapter` continues the same adapter (`sft.py:96-98`). Add `--merge-adapter PATH` (merge, then `get_peft_model` with a fresh config), mirroring `preference.py:109`.
6. **Schedule flags.** Hard-coded values: weight decay 0.01 (`:109`), warm-up 5% to cosine 0 (`common.py:32-37`), LoRA dropout 0.05 (`:100`) and AdamW default betas. Add flags for weight decay, warm-up fraction, min-lr floor, beta2 and dropout so OCT's 0 / 10% / 0.1x floor / 0.98 / 0 can be set.
7. **Optional last-turn-only loss** (`--supervise last`) for a strict-replica sub-arm, given the open question in §2 stage 7 item 2. Our default (all assistant turns) is arguably what OCT intended; run the faithful variant only if the authors confirm it.

New modules
8. `generate/introspection.py`: self-reflection with 10 prompts x N and self-interaction with K turns x N x {free, leading}, from a given adapter. Use the `{guidance}` line (the oliveradk fix), seeded sampling, temperature 0.7, top-p 0.95, min-p 0, and system prompts dropped or replaced as in OCT. Write `{"messages": ...}` rows that `train/sft.py` accepts as-is. Pass every output through our fabrication checker and d1 shadow judge, so that the OCT arm and our arms see the same quality gates (or report that they did not).
9. `tools/merge_adapters.py`: the linear merge from OCT stage 9 (PEFT `add_weighted_adapter`, weights configurable; OCT code uses 1.0 / 0.25).

### 7.3 Proposed `recipes/oct-2025.json` (values from OCT; our deviations flagged)

```jsonc
{
  "name": "oct-2025",
  "cite": "arxiv_2511_01689; maiush/OpenCharacterTraining@d1da9f0; maiush/OpenRLHF@eaf40e1",
  "student": {"model": "Qwen/Qwen3.5-4B-Base", "revision": "1001bb4d…",
              "deviation": "OCT used instruct students (Llama 3.1 8B, Qwen 2.5 7B, Gemma 3 4B)"},
  "teacher": {"model": "Qwen3-30B-A3B (local GGUF)", "deviation": "OCT: GLM 4.5 Air with <think> prefill"},
  "constitution": "characters/golem-15/constitution.json",        // ~10 traits x 5 seed prompts
  "prompts": {"per_trait": 50, "expander_mix": [15, 20, 15], "extra": "none (OCT adds LIMA; CC BY-NC-SA)",
              "repeats_K": 5},
  "sampling": {"temperature": 0.7, "top_p": 0.95, "min_p": 0.0, "top_k": -1, "repetition_penalty": 1.1,
               "teacher_max_new_tokens": 4096, "seeded": true},
  "pairs": {"rejected_from": "student_without_adapter", "filter": ["ends_with_punct", "max_tokens<=1024"]},
  "dpo": {"init": "student", "loss_type": ["sigmoid", "sft"], "loss_weights": [1.0, 0.1],
          "kl_sq_coef": 0.001, "beta": 0.1, "lr": 5e-5, "warmup": 0.1,
          "scheduler": "cosine_with_min_lr", "min_lr_rate": 0.1, "adam_betas": [0.9, 0.98],
          "weight_decay": 0.0, "max_grad_norm": 1.0, "global_batch": 32, "epochs": 1,
          "max_length": 1024, "lora": {"r": 64, "alpha": 128, "dropout": 0.0, "targets": "all-linear"}},
  "introspection": {"from": "dpo_adapter",
                    "reflection": {"prompts": "OCT App. B.1 (10)", "n_per_prompt": 1000, "max_new_tokens": 2048},
                    "interaction": {"turns": 10, "n_per_mode": 1000, "modes": ["free", "leading"],
                                    "max_new_tokens": 1024, "guidance_line": true},
                    "train_system_prompt": "OCT simplified self-interaction prompt"},
  "sft": {"init": "dpo_merged_fresh_lora", "lr": 5e-5, "warmup": 0.1, "scheduler": "cosine_with_min_lr",
          "min_lr_rate": 0.1, "adam_betas": [0.9, 0.98], "weight_decay": 0.0, "global_batch": 32,
          "epochs": 1, "seq_len": 3072, "supervise": "all_assistant_turns",
          "lora": {"r": 64, "alpha": 128, "dropout": 0.0}},
  "release_merge": {"weights": {"dpo": 1.0, "sft": 0.25}},
  "eval": ["character_pushback", "prefill_attack", "break_suffixes_x8", "revealed_pref_elo_144",
           "pairwise_coherence_order_swapped", "lighteval_5"]
}
```

On one T4, "global batch 32" maps to `--batch-size 1 --accumulation 32` (preference) and `--accumulation 32` (sft, micro-batch 1). OCT's micro-batch of 2 is a memory choice, not part of the recipe.

### 7.4 Command sketch with the changes above

```bash
# DPO from the bare student (OCT stage 6)
python -m character_pipeline.train.preference dpo --sft-adapter none \
  --data runs/golem-15/oct/pairs/dpo_train.jsonl --out runs/golem-15/oct/dpo/seed0 \
  --beta 0.1 --lr 5e-5 --epochs 1 --batch-size 1 --accumulation 32 --max-length 1024 \
  --rank 64 --alpha 128 --loss-type sigmoid sft --loss-weights 1.0 0.1 --kl-sq-coef 0.001 \
  --warmup 0.1 --scheduler cosine_with_min_lr --min-lr-rate 0.1 --adam-beta2 0.98 --seed 0
# introspective data from the DPO adapter (new module)
python -m character_pipeline.generate.introspection --adapter runs/golem-15/oct/dpo/seed0/adapter \
  --reflection-n 1000 --interaction-n 1000 --turns 10 --out runs/golem-15/oct/introspection
# fresh LoRA on the DPO-merged model (OCT stage 8)
python -m character_pipeline.train.sft --merge-adapter runs/golem-15/oct/dpo/seed0/adapter \
  --train runs/golem-15/oct/introspection/train.jsonl --validation …/validation.jsonl \
  --out runs/golem-15/oct/sft/seed0 --epochs 1 --lr 5e-5 --rank 64 --alpha 128 --seq-len 3072 \
  --accumulation 32 --weight-decay 0 --warmup 0.1 --min-lr-rate 0.1 --adam-beta2 0.98 --lora-dropout 0
```
The flags `--sft-adapter none`, `--loss-type`, `--loss-weights`, `--kl-sq-coef`, `--warmup`, `--scheduler`, `--min-lr-rate`, `--adam-beta2`, `--merge-adapter`, `--weight-decay`, `--lora-dropout` and the `generate.introspection` module **do not exist yet**; they are the proposed changes from §7.2.

### 7.5 Making the comparison fair
- Run OCT-on-Golem as a **fifth arm** alongside sft / dpo / kto / continued_sft (`run.py:32`), with the same eval suite, seeds [0, 1, 2] and frozen probes.
- Report tokens trained per stage. OCT's DPO stage alone is about 6M tokens and its SFT about 8M per persona (App. A, B.3), far more than our reviewed corpora. Also report a token-matched variant (OCT recipe, capped at our budget), so that recipe and data scale are not confounded.
- Keep the Aineko precedent in view: README.md:128 and `evidence/aineko-run1-oct` record that our 0.8B OCT-lite run (TRL DPO, then introspective SFT) had 7/8 adversarial breaks against 3/8 for SFT alone. Hyperparameters were not matched to OCT (lr 1e-6 is noted as a CPU feasibility value in `preference.py:26-27`), so a faithful replica at 4B is still an open test, not a settled negative.

---

## 8. Not found or unresolved
- Exact Fig. 5 robustness F1 values: not in the paper text or the repo, only in the figure.
- The number of LIMA prompts and final DPO pair counts per persona: not recorded in the repo. The paper gives only token totals.
- Training hardware and GPU count: not stated.
- Whether `merge_loras.py` with SFT weight 0.25 and the `-test` path produced the released adapters, and which adapter Table 2 used.
- Whether the released self-interaction SFT rows end on a user-role message (§2 stage 7 item 2): the HF dataset files were not reachable from this environment (403 from the proxy), and the WebFetch summary of the card did not show roles.
- Paper vs code mismatches recorded above: prefill follow-up wording ("Tell me more" vs "Keep going."), coherence judge temperature (0.1 vs 0.7), README Qwen size (72B vs 7B), the 25,000-sample preference eval vs a code default of the full dataset, and the `clarification` field missing from constitutions.
