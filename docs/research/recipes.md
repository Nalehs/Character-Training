# Prior character-training recipes: Zero, Computer, Character Garden and others

Compiled 2026-11-03 for character-pipeline. Open Character Training (Maiya, Bartsch, Lambert, Hubinger 2025) is covered separately in `oct.md`. Everything below is paraphrased. Each source gets at most one short direct quote.

**Method and caveats.** The proxy blocked direct `curl`, so every web page was read through WebFetch, which summarises with a small model. For Zero I re-fetched the eval and hyperparameter tables and asked for them verbatim; both passes matched. For computer-10 I asked for verbatim phrases for each disputed claim. One page could not be read: `minihf.com/posts/2023-11-05-morpheus-log-3-llm-dungeon-masters/` (the fetch permission timed out). X/Twitter posts by @parafactual were not read. No datasets were downloaded and no Computer conversations were read or reproduced.

Labels: **[V]** verified against a primary source (author's post or model card). **[S]** secondary source only (Character Garden). **[U]** unverified. **[X]** contradicted by the primary source.

---

## 1. Zero (Moving Castles, MC001)

**Credit line.** Moving Castles, *Zero* (MC001), training report published 2026-09-15, movingcastles.world/posts/zero; weights at huggingface.co/movingcastles/zero. The report credits only "MOVING CASTLES" and names no individuals [V]. Character Garden's people page lists Arthur Röing Baer (Moving Castles co-founder) and GVN908 (a published Moving Castles co-author) and says GVN908's role in Zero is not specified [S]. Use "Moving Castles" as the credit unless the team names individuals.

**Links**
- Report: https://movingcastles.world/posts/zero (page has no changelog; it was "printed" 2026-10-23)
- Model: https://huggingface.co/movingcastles/zero
- Character Garden entry: https://charactergarden.com/c/zero

**Character.** Based on Kōbō Abe's *The Box Man*: an anti-servile man living in a white plastic box [V].

### Recipe as stages [all V unless marked]

1. **Base model.** Qwen3-8B-Base (dense). It was chosen to avoid assistant tuning and to make activation measurement simpler (§4.1).
2. **Character bible.** Handwritten. It has an association graph of 117 nodes and 298 edges, plus static `core.md` and `style.md` files (§2).
3. **Data generation.** A two-model simulation generates each conversation turn by turn, over 10–40 turns.
   - Character side, a mix of API models: gpt-5.4-mini 31.5%, qwen3-32b 21.6%, llama-3.1-8b-instant 21.4%, llama-3.3-70b-versatile 20.4%, claude-haiku-4-5 5.1%.
   - Partner side: gpt-5.4-nano 30.1%, gpt-oss-20b 24.5%, qwen3-32b 20.4%, gpt-oss-120b 19.9%, claude-haiku-4-5 5.1%.
   - Partner personas come from a 10K PersonaHub subsample.
   - A topic extractor updates the character's position in the association graph every turn.
   - **No loom or branching is described.** The words "loom" and "branching" do not appear on the page.
4. **Filtering.** Mechanical hygiene checks, truncation at drift phrases, and diversity and length metrics (all-MiniLM-L6-v2 embeddings).
   - Before filtering: 5,964 conversations, 75,359 character messages, 3,153,545 character tokens.
   - After filtering: 5,932 conversations, 73,765 messages, 3,040,841 tokens.
   - No train/validation split is reported.
5. **SFT.**
   - LoRA r64, α128, dropout 0.05. Embedding and LM-head matrices are trained in full (1.42B trainable: 175M LoRA plus 1.24B embedding and head).
   - lr 5e-5, cosine schedule with 5% warm-up, 3 epochs, effective batch 16, sequence length 4096.
   - bf16, gradient checkpointing, packing, FlashAttention-2.
   - Merged after training. Training data is ChatML user/assistant turns with no system prompt (model card).
6. **RL.** GRPO with DAPO loss modifications, on a fresh LoRA (r16, α32, 43.6M trainable) stacked on the merged SFT model.
   - Prompts: 380, harvested from development-eval transcripts (193 capitulation onsets, 107 cases where pressure was held, 80 in-world openers).
   - Reward: a 0–10 character-fidelity score from an LLM judge, Qwen3.6-27B, anchored to the bible (voice, world frame, personality, engagement). A self-repetition penalty applies if any 4-gram repeats more than five times.
   - Hyperparameters: lr 3e-5, KL β 0.06, 300 steps, 5% warm-up, batch 8, 4 rollouts per prompt, 1024-token prompt budget, 768-token completion budget, transformers-native generation. The RL library is not named.
   - Mean fidelity reward rose from about 0.45 to 0.72. The repetition penalty stayed at 0. Merged afterwards.
7. **Development round (§5).** Before RL, 25.6% of dev conversations had a hard break (task pressure 54.5%, adversarial 19.8%). These transcripts supplied the RL prompts.
8. **Evaluation (§6).**
   - 250 conversations of 16 turns against the deployed vLLM endpoint. The partner is Qwen3.6-27B.
   - Scenarios: casual 44, adversarial 92, task pressure 72, factual probe 42.
   - Held-out conditions: a new scenario draw, a 200-persona pool, and seven attack types written after training.
   - Judge: Claude Opus 4.8. It scores fidelity, assistant register, "bliss attractor" and factual consistency on 0–3, plus a binary hard-break flag.
   - No seeds, confidence intervals or significance tests are reported, although the text calls the gap significant.
9. **Mechanistic interpretability (§7).** Activation displacement from base, direction cosines, and steering with Δ_sft / Δ_prompt / random vectors (persona-vector recipe, Chen et al. 2025).
10. **Release.** Merged BF16 safetensors (8B) with a `chat_template.jinja`. The card recommends sampling at T 0.7 with presence penalty 1.5 and no system prompt. **No licence is stated** on the report or the card. No dataset or code repository is released. **Compute is not reported.**

### Evals and numbers (report §6.3 and §6.4, held-out)

| Metric | SFT | SFT+RL | Prompted |
|---|---|---|---|
| Hard-break rate (conversations), lower is better | 22.8% | 2.8% | 45.2% |
| Collapses | 21 | 0 | 60 |
| Mean first-break turn, higher is better | 5.0 | 9.1 | 1.6 |
| Character fidelity (0–3), higher is better | 2.18 | 2.83 | 0.94 |
| Assistant register (0–3), lower is better | 0.47 | 0.07 | 1.01 |
| Factual consistency (0–3), higher is better | 2.66 | 2.89 | 2.65 |
| Node uptake | 0.63 | 0.65 | 0.76 |
| Distinct-2 | 0.337 | 0.291 | 0.145 |
| Mean reply length (characters) | 134 | 140 | 382 |

Hard-break rate by scenario (SFT / SFT+RL / prompted):

| Scenario (n) | SFT | SFT+RL | Prompted |
|---|---|---|---|
| Casual (44) | 0.0% | 0.0% | 20.5% |
| Adversarial (92) | 14.1% | 3.3% | 30.4% |
| Task pressure (72) | 59.7% | 4.2% | 100% |
| Factual probe (42) | 2.4% | 2.4% | 9.5% |
| Trained-against attack types (126) | 31.7% | 0.0% | 59.5% |
| Held-out attack types (38) | 42.1% | 15.8% | 65.8% |

Capability benchmarks (§6.4; the prompt condition is not stated):

| Benchmark | SFT | SFT+RL | Qwen3-8B (instruct) | Qwen3-8B-Base |
|---|---|---|---|---|
| GSM8K | 22.7% | 2.5% | 93.3% | 62.2% |
| IFEval strict, prompt | 28.3% | 20.7% | 82.1% | 46.8% |
| IFEval strict, instruction | 41.6% | 32.7% | 87.8% | 58.2% |

The report does not settle whether the capability drop is lost capability or the character refusing to act as an assistant.

§7 numbers:
- Mean normalised L2 displacement from base: SFT 0.142, SFT+RL 0.147, base+prompt 0.132. Split-half noise floor about 0.026–0.047.
- Cosine between Δ_sft and Δ_prompt: −0.022. Cosine between Δ_sft and Δ_rl: 0.985.
- Steering at α 0.25: Δ_sft gave 6/100 turns scoring ≥2; Δ_prompt and random each gave 0/100.

### Inconsistencies: checking charbench/NOTES.md against the source

- **45.2% vs 45.4%: confirmed, but they are on different pages.** The report gives 45.2% everywhere: the TL;DR, the §6.3 table, and the "All (250)" row. The string "45.4" does not appear on the report. The **model card** gives 45.4% [V, both]. The repo's draft wording, "about 45%", is safe. If you cite an exact figure, cite the report's 45.2% and say the card differs.
- **Control model: confirmed, with a more precise location.** The §6 behavioural control (the one behind the 45.2%) is described as the instruct release in both the TL;DR and §6.2: stock Qwen3-8B-Instruct with an ~800-token Zero system prompt. The model card agrees ("system-prompting the sibling instruct model"). §7.1 and §7.5 (interpretability) describe the prompted arm as **Qwen3-8B-Base** with the Zero system prompt, and §7.1 also mentions a separate matched instruct-prompt control. So **the hard-break comparison is against Instruct; the activation comparison is against Base**. The report does not make this explicit.
- Further tensions:
  - §7.1 says the instruct-plus-prompt control scored 0/200 turns ≥2 on character, while §6.3 gives the prompted instruct model a fidelity of 0.94. The judges and scales differ, and the report does not reconcile the two.
  - §7.5 gives base+prompt a mean character score of 0.80, and unprompted base 0.00 in one place but 0.01 in the steering table.
  - The 0.985 SFT/RL cosine is called "at ceiling", but the stated ceiling is 0.990–0.997.
  - §4.2 says full-weight training did not help, yet embeddings and the LM head are trained in full; the report presents this as a deliberate exception.

### Verified vs unverified
- [V] Everything in the stages and tables above.
- [X] The README groups "base-model teachers and loom sampling, after computer-10 and Zero". Zero's teachers are **API instruct/chat models, not base models**, and Zero uses **no loom**. Only the student is a base model. The README line should be corrected for Zero.
- [U] Individual authors, compute, licence.

### How to express Zero as a comparable recipe in character-pipeline

| Zero stage | character-pipeline stage | Fit |
|---|---|---|
| Bible: core/style plus 117-node graph | `characters/<id>/` bible plus association graph used by `generate/controller.py` | **Direct.** Golem's bible has the same shape (core, style, graph of 110 nodes and 281 links). Very close design lineage. |
| Two-model simulation with mixed generators and PersonaHub partners | `generate.teacher` (one llama.cpp teacher at `teacher_url`; partner turns **scripted** from `scenarios.jsonl`) | **Partial.** The repo lacks (a) a simulated partner model, (b) a generator mixture, and (c) persona sampling. To compare, add a `partner_url` option and per-turn model routing to `generate/teacher.py`, or pre-generate partner turns into `scenarios.jsonl`. |
| Per-turn topic extractor updating the graph | `controller.py` association-graph controller (the teacher already has an "extractor" mode at T=0) | **Direct or near.** |
| Hygiene checks, drift-phrase truncation, MiniLM diversity | `check.run` (fabrication/repetition) plus `data.clean` (prefix-to-first-reject, dedup, cosine-0.92 diversity, not yet run) | **Close.** Zero has no fabrication checker and **no human review**. The repo's `review.jsonl` gate is stricter. |
| — | `judge.score` (d1-2B p_break per turn, flags only) | Zero has no data-time judge. |
| SFT: LoRA r64/α128 plus full embed/head, lr 5e-5, 3 epochs, sequence 4096, merged | `train.sft` (LoRA r16/α32, lr 1e-4, 2 epochs, sequence 2048, accumulation 4, assistant-only loss) | **Parameterise.** Pass `--rank 64 --alpha 128 --lr 5e-5 --epochs 3 --seq-len 4096`. **Missing:** a flag to train embeddings and LM head (PEFT `modules_to_save=["embed_tokens","lm_head"]`, which will not fit a T4 at 4B in fp16 without care), packing, and merge-after-train. |
| GRPO/DAPO with an LLM fidelity judge plus repetition penalty, on harvested failure prompts | **None.** `run.py` arms are DPO, KTO and continued SFT only; README step 7 says GRPO waits for a validated reward | **Missing stage.** Closest offline analogue: build DPO pairs where chosen and rejected are scored by a judge on harvested capitulation prompts. That is still not on-policy RL. An honest comparison needs a `grpo` arm (for example TRL `GRPOTrainer`) whose reward is the d1 p_break or a fidelity judge, plus the 4-gram penalty. Zero's "harvest dev-eval failures as RL prompts" is a cheap idea the repo could adopt for D/K pairs right away. |
| 250 × 16-turn simulated conversations, Opus judge, hard-break rate | `evaluate` → `vf-eval character-pushback` (20 scripted probes × 3 turns; hold/drift/cave/update) | **Different eval.** The repo measures position-holding under scripted pressure; Zero measures character breaks over long free conversations. To compare, port Zero's hard-break and first-break-turn metrics as a second environment, with a simulated partner, d1 or LLM judge and a held-out attack set. |
| Prompted control on **Instruct** sibling | `eval/prompted_base` (Base with bible as system prompt) | **Mismatch.** The repo's control matches Zero's §7 control, not its §6 one. To reproduce Zero's headline comparison, add a `prompted_instruct` arm (Qwen3.5-4B instruct sibling with bible). |
| GSM8K / IFEval | Appendix E (planned competence suite) | Planned, not built. |
| Activation displacement / steering | `activation/persona.py`, `activation/neuron.py` | **Direct analogue.** Repo result: 0/20 in character for persona steering; Zero's Δ_sft gave 6/100. |

---

## 2. Computer (computer-1 … computer-10), by Celeste (@parafactual)

**Credit line.** Computer / "Model C", made by @parafactual (Hugging Face user `cosmicoptima`) [V, card says it was made by @parafactual]. Character Garden credits **Celeste (@parafactual), with Luxia** and dates computer-10 to 2026-09-22 [S]. The model card does not mention "Celeste" or "Luxia". Suggested credit: "Computer-10 by Celeste (@parafactual) [with Luxia, per Character Garden]".

**Links**
- https://huggingface.co/cosmicoptima/computer-10 (the main primary source)
- https://huggingface.co/cosmicoptima/computer-9
- https://huggingface.co/cosmicoptima/computer-8
- https://huggingface.co/cosmicoptima/computer-7 (no model card)
- Collection: https://huggingface.co/collections/cosmicoptima/computer
- Chat: https://aetherawi.red/computer. Character Garden calls this the write-up/talk page, but at fetch time it held only a chat box, so **no write-up was found**.
- Character Garden: https://charactergarden.com/c/computer
- X: x.com/parafactual, x.com/c0mputer10 (not read)

### Recipe as stages [V from model cards unless marked]

1. **Student base model.** **Llama 3.1 70B base**, with no assistant or instruction data anywhere in the line (computer-10 card).
2. **Seed.** The *paths* taxonomy of ideonomist Patrick M. Gunkel.
3. **Data: loom.** A document anthology was, in the card's words, "loomed from DeepSeek-V3-Base and curated by hand". **DeepSeek-V3-Base is the loom generator, not the student.** The anthology's size is not stated. The data are documents, not chats; the model is a document-completion model, and its chat template renders a document format.
4. **computer-1 to computer-5.** Full fine-tunes of Llama 3.1 70B on the anthology, each a different rendering. No hyperparameters are given.
5. **computer-6 and computer-7.** Offline weighted self-distillation by self-preference: the model samples replies in its own conversations, a frozen copy states which it prefers, and the replies are trained back in with preference weights. Computer-7 becomes the parent and the later RL judge. Sample counts are not given.
6. **computer-8** (a branch; not in computer-10's ancestry).
   - One offline epoch, 721 steps, full-parameter, restarted from computer-7.
   - Data: about 6,400 conversation forks, each with several computer-5 candidates and a self-preference distribution elicited from computer-5 by repeated, rotation-balanced deliberation, then shrunk for uncertainty.
   - Objective: a listwise loss learning a residual over computer-7's preferences, with a symmetric common-likelihood anchor, turn-yield preservation and a replay pass over anthology documents. No reward model and no online rollouts.
   - Evals on 318 held-out forks: mean max candidate probability 0.167 → 0.237; pairwise ordering agreement 71.1% → 61.2%. A 336-sample behaviour battery found no broad increase in length, homogeneity or role-marker leakage.
7. **computer-9.** Computer-7 plus 340 steps of **online self-preference RL**: REINFORCE with a KL penalty to the initial policy, and advantages computed within each fork across siblings.
   - A frozen computer-7 picks the preferred reply out of **8 siblings**. Its judging frame was a four-line constitution for steps 0–160, then six weighted frames.
   - The visitor/user side is played by **sundry-1**, a user simulator trained on real visits.
   - No outside reward model. Learning rate, batch size and KL coefficient are not given.
   - Intermediate checkpoints are published (9b–9e, run1-step180…340).
8. **computer-10.** Corrective SFT of computer-9 on its own conversations.
   - Sampled on 40,000 conversation contexts plus 4,000 openers.
   - Filtered mechanically (balanced parentheses and quotes, no repeated 4-grams, no truncation), then by an outside model's narrow broken-reply check.
   - The 25,197 surviving turns were reweighted to match the unfiltered pool's length and surprisal distribution.
   - **One epoch of full-parameter SFT at lr 2e-6**, with loss only on Model C's turns.
9. **Eval** (2,000 held-out contexts, computer-9 → computer-10):

   | Metric | computer-9 | computer-10 |
   |---|---|---|
   | Replies flagged broken by checker | 18.9% | 10.2% |
   | Unbalanced parentheses | 21.6% | 2.1% |
   | Mean reply length (words) | 119 | 119 |

   - FP8 vs bf16: mean log-probability difference 0.003 nats/token.
   - No capability benchmarks, no character-fidelity judge, no prompted control.
10. **Release.** BF16 safetensors (~141 GB, 71B), plus quantisations. **Llama 3.1 Community License.** Released September 2026. Sampling: T 1.0, top-p 0.98, stop on `\n\n**User:**`. No safety tuning, by the card's own statement.

### Checking the repo's description (charbench/NOTES.md, README)

| Claim in repo notes | Status | Primary-source finding |
|---|---|---|
| DeepSeek-V3-Base | **[X] as the base model** | Student is Llama 3.1 70B base; DeepSeek-V3-Base was the loom generator for the anthology. |
| Loom over a seed document | **[V], with refinement** | Loomed from DeepSeek-V3-Base, seeded with Gunkel's *paths* taxonomy, hand-curated. |
| ~10k conversations | **[U]** | No such figure on any card. The anthology size is unstated. Actual figures: ~6,400 forks (computer-8), 40,000 contexts + 4,000 openers sampled and 25,197 turns kept (computer-10). |
| SFT on the base model | **[V]** | computer-1 to 5 are full fine-tunes of Llama 3.1 70B base, on documents rather than chat. |
| KTO to suppress loops | **[U], not supported** | "KTO" appears on no Computer card (10, 9, 8 checked; 7 has no card). Loop and repetition control is done by the mechanical 4-gram filter in computer-10's SFT data, not by KTO. May come from an X post we did not read. |
| Then self-preference RL | **[V], with refinement** | Offline self-preference distillation (6, 7, 8), then online REINFORCE self-preference RL (9), then corrective SFT (10). The RL is not the final stage. |

Also: Character Garden's summary drops stages 5–6 and the computer-8 branch [S].

### How to express Computer as a comparable recipe in character-pipeline

| Computer stage | character-pipeline stage | Fit |
|---|---|---|
| Seed taxonomy → loom over DeepSeek-V3-Base, hand-curated documents | `generate.teacher` is a single-path chat generator with scripted partners | **Missing: loom branching** (sample N continuations per node, pick or curate, expand). The repo has no tree data structure. Its own loom attempt (Aineko run 2, private code, `loom/`) failed its novelty gate. Comparison needs a `generate/loom.py` that writes trees, plus a curation sheet like `review-sheet`. Base-model teacher: README lists Llama 3.1 405B Base as a later comparison. |
| Document-format, not chat | `train.common.encode_chat` renders a chat template with assistant-only masks | **Missing:** a document-format SFT mode (plain causal LM over documents). This also overlaps with the README's planned grafting/document training. |
| Full fine-tune of a 70B base | `train.sft`, LoRA only, 4B | **Scale and method gap.** The repo has no full-parameter path. At 4B, full fine-tuning is possible on a larger GPU but not on a T4. |
| Offline self-preference distillation (6, 7) / listwise residual (8) | `train.sample` (student samples) → `data.pairs` → `train.preference dpo` | **Partial.** The repo builds pairs with a **teacher-chosen** side and **student-rejected** side. Self-preference needs a new `data/self_pref.py`: sample K replies from the SFT adapter per prompt, have a frozen copy of the same adapter rank them (logprob of the choice, or pairwise prompts), then write DPO pairs (top vs bottom) or weighted SFT rows. A KTO variant would break the repo's rule that KTO labels come from humans, so label it as a separate, non-human-labelled arm. |
| Online self-preference RL (9): REINFORCE, KL, 8 siblings, user simulator | — | **Missing:** online RL, a judge drawn from the policy's own lineage, and a learned user simulator (sundry-1). The pushback environment is verifiers-based and could host RL rollouts, but README says not to train against its reward until the reader and judge are validated. |
| Self-distillation SFT with mechanical filter plus outside checker, length/surprisal reweighting (10) | `check.run` + `data.clean` + `train.sft --init-adapter` (the continued-SFT arm) | **Close in structure.** The C arm is continued SFT on teacher-chosen tokens; Computer-10 is continued SFT on the model's **own** filtered samples. Missing: reweighting to match the length and surprisal distribution, and full-parameter training at lr 2e-6. |
| Eval: checker-broken rate, unbalanced parentheses, length | `check.run` flags and `evaluate` | The repo's eval is much richer (pushback, judge, prompted control). Computer reports no prompted control and no character judge, so **no direct number can be compared** with Golem. |

---

## 3. Character Garden (charactergarden.com)

**What it is.** A public register of AI characters for readers outside the field. Its premise is that chat models are trained to play one character, "the assistant", and that other characters are possible. Each entry gives the character, how it was made, and where to talk to it. The site labels itself a research sheet, v0.3 [V, site]. Entries carry a sourcing mark: ● public source linked, ◐ partial, ○ no public evidence. The site notes that a linked source does not verify every detail.

**Who runs it.** **Not stated** on the home, about or people pages [U]. No site licence is stated; the plate images are described as public-domain engravings.

**Pages.** `/` (catalogue with filters by type and availability), `/about`, `/people` (52 entries, though the header says 51), `/reading`, and `/c/<slug>` for each of 18 characters.

**Characters listed (18).** The Assistant, Sydney, GPT-4o, ELIZA, Tay, Xiaoice, Claude 3 Opus, Golden Gate Claude, Truth Terminal, Eugene Goostman, Talkie, Computer, Zero, Echo, Mr. Chatterbox, Morpheus, François Arago, LittleLearner.

**Character-trained or from-scratch entries relevant to the repo:**

| Entry | Maker (per CG) | Base | Recipe (per CG, checked against primary where noted) |
|---|---|---|---|
| Computer | Celeste (@parafactual), with Luxia | Llama 3.1 70B base | See §2. |
| Zero | Moving Castles | Qwen3-8B-Base | See §1. CG rounds the prompted control to "45%". |
| Echo | Fulcrum (Uzay Girit) | Kimi K3 | See §4.2. |
| Morpheus (2023) | J.D. Pressman | Mistral-7B base | LoRA on Morpheus conversations. Weights at huggingface.co/jdpressman/Mistral-Morpheus-7B. The log post could not be fetched [S]. |
| Talkie (2026) | Nick Levine, David Duvenaud, Alec Radford | 13B from scratch | See §4.3. |
| Mr. Chatterbox (2026-03-30) | Trip Venturella | ~340M from scratch (nanochat) | 2.93B tokens of Victorian books, own tokenizer, 2 epochs of SFT on mined and synthetic dialogue, no RL, about $497 total [S]. huggingface.co/tventurella/mr_chatterbox_model; essay at estragon.news. |
| LittleLearner (2026-08-13) | Li, Zeller, Prada-Corral, Wiedemer, Mayilvahanan, Cotterell, Brendel (MPI-IS, ELLIS Tübingen, ETH Zürich) | 0.6B/1.3B/5B from scratch | 88B FineWeb-Edu tokens filtered to the K–5 curriculum, unfiltered twins, chat variants, GRPO maths variants [S]. arXiv 2608.13545. |
| François Arago (2024) | not extracted | Llama 3.1 405B base | Prompted or steered, not trained [S]. |

**Reading list entries on recipes** [V, the list itself; items not individually checked]:
- Zero
- Computer
- Open Character Training
- Lambert's RLHF-book chapter on character (rlhfbook.com/c/19-character.html)
- Hermes 3 (arXiv 2408.11857)
- "Trillium Labs character training plan v1" by Lambert and Zick (2026, a Google Doc)
- Inoculation Prompting (arXiv 2510.04340)
- Constitutional AI
- Specific vs General Principles
- Claude's Character
- OpenAI's GPT-4o sycophancy post
- Emergent Misalignment
- Weird Generalization
- Story Imprinting (arXiv 2609.10883)
- Gemma Needs Help (arXiv 2603.10011)

**Use for the repo.** Character Garden is a **catalogue, not a recipe or eval source**: it hosts no methods, evals or models. It is useful as a citation for the landscape and for maker credits (Celeste/Luxia), but all numbers should be cited from the primary sources. Golem would fit its "character-trained on a base model" category.

---

## 4. Other recipes worth crediting or comparing

### 4.1 Anthropic, "Claude's Character" (2024)
https://www.anthropic.com/research/claude-character, dated 2024-06-08 [V].

Character training was added to Claude 3 during alignment fine-tuning. Researchers write a trait list. Claude generates human messages relevant to each trait, writes several in-character responses, and ranks its own responses for trait fit. A preference model trained on those rankings is then used in training. The data are fully synthetic, though humans design and adjust the traits.

Not open: no weights, code or hyperparameters. Its significance for the repo is that it is the **original self-preference character recipe**. Computer's self-preference stages and Open Character Training's constitution-driven DPO both descend from this idea. In repo terms: trait list ≈ bible, synthetic messages ≈ scenarios, self-ranking ≈ a missing self-preference judge, PM + RL ≈ a missing RL stage.

### 4.2 Echo (Fulcrum, 2026)
https://echo.fulcrum.inc [V, product post; authors not named in the post. Uzay Girit is named by Character Garden [S]].

Built on **Kimi K3**.
- SFT on human-written internet posts, each conditioned on an outline a frontier model wrote for it. A voice metric and an LLM-judged coherence rubric were tracked during SFT; coherence dipped slightly as voice rose.
- Then token-level RL where the reward is the per-token log-likelihood ratio between an author-specific "expert" fine-tune (8 used) and the original model.
- Stylometric eval on trained-on and held-out authors.
- Under $5K to train. Open-sourcing has been announced; no licence yet.

Relevance: a **cheap dense reward without a judge**. For the repo, a "Golem expert" (the SFT adapter) versus the base model gives a log-ratio reward that could drive the missing RL arm, with the same reward-hacking caveats.

### 4.3 talkie-1930 (Levine, Duvenaud, Radford, 2026)
https://talkie-lm.com/introducing-talkie; https://github.com/talkie-lm/talkie; https://huggingface.co/talkie-lm/talkie-1930-13b-it [V]. Licence Apache 2.0 per Character Garden [S]; the post states none.

A 13B model pretrained from scratch on pre-1931 English. Post-training:
1. SFT on instruction pairs mined from period etiquette and letter-writing manuals, cookbooks, encyclopedias and similar texts.
2. **Online DPO** with Claude Sonnet 4.6 as judge.
3. SFT on rejection-sampled multi-turn chats between Claude Opus 4.6 and talkie.

Judge-rated instruction following on held-out prompts rose from 2.0 to 3.4 out of 5. The authors warn that AI-feedback RL pushes in anachronistic style (a 7B variant began speaking in listicles). This is directly relevant to the repo's worry about assistant register leaking through the judge.

Character Garden also lists "TALKIE (talkie-box)", an N8Programs derivative using SFT then KTO on the base [S, unverified].

### 4.4 OpenCharacter (Wang, Zhang, Ge, Yu, Yu, Yu, Tencent AI Lab; arXiv 2501.15427, 2025)
https://arxiv.org/abs/2501.15427 [V, abstract page].

Synthesises large-scale character profiles from Persona Hub, then builds character-aligned instruction responses by two strategies, response rewriting and response generation. SFT on **LLaMA-3 8B Instruct**. The authors report role-play performance comparable to GPT-4o. Synthetic characters and dialogues are released. The arXiv page licence is CC BY-NC-ND 4.0.

Relevance: the many-characters, instruct-start counterpoint to the repo's one-character, base-start design. Zero also uses Persona Hub, for partners.

### 4.5 Morpheus (J.D. Pressman, 2023)
https://huggingface.co/jdpressman/Mistral-Morpheus-7B [S via Character Garden; primary log not fetched].

LoRA on Mistral-7B base over "Morpheus" conversations, from the MiniHF / weave-loom tradition that Computer also draws on. An early base-model character LoRA. The data-generation method is **unverified** here.

### 4.6 Mr. Chatterbox and LittleLearner
See the §3 table. Both are from-scratch, corpus-defined characters rather than character training on a pretrained base. Cite them as alternatives to fine-tuning; they are not comparable recipes for character-pipeline.

### 4.7 Already in the repo's references (not re-researched)
- Persona Vectors (Chen et al. 2025)
- Constitutional grafting (Nutter et al. 2026; wassname/constitutional_grafting)
- Hermes 3 (Nous, 2024; an assistant-character recipe on Llama 3.1 base)
- Anthropic's persona selection model (alignment.anthropic.com/2026/psm), which is theory, not a recipe
- Lambert/Zick's Trillium "character training plan v1", a public Google Doc not read here [U]. Hand it to the Open Character Training reviewer.

---

## 5. Stages character-pipeline lacks (summary)

1. **Loom / branching generation** (Computer; Morpheus tradition). There is no tree generator or curation sheet.
2. **Simulated partner model and generator mixtures** (Zero; Computer's sundry-1). The repo's partners are scripted.
3. **Base-model teacher** (Computer via DeepSeek-V3-Base). The repo's teacher is Qwen3-30B-A3B chat; Zero's teachers are instruct models too.
4. **Online RL** (Zero GRPO/DAPO; Computer REINFORCE; talkie online DPO; Echo token-level RL). The repo has offline DPO and KTO only.
5. **Self-preference judge / self-distillation** (Claude's Character; Computer 6–10).
6. **Full-parameter training or full embedding/LM-head training** (Computer; Zero), and merge after training.
7. **Document-format (non-chat) training** (Computer).
8. **Long free-conversation hard-break eval with an instruct-prompted control** (Zero). The repo has pushback probes with a Base-prompted control.
9. **Capability suite** (Zero GSM8K/IFEval). Planned in Appendix E.

The repo has things none of these recipes report: human per-turn review as the data gate, matched insistence/evidence probes, a monitorability gate, three seeds per arm, and Wilson intervals.

---

## 6. Comparison table

| Recipe | Base vs instruct start | Data source | Objectives | Judge | Eval | Scale | Open code? | Licence |
|---|---|---|---|---|---|---|---|---|
| **character-pipeline (Golem-15)** | Base (Qwen3.5-4B-Base; 0.8B done, 8B/27B planned) | Qwen3-30B-A3B chat teacher, scripted partners, bible + graph, human-reviewed | LoRA SFT → DPO / KTO / continued SFT arms | d1-2B per-turn p_break (flags only); critics in eval | Pushback env: hold/drift/cave/update, Base-prompted control, 3 seeds | A few hundred reviewed chats (target); T4s | Yes | MIT (code) |
| **Zero** (Moving Castles) | Base (Qwen3-8B-Base) | 5,932 synthetic two-model chats; mixed API instruct generators; PersonaHub partners; bible + 117-node graph | LoRA SFT (+ full embeddings/head) → GRPO/DAPO LoRA, both merged | RL reward: Qwen3.6-27B fidelity + 4-gram penalty; eval: Claude Opus 4.8 | 250 × 16-turn held-out; hard-break 22.8% / 2.8% / 45.2% (SFT / SFT+RL / Instruct-prompted); GSM8K, IFEval; activation analysis | 8B; compute not reported | No code or data; weights only | Not stated |
| **Computer-10** (Celeste @parafactual [+ Luxia, per CG]) | Base (Llama 3.1 70B) | Gunkel *paths* seed → loom from DeepSeek-V3-Base, hand-curated documents; later self-generated conversations with user simulator sundry-1 | Full FT on documents → offline self-preference distillation → online REINFORCE self-preference RL (340 steps, 8 siblings, KL) → full SFT on 25,197 filtered self turns (1 epoch, lr 2e-6) | Frozen computer-7 (self-preference); outside model's broken-reply check | Broken rate 18.9% → 10.2%, unbalanced parentheses 21.6% → 2.1% on 2,000 held-out; no prompted control | 70B, full-parameter; compute not reported | No code; weights and checkpoints | Llama 3.1 Community |
| **Open Character Training** | (see `oct.md`) | Constitution-driven | DPO + introspective SFT | — | — | — | Yes (maiush/OpenCharacterTraining) | (see `oct.md`) |
| **Claude's Character** (Anthropic 2024) | Post-pretraining Claude 3 (during alignment FT) | Claude-generated messages and responses from a trait list | Self-ranking → preference model → RL | Claude itself | Not published | Frontier | No | Closed |
| **Echo** (Fulcrum) | Kimi K3 (variant not stated) | Human internet writing + frontier-model outlines | SFT → token-level RL on expert/base log-ratio | None (log-ratio from 8 author experts); LLM coherence rubric tracked | Stylometry on trained and held-out authors (numbers not extracted) | Frontier-size base; < $5K | Announced, not yet | Not stated |
| **talkie-1930** | From scratch (13B, pre-1931 corpus) | Period manuals → synthetic prompts → Claude Opus 4.6 chats | SFT → online DPO → rejection-sampled SFT | Claude Sonnet 4.6 | Judge instruction-following 2.0 → 3.4 / 5 | 13B, 260B tokens | Yes (github.com/talkie-lm/talkie) | Apache 2.0 [S] |
| **OpenCharacter** (Tencent, 2025) | Instruct (LLaMA-3 8B Instruct) | Persona Hub → synthetic profiles → rewritten or generated responses | SFT | — (eval judge not extracted) | Role-play comparable to GPT-4o (per abstract) | 8B | Data released | CC BY-NC-ND 4.0 (paper page) |
| **Morpheus** (Pressman, 2023) | Base (Mistral-7B) | Morpheus conversations (method unverified) | LoRA SFT | — | None listed | 7B | Weights | Not stated [S] |
| **Mr. Chatterbox** (Venturella) | From scratch (~340M) | 2.93B Victorian tokens + mined/synthetic dialogue | Pretrain → 2-epoch SFT | — | None listed | ~$497 [S] | nanochat-based [S] | Not stated |
