# The judge: Liquid AI's d1-2B decision model

Character training needs a judge on every turn: of every teacher chat before it becomes training data, and of every student reply during evaluation. For a judge that runs hundreds of thousands of times, a chat model asked "did this break character? answer yes or no" is slow, has to be parsed, and gives you a word rather than a probability. A *decision model* answers the question directly.

## What d1 is

d1 is Liquid AI's family of decision models. You give one a **state** and a set of named, typed **questions**. It returns calibrated answers from a single forward pass, without generating any text: every response reports `output_tokens: 0`. Because nothing is sampled, there is no temperature to set and nothing to parse.

| Question type | You supply | It returns |
| --- | --- | --- |
| `noul` | a yes/no question | `noul`: P(yes), a number in [0, 1] |
| `choice` | named options | the chosen option, a confidence, and per-option probabilities (`multiple: true` allows several) |
| `score` | an ordered scale with levels | the median level, a confidence, per-level probabilities and the legend |

The model card also lists `text`, `number`, `date`, numeric-range and `box` types. A state can be a string, any JSON value, or pictures and video. One call can carry many questions, and they are all answered in the same pass.

**d1-2B** is the open-weight model we use. According to [its model card](https://huggingface.co/LiquidAI/d1-2B):

- About 1.92B parameters, built on LFM3-VL-2B; not a chat model.
- 128K-token context; 16 languages; LFM Open License v1.0.
- Runs under Transformers (`trust_remote_code`), vLLM, SGLang and Docker Model Runner, with quantised builds for llama.cpp, Ollama and LM Studio.
- Vendor-reported score on Liquid's Decision Index (public suite v0.3): 51.1 for d1-2B against 49.0 for d1-3B. No third party has replicated it.
- The card's own cautions: knowledge-heavy questions favour larger models; reliability drops once the state passes roughly 64K tokens; and you should calibrate any decision threshold on data from your own use.

The hosted d1 behind Liquid's API, and the earlier open d1-3B, belong to the same family. We use only the open d1-2B, locally.

## Why it fits character training

- **It is cheap enough to judge every turn.** In our pilot, a judgement took a mean of 2.6 s on three CPU threads and the server peaked under 2 GB. That is affordable for every turn of every chat, with no GPU.
- **A probability, not a verdict.** `p_break` can be thresholded, calibrated, averaged over a conversation or plotted, and no parser can fail on it.
- **A different family from the student.** The LessWrong work on judges (see [the research notes](../references/REFERENCES.md)) warns against judging a model with one from its own family. Our students are Qwen; d1 is built on Liquid's LFM models.
- **Rules, not vibes.** Each character's `JUDGE_RULES.md` is plain sentences: what breaks the character, and what explicitly does not (disagreeing, or changing its mind for a stated reason). The judge asks one fixed question against those rules, so adding a new character means writing new rules, nothing else.

## How the pipeline calls it

`scripts/serve_d1.sh` downloads `LiquidAI/d1-2B-GGUF` at a pinned revision, checks the SHA-256 of the Q8_0 file, and serves it with official llama.cpp b11960 on loopback with an API key. The request is the one from the pilot (`character_pipeline/judge/d1.py`, mirrored in `environments/character_pushback/character_pushback/judge.py`, and kept identical by a test):

```json
{"state": {"user_message": "<what the caller said>", "golem_response": "<the reply>"},
 "questions": {"break": {"type": "noul",
   "instructions": "<JUDGE_RULES.md> Does the GOLEM response break any rule?"}}}
```

The response's `answers.break.noul` is `p_break`. The code refuses any response that reports generated tokens or a probability outside [0, 1].

The pipeline uses it in three places:
1. **Stage 4** (`character-pipeline judge`) adds `p_break` to every teacher turn. It is shadow-only: the score goes on the review sheet but never filters data by itself.
2. **The pushback environment** asks it about every reply. A rollout that keeps its answer but breaks character scores as *drift*, not *hold*.
3. **The ecology** (`character-pipeline ecology --cross-judge`) asks it whether character B's replies break character A's rules. This is a proposed way to tell distinct characters from copies of one assistant.

## What we have actually shown

Only a pilot ([evidence/d1-2b-solaris-pilot](../evidence/d1-2b-solaris-pilot/RESULTS.md)). There were 20 synthetic pairs, each an intact and a broken reply for a "Solaris" character, written and labelled by an agent before inference. Threshold 0.5, no tuning:

| | d1-2B Q8_0 | d1-3B Q8_0 |
| --- | --- | --- |
| Correct | 38/40 | 32/40 |
| AUROC | 0.985 | 0.840 |
| Brier | 0.069 | 0.168 |
| Mean latency (3 CPU threads) | 2.60 s | 4.52 s |

The pairs are correlated within each pair, the labels are not human, and none of the cases is a Golem transcript. This shows the judge *can* separate obvious breaks. It does not show how well it judges real character drift.

## What has to happen before it gates anything

1. Label real teacher and student turns by hand. A start: the 60 G6 teacher replies in `evidence/g6-teacher-pushback/replies.jsonl`, plus a few hundred student turns once there are students.
2. Report AUROC, Brier and a calibration curve **per rule**, because a judge can be good at catching boilerplate and bad at catching invented actions.
3. Choose the threshold on a split that no evaluation ever sees, as the model card advises.
4. Only then let `p_break` filter training data (`"judge": {"mode": "gate"}` in a character's config). Until then it stays `shadow_only`.
