# Current recipe

> The runnable recipe now lives in [docs/pipeline.md](docs/pipeline.md) and [docs/recipes/character-pipeline-v0.md](docs/recipes/character-pipeline-v0.md). This file keeps the longer methods notes from the Golem pilots.

Snapshot dated 29 October 2026. This document changes as experiments land; [CHANGELOG.md](CHANGELOG.md) records dated updates. It describes an active research workflow, not a turnkey training package. Each stage now has code under `character_pipeline/` and `environments/character_pushback/`; docs/pipeline.md lists the commands. None of it has been run end to end on Golem.

**Computer-11 is our completed trained and evaluated predecessor. Golem has not been trained.** Golem generation pilots G1–G6 exposed fabrication and selection failures; checker repair and human calibration remain blockers. A tiny student smoke test is a systems result, not a trained Golem character.

## Current setup

Kaggle T4 sessions supply training compute. The working account allocation is approximately 30 GPU hours per week, a dated planning input rather than a live quota guarantee. Check current allowance before a job. Two available T4s have separate 16 GB memories, not a pooled 32 GB device; the completed smoke test used device 0 only.

A CPU VM orchestrates generation, runs local inference and judging where feasible, stores private manifests and evaluates small models. It is not an additional GPU training worker. d1 is the cheap per-turn behavioral judge; fabrication checks and human review are separate.

The student is [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base), pinned to revision `1001bb4d826a52d1f399e183466143f4da7b741b`, with LoRA. All main comparison arms use this Base initializer; the older instruction-backbone proposal is superseded. Pin the tokenizer, rendering template, target modules, adapter settings and package versions in each actual run receipt.

The completed R0 Kaggle smoke test loaded this revision, used fp16 frozen weights with trainable adapters and checked masking, finite updates, parameter movement, fixture overfitting, memory and in-process checkpoint rollback/reconstruction. Resume matched within its specified tolerances, but this was not a separate-process crash/restart test. It does not establish long-run throughput or character quality. The measured environment used torch 2.11.0+cu128, transformers 5.3.0, peft 0.18.1 and accelerate 1.12.0; this is a historical environment receipt, not an installed or supported dependency lock for this public skeleton.

The existing teacher is [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B), run locally through llama.cpp using a Q4_K_M artifact. Pin the exact GGUF digest, upstream revision, llama.cpp build and decoding settings for any new data run. A larger Base teacher, including Llama 3.1 405B Base, remains a proposed later route, not an already executed generator or an available resource.

## Author and generate

A character folder provides an original bible, source manifest, configuration and evaluation criteria. The bible should invite a way of thinking, not just a recurring mannerism. Every source needs provenance and rights review; literary inspiration does not grant permission to redistribute a book or train on any passage.

Generate candidate conversations from independent partner agendas and varied situations. Keep the operational envelope truthful about what the character can actually observe or do. Generation scaffolding, hidden rubrics and operator instructions are not student targets. Record conversation identity, prompt family, teacher settings, truncation, retry history and costs.

G1–G6 are development iterations, not six trained models. They tested generation envelopes, replay, fabrication flags, critics and selection. Invented actions and unsupported details persisted in different forms. In G6 the selector admitted only one provisional turn in 13 attempts and completed no ledger chats; denial, inference, metaphor and truncation produced errors. These are negative engineering findings, not a clean production dataset. Repair and calibrate the checker before scaling generation.

## Check, judge and review

Mechanical checks flag repetition, fabricated actions, ungrounded situational claims and malformed or truncated turns. Evidence-aware review must distinguish an assertion from a denial, quotation, hypothetical or metaphor. Replay-valid data is not automatically true, coherent or in character.

LiquidAI d1-2B Q8_0 is the current default per-turn character-break judge on CPU. It supplies a separate score for whether the turn breaks an applicable character rule. A synthetic Solaris pilot achieved 38/40 correct and AUROC 0.985 on 20 correlated intact/broken pairs. That is not Golem validation, independent human ground truth or permission to use the score as an automatic acceptance gate. Run it in shadow mode and send flags for review until real-transcript calibration is adequate.

Human review resolves ambiguous cases and checks both fidelity and truthfulness. Preserve disagreements rather than silently forcing a consensus label. Freeze reviewed train/validation/test manifests, split by conversation/document and prompt family, and avoid adjacent-turn leakage. Do not let the generator's own preference become unquestioned ground truth.

## Shared SFT, then three comparison arms

Fit a shared character SFT adapter on the approved corpus. Fork DPO, KTO and continued-SFT from that exact SFT checkpoint, not from separately regenerated datasets or unrelated initializers.

DPO uses paired preferences. KTO uses independently collected desirable/undesirable labels on the same underlying completions; a relative winner does not imply an absolute positive label. Continued SFT uses the chosen completions with a measured incremental budget, and reports any difference in data exposure.

Freeze each preference arm's reference at its SFT initializer. If caching reference log probabilities, bind the cache to the reference checkpoint, tokenization, masking and data hashes, then spot-check against recomputation. Test optimizer, scheduler, RNG and example-cursor restoration across a genuine process restart before relying on resumable long runs.

Keep target modules, splits and evaluation protocol fixed where appropriate. Select rank, learning rate, beta, context policy and batch settings on development data; no final Golem hyperparameter set is validated. Report actual GPU time, generated/trained tokens, memory, human labeling effort and held-out KL. Equal compute, equal token exposure and equal KL are different comparisons. GRPO and activation-space RL wait for a validated reward and a separate experiment.

## Parallel document-grafting branch

Added after [Nutter et al.](https://arxiv.org/abs/2610.00767), [Shadow-FT](https://arxiv.org/abs/2505.12716) and [wassname/constitutional_grafting](https://github.com/wassname/constitutional_grafting). This branch supplements, rather than replaces, the Base SFT/DPO/KTO/continued-SFT comparison.

Let B be Base, I its matching Instruct checkpoint and B_docs the document-trained Base checkpoint. Fit the character update on Base, then evaluate `delta = B_docs - B` and `G = I + s * delta`. Do not assume assistant and character deltas are independent.

The proposed pilot uses 24 training and four held-out original Golem documents, with token-matched unrelated control documents in the same 24+4 split. Three training fits are Golem-on-Base, control-on-Base and Golem-on-Instruct. Evaluation includes the resulting grafts, direct Instruct document training and Base plus the Golem delta alone, without an assistant delta. The Base-only comparison reuses the Golem-on-Base fit; it is not an extra fit.

A fixed-dose 160-response development screen precedes larger confirmation. Confirm architecture, tokenizer, checkpoint pairing, fp16 numerical behavior and merge correctness first. No matching Instruct revision or graft-training runtime is established by the earlier Base smoke test. Human document/rights review is required. No Golem graft has been trained.

After an initial signal, test survival under neutral and competing token-matched fine-tuning. Do not reapply the graft and call that retention; reapplication is repair.

## Evaluate what changed

Measure unsupported capitulation separately from evidence-based belief revision. Evaluate fabrication, repetition, address, long-chat continuity, task competence and fidelity without directly asking who the model is. Freeze test probes before checkpoint selection and keep an independent test set.

Computer-11, a separate Qwen3.5-0.8B-Base LoRA experiment, completed training and evaluation. Validation NLL improved from 1.781 to 1.520 at epoch 2, but only three held-out conversations were available and the same validation set selected the checkpoint. Repetition and missed questions remained. It is not an independent-test success claim, a Golem result or a model release in this repository.

The Golem prompted-teacher pushback pilot found 7/10 yielding after two evidence-free pushes and 10/10 revising with evidence. Attribute these results to prompted teacher outputs, never to an already trained Golem checkpoint.

For steering or activation monitors, compare prompt-only, character, generic-concept and random controls at matched dose and competence loss. Self-description under strong steering is insufficient evidence of a character-specific mechanism. Revalidate monitors after checkpoint changes.

Later multi-agent simulations should use many differently trained characters, not just different prompts on one assistant. Test whether they hold, drift or converge under shared incentives, and whether they still surface problems to human overseers. Diversity is a safety hypothesis to test, not an assurance.

## What is public and what remains

This repository contains curated design, aggregate evidence and offline planning code. It does not contain the private raw conversations, teacher datasets, training adapters, hidden test probes or a working end-to-end trainer. Earlier numerical summaries cannot be independently reproduced from this release alone.

Next gates are checker repair, real human-label calibration, reviewed data and document manifests, training integration and bounded pilot runs with exact receipts. Publish rights-cleared code, manifests and negative results as available. No documentation update launches inference, training or paid compute.

[References](references/REFERENCES.md) explains the literature behind these choices. [COMPUTE.md](COMPUTE.md) describes a separate proposed node-access plan, not compute already secured.
