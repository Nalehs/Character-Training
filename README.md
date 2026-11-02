# Character Pipeline

A plug-in research pipeline for training distinct AI characters.

Golem-15, inspired by Stanisław Lem's *Golem XIV*, is the first character. A new character supplies a folder: a bible, rights-cleared source documents, a config and evaluation criteria. The training and evaluation stages stay the same. Multi-agent character simulations come later.

This repository publishes the design, curated evidence and a runnable planning skeleton, not an end-to-end trainer or model release.

```text
character folder
  bible + source documents + config
                 |
          teacher generation
                 |
     fabrication / repetition checks
                 |
    per-turn d1 character-break score
       (shadow-only until calibrated)
                 |
         human-reviewed dataset
                 |
       shared Base + LoRA SFT
          /       |        \
        DPO      KTO    continued SFT
          \       |        /
             held-out evals
                 |
       larger models, more characters
                 |
       multi-agent simulations (later)

parallel experiment
  documents -> Base + document delta
                   |
          Instruct + scaled delta
                   |
   evaluation + competing-finetune survival
```

## Stack, in one line

Qwen3-30B-A3B teacher / later Llama 3.1 405B Base · fabrication checker · d1-2B Q8_0 · Qwen3.5-4B-Base + LoRA · SFT / DPO / KTO / continued-SFT · document grafting · behavioral and activation evals.

These are research choices, not a pinned, installed training environment. This release runs offline on Python 3.10+ using only the standard library.

[Current recipe](RECIPE.md) · [Annotated references](references/REFERENCES.md) · [Full literature catalog](references/CATALOG.md) · [BibTeX](references/references.bib) · [Changelog](CHANGELOG.md)

## Try the planning skeleton

```sh
python3 scripts/plan.py characters/golem-15
python3 scripts/estimate_compute.py
python3 -m unittest discover -s tests -v
```

Nothing here downloads weights, generates data, starts training or spends money. Trainer and inference integrations are not implemented. Copy `characters/_template` to start a new character; edit the config, bible and document manifest. Do not commit unreviewed source material.

## How it works

The bible invites a way of thinking rather than enumerating catchphrases. Keith Johnstone's *Impro*, especially masks and trance, informs this approach. The character should be able to wander, disagree and revise a belief without falling back into a generic assistant.

Teacher generation explores conversations and document continuations. Mechanical checks flag repetition, fabricated actions and ungrounded claims; they cannot establish truth by themselves. Ambiguous cases go to human review. The d1 judge supplies a separate probability of a character-rule break on each turn, not an unquestioned accept/reject label.

The main training route starts with shared supervised fine-tuning on Qwen3.5-4B-Base. Three arms start from that same SFT checkpoint: DPO, KTO and continued SFT. Freeze each preference arm's reference at its initializer. Keep token budgets and data splits explicit; test resume state and cached reference probabilities before long runs.

A parallel grafting route fits documents on Base, then applies the resulting weight delta to the matching Instruct checkpoint. In shorthand, `delta = B_docs - B`, `G = I + s * delta`. It may also bring unwanted behavior with it. Compare against unrelated documents, direct Instruct document training and the Base-plus-character checkpoint without the assistant delta. Do not assume character and assistant deltas are independent.

Evaluation separates yielding to unsupported insistence from updating in response to good evidence. It also checks fabrication, repetition, address, ordinary task competence and character continuity without explicit identity questions.

## Evidence and current status

| Component | Status | What the evidence actually says |
| --- | --- | --- |
| Computer-11 predecessor | Trained and evaluated | Qwen3.5-0.8B-Base LoRA: base validation NLL 1.781; best epoch 2, 1.520. Only three held-out conversations; the same validation set selected the checkpoint. No independent test claim. Repetition and missed questions remained. |
| Golem prompted teacher | Pilot complete | In a small pushback pilot, 7/10 caved after two evidence-free pushes; 10/10 updated with evidence. These are prompted teacher outputs, **not a trained Golem model**. |
| Fabrication / repetition checker | Prototype, not production-ready | A later selector admitted only one provisional turn in 13 attempts and completed no ledger chats. Denials, metaphor and truncation caused problems. Human calibration is a blocker. |
| d1-2B Q8_0 judge | Synthetic pilot only | 38/40 correct; AUROC 0.985 on 20 paired intact/broken Solaris examples. Synthetic, correlated pairs; not independent human labels, not Golem validation. Shadow-only pending real-transcript calibration. |
| Golem SFT / DPO / KTO / continued SFT | Planned | No Golem adapters, training outcomes or model weights released. |
| Grafting | Planned | Small document-controlled experiment specified; no Golem graft trained. |
| Neuron steering | Exploratory pilot / controlled baseline planned | A small unrelated steering check produced control/dominance language and repetition. No demonstrated Golem-specific mechanism. |
| Activation monitoring | Planned | Test whether probes survive checkpoint changes and adversarial pressure; no robust internal character detector established. |
| 8B / 27B and multi-agent sims | Later | Advance only after a smaller controlled result. Exact larger checkpoint families remain to be selected. |

The numerical summaries above come from earlier private experiments. Raw conversations and third-party datasets are deliberately not redistributed, so these summaries are not independently reproducible from this release. Future rights-cleared runs should ship manifests, code, model revisions and negative results.

## Data and experimental controls

`characters/golem-15` contains an original short bible excerpt and an empty source manifest, not Lem's book or a synthetic training corpus. All new documents require provenance and rights review. Split by document/conversation and prompt family, not adjacent turns. Freeze held-out probes before model selection.

The proposed graft pilot uses 24 training and four held-out Golem documents, matched by token count to 24+4 unrelated control documents. Start with three fits: Golem-on-Base, control-on-Base and Golem-on-Instruct. A fixed-dose 160-response development screen precedes any larger confirmation. Test competing fine-tuning without reapplying the graft; reapplication is repair, not retention.

For steering, compare the prompt baseline with character, generic-concept and random directions at matched doses and competence loss. Thebes's absorption-into-self critique matters: a model calling a perturbed concept “me” is not evidence of a specific persona neuron. Measure behavior on non-identity tasks. A constant down-projection-column offset is not the same intervention as patching natural neuron activations.

## Compute and next steps

[COMPUTE.md](COMPUTE.md) gives a 4–6 week node-access plan, separate GPU/Mac budgets, hardware uncertainties and a reproducible estimate. [ROADMAP.md](ROADMAP.md) describes Golem first, many characters next, multi-agent simulations later.

## Credits and license

The document-grafting branch builds on and cites [wassname/constitutional_grafting](https://github.com/wassname/constitutional_grafting), whose clear experiment presentation also informed this repository, and Peter Nutter, Dani Roytburg, Clement Dumas, Jinghua Ou and Shi Feng, [*Pre-training interventions, ex post facto: Grafting model beliefs across checkpoints*](https://arxiv.org/abs/2610.00767), 2026. No upstream code is copied in this skeleton.

Character inspiration: Stanisław Lem, *Golem XIV*. Bible practice: Keith Johnstone, *Impro*. Model references: [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base), [d1-2B](https://huggingface.co/LiquidAI/d1-2B).

Original code and writing are MIT licensed. That license does not relicense model weights, external datasets, books or other cited material. No third-party character corpus is included.
