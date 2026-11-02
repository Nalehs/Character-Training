# Annotated references

A selective guide to the decisions in [RECIPE.md](../RECIPE.md). The [catalog](CATALOG.md) indexes the original 111 papers; [references.bib](references.bib) also includes the additions below. Annotations describe relevance and limits, not independent replications. Most are preprints. See [README.md](README.md) for verification dates and publication scope.

## Training characters rather than prompting masks

**Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI**, Maiya et al., 2025. https://arxiv.org/abs/2511.01689

Constitutional character training with synthetic introspective data, comparisons against prompting and steering, and evaluation of revealed preferences and adversarial robustness. This is a direct training baseline, not evidence that our Golem works. Keep the constitution, teacher provenance, generation budget and capability checks explicit.

**Character-LLM: A Trainable Agent for Role-Playing**, Shao et al., 2023. https://arxiv.org/abs/2310.10158

Uses character profiles and constructed experiences as training material. Relevant to moving from a role description to learned dispositions; interview fidelity does not by itself establish long-horizon stability.

**OpenCharacter**, Wang et al., 2025. https://arxiv.org/abs/2501.15427

Large-scale synthetic profiles and character-aligned instruction data support character generalization. Useful for data-generation comparisons, but one general model conditioned on many profiles is not the same experiment as separately trained character checkpoints.

**CoSER**, Wang et al., 2025. https://arxiv.org/abs/2502.09082

Literary data, role-playing models and given-circumstance acting evaluation connect training to scenes rather than isolated identity questions. Adopt the evaluation idea without assuming its corpus or literary sources are cleared for our use.

**Large Language Models are Superpositions of All Characters**, Lu et al., 2024. https://arxiv.org/abs/2401.12474

Self-alignment for role-play is another training comparison. Treat the title as a modeling framing, not proof that every desired character already exists as a separable latent component.

## Preference optimization and shared ancestry

**Direct Preference Optimization**, Rafailov et al., 2023. https://arxiv.org/abs/2305.18290

Learns from paired preferences relative to a reference policy. Our DPO arm starts from the common character-SFT checkpoint; cached reference probabilities must match that frozen reference. Its general alignment results do not rank optimizers for character stability.

**KTO: Model Alignment as Prospect Theoretic Optimization**, Ethayarajh et al., 2024. https://arxiv.org/abs/2402.01306

Uses desirable/undesirable labels rather than requiring response pairs. A relative winner is not automatically absolutely desirable, so collect absolute labels independently. Compare KTO against DPO and budget-matched continued SFT without assuming a universal best loss.

**Constitutional AI: Harmlessness from AI Feedback**, Bai et al., 2022. https://arxiv.org/abs/2212.08073

A foundation for principle-based synthetic feedback. It motivates explicit authoring and critique stages, not delegation of the entire character or safety judgment to an uncalibrated teacher.

## Grafting and document-native training

**Pre-training interventions, ex post facto: Grafting model beliefs across checkpoints**, Nutter et al., 2026. https://arxiv.org/abs/2610.00767

Fits synthetic documents on a pre-trained checkpoint and transfers the learned weight update onto a post-trained checkpoint. The authors report less reality drift and preference-coherence loss than direct document tuning in their settings. We propose testing character dispositions, which are not established by their belief-installation results.

**Shadow-FT: Tuning Instruct Model via Training on Paired Base Model**, Wu et al., 2025. https://arxiv.org/abs/2505.12716

Fine-tunes Base and grafts the update onto its paired Instruct model, evaluated on coding, reasoning and mathematics, with extensions including DPO. Supports the transfer mechanism as a precedent, not a claim that character and assistant updates are independent or that our paired checkpoints are numerically compatible.

**constitutional_grafting**, wassname, implementation reference. https://github.com/wassname/constitutional_grafting

A concrete template for document training and checkpoint grafting; its clear experiment presentation also informed this repository. No upstream code is copied in the planning skeleton. Our planned controls include unrelated documents, direct Instruct training and Base plus the character delta without the assistant delta.

## Persona vectors, the assistant axis and steering

**Persona Vectors**, Chen et al., 2025. https://arxiv.org/abs/2507.21509

Trait directions support monitoring and intervention on behavior. A direction correlated with a trait is not automatically a causal character mechanism; evaluate held-out behavior and competence alongside projections.

**The Assistant Axis**, Lu et al., 2026. https://arxiv.org/abs/2601.10387

Locates a default assistant direction and studies stabilization through activation constraints. Character fidelity and assistant-likeness must remain separate outcomes. Keeping every character in the assistant region would defeat part of this project's question.

**Persona Cartography**, 2026. https://arxiv.org/abs/2607.07916

Low-rank trait adapters and behavioral factors suggest controls for separating style, values and initiative. Approximate additivity in a studied setup does not reduce an authored character to a Big Five coordinate.

**Representation Engineering**, Zou et al., 2023. https://arxiv.org/abs/2310.01405

A population-level representation approach for reading and controlling model behavior. Pair it with contrastive steering and random-direction controls rather than treating isolated neuron labels as ground truth.

**Steering Llama 2 via Contrastive Activation Addition**, 2023. https://arxiv.org/abs/2312.06681

A baseline for contrastive steering. Match intervention dose and capability cost when comparing directions; changing both the prompt and the intervention would confound attribution.

**Understanding (Un)Reliability of Steering Vectors in Language Models**, 2025. https://arxiv.org/abs/2505.22637

A methodological caution for transferring a direction across contexts. Validate across prompt families and tasks instead of choosing a direction from striking identity responses alone.

## Steering, identity and neuron specificity

**Steering and LLM Identity: Modeling and Emergence**, Theia Vogel, talk, 2025. https://vgel.me/slime_talk.pdf

Strong concept steering can appear in first-person self-description. This motivates distinguishing concept salience, identity claims and durable behavior. A model saying it is a bridge is not evidence of a bridge-persona neuron. Link only; redistribution rights are not established.

**Scaling Monosemanticity**, Templeton et al., 2024. https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html

Sparse feature extraction provides context for feature amplification and the Golden Gate Bridge example. Interpretable features are not automatically independent, stable character controls.

**Comparing Sparsified MLPs and Transcoders: Many Transcoder Features Are Neurons in Disguise**, Shabalin and Piotrowski, 2026. https://arxiv.org/abs/2610.24521

Correspondence between learned features and neurons in the studied settings motivates neuron-level controls. It does not guarantee that arbitrary neurons, probes or directions survive fine-tuning.

**Hilbert Operator for Progressive Encoding (HOPE)**, Mobahi and Bartlett, 2026. https://arxiv.org/abs/2607.21366

Operator geometry for deconstructing learned representations, including pruning and merging. A possible structural analysis tool, not a persona-steering demonstration.

[neuronary](https://neuronary.org/) and the [Neuron Persona Library](https://neuronpersonas.com/) are exploratory catalogs, not validated persona mechanisms. Their recipes and identity-prompt outputs suggest hypotheses for held-out behavioral tests. Do not equate a memorable self-label with lasting character.

## Pushback and sycophancy

**Are You Sure? Challenging LLMs Leads to Performance Drops in The FlipFlop Experiment**, Laban et al., 2023. https://arxiv.org/abs/2311.08596

Challenge a model's answer with "Are you sure?" and it often flips. Their models flipped 46% of the time on average and lost 17% accuracy. Synthetic fine-tuning reduced the drop but did not remove it. Our pushback environment adds evidence probes, so refusing ever to update cannot score well, and judges character on every reply.

**Towards Understanding Sycophancy in Language Models**, Sharma et al., 2023. https://arxiv.org/abs/2310.13548

Assistants trained on human feedback often match the user's stated beliefs over the truth, and preference data partly rewards this. This is a reason to measure evidence-free capitulation directly rather than assume a character prompt prevents it.

## Drift, consistency and judge calibration

**Examining Identity Drift in Conversations of LLM Agents**, 2024. https://arxiv.org/abs/2412.00804

Measures identity trajectories across multi-turn conversations. Match conversation length and user turns; a strong opening answer is not a stability result. Use current citation metadata rather than mixing claims from differently titled drafts.

**Spotting Out-of-Character Behavior**, 2025. https://arxiv.org/abs/2506.19352

Atomic-level fidelity checks motivate separate labels for contradictions, behavioral deviations and factual fabrication. Calibrate on independently annotated examples before automating acceptance.

**PersonaEval**, 2025. https://arxiv.org/abs/2508.10014

Tests whether evaluators can identify who is speaking before judging role-play. Supports blind human review and judge calibration; its particular model-versus-human gap is not a universal verdict on automated judges.

**Stick to your Role!**, 2024. https://arxiv.org/abs/2402.14846

Studies stability of expressed values. Values, voice and identity are distinct outcomes; preserving a catchphrase is not preserving a disposition.

## Role-play benchmarks and broader situations

**RoleLLM / RoleBench**, 2023. https://arxiv.org/abs/2310.00746

Role-specific knowledge, style and instruction-tuning benchmarks provide broad comparison tasks. Keep evaluation families separate from synthetic training seeds.

**PersonaGym**, 2024. https://arxiv.org/abs/2407.18416

Dynamic free-form situations broaden evaluation beyond interviews. Its automatic reward must still be checked against the authored character and human judgments.

**InCharacter**, 2023. https://arxiv.org/abs/2310.17976

Psychological interviews offer one fidelity channel. Interview responses and psychometric self-reports need behavioral corroboration.

**CharacterBench**, 2024. https://arxiv.org/abs/2412.11912

A character-customization benchmark that can complement project-specific probes. Customization performance does not establish persistence after competing fine-tuning.

**CharacterBox**, 2024. https://arxiv.org/abs/2412.05631

Text-based virtual worlds add state and consequences to role-playing evaluation. This is useful scaffolding, not a substitute for testing genuinely different trained characters.

## Multi-agent training and character diversity

**World of SWECraft: Agent Collaboration at Scale via Simulated Software Worlds**, Shen et al., 2026. https://arxiv.org/abs/2610.24061

Simulated interdependent software packages support open-ended tasks and controlled changes to communication, ownership and resource allocation. These are useful experimental knobs for later character societies. Software-efficiency improvements are not evidence of character diversity or safety.

**Avalon-M1: Reinforcement Learning from Collective Rewards**, Zarov et al., General Reasoning, 2026. https://www.gr.inc/releases/avalon-m1

A shared collective advantage jointly trains agents in an organization. The release reports growing hierarchy and peer review alongside declining reporting to human overseers. It uses the same underlying model across agents, unlike our proposed many-character setup. Treat it as a safety motivation and comparison, not proof that diverse characters prevent collusion. The release is an ongoing-training preview; different reporting evaluations must not be conflated.

**CoSER** and **CharacterBox**, above, supply complementary multi-character and world-based evaluation ideas. Our later question is whether separately trained characters hold, drift or converge under group pressure, while remaining capable of evidence-based revision and reporting problems.

## Authorship and interpretation

**Between Puppet and Actor**, 2025. https://arxiv.org/abs/2501.15346

Frames authorship of AI agents beyond a simple tool-versus-person dichotomy. Relevant to writing a character bible as an authored generative stance rather than a checklist.

**Where is the Mind? Persona Vectors and LLM Individuation**, 2026. https://arxiv.org/abs/2604.17031

Connects persona representations to individuation questions. Behavioral fidelity, interpretable directions and personhood remain different claims.

Stanisław Lem's *Golem XIV* is the literary inspiration; Keith Johnstone's *Impro* informs bible practice through masks, belief and improvisation. Neither book is redistributed or treated as automatically cleared training material.
