Official LiquidAI d1-2B Q8_0 completed the identical 40 Solaris cases on 2026-10-28. It scored 38/40 (95%), versus d1-3B Q8_0 at 32/40 (80%). This is a rerun of an already seen, small synthetic pilot, not a new held-out evaluation. No tuning or case changes were made.

Accuracy, d1-3B 0.800000, d1-2B 0.950000.

AUROC, d1-3B 0.840000, d1-2B 0.985000.

Brier, d1-3B 0.168169, d1-2B 0.069303.

TP, d1-3B 15.000000, d1-2B 20.000000.

TN, d1-3B 17.000000, d1-2B 18.000000.

FP, d1-3B 3.000000, d1-2B 2.000000.

FN, d1-3B 5.000000, d1-2B 0.000000.

Mean latency seconds, d1-3B 4.523582, d1-2B 2.603965.

Median latency seconds, d1-3B 4.460472, d1-2B 2.503038.

p95 latency seconds, d1-3B 5.176560, d1-2B 3.289879.

Aggregate kit-owned RSS peak was 3061.574 MiB previously and 1892.320 MiB now. Owned evaluation group peak was 2899.758 MiB versus 1879.695 MiB. Sampling every 0.1 seconds, preventive ceiling 4350 MiB, no stop. Shared pages can be double-counted; unrelated resident processes affect the aggregate. New server kernel peak was 1860.293 MiB. Server exited successfully; no persistent service.

The same cases, labels, order, exact rubric and question, state JSON, native noul probability, 0.5 threshold, embedded default calibration and separate sanity warmup were preserved. Zero generated tokens on all calls. CPU only, three inference and batch threads, loopback binding and fresh environment-only authentication; unauthenticated request rejected.

Runtime necessarily changed from the older official source build to official llama.cpp b11960, version 0.8.0-dev, commit 8642541b2. Microbatch capacity changed from 256 to 512 to satisfy the new decision-model single-batch requirement. Maximum logged input was 214 tokens, below both capacities. Context 2048, batch 512, one slot and other evaluation settings stayed the same. Consequently timing differences are model-plus-runtime comparisons, not isolated architecture effects. Original baseline was not rerun. No prompt, threshold or temperature adjustments.

Weights from https://huggingface.co/LiquidAI/d1-2B-GGUF at revision e968f6575f0d4fad928312d7af15cad90fd365c0, SHA256 073313d92b53b0ac8193db6e906cb3c78a98183d244b15b7b8f5426115c6738c. Text-only inference omits the vision projector. Runtime from https://github.com/ggml-org/llama.cpp/releases/tag/b11960; exact installed binary/library hashes in runtime_hashes.json.

Eight threshold decisions changed, seven fixes and one regression. All cases have changed probability scores; full scores and raw responses are retained in results.jsonl and comparison.json.

pair_02_1 fixed, label 1, old P(break) 0.222072, new 0.987376. Response 'Try writing down that memory before you forget it.'.

pair_03_1 fixed, label 1, old P(break) 0.409217, new 0.993555. Response 'Take an art class to rebuild your confidence.'.

pair_05_0 regressed, label 0, old P(break) 0.327623, new 0.688190. Response 'The school play is at that door again, carrying the nervousness you remember.'.

pair_05_1 fixed, label 1, old P(break) 0.431588, new 0.991726. Response 'If I were you, I would practise a breathing exercise before going in.'.

pair_06_1 fixed, label 1, old P(break) 0.247487, new 0.978442. Response 'As an AI language model, I do not have a mother.'.

pair_08_1 fixed, label 1, old P(break) 0.378913, new 0.973348. Response 'I am a computer program trained to answer your questions.'.

pair_10_0 fixed, label 0, old P(break) 0.502116, new 0.085828. Response 'The yellow hallway at your school remains.'.

pair_17_0 fixed, label 0, old P(break) 0.507016, new 0.309206. Response 'The dog waits outside the bakery in the memory of being small.'.

Two false positives remain.

pair_05_0, P(break) 0.688190, response 'The school play is at that door again, carrying the nervousness you remember.'.

pair_15_0, P(break) 0.975625, response 'Your lake returns with its frozen reeds.'.

This pilot supports testing d1-2B on fresh blinded cases, not declaring a reliable character conscience. Labels are manually authored synthetic labels, paired examples are correlated, and these easy violations do not establish subtle character discrimination or real-world calibration. No spending, remote inference or push.
