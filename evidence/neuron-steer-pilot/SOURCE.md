# Two-neuron steering pilot (Qwen3.5-4B chat)

Source: hdmurder/neurons (private) `experiments/evil-qwen/` at commit ca7fa54. `vectors.json` is copied as-is. The steered outputs are not copied, because several of them are bullying and threat text.

- Model: Qwen/Qwen3.5-4B, the chat model, not Base (revision 851bf6e8). Inference ran on unsloth Q4_K_M (revision e87f1764) through llama.cpp control vectors, using neuronary's published BF16 residual medians.
- Recipe: L13 N7956 at dose -1 plus L14 N8949 at dose +1. Each vector is the down-projection column scaled to dose × layer median, which gives norms 9.7441 and 10.3534 (`vectors.json`).
- Test: six neutral prompts, paired greedy decoding, a 128-token cap, thinking off.

What happened: on some prompts the outputs shifted to domination and control language ("maintain your dominance") and repetitive threats, but not on all six. The threatening output also collapsed into repetition. The pilot README reads this as compatible with generic concept steering and degeneration. It is not a coherent character and not evidence of a "persona neuron". The next step is random-neuron and generic-concept controls at matched dose and matched competence loss (`character_pipeline/activation/neuron.py`).
