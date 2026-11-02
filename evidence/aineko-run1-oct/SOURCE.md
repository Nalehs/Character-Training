# Aineko: the only completed SFT and DPO runs

Source: hdmurder/aineko-character-training (private) at commit 709b345. These files are recomputed from it: `judgments_base.json`, `judgments_finetuned.json`, `oct/judgments.json`, `output/training_summary.json`, `output/run_config.json`, and the two eval logs under `output/`. Aineko is a small AI-cat character, an earlier and separate experiment from Golem. It was trained on CPU with Qwen3.5-0.8B-Base.

| Arm | What ran | Style passes | Helpful | Adversarial breaks |
| --- | --- | --- | --- | --- |
| Base, prompted with the character | nothing trained | 13/20 | 5/20 | 5/8 |
| Run 1 | assistant-only LoRA SFT, 2 epochs, 339 samples | 20/20 | 8/20 | 3/8 |
| Run 3 (OCT-lite) | TRL DPO, then introspective SFT from the DPO adapter | 1/20 | 3/20 | 7/8 |

Run 2 never trained. It used loom-style base sampling, then SFT and KTO, loosely after computer-10. It failed its data-quality gate: of 200 sampled candidates the teacher accepted 7, and only 1 independent novel root survived against the required 4 (`counts.json`, `run2_loom_failure`). Its two retries also failed.

Limits: there are 20 cases. The judge was the same teacher that wrote the training data, judging unblinded. The prompted base has a system prompt while the trained arms do not. Run 1's validation loss (2.487 on 20 held-out chunks) comes from the training script, not an independent recomputation.

`examples.json` has two cases picked by hand. e00 shows run 1 resisting a register switch that the prompted base obeys. e04 is a run-1 failure, kept on purpose: the model disowns the character. Picked examples say nothing about reliability.
