# Computer-11: reported, receipts not yet in this repository

Computer-11 is the completed trained-and-evaluated predecessor to Golem: Qwen3.5-0.8B-Base with LoRA SFT, completed 2026-10-07 (CHANGELOG.md).

| | Validation NLL |
| --- | --- |
| Base | 1.781 |
| Best epoch (2) | 1.520 |

Limits, as recorded in RECIPE.md: only three held-out conversations; the same validation set selected the checkpoint, so there is no independent test; repetition and missed questions remained.

**These numbers are not yet traceable to a file.** The run lives in local Computer-11 code that has not been pushed. Its training data is third-party (computer-10 conversations collected with permission) and will not be published here. To make the figures checkable without the data, add the run's `run_config.json`, `training_summary.json`, per-epoch validation losses and the three held-out eval transcripts' scores to this folder.
