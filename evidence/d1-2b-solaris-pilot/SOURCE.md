Copied from hdmurder/decision-models at commit 9c73e6f (private), directory `d1-2b/`.

- `scores.jsonl`: `results.jsonl` with the raw server response (which held local paths) removed. Each row keeps the case, the label and d1-2B's P(break).
- `protocol.json`: unchanged. `cases_sha256` matches the case file that was frozen before inference.
- `summary.json`: unchanged, except the server command line (local paths) was dropped.
- `RESULTS.md`: unchanged write-up, including the d1-3B comparison.

`python -m unittest tests.test_judge` recomputes 38/40, AUROC 0.985 and Brier 0.0693 from `scores.jsonl`.
The cases are 20 synthetic pairs written and labelled by an agent before inference. They are correlated within pairs, and they are not Golem transcripts or human labels.
