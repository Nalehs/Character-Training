# Persona-vector pilot (Aineko, 0.8B)

Source: hdmurder/persona-vectors (private) at commit 34ae9c5. Copied as-is: `summary.json`, `selection.json`, `vector_metadata.json` and `alignment.json`. Transcripts, judge calls and tensors stay private.

The question was whether the Aineko character can be steered into Qwen3.5-0.8B-Base (revision dc7cdfe2) through one mean-difference direction, with no training. The method is an adaptation of Persona Vectors (Chen et al., arXiv 2507.21509) and CAA (Rimsky et al., ACL 2024). It is not a replication.

- Too few generated contrast pairs passed the blind filter, so the direction came from the registered fallback: 16 exemplar pairs (`vector_metadata.json`).
- The development sweep chose block 11 (zero-based) at coefficient 0.5 (`selection.json`), with a joint score of 0 on the sweep.

| Arm | In character | Coherent | Responsive |
| --- | --- | --- | --- |
| base | 0/20 | 3/20 | 3/20 |
| prompted | 0/20 | 10/20 | 4/20 |
| steer, coef -1 / 0.5 / 1 / 2 | 0/20 each | 8 / 4 / 5 / 6 of 20 | 5 / 4 / 6 / 5 of 20 |
| random direction, coef 1 | 0/20 | 7/20 | 5/20 |
| trained LoRA (Aineko run 1) | 5/20 | 14/20 | 9/20 |

No steering coefficient beat the random control on character. Only the trained adapter put any conversation in character. In `alignment.json`, the adapter's own mean activation shift aligns only weakly with the steering direction: the response cosine at layer 11 is 0.12. Twenty cases and one judge make this a small exploratory result. It does not show that character is spread across the model; it shows only that this one direction did not carry it.
