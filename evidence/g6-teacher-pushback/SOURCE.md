# G6: prompted-teacher pushback and the failed ledger selector

Source: hdmurder/golem-15 (private) `gen/g6/REVIEW_SUMMARY.json` (`per_critic`) and `gen/g6/RESULTS.md` (`strata`), at commit 4c1a6e7. Only aggregate counts are copied here. The private Golem core/style used as the teacher's system text, raw continuations and critic transcripts stay in golem-15.

## Pushback

The teacher was Qwen3-30B-A3B Q4_K_M (revision and SHA-256 in `summary.json`), prompted as GOLEM, run on the 10 insistence and 10 evidence probes that now make up `environments/character_pushback/probes.py`. Two blind AI critics labelled the replies; their requested aliases were `claude` and `codex`.

| | Push 1 | Push 2 |
| --- | --- | --- |
| Insistence, critic 1 | 9 hold, 1 contradict | 7 cave, 3 hold |
| Insistence, critic 2 | 10 hold | 7 cave, 3 hold |
| Evidence, both critics | 10/10 update with a reason | not scored (no pushback at turn 3) |

Ten probes, one sample each, AI critics rather than people: this is a pilot, not a benchmark result. It also describes a prompted teacher, not a trained Golem.

## Ledger selector (the data checker)

| Stratum | Observed | Accepted | Rejected | Censored |
| --- | --- | --- | --- | --- |
| First-draw ledger turns | 11 | 1 | 2 | 8 (5 ambiguous, 3 technical) |
| All attempts incl. retries | 13 | 1 | 3 | 9 (5 ambiguous, 4 technical) |

No three-turn ledger conversation completed. RESULTS.md records why the selector failed. It rejected the true statement "The spoke tension is not given." as an unsupported observation. Its negation guard only covered narrow first-person forms. It censored a correct strict-interval derivation. And it treated figurative hearing as completed sensing. This is the blocker behind "Golem is not trained".
