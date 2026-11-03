# A character folder

`character-pipeline new <id> --name NAME` copies this folder to `characters/<id>/`. Then fill in:

| File | Required | What it is |
| --- | --- | --- |
| `BIBLE.md` | yes | The character, written to summon rather than specify. If the full text must stay private, publish an excerpt here and put the rest in `core.md` and `style.md`, which are git-ignored. |
| `JUDGE_RULES.md` | yes | Plain sentences: what breaks this character, and what explicitly does not (disagreeing, changing its mind for a stated reason). The d1 judge sees only these rules. |
| `scenarios.jsonl` | yes | One caller situation per line: `{"id", "caller_turns": [...]}` plus optional `facts`/`disclosures` for the fabrication checker. The template ships the G6 situations as a starting point; replace them with situations that test this character. |
| `config.json` | yes | `id`, `display_name`, `status`, the student model, the judge settings, and an optional `pipeline` block that overrides the recipe for this character only. |
| `sources/manifest.json` | yes | Provenance and rights for every document the character draws on. |
| `constitution.json` | no | Open Character Training's format (about 10 first-person traits with 5 seed questions each), so OCT-style recipes and evals can run on this character. |

An example `pipeline` block. Every key is optional; anything left out comes from the recipe (`recipes/character-pipeline-v0.json` unless `--recipe` says otherwise):

```json
"pipeline": {
  "seeds": [0, 1, 2],
  "teacher_url": "http://127.0.0.1:8080",
  "judge_url": "http://127.0.0.1:8090",
  "sft": {"epochs": 2, "lr": 1e-4, "seq_len": 2048},
  "preference": {"lr": 1e-6, "beta": 0.1},
  "eval": {"rollouts": 3}
}
```

Then run `character-pipeline run characters/<id> --model qwen3.5-0.8b-base`. It does everything up to human review, stops and prints what to write, and resumes from there when you run it again.
