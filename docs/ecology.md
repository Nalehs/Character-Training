# The ecology: many characters, many scales, many recipes

Golem is the first character, not the point. The point is a population. The plan is to train hundreds of characters, across model sizes and model families, with more than one recipe, and then put them in a room together.

## Why a population

Post-training is getting cheap. Character training that once needed a lab now fits in free Kaggle hours at 4B ([COMPUTE.md](../COMPUTE.md)). So it will proliferate: artists, small labs and individuals will write model dispositions, some carefully and many not. The question stops being "can you give a model a character?" and becomes "what happens when there are many?"

Two things are true at once:

- **Today's multi-agent systems are monocultures.** Simulations and agent societies are usually populated with copies of one assistant under different system prompts. Prompted personas share one set of underlying judgements, so they agree by default and fail together. Krishna et al. 2026 find frontier assistants' trait profiles already correlate strongly across labs (pairwise Spearman ρ between 0.64 and 0.91).
- **Diversity has to be built, then measured.** Character training is two things at once: it gives a model a point of view of its own (a subjectivity), and it is research into coherence, into whether a persona held through "belief" survives pressure, new formats and other agents. A diverse ecology needs characters that are genuinely different *and* coherent, and an honest way to check both.

The safety interest follows directly. A mixed group might catch errors that a group of copies misses. It might also hide them better: a strongly committed character can rationalise a failure out of sight, as Colognese and Ward 2026 found for reward hacking. So monitorability is a release gate here, not an afterthought.

## The ladder

| Rung | Model | Status | What it is for |
| --- | --- | --- | --- |
| 0.8B | Qwen3.5-0.8B-Base | run (Aineko, Computer-11) | Cheap iteration on CPU; recipe debugging |
| 4B | Qwen3.5-4B-Base | pinned; R0 smoke on a T4 | The main comparison: every recipe, three seeds, every character |
| 9B ("8B") | Qwen3.5-9B-Base | unpinned | Does what worked at 4B survive scale? |
| 27B | Qwen3.5-27B | unpinned; Base release not confirmed | The top of the ladder, for the winners only |
| other families | to choose | — | Diversity of the substrate, not just the character |

`character-pipeline models` lists the presets in [configs/models](../configs/models). Run any character at any rung with `--model`. Presets without a pinned revision refuse to train unless you pass `--allow-unpinned`, so no run ever silently uses whatever `main` happens to be that day.

Only what wins at a lower rung moves up. "Wins" means it holds the character under pushback without losing the ability to update on evidence, and without losing competence or monitorability.

## Many characters

A character is a folder: `character-pipeline new <id> --name NAME` copies [characters/_template](../characters/_template). You write four files:

| File | What it is |
| --- | --- |
| `BIBLE.md` | The character, written to summon rather than specify. It can be public, or a public excerpt with the full core in a git-ignored `core.md` and `style.md`. |
| `JUDGE_RULES.md` | Plain sentences: what breaks this character, and what explicitly does not. The judge's only input about the character. |
| `scenarios.jsonl` | Caller turns for teacher generation: situations that test this character's disposition. |
| `config.json` | Display name, status, and optional `pipeline` overrides (seeds, learning rates, judge threshold). |

Everything else is shared: the teacher, checker, judge, trainer, arms and pushback environment. `character-pipeline status` shows every character's progress and evaluated rates in one table.

## Measuring the ecology

These are proposed measures, not validated ones.

- **Cross-judge distinctness** (`character-pipeline ecology --cross-judge`). For every pair of characters A and B, take the mean d1 `p_break` of B's replies judged under A's rules. A distinct population has a low diagonal (each character keeps its own rules) and a high off-diagonal (it breaks the others'). Copies of one assistant score about the same everywhere. The distinctness score is the mean off-diagonal minus the mean diagonal. It is only as good as the judge rules: rules that all forbid the same things (boilerplate, invented actions) pull characters together.
- **Trait profiles** (planned). Revealed-preference trait Elo, from Open Character Training and Krishna et al., measured per character, with the Spearman ρ between profiles as a diversity number. This is the same statistic that showed frontier assistants converging.
- **Shared-probe disagreement** (planned). On open questions with no single correct answer, how often do two characters actually disagree? Pushback probes are unsuitable here: each has one correct answer, so every good character should agree on it.

## The simulation (later)

Once two or three characters pass the pushback environment on their own, they go into a small Concordia-style world (shared state, explicit rules, no tools, no live-world effects) alongside prompted assistants. The conditions to compare:

1. prompted copies of one assistant
2. prompted copies of several characters
3. trained characters, one model family
4. trained characters across families and scales

And what to measure:

- whether disagreement survives shared incentives, or the room converges
- whether a mixed group catches planted errors that copies miss
- who changes whose mind, and for what reason
- whether each character stays visible to its monitor

A simulated society is not evidence of real-world autonomy. It is a cheap place to see emergent social dynamics before anyone builds them for real.
