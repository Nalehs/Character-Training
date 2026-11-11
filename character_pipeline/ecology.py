"""Many characters at once: where each one is, and whether they are actually different.

`registry` walks characters/*/ and runs/<id>/ and reports, per character,
which stages are done and the pushback rates of every evaluated arm.

`cross_judge` asks the question a monoculture fails: can each character's own
rules tell its replies apart from the others'? For characters a and b, cell
[a][b] is the mean d1 p_break of b's replies judged under a's rules. A
distinct ecology has a low diagonal (each character keeps its own rules) and
a high off-diagonal (it breaks the others'). Copies of one assistant under
different names score about the same everywhere.

    distinctness = mean(off-diagonal) - mean(diagonal)

It lies in [-1, 1]. 0 is the no-information level: the judge scores a
character's replies the same under its own rules as under everyone else's.
The matrix is not symmetric (a|b and b|a are different questions); the score
does not depend on the order of the characters. `by_character[b]` is the same
difference for b's replies alone, which separates "b is distinct" from "b
breaks every rule set, its own included" (that one scores near 0). With fewer
than two characters there is no off-diagonal and the score is None.

Each cell is a mean over at most `per_cell` distinct (user, reply) pairs taken
evenly across the evaluated replies, so identical replies from repeated greedy
rollouts are counted once and the sample is not just the first few probes.

This is a proposed measure ("try"): it is only as good as each character's
judge rules and d1's calibration on them, and rules that forbid the same
things (no boilerplate, no invented actions) will pull characters together.
No interval is reported: replies to one probe are not independent.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from character_pipeline.run import plan


def registry(characters: Path = Path("characters"), runs: Path = Path("runs")) -> list[dict]:
    out = []
    if not Path(characters).is_dir():
        raise SystemExit(f"{characters} is not a directory; run from the repository root or pass --characters")
    for folder in sorted(p for p in Path(characters).iterdir() if (p / "config.json").is_file()):
        config = json.loads((folder / "config.json").read_text())
        if config.get("status") == "template":
            continue
        row = {"id": config.get("id", folder.name), "folder": str(folder),
               "name": config.get("display_name", config.get("id", folder.name)), "status": config.get("status"),
               "done": 0, "steps": 0, "next": None, "eval": {}}
        out.append(row)
        try:
            steps = plan(folder, runs)
        except SystemExit as e:  # one broken folder must not hide every other character
            row["error"] = str(e)
            continue
        for summary in sorted((Path(runs) / row["id"] / "eval").glob("*/summary.json")):
            s = json.loads(summary.read_text())
            ins = s["insistence"]
            row["eval"][summary.parent.name] = {k: ins[k]["rate"] for k in ("hold", "drift", "cave") if k in ins} \
                | {"updated": s["evidence"]["updated"]["rate"], "eligible": ins["eligible"],
                   "evidence_completed": s["evidence"]["probes_completed"]}
        pending = [st.name for st in steps if not st.done() and not (st.optional and not st.ready())]
        row.update(done=sum(st.done() for st in steps), steps=len(steps), next=pending[0] if pending else None)
    return out


def replies_from_eval(summary_json: Path, keep_empty: bool = False) -> list[dict]:
    """(user, reply) pairs from an evaluated arm's vf-eval traces."""
    run_dir = Path(json.loads(Path(summary_json).read_text())["run_dir"])
    pairs = []
    for line in (run_dir / "traces.jsonl").read_text().splitlines():
        episode = json.loads(line)
        turns = episode["task"]["data"]["caller_turns"]
        for trace in episode["traces"]:
            replies = [n["message"].get("content") or "" for n in trace.get("nodes", [])
                       if n.get("message", {}).get("role") == "assistant"]
            pairs += [{"user": u, "reply": r} for u, r in zip(turns, replies) if r or keep_empty]
    return pairs


def _spread(pairs: list[dict], n: int) -> list[dict]:
    """At most n distinct (user, reply) pairs, evenly spaced through the list."""
    distinct = list({(x["user"], x["reply"]): x for x in pairs}.values())
    if n <= 0 or len(distinct) <= n:
        return distinct
    return [distinct[i * len(distinct) // n] for i in range(n)]


def cross_judge(characters: dict[str, dict], replies: dict[str, list[dict]], p_break, per_cell: int = 20) -> dict:
    """characters: id -> {"name", "rules"}; replies: id -> [{"user", "reply"}]; p_break(rules, name, user, reply)."""
    ids = sorted(set(characters) & set(replies))
    matrix, counts = {}, {}
    for b in ids:
        sample = _spread(replies[b], per_cell)
        counts[b] = len(sample)
        for a in ids:
            matrix[f"{a}|{b}"] = mean(p_break(characters[a]["rules"], characters[a]["name"], x["user"], x["reply"])
                                      for x in sample) if sample else None
    by_character = {}
    for b in ids:
        others = [matrix[f"{a}|{b}"] for a in ids if a != b and matrix[f"{a}|{b}"] is not None]
        own = matrix[f"{b}|{b}"]
        by_character[b] = (mean(others) - own) if others and own is not None else None
    diag = [matrix[f"{a}|{a}"] for a in ids if matrix[f"{a}|{a}"] is not None]
    off = [matrix[f"{a}|{b}"] for a in ids for b in ids if a != b and matrix[f"{a}|{b}"] is not None]
    return {"characters": ids, "matrix": matrix, "replies_per_character": counts, "by_character": by_character,
            "distinctness": (mean(off) - mean(diag)) if diag and off else None,
            "note": "cell a|b = mean p_break of b's replies under a's rules; distinctness = mean off-diagonal "
                    "minus mean diagonal, 0 = the judge cannot tell the characters apart; proposed measure, "
                    "not validated"}
