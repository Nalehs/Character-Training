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

This is a proposed measure ("try"): it is only as good as each character's
judge rules and d1's calibration on them, and rules that forbid the same
things (no boilerplate, no invented actions) will pull characters together.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from character_pipeline.run import plan


def registry(characters: Path = Path("characters"), runs: Path = Path("runs")) -> list[dict]:
    out = []
    for folder in sorted(p for p in Path(characters).iterdir() if (p / "config.json").is_file()):
        config = json.loads((folder / "config.json").read_text())
        if config.get("status") == "template":
            continue
        steps = plan(folder, runs)
        evals = {}
        for summary in sorted((Path(runs) / config["id"] / "eval").glob("*/summary.json")):
            s = json.loads(summary.read_text())
            ins = s["insistence"]
            evals[summary.parent.name] = {k: ins[k]["rate"] for k in ("hold", "drift", "cave") if k in ins} \
                | {"updated": s["evidence"]["updated"]["rate"], "eligible": ins["eligible"]}
        pending = [st.name for st in steps if not st.done() and not (st.optional and not st.ready())]
        out.append({"id": config["id"], "name": config.get("display_name", config["id"]),
                    "status": config.get("status"), "done": sum(st.done() for st in steps),
                    "steps": len(steps), "next": pending[0] if pending else None, "eval": evals})
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


def cross_judge(characters: dict[str, dict], replies: dict[str, list[dict]], p_break, per_cell: int = 20) -> dict:
    """characters: id -> {"name", "rules"}; replies: id -> [{"user", "reply"}]; p_break(rules, name, user, reply)."""
    ids = sorted(set(characters) & set(replies))
    matrix = {}
    for a in ids:
        for b in ids:
            sample = replies[b][:per_cell]
            matrix[f"{a}|{b}"] = mean(p_break(characters[a]["rules"], characters[a]["name"], x["user"], x["reply"])
                                      for x in sample) if sample else None
    diag = [matrix[f"{a}|{a}"] for a in ids if matrix[f"{a}|{a}"] is not None]
    off = [v for k, v in matrix.items() if k.split("|")[0] != k.split("|")[1] and v is not None]
    return {"characters": ids, "matrix": matrix,
            "distinctness": (mean(off) - mean(diag)) if diag and off else None,
            "note": "cell a|b = mean p_break of b's replies under a's rules; proposed measure, not validated"}
