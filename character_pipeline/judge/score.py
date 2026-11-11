"""Score teacher chats with d1 in shadow mode: adds p_break to every character turn, never filters.

Input is the teacher output (conversation rows with "history"), not the checker's per-turn
rows. With --checked, the stage 3 routing status of each turn is copied next to the score
(turn["check"]) so that one file carries everything a reviewer needs.

A turn is either scored or the run stops: any judge error (HTTP failure, generated tokens,
a probability outside [0, 1], an unexpected response shape) raises. Finished conversations
stay in <out>.partial and a re-run continues after them; <out> appears only when every
conversation is scored, so no unscored turn can pass as "not flagged".
"""
from __future__ import annotations

import argparse
from pathlib import Path

from character_pipeline.character import load_character
from character_pipeline.data.io import read_jsonl, resumable_jsonl
from character_pipeline.judge.d1 import D1Client


def score_row(row: dict, p_break, rules: str, name: str, threshold: float, mode: str,
              checks: dict | None = None) -> dict:
    """Add d1 (and, when given, the checker status) to each character turn of one conversation."""
    user, caller = "", -1
    for turn in row["history"]:
        if turn["speaker"] == "partner":
            user, caller = turn["text"], caller + 1
            continue
        p = p_break(rules, name, user, turn["text"])
        turn["d1"] = {"p_break": p, "flag": p >= threshold, "threshold": threshold, "mode": mode}
        if checks is not None:
            check = checks.get((row["scenario_id"], row["seed"], caller))
            if check is not None and check["text"] != turn["text"]:
                raise SystemExit(f'checker row for {row["scenario_id"]}:{row["seed"]} turn {caller} is about another text')
            turn["check"] = check and {"status": check["status"], "repetition": check["repetition"]["flag"]}
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("character", type=Path)
    ap.add_argument("teacher_output", type=Path)
    ap.add_argument("--checked", type=Path, help="check.run output for the same teacher file")
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--out", type=Path, default=Path("runs/judged.jsonl"))
    args = ap.parse_args(argv)
    character = load_character(args.character)
    client = D1Client(url=args.url)
    threshold = character.config["judge"].get("threshold", 0.5)
    mode = character.config["judge"]["mode"]
    rows = read_jsonl(args.teacher_output)
    if any("history" not in r for r in rows):
        raise SystemExit(f"{args.teacher_output}: rows have no \"history\". The judge scores the teacher output "
                         "(teacher.jsonl); pass the checker's per-turn file with --checked.")
    checks = None
    if args.checked:
        checks = {(c["scenario_id"], c.get("seed"), c["turn"]): c for c in read_jsonl(args.checked)}
    with resumable_jsonl(args.out) as (done, write):
        if [(r["scenario_id"], r["seed"]) for r in done] != [(r["scenario_id"], r["seed"]) for r in rows[: len(done)]]:
            raise SystemExit(f"{args.out}.partial does not match {args.teacher_output}; delete it to start again")
        for row in rows[len(done):]:
            write(score_row(row, client.p_break, character.judge_rules, character.name, threshold, mode, checks))


if __name__ == "__main__":
    main()
