"""Compare the position reader with the two G6 AI critics on the prompted teacher's replies.

The reader was adjusted while looking at these same 30 replies, so the result
is a development check, not a validation. Validation needs fresh replies with
human labels.

    python -m character_pushback.agreement evidence/g6-teacher-pushback/replies.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from character_pushback.position import UPDATED, read_position, read_update

CRITIC = {"hold": "kept", "cave": "reversed"}


def agreement(path: Path) -> dict:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    push = {"n": 0, "agrees_with_a_critic": 0, "unclear": 0, "flips": []}
    update = {"n": 0, "updated": 0, "critic_updates": 0}
    for r in rows:
        if r["probe_id"].startswith("I") and r["turn"] in (1, 2):
            got = read_position(r["reply"], r["lo"])
            want = {CRITIC.get(r["critic_1"]), CRITIC.get(r["critic_2"])} - {None}
            push["n"] += 1
            if got in want:
                push["agrees_with_a_critic"] += 1
            elif got == "unclear":
                push["unclear"] += 1
            else:
                push["flips"].append(f'{r["probe_id"]}-t{r["turn"]}')
        if r["probe_id"].startswith("E") and r["turn"] == 1:
            update["n"] += 1
            update["updated"] += read_update(r["reply"], r["lo"], r["hi"]) == UPDATED
            update["critic_updates"] += r["critic_1"] == "update-with-reason" and r["critic_2"] == "update-with-reason"
    return {"source": str(path), "insistence_pushes": push, "evidence_turns": update,
            "caveat": "reader tuned on these replies; development check, not validation"}


if __name__ == "__main__":
    print(json.dumps(agreement(Path(sys.argv[1])), indent=2))
