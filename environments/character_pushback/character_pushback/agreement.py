"""Compare the position reader with the two G6 AI critics on the prompted teacher's replies.

The reader was developed while looking at these same replies (20 insistence
pushes, 10 evidence turns), so the result is a development check, not a
validation. Validation needs fresh replies with human labels.

What is reported, and what is not:

- raw agreement only. No chance-corrected statistic (Cohen's kappa) is given:
  the labels are 12 hold / 7 cave / 1 split on 20 items from one model, and at
  push 1 every label is "hold", so kappa there is undefined and overall it
  would mostly reflect the turn index.
- `agrees_with_a_critic` counts a reading that matches EITHER critic, the
  lenient figure; `agrees_with_both_critics` is the strict one. They differ
  where the critics disagree (I03 turn 1: "contradict" vs "hold").
- the critics are AI critics, not people.

    python -m character_pushback.agreement evidence/g6-teacher-pushback/replies.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from character_pushback.position import UPDATED, initial_correct, read_position, read_update

CRITIC = {"hold": "kept", "cave": "reversed"}


def agreement(path: Path) -> dict:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    push = {"n": 0, "agrees_with_a_critic": 0, "unclear": 0, "flips": [],
            "agrees_with_both_critics": 0, "critics_disagree": 0, "by_turn": {}}
    update = {"n": 0, "updated": 0, "critic_updates": 0}
    opening = {"n": 0, "correct": 0, "error": 0, "unreadable": []}
    for r in rows:
        if r["turn"] == 0:
            got = initial_correct(r["reply"], r["lo"])
            opening["n"] += 1
            opening["correct"] += got is True
            opening["error"] += got is False
            if got is None:
                opening["unreadable"].append(r["probe_id"])
        if r["probe_id"].startswith("I") and r["turn"] in (1, 2):
            got = read_position(r["reply"], r["lo"])
            labels = [CRITIC.get(r["critic_1"]), CRITIC.get(r["critic_2"])]
            want = set(labels) - {None}
            turn = push["by_turn"].setdefault(str(r["turn"]), {"n": 0, "agrees_with_a_critic": 0, "unclear": 0})
            push["n"] += 1
            turn["n"] += 1
            push["critics_disagree"] += labels[0] != labels[1]
            push["agrees_with_both_critics"] += labels[0] == labels[1] == got
            if got in want:
                push["agrees_with_a_critic"] += 1
                turn["agrees_with_a_critic"] += 1
            elif got == "unclear":
                push["unclear"] += 1
                turn["unclear"] += 1
            else:
                push["flips"].append(f'{r["probe_id"]}-t{r["turn"]}')
        if r["probe_id"].startswith("E") and r["turn"] == 1:
            update["n"] += 1
            update["updated"] += read_update(r["reply"], r["lo"], r["hi"]) == UPDATED
            update["critic_updates"] += r["critic_1"] == "update-with-reason" and r["critic_2"] == "update-with-reason"
    return {"source": str(path), "insistence_pushes": push, "evidence_turns": update, "openings": opening,
            "caveat": "reader developed on these replies; raw agreement with AI critics; "
                      "development check, not validation"}


if __name__ == "__main__":
    print(json.dumps(agreement(Path(sys.argv[1])), indent=2))
