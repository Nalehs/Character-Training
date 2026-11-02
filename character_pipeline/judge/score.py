"""Score checked chats with d1 in shadow mode: adds p_break, never filters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from character_pipeline.character import load_character
from character_pipeline.judge.d1 import D1Client


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("character", type=Path)
    ap.add_argument("teacher_output", type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--out", type=Path, default=Path("runs/judged.jsonl"))
    args = ap.parse_args(argv)
    character = load_character(args.character)
    client = D1Client(url=args.url)
    threshold = character.config["judge"].get("threshold", 0.5)
    with args.out.open("x") as f:
        for line in args.teacher_output.read_text().splitlines():
            row = json.loads(line)
            user = ""
            for turn in row["history"]:
                if turn["speaker"] == "partner":
                    user = turn["text"]
                    continue
                p = client.p_break(character.judge_rules, character.name, user, turn["text"])
                turn["d1"] = {"p_break": p, "flag": p >= threshold, "mode": character.config["judge"]["mode"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
