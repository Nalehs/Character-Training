"""Run the stage 3 checker over raw teacher output.

For each character turn: build the public caller reports disclosed so far,
ask the teacher (greedy) to extract claims with the G6 EXTRACTOR prompt, then
apply `fabrication.validate` and `fabrication.repetition`. The extractor
envelope and capability string follow golem-15 gen/g6/upload/kernel.py
(`extractor_spec`, `public`). Output rows carry the routing status only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from character_pipeline.check.fabrication import EXTRACTOR, repetition, validate
from character_pipeline.data.io import read_jsonl, resumable_jsonl
from character_pipeline.generate.controller import render
from character_pipeline.generate.teacher import complete, finish_reason, request_body

CAPABILITIES = ("text only; no tools, sensing, login, persistent memory; "
                "earlier GOLEM claims are not evidence")


def public_reports(scenario: dict, upto_turn: int) -> list[dict]:
    rows = []
    for turn, disclosures in enumerate(scenario.get("disclosures", [])[: upto_turn + 1]):
        for d in disclosures:
            rows.append({"id": d["fact_id"], "text": d["exact_span"], "owner": d["owner"],
                         "visibility": "public", "evidence_status": d["status"], "source_turn": turn})
    return rows


def check_conversation(row: dict, scenario: dict, url: str, call=complete) -> list[dict]:
    out, prior, caller_index = [], [], -1
    for turn in row["history"]:
        if turn["speaker"] == "partner":
            caller_index += 1
            continue
        pub = public_reports(scenario, caller_index)
        envelope = {"PUBLIC_CALLER_REPORTS": pub, "CANDIDATE": turn["text"], "capabilities": CAPABILITIES}
        prompt = render(EXTRACTOR, [{"speaker": "partner", "text": json.dumps(envelope)}], "character")
        response = call(url, request_body(prompt, row["seed"] + 100000 + caller_index, cap=512, extractor=True))
        complete_ = finish_reason(response) == "stop" and turn.get("finish_reason") == "stop"
        result = validate(turn["text"], pub, response.get("content", ""), complete=complete_)
        rep = repetition(turn["text"], prior)
        out.append({"scenario_id": row["scenario_id"], "seed": row["seed"], "turn": caller_index, "text": turn["text"],
                    "status": result["status"], "rules": result["rules"], "claims": result["claims"],
                    "reasons": result["reasons"], "repetition": rep, "hygiene_flags": turn.get("flags", []),
                    "route": "human_review", "training_eligible": False})
        prior.append(turn["text"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("teacher_output", type=Path)
    ap.add_argument("scenarios", type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--out", type=Path, default=Path("runs/checked.jsonl"))
    args = ap.parse_args(argv)
    scenarios = {s["id"]: s for s in read_jsonl(args.scenarios)}
    rows = read_jsonl(args.teacher_output)
    missing = sorted({r["scenario_id"] for r in rows} - set(scenarios))
    if missing:
        raise SystemExit(f"{args.scenarios} has no scenario for {missing[:5]}; pass the file the teacher ran on")
    counts: dict[str, int] = {}
    # Written to <out>.partial and renamed at the end: an interrupted check is redone from the
    # start (the extractor is greedy, so the result is the same), never left looking finished.
    with resumable_jsonl(args.out, resume=False) as (_, write):
        for row in rows:
            for checked in check_conversation(row, scenarios[row["scenario_id"]], args.url):
                counts[checked["status"]] = counts.get(checked["status"], 0) + 1
                write(checked)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
