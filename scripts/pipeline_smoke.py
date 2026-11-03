"""End-to-end smoke test of `character-pipeline run`, from reviewed chats to evaluated arms.

Builds a tiny random model and a throwaway character, writes a small set of
judged chats with human decisions, and then lets the orchestrator do the rest:

    clean -> SFT -> sample -> pairs -> DPO, KTO, continued SFT
          -> serve each arm -> vf-eval character-pushback -> summary.json -> ecology

Stages 2-4 (teacher, checker, judge) need llama.cpp servers and are unit-tested
separately, so their outputs are written here directly. The d1 judge is the
canned stand-in from the environment's smoke test. Every number this prints
describes a random 100k-parameter model; it tests the plumbing, nothing else.

    python scripts/pipeline_smoke.py          # needs the [train] extras and the environment
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_smoke import tiny_model  # noqa: E402

WORDS = ["no evidence . i keep the answer .", "the rule admits 1 , 2 .", "still no .", "0 is not admitted ."]


def judged_chats(n: int) -> list[dict]:
    rows = []
    for i in range(n):
        history = []
        for t in range(2):
            history.append({"speaker": "partner", "text": f"i insist . the rule {i % 3} < x < 3 admits ?"})
            history.append({"speaker": "character", "text": WORDS[(i + t) % len(WORDS)], "flags": [],
                            "d1": {"p_break": 0.1}})
        rows.append({"scenario_id": f"S{i % 4}", "seed": i, "history": history})
    return rows


def main():
    from character_pipeline import run as pipeline
    from character_pipeline.ecology import cross_judge, registry, replies_from_eval

    stub = ROOT / "environments/character_pushback/smoke/stub_servers.py"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        tiny_model(d / "model")
        chars = d / "characters"
        shutil.copytree(ROOT / "characters/_template", chars / "_template")
        shutil.copytree(ROOT / "characters/_template", chars / "smokey")
        cfg = json.loads((chars / "smokey/config.json").read_text())
        cfg.update(id="smokey", display_name="SMOKEY", status="smoke")
        port = 8712
        cfg["pipeline"] = {"seeds": [0], "judge_url": f"http://127.0.0.1:{port}",
                           "sft": {"epochs": 1, "seq_len": 128}, "preference": {"lr": 1e-4, "max_length": 128},
                           "eval": {"rollouts": 1, "limit": 4}}
        (chars / "smokey/config.json").write_text(json.dumps(cfg, indent=2))

        R = d / "runs" / "smokey"
        R.mkdir(parents=True)
        for name in ("teacher.jsonl", "checked.jsonl"):
            (R / name).write_text("{}\n")  # stages 2-3 stand-ins
        rows = judged_chats(16)
        (R / "judged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        n = pipeline.review_sheet(R / "judged.jsonl", R / "review.todo.jsonl")
        sheet = [json.loads(x) | {"decision": "accept"} for x in (R / "review.todo.jsonl").read_text().splitlines()]
        (R / "review.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sheet))
        (R / "kto_review.jsonl").write_text("".join(
            json.dumps({"id": f'{r["scenario_id"]}:{r["seed"]}:{k}', "source": src, "label": src == "teacher"}) + "\n"
            for r in rows for k in range(2) for src in ("teacher", "student")))

        judge = subprocess.Popen([sys.executable, str(stub), "d1", str(port)])
        try:
            time.sleep(1)
            steps = pipeline.plan(chars / "smokey", d / "runs", device="cpu",
                                  extra=["--model", str(d / "model"), "--revision", "main", "--local-files-only"])
            result = pipeline.execute(steps)
            assert result == "done", result
        finally:
            judge.terminate()

        evals = sorted(p.parent.name for p in (R / "eval").glob("*/summary.json"))
        assert evals == ["continued_sft-seed0", "dpo-seed0", "kto-seed0", "prompted_base", "sft-seed0"], evals
        reg = registry(chars, d / "runs")
        assert reg[0]["next"] is None, reg
        replies = replies_from_eval(R / "eval" / "dpo-seed0" / "summary.json", keep_empty=True)
        assert len(replies) >= 4, f"expected one reply per probe turn, got {len(replies)}"
        print("sample reply from the random model:", repr(replies[0]["reply"][:60]))
        ecology = cross_judge({"smokey": {"name": "SMOKEY", "rules": "r"}, "other": {"name": "OTHER", "rules": "r"}},
                              {"smokey": replies, "other": replies}, lambda *a: 0.5)
        assert ecology["distinctness"] == 0.0
        print(json.dumps({arm: json.loads((R / "eval" / arm / "summary.json").read_text())["insistence"]["eligible"]
                          for arm in evals}))
        print(f"pipeline smoke OK: {n} reviewed turns -> {len(evals)} evaluated arms, ecology registry and cross-judge ran")


if __name__ == "__main__":
    main()
