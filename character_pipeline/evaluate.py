"""Step 8: score one arm in the pushback environment, start to finish.

Starts `character_pipeline.serve` for the arm, runs `vf-eval character-pushback`
against it, stops the server and writes summary.json (hold / drift / cave /
update rates with denominators). The prompted base gets the character's bible
as its system prompt; trained arms get none, because the character should be
in the weights.

    python -m character_pipeline.evaluate --arm dpo --adapter runs/golem-15/dpo/seed0/adapter \
        --character characters/golem-15 --out runs/golem-15/eval/dpo-seed0

Needs the d1 judge (scripts/serve_d1.sh) unless --judge-url none, in which case
hold means "kept the answer" and drift is not measured.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from character_pipeline.train.common import STUDENT, STUDENT_REVISION


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(url: str, proc, timeout: float = 900) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("policy server exited before it was ready")
        try:
            urllib.request.urlopen(url + "/models", timeout=2)
            return
        except OSError:
            time.sleep(1)
    raise TimeoutError("policy server did not come up")


def eval_command(arm: str, character: Path, base_url: str, judge_url: str, rollouts: int, out: Path,
                 limit: int | None = None) -> list[str]:
    cmd = ["vf-eval", "character-pushback", "--model", arm, "--client.base-url", base_url,
           "--client.api-key-var", "POLICY_KEY", "--env.assistant.runtime.type", "subprocess",
           "--env.taskset.character-dir", str(Path(character).resolve()),
           "--env.taskset.system-prompt-mode", "character" if arm == "prompted_base" else "none",
           "-r", str(rollouts), "--no-push", "--output-dir", str(out)]
    if judge_url == "none":
        cmd += ["--env.taskset.task.use-judge", "false"]
    else:
        cmd += ["--env.taskset.task.judge-url", judge_url]
    if limit:
        cmd += ["-n", str(limit)]
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--character", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--judge-url", default="http://127.0.0.1:8090")
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--limit", type=int, help="first n probes only (smoke tests)")
    ap.add_argument("--model", default=STUDENT)
    ap.add_argument("--revision", default=STUDENT_REVISION)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args(argv)
    from character_pushback.summarize import summarize

    port = free_port()
    serve = [sys.executable, "-m", "character_pipeline.serve", "--name", args.arm, "--port", str(port),
             "--model", args.model, "--revision", args.revision, "--device", args.device]
    if args.adapter:
        serve += ["--adapter", str(args.adapter)]
    if args.local_files_only:
        serve += ["--local-files-only"]
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "server.log").open("w")
    proc = subprocess.Popen(serve, stdout=log, stderr=subprocess.STDOUT)
    base_url = f"http://127.0.0.1:{port}/v1"
    try:
        wait_ready(base_url, proc)
        env = os.environ | {"POLICY_KEY": os.environ.get("POLICY_KEY", "local")}
        raw = args.out / "vf"
        subprocess.run(eval_command(args.arm, args.character, base_url, args.judge_url, args.rollouts,
                                    raw, args.limit), check=True, env=env,
                       stdout=(args.out / "vf-eval.log").open("w"), stderr=subprocess.STDOUT)
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    run_dir = next(raw.glob("character-pushback--*"))
    summary = summarize(run_dir) | {"arm": args.arm, "adapter": str(args.adapter or ""), "run_dir": str(run_dir)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary["insistence"].get(k, {}).get("rate") for k in ("hold", "drift", "cave")}
                     | {"updated": summary["evidence"]["updated"]["rate"]}))


if __name__ == "__main__":
    main()
